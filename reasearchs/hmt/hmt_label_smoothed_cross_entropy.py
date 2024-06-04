# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class HmtLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def Latency_loss(
    model,
    src_lens,
    lprobs,
    target,
    transition_lprob,
    epsilon,
    ignore_index=None,
    reduce=True,
):
    bsz, tgt_len, cands_per_token, voc = lprobs.size()
    target = target.unsqueeze(-1).repeat(1, 1, cands_per_token)
    target = target.unsqueeze(-1)
    first_read = model.decoder.first_read
    cands_per_token = model.decoder.cands_per_token

    idea_latency = (
        torch.arange(first_read, first_read + tgt_len, device=lprobs.device)
        .unsqueeze(1)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(bsz, 1, 1, 1)
    )
    idea_latency = idea_latency.min(src_lens.unsqueeze(1).unsqueeze(2).unsqueeze(3))
    cands = model.decoder.cands
    cands = cands.contiguous().view(1, tgt_len, cands_per_token, 1)
    cands = cands.min(src_lens.unsqueeze(1).unsqueeze(2).unsqueeze(3))
    delay = cands - idea_latency

    not_transition_lprob = transition_lprob[:, :, :, 0:1]
    cum_not_transition_lprob = torch.cumsum(not_transition_lprob, dim=2)
    cum_not_transition_lprob = torch.cat(
        (
            torch.zeros(
                (
                    cum_not_transition_lprob.size(0),
                    cum_not_transition_lprob.size(1),
                    1,
                    cum_not_transition_lprob.size(3),
                ),
                device=not_transition_lprob.device,
            ),
            cum_not_transition_lprob[:, :, :-1, :],
        ),
        dim=2,
    )
    construct_transition_lprob = (
        transition_lprob[:, :, :, 1:] + cum_not_transition_lprob
    )

    alpha_list = []
    alpha = construct_transition_lprob[:, :1, :, :].transpose(2, 3)
    alpha_list.append(alpha)

    for i in range(tgt_len - 1):
        construct_transition_lprob_i = (
            construct_transition_lprob[:, i + 1 : i + 2, :, :]
            .repeat(1, 1, 1, cands_per_token)
            .transpose(2, 3)
        )
        construct_transition_lprob_i = construct_transition_lprob_i.masked_fill(
            (
                cands[:, i : i + 1, :, :]
                > cands[:, i + 1 : i + 2, :, :].transpose(-1, -2)
            ),
            float("-inf"),
        )
        construct_transition_lprob_i = construct_transition_lprob_i - torch.logsumexp(
            (construct_transition_lprob_i), dim=-1, keepdim=True
        )
        alpha = torch.logsumexp(
            (alpha.transpose(2, 3) + construct_transition_lprob_i), dim=2, keepdim=True
        )
        alpha_list.append(alpha)

    alphas = torch.exp(torch.cat(alpha_list, dim=1))
    latency_loss = alphas.transpose(2, 3) * delay

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        latency_loss = latency_loss.masked_fill(pad_mask, 0.0)

    latency_loss = latency_loss / (latency_loss.sum(dim=2, keepdim=True) > 0).sum(
        dim=1, keepdim=True
    ).clamp(1, tgt_len)
    latency_loss = torch.abs(latency_loss.sum())

    return latency_loss


def HMM_loss(
    model,
    src_lens,
    lprobs,
    target,
    transition_lprob,
    epsilon,
    ignore_index=None,
    reduce=True,
):
    bsz, tgt_len, cands_per_token, voc = lprobs.size()
    target = target.unsqueeze(-1).repeat(1, 1, cands_per_token)
    target = target.unsqueeze(-1)
    gt_lprob = lprobs.gather(dim=-1, index=target)
    not_transition_lprob = transition_lprob[:, :, :, 0:1]
    cum_not_transition_lprob = torch.cumsum(not_transition_lprob, dim=2)
    cum_not_transition_lprob = torch.cat(
        (
            torch.zeros(
                (
                    cum_not_transition_lprob.size(0),
                    cum_not_transition_lprob.size(1),
                    1,
                    cum_not_transition_lprob.size(3),
                ),
                device=not_transition_lprob.device,
            ),
            cum_not_transition_lprob[:, :, :-1, :],
        ),
        dim=2,
    )

    construct_transition_lprob = (
        transition_lprob[:, :, :, 1:] + cum_not_transition_lprob
    )

    cands = model.decoder.cands
    cands = cands.contiguous().view(1, tgt_len, cands_per_token, 1)
    cands = cands.min(src_lens.unsqueeze(1).unsqueeze(2).unsqueeze(3))
    first_read = model.decoder.first_read
    cands_per_token = model.decoder.cands_per_token

    alpha_list = []
    alpha = construct_transition_lprob[:, :1, :, :].transpose(2, 3) + gt_lprob[
        :, 0:1, :, :
    ].transpose(2, 3)
    alpha_list.append(alpha)

    for i in range(tgt_len - 1):
        gt_lprob_i = gt_lprob[:, i + 1 : i + 2, :, :]
        construct_transition_lprob_i = (
            construct_transition_lprob[:, i + 1 : i + 2, :, :]
            .repeat(1, 1, 1, cands_per_token)
            .transpose(2, 3)
        )
        construct_transition_lprob_i = construct_transition_lprob_i.masked_fill(
            (
                cands[:, i : i + 1, :, :]
                > cands[:, i + 1 : i + 2, :, :].transpose(-1, -2)
            ),
            float("-inf"),
        )
        construct_transition_lprob_i = construct_transition_lprob_i - torch.logsumexp(
            (construct_transition_lprob_i), dim=-1, keepdim=True
        )
        alpha = torch.logsumexp(
            (alpha.transpose(2, 3) + construct_transition_lprob_i), dim=2, keepdim=True
        ) + gt_lprob_i.transpose(2, 3)
        alpha_list.append(alpha)

    alphas = torch.cat(alpha_list, dim=1)
    tgt_lens = (target != ignore_index).sum(dim=1, keepdim=True).transpose(-1, -2)
    last_alpha = alphas.gather(dim=1, index=tgt_lens - 1)

    alpha = torch.logsumexp(last_alpha, dim=-1)
    alpha = alpha.type_as(lprobs)
    nll_loss = -1 * alpha.sum()
    loss = nll_loss

    return loss, nll_loss


def CE_loss(lprobs, target, transition_prob, epsilon, ignore_index=None, reduce=True):
    bsz, tgt_len, cands_per_token, voc = lprobs.size()
    target = target.unsqueeze(-1).repeat(1, 1, cands_per_token)
    target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    loss = loss / cands_per_token
    nll_loss = nll_loss / cands_per_token

    return loss, nll_loss


def bulid_transition_mask(first_read, cands_per_token):

    a = torch.arange(first_read, first_read + cands_per_token, device="cuda")
    b = torch.arange(first_read + 1, first_read + cands_per_token + 1, device="cuda")
    transition_mask = a.unsqueeze(1) > b.unsqueeze(0)
    return transition_mask


@register_criterion(
    "hmt_label_smoothed_cross_entropy",
    dataclass=HmtLabelSmoothedCrossEntropyCriterionConfig,
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        src_len = sample["net_input"]["src_tokens"].size(1)
        x, transition_prob, extra = model(**sample["net_input"])
        # loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        lprobs = model.get_normalized_probs(
            (x, extra), log_probs=True
        )  # bsz * tgt_len * cands_per_token * voc
        transition_lprob = model.get_normalized_probs(
            (transition_prob, extra), log_probs=True
        )
        # prevent overflowing
        lprobs = lprobs.float()
        transition_lprob = transition_lprob.float()
        target = model.get_targets(sample, (x, extra))  # bsz*tgt_len

        hmm_loss, hmm_nll_loss = HMM_loss(
            model,
            sample["net_input"]["src_lengths"],
            lprobs,
            target,
            transition_lprob,
            self.eps,
            ignore_index=self.padding_idx,
        )
        state_loss, state_nll_loss = CE_loss(
            lprobs, target, transition_lprob, self.eps, ignore_index=self.padding_idx
        )
        latency_loss = Latency_loss(
            model,
            sample["net_input"]["src_lengths"],
            lprobs,
            target,
            transition_lprob,
            self.eps,
            ignore_index=self.padding_idx,
        )

        loss = hmm_loss + state_loss + latency_loss
        nll_loss = hmm_nll_loss + state_nll_loss + latency_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "hmm_loss": hmm_loss.data,
            "state_loss": state_loss.data,
            "latency_loss": latency_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        hmm_loss_sum = sum(log.get("hmm_loss", 0) for log in logging_outputs)
        state_loss_sum = sum(log.get("state_loss", 0) for log in logging_outputs)
        latency_loss_sum = sum(log.get("latency_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "hmm_loss", hmm_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "state_loss",
            state_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "latency_loss",
            latency_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: (
                    round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3)
                    if meters["total"].sum > 0
                    else float("nan")
                ),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
