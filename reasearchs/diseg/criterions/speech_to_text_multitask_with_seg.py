# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


def seg2weight(seg_prob):
    bsz, src_len = seg_prob.size()
    tmp = (
        torch.arange(0, src_len, device=seg_prob.device)
        .unsqueeze(0)
        .unsqueeze(1)
        .repeat(bsz, src_len, 1)
    )
    idx = (
        torch.arange(0, src_len, device=seg_prob.device)
        .unsqueeze(0)
        .unsqueeze(2)
        .repeat(bsz, 1, 1)
    )
    left = tmp < idx
    right = tmp > idx
    equal = tmp == idx

    res = seg_prob.unsqueeze(1).repeat(1, src_len, 1)
    res_left = res.masked_fill((right | equal), 0)
    res_right = res.masked_fill(left, 0)

    cumprod_left = torch.cumprod((1 - res_left).flip(-1), dim=-1).flip(-1)
    cumprod_right = torch.cumprod((1 - res_right), dim=-1)
    cumprod_right = torch.cat(
        (
            torch.zeros(
                (cumprod_right.size(0), cumprod_right.size(1), 1),
                device=seg_prob.device,
            ).type_as(cumprod_right),
            cumprod_right[:, :, :-1],
        ),
        dim=-1,
    )
    cumprod_eq = torch.ones_like(res)

    seg_weight = (
        cumprod_left * left.type_as(cumprod_left)
        + cumprod_eq * equal.type_as(cumprod_eq)
        + cumprod_right * right.type_as(cumprod_right)
    )
    return seg_weight / seg_weight.sum(dim=-1)


@dataclass
class SpeechToTextMultitaskwithSegCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    mt_training: bool = field(
        default=False,
        metadata={"help": "add mt multi-task"},
    )
    asr_training: bool = field(
        default=False,
        metadata={"help": "add asr multi-task"},
    )
    seg_speech: bool = field(
        default=False,
        metadata={"help": "segment speech"},
    )
    add_speech_seg_text_ctr: bool = field(
        default=False,
        metadata={"help": "add_speech_seg_text_ctr"},
    )


@register_criterion(
    "speech_to_text_multitask_with_seg",
    dataclass=SpeechToTextMultitaskwithSegCriterionConfig,
)
class SpeechToTextMultitaskwithSegCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_training=False,
        asr_training=False,
        seg_speech=False,
        add_speech_seg_text_ctr=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.mt_training = mt_training
        self.asr_training = asr_training
        self.seg_speech = seg_speech
        self.add_speech_seg_text_ctr = add_speech_seg_text_ctr

    def forward_st(self, model, sample, update_num, reduce, training_lagging_seg=None):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "seg_speech": self.seg_speech,  # if (update_num is None or update_num>5000) else False,
            "update_num": update_num,
        }
        st_output, speech_encoder_out = model(
            **audio_input,
            # training_lagging_seg=training_lagging_seg
        )

        loss, _ = self.compute_loss(model, st_output, sample, reduce=reduce)
        return loss, speech_encoder_out

    def forward_mt(self, model, sample, update_num, reduce, training_lagging_seg=None):
        text_input = {
            "src_tokens": sample["net_input"]["transcription"],
            "src_lengths": sample["net_input"]["transcription_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        mt_output, text_encoder_out = model(
            **text_input, training_lagging_seg=training_lagging_seg
        )
        loss, _ = self.compute_loss(model, mt_output, sample, reduce=reduce)
        return loss, text_encoder_out

    def forward_asr(
        self,
        model,
        sample,
        update_num,
        reduce,
        speech_encoder_out=None,
        training_lagging_seg=None,
    ):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_transcription_tokens"],
            "seg_speech": self.seg_speech,  # if update_num>60000 else False,
            "update_num": update_num,
        }
        asr_output, _ = model(
            **audio_input,
            speech_encoder_out=speech_encoder_out,
            training_lagging_seg=training_lagging_seg,
        )
        loss, _ = self.compute_loss(
            model, asr_output, {"target": sample["transcription"]}, reduce=reduce
        )
        return loss

    def forward_ext_mt(self, model, sample, update_num, reduce):
        text_output, _ = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def calculate_seg_num_loss(self, speech_encoder_out, sample):
        # loss of segment number
        seg_prob = speech_encoder_out["seg_prob"]
        src_len = seg_prob.size(1)

        number = (sample["transcription_tokens_num"].float() - 1).clamp(1, 9999)
        bsz = seg_prob.size(0)

        seg_prob_pooling_sum_list = []
        for i in range(number.size(0)):
            kernel_size = max(math.floor(src_len / number[i].item()), 1)
            m = torch.nn.MaxPool1d(
                kernel_size,
                stride=None,
                padding=0,
            )
            seg_prob_pooling = m(seg_prob[i : i + 1].unsqueeze(0)).squeeze(0)
            seg_prob_pooling_sum = seg_prob_pooling.sum(dim=-1)
            seg_prob_pooling_sum_list.append(seg_prob_pooling_sum)
        seg_num_loss = (
            torch.dist(seg_prob.sum(dim=-1), number, p=2)
            + torch.sqrt(
                (
                    torch.pow(torch.cat(seg_prob_pooling_sum_list, dim=0) - number, 2)
                    * (src_len / number)
                ).sum(dim=-1)
                + 1e-6
            ).sum()
        )
        return seg_num_loss

    def expected_speech_seg(self, seg_prob, seg_num):
        # expected feature-to-segment mapping
        bsz, src_len = seg_prob.size()
        expected_segs = []
        pad = torch.nn.ZeroPad2d(padding=(0, 0, 1, 0))

        def pad_x(x):
            return pad(x)[:, :-1, :]

        expected_seg_i = torch.cat(
            (
                torch.ones((bsz, 1, 1), device=seg_prob.device),
                torch.zeros((bsz, seg_num - 1, 1), device=seg_prob.device),
            ),
            dim=1,
        ).type_as(seg_prob)

        expected_segs.append(expected_seg_i)
        for i in range(src_len - 1):
            c_i = seg_prob[:, i : i + 1].unsqueeze(1)
            expected_seg_i = pad_x(expected_seg_i) * c_i + expected_seg_i * (1 - c_i)
            expected_seg_i = expected_seg_i.clamp(1e-4, 1)
            expected_segs.append(expected_seg_i)

        expected_speech_seg = torch.cat(expected_segs, dim=-1)
        expected_speech_seg / (expected_speech_seg.sum(dim=1, keepdim=True) + 1e-4)
        return expected_speech_seg / (
            expected_speech_seg.sum(dim=-1, keepdim=True) + 1e-4
        )

    def calculate_expected_ctr_loss(self, speech_encoder_out, text_encoder_out, sample):
        from torch_scatter import scatter_mean

        speech_hidden_states = speech_encoder_out["encoder_embedding"][0].transpose(
            0, 1
        )
        text_hidden_states = text_encoder_out["encoder_embedding"][0]
        number = (sample["transcription_tokens_num"].int()).clamp(1, 999999999)

        # subword-to-word mapping
        sub2token_id = sample["sub2token_id"]
        seg_prob = speech_encoder_out["seg_prob"].type_as(speech_hidden_states)

        bsz = speech_hidden_states.size(0)
        ctr_loss = 0

        # expected segment representation
        expected_speech_seg = self.expected_speech_seg(seg_prob, number.max().item())
        expected_speech_seg_hidden_states = torch.bmm(
            expected_speech_seg, speech_hidden_states
        )

        # word representation
        detok_text_hidden_states = scatter_mean(
            text_hidden_states, sub2token_id, dim=1
        )[:, 1:, :]

        logits = torch.nn.functional.cosine_similarity(
            expected_speech_seg_hidden_states.unsqueeze(2).float(),
            detok_text_hidden_states.unsqueeze(1).float(),
            dim=-1,
        )

        mask = torch.arange(0, logits.size(1), device=logits.device).unsqueeze(
            0
        ).unsqueeze(1).repeat(bsz, 1, 1) >= number.unsqueeze(1).unsqueeze(2)

        logits = logits.masked_fill(mask, float("-inf"))

        logits /= 0.1
        ctr_loss = -torch.nn.LogSoftmax(2)(logits)
        ctr_loss = ctr_loss.masked_fill(mask.transpose(1, 2) | mask, 0)
        token_level_ctr_loss = torch.diagonal(ctr_loss, dim1=-2, dim2=-1).sum()

        return token_level_ctr_loss

    def forward(self, model, sample, reduce=True, update_num=None):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, asr_loss, ext_mt_loss, seg_loss, ctr_loss = (
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
            torch.Tensor([0]).cuda(),
        )
        st_size, mt_size, asr_size, ext_mt_size = 0, 0, 0, 0

        update_num = model.encoder.num_updates

        if update_num is not None and update_num > 10000:
            # multipath training
            training_lagging_seg = random.randint(
                3, max(10, sample["transcription_tokens_num"].min().item())
            )
        else:
            training_lagging_seg = None

        if not model.training:
            training_lagging_seg = 3

        mode = sample["net_input"]["mode"]
        if mode == "st":
            st_loss, speech_encoder_out = self.forward_st(
                model,
                sample,
                update_num,
                reduce,
                training_lagging_seg=training_lagging_seg,
            )

            if self.seg_speech and (update_num is None or update_num > 4000):
                seg_loss = self.calculate_seg_num_loss(speech_encoder_out, sample)
            st_size = sample_size = sample["ntokens"]

            if self.mt_training and self.training:
                mt_loss, text_encoder_out = self.forward_mt(
                    model,
                    sample,
                    update_num,
                    reduce,
                    training_lagging_seg=training_lagging_seg,
                )
                mt_size = sample["ntokens"]
                if (
                    self.add_speech_seg_text_ctr
                    and self.seg_speech
                    and (update_num is None or update_num > 20000)
                ):
                    ctr_loss = self.calculate_expected_ctr_loss(
                        speech_encoder_out, text_encoder_out, sample
                    )

            if self.asr_training and self.training:
                asr_loss = self.forward_asr(
                    model,
                    sample,
                    update_num,
                    reduce,
                    speech_encoder_out=speech_encoder_out,
                    training_lagging_seg=training_lagging_seg,
                )
                asr_size = sample["src_ntokens"]

            loss = st_loss + mt_loss + asr_loss + seg_loss + 0.1 * ctr_loss

        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "asr_loss": asr_loss.data,
            "asr_sample_size": asr_size,
            "seg_loss": seg_loss.data,
            "ctr_loss": ctr_loss.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        seg_loss_sum = sum(log.get("seg_loss", 0) for log in logging_outputs)
        ctr_loss_sum = sum(log.get("ctr_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        asr_sample_size = sum(log.get("asr_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(
            log.get("ext_mt_sample_size", 0) for log in logging_outputs
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss",
            st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0,
            st_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "mt_loss",
            mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0,
            mt_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "asr_loss",
            asr_loss_sum / asr_sample_size / math.log(2) if asr_sample_size != 0 else 0,
            asr_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "seg_loss",
            seg_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0,
            st_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "ctr_loss",
            ctr_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0,
            st_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "ext_mt_loss",
            (
                ext_mt_loss_sum / ext_mt_sample_size / math.log(2)
                if ext_mt_sample_size != 0
                else 0
            ),
            ext_mt_sample_size,
            round=3,
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
