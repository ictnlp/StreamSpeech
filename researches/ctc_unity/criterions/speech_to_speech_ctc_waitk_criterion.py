# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections import OrderedDict
import random
import torch
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion
from fairseq.criterions.label_smoothed_cross_entropy_with_rdrop import (
    RdropLabelSmoothedCrossEntropyCriterion,
    RdropLabelSmoothedCrossEntropyCriterionConfig,
    duplicate_input,
)
import torch.nn.functional as F
from fairseq.criterions.tacotron2_loss import (
    Tacotron2Criterion,
    Tacotron2CriterionConfig,
)
from fairseq.criterions.speech_to_speech_criterion import (
    Tacotron2CriterionConfig,
    SpeechToUnit2passMultitaskTaskCriterion,
    SpeechToSpectrogram2passMultitaskTaskCriterion,
)
from fairseq.data.data_utils import post_process

logger = logging.getLogger(__name__)


@dataclass
class SpeechToUnit2passCTCWaitkCriterionConfig(
    RdropLabelSmoothedCrossEntropyCriterionConfig
):
    k1: int = field(
        default=3,
        metadata={"help": "k1"},
    )
    k2: int = field(
        default=3,
        metadata={"help": "k1"},
    )
    n1: int = field(
        default=3,
        metadata={"help": "k1"},
    )
    n2: int = field(
        default=3,
        metadata={"help": "k1"},
    )
    unit_per_subword: int = field(
        default=10,
        metadata={"help": "k1"},
    )
    segment_size: int = field(
        default=280,
        metadata={"help": "k1"},
    )
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )


@register_criterion(
    "speech_to_unit_2pass_ctc_waitk", dataclass=SpeechToUnit2passCTCWaitkCriterionConfig
)
class SpeechToUnit2passCTCWaitkMultitaskTaskCriterion(
    SpeechToUnit2passMultitaskTaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
        k1=3,
        k2=1,
        n1=3,
        n2=3,
        unit_per_subword=10,
        segment_size=280,
        post_process="letter",
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            rdrop_alpha,
        )
        self.k1 = k1
        self.k2 = k2
        self.n1 = n1
        self.n2 = n2
        self.unit_per_subword = unit_per_subword
        self.segment_size = segment_size
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )

        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = post_process

    def forward(self, model, sample, reduce=True):
        net_input_concat = {
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "prev_output_tokens_mt": sample["multitask"][model.mt_task_name][
                "net_input"
            ]["prev_output_tokens"],
            "tgt_speaker": sample["net_input"].get("tgt_speaker", None),
            "return_all_hiddens": True,
        }
        if getattr(model, "asr_task_name", None) is not None:
            net_input_concat["prev_output_tokens_asr"] = sample["multitask"][
                model.asr_task_name
            ]["net_input"]["prev_output_tokens"]

        if self.rdrop_alpha > 0 or self.rdrop_alpha_mtl > 0:
            net_input_concat = duplicate_input(net_input_concat)

        streaming_config = {
            "k1": (
                self.k1
                if self.k1 >= 0
                else random.randint(
                    0,
                    1
                    + sample["net_input"]["src_tokens"].size(1)
                    * 40
                    // self.segment_size,
                )
            ),
            "n1": (
                self.n1
                if self.n1 >= 0
                else random.randint(
                    2, 1 + net_input_concat["prev_output_tokens_mt"].size(1)
                )
            ),
            "k2": (
                self.k2
                if self.k2 >= 0
                else random.randint(
                    0, 1 + net_input_concat["prev_output_tokens_mt"].size(1)
                )
            ),
            "n2": (
                self.n2
                if self.n1 >= 0
                else random.randint(
                    2, 1 + net_input_concat["prev_output_tokens_mt"].size(1)
                )
            ),
            "segment_size": self.segment_size,
        }
        net_output, extra = model(**net_input_concat, streaming_config=streaming_config)
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, [net_output, extra], sample, reduce=reduce
        )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy and not model.training:
            n_correct, total = self.compute_accuracy(model, [net_output, extra], sample)
            logging_output["n_correct"] = n_correct
            logging_output["total"] = total
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True).transpose(0, 1)
        target = model.get_targets(sample, net_output)

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        if net_output[-1]["decoder_padding_mask"] is not None:
            non_padding_mask = ~net_output[-1]["decoder_padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(0),), lprobs.size(1), dtype=torch.long
            )

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                target,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )

        if self.rdrop_alpha > 0:
            pad_mask = target[: target.size(0) // 2].unsqueeze(-1).eq(self.padding_idx)
            rdrop_kl_loss = compute_kl_loss(model, net_output, pad_mask)
            loss += self.rdrop_alpha * rdrop_kl_loss
        else:
            rdrop_kl_loss = loss.new_zeros(1)
        return loss, loss, rdrop_kl_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True).transpose(0, 1)
        target = model.get_targets(sample, net_output)

        if net_output[-1]["decoder_padding_mask"] is not None:
            non_padding_mask = ~net_output[-1]["decoder_padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_full(
                (lprobs.size(0),), lprobs.size(1), dtype=torch.long
            )

        logging_output = {}
        import editdistance

        with torch.no_grad():
            lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            wv_errs = 0
            for lp, t, inp_l in zip(
                lprobs_t,
                (
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"]
                ),
                input_lengths,
            ):
                lp = lp[:inp_l].unsqueeze(0)

                decoded = None

                p = (t != self.task.target_dictionary.pad()) & (
                    t != self.task.target_dictionary.eos()
                )
                targ = t[p]
                targ_units = self.task.target_dictionary.string(targ)
                targ_units_arr = targ.tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                c_len += len(targ_units_arr)

                targ_words = post_process(targ_units, self.post_process).split()

                pred_units = self.task.target_dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units, self.post_process).split()

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    w_errs += editdistance.eval(pred_words, targ_words)
                    wv_errs += editdistance.eval(pred_words_raw, targ_words)
                else:
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist

                w_len += len(targ_words)

            logging_output["wv_errors"] = wv_errs
            logging_output["w_errors"] = w_errs
            logging_output["w_total"] = w_len
            logging_output["c_errors"] = c_err
            logging_output["c_total"] = c_len
        return (
            logging_output["c_total"] - logging_output["c_errors"],
            logging_output["c_total"],
        )


def compute_kl_loss(model, net_output, pad_mask=None, reduce=True):
    net_prob = model.get_normalized_probs(net_output, log_probs=True)
    net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

    net_prob = net_prob.view(-1, net_prob.size(-1))
    net_prob_tec = net_prob_tec.view(-1, net_prob_tec.size(-1))

    p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
    p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

    p_loss = torch.nn.functional.kl_div(p, q_tec, reduction="none")
    q_loss = torch.nn.functional.kl_div(q, p_tec, reduction="none")

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    if reduce:
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
