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
from fairseq.criterions.tacotron2_loss import (
    Tacotron2Criterion,
    Tacotron2CriterionConfig,
)
from fairseq.criterions.speech_to_speech_criterion import (
    Tacotron2CriterionConfig,
    SpeechToUnit2passMultitaskTaskCriterion,
    SpeechToSpectrogram2passMultitaskTaskCriterion,
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechToUnit2passWaitkCriterionConfig(
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


@register_criterion(
    "speech_to_unit_2pass_waitk", dataclass=SpeechToUnit2passWaitkCriterionConfig
)
class SpeechToUnit2passWaitkMultitaskTaskCriterion(
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
                if self.k1 > 0
                else random.randint(
                    5,
                    1
                    + sample["net_input"]["src_tokens"].size(1)
                    * 40
                    // self.segment_size,
                )
            ),
            "k2": self.k2,
            "n1": self.n1,
            "n2": self.n2,
            "unit_per_subword": self.unit_per_subword,
            "segment_size": self.segment_size,
        }
        net_output, extra = model(**net_input_concat, streaming_config=streaming_config)
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, [net_output], sample, reduce=reduce
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
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, [net_output], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output
