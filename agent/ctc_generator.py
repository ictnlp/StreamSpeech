import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion

logger = logging.getLogger(__name__)


class CTCSequenceGenerator(nn.Module):
    def __init__(self, tgt_dict, models, use_incremental_states=False):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.unk = tgt_dict.unk()
        self.models = models
        self.tgt_dict = tgt_dict
        self.use_incremental_states = use_incremental_states
        self.incremental_states = None

    def reset_incremental_states(self):
        self.incremental_states = None

    @torch.no_grad()
    def generate(self, encoder_out, prefix=None, aux_task_name=None, **kwargs):
        if self.use_incremental_states:
            if self.incremental_states is None:
                incremental_states = torch.jit.annotate(
                    List[Dict[str, Dict[str, Optional[Tensor]]]],
                    [
                        torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                        for i in range(1)
                    ],
                )
                self.incremental_states = incremental_states
            else:
                incremental_states = self.incremental_states
        else:
            incremental_states = None

        # currently only support viterbi search for stacked units
        model = self.models[0]
        model.eval()

        max_len = model.max_decoder_positions()
        # TODO: incorporate max_len_a and max_len_b

        incremental_state = {}
        pred_out, attn, scores = [], [], []

        prev_output_tokens = None
        decoder_name = f"{aux_task_name}_decoder" if aux_task_name else "decoder"
        ctc_decoder = getattr(model, decoder_name)
        ctc_out, ctc_extra = ctc_decoder(
            None,
            encoder_out=encoder_out,
            incremental_state=(
                incremental_states[0] if self.use_incremental_states else None
            ),
            **kwargs,
        )
        lprobs = model.get_normalized_probs([ctc_out], log_probs=True)

        # never select pad, unk
        lprobs[:, :, self.pad] = -math.inf
        lprobs[:, :, self.unk] = -math.inf

        cur_pred_lprob, cur_pred_out = torch.max(lprobs, dim=2)
        scores = cur_pred_lprob
        pred_out = cur_pred_out

        attn = ctc_extra["attn"][0]
        alignment = None

        def _ctc_postprocess(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [
                v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]
            ]
            hyp = [
                v
                for v in deduplicated_toks
                if (v != self.tgt_dict.blank_index) and (v != self.tgt_dict.pad_index)
            ]
            return torch.tensor(hyp)

        if prefix is not None:
            if self.use_incremental_states:
                pred_out = torch.cat((prefix, pred_out), dim=1)
            else:
                pred_out = torch.cat((prefix, pred_out[:, prefix.size(1) :]), dim=1)

        hypos = [
            [
                {
                    "tokens": _ctc_postprocess(pred_out[b]),
                    "org_tokens": pred_out[b],
                    "attn": None,
                    "alignment": None,
                    "positional_scores": scores[b],
                    "score": utils.item(scores[b].sum().data),
                }
            ]
            for b in range(pred_out.size(0))
        ]

        return hypos
