# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq.models.speech_to_text import S2TTransformerEncoder
from fairseq.models.text_to_speech import TTSTransformerDecoder
from fairseq.models.transformer import Linear, TransformerModelBase

from ctc_unity.modules.transformer_decoder import TransformerDecoder

logger = logging.getLogger(__name__)


class CTCTransformerUnitDecoder(TransformerDecoder):
    """Based on Transformer decoder, with support to decoding stacked units"""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn, output_projection
        )
        self.n_frames_per_step = args.n_frames_per_step

        self.out_proj_n_frames = (
            Linear(
                self.output_embed_dim,
                self.output_embed_dim * self.n_frames_per_step,
                bias=False,
            )
            if self.n_frames_per_step > 1
            else None
        )

        self.ctc_upsample_rate = args.ctc_upsample_rate

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        streaming_config=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            streaming_config=streaming_config,
        )

        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.output_layer(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if (
                incremental_state is None and self.n_frames_per_step > 1
            ):  # teacher-forcing mode in training
                x = x[
                    :, : -(self.n_frames_per_step - 1), :
                ]  # remove extra frames after <eos>

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        streaming_config=None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            streaming_config,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        streaming_config=None,
    ):

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        slen, bs, embed = enc.size()
        x = (
            enc.unsqueeze(1)
            .repeat(1, self.ctc_upsample_rate, 1, 1)
            .contiguous()
            .view(slen * self.ctc_upsample_rate, bs, embed)
        )
        _x = x.contiguous()

        prev_key_length = 0
        if (
            incremental_state is not None
            and self.layers[0].self_attn._get_input_buffer(incremental_state) != {}
        ):
            prev_key_length = (
                self.layers[0]
                .self_attn._get_input_buffer(incremental_state)["prev_key"]
                .size(-2)
            )

            if x.size(0) > prev_key_length:
                x = x[prev_key_length:]

        if self.embed_positions is not None:
            positions = self.embed_positions(
                x[:, :, 0], incremental_state=incremental_state
            )

        x += positions
        x = self.dropout_module(x)

        self_attn_padding_mask: Optional[Tensor] = None

        if padding_mask is not None and (
            self.cross_self_attention or padding_mask.any()
        ):
            self_attn_padding_mask = (
                padding_mask.unsqueeze(2)
                .repeat(1, 1, self.ctc_upsample_rate)
                .contiguous()
                .view(bs, slen * self.ctc_upsample_rate)
            )

        if streaming_config is not None:
            if (
                "streaming_mask" in streaming_config.keys()
                and streaming_config["streaming_mask"] is not None
            ):
                streaming_mask = streaming_config["streaming_mask"]
                streaming_mask = streaming_mask[:, prev_key_length:]
            else:
                streaming_mask = self.build_streaming_mask(
                    x,
                    enc.size(0),
                    _x.size(0),
                    streaming_config["src_wait"],
                    streaming_config["src_step"],
                    streaming_config["src_step"] * self.ctc_upsample_rate,
                )
                streaming_mask = streaming_mask[prev_key_length:]

        else:
            streaming_mask = None

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):

            self_attn_mask = self.buffered_future_mask(_x)
            self_attn_mask = self_attn_mask[-1 * x.size(0) :]

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                extra={"streaming_mask": streaming_mask},
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {
            "attn": [attn],
            "inner_states": inner_states,
            "decoder_padding_mask": self_attn_padding_mask,
        }

    def build_streaming_mask(self, x, src_len, tgt_len, src_wait, src_step, tgt_step):
        idx = torch.arange(0, tgt_len, device=x.device).unsqueeze(1)
        idx = (idx // tgt_step + 1) * src_step + src_wait
        idx = idx.clamp(1, src_len)
        tmp = torch.arange(0, src_len, device=x.device).unsqueeze(0).repeat(tgt_len, 1)
        return tmp >= idx

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]
