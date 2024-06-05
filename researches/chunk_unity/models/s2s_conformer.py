# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from email.policy import default
import logging
from pathlib import Path

import torch

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.speech_to_speech.s2s_transformer import S2UTTransformerModel
from chunk_unity.models.s2t_conformer import ChunkS2TConformerEncoder
from fairseq.models.transformer import Linear

logger = logging.getLogger(__name__)


def build_s2s_chunk_conformer_encoder(args):
    encoder = ChunkS2SConformerEncoder(args)
    pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
    if pretraining_path is not None:
        if not Path(pretraining_path).exists():
            logger.warning(
                f"skipped pretraining because {pretraining_path} does not exist"
            )
        else:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=pretraining_path
            )
            logger.info(f"loaded pretrained encoder from: {pretraining_path}")
    return encoder


class ChunkS2SConformerEncoder(ChunkS2TConformerEncoder):
    """Based on S2T transformer encoder, with support
    to incorporate target speaker embedding."""

    def __init__(self, args):
        super().__init__(args)

        self.spk_emb_proj = None
        if args.target_speaker_embed:
            self.spk_emb_proj = Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

    def forward(
        self, src_tokens, src_lengths, tgt_speaker=None, return_all_hiddens=False
    ):
        out = super().forward(src_tokens, src_lengths, return_all_hiddens)

        if self.spk_emb_proj:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x

        return out


class ChunkS2UTConformerModel(S2UTTransformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + Transformer discrete unit decoder
    """

    @staticmethod
    def add_args(parser):
        S2UTTransformerModel.add_args(parser)
        parser.add_argument(
            "--depthwise-conv-kernel-size",
            type=int,
            metavar="N",
            help="kernel size of depthwise convolution layers",
        )
        parser.add_argument(
            "--attn-type",
            type=str,
            metavar="STR",
            help="If not specified uses fairseq MHA. Other valid option is espnet for using conformer",
        )
        parser.add_argument(
            "--pos-enc-type",
            type=str,
            metavar="STR",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            metavar="N",
            default=-1,
            help="chunk size",
        )

    @classmethod
    def build_encoder(cls, args):
        return build_s2s_chunk_conformer_encoder(args)
