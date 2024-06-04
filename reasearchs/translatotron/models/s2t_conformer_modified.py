# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture

from fairseq.models.speech_to_text.s2t_conformer import (
    S2TConformerModel,
    conformer_base_architecture,
)


logger = logging.getLogger(__name__)


@register_model("s2t_conformer_modified")
class S2TConformerModelModified(S2TConformerModel):

    @staticmethod
    def add_args(parser):
        S2TConformerModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-s2t-from",
            type=str,
            help="path to pretrained s2t conformer model",
        )

    @classmethod
    def build_model(cls, args, task):
        base_model = super().build_model(args, task)

        if getattr(args, "load_pretrained_s2t_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrained_s2t_from
            )["model"]
            del state_dict["decoder.embed_tokens.weight"]
            del state_dict["decoder.output_projection.weight"]
            base_model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"Successfully load pretrained S2T model from {args.load_pretrained_s2t_from}."
            )

        return base_model


@register_model_architecture("s2t_conformer_modified", "s2t_conformer_modified")
def conformer_base_modified_architecture(args):
    conformer_base_architecture(args)
