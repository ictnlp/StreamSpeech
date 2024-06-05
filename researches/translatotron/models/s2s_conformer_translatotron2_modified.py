# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import OrderedDict
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.s2s_conformer_translatotron2 import (
    S2SpecT2ConformerModel,
    s2spect2_conformer_architecture_base,
)

logger = logging.getLogger(__name__)


@register_model("s2spect2_conformer_modified")
class S2SpecT2ConformerModelModified(S2SpecT2ConformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + TTS Transformer decoder
    Modified version: support load pretrained S2T model
    """

    @staticmethod
    def add_args(parser):
        S2SpecT2ConformerModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-mt-from",
            type=str,
            help="path to pretrained s2t transformer model",
        )

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args)
        base_model = cls(encoder, decoder)

        # set up multitask decoders
        base_model.mt_task_name = None
        base_model.multitask_decoders = {}
        has_first_pass_decoder = False
        for task_name, task_obj in task.multitask_tasks.items():
            if task_obj.is_first_pass_decoder:
                has_first_pass_decoder = True
                base_model.mt_task_name = task_name

            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )
            task_decoder = cls.build_multitask_decoder(
                task_obj.args,
                task_obj.target_dictionary,
                in_dim,
                task_obj.is_first_pass_decoder,
                getattr(args, "translation_decoder_layers", 4),
                getattr(args, "decoder_embed_dim", 256),
                getattr(args, "decoder_attention_heads", 4),
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        assert has_first_pass_decoder, "set at least one intermediate non-CTC decoder"

        # set up encoder on top of the auxiliary MT decoder
        if getattr(args, "synthesizer_encoder_layers", 0) > 0:
            base_model.synthesizer_encoder = cls.build_text_encoder(args)
        else:
            base_model.synthesizer_encoder = None

        if getattr(args, "load_pretrained_mt_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrained_mt_from
            )["model"]
            encoder_state_dict = OrderedDict()
            decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    encoder_state_dict[subkey] = state_dict[key]
                elif key.startswith("decoder"):
                    decoder_state_dict[key] = state_dict[key]
            base_model.encoder.load_state_dict(encoder_state_dict)
            base_model.multitask_decoders[base_model.mt_task_name].load_state_dict(
                decoder_state_dict
            )
            logger.info(
                f"Successfully load pretrained Conformer from {args.load_pretrained_mt_from}."
            )

        return base_model


@register_model_architecture(
    model_name="s2spect2_conformer_modified", arch_name="s2spect2_conformer_modified"
)
def s2spect2_conformer_modified_architecture_base(args):
    s2spect2_conformer_architecture_base(args)
