# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import OrderedDict
from pathlib import Path
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.s2s_conformer_unity import (
    UnityConformerModel,
    unity_conformer_architecture_base,
)

from fairseq.models.transformer import (
    TransformerEncoderBase,
)
from fairseq.models.speech_to_speech.s2s_transformer import S2STransformerEncoder
from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerEncoder

logger = logging.getLogger(__name__)


@register_model("unity_transformer_modified")
class UnityTransformerModelModified(UnityConformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + Transformer discrete unit decoder
    Modified version: support load pretrained S2T model
    """

    @staticmethod
    def add_args(parser):
        UnityConformerModel.add_args(parser)
        parser.add_argument(
            "--load-pretrained-mt-from",
            type=str,
            help="path to pretrained s2t transformer model",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = S2STransformerEncoder(args)
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

    @classmethod
    def build_model(cls, args, task):
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(
            args,
            task.target_dictionary,
            aug_attn=getattr(args, "synthesizer_augmented_cross_attention", False),
        )
        base_model = cls(encoder, decoder)

        base_model.t2u_augmented_cross_attn = getattr(
            args, "synthesizer_augmented_cross_attention", False
        )

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

    # def forward(
    #     self,
    #     src_tokens,
    #     src_lengths,
    #     prev_output_tokens,
    #     prev_output_tokens_mt,
    #     tgt_speaker=None,
    #     return_all_hiddens=False,
    # ):
    #     mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")

    #     encoder_out = self.encoder(
    #         src_tokens,
    #         src_lengths=src_lengths,
    #         return_all_hiddens=return_all_hiddens,
    #     )

    #     # 1. MT decoder
    #     mt_decoder_out = mt_decoder(
    #         prev_output_tokens_mt,
    #         encoder_out=encoder_out,
    #     )
    #     x = mt_decoder_out[1]["inner_states"][-1]
    #     if mt_decoder.layer_norm is not None:
    #         x = mt_decoder.layer_norm(x)

    #     mt_decoder_padding_mask = None
    #     if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
    #         mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

    #     # 2. T2U encoder
    #     if self.synthesizer_encoder is not None:
    #         t2u_encoder_out = self.synthesizer_encoder(
    #             x,
    #             mt_decoder_padding_mask,
    #             return_all_hiddens=return_all_hiddens,
    #         )
    #     else:
    #         t2u_encoder_out = {
    #             "encoder_out": [x],  # T x B x C
    #             "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
    #         }

    #     # 3. T2U decoder
    #     if self.t2u_augmented_cross_attn:
    #         decoder_out = self.decoder(
    #             prev_output_tokens,
    #             encoder_out=encoder_out,
    #             encoder_out_aug=t2u_encoder_out,
    #         )
    #     else:
    #         decoder_out = self.decoder(
    #             prev_output_tokens,
    #             encoder_out=t2u_encoder_out,
    #         )
    #     if return_all_hiddens:
    #         decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
    #         decoder_out[-1]["encoder_padding_mask"] = encoder_out[
    #             "encoder_padding_mask"
    #         ]
    #     decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
    #     return decoder_out


@register_model_architecture(
    model_name="unity_transformer_modified", arch_name="unity_transformer_modified"
)
def unity_Transformer_modified_architecture_base(args):
    unity_conformer_architecture_base(args)
