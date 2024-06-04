# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import torch
from typing import OrderedDict

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq.models.speech_to_speech.modules.transformer_decoder_aug import (
    AugTransformerUnitDecoder,
)
from chunk_unity.modules.transformer_encoder import (
    UniTransformerEncoderNoEmb,
)
from chunk_unity.models.s2s_conformer import ChunkS2UTConformerModel
from fairseq.models.speech_to_speech.s2s_transformer import (
    base_multitask_text_transformer_decoder_arch,
    s2ut_architecture_base,
)
from chunk_unity.models.s2s_transformer import (
    TransformerUnitDecoder,
)
from fairseq.models.transformer import TransformerModelBase
from chunk_unity.modules.transformer_decoder import TransformerDecoder
from fairseq import checkpoint_utils

logger = logging.getLogger(__name__)


def multitask_text_transformer_decoder_arch(
    args, decoder_layers, decoder_embed_dim=256, decoder_attention_heads=4
):
    args.decoder_layers = decoder_layers
    args.decoder_embed_dim = decoder_embed_dim
    args.decoder_attention_heads = decoder_attention_heads
    base_multitask_text_transformer_decoder_arch(args)


@register_model("chunk_unity_conformer")
class ChunkUnityConformerModel(ChunkS2UTConformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + Transformer discrete unit decoder
    """

    @staticmethod
    def add_args(parser):
        ChunkS2UTConformerModel.add_args(parser)
        parser.add_argument(
            "--translation-decoder-layers",
            type=int,
            default=4,
            metavar="N",
            help="num decoder layers in the first-pass translation module",
        )
        parser.add_argument(
            "--synthesizer",
            default="transformer",
            choices=["transformer"],
            help="",
        )
        parser.add_argument(
            "--synthesizer-encoder-layers",
            type=int,
            default=0,
            metavar="N",
            help="num encoder layers in the second-pass synthesizer module",
        )
        parser.add_argument(
            "--synthesizer-augmented-cross-attention",
            action="store_true",
            default=False,
            help="augmented cross-attention over speech encoder output",
        )
        parser.add_argument(
            "--load-pretrained-mt-from",
            type=str,
            help="path to pretrained s2t transformer model",
        )
        parser.add_argument(
            "--uni-encoder",
            action="store_true",
            default=False,
            help="apply unidirectional encoder",
        )

    @classmethod
    def build_multitask_decoder(
        cls,
        args,
        tgt_dict,
        in_dim,
        is_first_pass_decoder,
        decoder_layers,
        decoder_embed_dim,
        decoder_attention_heads,
    ):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            if is_first_pass_decoder:
                multitask_text_transformer_decoder_arch(
                    decoder_args,
                    decoder_layers,
                    decoder_embed_dim,
                    decoder_attention_heads,
                )  # 4L
            else:
                base_multitask_text_transformer_decoder_arch(decoder_args)  # 2L
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            task_decoder = CTCDecoder(
                dictionary=tgt_dict,
                in_dim=in_dim,
            )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, aug_attn=False):
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )

        _args = copy.deepcopy(args)
        _args.encoder_embed_dim = args.decoder_embed_dim

        decoder_cls = AugTransformerUnitDecoder if aug_attn else TransformerUnitDecoder
        return decoder_cls(
            _args,
            tgt_dict,
            embed_tokens,
        )

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

    @classmethod
    def build_text_encoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.synthesizer_encoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return UniTransformerEncoderNoEmb(_args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_output_tokens_mt,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):
        mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )

        # 1. MT decoder
        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=encoder_out,
        )
        x = mt_decoder_out[1]["inner_states"][-1]
        if mt_decoder.layer_norm is not None:
            x = mt_decoder.layer_norm(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if self.synthesizer_encoder is not None:
            t2u_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
                return_all_hiddens=return_all_hiddens,
            )
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
            }

        # 3. T2U decoder
        if self.t2u_augmented_cross_attn:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                encoder_out_aug=t2u_encoder_out,
            )
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=t2u_encoder_out,
            )
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out


@register_model_architecture(
    model_name="chunk_unity_conformer", arch_name="chunk_unity_conformer"
)
def chunk_unity_conformer_architecture_base(args):
    args.conv_version = getattr(args, "conv_version", "convtransformer")
    args.attn_type = getattr(args, "attn_type", None)
    args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    args.max_source_positions = getattr(args, "max_source_positions", 6000)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
    s2ut_architecture_base(args)
