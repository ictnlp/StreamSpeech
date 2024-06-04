#!/usr/bin/env python3

import logging
import math
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.modules.convolution import infer_conv_output_dim
from fairseq.models.speech_to_text.convtransformer import base_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.speech_to_text.modules.convolution import (
    Conv1dSubsampler,
    Conv2dSubsampler,
)

from fairseq.distributed import fsdp_wrap

from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from diseg.modules.seg_encoder_layer import SegEncoderLayer
from diseg.modules.waitseg_decoder_layer import WaitSegDecoderLayer

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

logger = logging.getLogger(__name__)


@register_model("convtransformer_seg")
class ConvTransformerModelWac2VecSeg(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="INT",
            help="the number of output channels of conv layer",
        )
        # parser.add_argument(
        #     "--w2v2-model-path",
        #     default="/path/wav2vec_small.pt",
        #     type=str,
        #     help="path to wav2vec model",
        # )
        parser.add_argument(
            "--uni-encoder", default=False, type=bool, help="unidirectional encoder"
        )
        parser.add_argument(
            "--seg-encoder-layers",
            default=1,
            type=int,
            help="number of seg encoder layers",
        )

        # parser.add_argument(
        #     "--uni-wav2vec", default=False, type=bool, help="unidirectional encoder"
        # )
        # pretrain
        parser.add_argument(
            "--load-pretrained-mt-encoder-decoder-from",
            type=str,
            help="model to take mt encoder/decoder weight from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-full-st-from",
            type=str,
            help="model to take full st weight from (for initialization)",
        )
        # seg
        parser.add_argument(
            "--noise-var", type=float, default=1.0, help="Variance of discretness noise"
        )
        parser.add_argument(
            "--noise-mean", type=float, default=0.0, help="Mean of discretness noise"
        )

    @classmethod
    def build_encoder(cls, args, task, embed_tokens):
        encoder = ConvTransformerSegEncoder(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = WaitSegTransformerDecoder(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        # load pretrained mt models
        mt_pretrained_path = getattr(
            args, "load_pretrained_mt_encoder_decoder_from", None
        )
        if mt_pretrained_path is not None and Path(mt_pretrained_path).exists():
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(mt_pretrained_path)[
                "model"
            ]
            mt_encoder_state_dict = OrderedDict()
            mt_decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if "wav2vec_model" in key or "subsampler" in key:
                    continue
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    mt_encoder_state_dict[subkey] = state_dict[key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    mt_decoder_state_dict[subkey] = state_dict[key]
            encoder.load_state_dict(mt_encoder_state_dict, strict=False)
            decoder.load_state_dict(mt_decoder_state_dict, strict=False)
            logger.info(
                f"load pretrained mt encoder/decoder from: {mt_pretrained_path}"
            )
        full_st_pretrained_path = getattr(args, "load_pretrained_full_st_from", None)
        if (
            full_st_pretrained_path is not None
            and Path(full_st_pretrained_path).exists()
        ):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                full_st_pretrained_path
            )["model"]
            st_encoder_state_dict = OrderedDict()
            st_decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    st_encoder_state_dict[subkey] = state_dict[key]
                if key.startswith("decoder"):
                    subkey = key[len("decoder") + 1 :]
                    st_decoder_state_dict[subkey] = state_dict[key]
            encoder.load_state_dict(st_encoder_state_dict, strict=False)
            decoder.load_state_dict(st_decoder_state_dict, strict=False)
            logger.info(f"load pretrained full st from: {full_st_pretrained_path}")

        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        mode="st",
        update_num=None,
        seg_speech=False,
        speech_encoder_out=None,
        training_lagging_seg=None,
    ):
        if speech_encoder_out:
            encoder_out = speech_encoder_out
        else:
            encoder_out = self.encoder(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                mode=mode,
                update_num=update_num,
                seg_speech=seg_speech,
            )
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            training_lagging_seg=training_lagging_seg,
        )
        return decoder_out, encoder_out


class ConvTransformerSegEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(
        self,
        args,
        dictionary=None,
        embed_tokens=None,
    ):
        """Construct an Encoder object."""
        super().__init__(None)
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )
        transformer_input_dim = infer_conv_output_dim(
            self.in_channels, self.input_dim, args.conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)

        # use no conv
        self.dropout = args.dropout
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        max_source_positions = args.max_source_positions
        if max_source_positions < 3200000:
            max_source_positions = 3200000
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.embed_tokens = embed_tokens
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(
                embed_tokens.embedding_dim, export=export
            )
        else:
            self.layernorm_embedding = None

        self.uni_encoder = getattr(args, "uni_encoder", False)

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [SegEncoderLayer(args) for i in range(args.seg_encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self._future_mask = torch.empty(0)

        self.w2 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.w1 = nn.Linear(args.encoder_embed_dim, 1)

        self.noise_mean = getattr(args, "noise_mean", 0.0)
        self.noise_var = getattr(args, "noise_var", 1.0)

    def _get_w2v_feature(self, src_tokens, src_lengths):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_feature, padding_mask = self.wav2vec_model.extract_features(
            src_tokens, padding_mask
        )
        output_length = (1 - padding_mask.int()).sum(dim=1)

        return w2v_feature, padding_mask, output_length

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self, src_tokens, src_lengths, mode="st", update_num=None, seg_speech=False
    ):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if mode == "st":
            # w2v_feature, encoder_padding_mask, input_lengths = self._get_w2v_feature(
            #     src_tokens, src_lengths
            # )

            # x = torch.transpose(w2v_feature, 1, 0)

            # x = self.dim_proj(x)
            # x = x_emb = self.embed_scale * x
            # seg_x = x.detach()

            # encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            # positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            # x += positions

            # if self.layernorm_embedding is not None:
            #     x = self.layernorm_embedding(x)
            # x = self.dropout_module(x)

            # bsz, max_seq_len, _ = src_tokens.size()
            # x = (
            #     src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            #     .transpose(1, 2)
            #     .contiguous()
            # )
            # x = self.conv(x)
            # bsz, _, output_seq_len, _ = x.size()
            # x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
            # x = self.out(x)
            # x = x_emb = self.embed_scale * x
            # seg_x = x.detach()

            # subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
            # input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
            # input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(
            #     input_len_0.device
            # )
            # input_lengths = torch.min(input_len_0, input_len_1)

            # encoder_padding_mask = lengths_to_padding_mask(input_lengths)

            # positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            # x += positions

            x, input_lengths = self.subsample(src_tokens, src_lengths)
            x = x_emb = self.embed_scale * x
            seg_x = x.detach()

            encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            # x = self.dropout_module(x)

            # if self.layernorm_embedding is not None:
            #     x = self.layernorm_embedding(x)
            x = self.dropout_module(x)

        else:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
            x, x_emb = self.forward_embedding(src_tokens)
            if has_pads:
                x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
            x = x.transpose(0, 1)

        if seg_speech:

            seg_energy = self.w1(torch.relu(self.w2(seg_x))).transpose(0, 1)

            noise = 0
            if (
                self.training
                and self.noise_var > 0
                and not (update_num is None or update_num <= 4000)
            ):
                var = self.noise_var
                noise = torch.normal(
                    self.noise_mean, var, seg_energy.size(), device=seg_energy.device
                ).type_as(seg_energy)

            seg_prob = torch.sigmoid(seg_energy + noise).squeeze(-1)
            seg_prob = seg_prob.masked_fill(encoder_padding_mask, 0.0)
            _seg_prob = torch.sigmoid(seg_energy).squeeze(-1)
            if not self.training:
                seg_prob = seg_prob.round()
            seg_weight = self.seg2beta_weight(seg_prob)
        else:
            seg_weight = None
            _seg_prob = None
        for seg_layer in self.transformer_layers:
            x = seg_layer(
                x,
                encoder_padding_mask,
                (
                    self.buffered_future_mask(x)
                    if self.uni_encoder or mode == "mt"
                    else None
                ),
                seg_weight=seg_weight if seg_speech else None,
            )

        maybe_encoder_padding_mask = encoder_padding_mask
        return {
            "encoder_out": [x],
            "encoder_padding_mask": (
                [maybe_encoder_padding_mask]
                if maybe_encoder_padding_mask is not None
                else []
            ),
            "encoder_embedding": [x_emb],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [input_lengths] if mode == "st" else [src_lengths],
            "seg_prob": _seg_prob,
        }

    def seg2beta_weight(self, seg_prob):
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
            cumprod_eq * left.type_as(cumprod_eq)
            + cumprod_eq * equal.type_as(cumprod_eq)
            + cumprod_right * right.type_as(cumprod_right)
        )
        return seg_weight

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim], device=tensor.device)),
                1,
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(1, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)
        if encoder_out["seg_prob"] is not None:
            seg_prob = encoder_out["seg_prob"].index_select(0, new_order)
        else:
            seg_prob = None
        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
            "seg_prob": seg_prob,
        }


class WaitSegTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
                BaseLayer(args),
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = WaitSegDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

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
        training_lagging_seg=None,
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
            training_lagging_seg=training_lagging_seg,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        training_lagging_seg=None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            training_lagging_seg=training_lagging_seg,
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
        training_lagging_seg=None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None:
            enc = encoder_out["encoder_out"][0]
            padding_mask = encoder_out["encoder_padding_mask"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=False,  # bool((idx == alignment_layer)),
                need_head_weights=False,  # bool((idx == alignment_layer)),
                seg_prob=encoder_out["seg_prob"],
                training_lagging_seg=training_lagging_seg,
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

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict["{}.embed_positions._float_tensor".format(name)] = (
                torch.FloatTensor(1)
            )

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict["{}.layers.{}.{}.{}".format(name, i, new, m)] = (
                            state_dict[k]
                        )
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


# @register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
# def base_architecture(args):
#     args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_layers = getattr(args, "encoder_layers", 6)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
#     args.decoder_ffn_embed_dim = getattr(
#         args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
#     )
#     args.decoder_layers = getattr(args, "decoder_layers", 6)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
#     args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.0)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.0)
#     args.activation_fn = getattr(args, "activation_fn", "relu")
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
#     args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
#     args.share_decoder_input_output_embed = getattr(
#         args, "share_decoder_input_output_embed", False
#     )
#     args.no_token_positional_embeddings = getattr(
#         args, "no_token_positional_embeddings", False
#     )
#     args.adaptive_input = getattr(args, "adaptive_input", False)
#     args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
#
#     args.decoder_output_dim = getattr(
#         args, "decoder_output_dim", args.decoder_embed_dim
#     )
#     args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
#     args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
#     args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
#     args.max_source_positions = getattr(args, "max_source_positions", 3000)
#     args.max_target_positions = getattr(args, "max_target_positions", 1024)
#     args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
#     args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


@register_model_architecture("convtransformer_seg", "convtransformer_espnet_seg")
def convtransformer_espnet(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


@register_model_architecture("convtransformer_seg", "convtransformer_espnet_base_seg")
def convtransformer_espnet_base(args):
    # args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    # args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    # args.encoder_layers = getattr(args, "encoder_layers", 6)
    # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    # args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    # args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    # args.decoder_ffn_embed_dim = getattr(
    #     args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    # )
    # args.decoder_layers = getattr(args, "decoder_layers", 6)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    # args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    # args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    # args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    # args.activation_fn = getattr(args, "activation_fn", "relu")
    # args.dropout = getattr(args, "dropout", 0.1)
    # args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    # args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    # args.share_decoder_input_output_embed = getattr(
    #     args, "share_decoder_input_output_embed", False
    # )
    # args.no_token_positional_embeddings = getattr(
    #     args, "no_token_positional_embeddings", False
    # )
    # args.adaptive_input = getattr(args, "adaptive_input", False)
    # args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    # args.decoder_output_dim = getattr(
    #     args, "decoder_output_dim", args.decoder_embed_dim
    # )
    # args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    # args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    # args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    # args.max_source_positions = getattr(args, "max_source_positions", 3000)
    # args.max_target_positions = getattr(args, "max_target_positions", 1024)
    # args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    # args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)
    # args.seg_encoder_layers = getattr(args, "seg_encoder_layers", 6)

    # args.input_channels = getattr(args, "input_channels", 1)
    # args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    # args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    # # args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    # args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.input_channels = getattr(args, "input_channels", 1)
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")  # for Conv1d
    args.conv_channels = getattr(args, "conv_channels", 1024)  # for Conv1d
    args.conv_out_channels = getattr(args, "conv_out_channels", 256)  # for Conv2d
    args.conv_version = getattr(args, "conv_version", "s2t_transformer")
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.seg_encoder_layers = getattr(args, "seg_encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
