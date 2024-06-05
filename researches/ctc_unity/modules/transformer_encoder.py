# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from fairseq import utils
import torch.nn as nn
import math

from fairseq.models import FairseqEncoder
from fairseq.modules import LayerNorm, PositionalEmbedding, FairseqDropout
from ctc_unity.modules.transformer_layer import TransformerEncoderLayer


class UniTransformerEncoderNoEmb(FairseqEncoder):
    """Transformer encoder without token embeddings."""

    def __init__(self, args):
        super().__init__(None)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self._future_mask = torch.empty(0)
        self.unidirectional = getattr(args, "uni_encoder", False)

    def forward(
        self, x, encoder_padding_mask, return_all_hiddens=False, streaming_config=None
    ):

        encoder_states = []

        if streaming_config is None:
            extra = {
                "encoder_mask": (
                    self.buffered_future_mask(x) if self.unidirectional else None
                )
            }
        else:

            encoder_mask = None
            if (
                "encoder_mask" in streaming_config.keys()
                and streaming_config["encoder_mask"] is not None
            ):
                encoder_mask = streaming_config["encoder_mask"]
            else:
                encoder_mask = self.buffered_chunk_mask(
                    x, tgt_step=streaming_config["tgt_step"]
                )
            extra = {"encoder_mask": encoder_mask}

        for layer in self.layers:
            x = layer(x, encoder_padding_mask, extra=extra)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": (
                [encoder_padding_mask]
                if encoder_padding_mask is not None and encoder_padding_mask.any()
                else []
            ),  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

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

    def buffered_chunk_mask(self, tensor, tgt_step):
        dim = tensor.size(0)
        idx = torch.arange(0, dim, device=tensor.device).unsqueeze(1)
        idx = (idx // tgt_step + 1) * tgt_step
        idx = idx.clamp(1, dim)
        tmp = torch.arange(0, dim, device=tensor.device).unsqueeze(0).repeat(dim, 1)
        chunk_mask = torch.where(
            idx <= tmp, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        return chunk_mask[:dim, :dim]

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class UniTransformerEncoderWithEmb(FairseqEncoder):
    """Transformer encoder without token embeddings."""

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
    ):
        super().__init__(dictionary)

        self.dropout_module = FairseqDropout(args.dropout)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=False,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self._future_mask = torch.empty(0)
        self.unidirectional = getattr(args, "uni_encoder", False)

    def forward_embedding(self, src_tokens):
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        encoder_padding_mask,
        return_all_hiddens=False,
        streaming_config=None,
    ):

        x, encoder_embedding = self.forward_embedding(src_tokens)
        x = x.transpose(0, 1)

        encoder_states = []

        if streaming_config is None:
            extra = {
                "encoder_mask": (
                    self.buffered_future_mask(x) if self.unidirectional else None
                )
            }
        else:

            encoder_mask = None
            if (
                "encoder_mask" in streaming_config.keys()
                and streaming_config["encoder_mask"] is not None
            ):
                encoder_mask = streaming_config["encoder_mask"]
            else:
                encoder_mask = self.buffered_chunk_mask(
                    x, tgt_step=streaming_config["tgt_step"]
                )
            extra = {"encoder_mask": encoder_mask}

        for layer in self.layers:
            x = layer(x, encoder_padding_mask, extra=extra)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": (
                [encoder_padding_mask]
                if encoder_padding_mask is not None and encoder_padding_mask.any()
                else []
            ),  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

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

    def buffered_chunk_mask(self, tensor, tgt_step):
        dim = tensor.size(0)
        idx = torch.arange(0, dim, device=tensor.device).unsqueeze(1)
        idx = (idx // tgt_step + 1) * tgt_step
        idx = idx.clamp(1, dim)
        tmp = torch.arange(0, dim, device=tensor.device).unsqueeze(0).repeat(dim, 1)
        chunk_mask = torch.where(
            idx <= tmp, torch.tensor(float("-inf")), torch.tensor(0.0)
        )
        return chunk_mask[:dim, :dim]

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }
