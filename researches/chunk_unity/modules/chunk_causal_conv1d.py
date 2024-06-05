##########################################
# Chunk-based Conv1d in StreamSpeech
#
# StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning (ACL 2024)
##########################################

from typing import Optional

import torch
import torch.nn.functional as F


class ChunkCausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        chunk_size=8,
    ):
        super(ChunkCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size // 2) * dilation
        self.chunk_size = chunk_size

    def forward(self, input):
        if self.chunk_size > 0 and self.chunk_size < 999:
            self.__k = self.__padding + self.chunk_size
            output_len = (
                input.size(-1) + 2 * self.__padding - self.kernel_size[0]
            ) // self.stride[0] + 1
            padded_input = self.pad_to_chunk_size(input)
            unfolded_input = padded_input.unfold(
                -1, self.__k, self.__k - self.__padding
            )
            unfolded_input = F.pad(unfolded_input, (0, self.__padding))
            bsz, n_channels, chunks, seq_length = unfolded_input.size()
            unfolded_input = (
                unfolded_input.transpose(1, 2)
                .contiguous()
                .view(-1, n_channels, seq_length)
            )
            res = super(ChunkCausalConv1d, self).forward(unfolded_input)
            res = (
                res.contiguous()
                .view(bsz, chunks, self.out_channels, -1)
                .transpose(1, 2)
            )
            res = res.contiguous().view(bsz, self.out_channels, -1)[:, :, :output_len]
        else:
            unfolded_input = F.pad(input, (self.__padding, 0))
            unfolded_input = F.pad(unfolded_input, (0, self.__padding))
            res = super(ChunkCausalConv1d, self).forward(unfolded_input)

        return res

    def pad_to_chunk_size(self, input_tensor):
        batch_size, num_channels, seq_length = input_tensor.size()

        input_tensor = F.pad(input_tensor, (self.__padding, 0))
        padding_size = (
            self.chunk_size - (seq_length % self.chunk_size)
        ) % self.chunk_size
        padded_tensor = F.pad(input_tensor, (0, padding_size))
        return padded_tensor
