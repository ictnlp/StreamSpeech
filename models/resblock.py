"""
Source code inspired from:
- fairseq: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/hifigan.py
- jik876: https://github.com/jik876/hifi-gan/blob/master/models.py

For detailed model architecture details, refer to the original paper: https://arxiv.org/abs/2010.05646

For in-depth details for residual blocks/nets, refer to its seminal paper: https://arxiv.org/abs/1512.03385
"""

import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn.functional import leaky_relu

from .utils import init_weights


# Empirically, 0.1 worked best in GAN-based vocoders for audio stability
# The small negative slope (0.1) allows a small gradient to flow even for negative activations
# Helpful in preventing "dead" neurons
LEAKY_RELU_SLOPE = 0.1


# Residual Blocks that form the Multi-Receptive Field Fusion (MRF) Module
class ResBlock(nn.Module):

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()  # with dilations
        self.convs2 = nn.ModuleList()  # no dilations

        # weight_norm() is crucial to stabilize training by normalizing the scale of each filter
        # Helps in avoiding exploding or vanishing gradients and improves convergence
        for dilation in dilations:
            # With dilations
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=((kernel_size * dilation - dilation) // 2)
                    )
                )
            )

            # No dilations
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=((kernel_size - 1) // 2)
                    )
                )
            )

        # Override default weights to stabilize training and reduce artifacts
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)


    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x

            x = leaky_relu(x, LEAKY_RELU_SLOPE)
            x = conv1(x)
            x = leaky_relu(x, LEAKY_RELU_SLOPE)
            x = conv2(x)
            x = x + residual

        return x


    def remove_weight_norm(self):
        """
        Use during inference, since normalization overhead is not needed anymore
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)
