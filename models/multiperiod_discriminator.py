"""
Multi-Period Discriminator

The MPD is made up of several smaller discriminators, each focusing on a different time period 
of the audio (e.g., every 2, 3, 5, 7, or 11 samples). This helps the model detect repeating patterns 
like pitch or rhythm that occur at different frequencies. 

Each sub-discriminator reshapes the 1D waveform into a 2D map so it can look at periodic structures 
more effectively using 2D convolutions. This design allows MPD to catch unnatural repeating noises or 
distortions in generated speech while keeping gradients connected across all time steps for stable training.

Reference: https://arxiv.org/pdf/2010.05646 (Section 2.3 - Multi-Period Discriminator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from .utils import LEAKY_RELU_SLOPE, get_padding


class PeriodDiscriminator(nn.Module):
    """
    Discriminator that inspects the waveform periodically.
    It reshapes the input into a 2D map [batch, 1, time//period, period]
    to detect periodic patterns in the waveform.
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period

        # According to paper, they used weight_norm for MPD
        norm = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ),
            norm(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=128,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ),
            norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=512,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ),
            norm(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ),
            norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(get_padding(kernel_size, 1), 0)
                )
            ),
        ])

        self.conv_post = norm(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0)
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: waveform [B, 1, T]
        Returns:
            score: final scalar logits
            feature_maps: list of intermediate activations for feature matching
        """

        batch_size, channels, time_steps = x.shape

        # Ensure divisible by period
        if time_steps % self.period != 0:
            pad_len = self.period - (time_steps % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            time_steps = time_steps + pad_len

        # Reshape from 1D to 2D [B, 1, T//P, P]
        x = x.view(batch_size, channels, time_steps // self.period, self.period)

        feature_maps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LEAKY_RELU_SLOPE)
            feature_maps.append(x)

        score = self.conv_post(x)
        feature_maps.append(score)

        # Flatten score map for discriminator loss
        return score.flatten(1, -1), feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    Combines multiple PeriodDiscriminators, each with different periodicities.
    Detects pitch-related periodic artifacts at various temporal resolutions.
    """

    # Used the same values as original HiFi-GAN paper
    # Chose these specific values to minimize overlap
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [ PeriodDiscriminator(p) for p in periods ]
        )


    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio: waveform [B, 1, T]
        Returns:
            scores: List of logits
            feature_maps: List of intermediate activations for feature matching
        """
        scores = []
        feature_maps = []

        for discriminator in self.sub_discriminators:
            score, feats = discriminator(audio)
            scores.append(score)
            feature_maps.append(feats)

        return scores, feature_maps