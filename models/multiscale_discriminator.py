"""
Multi-Scale Discriminator

The MSD evaluates audio quality at multiple time resolutions by running three sub-discriminators on raw,
2x downsampled, and 4x downsampled waveforms. This allows it to detect broad, long-range artifacts that
may be missed at a single scale. 

Each sub-discriminator uses strided and grouped 1D convolutions with LeakyReLU to analyze smoothed waveform structure, 
while spectral normalization is applied only to the first (raw) scale for training stability. 
Unlike the MPD, which inspects disjoint periodic samples, the MSD continuously analyzes the audio sequence 
to capture overall timbre, clarity, and temporal consistency.

Reference: https://arxiv.org/pdf/2010.05646 (Section 2.3 - Multi-Scale Discriminator)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from .utils import LEAKY_RELU_SLOPE


class ScaleDiscriminator(nn.Module):
    """
    Discriminator that inspects the waveform at different time resolutions.
    It applies strided 1D convolutions (with grouped kernels) to detect
    broad, long-range artifacts in the audio such as timbre or smoothness.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()

        # HiFi-GAN applies spectral norm only for the first scale.
        norm = spectral_norm if use_spectral_norm else weight_norm

        self.conv_layers = nn.ModuleList([
            norm(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=128,
                    kernel_size=15,
                    stride=1,
                    padding=7
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=41,
                    stride=2,
                    padding=20,
                    groups=4
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=41,
                    stride=2,
                    padding=20,
                    groups=16
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=41,
                    stride=1,
                    padding=20,
                    groups=16
                )
            ),
            norm(
                nn.Conv1d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=5,
                    stride=1,
                    padding=2
                )
            )
        ])

        self.conv_post = norm(
            nn.Conv1d(
                in_channels=1024,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )


    def forward(self, x: torch.Tensor):
        feature_maps = []

        for conv in self.conv_layers:
            x = conv(x)
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            feature_maps.append(x)

        # Final logit prediction
        logits = self.conv_post(x)
        feature_maps.append(logits)

        # Flatten time dimension
        logits = logits.flatten(1, -1)
        return logits, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """
    The MSD runs three identical ScaleDiscriminators on:
    - the original waveform
    - a 2x time-downsampled waveform
    - a 4x time-downsampled waveform

    Only the first scale uses spectral normalization to stabilize training.
    """

    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),   # highest stability
            ScaleDiscriminator(use_spectral_norm=False),  # more flexibility
            ScaleDiscriminator(use_spectral_norm=False),
        ])


    def forward(self, audio: torch.Tensor):
        scores = []
        feature_maps = []

        for disc in self.discriminators:
            score, feats = disc(audio)
            scores.append(score)
            feature_maps.append(feats)

            # Downsample for next discriminator
            audio = F.avg_pool1d(audio, kernel_size=4, stride=2, padding=2)

        return scores, feature_maps

