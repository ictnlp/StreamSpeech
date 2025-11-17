"""
Source code inspired from:
- fairseq:
    - Core HiFi-GAN: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/hifigan.py
    - Unit HiFi-GAN: https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/text_to_speech/codehifigan.py
- jik876: https://github.com/jik876/hifi-gan/blob/master/models.py

For detailed model architecture details, refer to the original paper: https://arxiv.org/abs/2010.05646
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from .resblock import ResBlock
from .film import FiLM
from .utils import init_weights


# Empirically, 0.1 worked best in GAN-based vocoders for audio stability
# The small negative slope (0.1) allows a small gradient to flow even for negative activations
# Helpful in preventing "dead" neurons
LEAKY_RELU_SLOPE = 0.1


class UnitHiFiGANGenerator(nn.Module):
    """
    ## Overview
    Unit-based HiFi-GAN Generator with optional FiLM conditioning 
    that converts discrete speech units into audio waveforms.

    - **Input**:  Discrete unit IDs (LongTensor [B, T])
    - **Output**: Audio waveform (FloatTensor [B, 1, L])

    ## Requirements
    Config dictionary (`config.json`) expected to contain:

    ### Embedding/Input
    - `num_embeddings`: `int`            (e.g., `1000`)
    - `embedding_dim`: `int`             (e.g., `128`)
    - `model_in_dim`: `int`              (e.g., `128` - must match channel count fed to `conv_pre`)

    ### Generator
    - `upsample_initial_channel`: `int`                      (e.g., `512`)
    - `upsample_rates`: `List[int]`                          (e.g., `[5,4,4,2,2]`)
    - `upsample_kernel_sizes`: `List[int]`                   (e.g., `[11,8,8,4,4]`)
    - `resblock_kernel_sizes`: `List[int]`                   (e.g., `[3,7,11]`)
    - `resblock_dilation_sizes`: `List[Tuple[int,int,int]]`  (e.g., `[(1,3,5), ...]`)

    [Link to downloadable config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
    """

    def __init__(self, config: dict, use_film: bool = False):
        super().__init__()
        self.use_film = use_film

        # Converts unit IDs to become continuous, learnable feature vectors
        # Should have shape (1000, 128)
        self.dict = nn.Embedding(config["num_embeddings"], config["embedding_dim"])

        # Sanity Check: embedding_dim must equal model_in_dim for checkpoint compatibility
        in_dim = config.get("model_in_dim", config["embedding_dim"])
        assert in_dim == config["embedding_dim"], (
            f"model_in_dim ({in_dim}) must equal embedding_dim ({config['embedding_dim']})"
        )

        # Converts embedding vectors into the generator’s internal channels
        # input channel (128) -> output channel (512)
        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=config["upsample_initial_channel"],
                kernel_size=7,
                padding=3
            )
        )

        # Upsampling x Multi-Receptive Field Fusion (MRF) module x FiLM conditioning
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.film_layers = nn.ModuleList() if self.use_film else None

        in_ch = config["upsample_initial_channel"]

        for stride, k_up in zip(config["upsample_rates"], config["upsample_kernel_sizes"]):
            out_ch = in_ch // 2

            # Upsampling (ConvTranspose)
            # At each stage, channels halve (e.g., 512 -> 256 -> 128 -> ...)
            self.ups.append(
                weight_norm(
                    # Upsamples time by stride (e.g., 5, 4, 4, 2, 2)
                    nn.ConvTranspose1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=k_up,
                        stride=stride,
                        padding=(k_up - stride) // 2
                    )
                )
            )

            # MRF
            # For each upsample stage, create several ResBlocks:
            #   e.g., kernels [3, 7, 11] with dilations (1, 3, 5) for each
            for ks, ds in zip(config["resblock_kernel_sizes"], config["resblock_dilation_sizes"]):
                self.resblocks.append(
                    ResBlock(
                        channels=out_ch,
                        kernel_size=ks,
                        dilations=tuple(ds)
                    )
                )

            # FiLM
            if self.use_film:
                self.film_layers.append(
                    FiLM(
                        in_channels=out_ch,
                        cond_dim=config.get("film_cond_dim", 512),
                        use_mlp=config.get("use_film_mlp", False),
                        hidden_dim=config.get("film_hidden_dim", 256),
                        dropout_p=config.get("film_dropout_p", 0.1),
                    )
                )

            # Update for next iteration
            in_ch = out_ch

        # Converts the final features into 1-channel waveform output
        self.conv_post = weight_norm(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=1,
                kernel_size=7,
                padding=3
            )
        )

        self.num_kernels = len(config["resblock_kernel_sizes"])

        # Project speaker + emotion embeddings (960 → 512)
        # (192 speaker + 768 emotion = 960 total dims)
        self.cond_proj = nn.Linear(960, config.get("film_cond_dim", 512))

        # Override default weights to stabilize training and reduce artifacts
        self.conv_pre.apply(init_weights)
        for up in self.ups:
            up.apply(init_weights)
        self.conv_post.apply(init_weights)


    def forward(
        self,
        units: torch.LongTensor,
        speaker: torch.Tensor | None = None,
        emotion: torch.Tensor | None = None,
        global_step: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            units: [B, T] discrete speech unit IDs.
            speaker: [B, D_s] speaker embedding (optional).
            emotion: [B, D_e] emotion embedding (optional).
        """

        # 1. Embed discrete speech units -> [B, C, T]
        x = self.dict(units).transpose(1, 2)

        # 2. Pre-conv to match generator's internal channels
        x = self.conv_pre(x)

        # 3. Concatenate conditioning if FiLM is active
        cond = None
        if self.use_film:
            if speaker is not None and emotion is not None:
                speaker_norm = F.normalize(speaker, dim=-1)
                emotion_norm = F.normalize(emotion, dim=-1)
                cond = torch.cat([speaker_norm, emotion_norm], dim=-1)
                # cond = torch.cat([speaker, emotion], dim=-1)
            elif speaker is not None:
                cond = F.normalize(speaker, dim=-1)
                # cond = speaker
            elif emotion is not None:
                cond = F.normalize(emotion, dim=-1)
                # cond = emotion

            if cond is not None:
                # NOTE: Removed normalization temporarily since it might be erasing meaningful magnitude differences from the embeddings
                # Normalize to prevent FiLM from over-conditioning due to large embedding magnitudes
                # cond = cond / (cond.norm(dim=-1, keepdim=True) + 1e-8)
                cond = self.cond_proj(cond)

        # 4. Upsample x FiLM (optional) x MRF 
        for i, upsample in enumerate(self.ups):
            # Upsample
            x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
            x = upsample(x)

            # Apply FiLM to inject speaker/emotion conditioning
            if self.use_film and cond is not None:
                self.film_layers[i].global_step = global_step
                x = self.film_layers[i](x, cond)

            # Sum & average MRF outputs
            res_outputs = []
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                y = self.resblocks[idx](x)
                res_outputs.append(y)
            x = sum(res_outputs) / self.num_kernels

        # 5. Post-conv to generate single channel waveform output
        x = F.leaky_relu(x, LEAKY_RELU_SLOPE)
        x = torch.tanh(self.conv_post(x))
        return x


    def remove_weight_norm(self):
        """Strip weight norm for faster inference"""
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

        for up in self.ups:
            remove_weight_norm(up)

        # Has internal remove_weight_norm
        for rb in self.resblocks:
            rb.remove_weight_norm()
