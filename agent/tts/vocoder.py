# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fairseq.data.audio.audio_utils import (
    TTSSpectrogram,
    get_fourier_basis,
    get_mel_filters,
    get_window,
)
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.models import BaseFairseqModel, register_model
from agent.tts.codehifigan import CodeGenerator as CodeHiFiGANModel
from fairseq.models.text_to_speech.hifigan import Generator as HiFiGANModel
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface

logger = logging.getLogger(__name__)


@register_model("CodeHiFiGANVocoderWithDur")
class CodeHiFiGANVocoderWithDur(BaseFairseqModel):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = CodeHiFiGANModel(model_cfg)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        if fp16:
            self.model.half()
        self.model.remove_weight_norm()
        logger.info(f"loaded CodeHiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: Dict[str, torch.Tensor], dur_prediction=False) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)
        wav, dur = self.model(**x)
        return wav.detach().squeeze(), dur

    @classmethod
    def from_data_cfg(cls, args, data_cfg):
        vocoder_cfg = data_cfg.vocoder
        assert vocoder_cfg is not None, "vocoder not specified in the data config"
        with open(vocoder_cfg["config"]) as f:
            model_cfg = json.load(f)
        return cls(vocoder_cfg["checkpoint"], model_cfg, fp16=args.fp16)

    @classmethod
    def hub_models(cls):
        base_url = "http://dl.fbaipublicfiles.com/fairseq/vocoder"
        model_ids = [
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur",
            "unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10_dur",
            "unit_hifigan_HK_layer12.km2500_frame_TAT-TTS",
        ]
        return {i: f"{base_url}/{i}.tar.gz" for i in model_ids}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        config="config.json",
        fp16: bool = False,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            config_yaml=config,
            fp16=fp16,
            is_vocoder=True,
            **kwargs,
        )

        with open(f"{x['args']['data']}/{config}") as f:
            vocoder_cfg = json.load(f)
        assert len(x["args"]["model_path"]) == 1, "Too many vocoder models in the input"

        vocoder = CodeHiFiGANVocoderWithDur(x["args"]["model_path"][0], vocoder_cfg)
        return VocoderHubInterface(vocoder_cfg, vocoder)
