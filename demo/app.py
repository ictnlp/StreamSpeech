
##########################################
# Simultaneous Speech-to-Speech Translation Agent for StreamSpeech
#
# StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning (ACL 2024)
##########################################
import sys
import os
# Add fairseq to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fairseq'))

from flask import Flask, request, jsonify, render_template, send_from_directory,url_for
import os
import json
import logging

# Re-enable Flask request logging (comment out to hide logs)
# logging.getLogger('werkzeug').setLevel(logging.ERROR)
import pdb
import argparse
from pydub import AudioSegment
import math
import numpy as np
import shutil

from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import WriteAction, ReadAction
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fairseq.data.audio.audio_utils import convert_waveform
# Import data_utils directly from the file path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fairseq', 'examples', 'speech_to_text'))
from data_utils import extract_fbank_features
import ast
import math
import os
import json
import numpy as np
from copy import deepcopy
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.file_io import PathManager
from fairseq import search
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
import soundfile
import argparse

SHIFT_SIZE = 10
WINDOW_SIZE = 25
ORG_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_EOS = 2
OFFSET_MS=-1
Finished=False

ASR={}

S2TT={}

S2ST=[]
S2ST_ORIGINAL=[]  # For dual mode: original vocoder output
S2ST_MODIFIED=[]  # For dual mode: modified vocoder output

class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args, cfg):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = "cuda" if torch.cuda.is_available()  else "cpu"
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            {"feature_transforms": ["utterance_cmvn"]}
        )

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples, sr=ORG_SAMPLE_RATE):
        samples = new_samples

        # # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # # the number of frames used for feature extraction
        # # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )
        samples = samples[:effective_num_samples]
        # Simple audio conversion without sox dependency
        waveform = torch.tensor([samples])
        if sr != 16000:
            # Simple resampling using torch.nn.functional.interpolate
            # waveform is 2D: [1, samples_length]
            target_length = int(len(samples) * 16000 / sr)
            # For linear interpolation, we need 3D input: [batch, channels, length]
            waveform = waveform.unsqueeze(0)  # Now [1, 1, samples_length]
            waveform = torch.nn.functional.interpolate(
                waveform, size=target_length, mode='linear', align_corners=False
            ).squeeze(0)  # Back to [1, target_length]
        sample_rate = 16000
        output = extract_fbank_features(waveform, 16000)
        output = self.transform(output)
        return torch.tensor(output, device=self.device)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x

class StreamSpeechS2STAgent(SpeechToSpeechAgent):
    """
    Incrementally feed text to this offline Fastspeech2 TTS model,
    with a minimum numbers of phonemes every chunk.
    """

    def __init__(self, args):
        super().__init__(args)
        self.eos = DEFAULT_EOS

        self.gpu = torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.args = args

        self.load_model_vocab(args)

        self.max_len = args.max_len

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

        tgt_dict_mt = self.dict[f"{self.models[0].mt_task_name}"]
        tgt_dict = self.dict["tgt"]
        tgt_dict_asr = self.dict["source_unigram"]
        tgt_dict_st = self.dict["ctc_target_unigram"]
        args.user_dir=args.agent_dir
        utils.import_user_module(args)
        from agent.sequence_generator import SequenceGenerator
        from agent.ctc_generator import CTCSequenceGenerator
        from agent.ctc_decoder import CTCDecoder

        self.ctc_generator = CTCSequenceGenerator(
            tgt_dict, self.models, use_incremental_states=False
        )

        self.asr_ctc_generator = CTCDecoder(tgt_dict_asr, self.models)
        self.st_ctc_generator = CTCDecoder(tgt_dict_st, self.models)

        self.generator = SequenceGenerator(
            self.models,
            tgt_dict,
            beam_size=1,
            max_len_a=1,
            max_len_b=200,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict),
            eos=tgt_dict.eos(),
            symbols_to_strip_from_output=None,
        )

        self.generator_mt = SequenceGenerator(
            self.models,
            tgt_dict_mt,
            beam_size=1,
            max_len_a=0,
            max_len_b=100,
            max_len=0,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=search.BeamSearch(tgt_dict_mt),
            eos=tgt_dict_mt.eos(),
            symbols_to_strip_from_output=None,
            use_incremental_states=False,
        )

        # Initialize vocoder wrapper
        from vocoder_wrapper import VocoderWrapper, DualVocoderWrapper
        
        vocoder_type = getattr(args, 'vocoder_type', 'original')  # default to original
        
        if vocoder_type == 'dual':
            # Initialize dual vocoder system for side-by-side comparison
            self.vocoder = DualVocoderWrapper(
                original_vocoder_path=getattr(args, 'original_vocoder', args.vocoder),
                original_vocoder_cfg=getattr(args, 'original_vocoder_cfg', args.vocoder_cfg),
                modified_vocoder_path=getattr(args, 'modified_vocoder', args.vocoder),
                modified_vocoder_cfg=getattr(args, 'modified_vocoder_cfg', args.vocoder_cfg),
                device=self.device
            )
            self.dual_mode = True
        else:
            self.vocoder = VocoderWrapper(
                vocoder_type=vocoder_type,
                vocoder_path=args.vocoder,
                vocoder_cfg_path=args.vocoder_cfg,
                device=self.device
            )
            self.dual_mode = False
        
        self.dur_prediction = args.dur_prediction

        self.lagging_k1 = args.lagging_k1
        self.lagging_k2 = args.lagging_k2
        self.segment_size = args.segment_size
        self.stride_n = args.stride_n

        self.unit_per_subword = args.unit_per_subword
        self.stride_n2 = args.stride_n2

        if args.extra_output_dir is not None:
            self.asr_file = Path(args.extra_output_dir + "/asr.txt")
            self.st_file = Path(args.extra_output_dir + "/st.txt")
            self.unit_file = Path(args.extra_output_dir + "/unit.txt")
            self.quiet = False
        else:
            self.quiet = True

        self.output_asr_translation = args.output_asr_translation

        self.segment_size=args.segment_size

        if args.segment_size >= 640:
            self.whole_word = True
        else:
            self.whole_word = False
        
        self.states = self.build_states()
        self.reset()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="path to your pretrained model.",
        )
        parser.add_argument(
            "--data-bin", type=str, required=True, help="Path of data binary"
        )
        parser.add_argument(
            "--config-yaml", type=str, default=None, help="Path to config yaml file"
        )
        parser.add_argument(
            "--multitask-config-yaml",
            type=str,
            default=None,
            help="Path to config yaml file",
        )
        parser.add_argument(
            "--global-stats",
            type=str,
            default=None,
            help="Path to json file containing cmvn stats",
        )
        parser.add_argument(
            "--tgt-splitter-type",
            type=str,
            default="SentencePiece",
            help="Subword splitter type for target text",
        )
        parser.add_argument(
            "--tgt-splitter-path",
            type=str,
            default=None,
            help="Subword splitter model path for target text",
        )
        parser.add_argument(
            "--user-dir",
            type=str,
            default="researches/ctc_unity",
            help="User directory for model",
        )
        parser.add_argument(
            "--agent-dir",
            type=str,
            default="agent",
            help="User directory for agents",
        )
        parser.add_argument(
            "--max-len", type=int, default=200, help="Max length of translation"
        )
        parser.add_argument(
            "--force-finish",
            default=False,
            action="store_true",
            help="Force the model to finish the hypothsis if the source is not finished",
        )
        parser.add_argument(
            "--shift-size",
            type=int,
            default=SHIFT_SIZE,
            help="Shift size of feature extraction window.",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=WINDOW_SIZE,
            help="Window size of feature extraction window.",
        )
        parser.add_argument(
            "--sample-rate", type=int, default=ORG_SAMPLE_RATE, help="Sample rate"
        )
        parser.add_argument(
            "--feature-dim",
            type=int,
            default=FEATURE_DIM,
            help="Acoustic feature dimension.",
        )
        parser.add_argument(
            "--vocoder", type=str, required=True, help="path to the CodeHiFiGAN vocoder"
        )
        parser.add_argument(
            "--vocoder-cfg",
            type=str,
            required=True,
            help="path to the CodeHiFiGAN vocoder config",
        )
        parser.add_argument(
            "--vocoder-type",
            type=str,
            default="original",
            choices=["original", "modified", "dual"],
            help="Type of vocoder to use: 'original' (CodeHiFiGAN), 'modified' (UnitHiFiGAN+FiLM), or 'dual' (both for comparison)",
        )
        parser.add_argument(
            "--original-vocoder",
            type=str,
            required=False,
            help="path to the original CodeHiFiGAN vocoder (for dual mode)",
        )
        parser.add_argument(
            "--original-vocoder-cfg",
            type=str,
            required=False,
            help="path to the original CodeHiFiGAN vocoder config (for dual mode)",
        )
        parser.add_argument(
            "--modified-vocoder",
            type=str,
            required=False,
            help="path to the modified UnitHiFiGAN vocoder (for dual mode)",
        )
        parser.add_argument(
            "--modified-vocoder-cfg",
            type=str,
            required=False,
            help="path to the modified UnitHiFiGAN vocoder config (for dual mode)",
        )
        parser.add_argument(
            "--dur-prediction",
            action="store_true",
            help="enable duration prediction (for reduced/unique code sequences)",
        )
        parser.add_argument("--lagging-k1", type=int, default=0, help="lagging number")
        parser.add_argument("--lagging-k2", type=int, default=0, help="lagging number")
        parser.add_argument(
            "--segment-size", type=int, default=320, help="segment-size"
        )
        parser.add_argument("--stride-n", type=int, default=1, help="lagging number")
        parser.add_argument("--stride-n2", type=int, default=1, help="lagging number")
        parser.add_argument(
            "--unit-per-subword", type=int, default=15, help="lagging number"
        )
        parser.add_argument(
            "--extra-output-dir", type=str, default=None, help="extra output dir"
        )
        parser.add_argument(
            "--output-asr-translation",
            type=bool,
            default=False,
            help="extra output dir",
        )

    def reset(self):
        self.src_seg_num = 0
        self.tgt_subwords_indices = None
        self.src_ctc_indices = None
        self.src_ctc_prefix_length = 0
        self.tgt_ctc_prefix_length = 0
        self.tgt_units_indices = None
        self.prev_output_tokens_mt = None
        self.tgt_text = []
        self.mt_decoder_out = None
        self.unit = None
        self.wav = []
        self.post_transcription = ""
        self.unfinished_wav = None
        self.states.reset()
        try:
            self.generator_mt.reset_incremental_states()
            self.ctc_generator.reset_incremental_states()
        except:
            pass

    def set_chunk_size(self,segment_size):
        # print(segment_size)
        self.segment_size=segment_size
        chunk_size = segment_size // 40


        for model in self.models:
            model.encoder.chunk_size = chunk_size

            if chunk_size >= 16:
                chunk_size = 16
            else:
                chunk_size = 8
            for conv in model.encoder.subsample.conv_layers:
                conv.chunk_size = chunk_size
            for layer in model.encoder.conformer_layers:
                layer.conv_module.depthwise_conv.chunk_size = chunk_size

        if segment_size >= 640:
            self.whole_word = True
        else:
            self.whole_word = False


    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        state["cfg"].common['user_dir']=args.user_dir
        utils.import_user_module(state["cfg"].common)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        args.global_cmvn = None
        if args.config_yaml is not None:
            task_args.config_yaml = args.config_yaml
            with open(os.path.join(args.data_bin, args.config_yaml), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        self.feature_extractor = OnlineFeatureExtractor(args, config)

        if args.multitask_config_yaml is not None:
            task_args.multitask_config_yaml = args.multitask_config_yaml

        task = tasks.setup_task(task_args)
        self.task = task

        overrides = ast.literal_eval(state["cfg"].common_eval.model_overrides)

        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(filename),
            arg_overrides=overrides,
            task=task,
            suffix=state["cfg"].checkpoint.checkpoint_suffix,
            strict=(state["cfg"].checkpoint.checkpoint_shard_count == 1),
            num_shards=state["cfg"].checkpoint.checkpoint_shard_count,
        )

        chunk_size = args.segment_size // 40

        self.models = models

        for model in self.models:
            model.eval()
            model.share_memory()
            if self.gpu:
                model.cuda()
            model.encoder.chunk_size = chunk_size

            if chunk_size >= 16:
                chunk_size = 16
            else:
                chunk_size = 8
            for conv in model.encoder.subsample.conv_layers:
                conv.chunk_size = chunk_size
            for layer in model.encoder.conformer_layers:
                layer.conv_module.depthwise_conv.chunk_size = chunk_size

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

        for k, v in task.multitask_tasks.items():
            self.dict[k] = v.tgt_dict

    @torch.inference_mode()
    def policy(self):
        # print(self.states.source)
        feature = self.feature_extractor(self.states.source)

        if feature.size(0) == 0 and not self.states.source_finished:
            return ReadAction()

        src_indices = feature.unsqueeze(0)
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long()

        self.encoder_outs = self.generator.model.forward_encoder(
            {"src_tokens": src_indices, "src_lengths": src_lengths}
        )

        finalized_asr = self.asr_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="source_unigram"
        )
        asr_probs = torch.exp(finalized_asr[0][0]["lprobs"])

        for i, hypo in enumerate(finalized_asr):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()
            src_ctc_indices = tmp
            src_ctc_index = hypo[i_beam]["index"]
            text = "".join([self.dict["source_unigram"][c] for c in tmp])
            text = text.replace("_", " ")
            text = text.replace("▁", " ")
            text = text.replace("<unk>", " ")
            text = text.replace("<s>", "")
            text = text.replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            if self.states.source_finished and not self.quiet:
                with open(self.asr_file, "a") as file:
                    print(text, file=file)
            if self.output_asr_translation:
                print("Streaming ASR:", text)

            ASR[len(self.states.source)]=text
            

        finalized_st = self.st_ctc_generator.generate(
            self.encoder_outs[0], aux_task_name="ctc_target_unigram"
        )
        st_probs = torch.exp(finalized_st[0][0]["lprobs"])

        for i, hypo in enumerate(finalized_st):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()
            tgt_ctc_indices = tmp
            tgt_ctc_index = hypo[i_beam]["index"]
            text = "".join([self.dict["ctc_target_unigram"][c] for c in tmp])
            text = text.replace("_", " ")
            text = text.replace("▁", " ")
            text = text.replace("<unk>", " ")
            text = text.replace("<s>", "")
            text = text.replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]

        if not self.states.source_finished:
            src_ctc_prefix_length = src_ctc_indices.size(-1)
            tgt_ctc_prefix_length = tgt_ctc_indices.size(-1)

            self.src_ctc_indices = src_ctc_indices
            if (
                src_ctc_prefix_length < self.src_ctc_prefix_length + self.stride_n
                or tgt_ctc_prefix_length < self.tgt_ctc_prefix_length + self.stride_n
            ):
                return ReadAction()
            self.src_ctc_prefix_length = max(
                src_ctc_prefix_length, self.src_ctc_prefix_length
            )
            self.tgt_ctc_prefix_length = max(
                tgt_ctc_prefix_length, self.tgt_ctc_prefix_length
            )
            subword_tokens = (
                (tgt_ctc_prefix_length - self.lagging_k1) // self.stride_n
            ) * self.stride_n

            if self.whole_word:
                subword_tokens += 1
            new_subword_tokens = (
                (subword_tokens - self.tgt_subwords_indices.size(-1))
                if self.tgt_subwords_indices is not None
                else subword_tokens
            )

            if new_subword_tokens < 1:
                return ReadAction()
        else:
            self.src_ctc_indices = src_ctc_indices
            new_subword_tokens = -1

        new_subword_tokens = int(new_subword_tokens)

        single_model = self.generator.model.single_model
        mt_decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")

        # 1. MT decoder
        finalized_mt = self.generator_mt.generate_decoder(
            self.encoder_outs,
            src_indices,
            src_lengths,
            {
                "id": 1,
                "net_input": {"src_tokens": src_indices, "src_lengths": src_lengths},
            },
            self.tgt_subwords_indices,
            None,
            None,
            aux_task_name=single_model.mt_task_name,
            max_new_tokens=new_subword_tokens,
        )

        if finalized_mt[0][0]["tokens"][-1] == 2:
            tgt_subwords_indices = finalized_mt[0][0]["tokens"][:-1].unsqueeze(0)
        else:
            tgt_subwords_indices = finalized_mt[0][0]["tokens"].unsqueeze(0)

        if self.whole_word:
            j = 999999
            if not self.states.source_finished:
                for j in range(tgt_subwords_indices.size(-1) - 1, -1, -1):
                    if self.generator_mt.tgt_dict[
                        tgt_subwords_indices[0][j]
                    ].startswith("▁"):
                        break
                tgt_subwords_indices = tgt_subwords_indices[:, :j]
                finalized_mt[0][0]["tokens"] = finalized_mt[0][0]["tokens"][:j]

                if j == 0:
                    return ReadAction()

                new_incremental_states = [{}]
                if (
                    self.generator_mt.incremental_states is not None
                    and self.generator_mt.use_incremental_states
                ):
                    for k, v in self.generator_mt.incremental_states[0].items():
                        if v["prev_key"].size(2) == v["prev_value"].size(2):
                            new_incremental_states[0][k] = {
                                "prev_key": v["prev_key"][:, :, :j, :].contiguous(),
                                "prev_value": v["prev_value"][:, :, :j, :].contiguous(),
                                "prev_key_padding_mask": None,
                            }
                        else:
                            new_incremental_states[0][k] = {
                                "prev_key": v["prev_key"],
                                "prev_value": v["prev_value"][:, :, :j, :].contiguous(),
                                "prev_key_padding_mask": None,
                            }
                    self.generator_mt.incremental_states = deepcopy(
                        new_incremental_states
                    )

        max_tgt_len = max([len(hypo[0]["tokens"]) for hypo in finalized_mt])
        if self.whole_word:
            max_tgt_len += 1
        prev_output_tokens_mt = (
            src_indices.new_zeros(src_indices.shape[0], max_tgt_len)
            .fill_(mt_decoder.padding_idx)
            .int()
        )

        for i, hypo in enumerate(finalized_mt):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()
            prev_output_tokens_mt[i, 0] = self.generator_mt.eos
            if tmp[-1] == self.generator_mt.eos:
                tmp = tmp[:-1]
            prev_output_tokens_mt[i, 1 : len(tmp) + 1] = tmp

            tokens = [self.generator_mt.tgt_dict[c] for c in tmp]

            text = "".join(tokens)
            text = text.replace("_", " ")
            text = text.replace("▁", " ")
            text = text.replace("<unk>", " ")
            text = text.replace("<s>", "")
            text = text.replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            if self.states.source_finished and not self.quiet:
                with open(self.st_file, "a") as file:
                    print(text, file=file)
            if self.output_asr_translation:
                print("Simultaneous translation:", text)

            S2TT[len(self.states.source)]=text

        if self.tgt_subwords_indices is not None and torch.equal(
            self.tgt_subwords_indices, tgt_subwords_indices
        ):
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(
                            self.unfinished_wav.tolist()
                            if self.unfinished_wav is not None
                            else []
                        ),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )
        self.tgt_subwords_indices = tgt_subwords_indices

        if not self.states.source_finished:
            if self.prev_output_tokens_mt is not None:
                if torch.equal(
                    self.prev_output_tokens_mt, prev_output_tokens_mt
                ) or prev_output_tokens_mt.size(-1) <= self.prev_output_tokens_mt.size(
                    -1
                ):
                    return ReadAction()
        self.prev_output_tokens_mt = prev_output_tokens_mt
        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=self.encoder_outs[0],
            features_only=True,
        )[0].transpose(0, 1)

        if self.mt_decoder_out is None:
            self.mt_decoder_out = mt_decoder_out
        else:
            self.mt_decoder_out = torch.cat(
                (self.mt_decoder_out, mt_decoder_out[self.mt_decoder_out.size(0) :]),
                dim=0,
            )
        self.mt_decoder_out = mt_decoder_out
        x = self.mt_decoder_out

        if getattr(single_model, "proj", None) is not None:
            x = single_model.proj(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if getattr(single_model, "synthesizer_encoder", None) is not None:
            t2u_encoder_out = single_model.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
            )
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": (
                    [mt_decoder_padding_mask]
                    if mt_decoder_padding_mask is not None
                    else []
                ),  # B x T
                "encoder_embedding": [],
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
            }

        if getattr(single_model, "t2u_augmented_cross_attn", False):
            encoder_outs_aug = [t2u_encoder_out]
        else:
            encoder_outs = [t2u_encoder_out]
            encoder_outs_aug = None
        finalized = self.ctc_generator.generate(
            encoder_outs[0],
            prefix=self.tgt_units_indices,
        )

        if len(finalized[0][0]["tokens"]) == 0:
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(
                            self.unfinished_wav.tolist()
                            if self.unfinished_wav is not None
                            else []
                        ),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )

        for i, hypo in enumerate(finalized):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
            if tmp[-1] == self.generator.eos:
                tmp = tmp[:-1]
            unit = []
            for c in tmp:
                u = self.generator.tgt_dict[c].replace("<s>", "").replace("</s>", "")
                if u != "":
                    unit.append(int(u))

            if len(unit) > 0 and unit[0] == " ":
                unit = unit[1:]
            text = " ".join([str(_) for _ in unit])
            if self.states.source_finished and not self.quiet:
                with open(self.unit_file, "a") as file:
                    print(text, file=file)
        cur_unit = unit if self.unit is None else unit[len(self.unit) :]
        if len(unit) < 1 or len(cur_unit) < 1:
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(
                            self.unfinished_wav.tolist()
                            if self.unfinished_wav is not None
                            else []
                        ),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )

        # Extract discrete units before vocoder
        if self.states.source_finished and len(unit) > 0:
            try:
                from extract_intermediates import save_discrete_units
                # Use source filename if available, otherwise default to "output"
                filename_prefix = getattr(self, 'current_source_filename', 'output')
                save_discrete_units(unit, filename_prefix=f"{filename_prefix}_output")
            except Exception as e:
                import traceback
                print(f"⚠️  Failed to save discrete units: {e}")
                traceback.print_exc()
        
        x = {
            "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(
                1, -1
            ),
        }
        
        # Handle dual mode or single mode
        if self.dual_mode:
            # Dual mode: generate both outputs
            vocoder_outputs = self.vocoder(x, self.dur_prediction)
            wav_original, dur_original = vocoder_outputs['original']
            wav_modified, dur_modified = vocoder_outputs['modified']
            
            # Process original output
            cur_wav_length_original = dur_original[:, -len(cur_unit) :].sum() * 320
            new_wav_original = wav_original[-cur_wav_length_original:]
            if hasattr(self, 'unfinished_wav_original') and self.unfinished_wav_original is not None and len(self.unfinished_wav_original) > 0:
                new_wav_original = torch.cat((self.unfinished_wav_original, new_wav_original), dim=0)
            
            # Process modified output
            cur_wav_length_modified = dur_modified[:, -len(cur_unit) :].sum() * 320
            new_wav_modified = wav_modified[-cur_wav_length_modified:]
            if hasattr(self, 'unfinished_wav_modified') and self.unfinished_wav_modified is not None and len(self.unfinished_wav_modified) > 0:
                new_wav_modified = torch.cat((self.unfinished_wav_modified, new_wav_modified), dim=0)
            
            # Store both outputs
            self.wav_original = wav_original
            self.wav_modified = wav_modified
            self.wav = wav_original  # For compatibility
            self.unit = unit
            
            # Add to global lists
            global S2ST_ORIGINAL, S2ST_MODIFIED
            S2ST_ORIGINAL.extend(new_wav_original.tolist())
            S2ST_MODIFIED.extend(new_wav_modified.tolist())
            
            # Also add original to S2ST for backward compatibility
            S2ST.extend(new_wav_original.tolist())
            
            # Use original for return (primary output)
            new_wav = new_wav_original
            wav = wav_original
            dur = dur_original
        else:
            # Single mode: original behavior
            wav, dur = self.vocoder(x, self.dur_prediction)
            
            cur_wav_length = dur[:, -len(cur_unit) :].sum() * 320
            new_wav = wav[-cur_wav_length:]
            if self.unfinished_wav is not None and len(self.unfinished_wav) > 0:
                new_wav = torch.cat((self.unfinished_wav, new_wav), dim=0)
            
            self.wav = wav
            self.unit = unit
            
            S2ST.extend(new_wav.tolist())

        global OFFSET_MS
        if OFFSET_MS==-1:
            OFFSET_MS=1000*len(self.states.source)/ORG_SAMPLE_RATE

        # A SpeechSegment has to be returned for speech-to-speech translation system
        if self.states.source_finished and new_subword_tokens == -1:
            self.states.target_finished = True
            # self.reset()

        return WriteAction(
            SpeechSegment(
                content=new_wav.tolist(),
                sample_rate=SAMPLE_RATE,
                finished=self.states.source_finished,
            ),
            finished=self.states.target_finished,
        )
    
def run(source):
    # if len(S2ST)!=0: return
    
    # Handle MP3 files by converting to WAV first
    if source.lower().endswith('.mp3'):
        print(f"Converting MP3 to WAV: {source}")
        audio = AudioSegment.from_mp3(source)
        # Create a temporary WAV file
        wav_path = source.rsplit('.', 1)[0] + '_temp.wav'
        audio.export(wav_path, format='wav')
        samples, sr = soundfile.read(wav_path, dtype="float32")
        # Clean up temp file
        try:
            os.remove(wav_path)
        except:
            pass
    else:
        samples, sr = soundfile.read(source, dtype="float32")
    
    # Extract source audio at 16kHz
    source_filename = os.path.basename(source).split('.')[0]
    from extract_intermediates import save_source_audio
    extracted_wav_path = save_source_audio(samples, ORG_SAMPLE_RATE, filename_prefix=source_filename)
    
    # Store filename for later use in discrete units extraction
    agent.current_source_filename = source_filename
    
    # Pass extracted 16kHz WAV to vocoder for embedding extraction (modified vocoder only)
    # This ensures ECAPA and Emotion2Vec get clean, properly formatted audio
    agent.vocoder.set_source_audio(extracted_wav_path)
    
    # Resample to expected sample rate if needed
    if sr != ORG_SAMPLE_RATE:
        print(f"Resampling from {sr}Hz to {ORG_SAMPLE_RATE}Hz")
        # Simple resampling using torch
        samples_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0)  # [1, 1, length]
        target_length = int(len(samples) * ORG_SAMPLE_RATE / sr)
        samples_tensor = torch.nn.functional.interpolate(
            samples_tensor, size=target_length, mode='linear', align_corners=False
        )
        samples = samples_tensor.squeeze().numpy()
    
    # Normalize input audio to prevent loud playback
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.8  # Normalize and scale to 80%
    
    agent.reset()

    interval=int(agent.segment_size*(ORG_SAMPLE_RATE/1000))
    print(f"🔄 Processing with segment_size={agent.segment_size}ms, interval={interval} samples")
    cur_idx=0
    while not agent.states.target_finished:
        cur_idx+=interval
        agent.states.source=samples[:cur_idx]
        agent.states.source_finished=cur_idx>len(samples)
        action=agent.policy()
        # print("ASR_RESULT",ASR)
        # print("S2ST_RESULT",S2ST)

def reset():
    global OFFSET_MS
    OFFSET_MS=-1
    global ASR
    ASR={}
    global S2TT
    S2TT={}
    global S2ST, S2ST_ORIGINAL, S2ST_MODIFIED
    S2ST=[]
    S2ST_ORIGINAL=[]
    S2ST_MODIFIED=[]


def find_largest_key_value(dictionary, N):
    keys = [key for key in dictionary.keys() if key < N]
    if not keys:
        return ""
    largest_key = max(keys)
    return dictionary[largest_key]

def merge_audio(left_audio_path, right_audio_path, offset_ms):
    # Use soundfile instead of pydub to avoid ffmpeg dependency
    left_data, left_sr = soundfile.read(left_audio_path, dtype='float32')
    right_data, right_sr = soundfile.read(right_audio_path, dtype='float32')
    
    # Convert offset from ms to samples
    offset_samples = int(offset_ms * right_sr / 1000)
    
    # Add silence at the beginning of right audio
    right_data = np.concatenate([np.zeros(offset_samples), right_data])
    
    # Ensure both audio files have the same length
    max_length = max(len(left_data), len(right_data))
    
    if len(left_data) < max_length:
        left_data = np.concatenate([left_data, np.zeros(max_length - len(left_data))])
    if len(right_data) < max_length:
        right_data = np.concatenate([right_data, np.zeros(max_length - len(right_data))])
    
    # Normalize audio data before creating AudioSegment objects
    left_max = np.max(np.abs(left_data))
    if left_max > 0:
        left_data = left_data / left_max * 0.8
    
    right_max = np.max(np.abs(right_data))
    if right_max > 0:
        right_data = right_data / right_max * 0.8
    
    # Convert to int16 for AudioSegment (standard format)
    left_data_int16 = (left_data * 32767).astype(np.int16)
    right_data_int16 = (right_data * 32767).astype(np.int16)
    
    # Create AudioSegment objects for compatibility with the rest of the code
    left_audio = AudioSegment(
        left_data_int16.tobytes(),
        frame_rate=left_sr,
        sample_width=2,  # int16 = 2 bytes
        channels=1
    )
    right_audio = AudioSegment(
        right_data_int16.tobytes(),
        frame_rate=right_sr,
        sample_width=2,  # int16 = 2 bytes
        channels=1
    )
    
    # Audio normalization is now handled at the source when writing the file
    
    return left_audio, right_audio

# Flask routes will be defined after app initialization

# Load main configuration
with open('config.json', 'r') as f:
    main_config = json.load(f)

# Load paths configuration
with open('paths_config.json', 'r') as f:
    paths_config = json.load(f)

# Merge configurations
args_dict = main_config.copy()
if main_config.get('use_paths_config', False):
    # Determine which vocoder to use
    vocoder_type = args_dict.get('vocoder-type', 'original')
    
    if vocoder_type == 'dual':
        # Provide both vocoder paths for dual mode
        vocoder_paths = {
            'original-vocoder': paths_config['vocoder']['checkpoint'],
            'original-vocoder-cfg': paths_config['vocoder']['config'],
            'modified-vocoder': paths_config['modified_vocoder']['checkpoint'],
            'modified-vocoder-cfg': paths_config['modified_vocoder']['config'],
            # Default to original for backward compatibility
            'vocoder': paths_config['vocoder']['checkpoint'],
            'vocoder-cfg': paths_config['vocoder']['config']
        }
    elif vocoder_type == 'modified':
        vocoder_paths = {
            'vocoder': paths_config['modified_vocoder']['checkpoint'],
            'vocoder-cfg': paths_config['modified_vocoder']['config']
        }
    else:
        vocoder_paths = {
            'vocoder': paths_config['vocoder']['checkpoint'],
            'vocoder-cfg': paths_config['vocoder']['config']
        }
    
    # Add paths from paths_config.json
    args_dict.update({
        'data-bin': paths_config['configs']['data_bin'],
        'user-dir': paths_config['configs']['user_dir'],
        'agent-dir': paths_config['configs']['agent_dir'],
        'model-path': paths_config['models']['simultaneous'],
        **vocoder_paths
    })

# Initialize Flask app with config
app = Flask(__name__)
# Set upload folder from paths config
upload_folder = paths_config.get('demo', {}).get('upload_folder', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize agent
parser = argparse.ArgumentParser()
StreamSpeechS2STAgent.add_args(parser)

# Create the list of arguments from args_dict
args_list = []
# pdb.set_trace()
for key, value in args_dict.items():
    # Skip non-argument fields
    if key.startswith('_') or key in ['use_paths_config', 'language_pair']:
        continue
    if isinstance(value, bool):
        if value:
            args_list.append(f'--{key}')
    else:
        args_list.append(f'--{key}')
        args_list.append(str(value))

args = parser.parse_args(args_list)

agent = StreamSpeechS2STAgent(args)

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Return just the filename, not the full path
        return file.filename

@app.route('/process/<path:filepath>')
def uploaded_file(filepath):
    latency = request.args.get('latency', default=320, type=int)
    print(f"📊 Received latency parameter: {latency} ms")
    agent.set_chunk_size(latency)
    print(f"✅ Agent segment_size updated to: {agent.segment_size} ms")

    # Construct full path from upload folder and filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    # pdb.set_trace()
    # if len(S2ST)==0:
    reset()
    run(path)
    filename = os.path.basename(path)
    output_path = os.path.join(os.path.dirname(path), 'output.'+filename)
    
    # Check if dual mode is enabled
    if agent.dual_mode and len(S2ST_ORIGINAL) > 0 and len(S2ST_MODIFIED) > 0:
        # DUAL MODE: Save both original and modified outputs
        
        # Save original vocoder output
        output_path_original = os.path.join(os.path.dirname(path), 'output_original.'+filename)
        audio_data_original = np.array(S2ST_ORIGINAL, dtype=np.float32)
        max_val_original = np.max(np.abs(audio_data_original))
        if max_val_original > 0:
            audio_data_original = audio_data_original / max_val_original * 0.8
        soundfile.write(output_path_original, audio_data_original, SAMPLE_RATE)
        
        # Save modified vocoder output
        output_path_modified = os.path.join(os.path.dirname(path), 'output_modified.'+filename)
        audio_data_modified = np.array(S2ST_MODIFIED, dtype=np.float32)
        max_val_modified = np.max(np.abs(audio_data_modified))
        if max_val_modified > 0:
            audio_data_modified = audio_data_modified / max_val_modified * 0.8
        soundfile.write(output_path_modified, audio_data_modified, SAMPLE_RATE)
        
        # Also save as default output (use original)
        soundfile.write(output_path, audio_data_original, SAMPLE_RATE)
        
        print(f"✓ DUAL MODE: Saved both outputs")
        print(f"  - Original: {output_path_original}")
        print(f"  - Modified: {output_path_modified}")
    else:
        # SINGLE MODE: Original behavior
        # Normalize the audio data to prevent it from being too loud
        if len(S2ST) > 0:
            # Convert to numpy array and normalize
            audio_data = np.array(S2ST, dtype=np.float32)
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8  # Scale to 80% of max to be safe
            soundfile.write(output_path, audio_data, SAMPLE_RATE)
        else:
            # Create silent audio if no data
            soundfile.write(output_path, np.zeros(1000), SAMPLE_RATE)
    
    left,right=merge_audio(path, output_path, OFFSET_MS)
    input_path = os.path.join(os.path.dirname(path), 'input.'+filename)
    left.export(input_path, format="wav")
    right.export(output_path, format="wav")
    # left=left.split_to_mono()[0]
    # right=right.split_to_mono()[1]
    # pdb.set_trace()
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'input.'+filename)

@app.route('/output/<path:filepath>')
def uploaded_output_file(filepath):
    # filepath is just the filename
    filename = filepath
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output.'+filename)

@app.route('/output_original/<path:filepath>')
def uploaded_output_original_file(filepath):
    # filepath is just the filename
    filename = filepath
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_original.'+filename)

@app.route('/output_modified/<path:filepath>')
def uploaded_output_modified_file(filepath):
    # filepath is just the filename
    filename = filepath
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_modified.'+filename)

@app.route('/is_dual_mode', methods=['GET'])
def is_dual_mode():
    return jsonify(dual_mode=agent.dual_mode)


@app.route('/asr/<current_time>', methods=['GET'])
def asr(current_time):
    try:
        current_time = float(current_time)
    except ValueError:
        return jsonify(result="")
    
    # asr_result = f"ABCD... {int(current_time * 1000)}"
    N = current_time*ORG_SAMPLE_RATE

    asr_result=find_largest_key_value(ASR, N)
    return jsonify(result=asr_result)

@app.route('/translation/<current_time>', methods=['GET'])
def translation(current_time):
    try:
        current_time = float(current_time)
    except ValueError:
        return jsonify(result="")
    
    N = current_time*ORG_SAMPLE_RATE

    translation_result=find_largest_key_value(S2TT, N)
    # translation_result = f"1234... {int(current_time * 1000)}"
    return jsonify(result=translation_result)

@app.route('/favicon.ico')
def favicon():
    # Return a simple 204 No Content response to stop the 404 error
    return '', 204

if __name__ == '__main__':
    host = paths_config.get('demo', {}).get('host', '0.0.0.0')
    port = paths_config.get('demo', {}).get('port', 7860)
    app.run(host=host, port=port, debug=True)
