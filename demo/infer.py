import os
import json
import threading
import queue
import time
import math
from typing import List, Dict, Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal

import ast
from fairseq import checkpoint_utils, tasks, utils, search
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    S2TTransformerDecoder,
)
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

# Constants
SAMPLE_RATE = 16000
FEATURE_DIM = 80
SHIFT_SIZE = 10
WINDOW_SIZE = 25
ORG_SAMPLE_RATE = 16000

@dataclass
class Config:
    model_path: str
    data_bin: str
    config_yaml: str
    multitask_config_yaml: str
    global_cmvn: str
    vocoder: str
    vocoder_cfg: str
    tgt_lang: str
    output_dir: str
    segment_size: int = 320
    beam_size: int = 5
    max_len: int = 200
    force_finish: bool = False
    use_gpu: bool = torch.cuda.is_available()
    lagging_k1: int = 0
    lagging_k2: int = 0
    stride_n: int = 1
    stride_n2: int = 1
    unit_per_subword: int = 15
    user_dir: str = "researches/ctc_unity"
    shift_size: int = SHIFT_SIZE
    window_size: int = WINDOW_SIZE
    sample_rate: int = ORG_SAMPLE_RATE
    feature_dim: int = FEATURE_DIM
    dur_prediction: bool = False

class SpeechSegment(NamedTuple):
    content: List[float]
    sample_rate: int
    finished: bool

class WriteAction(NamedTuple):
    segment: SpeechSegment
    finished: bool

class ReadAction:
    pass

class OnlineFeatureExtractor:
    def __init__(self, args, cfg):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            {"feature_transforms": ["utterance_cmvn"]}
        )

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples, sr=ORG_SAMPLE_RATE):
        samples = new_samples
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )
        samples = samples[:effective_num_samples]
        waveform, sample_rate = convert_waveform(
            torch.tensor([samples]), sr, to_mono=True, to_sample_rate=16000
        )
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

class TranslationCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

class CTCSequenceGenerator:
    def __init__(self, tgt_dict, models, use_incremental_states=False):
        self.tgt_dict = tgt_dict
        self.models = models
        self.use_incremental_states = use_incremental_states
        self.incremental_states = None

    def generate(self, encoder_out, prefix=None, aux_task_name=None):
        model = self.models[0]
        encoder_out_dict = encoder_out
        if aux_task_name is not None:
            decoder = getattr(model, f"{aux_task_name}_decoder")
        else:
            decoder = model.decoder

        if prefix is None:
            prefix = torch.LongTensor([[self.tgt_dict.bos()]]).to(encoder_out_dict["encoder_out"][0].device)

        incremental_state = self.incremental_states[0] if self.use_incremental_states else None
        lprobs, _ = decoder(prefix, encoder_out=encoder_out_dict, incremental_state=incremental_state)
        
        topk_lprobs, topk_indices = lprobs.topk(1, dim=-1)
        topk_lprobs = topk_lprobs.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        hypos = []
        for i in range(topk_lprobs.size(0)):
            hypos.append(
                {
                    "tokens": torch.cat([prefix[i], topk_indices[i].unsqueeze(0)], dim=0),
                    "score": topk_lprobs[i].item(),
                    "attention": None,
                    "alignment": None,
                    "positional_scores": topk_lprobs[i].unsqueeze(0),
                }
            )

        return [(hypos, [])]

    def reset_incremental_states(self):
        self.incremental_states = None

class CTCDecoder:
    def __init__(self, tgt_dict, models):
        self.tgt_dict = tgt_dict
        self.models = models

    def generate(self, encoder_out, aux_task_name=None):
        model = self.models[0]
        encoder_out_dict = encoder_out
        if aux_task_name is not None:
            decoder = getattr(model, f"{aux_task_name}_decoder")
        else:
            decoder = model.decoder

        logits = decoder(encoder_out_dict)
        lprobs = F.log_softmax(logits, dim=-1)
        
        best_paths = lprobs.argmax(dim=-1)
        hypos = []
        for i in range(best_paths.size(0)):
            hypos.append(
                {
                    "tokens": best_paths[i],
                    "score": lprobs[i].max().item(),
                    "lprobs": lprobs[i],
                    "index": i,
                }
            )

        return [(hypos, [])]

class CodeHiFiGANVocoderWithDur(nn.Module):
    def __init__(self, vocoder_path, vocoder_cfg):
        super().__init__()
        with open(vocoder_cfg) as f:
            cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder.from_pretrained(vocoder_path, cfg)
    
    def forward(self, x, dur_prediction=False):
        code = x["code"]
        if dur_prediction:
            dur = self.vocoder.duration_predictor(code)
        else:
            dur = x.get("dur", None)
        
        if dur is None:
            dur = torch.ones_like(code).float()
        
        y = self.vocoder(code, dur)
        return y, dur

class StreamSpeechS2STAgent:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu else "cpu")
        self.load_model_vocab(config)
        self.setup_generators()
        self.setup_vocoder()
        self.reset()
        self.cache = TranslationCache()
        self.whole_word = config.segment_size >= 640
        self.states = SimpleNamespace(source=None, source_finished=False, target_finished=False)

    def load_model_vocab(self, config):
        filename = config.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        state["cfg"].common['user_dir'] = config.user_dir
        utils.import_user_module(state["cfg"].common)

        task_args = state["cfg"]["task"]
        task_args.data = config.data_bin

        if config.config_yaml is not None:
            task_args.config_yaml = config.config_yaml
            with open(os.path.join(config.data_bin, config.config_yaml), "r") as f:
                config_yaml = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config_yaml:
                config.global_cmvn = np.load(config_yaml["global_cmvn"]["stats_npz_path"])

        if config.multitask_config_yaml is not None:
            task_args.multitask_config_yaml = config.multitask_config_yaml

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

        chunk_size = config.segment_size // 40

        self.models = models

        for model in self.models:
            model.eval()
            model.share_memory()
            if self.config.use_gpu:
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

        self.feature_extractor = OnlineFeatureExtractor(config, config_yaml)

    def setup_generators(self):
        tgt_dict = self.dict["tgt"]
        tgt_dict_mt = self.dict[f"{self.models[0].mt_task_name}"]
        tgt_dict_asr = self.dict["source_unigram"]
        tgt_dict_st = self.dict["ctc_target_unigram"]

        self.ctc_generator = CTCSequenceGenerator(
            tgt_dict, self.models, use_incremental_states=False
        )

        self.asr_ctc_generator = CTCDecoder(tgt_dict_asr, self.models)
        self.st_ctc_generator = CTCDecoder(tgt_dict_st, self.models)

        self.generator = search.SequenceGenerator(
            self.models,
            tgt_dict,
            beam_size=self.config.beam_size,
            max_len_a=1,
            max_len_b=self.config.max_len,
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

        self.generator_mt = search.SequenceGenerator(
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
        )

    def setup_vocoder(self):
        self.vocoder = CodeHiFiGANVocoderWithDur(self.config.vocoder, self.config.vocoder_cfg)
        if self.device == "cuda":
            self.vocoder = self.vocoder.cuda()

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
        self.states = SimpleNamespace(source=None, source_finished=False, target_finished=False)
        try:
            self.generator_mt.model.decoder.clear_cache()
            self.ctc_generator.reset_incremental_states()
        except AttributeError:
            pass

    def policy(self):
        feature = self.feature_extractor(self.states.source)

        if feature.size(0) == 0 and not self.states.source_finished:
            return ReadAction()

        src_indices = feature.unsqueeze(0)
        src_lengths = torch.tensor([feature.size(0)], device=self.device).long()

        self.encoder_outs = self.generator.model.encoder(src_indices, src_lengths)

        finalized_asr = self.asr_ctc_generator.generate(self.encoder_outs, aux_task_name="source_unigram")
        asr_probs = torch.exp(finalized_asr[0][0]["lprobs"])

        for i, hypo in enumerate(finalized_asr[0]):
            tmp = hypo["tokens"].int()
            src_ctc_indices = tmp
            text = "".join([self.dict["source_unigram"][c] for c in tmp])
            text = text.replace("_", " ").replace("▁", " ").replace("<unk>", " ").replace("<s>", "").replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            self.ASR = text

        finalized_st = self.st_ctc_generator.generate(self.encoder_outs, aux_task_name="ctc_target_unigram")
        st_probs = torch.exp(finalized_st[0][0]["lprobs"])

        for i, hypo in enumerate(finalized_st[0]):
            tmp = hypo["tokens"].int()
            tgt_ctc_indices = tmp
            text = "".join([self.dict["ctc_target_unigram"][c] for c in tmp])
            text = text.replace("_", " ").replace("▁", " ").replace("<unk>", " ").replace("<s>", "").replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]

        if not self.states.source_finished:
            src_ctc_prefix_length = src_ctc_indices.size(-1)
            tgt_ctc_prefix_length = tgt_ctc_indices.size(-1)

            self.src_ctc_indices = src_ctc_indices
            if (
                src_ctc_prefix_length < self.src_ctc_prefix_length + self.config.stride_n
                or tgt_ctc_prefix_length < self.tgt_ctc_prefix_length + self.config.stride_n
            ):
                return ReadAction()
            self.src_ctc_prefix_length = max(src_ctc_prefix_length, self.src_ctc_prefix_length)
            self.tgt_ctc_prefix_length = max(tgt_ctc_prefix_length, self.tgt_ctc_prefix_length)
            subword_tokens = ((tgt_ctc_prefix_length - self.config.lagging_k1) // self.config.stride_n) * self.config.stride_n

            if self.whole_word:
                subword_tokens += 1
            new_subword_tokens = (subword_tokens - self.tgt_subwords_indices.size(-1)) if self.tgt_subwords_indices is not None else subword_tokens

            if new_subword_tokens < 1:
                return ReadAction()
        else:
            self.src_ctc_indices = src_ctc_indices
            new_subword_tokens = -1

        new_subword_tokens = int(new_subword_tokens)

        single_model = self.generator.model
        mt_decoder = getattr(single_model, f"{single_model.mt_task_name}_decoder")

        finalized_mt = self.generator_mt.generate([single_model], self.encoder_outs, max_new_tokens=new_subword_tokens)

        if finalized_mt[0][0]["tokens"][-1] == self.generator_mt.eos:
            tgt_subwords_indices = finalized_mt[0][0]["tokens"][:-1].unsqueeze(0)
        else:
            tgt_subwords_indices = finalized_mt[0][0]["tokens"].unsqueeze(0)

        if self.whole_word:
            j = tgt_subwords_indices.size(-1) - 1
            if not self.states.source_finished:
                for j in range(tgt_subwords_indices.size(-1) - 1, -1, -1):
                    if self.generator_mt.tgt_dict[tgt_subwords_indices[0][j]].startswith("▁"):
                        break
                tgt_subwords_indices = tgt_subwords_indices[:, :j+1]
                finalized_mt[0][0]["tokens"] = finalized_mt[0][0]["tokens"][:j+1]

                if j == 0:
                    return ReadAction()

        max_tgt_len = tgt_subwords_indices.size(1)
        prev_output_tokens_mt = torch.full((1, max_tgt_len), mt_decoder.padding_idx, dtype=torch.long, device=self.device)
        prev_output_tokens_mt[:, 0] = self.generator_mt.eos
        prev_output_tokens_mt[:, 1:] = tgt_subwords_indices[:, :-1]

        for i, hypo in enumerate(finalized_mt):
            tokens = [self.generator_mt.tgt_dict[c] for c in hypo[0]["tokens"]]
            text = "".join(tokens)
            text = text.replace("_", " ").replace("▁", " ").replace("<unk>", " ").replace("<s>", "").replace("</s>", "")
            if len(text) > 0 and text[0] == " ":
                text = text[1:]
            self.S2TT = text

        if self.tgt_subwords_indices is not None and torch.equal(self.tgt_subwords_indices, tgt_subwords_indices):
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(self.unfinished_wav.tolist() if self.unfinished_wav is not None else []),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )
        self.tgt_subwords_indices = tgt_subwords_indices

        if not self.states.source_finished:
            if self.prev_output_tokens_mt is not None:
                if torch.equal(self.prev_output_tokens_mt, prev_output_tokens_mt) or prev_output_tokens_mt.size(-1) <= self.prev_output_tokens_mt.size(-1):
                    return ReadAction()
        self.prev_output_tokens_mt = prev_output_tokens_mt
        mt_decoder_out = mt_decoder(prev_output_tokens_mt, encoder_out=self.encoder_outs, features_only=True)[0]

        if self.mt_decoder_out is None:
            self.mt_decoder_out = mt_decoder_out
        else:
            self.mt_decoder_out = torch.cat((self.mt_decoder_out, mt_decoder_out[:, self.mt_decoder_out.size(1):]), dim=1)
        x = self.mt_decoder_out

        if hasattr(single_model, "proj"):
            x = single_model.proj(x)

        mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        if hasattr(single_model, "synthesizer_encoder"):
            t2u_encoder_out = single_model.synthesizer_encoder(x, mt_decoder_padding_mask)
        else:
            t2u_encoder_out = {
                "encoder_out": [x],
                "encoder_padding_mask": [mt_decoder_padding_mask] if mt_decoder_padding_mask is not None else [],
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
        finalized = self.ctc_generator.generate(encoder_outs[0], prefix=self.tgt_units_indices)

        if len(finalized[0][0]["tokens"]) == 0:
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(self.unfinished_wav.tolist() if self.unfinished_wav is not None else []),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )

        for i, hypo in enumerate(finalized):
            tmp = hypo[0]["tokens"].int()
            if tmp[-1] == self.generator.eos:
                tmp = tmp[:-1]
            unit = []
            for c in tmp:
                u = self.generator.tgt_dict[c].replace("<s>", "").replace("</s>", "")
                if u != "":
                    unit.append(int(u))

            if len(unit) > 0 and unit[0] == " ":
                unit = unit[1:]

        cur_unit = unit if self.unit is None else unit[len(self.unit):]
        if len(unit) < 1 or len(cur_unit) < 1:
            if not self.states.source_finished:
                return ReadAction()
            else:
                return WriteAction(
                    SpeechSegment(
                        content=(self.unfinished_wav.tolist() if self.unfinished_wav is not None else []),
                        sample_rate=SAMPLE_RATE,
                        finished=True,
                    ),
                    finished=True,
                )

        x = {
            "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(1, -1),
        }
        wav, dur = self.vocoder(x, self.config.dur_prediction)

        cur_wav_length = dur[:, -len(cur_unit):].sum() * 320
        new_wav = wav[:, -cur_wav_length:]
        if self.unfinished_wav is not None and self.unfinished_wav.size(1) > 0:
            new_wav = torch.cat((self.unfinished_wav, new_wav), dim=1)

        self.wav = wav
        self.unit = unit

        if self.states.source_finished and new_subword_tokens == -1:
            self.states.target_finished = True

        self.S2ST = new_wav.squeeze(0).cpu().numpy().tolist()

        return WriteAction(
            SpeechSegment(
                content=new_wav.squeeze(0).cpu().numpy().tolist(),
                sample_rate=SAMPLE_RATE,
                finished=self.states.source_finished,
            ),
            finished=self.states.target_finished,
        )

class Visualizer:
    @staticmethod
    def plot_waveform(audio: np.ndarray, sample_rate: int, output_path: str):
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(audio)) / sample_rate, audio)
        plt.title("Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sample_rate: int, output_path: str):
        f, t, Sxx = signal.spectrogram(audio, sample_rate)
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title("Spectrogram")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.savefig(output_path)
        plt.close()

class ImprovedS2STRunner:
    def __init__(self, config: Config):
        self.config = config
        self.agent = StreamSpeechS2STAgent(config)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

    def process_input(self, source_path: str):
        with soundfile.SoundFile(source_path) as sf:
            for chunk in sf.blocks(blocksize=self.config.segment_size):
                self.input_queue.put(chunk)
            self.input_queue.put(None)  # Signal end of input

    def process_output(self):
        output_audio = []
        with tqdm(total=100, desc="Translation Progress") as pbar:
            while True:
                segment = self.output_queue.get()
                if segment is None:
                    break
                output_audio.extend(segment.content)
                pbar.update(1)
        
        output_path = os.path.join(self.config.output_dir, "output.wav")
        soundfile.write(output_path, output_audio, SAMPLE_RATE)
        
        Visualizer.plot_waveform(np.array(output_audio), SAMPLE_RATE, 
                                 os.path.join(self.config.output_dir, "output_waveform.png"))
        Visualizer.plot_spectrogram(np.array(output_audio), SAMPLE_RATE, 
                                    os.path.join(self.config.output_dir, "output_spectrogram.png"))

    def run_translation(self):
        while True:
            chunk = self.input_queue.get()
            if chunk is None:
                self.agent.states.source_finished = True
            else:
                self.agent.states.source = chunk

            cache_key = self.agent.states.source.tobytes()
            cached_result = self.agent.cache.get(cache_key)
            
            if cached_result:
                action = cached_result
            else:
                action = self.agent.policy()
                self.agent.cache.set(cache_key, action)
            
            if isinstance(action, WriteAction):
                self.output_queue.put(action.segment)
            
            if self.agent.states.target_finished:
                self.output_queue.put(None)  # Signal end of output
                break

    def run(self, source_path: str):
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        input_thread = threading.Thread(target=self.process_input, args=(source_path,))
        output_thread = threading.Thread(target=self.process_output)
        translation_thread = threading.Thread(target=self.run_translation)

        input_thread.start()
        output_thread.start()
        translation_thread.start()

        input_thread.join()
        translation_thread.join()
        output_thread.join()

        print(f"Translation completed. Output saved in {self.config.output_dir}")
        print("ASR Result:", self.agent.ASR)
        print("S2TT Result:", self.agent.S2TT)
        print("S2ST Result: Audio saved as output.wav")

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

def convert_waveform(waveform, sample_rate, to_mono=True, to_sample_rate=16000):
    if waveform.dim() == 2 and to_mono:
        waveform = waveform.mean(dim=0)
    if sample_rate != to_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, to_sample_rate)
    return waveform, to_sample_rate

def extract_fbank_features(waveform, sample_rate):
    features = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sample_rate
    )
    return features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = ImprovedS2STRunner(config)
    runner.run(args.input)
