##########################################
# Simultaneous Speech-to-Speech Translation Agent for StreamSpeech
#
# StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning (ACL 2024)
##########################################

from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import WriteAction, ReadAction
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from pathlib import Path
from typing import Any, Dict, Optional, Union
from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import extract_fbank_features
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


SHIFT_SIZE = 10
WINDOW_SIZE = 25
ORG_SAMPLE_RATE = 48000
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"
DEFAULT_EOS = 2


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
        self.device = "cuda" if args.device == "gpu" else "cpu"
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


@entrypoint
class StreamSpeechS2STAgent(SpeechToSpeechAgent):
    """
    Incrementally feed text to this offline Fastspeech2 TTS model,
    with a minimum numbers of phonemes every chunk.
    """

    def __init__(self, args):
        super().__init__(args)
        self.eos = DEFAULT_EOS

        self.gpu = self.args.device == "gpu"
        self.device = "cuda" if args.device == "gpu" else "cpu"

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
        from agent.tts.vocoder import CodeHiFiGANVocoderWithDur

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

        with open(args.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoderWithDur(args.vocoder, vocoder_cfg)
        if self.device == "cuda":
            self.vocoder = self.vocoder.cuda()
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

        if args.source_segment_size >= 640:
            self.whole_word = True
        else:
            self.whole_word = False

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

        chunk_size = args.source_segment_size // 40

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

        x = {
            "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(
                1, -1
            ),
        }
        wav, dur = self.vocoder(x, self.dur_prediction)

        cur_wav_length = dur[:, -len(cur_unit) :].sum() * 320
        new_wav = wav[-cur_wav_length:]
        if self.unfinished_wav is not None and len(self.unfinished_wav) > 0:
            new_wav = torch.cat((self.unfinished_wav, new_wav), dim=0)

        self.wav = wav
        self.unit = unit

        # A SpeechSegment has to be returned for speech-to-speech translation system
        if self.states.source_finished and new_subword_tokens == -1:
            self.states.target_finished = True
            self.reset()

        return WriteAction(
            SpeechSegment(
                content=new_wav.tolist(),
                sample_rate=SAMPLE_RATE,
                finished=self.states.source_finished,
            ),
            finished=self.states.target_finished,
        )
