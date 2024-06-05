# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import os.path as op
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from diseg.datasets.audio_utils import get_fbank, get_waveform, get_segment_waveform
from diseg.datasets.data_cfg import S2TDataConfig

logger = logging.getLogger(__name__)


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def is_flac_or_wav_data(data: bytes) -> bool:
    is_flac = data[0] == 102 and data[1] == 76
    is_wav = data[0] == 82 and data[1] == 73
    return is_flac or is_wav


def read_from_uncompressed_zip(file_path, offset, file_size) -> bytes:
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def get_features_from_npy_or_audio(path):
    ext = op.splitext(op.basename(path))[1]
    if ext not in {".npy", ".flac", ".wav"}:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_uncompressed_zip(
    path, byte_offset, byte_size, need_waveform=False
):
    assert path.endswith(".zip")
    data = read_from_uncompressed_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_flac_or_wav_data(data):
        features_or_waveform = get_waveform(f)[0] if need_waveform else get_fbank(f)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_raw_waveform_from_audio(path, byte_offset, byte_size):
    return get_segment_waveform(path, byte_offset, byte_size)[0]


def get_features_or_waveform(path: str, need_waveform=False):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, *extra = path.split(":")
    if not op.exists(_path):
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0:
        if need_waveform:
            return get_waveform(_path)
        return get_features_from_npy_or_audio(_path)
    elif len(extra) == 2:
        extra = [int(i) for i in extra]
        if _path.endswith(".zip"):
            features_or_waveform = get_features_or_waveform_from_uncompressed_zip(
                _path, extra[0], extra[1], need_waveform=need_waveform
            )
        else:  # _path.endswith(".wav")
            features_or_waveform = get_raw_waveform_from_audio(
                _path, int(extra[0]), int(extra[1])
            )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


@dataclass
class SpeechToTextTripleDatasetItem(object):
    index: int
    audio: torch.Tensor
    transcription: torch.Tensor
    target: torch.Tensor
    speaker_id: Optional[int] = None


class SpeechToTextTripleDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        n_frames_per_step=1,
        speaker_to_id=None,
        append_eos=True,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.cfg = cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.speakers = speakers
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.n_frames_per_step = n_frames_per_step
        self.speaker_to_id = speaker_to_id

        self.src_lens = self.get_src_lens_and_check_oov()
        self.tgt_lens = self.get_tgt_lens_and_check_oov()
        self.append_eos = append_eos

        logger.info(self.__repr__())

    def get_src_lens_and_check_oov(self):
        if self.src_texts is None:
            return [0 for _ in range(self.n_samples)]
        src_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_src_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            src_lens.append(len(tokenized))
        logger.info(f"'{self.split}-src' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return src_lens

    def get_tgt_lens_and_check_oov(self):
        if self.tgt_texts is None:
            return [0 for _ in range(self.n_samples)]
        tgt_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_tgt_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.tgt_dict.index(t) == self.tgt_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            tgt_lens.append(len(tokenized))
        logger.info(f"'{self.split}-tgt' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return tgt_lens

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples:_}, '
            f"prepend_tgt_lang_tag={self.cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms}, "
            f"n_frames_per_step={self.n_frames_per_step}"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    @classmethod
    def tokenize(cls, tokenizer, text: str):
        return text if tokenizer is None else tokenizer.encode(text)

    def get_tokenized_src_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer, self.src_texts[index])
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def get_tokenized_tgt_text(self, index: int):
        text = self.tokenize(self.pre_tokenizer, self.tgt_texts[index])
        text = self.tokenize(self.bpe_tokenizer, text)
        return text

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    @classmethod
    def get_lang_tag_idx(cls, lang: str, dictionary: Dictionary):
        lang_tag_idx = dictionary.index(cls.LANG_TAG_TEMPLATE.format(lang))
        assert lang_tag_idx != dictionary.unk()
        return lang_tag_idx

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def _get_source_audio(self, index: int) -> torch.Tensor:
        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.cfg.use_audio_input
        )
        # print(source)
        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).float()
        if self.cfg.use_audio_input:
            source = source.squeeze(0)
        return source

    def __getitem__(self, index: int) -> SpeechToTextTripleDatasetItem:

        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.cfg.use_audio_input
        )
        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).float()
        if self.cfg.use_audio_input:
            source = source.squeeze(0)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        transcription = None
        if self.src_texts is not None:
            text = self.src_texts[index]
            # text = re.sub("[^a-z'() ]", "", text.strip().lower())
            # text = re.sub(" +", " ", text.strip().lower())
            # text = re.sub(" +", " ", text.strip())
            tokenized = self.tokenize_text(text)
            transcription = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=False
            ).long()
            if self.cfg.prepend_src_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                transcription = torch.cat(
                    (torch.LongTensor([lang_tag_idx]), transcription), 0
                )

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]

        return SpeechToTextTripleDatasetItem(
            index=index,
            audio=source,
            transcription=transcription,
            target=target,
            speaker_id=speaker_id,
        )

    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SpeechToTextTripleDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.audio for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.audio.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        transcription = fairseq_data_utils.collate_tokens(
            [x.transcription for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        transcription = transcription.index_select(0, order)
        transcription_lengths = torch.tensor(
            [x.transcription.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)

        target = fairseq_data_utils.collate_tokens(
            [x.target for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        target = target.index_select(0, order)
        target_lengths = torch.tensor(
            [x.target.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)
        prev_output_tokens = fairseq_data_utils.collate_tokens(
            [x.target for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, order)

        # torch.cat((x.transcription,self.tgt_dict.eos()), 0)
        """
        transcription_eos = fairseq_data_utils.collate_tokens(
            [torch.cat((x.transcription,torch.LongTensor([self.tgt_dict.eos()])), 0) for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        transcription_eos = transcription_eos.index_select(0, order)
        

        prev_transcription_tokens_eos  = fairseq_data_utils.collate_tokens(
            [torch.cat((x.transcription,torch.LongTensor([self.tgt_dict.eos()])), 0) for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        prev_transcription_tokens_eos = prev_transcription_tokens_eos.index_select(0, order)
        """
        prev_transcription_tokens = fairseq_data_utils.collate_tokens(
            [x.transcription for x in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True,
        )
        prev_transcription_tokens = prev_transcription_tokens.index_select(0, order)

        ntokens = sum(x.target.size(0) for x in samples)
        src_ntokens = sum(x.transcription.size(0) for x in samples)

        speaker = None
        if self.speaker_to_id is not None:
            speaker = (
                torch.tensor([s.speaker_id for s in samples], dtype=torch.long)
                .index_select(0, order)
                .view(-1, 1)
            )

        is_sub = fairseq_data_utils.collate_tokens(
            [
                torch.tensor(
                    [
                        int(sub.startswith("▁"))
                        for sub in self.tgt_dict.string(x.transcription).split(" ")
                    ],
                    dtype=torch.long,
                )
                for x in samples
            ],
            0,
            eos_idx=0,
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)
        """

        is_sub = fairseq_data_utils.collate_tokens(
            [ torch.tensor([ int(not sub.startswith('<')) for sub in self.tgt_dict.string(x.transcription).split(' ')],dtype=torch.long) for x in samples],
            0,
            eos_idx=0,
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)
        """

        transcription_tokens_num = is_sub.sum(dim=-1, keepdim=False)
        sub2token_id = torch.cumsum(is_sub, dim=-1)

        net_input = {
            "audio": frames,
            "audio_lengths": n_frames,
            "transcription": transcription,
            "transcription_lengths": transcription_lengths,
            "prev_output_tokens": prev_output_tokens,
            "prev_transcription_tokens": prev_transcription_tokens,
        }
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": speaker,
            "target": target,
            "target_lengths": target_lengths,
            "transcription": transcription,
            "transcription_tokens_num": transcription_tokens_num,
            "sub2token_id": sub2token_id,
            "ntokens": ntokens,
            "src_ntokens": src_ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        return self.n_frames[index], self.src_lens[index], self.tgt_lens[index]

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToTextTripleDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
    ) -> SpeechToTextTripleDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextTripleDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
        )

    @classmethod
    def get_size_ratios(
        cls, datasets: List[SpeechToTextTripleDataset], alpha: float = 1.0
    ) -> List[float]:
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""

        id_to_lp, lp_to_sz = {}, defaultdict(int)
        for ds in datasets:
            lang_pairs = {f"{s}->{t}" for s, t in zip(ds.src_langs, ds.tgt_langs)}
            assert len(lang_pairs) == 1
            lang_pair = list(lang_pairs)[0]
            id_to_lp[ds.split] = lang_pair
            lp_to_sz[lang_pair] += sum(ds.n_frames)

        sz_sum = sum(v for v in lp_to_sz.values())
        lp_to_prob = {k: v / sz_sum for k, v in lp_to_sz.items()}
        lp_to_tgt_prob = {k: v**alpha for k, v in lp_to_prob.items()}
        prob_sum = sum(v for v in lp_to_tgt_prob.values())
        lp_to_tgt_prob = {k: v / prob_sum for k, v in lp_to_tgt_prob.items()}
        lp_to_sz_ratio = {
            k: (lp_to_tgt_prob[k] * sz_sum) / v for k, v in lp_to_sz.items()
        }
        size_ratio = [lp_to_sz_ratio[id_to_lp[ds.split]] for ds in datasets]

        p_formatted = {
            k: f"{lp_to_prob[k]:.3f}->{lp_to_tgt_prob[k]:.3f}" for k in lp_to_sz
        }
        logger.info(f"sampling probability balancing: {p_formatted}")
        sr_formatted = {ds.split: f"{r:.3f}" for ds, r in zip(datasets, size_ratio)}
        logger.info(f"balanced sampling size ratio: {sr_formatted}")
        return size_ratio

    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        n_frames_per_step,
        speaker_to_id,
    ) -> SpeechToTextTripleDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        n_frames_per_step: int = 1,
        speaker_to_id=None,
    ) -> SpeechToTextTripleDataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                tgt_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                n_frames_per_step,
                speaker_to_id,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
