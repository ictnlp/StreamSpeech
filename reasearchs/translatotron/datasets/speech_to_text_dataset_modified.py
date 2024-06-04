from pathlib import Path
from typing import Dict, List, Optional

from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    SpeechToTextMultitaskDataset,
    TextTargetMultitaskData,
)


class SpeechToTextDatasetModifiedCreator(SpeechToTextDatasetCreator):
    DEFAULT_TGT_TEXT = ""

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
        multitask: Optional[Dict] = None,
    ) -> SpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s.get(cls.KEY_TGT_TEXT, cls.DEFAULT_TGT_TEXT) for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            SpeechToTextMultitaskDataset if has_multitask else SpeechToTextDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
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

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds
