from pathlib import Path

from fairseq.tasks import register_task
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
from translatotron.datasets.speech_to_speech_dataset_modified import (
    SpeechToSpeechDatasetModifiedCreator,
)
from translatotron.datasets.speech_to_speech_data_cfg_modified import (
    S2SDataConfigModified,
)


@register_task("speech_to_speech_modified")
class SpeechToSpeechTaskModified(SpeechToSpeechTask):

    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, tgt_dict, infer_tgt_lang_id)
        self.data_cfg = S2SDataConfigModified(Path(args.data) / args.config_yaml)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = SpeechToSpeechDatasetModifiedCreator.from_tsv(
            root=self.args.data,
            cfg=self.data_cfg,
            splits=split,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.tgt_dict,
            is_train_split=split.startswith("train"),
            n_frames_per_step=self.args.n_frames_per_step,
            # multitask=self.multitask_tasks,
            multitask=self.multitask_tasks if split.startswith("train") else None,
        )
