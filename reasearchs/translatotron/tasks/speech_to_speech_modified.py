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

    def build_generator_dual_decoder(
        self,
        models,
        args,
        extra_gen_cls_kwargs=None,
    ):
        from translatotron.sequence_generator_multi_decoder import (
            MultiDecoderSequenceGenerator,
        )

        return MultiDecoderSequenceGenerator(
            models,
            self.target_dictionary,
            self.target_dictionary_mt,
            beam_size=max(1, getattr(args, "beam", 1)),
            beam_size_mt=max(1, getattr(args, "beam_mt", 1)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            max_len_a_mt=getattr(args, "max_len_a_mt", 0),
            max_len_b_mt=getattr(args, "max_len_b_mt", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            **extra_gen_cls_kwargs,
        )
