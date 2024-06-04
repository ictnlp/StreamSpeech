from pathlib import Path

from fairseq.tasks import register_task
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
from ctc_unity.datasets.speech_to_speech_dataset_modified import (
    SpeechToSpeechDatasetModifiedCreator,
)
from ctc_unity.datasets.speech_to_speech_data_cfg_modified import S2SDataConfigModified


@register_task("speech_to_speech_ctc")
class SpeechToSpeechCTCTask(SpeechToSpeechTask):

    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        tgt_blank_index = tgt_dict.add_symbol("<blank>")
        self.tgt_dict = tgt_dict
        self.tgt_dict.blank_index = tgt_blank_index
        super().__init__(args, tgt_dict, infer_tgt_lang_id)
        self.blank_symbol = "<blank>"

    def build_generator_dual_decoder(
        self,
        models,
        args,
        extra_gen_cls_kwargs=None,
    ):
        from ctc_unity.sequence_generator_multi_decoder_ctc import (
            CTCMultiDecoderSequenceGenerator,
        )

        return CTCMultiDecoderSequenceGenerator(
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
