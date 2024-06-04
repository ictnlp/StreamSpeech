# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from email.policy import default
import torch
import json
import logging
import numpy as np
from pathlib import Path
from argparse import Namespace
from typing_extensions import Concatenate

from fairseq import utils, metrics
from fairseq.data import Dictionary, encoders
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)


from diseg.datasets.data_cfg import S2TDataConfig
from diseg.datasets.speech_transcription_text_triple_dataset import (
    SpeechToTextTripleDataset,
    SpeechToTextTripleDatasetCreator,
    get_features_or_waveform,
)
import itertools
import os
from typing import Optional
from omegaconf import II
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq import search

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


@register_task("speech_to_text_multitask")
class SpeechToTextMultitask(LegacyFairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument("--text-data", help="manifest root path for text data")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-audio-positions",
            default=900000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-source-positions",
            default=512,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=512,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--batch-size-text",
            type=int,
            metavar="N",
            help="batch size for encoder text input ",
        )
        parser.add_argument(
            "--st-training", action="store_true", help="speech translation training"
        )
        parser.add_argument(
            "--ext-mt-training",
            action="store_true",
            help="external machine transaltion training",
        )
        parser.add_argument("--tgt-lang", type=str, help="target language")
        parser.add_argument(
            "--st-sample-ratio",
            type=float,
            default=1.0,
            help="sample ratio of st dataset",
        )
        parser.add_argument(
            "--mt-sample-ratio",
            type=float,
            default=1.0,
            help="sample ratio of ext mt dataset",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq > 1",
        )
        parser.add_argument(
            "--eval-task", type=str, help="task for evaluation, st or mt"
        )
        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu",
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-detok",
            type=str,
            default="space",
            help="detokenize before computing BLEU (e.g., 'moses'); "
            "required if using --eval-bleu; use 'space' to "
            "disable detokenization; see fairseq.data.encoders "
            "for other options",
        )
        parser.add_argument(
            "--eval-bleu-detok-args",
            type=str,
            metavar="JSON",
            help="args for building the tokenizer, if needed",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            default=False,
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-args",
            type=str,
            metavar="JSON",
            help="generation args for BLUE scoring, "
            'e.g., \'{"beam": 4, "lenpen": 0.6}\'',
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            action="store_true",
            help="print sample generations during validation",
        )
        parser.add_argument(
            "--eval-bleu-bpe",
            type=str,
            metavar="BPE",
            default=None,
            help="args for building the bpe, if needed",
        )
        parser.add_argument(
            "--eval-bleu-bpe-path",
            type=str,
            metavar="BPE",
            help="args for building the bpe, if needed",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()

        self.pre_tokenizer = self.build_tokenizer(self.args)
        self.bpe_tokenizer = self.build_bpe(self.args)

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        src_dict = tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, src_dict, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        concat_dataset = []

        if getattr(self.args, "eval_task", "st") == "st":
            self.args.st_training = True
        if getattr(self.args, "eval_task", "st") == "mt":
            self.args.st_training = True
        if getattr(self.args, "eval_task", "st") == "ext_mt":
            self.args.ext_mt_training = True

        assert self.args.st_training or self.args.ext_mt_training
        if self.args.st_training:
            st_dataset = self.load_st_dataset(split, epoch)
            concat_dataset.append(
                ModalityDatasetItem(
                    "st",
                    st_dataset,
                    [
                        self.args.max_audio_positions,
                        self.args.max_source_positions,
                        self.args.max_target_positions,
                    ],
                    self.args.max_tokens,
                    self.args.batch_size,
                )
            )
        if self.args.ext_mt_training and (is_train_split or not self.args.st_training):
            mt_dataset = self.load_mt_dataset(split)
            concat_dataset.append(
                ModalityDatasetItem(
                    "ext_mt",
                    mt_dataset,
                    [self.args.max_source_positions, self.args.max_target_positions],
                    self.args.max_tokens_text,
                    self.args.batch_size_text,
                )
            )
        self.datasets[split] = MultiModalityDataset(concat_dataset)

    def load_st_dataset(self, split, epoch):
        is_train_split = split.startswith("train")
        return SpeechToTextTripleDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.pre_tokenizer,
            self.bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
        )

    def load_mt_dataset(self, split):
        if split == "dev":
            split = "valid"
        return load_langpair_dataset(
            self.args.text_data,
            split,
            "en",
            self.src_dict,
            self.args.tgt_lang,
            self.tgt_dict,
            prepend_src_lang_tag=self.data_cfg.prepend_src_lang_tag,
            prepend_tgt_lang_tag=self.data_cfg.prepend_tgt_lang_tag,
            combine=True,
            dataset_impl=None,
            upsample_primary=1,
            left_pad_source=False,
            left_pad_target=False,
            remove_eos_from_source=True,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=False,
            truncate_source=False,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        if getattr(self.args, "eval_task", "st") == "st":
            return None
        return self.src_dict

    def max_positions(self):
        return (
            self.args.max_audio_positions,
            self.args.max_source_positions,
            self.args.max_target_positions,
        )

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        args.speaker_to_id = self.speaker_to_id
        model = super(SpeechToTextMultitask, self).build_model(args)

        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextTripleDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )

    def begin_epoch(self, epoch, model):
        model.set_epoch(epoch)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        num_dataset = len(dataset.datasets)
        if num_dataset == 1:
            mult_ratio = [1.0]
        elif num_dataset == 2:
            mult_ratio = [
                self.args.st_sample_ratio,
                self.args.mt_sample_ratio,
            ]

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            # mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            mult_rate=1,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        debug=False,
    ):
        if getattr(self.args, "eval_task", "st") in ["st", "mt", "ext_mt"]:
            if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
                raise ValueError(
                    'Please set "--prefix-size 1" since '
                    "target language ID token is prepended as BOS."
                )
            lang_token_ids = {
                i
                for s, i in self.tgt_dict.indices.items()
                if SpeechToTextTripleDataset.is_lang_tag(s)
            }

            if extra_gen_cls_kwargs is None:
                extra_gen_cls_kwargs = {}
            extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids

            eos_token = (
                args.eos_token
                if "eos_token" in args and args.eos_token is not None
                else self.data_cfg.config.get("eos_token", None)
            )
            """
            if self.data_cfg.prepend_bos_and_append_tgt_lang_tag and not eos_token:
                raise Warning(
                    "Please provide --eos_token to replace eos in sequence generator"
                )
            """
            eos_id = self.tgt_dict.index(eos_token) if eos_token else None
            extra_gen_cls_kwargs["eos"] = eos_id

        if debug:
            return self.build_generator_debug(
                models,
                args,
                seq_gen_cls=None,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )
        else:
            return super().build_generator(
                models,
                args,
                seq_gen_cls=None,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )

    def build_generator_debug(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator_debug import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if (
            getattr(self.args, "eval_task", "st") == "st"
            and "audio" in sample["net_input"]
        ):
            net_input = {
                "src_tokens": sample["net_input"]["audio"],
                "src_lengths": sample["net_input"]["audio_lengths"],
                # "mode": "st",
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
                # 'seg_speech': True,
            }
            sample["net_input"] = net_input
        elif getattr(self.args, "eval_task", "st") == "mt":
            net_input = {
                "src_tokens": sample["net_input"]["transcription"],
                "src_lengths": sample["net_input"]["transcription_lengths"],
                "mode": "mt",
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            }
            sample["net_input"] = net_input
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                if key in logging_outputs[0]:
                    return sum(log[key].cpu().numpy() for log in logging_outputs)
                    # return sum(log[key] for log in logging_outputs)
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.corpus_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe_tokenizer is not None:
                s = self.bpe_tokenizer.decode(s)
            if self.pre_tokenizer is not None:
                s = self.pre_tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            hyps.append(hyp)
            refs.append(ref)

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    remove_eos_from_source,
    max_source_positions,
    max_target_positions,
    prepend_src_lang_tag=False,
    prepend_tgt_lang_tag=False,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in [0]:  # itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_src_lang_tag:
        lang_tag = LANG_TAG_TEMPLATE.format(src)
        src_dataset = PrependTokenDataset(src_dataset, src_dict.index(lang_tag))

    if prepend_tgt_lang_tag:
        lang_tag = LANG_TAG_TEMPLATE.format(tgt)
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.index(lang_tag))
    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        remove_eos_from_source=remove_eos_from_source,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )
