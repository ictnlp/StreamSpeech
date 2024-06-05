#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    print(saved_cfg.task)
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    rw = []
    gs = []
    has_target = True
    wps_meter = TimeMeter()

    translation_results = {}

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos, d, srcs, extra = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            rw01 = d201(d[sample_id], sample["net_input"]["src_lengths"][i])
            rw.append(rw01)

            translation_results[sample_id] = {}
            translation_results[sample_id]["transition_probs"] = extra[
                "transition_probs"
            ][sample_id].cpu()
            translation_results[sample_id]["attention"] = extra["attention"][
                sample_id
            ].cpu()
            translation_results[sample_id]["states_output"] = tgt_dict.string(
                extra["states_output"][sample_id], include_eos=True
            )

            cur_scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    bpe_src_str = src_dict.string(src_tokens, include_eos=True)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )
                    bpe_ref_str = tgt_dict.string(target_tokens, include_eos=True)

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )

                bpe_tgt_str = tgt_dict.string(
                    hypo["tokens"].int().cpu(), include_eos=True
                )
                detok_hypo_str = decode_fn(hypo_str)

                print(
                    "RW-{}\t{}\t{}".format(sample_id, np.around(RW2AL(rw01), 2), rw01),
                    file=output_file,
                )
                translation_results[sample_id]["RW"] = rw01

                translation_results[sample_id]["AL"] = np.around(RW2AL(rw01), 2)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2

                    print("BS-{}\t{}".format(sample_id, bpe_src_str), file=output_file)
                    translation_results[sample_id]["bpe_source"] = bpe_src_str
                    translation_results[sample_id]["bpe_reference"] = bpe_ref_str

                    print(
                        "BT-{}\t{}".format(sample_id, bpe_tgt_str),
                        file=output_file,
                    )

                    translation_results[sample_id]["bpe_target"] = bpe_tgt_str

                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    """
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )
                    """

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                        cur_scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)
                        cur_scorer.add(target_tokens, hypo_tokens)
                    translation_results[sample_id]["BLEU"] = cur_scorer.result_string()

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )
    cw, ap, al, dal = compute_delay(rw, is_weight_ave=True)

    print("CW score: ", cw)
    print("AP score: ", ap)
    print("AL score: ", al)
    print("DAL score: ", dal)

    return scorer


def d201(d, src):
    # print("+++",d)
    s = "0 " * int(d[0]) + "1 "
    for i in range(1, len(d)):
        s = s + "0 " * int((min(d[i], src) - min(d[i - 1], src))) + "1 "
    if src > d[-1]:
        s = s + "0 " * int(src - d[-1])
    return s


def compute_delay(rw, is_weight_ave=False):
    CWs, ALs, APs, DALs, Lsrc = [], [], [], [], []
    for line in rw:
        line = line.strip()
        al_ans = RW2AL(line)
        dal_ans = RW2DAL(line)
        ap_ans = RW2AP(line)
        cw_ans = RW2CW(line)
        if al_ans is not None:
            ALs.append(al_ans)
            DALs.append(dal_ans)
            APs.append(ap_ans)
            CWs.append(cw_ans)
            Lsrc.append(line.count("0"))

    CW = np.average(CWs) if is_weight_ave else np.average(CWs, weights=Lsrc)
    AL = np.average(ALs) if is_weight_ave else np.average(ALs, weights=Lsrc)
    DAL = np.average(DALs) if is_weight_ave else np.average(DALs, weights=Lsrc)
    AP = np.average(APs) if is_weight_ave else np.average(APs, weights=Lsrc)
    return CW, AP, AL, DAL


def RW2CW(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0
    c = s.count("01")

    if c == 0:
        return 0
    else:
        return x / c


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AP(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    return sum(curr) / x / y


# s is RW sequence, in format of '0 0 0 1 1 0 1 0 1', or 'R R R W W R W R W', flexible on blank/comma
def RW2AL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
        if i == "1" and count == x:
            break
    y1 = len(curr)
    diag = [(t - 1) / rate for t in range(1, y1 + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr, diag)) / y1


def RW2DAL(s, add_eos=False):
    trantab = str.maketrans("RrWw", "0011")
    if isinstance(s, str):
        s = s.translate(trantab).replace(" ", "").replace(",", "")
        if (
            add_eos
        ):  # to add eos token for both src and tgt if you did not do it during RW generating
            idx = s.rfind("0")
            s = (
                s[: idx + 1] + "0" + s[idx + 1 :] + "1"
            )  # remove last 0/1(<eos>) to keep actuall setence length
            # s = (s[:idx]+s[idx+1:])[:-1]  # remove last 0/1(<eos>) to keep actuall setence length
    else:
        return None
    x, y = s.count("0"), s.count("1")
    if x == 0 or y == 0:
        return 0

    count = 0
    rate = y / x
    curr = []
    curr1 = []
    for i in s:
        if i == "0":
            count += 1
        else:
            curr.append(count)
    curr1.append(curr[0])
    for i in range(1, y):
        curr1.append(max(curr[i], curr1[i - 1] + 1 / rate))

    diag = [(t - 1) / rate for t in range(1, y + 1)]
    return sum(l1 - l2 for l1, l2 in zip(curr1, diag)) / y


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="hmt_transformer_iwslt_de_en",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
