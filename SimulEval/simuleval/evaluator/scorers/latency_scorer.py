# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from statistics import mean
from pathlib import Path
import subprocess
import logging
import textgrid
import sys
import shutil
from typing import List, Union, Dict
from simuleval.evaluator.instance import (
    TextInputInstance,
    TextOutputInstance,
    SpeechOutputInstance,
    Instance,
    LogInstance,
    SpeechOutputInstance,
)
from argparse import ArgumentParser, Namespace
from subprocess import Popen, PIPE

logger = logging.getLogger("simuleval.latency_scorer")

LATENCY_SCORERS_DICT = {}
LATENCY_SCORERS_NAME_DICT = {}


def register_latency_scorer(name):
    def register(cls):
        LATENCY_SCORERS_DICT[name] = cls
        LATENCY_SCORERS_NAME_DICT[cls.__name__] = name
        return cls

    return register


class LatencyScorer:
    metric = None
    add_duration = False

    def __init__(
        self, computation_aware: bool = False, use_ref_len: bool = True
    ) -> None:
        super().__init__()
        self.use_ref_len = use_ref_len
        self.computation_aware = computation_aware

    @property
    def timestamp_type(self):
        return "delays" if not self.computation_aware else "elapsed"

    def compute(self, *args):
        raise NotImplementedError

    def get_delays_lengths(self, ins: Instance):
        """
        Args:
            ins Instance: one instance

        Returns:
            A tuple with the 3 elements:
            delays (List[Union[float, int]]): Sequence of delays.
            src_len (Union[float, int]): Length of source sequence.
            tgt_len (Union[float, int]): Length of target sequence.
        """
        delays = getattr(ins, self.timestamp_type, None)
        assert delays

        if not self.use_ref_len or ins.reference is None:
            tgt_len = len(delays)
        else:
            tgt_len = ins.reference_length
        src_len = ins.source_length
        return delays, src_len, tgt_len

    @property
    def metric_name(self) -> str:
        return LATENCY_SCORERS_NAME_DICT[self.__class__.__name__]

    def __call__(self, instances: Dict[int, Instance]) -> float:
        scores = []
        for index, ins in instances.items():
            if isinstance(ins, TextInputInstance):
                if self.computation_aware:
                    raise RuntimeError(
                        "The computation aware latency is not supported on text input."
                    )
            delays = getattr(ins, self.timestamp_type, None)
            if delays is None or len(delays) == 0:
                logger.warn(f"Instance {index} has no delay information. Skipped")
                continue
            score = self.compute(ins)
            ins.metrics[self.metric_name] = score
            scores.append(score)

        return mean(scores)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            computation_aware=args.computation_aware,
            use_ref_len=not args.no_use_ref_len,
        )


@register_latency_scorer("AL")
class ALScorer(LatencyScorer):
    r"""
    Average Lagging (AL) from
    `STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework <https://arxiv.org/abs/1810.08398>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        AL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{|Y|}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length

    Usage:
        ----latency-metrics AL
    """  # noqa: E501

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)

        if delays[0] > source_length:
            return delays[0]

        AL = 0
        gamma = target_length / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            AL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        AL /= tau
        return AL


@register_latency_scorer("LAAL")
class LAALScorer(ALScorer):
    r"""
    Length Adaptive Average Lagging (LAAL) as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_.
    The name was suggested in `Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length difference between the hypothesis and reference.

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,

    .. math::

        LAAL = \frac{1}{\tau} \sum_i^\tau D_i - (i - 1) \frac{|X|}{max(|Y|,|Y*|)}

    Where

    .. math::

        \tau = argmin_i(D_i = |X|)

    When reference was given, :math:`|Y|` would be the reference length, and :math:`|Y*|` is the length of the hypothesis.

    Usage:
        ----latency-metrics LAAL
    """

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins: Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if delays[0] > source_length:
            return delays[0]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        LAAL /= tau
        return LAAL


@register_latency_scorer("AP")
class APScorer(LatencyScorer):
    r"""
    Average Proportion (AP) from
    `Can neural machine translation do simultaneous translation? <https://arxiv.org/abs/1606.02012>`_

    Give source :math:`X`, target :math:`Y`, delays :math:`D`,
    the AP is calculated as:

    .. math::

        AP = \frac{1}{|X||Y]} \sum_i^{|Y|} D_i

    Usage:
        ----latency-metrics AP
    """

    def compute(self, ins: Instance) -> float:
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        return sum(delays) / (source_length * target_length)


@register_latency_scorer("DAL")
class DALScorer(LatencyScorer):
    r"""
    Differentiable Average Lagging (DAL) from
    Monotonic Infinite Lookback Attention for Simultaneous Machine Translation
    (https://arxiv.org/abs/1906.05218)

    Usage:
        ----latency-metrics DAL
    """

    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)

        DAL = 0
        target_length = len(delays)
        gamma = target_length / source_length
        g_prime_last = 0
        for i_minus_1, g in enumerate(delays):
            if i_minus_1 + 1 == 1:
                g_prime = g
            else:
                g_prime = max([g, g_prime_last + 1 / gamma])

            DAL += g_prime - i_minus_1 / gamma
            g_prime_last = g_prime

        DAL /= target_length
        return DAL


@register_latency_scorer("ATD")
class ATDScorer(LatencyScorer):
    r"""
    Average Token Delay (ATD) from
    Average Token Delay: A Latency Metric for Simultaneous Translation
    (https://arxiv.org/abs/2211.13173)

    Different from speech segments, text tokens have no length
    and multiple tokens can be output at the same time like subtitle.
    Therefore, we set its length to be 0. However, to calculate latency in text-text,
    we give virtual time 1 for the length of text tokens.

    Usage:
        ----latency-metrics ATD
    """

    def __call__(self, instances) -> float:  # noqa C901
        if isinstance(instances[0], TextInputInstance):
            TGT_TOKEN_LEN = 1
            SRC_TOKEN_LEN = 1
            INPUT_TYPE = "text"
            OUTPUT_TYPE = "text"
        else:
            SRC_TOKEN_LEN = 300  # 300ms per word
            INPUT_TYPE = "speech"
            if isinstance(instances[0], TextOutputInstance) or isinstance(
                instances[0], LogInstance
            ):
                TGT_TOKEN_LEN = 0
                OUTPUT_TYPE = "text"
            else:
                TGT_TOKEN_LEN = 300
                OUTPUT_TYPE = "speech"

        scores = []
        for index, ins in instances.items():
            delays = getattr(ins, "delays", None)
            if delays is None or len(delays) == 0:
                logger.warn(f"Instance {index} has no delay information. Skipped")
                continue

            if self.computation_aware:
                elapsed = getattr(ins, "elapsed", None)
                if elapsed is None or len(elapsed) == 0:
                    logger.warn(
                        f"Instance {index} has no computational delay information. Skipped"
                    )
                    continue
                if elapsed != [0] * len(delays):
                    compute_elapsed = self.subtract(elapsed, delays)
                    compute_times = self.subtract(
                        compute_elapsed, [0] + compute_elapsed[:-1]
                    )
                else:
                    compute_times = elapsed
            else:
                compute_times = [0] * len(delays)

            chunk_sizes = {"src": [0], "tgt": [0]}
            token_to_chunk = {"src": [0], "tgt": [0]}
            token_to_time = {"src": [0], "tgt": [0]}

            tgt_token_lens = []
            delays_no_duplicate = sorted(set(delays), key=delays.index)

            if OUTPUT_TYPE == "text":
                prev_delay = None
                for delay in delays:
                    if delay != prev_delay:
                        chunk_sizes["tgt"].append(1)
                    else:
                        chunk_sizes["tgt"][-1] += 1
                    prev_delay = delay
                for i, chunk_size in enumerate(chunk_sizes["tgt"][1:], 1):
                    token_to_chunk["tgt"] += [i] * chunk_size
                tgt_token_lens = [TGT_TOKEN_LEN] * len(delays)
            else:
                s2s_delays = []
                s2s_compute_times = []
                chunk_durations = []
                chunk_compute_times = []
                prev_delay = None
                for delay, compute_time, duration in zip(
                    delays, compute_times, ins.durations
                ):
                    if delay != prev_delay:
                        chunk_durations.append(duration)
                        chunk_compute_times.append(compute_time)
                    else:
                        chunk_durations[-1] += duration
                        chunk_compute_times[-1] += compute_time
                    prev_delay = delay
                for i, chunk_duration in enumerate(chunk_durations, 1):
                    num_tokens, rest = divmod(chunk_duration, TGT_TOKEN_LEN)
                    token_lens = int(num_tokens) * [TGT_TOKEN_LEN] + (
                        [rest] if rest != 0 else []
                    )
                    tgt_token_lens += token_lens
                    chunk_sizes["tgt"] += [len(token_lens)]
                    token_to_chunk["tgt"] += [i] * len(token_lens)
                    s2s_delays += [delays_no_duplicate[i - 1]] * len(token_lens)
                    s2s_compute_times += [
                        chunk_compute_times[i - 1] / len(token_lens)
                    ] * len(token_lens)
                delays = s2s_delays
                compute_times = s2s_compute_times

            if INPUT_TYPE == "text":
                chunk_sizes["src"] += self.subtract(
                    delays_no_duplicate, [0] + delays_no_duplicate[:-1]
                )
                for i, chunk_size in enumerate(chunk_sizes["src"][1:], 1):
                    token_lens = chunk_size * [SRC_TOKEN_LEN]
                    for token_len in token_lens:
                        token_to_time["src"].append(
                            token_to_time["src"][-1] + token_len
                        )
                        token_to_chunk["src"].append(i)
            else:
                chunk_durations = self.subtract(
                    delays_no_duplicate, [0] + delays_no_duplicate[:-1]
                )
                for i, chunk_duration in enumerate(chunk_durations, 1):
                    num_tokens, rest = divmod(chunk_duration, SRC_TOKEN_LEN)
                    token_lens = int(num_tokens) * [SRC_TOKEN_LEN] + (
                        [rest] if rest != 0 else []
                    )
                    chunk_sizes["src"] += [len(token_lens)]
                    for token_len in token_lens:
                        token_to_time["src"].append(
                            token_to_time["src"][-1] + token_len
                        )
                        token_to_chunk["src"].append(i)

            for delay, compute_time, token_len in zip(
                delays, compute_times, tgt_token_lens
            ):
                tgt_start_time = max(delay, token_to_time["tgt"][-1])
                token_to_time["tgt"].append(tgt_start_time + token_len + compute_time)

            scores.append(self.compute(chunk_sizes, token_to_chunk, token_to_time))

        return mean(scores)

    def subtract(self, arr1, arr2):
        return [x - y for x, y in zip(arr1, arr2)]

    def compute(
        self,
        chunk_sizes: Dict[str, List[Union[float, int]]],
        token_to_chunk: Dict[str, List[Union[float, int]]],
        token_to_time: Dict[str, List[Union[float, int]]],
    ) -> float:
        """
        Function to compute latency on one sentence (instance).
        Args:
            chunk_sizes Dict[str, List[Union[float, int]]]: Sequence of chunk sizes for source and target.
            token_to_chunk Dict[str, List[Union[float, int]]]: Sequence of chunk indices to which the tokens belong for source and target.
            token_to_time Dict[str, List[Union[float, int]]]: Sequence of ending times of tokens for source and target.

        Returns:
            float: the latency score on one sentence.
        """  # noqa C501

        tgt_to_src = []

        for t in range(1, len(token_to_chunk["tgt"])):
            chunk_id = token_to_chunk["tgt"][t]
            AccSize_x = sum(chunk_sizes["src"][:chunk_id])
            AccSize_y = sum(chunk_sizes["tgt"][:chunk_id])

            S = t - max(0, AccSize_y - AccSize_x)
            current_src_size = sum(chunk_sizes["src"][: chunk_id + 1])

            if S < current_src_size:
                tgt_to_src.append((t, S))
            else:
                tgt_to_src.append((t, current_src_size))

        atd_delays = []

        for t, s in tgt_to_src:
            atd_delay = token_to_time["tgt"][t] - token_to_time["src"][s]
            atd_delays.append(atd_delay)

        return float(mean(atd_delays))


@register_latency_scorer("NumChunks")
class NumChunksScorer(LatencyScorer):
    """Number of chunks (of speech/text) in output

    Usage:
        ----latency-metrics NumChunks

    """

    def compute(self, ins: Instance):
        delays, _, _ = self.get_delays_lengths(ins)
        return len(delays)


@register_latency_scorer("DiscontinuitySum")
class DiscontinuitySumScorer(LatencyScorer):
    """Sum of discontinuity in speech output

    Usage:
        ----latency-metrics DiscontinuitySum

    """

    def compute(self, ins: Instance):
        assert isinstance(ins, SpeechOutputInstance)
        return sum(ins.silences)


@register_latency_scorer("DiscontinuityAve")
class DiscontinuityAveScorer(LatencyScorer):
    """Average of discontinuities in speech output

    Usage:
        ----latency-metrics DiscontinuityAve

    """

    def compute(self, ins: Instance):
        assert isinstance(ins, SpeechOutputInstance)
        if len(ins.silences) == 0:
            return 0
        return sum(ins.silences) / len(ins.silences)


@register_latency_scorer("DiscontinuityNum")
class DiscontinuityNumScorer(LatencyScorer):
    """Number of discontinuities in speech output

    Usage:
        ----latency-metrics DiscontinuityNum

    """

    def compute(self, ins: Instance):
        assert isinstance(ins, SpeechOutputInstance)
        return len(ins.silences)


@register_latency_scorer("StartOffset")
class StartOffsetScorer(LatencyScorer):
    """Starting offset of the translation

    Usage:
        ----latency-metrics StartOffset

    """

    def compute(self, ins: Instance):
        delays, _, _ = self.get_delays_lengths(ins)
        return delays[0]


@register_latency_scorer("EndOffset")
class EndOffsetScorer(LatencyScorer):
    """Ending offset of the translation

    Usage:
        ----latency-metrics EndOffset

    """

    def compute(self, ins: Instance):
        delays, source_length, _ = self.get_delays_lengths(ins)
        if isinstance(ins, SpeechOutputInstance) or (
            isinstance(ins, LogInstance) and len(ins.intervals) > 0
        ):
            delays = [start + duration for start, duration in ins.intervals]
        return delays[-1] - source_length


@register_latency_scorer("RTF")
class RTFScorer(LatencyScorer):
    """Compute Real Time Factor (RTF)

    Usage:
        ----latency-metrics (RTF)

    """

    def compute(self, ins: Instance):
        delays, source_length, _ = self.get_delays_lengths(ins)
        if isinstance(ins, SpeechOutputInstance):
            delays = [start + duration for start, duration in ins.intervals]
        return delays[-1] / source_length


def speechoutput_alignment_latency_scorer(scorer_class):  # noqa C901
    class Klass(scorer_class):
        def __init__(self, **kargs) -> None:
            assert getattr(self, "boundary_type", None) in [
                "BOW",
                "EOW",
                "COW",
            ], self.boundary_type
            super().__init__(**kargs)
            if self.computation_aware:
                raise RuntimeError(
                    "The computation aware latency for speech output is not supported yet"
                )

        @property
        def timestamp_type(self):
            return "aligned_delays"

        def __call__(self, instances) -> float:
            self.prepare_alignment(instances)
            return super().__call__(instances)

        def prepare_alignment(self, instances):
            try:
                subprocess.check_output(
                    "mfa version", shell=True, stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as grepexc:
                logger.error(grepexc.output.decode("utf-8").strip())
                logger.error("Please make sure the mfa>=2.0.6 is correctly installed. ")
                sys.exit(1)

            output_dir = Path(instances[0].prediction).absolute().parent.parent
            align_dir = output_dir / "align"
            if not align_dir.exists():
                logger.info("Align target transcripts with speech.")
                temp_dir = Path(output_dir) / "mfa"
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir.mkdir(exist_ok=True)
                original_model_path = Path.home() / "Documents/MFA/pretrained_models"
                acoustic_model_path = temp_dir / "acoustic.zip"
                acoustic_model_path.symlink_to(
                    original_model_path / "acoustic" / "english_mfa.zip"
                )
                dictionary_path = temp_dir / "dict"
                dictionary_path.symlink_to(
                    original_model_path / "dictionary" / "english_mfa.dict"
                )
                mfa_command = (
                    f"mfa align {output_dir  / 'wavs'} {dictionary_path.as_posix()} {acoustic_model_path.as_posix()}"
                    + f" {align_dir.as_posix()} --clean --overwrite --temporary_directory  {temp_dir.as_posix()}"
                )
                logger.info(mfa_command)

                subprocess.run(
                    mfa_command,
                    shell=True,
                    check=True,
                )
            else:
                logger.info("Found existing alignment")

            for file in align_dir.iterdir():
                if file.name.endswith("TextGrid"):
                    index = int(file.name.split("_")[0])
                    target_offset = instances[index].delays[0]
                    info = textgrid.TextGrid.fromFile(file)
                    delays = []
                    for interval in info[0]:
                        if len(interval.mark) > 0:
                            if self.boundary_type == "BOW":
                                delays.append(target_offset + 1000 * interval.minTime)
                            elif self.boundary_type == "EOW":
                                delays.append(target_offset + 1000 * interval.maxTime)
                            else:
                                delays.append(
                                    target_offset
                                    + 0.5 * (interval.maxTime + interval.minTime) * 1000
                                )
                    setattr(instances[index], self.timestamp_type, delays)

    return Klass


for boundary_type in ["BOW", "COW", "EOW"]:
    for metric in ["AL", "LAAL", "AP", "DAL", "ATD", "StartOffset", "EndOffset"]:

        @register_latency_scorer(f"{metric}_SpeechAlign_{boundary_type}")
        @speechoutput_alignment_latency_scorer
        class SpeechAlignScorer(LATENCY_SCORERS_DICT[metric]):  # type: ignore
            f"""Compute {metric} based on alignment ({boundary_type})

            Usage:
                ----latency-metrics {metric}_SpeechAlign_{boundary_type}
            """
            boundary_type = boundary_type
            __name__ = f"{metric}SpeechAlign{boundary_type}Scorer"
