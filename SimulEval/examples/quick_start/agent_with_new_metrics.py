# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from statistics import mean
from simuleval.utils import entrypoint
from simuleval.evaluator.scorers.latency_scorer import (
    register_latency_scorer,
    LatencyScorer,
)
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction


@register_latency_scorer("RTF")
class RTFScorer(LatencyScorer):
    """Real time factor

    Usage:
        --latency-metrics RTF
    """

    def __call__(self, instances) -> float:
        scores = []
        for ins in instances.values():
            scores.append(ins.delays[-1] / ins.source_length)

        return mean(scores)


@entrypoint
class DummyWaitkTextAgent(TextToTextAgent):
    waitk = 3
    vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)

        if lagging >= self.waitk or self.states.source_finished:
            prediction = random.choice(self.vocab)

            return WriteAction(prediction, finished=(lagging <= 1))
        else:
            return ReadAction()
