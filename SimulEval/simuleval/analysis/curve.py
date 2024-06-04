# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas
from pathlib import Path
from typing import Dict, List, Union


class SimulEvalResults:
    def __init__(self, path: Union[Path, str]) -> None:
        self.path = Path(path)
        scores_path = self.path / "scores"
        if scores_path.exists():
            self.is_finished = True
            with open(self.path / "scores") as f:
                self.scores = json.load(f)
        else:
            self.is_finished = False
            self.scores = {}

    @property
    def quality(self) -> float:
        if self.is_finished:
            if self.scores is None:
                return 0
            return self.scores["Quality"]["BLEU"]
        else:
            return 0

    @property
    def bleu(self) -> float:
        return self.quality

    @property
    def latency(self) -> Dict[str, float]:
        if self.is_finished:
            return self.scores["Latency"]
        else:
            return {}

    @property
    def average_lagging(self):
        return self.latency.get("AL", 0)

    @property
    def average_lagging_ca(self):
        return self.latency.get("AL_CA", 0)

    @property
    def average_proportion(self):
        return self.latency.get("AP", 0)

    @property
    def name(self):
        return self.path.name


class S2SSimulEvalResults(SimulEvalResults):
    @property
    def bow_average_lagging(self):
        return self.latency.get("BOW", {}).get("AL", 0)

    @property
    def cow_average_lagging(self):
        return self.latency.get("COW", {}).get("AL", 0)

    @property
    def eow_average_lagging(self):
        return self.latency.get("EOW", {}).get("AL", 0)


class QualityLatencyAnalyzer:
    def __init__(self) -> None:
        self.score_list: List[SimulEvalResults] = []

    def add_scores_from_path(self, path: Path):
        self.score_list.append(SimulEvalResults(path))

    @classmethod
    def from_paths(cls, path_list: List[Path]):
        analyzer = cls()
        for path in path_list:
            analyzer.add_scores_from_path(path)
        return analyzer

    def summarize(self):
        results = []
        for score in self.score_list:
            if score.bleu == 0:
                continue
            results.append(
                [
                    score.name,
                    round(score.average_lagging / 1000, 2),
                    round(score.average_lagging_ca / 1000, 2),
                    round(score.average_proportion, 2),
                    round(score.bleu, 2),
                ]
            )
        results.sort(key=lambda x: x[1])
        return pandas.DataFrame(results, columns=["name", "AL", "AL(CA)", "AP", "BLEU"])


class S2SQualityLatencyAnalyzer(QualityLatencyAnalyzer):
    def add_scores_from_path(self, path: Path):
        self.score_list.append(S2SSimulEvalResults(path))

    def summarize(self):
        results = []
        for score in self.score_list:
            if score.bleu == 0:
                continue
            results.append(
                [
                    score.name,
                    round(score.bow_average_lagging / 1000, 2),
                    round(score.cow_average_lagging / 1000, 2),
                    round(score.eow_average_lagging / 1000, 2),
                    round(score.bleu, 2),
                ]
            )
        results.sort(key=lambda x: x[1])
        return pandas.DataFrame(
            results, columns=["name", "BOW_AL", "COW_AL", "EOW_AL", "BLEU"]
        )
