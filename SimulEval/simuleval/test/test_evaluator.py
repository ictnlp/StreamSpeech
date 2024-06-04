# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path

import simuleval.cli as cli

ROOT_PATH = Path(__file__).parents[2]


def test_score_only(root_path=ROOT_PATH):
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--agent",
            os.path.join(root_path, "examples", "quick_start", "first_agent.py"),
            "--source",
            os.path.join(root_path, "examples", "quick_start", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "quick_start", "target.txt"),
            "--output",
            tmpdirname,
        ]
        cli.main()
        cli.sys.argv[1:] = ["--score-only", "--output", tmpdirname]
        cli.main()
