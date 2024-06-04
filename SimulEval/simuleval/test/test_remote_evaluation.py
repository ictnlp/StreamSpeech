# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import time
from multiprocessing import Process
from pathlib import Path

import simuleval.cli as cli
from simuleval.utils.functional import find_free_port

ROOT_PATH = Path(__file__).parents[2]


def p1(port, root_path):
    cli.sys.argv[1:] = [
        "--standalone",
        "--remote-port",
        str(port),
        "--agent",
        os.path.join(root_path, "examples", "quick_start", "first_agent.py"),
    ]
    cli.main()
    time.sleep(5)


def p2(port, root_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        cli.sys.argv[1:] = [
            "--remote-eval",
            "--remote-port",
            str(port),
            "--source",
            os.path.join(root_path, "examples", "quick_start", "source.txt"),
            "--target",
            os.path.join(root_path, "examples", "quick_start", "target.txt"),
            "--dataloader",
            "text-to-text",
            "--output",
            tmpdirname,
        ]
        cli.main()


def test_remote_eval(root_path=ROOT_PATH):
    port = find_free_port()

    p_1 = Process(target=p1, args=(port, root_path))
    p_1.start()

    p_2 = Process(target=p2, args=(port, root_path))
    p_2.start()

    p_1.kill()
    p_2.kill()
