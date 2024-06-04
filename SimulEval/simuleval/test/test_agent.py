# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from pathlib import Path

import simuleval.cli as cli
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import TextSegment

ROOT_PATH = Path(__file__).parents[2]


def test_agent(root_path=ROOT_PATH):
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


def test_statelss_agent(root_path=ROOT_PATH):
    class DummyWaitkTextAgent(TextToTextAgent):
        waitk = 0
        vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

        def policy(self, states=None):
            if states is None:
                states = self.states

            lagging = len(states.source) - len(states.target)

            if lagging >= self.waitk or states.source_finished:
                prediction = self.vocab[len(states.source)]

                return WriteAction(prediction, finished=(lagging <= 1))
            else:
                return ReadAction()

    args = None
    agent_stateless = DummyWaitkTextAgent.from_args(args)
    agent_state = agent_stateless.build_states()
    agent_stateful = DummyWaitkTextAgent.from_args(args)

    for _ in range(10):
        segment = TextSegment(0, "A")
        output_1 = agent_stateless.pushpop(segment, agent_state)
        output_2 = agent_stateful.pushpop(segment)
        assert output_1.content == output_2.content
