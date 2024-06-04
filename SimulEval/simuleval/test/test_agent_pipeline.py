# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import simuleval.cli as cli
from simuleval.agents import AgentPipeline, TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import TextSegment

ROOT_PATH = Path(__file__).parents[2]


def test_pipeline_cmd(root_path=ROOT_PATH):
    cli.sys.argv[1:] = [
        "--agent",
        os.path.join(root_path, "examples", "quick_start", "agent_pipeline.py"),
        "--source",
        os.path.join(root_path, "examples", "quick_start", "source.txt"),
        "--target",
        os.path.join(root_path, "examples", "quick_start", "target.txt"),
    ]
    cli.main()


def test_pipeline():
    class DummyWaitkTextAgent(TextToTextAgent):
        waitk = 0
        vocab = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

        def policy(self):
            lagging = len(self.states.source) - len(self.states.target)

            if lagging >= self.waitk or self.states.source_finished:
                prediction = self.vocab[len(self.states.source)]

                return WriteAction(prediction, finished=(lagging <= 1))
            else:
                return ReadAction()

    class DummyWait2TextAgent(DummyWaitkTextAgent):
        waitk = 2

    class DummyWait4TextAgent(DummyWaitkTextAgent):
        waitk = 4

    class DummyPipeline(AgentPipeline):
        pipeline = [DummyWait2TextAgent, DummyWait4TextAgent]

    args = None
    agent_1 = DummyPipeline.from_args(args)
    agent_2 = DummyPipeline.from_args(args)
    for _ in range(10):
        segment = TextSegment(0, "A")
        output_1 = agent_1.pushpop(segment)
        agent_2.push(segment)
        output_2 = agent_2.pop()
        assert output_1.content == output_2.content
