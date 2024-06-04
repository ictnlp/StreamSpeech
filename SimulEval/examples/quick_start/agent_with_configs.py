# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from argparse import Namespace, ArgumentParser


@entrypoint
class DummyWaitkTextAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        """Initialize your agent here.
        For example loading model, vocab, etc
        """
        super().__init__(args)
        self.waitk = args.waitk
        with open(args.vocab) as f:
            self.vocab = [line.strip() for line in f]

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add customized command line arguments"""
        parser.add_argument("--waitk", type=int, default=3)
        parser.add_argument("--vocab", type=str)

    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)

        if lagging >= self.waitk or self.states.source_finished:
            prediction = random.choice(self.vocab)

            return WriteAction(prediction, finished=(lagging <= 1))
        else:
            return ReadAction()
