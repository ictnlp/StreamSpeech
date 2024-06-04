# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .evaluator import SentenceLevelEvaluator
from .remote import RemoteEvaluator


def build_evaluator(args):
    return SentenceLevelEvaluator.from_args(args)


def build_remote_evaluator(args):
    return RemoteEvaluator(build_evaluator(args))
