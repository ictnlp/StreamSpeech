# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
from simuleval import options
from simuleval.utils.agent import build_system_args
from simuleval.utils.slurm import submit_slurm_job
from simuleval.utils.arguments import check_argument
from simuleval.utils import EVALUATION_SYSTEM_LIST
from simuleval.evaluator import (
    build_evaluator,
    build_remote_evaluator,
    SentenceLevelEvaluator,
)
from simuleval.agents.service import start_agent_service
from simuleval.agents import GenericAgent


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)


logger = logging.getLogger("simuleval.cli")


def main():
    if check_argument("remote_eval"):
        remote_evaluate()
        return

    if check_argument("score_only"):
        scoring()
        return

    if check_argument("slurm"):
        submit_slurm_job()
        return

    system, args = build_system_args()

    if check_argument("standalone"):
        start_agent_service(system)
        return

    # build evaluator
    evaluator = build_evaluator(args)
    # evaluate system
    evaluator(system)


def evaluate(system_class: GenericAgent, config_dict: dict = {}):
    EVALUATION_SYSTEM_LIST.append(system_class)

    if check_argument("slurm", config_dict):
        submit_slurm_job(config_dict)
        return

    system, args = build_system_args(config_dict)

    # build evaluator
    evaluator = build_evaluator(args)
    # evaluate system
    evaluator(system)


def scoring():
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser)
    options.add_dataloader_args(parser)
    args = parser.parse_args()
    evaluator = SentenceLevelEvaluator.from_args(args)
    print(evaluator.results)


def remote_evaluate():
    # build evaluator
    parser = options.general_parser()
    options.add_dataloader_args(parser)
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser)
    args = parser.parse_args()
    evaluator = build_remote_evaluator(args)

    # evaluate system
    evaluator.remote_eval()


if __name__ == "__main__":
    main()
