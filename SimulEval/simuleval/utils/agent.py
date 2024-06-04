# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import yaml
import logging
import importlib
from argparse import Namespace
from typing import Union, Optional, Tuple
from pathlib import Path
from simuleval import options
from simuleval.agents import GenericAgent
from simuleval.utils.arguments import cli_argument_list, check_argument

EVALUATION_SYSTEM_LIST = []

logger = logging.getLogger("simuleval.utils.agent")


def import_file(file_path):
    spec = importlib.util.spec_from_file_location("agents", file_path)
    agent_modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_modules)


def get_agent_class(config_dict: Optional[dict] = None) -> GenericAgent:
    class_name = check_argument("agent_class", config_dict)

    if class_name is not None:
        if check_argument("agent"):
            raise RuntimeError("Use either --agent or --agent-class, not both.")
        EVALUATION_SYSTEM_LIST.append(get_agent_class_from_string(class_name))

    system_dir = check_argument("system_dir")
    config_name = check_argument("system_config")

    if system_dir is not None:
        EVALUATION_SYSTEM_LIST.append(get_agent_class_from_dir(system_dir, config_name))

    agent_file = check_argument("agent")
    if agent_file is not None:
        import_file(agent_file)

    if len(EVALUATION_SYSTEM_LIST) == 0:
        raise RuntimeError(
            "Please use @entrypoint decorator to indicate the system you want to evaluate."
        )
    if len(EVALUATION_SYSTEM_LIST) > 1:
        raise RuntimeError("More than one system is not supported right now.")
    return EVALUATION_SYSTEM_LIST[0]


def get_system_config(path: Union[Path, str], config_name) -> dict:
    path = Path(path)
    with open(path / config_name, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error(f"Failed to load configs from {path / config_name}.")
            logging.error(exc)
            sys.exit(1)
    return config_dict


def get_agent_class_from_string(class_name: str) -> GenericAgent:
    try:
        agent_module = importlib.import_module(".".join(class_name.split(".")[:-1]))
        agent_class = getattr(agent_module, class_name.split(".")[-1])
    except Exception as e:
        logger.error(f"Not able to load {class_name}.")
        raise e
    return agent_class


def get_agent_class_from_dir(
    path: Union[Path, str], config_name: str = "main.yaml"
) -> GenericAgent:
    config_dict = get_system_config(path, config_name)
    assert "agent_class" in config_dict
    class_name = config_dict["agent_class"]
    return get_agent_class_from_string(class_name)


def build_system_from_dir(
    path: Union[Path, str],
    config_name: str = "main.yaml",
    overwrite_config_dict: Optional[dict] = None,
) -> GenericAgent:
    path = Path(path)
    config_dict = get_system_config(path, config_name)
    if overwrite_config_dict is not None:
        for key, value in overwrite_config_dict:
            config_dict[key] = value
    agent_class = get_agent_class_from_dir(path, config_name)

    parser = options.general_parser()
    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(config_dict))
    sys.path.append(path.as_posix())

    cur_dir = os.getcwd()
    os.chdir(path.as_posix())
    system = agent_class.from_args(args)
    os.chdir(cur_dir)
    return system


def build_system_args(
    config_dict: Optional[dict] = None,
) -> Tuple[GenericAgent, Namespace]:
    parser = options.general_parser()
    cli_arguments = cli_argument_list(config_dict)
    options.add_evaluator_args(parser)
    options.add_scorer_args(parser, cli_arguments)
    options.add_slurm_args(parser)
    options.add_dataloader_args(parser, cli_arguments)

    if check_argument("system_dir"):
        system = build_system_from_dir(
            check_argument("system_dir"), check_argument("system_config"), config_dict
        )
    else:
        system_class = get_agent_class(config_dict)
        system_class.add_args(parser)
        args, _ = parser.parse_known_args(cli_argument_list(config_dict))
        system = system_class.from_args(args)

    args = parser.parse_args(cli_argument_list(config_dict))

    logger.info(f"System will run on device: {args.device}.")
    system.to(args.device)

    args.source_type = system.source_type
    args.target_type = system.target_type
    return system, args
