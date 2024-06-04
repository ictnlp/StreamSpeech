# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from pathlib import Path
from typing import Callable, List, Union, Optional
from .dataloader import GenericDataloader
from simuleval.data.dataloader import register_dataloader
from argparse import Namespace


@register_dataloader("text-to-text")
class TextToTextDataloader(GenericDataloader):
    def __init__(
        self, source_list: List[str], target_list: Union[List[str], List[None]]
    ) -> None:
        super().__init__(source_list, target_list)
        self.source_splitter = lambda x: x.split()
        self.target_splitter = lambda x: x

    def set_source_splitter(self, function: Callable) -> None:
        # TODO, make is configurable
        self.splitter = function

    def preprocess_source(self, source: str) -> List:
        return self.source_splitter(source)

    def preprocess_target(self, target: str) -> List:
        return self.target_splitter(target)

    @classmethod
    def from_files(
        cls, source: Union[Path, str], target: Optional[Union[Path, str]]
    ) -> TextToTextDataloader:
        assert source
        with open(source) as f:
            source_list = f.readlines()
        if target:
            with open(target) as f:
                target_list = f.readlines()
        else:
            target_list = [None for _ in source_list]
        dataloader = cls(source_list, target_list)
        return dataloader

    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "text"
        args.target_type = "text"
        return cls.from_files(args.source, args.target)
