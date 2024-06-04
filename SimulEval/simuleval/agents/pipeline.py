# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
from simuleval.data.segments import Segment
from .agent import GenericAgent, AgentStates


class AgentPipeline(GenericAgent):
    """A pipeline of agents

    Attributes:
        pipeline (list): a list of agent classes.

    """

    pipeline: List = []

    def __init__(self, module_list: List[GenericAgent]) -> None:
        self.module_list = module_list
        self.check_pipeline_types()

    def check_pipeline_types(self):
        if len(self.pipeline) > 1:
            for i in range(1, len(self.pipeline)):
                if (
                    self.module_list[i].source_type
                    != self.module_list[i - 1].target_type
                ):
                    raise RuntimeError(
                        f"{self.module_list[i]}.source_type({self.module_list[i].source_type}) != {self.pipeline[i-1]}.target_type({self.pipeline[i - 1].target_type}"  # noqa F401
                    )

    @property
    def source_type(self) -> Optional[str]:
        return self.module_list[0].source_type

    @property
    def target_type(self) -> Optional[str]:
        return self.module_list[-1].target_type

    def reset(self) -> None:
        for module in self.module_list:
            module.reset()

    def build_states(self) -> List[AgentStates]:
        return [module.build_states() for module in self.module_list]

    def push(
        self, segment: Segment, states: Optional[List[Optional[AgentStates]]] = None
    ) -> None:
        if states is None:
            states = [None for _ in self.module_list]
        else:
            assert len(states) == len(self.module_list)

        for index, module in enumerate(self.module_list[:-1]):
            segment = module.pushpop(segment, states[index])
        self.module_list[-1].push(segment, states[-1])

    def pop(self, states: Optional[List[Optional[AgentStates]]] = None) -> Segment:
        if states is None:
            last_states = None
        else:
            assert len(states) == len(self.module_list)
            last_states = states[-1]

        return self.module_list[-1].pop(last_states)

    @classmethod
    def add_args(cls, parser) -> None:
        for module_class in cls.pipeline:
            module_class.add_args(parser)

    @classmethod
    def from_args(cls, args):
        assert len(cls.pipeline) > 0
        return cls([module_class.from_args(args) for module_class in cls.pipeline])

    def __repr__(self) -> str:
        pipline_str = "\n\t".join(
            "\t".join(str(module).splitlines(True)) for module in self.module_list
        )
        return f"{self.__class__.__name__}(\n\t{pipline_str}\n)"

    def __str__(self) -> str:
        return self.__repr__()
