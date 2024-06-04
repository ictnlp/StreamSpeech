# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, List
from dataclasses import dataclass
from simuleval.data.segments import Segment


class Action:
    """
    Abstract Action class
    """

    def is_read(self) -> bool:
        """
        Whether the action is a read action

        Returns:
            bool: True if the action is a read action.
        """
        assert NotImplementedError


class ReadAction(Action):
    """
    Action to return when policy decide to read one more source segment.
    The only way to use it is to return :code:`ReadAction()`
    """

    def is_read(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "ReadAction()"


@dataclass
class WriteAction(Action):
    """
    Action to return when policy decide to generate a prediction

    Args:
        content (Union[str, List[float]]): The prediction.
        finished (bool): Indicates if current sentence is finished.

    .. note:: For text the prediction a str; for speech, it's a list.

    """

    content: Union[str, List[float], Segment]
    finished: bool

    def is_read(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"WriteAction({self.content}, finished={self.finished})"
