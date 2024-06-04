# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from simuleval.data.segments import Segment, TextSegment, EmptySegment, SpeechSegment


class AgentStates:
    """
    Tracker of the decoding progress.

    Attributes:
        source (list): current source sequence.
        target (list): current target sequence.
        source_finished (bool): if the source is finished.
        target_finished (bool): if the target is finished.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset Agent states"""
        self.source = []
        self.target = []
        self.source_finished = False
        self.target_finished = False
        self.source_sample_rate = 0
        self.target_sample_rate = 0

    def update_source(self, segment: Segment):
        """
        Update states from input segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if isinstance(segment, EmptySegment):
            return
        elif isinstance(segment, TextSegment):
            self.source.append(segment.content)
        elif isinstance(segment, SpeechSegment):
            self.source += segment.content
            self.source_sample_rate = segment.sample_rate
        else:
            raise NotImplementedError

    def update_target(self, segment: Segment):
        """
        Update states from output segment

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.target_finished = segment.finished
        if not self.target_finished:
            if isinstance(segment, EmptySegment):
                return
            elif isinstance(segment, TextSegment):
                self.target.append(segment.content)
            elif isinstance(segment, SpeechSegment):
                self.target += segment.content
                self.target_sample_rate = segment.sample_rate
            else:
                raise NotImplementedError
