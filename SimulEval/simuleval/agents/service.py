# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import logging
from tornado import web, ioloop
from simuleval.data.segments import segment_from_json_string
from simuleval import options

logger = logging.getLogger("simuleval.agent_server")


class SystemHandler(web.RequestHandler):
    def initialize(self, system):
        self.system = system

    def get(self):
        self.write(json.dumps({"info": str(self.system)}))


class ResetHandle(SystemHandler):
    def post(self):
        self.system.reset()


class OutputHandler(SystemHandler):
    def get(self):
        output_segment = self.system.pop()
        self.write(output_segment.json())


class InputHandler(SystemHandler):
    def put(self):
        segment = segment_from_json_string(self.request.body)
        self.system.push(segment)


def start_agent_service(system):
    parser = options.general_parser()
    options.add_evaluator_args(parser)
    args, _ = parser.parse_known_args()
    app = web.Application(
        [
            (r"/reset", ResetHandle, {"system": system}),
            (r"/input", InputHandler, {"system": system}),
            (r"/output", OutputHandler, {"system": system}),
            (r"/", SystemHandler, {"system": system}),
        ],
        debug=False,
    )

    app.listen(args.remote_port, max_buffer_size=1024**3)

    logger.info(
        f"Simultaneous Translation Server Started (process id {os.getpid()}). Listening to port {args.remote_port} "
    )
    ioloop.IOLoop.current().start()
