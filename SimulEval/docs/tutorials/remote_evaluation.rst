Remote Evaluation
=================

Stand Alone Agent
-----------------
The agent can run in stand alone mode,
by using :code:`--standalone` option.
The SimulEval will kickoff a server that host the agent.
For instance, with the agent in :ref:`first-agent`,

.. code-block:: bash

    > simuleval --standalone --remote-port 8888 --agent first_agent.py.py
    2022-12-06 19:12:26 | INFO | simuleval.cli | Evaluate system: DummyWaitkTextAgent
    2022-12-06 19:12:26 | INFO | simuleval.agent_server | Simultaneous Translation Server Started (process id 53902). Listening to port 8888

For detailed RESTful APIs, please see (TODO)

Docker
-----------------
You can also use a docker image to run the simuleval.
An minimal example of :code:`Dockerfile` is

.. literalinclude:: ../../examples/quick_start/Dockerfile
   :language: docker

Build and run the docker image:

.. code-block:: bash

    cd examples/quick_start && docker build -t simuleval_agent .
    docker run -p 8888:8888 simuleval_agent:latest

Remote Evaluation
------------------
If there is an agent server or docker image available,
(let's say the one we just kickoff at localhost:8888)
We can start a remote evaluator as follow. For simplicity we assume they are on the same machine

.. code-block:: bash

    simuleval --remote-eval --remote-port 8888 \
        --source source.txt --target target.txt \
        --source-type text --target-type text
