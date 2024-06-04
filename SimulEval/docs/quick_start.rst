.. _first-agent:

Quick Start
===========

This section will introduce a minimal example on how to use SimulEval for simultaneous translation evaluation.
The code in the example can be found in :code:`examples/quick_start`.

The agent in SimulEval is core for simultaneous evaluation.
It's a carrier of user's simultaneous system.
The user has to implement the agent based on their system for evaluation.
The example simultaneous system is a dummy wait-k agent, which

- Runs `wait-k <https://aclanthology.org/P19-1289/>`_ policy.
- Generates random characters the policy decide to write.
- Stops the generation k predictions after source input. For simplicity, we just set :code:`k=3` in this example.

The implementation of this agent is shown as follow.

.. literalinclude:: ../examples/quick_start/first_agent.py
   :language: python
   :lines: 6-

There two essential components for an agent:

- :code:`states`:  The attribute keeps track of the source and target information.
- :code:`policy`:  The method makes decisions when the there is a new source segment.

Once the agent is implemented and saved at :code:`first_agent.py`,
run the following command for latency evaluation on:

.. code-block:: bash

    simuleval --source source.txt --reference target.txt --agent first_agent.py

where :code:`--source` is the input file while :code:`--target` is the reference file.

By default, the SimulEval will give the following output --- one quality and three latency metrics.

.. code-block:: bash

   2022-12-05 13:43:58 | INFO | simuleval.cli | Evaluate system: DummyWaitkTextAgent
   2022-12-05 13:43:58 | INFO | simuleval.dataloader | Evaluating from text to text.
   2022-12-05 13:43:58 | INFO | simuleval.sentence_level_evaluator | Results:
   BLEU  AL    AP  DAL
   1.541 3.0 0.688  3.0

The average lagging is expected since we are running an wait-3 system where the source and target always have the same length.
Notice that we have a very low yet random BLEU score. It's because we are randomly generate the output.
