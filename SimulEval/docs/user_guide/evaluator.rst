Evaluator
=========

The evaluation in SimulEval implemented as the Evaluator shown below.
It runs on sentence level, and will score the translation on quality and latency.
The user can use :code:`--quality-metrics` and :code:`--latency-metrics` to choose the metrics.
The final results along with the logs will be saved at :code:`--output` if given.

.. autoclass:: simuleval.evaluator.evaluator.SentenceLevelEvaluator

Quality Scorers
---------------

.. autoclass:: simuleval.evaluator.scorers.quality_scorer.SacreBLEUScorer
.. autoclass:: simuleval.evaluator.scorers.quality_scorer.ASRSacreBLEUScorer

Latency Scorers
---------------

.. autoclass:: simuleval.evaluator.scorers.latency_scorer.ALScorer
   :members:

.. autoclass:: simuleval.evaluator.scorers.latency_scorer.APScorer
   :members:

.. autoclass:: simuleval.evaluator.scorers.latency_scorer.DALScorer
   :members:

Customized Scorers
------------------
To add customized scorers, the user can use :code:`@register_latency_scorer` or :code:`@register_quality_scorer` to decorate a scorer class.
and use :code:`--quality-metrics` and :code:`--latency-metrics` to call the scorer. For example:

.. literalinclude:: ../../examples/quick_start/agent_with_new_metrics.py
   :lines: 6-

.. code-block:: bash

    > simuleval --source source.txt --target target.txt --agent agent_with_new_metrics.py --latency-metrics RTF
    2022-12-06 12:56:01 | INFO | simuleval.cli | Evaluate system: DummyWaitkTextAgent
    2022-12-06 12:56:01 | INFO | simuleval.dataloader | Evaluating from text to text.
    2022-12-06 12:56:01 | INFO | simuleval.sentence_level_evaluator | Results:
    BLEU   RTF
    1.593 1.078
