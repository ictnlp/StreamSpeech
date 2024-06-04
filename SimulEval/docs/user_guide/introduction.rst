Introduction
============
Different from offline translation system, the evaluation of simultaneous translation requires incremental decoding with an streaming input.
The simultaneous introduce the a front-end / back-end setup, shown as follow.

The back-end contains one or multiple user-defined agents which make decisions of whether to generate prediction at a certain point.
The agent can also considered as queue, where the input are keep pushed in and policy decides the timing to pop the output.

The front-end on the other side, represent the source of input and recipient of the system prediction.
In deployment, the front-end can be web page or cell phone app.
In SimulEval, the front-end is the evaluator , which feeds streaming input to back-end, receive prediction and track the delays.
The front-end and back-end can run separately for different purpose.

The evaluation process can summarized as follow pseudocode

.. code-block:: python

    for instance in evaluator.instances:
        while not instance.finished:
            input_segment = instance.send_source()
            prediction = agent.pushpop(input_segment)
            if prediction is not None:
                instance.receive_prediction(prediction)

    results = [scorer.score() for scorer in evaluate.scorers]



The common usage of SimulEval is as follow

.. code-block:: bash

    simuleval DATALOADER_OPTIONS EVALUATOR_OPTIONS --agent $AGENT_FILE AGENT_OPTIONS

We will introduce the usage of the toolkit based on these three major components: Agent, Dataloader and Evaluator.