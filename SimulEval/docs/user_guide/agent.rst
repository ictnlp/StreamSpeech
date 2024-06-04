Agent
=====

To evaluate the simultaneous translation system,
the users need to implement agent class which operate the system logics.
This section will introduce how to implement an agent.

Source-Target Types
-------------------
First of all,
we must declare the source and target types of the agent class.
It can be done by inheriting from

- One of the following four built-in agent types

    - :class:`simuleval.agents.TextToTextAgent`
    - :class:`simuleval.agents.SpeechToTextAgent`
    - :class:`simuleval.agents.TextToSpeechAgent`
    - :class:`simuleval.agents.SpeechToSpeechAgent`

- Or :class:`simuleval.agents.GenericAgent`, with explicit declaration of :code:`source_type` and  :code:`target_type`.

The follow two examples are equivalent.

.. code-block:: python

    from simuleval import simuleval
    from simuleval.agents import GenericAgent

    class MySpeechToTextAgent(GenericAgent):
        source_type = "Speech"
        target_type = "Text"
        ....

.. code-block:: python

    from simuleval.agents import SpeechToSpeechAgent

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        ....

.. _agent_policy:

Policy
------

The agent must have a :code:`policy` method which must return one of two actions, :code:`ReadAction` and :code:`WriteAction`.
For example, an agent with a :code:`policy` method should look like this

.. code-block:: python

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        def policy(self):
            if do_we_need_more_input(self.states):
                return ReadAction()
            else:
                prediction = generate_a_token(self.states)
                finished = is_sentence_finished(self.states)
                return WriteAction(prediction, finished=finished)


..
    .. autoclass:: simuleval.agents.actions.WriteAction

..
    .. autoclass:: simuleval.agents.actions.ReadAction

States
------------
Each agent has the attribute the :code:`states` to keep track of the progress of decoding.
The :code:`states` attribute will be reset at the beginning of each sentence.
SimulEval provide an built-in states :class:`simuleval.agents.states.AgentStates`,
which has some basic attributes such source and target sequences.
The users can also define customized states with :code:`Agent.build_states` method:

.. code-block:: python

    from simuleval.agents.states import AgentStates
    from dataclasses import dataclass

    @dataclass
    class MyComplicatedStates(AgentStates)
        some_very_useful_variable: int

        def reset(self):
            super().reset()
            # also remember to reset the value
            some_very_useful_variable = 0

    class MySpeechToTextAgent(SpeechToSpeechAgent):
        def build_states(self):
            return MyComplicatedStates(0)

        def policy(self):
            some_very_useful_variable = self.states.some_very_useful_variable
            ...
            self.states.some_very_useful_variable = new_value
            ...

..
    .. autoclass:: simuleval.agents.states.AgentStates
        :members:


Pipeline
--------
The simultaneous system can consist several different components.
For instance, a simultaneous speech-to-text translation can have a streaming automatic speech recognition system and simultaneous text-to-text translation system.
SimulEval introduces the agent pipeline to support this function.
The following is a minimal example.
We concatenate two wait-k systems with different rates (:code:`k=2` and :code:`k=3`)
Note that if there are more than one agent class define,
the :code:`@entrypoint` decorator has to be used to determine the entry point

.. literalinclude:: ../../examples/quick_start/agent_pipeline.py
   :language: python
   :lines: 7-

Customized Arguments
-----------------------

It is often the case that we need to pass some customized arguments for the system to configure different settings.
The agent class has a built-in static method :code:`add_args` for this purpose.
The following is an updated version of the dummy agent from :ref:`first-agent`.

.. literalinclude:: ../../examples/quick_start/agent_with_configs.py
   :language: python
   :lines: 6-

Then just simply pass the arguments through command line as follow.

.. code-block:: bash

    simuleval \
        --source source.txt --source target.txt \ # data arguments
        --agent dummy_waitk_text_agent_v2.py \
        --waitk 3 --vocab data/dict.txt # agent arguments

Load Agents from Python Class
-----------------------------

If you have the agent class in the python environment, for instance 

.. literalinclude:: ../../examples/quick_start/agent_with_configs.py
   :language: python
   :lines: 6-

You can also start the evaluation with following command

.. code-block:: bash

    simuleval \
        --source source.txt --source target.txt \ # data arguments
        --agent-class DummyWaitkTextAgent \
        --waitk 3 --vocab data/dict.txt # agent arguments


Load Agents from Directory
--------------------------

Agent can also be loaded from a directory, which will be referred to as system directory.
The system directory should have everything required to start the agent. Again use the following agent as example

.. literalinclude:: ../../examples/quick_start/agent_with_configs.py
   :language: python
   :lines: 6-

and the system directory has 

.. code-block:: bash

    > ls ${system_dir}
    main.yaml dict.txt 

Where the `main.yaml` has all the command line options. The path will be the relative path to the `${system_dir}`.

.. code-block:: yaml

    waitk: 3
    vocab: dict.txt

The agent can then be started as following

.. code-block:: bash

    simuleval \
        --source source.txt --source target.txt \ # data arguments
        --system-dir ${system_dir}

By default, the `main.yaml` will be read. You can also have multiple YAML files in the system directory and pass them through command line arguments

.. code-block:: bash
     > ls ${system_dir}
    main.yaml dict.txt v1.yaml 

    > simuleval \
        --source source.txt --source target.txt \ # data arguments
        --system-dir ${system_dir} --system-config v1.yaml