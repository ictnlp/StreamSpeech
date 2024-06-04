# Quick Start
Following are some minimal examples to use SimulEval. More details can be found [here](https://simuleval.readthedocs.io/en/v1.1.0/quick_start.html).

## First Agent
To evaluate a text-to-text wait-3 system with random output:

```
> simuleval --source source.txt --target target.txt --agent first_agent.py

2022-12-05 13:43:58 | INFO | simuleval.cli | Evaluate system: DummyWaitkTextAgent
2022-12-05 13:43:58 | INFO | simuleval.dataloader | Evaluating from text to text.
2022-12-05 13:43:58 | INFO | simuleval.sentence_level_evaluator | Results:
BLEU  AL    AP  DAL
1.541 3.0 0.688  3.0

```

## Agent with Command Line Arguments
```
simuleval --source source.txt --target target.txt --agent agent_with_configs.py --waitk 3 --vocab dict.txt
```

## Agent Pipeline
```
simuleval --source source.txt --target target.txt --agent agent_pipeline.py
```

## Agent with New Metrics
```
simuleval --source source.txt --target target.txt --agent agent_with_new_metrics.py
```

## Standalone Agent & Remote Evaluation
Start an agent server:
```
simuleval --standalone --remote-port 8888 --agent agent_with_new_metrics.py
```
Or with docker
```
docker build -t simuleval_agent .
docker run -p 8888:8888 simuleval_agent:latest
```

Start a remote evaluator:
```
simuleval --remote-eval --source source.txt --target target.txt --source-type text --target-type text --remote-port 8888
```
