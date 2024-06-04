## Simultaneous Speech-to-Speech Translation

This tutorial provides a minimal example on how to evaluate a simultaneous speech-to-speech translation system.

### Requirements

To run this example, the following package is required

- [`whisper`](https://github.com/openai/whisper): for quality evaluation (`WHISPER_ASR_BLEU`).

### Agent

The speech-to-speech agent ([english_counter_agent.py](english_counter_agent.py)) in this example is a counter, which generates a piece of audio every second after an initial wait.
The policy of the agent is show follow. The agent will wait for `self.wait_seconds` seconds,
and generate the audio of `{length_in_seconds} mississippi` every second afterward.

```python
 def policy(self):
        length_in_seconds = round(
            len(self.states.source) / self.states.source_sample_rate
        )
        if not self.states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()
        print(length_in_seconds)
        samples, fs = self.tts_model.synthesize(f"{length_in_seconds} mississippi")

        # A SpeechSegment has to be returned for speech-to-speech translation system
        return WriteAction(
            SpeechSegment(
                content=samples,
                sample_rate=fs,
                finished=self.states.source_finished,
            ),
            finished=self.states.source_finished,
        )
```

Notice that for speech output agent, the `WriteAction` has to contain a `SpeechSegment` class.

### Evaluation

The following command will start an evaluation

```bash
simuleval \
    --agent english_counter_agent.py --output output \
    --source source.txt --target reference/en.txt --source-segment-size 1000\
    --quality-metrics WHISPER_ASR_BLEU \
    --target-speech-lang en --transcript-lowercase --transcript-non-punctuation\
    --latency-metrics StartOffset EndOffset ATD
```

For quality evaluation, we use ASR_BLEU, that is transcribing the speech output and compute BLEU score with the reference text. To use this feature, `whisper` has to be installed.

We use three metrics for latency evaluation

- `StartOffset`: The starting offset of translation comparing with source audio
- `EndOffset`: The ending offset of translation comparing with source audio
- `ATD`: Average Token Delay

The results of the evaluation should be as following. The transcripts and alignments can be found in the `output` directory.

```
 WHISPER_ASR_BLEU  StartOffset  EndOffset      ATD
            100.0       1000.0   1490.703 1248.261
```
