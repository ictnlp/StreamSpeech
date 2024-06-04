simuleval \
    --agent english_counter_agent.py --output output \
    --source source.txt --target reference/en.txt --source-segment-size 1000\
    --quality-metrics WHISPER_ASR_BLEU \
    --target-speech-lang en --transcript-lowercase --transcript-non-punctuation --whisper-model-size large \
    --latency-metrics StartOffset EndOffset ATD
