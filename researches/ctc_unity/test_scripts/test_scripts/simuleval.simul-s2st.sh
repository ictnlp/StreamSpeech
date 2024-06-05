export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech
DATA_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
PRETRAIN_ROOT=/data/zhangshaolei/pretrain_models
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

LANG=fr
file=/data/zhangshaolei/StreamSpeech_model/streamspeech.simultaneous.${LANG}-en.pt
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2st

chunk_size=960

PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${DATA_ROOT}/${LANG}-en/fbank2unit \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source ${DATA_ROOT}/${LANG}-en/simuleval/test/wav_list.txt --target ${DATA_ROOT}/${LANG}-en/simuleval/test/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_speech.streamspeech.agent.py \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction \
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics ASR_BLEU  --target-speech-lang en --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks DiscontinuitySum DiscontinuityAve DiscontinuityNum RTF \
    --device gpu --computation-aware 


# # To calculate ASR-BLEU w/o silence,
# # Another way: You can simply comment out Line 358 to Line 360 of StreamSpeech/SimulEval/simuleval/evaluator/instance.py to prevent silence from being added to the result within SimulEval.
#
# cd $ROOT/asr_bleu_rm_silence
# python compute_asr_bleu.py --reference_path ${DATA_ROOT}/${LANG}-en/simuleval/test/target.txt --lang en --audio_dirpath $output_dir/chunk_size=$chunk_size/wavs --reference_format txt --transcripts_path $output_dir/chunk_size=$chunk_size/rm_silence_asr_transcripts.txt --results_dirpath $output_dir/chunk_size=$chunk_size/rm_silence_asr_bleu
# cd $ROOT