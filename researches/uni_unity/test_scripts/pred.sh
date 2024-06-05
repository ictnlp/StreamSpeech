export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech
DATA_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
PRETRAIN_ROOT=/data/zhangshaolei/pretrain_models
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json

LANG=fr
DATA=${DATA_ROOT}/${LANG}-en/fbank2unit
SPLIT=test
BEAM=10

file=/data/zhangshaolei/StreamSpeech_model/unity.${LANG}-en.pt

mkdir res
output_dir=res/unity.${LANG}-en
mkdir -p $output_dir

PYTHONPATH=$ROOT/fairseq fairseq-generate ${DATA} \
    --user-dir researches/uni_unity \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_unity.yaml \
    --task speech_to_speech_modified --target-is-code --target-code-size 1000 --vocoder code_hifigan \
    --path $file --gen-subset $SPLIT \
    --beam-mt $BEAM --beam 1 --max-len-a 1 \
    --max-tokens 10000 \
    --required-batch-size-multiple 1 \
    --results-path $output_dir > $output_dir/generate-$SPLIT.log 2>&1

grep '^D-' $output_dir/generate-$SPLIT.log | sort -t'-' -k2,2n | cut -f2 > $output_dir/generate-$SPLIT.tgt

echo '################### Speech-to-text target text BLEU ###################' >> $output_dir/res.txt
sacrebleu $DATA/$SPLIT.txt -i $output_dir/generate-$SPLIT.tgt -w 3 >> $output_dir/res.txt

grep "^D\-" $output_dir/generate-$SPLIT.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
> $output_dir/generate-$SPLIT.unit

echo '################### Speech-to-unit target unit BLEU ###################' >> $output_dir/res.txt
sacrebleu $DATA/$SPLIT.unit -i $output_dir/generate-$SPLIT.unit -w 3 >> $output_dir/res.txt

python $ROOT/fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
    --in-code-file $output_dir/generate-$SPLIT.unit \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --results-path $output_dir/pred_wav --dur-prediction

cd $ROOT/asr_bleu
python compute_asr_bleu.py \
    --lang en \
    --audio_dirpath ../$output_dir/pred_wav \
    --reference_path $DATA/$SPLIT.txt \
    --reference_format txt > ../$output_dir/asr_bleu.log 2>&1

cd ..

echo '################### Speech-to-speech target speech ASR-BLEU ###################' >> $output_dir/res.txt
tail -n 1 $output_dir/asr_bleu.log >> $output_dir/res.txt

