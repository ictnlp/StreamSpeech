lang=$1
CVSS_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
ROOT=/data/zhangshaolei/StreamSpeech


PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/prep_cvss_c_multitask_data.py \
    --data-dir $CVSS_ROOT/${lang}-en/fbank2unit \
    --output-dir $CVSS_ROOT/${lang}-en/tgt_unigram6000 \
    --lang $lang \
    --vocab-type unigram --vocab-size 6000