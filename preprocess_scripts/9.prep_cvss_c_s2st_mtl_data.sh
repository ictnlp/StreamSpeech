lang=$1
CVSS_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
ROOT=/data/zhangshaolei/StreamSpeech

PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/convert_s2st_tsv_to_s2tt_mtl_tsv.py \
    --s2st-tsv-dir $CVSS_ROOT/${lang}-en/fbank2unit \
    --s2tt-mtl-tsv-dir $CVSS_ROOT/${lang}-en/fbank2text_mtl \
    --src-lang $lang \
    --tgt-lang en \
    --vocab-size 10000 
