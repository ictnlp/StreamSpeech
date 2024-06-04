lang=$1
CVSS_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
ROOT=/data/zhangshaolei/StreamSpeech


for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_simuleval_unit.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --wav-list $CVSS_ROOT/${lang}-en/simuleval/$split/wav_list.txt \
        --output-unit $CVSS_ROOT/${lang}-en/simuleval/$split/unit.txt
done
