
lang=$1
CVSS_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
ROOT=/data/zhangshaolei/StreamSpeech

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_ref_txt.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-txt $CVSS_ROOT/${lang}-en/fbank2unit/$split.txt
done

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_ref_unit.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-unit $CVSS_ROOT/${lang}-en/fbank2unit/$split.unit
done

for split in train dev test
do
    PYTHONPATH=$ROOT/fairseq python $ROOT/preprocess_scripts/extract_src_txt.py \
        --input-tsv $CVSS_ROOT/${lang}-en/fbank2unit/$split.tsv \
        --output-txt $CVSS_ROOT/${lang}-en/fbank2unit/$split.src
done