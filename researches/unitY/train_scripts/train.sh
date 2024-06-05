export CUDA_VISIBLE_DEVICES=0,1,2,3

LANG=fr
DATA_ROOT=/data/zhangshaolei/datasets/cvss/cvss-c
DATA=$DATA_ROOT/${LANG}-en/fbank2unit
model=unity

fairseq-train $DATA\
  --user-dir researches/translatotron \
  --config-yaml config_gcmvn.yaml --multitask-config-yaml config_unity.yaml \
  --task speech_to_speech --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit_2pass --label-smoothing 0.1 --rdrop-alpha 0.0 \
  --arch unity_conformer_modified --share-decoder-input-output-embed \
  --encoder-layers 12 --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4 \
  --translation-decoder-layers 4 --synthesizer-encoder-layers 2 \
  --decoder-layers 2  --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 8 \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir checkpoints/$model \
  --validate-interval 1000 --validate-interval-updates 1000 \
  --save-interval 1 --save-interval-updates 2000 \
  --keep-last-epochs 10 \
  --no-progress-bar --log-format json --log-interval 100 \
  --lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 1.0 \
  --max-tokens 40000 --max-target-positions 1200 --update-freq 1 \
  --attn-type espnet --pos-enc-type rel_pos \
  --keep-interval-updates 40 \
  --report-accuracy --keep-best-checkpoints 20 \
  --seed 1 --fp16 --num-workers 8 
