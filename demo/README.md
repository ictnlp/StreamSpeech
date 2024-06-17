# GUI Demo for StreamSpeech

## One Speech Inference
```shell
export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech # path to StreamSpeech repo
PRETRAIN_ROOT=/data/zhangshaolei/pretrain_models 
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000 # path to downloaded Unit-based HiFi-GAN Vocoder
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json # path to downloaded Unit-based HiFi-GAN Vocoder

LANG=fr
file=/streamspeech.simultaneous.${LANG}-en.pt # path to downloaded StreamSpeech model
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2st

chunk_size=320 #ms
PYTHONPATH=$ROOT/fairseq python infer.py --data-bin ${ROOT}/configs/${LANG}-en \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --segment-size $chunk_size \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction 
```

## GUI Demo
1. Modify the config file [config.json](./config.json)
2. Run GUI Inference
```shell
python app.py
```
The Web UI will be displayed at http://0.0.0.0:7860/ and can be accessed using a browser.