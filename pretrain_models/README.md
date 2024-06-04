# Download Pretrained Models

Download unit-based HiFi-GAN vocoder and HuBERT for unit extraction:

```shell
ROOT=/data/zhangshaolei/StreamSpeech # change to your local address
PRETRAIN_ROOT=$ROOT/pretrained_models

# unit-based_HiFi-GAN_vocoder
mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder
mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json

mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.es
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.es https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.es https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json

mkdir $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.fr
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.fr https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000
wget -P $PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.fr https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json

# HuBERT
mkdir $PRETRAIN_ROOT/mHuBERT
wget -P $PRETRAIN_ROOT/mHuBERT https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt
wget -P $PRETRAIN_ROOT/mHuBERT https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
```