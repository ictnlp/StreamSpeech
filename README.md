# StreamSpeech

[![arXiv](https://img.shields.io/badge/arXiv-2406.03049-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.03049)
[![project](https://img.shields.io/badge/%F0%9F%8E%A7%20Demo-Listen%20to%20StreamSpeech-orange.svg)](https://ictnlp.github.io/StreamSpeech-site/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20-StreamSpeech_Models-blue.svg)](https://huggingface.co/ICTNLP/StreamSpeech_Models/tree/main)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fictnlp%2FStreamSpeech&count_bg=%2379C83D&title_bg=%23555555&icon=awesomelists.svg&icon_color=%23E7E7E7&title=Visitors&edge_flat=false)](https://hits.seeyoufarm.com)

[![twitter](https://img.shields.io/badge/Twitter-@Gorden%20Sun-black?logo=X&logoColor=black)](https://x.com/Gorden_Sun/status/1798742796524007845) [![twitter](https://img.shields.io/badge/Twitter-@imxiaohu-black?logo=X&logoColor=black)](https://x.com/imxiaohu/status/1798999363987124355)

> **Authors**: **[Shaolei Zhang](https://zhangshaolei1998.github.io/), [Qingkai Fang](https://fangqingkai.github.io/), [Shoutao Guo](https://scholar.google.com.hk/citations?user=XwHtPyAAAAAJ&hl), [Zhengrui Ma](https://scholar.google.com.hk/citations?user=dUgq6tEAAAAJ), [Min Zhang](https://scholar.google.com.hk/citations?user=CncXH-YAAAAJ), [Yang Feng*](https://people.ucas.edu.cn/~yangfeng?language=en)**


Code for ACL 2024 paper "[StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning](https://arxiv.org/pdf/2406.03049)".

<p align="center" width="100%">
<img src="./assets/streamspeech.png" alt="StreamSpeech" style="width: 70%; min-width: 300px; display: block; margin: auto;">
</p>
<p align="center">
  🎧 Listen to <a href="https://ictnlp.github.io/StreamSpeech-site/">StreamSpeech's translated speech</a> 🎧 
</p>

💡**Highlight**:
1. StreamSpeech achieves **SOTA performance** on both offline and simultaneous speech-to-speech translation.
2. StreamSpeech performs **streaming ASR**, **simultaneous speech-to-text translation** and **simultaneous speech-to-speech translation** via an "All in One" seamless model.
3. StreamSpeech can present intermediate results (i.e., ASR or translation results) during simultaneous translation, offering a more comprehensive low-latency communication experience.

## 🔥News
- [06.17] Add [Web GUI demo](./demo), now you can experience StreamSpeech in your local browser.
- [06.05] [Paper](https://arxiv.org/pdf/2406.03049), [code](https://github.com/ictnlp/StreamSpeech), [models](https://huggingface.co/ICTNLP/StreamSpeech_Models/tree/main) and [demo](https://ictnlp.github.io/StreamSpeech-site/) of StreamSpeech are available!

## ⭐Features

### Support 8 Tasks
- **Offline**: Speech Recognition (ASR)✅, Speech-to-Text Translation (S2TT)✅, Speech-to-Speech Translation (S2ST)✅, Speech Synthesis (TTS)✅
- **Simultaneous**: Streaming ASR✅, Simultaneous S2TT✅, Simultaneous S2ST✅, Real-time TTS✅ under any latency (with one model)

### GUI Demo

https://github.com/ictnlp/StreamSpeech/assets/34680227/4d9bdabf-af66-4320-ae7d-0f23e721cd71
<p align="center">
  Simultaneously provide ASR, translation, and synthesis results via a seamless model
</p>

### Case

> **Speech Input**: [example/wavs/common_voice_fr_17301936.mp3](./example/wavs/common_voice_fr_17301936.mp3)
>
> **Transcription** (ground truth): jai donc lexpérience des années passées jen dirai un mot tout à lheure
>
> **Translation** (ground truth): i therefore have the experience of the passed years i'll say a few words about that later

| StreamSpeech                                    | Simultaneous                                                 | Offline                                                      |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Speech Recognition**                          | jai donc expérience des années passé jen dirairai un mot tout à lheure | jai donc lexpérience des années passé jen dirairai un mot tout à lheure |
| **Speech-to-Text Translation**                  | i therefore have an experience of last years i will tell a word later | so i have the experience in the past years i'll say a word later |
| **Speech-to-Speech Translation**                | <video src='https://github.com/zhangshaolei1998/StreamSpeech_dev/assets/34680227/ed41ba13-353b-489b-acfa-85563d0cc2cb' width="30%"/>                          | <video src='https://github.com/zhangshaolei1998/StreamSpeech_dev/assets/34680227/ca482ba6-76da-4619-9dfd-24aa2eb3339a' width="30%"/>                          |
| **Text-to-Speech Synthesis** (*incrementally synthesize speech word by word*) | <video src='https://github.com/zhangshaolei1998/StreamSpeech_dev/assets/34680227/294f1310-eace-4914-be30-5cd798e8592e' width="30%"/>                          | <video src='https://github.com/zhangshaolei1998/StreamSpeech_dev/assets/34680227/52854163-7fc5-4622-a5a6-c133cbd99e58' width="30%"/>                          |



## ⚙Requirements

- Python == 3.10, PyTorch == 2.0.1, Install fairseq & SimulEval

  ```bash
  cd fairseq
  pip install --editable ./ --no-build-isolation
  cd SimulEval
  pip install --editable ./
  ```

## 🚀Quick Start

### 1. Model Download

#### (1) StreamSpeech Models

| Language | UnitY                                                        | StreamSpeech (offline)                                       | StreamSpeech (simultaneous)                                  |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fr-En    | unity.fr-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.fr-en.pt)] [[Baidu](https://pan.baidu.com/s/10uGYgl0xTej9FP43iKx7Cg?pwd=nkvu)] | streamspeech.offline.fr-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.fr-en.pt)] [[Baidu](https://pan.baidu.com/s/1GFckHGP5SNLuOEj6mbIWhQ?pwd=pwgq)] | streamspeech.simultaneous.fr-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.fr-en.pt)] [[Baidu](https://pan.baidu.com/s/1edCPFljogyDHgGXkUV8_3w?pwd=8gg3)] |
| Es-En    | unity.es-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.es-en.pt)] [[Baidu](https://pan.baidu.com/s/1RwIEHye8jjw3kiIgrCHA3A?pwd=hde4)] | streamspeech.offline.es-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.es-en.pt)] [[Baidu](https://pan.baidu.com/s/1T89G4NC4J0Ofzcsc8Rt2Ww?pwd=yuhd)] | streamspeech.simultaneous.es-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.es-en.pt)] [[Baidu](https://pan.baidu.com/s/1NbLEVcYWHIdqqLD17P1s9g?pwd=p1pc)] |
| De-En    | unity.de-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.de-en.pt)] [[Baidu](https://pan.baidu.com/s/1Mg_PBeZ5acEDhl5wRJ_-7w?pwd=egvv)] | streamspeech.offline.de-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.de-en.pt)] [[Baidu](https://pan.baidu.com/s/1mTE4eHuVLJPB7Yg9AackEg?pwd=6ga8)] | streamspeech.simultaneous.de-en.pt [[Huggingface](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.de-en.pt)] [[Baidu](https://pan.baidu.com/s/1DYPMg3mdDopLY70BYQTduQ?pwd=r7kw)] |

#### (2) Unit-based HiFi-GAN Vocoder

| Unit config       | Unit size | Vocoder language | Dataset                                             | Model                                                        |
| ----------------- | --------- | ---------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| mHuBERT, layer 11 | 1000      | En               | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json) |

### 2. Prepare Data and Config (only for test/inference)

#### (1) Config Files

Replace `/data/zhangshaolei/StreamSpeech` in files [configs/fr-en/config_gcmvn.yaml](./configs/fr-en/config_gcmvn.yaml) and [configs/fr-en/config_mtl_asr_st_ctcst.yaml](./configs/fr-en/config_mtl_asr_st_ctcst.yaml) with your local address of StreamSpeech repo.

#### (2) Test Data

Prepare test data following [SimulEval](https://github.com/facebookresearch/SimulEval) format. [example/](./example) provides an example:

- [wav_list.txt](./example/wav_list.txt): Each line records the path of a source speech.
- [target.txt](./example/target.txt): Each line records the reference text, e.g., target translation or source transcription (used to calculate the metrics).

### 3. Inference with SimulEval

Run these scripts to inference StreamSpeech on streaming ASR, simultaneous S2TT and  simultaneous S2ST.

> `--source-segment-size`: set the chunk size (millisecond) to any value to control the latency

<details>
<summary>Simultaneous Speech-to-Speech Translation</summary>

`--output-asr-translation`: whether to output the intermediate ASR and translated text results during simultaneous speech-to-speech translation.

```shell
export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech # path to StreamSpeech repo
PRETRAIN_ROOT=/data/zhangshaolei/pretrain_models 
VOCODER_CKPT=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000 # path to downloaded Unit-based HiFi-GAN Vocoder
VOCODER_CFG=$PRETRAIN_ROOT/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json # path to downloaded Unit-based HiFi-GAN Vocoder

LANG=fr
file=streamspeech.simultaneous.${LANG}-en.pt # path to downloaded StreamSpeech model
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2st

chunk_size=320 #ms
PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${ROOT}/configs/${LANG}-en \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source example/wav_list.txt --target example/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_speech.streamspeech.agent.py \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG --dur-prediction \
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics ASR_BLEU  --target-speech-lang en --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks DiscontinuitySum DiscontinuityAve DiscontinuityNum RTF \
    --device gpu --computation-aware \
    --output-asr-translation True
```

You should get the following outputs:

```
fairseq plugins loaded...
fairseq plugins loaded...
fairseq plugins loaded...
fairseq plugins loaded...
2024-06-06 09:45:46 | INFO     | fairseq.tasks.speech_to_speech | dictionary size: 1,004
import agents...
Removing weight norm...
2024-06-06 09:45:50 | INFO     | agent.tts.vocoder | loaded CodeHiFiGAN checkpoint from /data/zhangshaolei/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
2024-06-06 09:45:50 | INFO     | simuleval.utils.agent | System will run on device: gpu.
2024-06-06 09:45:50 | INFO     | simuleval.dataloader | Evaluating from speech to speech.
  0%|                                                                                                                                                                              | 0/2 [00:00<?, ?it/s]
Streaming ASR: 
Streaming ASR: 
Streaming ASR: je
Simultaneous translation: i would
Streaming ASR: je voudrais
Simultaneous translation: i would like to
Streaming ASR: je voudrais soumettre
Simultaneous translation: i would like to sub
Streaming ASR: je voudrais soumettre cette
Simultaneous translation: i would like to submit
Streaming ASR: je voudrais soumettre cette idée
Simultaneous translation: i would like to submit this
Streaming ASR: je voudrais soumettre cette idée à la
Simultaneous translation: i would like to submit this idea to
Streaming ASR: je voudrais soumettre cette idée à la réflexion
Simultaneous translation: i would like to submit this idea to the
Streaming ASR: je voudrais soumettre cette idée à la réflexion de
Simultaneous translation: i would like to submit this idea to the reflection
Streaming ASR: je voudrais soumettre cette idée à la réflexion de lassemblée
Simultaneous translation: i would like to submit this idea to the reflection of
Streaming ASR: je voudrais soumettre cette idée à la réflexion de lassemblée nationale
Simultaneous translation: i would like to submit this idea to the reflection of the
Streaming ASR: je voudrais soumettre cette idée à la réflexion de lassemblée nationale
Simultaneous translation: i would like to submit this idea to the reflection of the national assembly
 50%|███████████████████████████████████████████████████████████████████████████████████                                                                                   | 1/2 [00:04<00:04,  4.08s/it]
Streaming ASR: 
Streaming ASR: 
Streaming ASR: 
Streaming ASR: 
Streaming ASR: jai donc
Simultaneous translation: i therefore
Streaming ASR: jai donc
Streaming ASR: jai donc expérience des
Simultaneous translation: i therefore have an experience
Streaming ASR: jai donc expérience des années
Streaming ASR: jai donc expérience des années passé
Simultaneous translation: i therefore have an experience of last
Streaming ASR: jai donc expérience des années passé jen
Simultaneous translation: i therefore have an experience of last years
Streaming ASR: jai donc expérience des années passé jen dirairai
Simultaneous translation: i therefore have an experience of last years i will
Streaming ASR: jai donc expérience des années passé jen dirairai un mot
Simultaneous translation: i therefore have an experience of last years i will tell a
Streaming ASR: jai donc expérience des années passé jen dirairai un mot tout à lheure
Simultaneous translation: i therefore have an experience of last years i will tell a word
Streaming ASR: jai donc expérience des années passé jen dirairai un mot tout à lheure
Simultaneous translation: i therefore have an experience of last years i will tell a word later
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:06<00:00,  3.02s/it]
2024-06-06 09:45:56 | WARNING  | simuleval.scorer.asr_bleu | Beta feature: Evaluating speech output. Faieseq is required.
2024-06-06 09:46:12 | INFO | fairseq.tasks.audio_finetuning | Using dict_path : /data/zhangshaolei/.cache/ust_asr/en/dict.ltr.txt
Transcribing predictions: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.63it/s]
2024-06-06 09:46:21 | INFO     | simuleval.sentence_level_evaluator | Results:
 ASR_BLEU       AL    AL_CA    AP  AP_CA      DAL  DAL_CA  StartOffset  StartOffset_CA  EndOffset  EndOffset_CA     LAAL  LAAL_CA      ATD   ATD_CA  NumChunks  NumChunks_CA  DiscontinuitySum  DiscontinuitySum_CA  DiscontinuityAve  DiscontinuityAve_CA  DiscontinuityNum  DiscontinuityNum_CA   RTF  RTF_CA
   15.448 1724.895 2913.508 0.425  0.776 1358.812 3137.55       1280.0        2213.906     1366.0        1366.0 1724.895 2913.508 1440.146 3389.374        9.5           9.5             110.0                110.0              55.0                 55.0                 1                    1 1.326   1.326

```

Logs and evaluation results are stored in ` $output_dir/chunk_size=$chunk_size`:

```
$output_dir/chunk_size=$chunk_size
├── wavs/
│   ├── 0_pred.wav # generated speech
│   ├── 1_pred.wav 
│   ├── 0_pred.txt # asr transcription for ASR-BLEU tookit
│   ├── 1_pred.txt 
├── config.yaml
├── asr_transcripts.txt # ASR-BLEU transcription results
├── metrics.tsv
├── scores.tsv
├── asr_cmd.bash
└── instances.log # logs of Simul-S2ST
```

</details>

<details>
<summary>Simultaneous Speech-to-Text Translation</summary>

```shell
export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech # path to StreamSpeech repo

LANG=fr
file=streamspeech.simultaneous.${LANG}-en.pt # path to downloaded StreamSpeech model
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/simul-s2tt

chunk_size=320 #ms
PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${ROOT}/configs/${LANG}-en \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source example/wav_list.txt --target example/target.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_text.s2tt.streamspeech.agent.py\
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics BLEU  --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu --computation-aware 
```
</details>

<details>
<summary>Streaming ASR</summary>

```shell
export CUDA_VISIBLE_DEVICES=0

ROOT=/data/zhangshaolei/StreamSpeech # path to StreamSpeech repo

LANG=fr
file=streamspeech.simultaneous.${LANG}-en.pt # path to downloaded StreamSpeech model
output_dir=$ROOT/res/streamspeech.simultaneous.${LANG}-en/streaming-asr

chunk_size=320 #ms
PYTHONPATH=$ROOT/fairseq simuleval --data-bin ${ROOT}/configs/${LANG}-en \
    --user-dir ${ROOT}/researches/ctc_unity --agent-dir ${ROOT}/agent \
    --source example/wav_list.txt --target example/source.txt \
    --model-path $file \
    --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
    --agent $ROOT/agent/speech_to_text.asr.streamspeech.agent.py\
    --output $output_dir/chunk_size=$chunk_size \
    --source-segment-size $chunk_size \
    --quality-metrics BLEU  --latency-metrics AL AP DAL StartOffset EndOffset LAAL ATD NumChunks RTF \
    --device gpu --computation-aware 
```
</details>

## 🎈Develop Your Own StreamSpeech

### 1. Data Preprocess

- Follow [`./preprocess_scripts`](./preprocess_scripts) to process CVSS-C data. 

### 2. Training

> [!Note]
> You can directly use the [downloaded StreamSpeech model](#1-model-download) for evaluation and skip training.

<p align="center" width="100%">
<img src="./assets/model.png" alt="model" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

- Follow [`researches/ctc_unity/train_scripts/train.simul-s2st.sh`](./researches/ctc_unity/train_scripts/train.simul-s2st.sh) to train StreamSpeech for simultaneous speech-to-speech translation.
- Follow [`researches/ctc_unity/train_scripts/train.offline-s2st.sh`](./researches/ctc_unity/train_scripts/train.offline-s2st.sh) to train StreamSpeech for offline speech-to-speech translation.
- We also provide some other StreamSpeech variants and baseline implementations.

| Model             | --user-dir                 | --arch                            | Description                                                  |
| ----------------- | -------------------------- | --------------------------------- | ------------------------------------------------------------ |
| **Translatotron 2** | `researches/translatotron` | `s2spect2_conformer_modified`     | [Translatotron 2](https://proceedings.mlr.press/v162/jia22b.html) |
| **UnitY**         | `researches/translatotron` | `unity_conformer_modified`        | [UnitY](https://aclanthology.org/2023.acl-long.872/)         |
| **Uni-UnitY**     | `researches/uni_unity`     | `uni_unity_conformer`             | Change all encoders in UnitY into unidirectional             |
| **Chunk-UnitY**   | `researches/chunk_unity`   | `chunk_unity_conformer`           | Change the Conformer in UnitY into Chunk-based Conformer     |
| **StreamSpeech**  | `researches/ctc_unity`     | `streamspeech`                    | StreamSpeech                                                 |
| **StreamSpeech (cascade)** | `researches/ctc_unity` | `streamspeech_cascade` | Cascaded StreamSpeech of S2TT and TTS. TTS module can be used independently for real-time TTS given incremental text. |
| **HMT**           | `researches/hmt`           | `hmt_transformer_iwslt_de_en`     | [HMT](https://openreview.net/forum?id=9y0HFvaAYD6): strong simultaneous text-to-text translation method |
| **DiSeg**         | `researches/diseg`         | `convtransformer_espnet_base_seg` | [DiSeg](https://aclanthology.org/2023.findings-acl.485/): strong simultaneous speech-to-text translation method |

> [!Tip]
> The `train_scripts/` and `test_scripts/` in directory `--user-dir` give the training and testing scripts for each model.
> Refer to official repo of [UnitY](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/speech_to_speech/s2s_conformer_unity.py), [Translatotron 2](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/speech_to_speech/s2s_conformer_translatotron2.py), [HMT](https://github.com/ictnlp/HMT) and [DiSeg](https://github.com/ictnlp/DiSeg) for more details.

### 3. Evaluation

#### (1) Offline Evaluation

Follow [`pred.offline-s2st.sh`](./researches/ctc_unity/test_scripts/pred.offline-s2st.sh) to evaluate the offline performance of StreamSpeech on ASR, S2TT and S2ST.

#### (2) Simultaneous Evaluation

A trained StreamSpeech model can be used for streaming ASR, simultaneous speech-to-text translation and simultaneous speech-to-speech translation. We provide [agent/](./agent) for these three tasks:

- `agent/speech_to_speech.streamspeech.agent.py`: simultaneous speech-to-speech translation
- `agent/speech_to_text.s2tt.streamspeech.agent.py`: simultaneous speech-to-text translation
- `agent/speech_to_text.asr.streamspeech.agent.py`: streaming ASR

Follow [`simuleval.simul-s2st.sh`](./researches/ctc_unity/test_scripts/simuleval.simul-s2st.sh), [`simuleval.simul-s2tt.sh`](./researches/ctc_unity/test_scripts/simuleval.simul-s2tt.sh), [`simuleval.streaming-asr.sh`](./researches/ctc_unity/test_scripts/simuleval.streaming-asr.sh)  to evaluate StreamSpeech.

### 4. Our Results

Our project page ([https://ictnlp.github.io/StreamSpeech-site/](https://ictnlp.github.io/StreamSpeech-site/)) provides some translated speech generated by StreamSpeech, listen to it 🎧.

#### (1) Offline Speech-to-Speech Translation  ( ASR-BLEU: quality )

<p align="center" width="100%">
<img src="./assets/offline_results.png" alt="offline" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

#### (2) Simultaneous Speech-to-Speech Translation  ( AL: latency  |  ASR-BLEU: quality )

<p align="center" width="100%">
<img src="./assets/simultaneous_results.png" alt="simul" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

#### (3) Simultaneous Speech-to-Text Translation  ( AL: latency  |  BLEU: quality )

<p align="center" width="100%">
<img src="./assets/s2tt.png" alt="simul" style="width: 38%; min-width: 300px; display: block; margin: auto;">
</p>

#### (4) Streaming ASR  ( AL: latency  |  WER: quality )

<p align="center" width="100%">
<img src="./assets/asr.png" alt="simul" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

## 🖋Citation

If you have any questions, please feel free to submit an issue or contact `zhangshaolei20z@ict.ac.cn`.

If our work is useful for you, please cite as:

```
@inproceedings{streamspeech,
      title={StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning}, 
      author={Shaolei Zhang and Qingkai Fang and Shoutao Guo and Zhengrui Ma and Min Zhang and Yang Feng},
      year={2024},
      booktitle = {Proceedings of the 62th Annual Meeting of the Association for Computational Linguistics (Long Papers)},
      publisher = {Association for Computational Linguistics}
}
```
