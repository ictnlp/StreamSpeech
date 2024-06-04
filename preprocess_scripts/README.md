# Preprocess of CVSS-C Data

These scripts process the CVSS-C data from scratch for training and testing StreamSpeech.

- [Step1: download CoVoST 2 & CVSS-C data](#step1-download-covost-2--cvss-c-data)
- [Step2: download pretrained models](#step2-download-pretrained-models)
- [Step3: process training and testing data](step3-process-training-and-testing-data)
- [Final Version](#final-version)

### Step1: download CoVoST 2 & CVSS-C data

- Download CoVoST 2 & CVSS-C data from here:

  - [CoVoST: A Large-Scale Multilingual Speech-To-Text Translation Corpus](https://github.com/facebookresearch/covost)

  - [CVSS: A Massively Multilingual Speech-to-Speech Translation Corpus](https://github.com/google-research-datasets/cvss)

- The directory structure of data is as follows. Replace `/data/zhangshaolei/datasets` with your local address `XXX`.

```
/data/zhangshaolei/datasets
├── covost2/
│   ├── fr/
│   │   ├── clips/
│   │   ├── covost_v2.fr_en.tsv
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   ├── other.tsv
│   │   ├── validated.tsv
│   │   ├── invalidated.tsv
│   │   └── covost_v2.fr_en.tsv.tar.gz
│   ├── de/
│   │   ├── clips/
│   │   ├── covost_v2.de_en.tsv
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   ├── other.tsv
│   │   ├── validated.tsv
│   │   ├── invalidated.tsv
│   │   └── covost_v2.de_en.tsv.tar.gz
│   ├── es/
│   │   ├── clips/
│   │   ├── covost_v2.es_en.tsv
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   ├── other.tsv
│   │   ├── validated.tsv
│   │   ├── invalidated.tsv
│   │   └── covost_v2.es_en.tsv.tar.gz
│   └── ...
└── cvss/
│   ├── cvss-c/
│   │   ├── fr-en/
│   │   │   ├── train/
│   │   │   ├── dev/
│   │   │   ├── test/
│   │   │   ├── train.tsv
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── cvss_c_fr_en_v1.0.tar.gz
│   │   ├── de-en
│   │   │   ├── train/
│   │   │   ├── dev/
│   │   │   ├── test/
│   │   │   ├── train.tsv
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── cvss_c_de_en_v1.0.tar.gz
│   │   ├── es-en
│   │   │   ├── train/
│   │   │   ├── dev/
│   │   │   ├── test/
│   │   │   ├── train.tsv
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   └── cvss_c_es_en_v1.0.tar.gz
│   │   ├── ...
└──...
```



### Step2: download pretrained models

- Replace `ROOT` in [0.download_pretrain_models.sh](./0.download_pretrain_models.sh) with your local address of the StreamSpeech repo, run:

```shell
bash 0.download_pretrain_models.sh
```

### Step3: process training and testing data

- Replace `ROOT`,`DATA_ROOT` in `1.XXX.sh`, `2.XXX.sh`, ... and `9.XXX.sh`  with your local address `XXX`, run:

```shell
bash preprocess.sh
```

- Modify the absolute path in the config files `./configs/fr-en/config_gcmvn.yaml` and `./configs/fr-en/config_mtl_asr_st_ctcst.yaml` to your local address `XXX`, then put them into  `cvss-c/fr-en/fbank2unit`.

  `cvss-c/fr-en/fbank2unit/config_gcmvn.yaml` should be like:

  ```yaml
  global_cmvn:
    stats_npz_path: /XXX/cvss/cvss-c/fr-en/gcmvn.npz 
  input_channels: 1
  input_feat_per_channel: 80
  specaugment:
    freq_mask_F: 27
    freq_mask_N: 1
    time_mask_N: 1
    time_mask_T: 100
    time_mask_p: 1.0
    time_wrap_W: 0
  transforms:
    '*':
    - global_cmvn
    _train:
    - global_cmvn
    - specaugment
  vocoder:
    checkpoint: /XXX/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000
    config: /XXX/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json
    type: code_hifigan
  ```

  `cvss-c/fr-en/fbank2unit/config_mtl_asr_st_ctcst.yaml` should be like:

  ```yaml
  target_unigram:
     decoder_type: transformer
     dict: /XXX/cvss/cvss-c/fr-en/tgt_unigram6000/spm_unigram_fr.txt
     data: /XXX/cvss/cvss-c/fr-en/tgt_unigram6000
     loss_weight: 8.0
     rdrop_alpha: 0.0
     decoder_args:
        decoder_layers: 4
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
     label_smoothing: 0.1
  source_unigram:
     decoder_type: ctc
     dict: /XXX/cvss/cvss-c/fr-en/src_unigram6000/spm_unigram_fr.txt
     data: /XXX/cvss/cvss-c/fr-en/src_unigram6000
     loss_weight: 4.0
     rdrop_alpha: 0.0
     decoder_args:
        decoder_layers: 0
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
     label_smoothing: 0.1
  ctc_target_unigram:
     decoder_type: ctc
     dict: /XXX/cvss/cvss-c/fr-en/tgt_unigram6000/spm_unigram_fr.txt
     data: /XXX/cvss/cvss-c/fr-en/tgt_unigram6000
     loss_weight: 4.0
     rdrop_alpha: 0.0
     decoder_args:
        decoder_layers: 0
        decoder_embed_dim: 512
        decoder_ffn_embed_dim: 2048
        decoder_attention_heads: 8
     label_smoothing: 0.1
  ```

### Final Version

- The directory structure of CVSS-C should be:

```
/data/zhangshaolei/datasets
├── covost2/
│   └── ... # no change
└── cvss/
│   ├── cvss-c/
│   │   ├── fr-en/
│   │   │   ├── train/
│   │   │   ├── dev/
│   │   │   ├── test/
│   │   │   ├── train.tsv
│   │   │   ├── dev.tsv
│   │   │   ├── test.tsv
│   │   │   ├── cvss_c_fr_en_v1.0.tar.gz
│   │   │   ├── fbank2unit/ # for StreamSpeech training
│   │   │   │   ├── config_gcmvn.yaml
│   │   │   │   ├── config_mtl_asr_st_ctcst.yaml
│   │   │   │   ├── train.tsv
│   │   │   │   ├── dev.tsv
│   │   │   │   ├── test.tsv
│   │   │   │   ├── train.src
│   │   │   │   ├── dev.src
│   │   │   │   ├── test.src
│   │   │   │   ├── train.txt
│   │   │   │   ├── dev.txt
│   │   │   │   ├── test.txt
│   │   │   │   ├── train.unit
│   │   │   │   ├── dev.unit
│   │   │   │   └── test.unit
│   │   │   ├── fbank2text/ # for S2TT training, no involved in StreamSpeech
│   │   │   │   ├── config_gcmvn.yaml
│   │   │   │   ├── train.tsv
│   │   │   │   ├── dev.tsv
│   │   │   │   ├── test.tsv
│   │   │   │   ├── spm_unigram_fr.model
│   │   │   │   ├── spm_unigram_fr.txt
│   │   │   │   └── spm_unigram_fr.vocab
│   │   │   ├── src_unigram6000/ # for multitask learning of StreamSpeech
│   │   │   │   ├── train.tsv
│   │   │   │   ├── dev.tsv
│   │   │   │   ├── test.tsv
│   │   │   │   ├── spm_unigram_fr.model
│   │   │   │   ├── spm_unigram_fr.txt
│   │   │   │   └── spm_unigram_fr.vocab
│   │   │   ├── tgt_unigram6000/ # for multitask learning of StreamSpeech
│   │   │   │   ├── train.tsv
│   │   │   │   ├── dev.tsv
│   │   │   │   ├── test.tsv
│   │   │   │   ├── spm_unigram_fr.model
│   │   │   │   ├── spm_unigram_fr.txt
│   │   │   │   └── spm_unigram_fr.vocab
│   │   │   ├── simuleval/ # for simuleval of StreamSpeech
│   │   │   │   ├── train/
│   │   │   │   │   ├── wav_list.txt
│   │   │   │   │   └── target.txt
│   │   │   │   ├── dev/
│   │   │   │   │   ├── wav_list.txt
│   │   │   │   │   └── target.txt
│   │   │   │   ├── test/
│   │   │   │   │   ├── wav_list.txt
│   │   │   │   │   └── target.txt
│   │   ├── de-en
│   │   │   └── ...
│   │   ├── es-en
│   │   │   └── ...
└──...
```

- The training tsv file `/XXX/cvss/cvss-c/fr-en/fbank2unit/train.tsv` (as well as dev and test) should be like:

```tsv
id	src_audio	src_n_frames	src_text	tgt_text	tgt_audio	tgt_n_frames
common_voice_fr_17732749	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:17614448698:126208	394	Madame la baronne Pfeffers.	madam pfeffers the baroness	63 991 162 73 338 359 761 430 901 921 549 413 366 896 627 915 143 390 479 330 776 576 384 879 70 958 66 776 663 198 711 124 884 393 946 734 870 290 978 484 249 466 663 179 961 931 428 377 32 835 67 297 265 675 755 237 193 415 772	59
common_voice_fr_17732750	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:18841732828:226048	706	Vous savez aussi bien que moi que de nombreuses molécules innovantes ont malheureusement déçu.	you know as well as i do that many new molecules have unfortunately been disappointing	63 644 553 258 436 139 340 575 116 281 62 783 803 791 563 52 483 366 873 641 124 337 243 935 101 741 803 693 521 453 366 641 124 362 530 733 664 196 721 250 549 139 340 846 726 603 857 662 459 173 945 29 609 710 892 73 889 172 871 877 384 120 179 207 950 974 86 116 372 139 340 498 324 338 359 655 764 259 453 366 998 319 501 445 137 74 205 521 711 510 337 152 784 429 558 167 650 816 915 143 38 479 330 435 592 103 934 477 59 179 961 931 428 366 901 29 518 56 321 948 86 290 943 488 620 352 915 721 250 333 432 882 924 586 362 734 870 251 0 41 740 908 211 81 664 274 398 53 455 309 584 415	152
common_voice_fr_17732751	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:16402047244:129408	404	Oh ! parce que maintenant, quand on parle de boire, je m’en vais !	oh because now when we talk about drinking i leave	63 644 254 27 908 52 611 916 726 603 987 752 63 662 260 978 56 165 319 263 501 137 576 167 104 713 873 711 124 510 337 243 116 281 62 384 907 597 611 916 726 902 224 286 111 277 300 63 644 991 535 935 101 741 366 620 352 915 271 930 105 244 583 167 246 764 246 268 501 137 366 849 907 597 29 542 777 728 647 503 488 212 325 409 501 398 212 455 385 942 115 224 121 6 334 226 666 985 254 27 530 733 259 781 303 485 321 948 885 148 417 755 603 752 544	115
common_voice_fr_17732752	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:24324038302:176128	550	Les questions sanitaires placent l’enfant au cœur de la problématique de l’évolution humaine.	the sanitary issues put the child at the heart of the human evolution question	63 665 991 393 946 734 432 742 519 26 204 280 668 384 879 961 931 428 366 523 793 794 75 583 15 576 663 466 56 406 25 771 333 432 882 431 531 976 534 139 340 198 711 510 243 850 561 213 41 740 677 355 660 555 29 202 393 946 734 793 105 326 531 668 167 655 837 81 693 521 555 944 565 173 945 29 393 946 734 470 821 655 764 969 555 208 944 932 148 202 393 946 734 470 821 258 436 139 340 748 872 336 366 620 352 112 659 538 423 565 879 577 692 154 302 259 854 340 817 146 283 352 143 38 272 119 607 167 384 879 70 170 731 600 477 283 377 385 584 16 584 819 415	143
common_voice_fr_17732753	/XXX/cvss/cvss-c/fr-en/src_fbank80.zip:16893259334:180608	564	J’ai au moins une satisfaction personnelle : j’ai ému Monsieur Piron.	i ve got at least one personal satisfaction i managed to got mister piron emotional	63 254 27 530 655 764 530 733 630 458 942 410 115 286 111 666 63 665 689 202 881 331 822 89 664 319 416 836 167 462 104 430 945 944 503 173 437 945 29 781 303 485 948 835 940 118 233 535 935 101 741 246 650 816 112 654 343 143 38 412 260 241 647 663 969 346 540 295 59 353 716 205 521 98 519 26 204 280 668 167 761 430 70 185 944 59 432 170 683 589 337 66 776 6 761 430 70 219 727 146 283 385 309 584 224 121 704 135 499 704 719 499 616 666 985 505 99 254 504 530 733 498 889 172 338 877 384 879 179 961 428 333 873 705 431 884 79 868 220 998 263 416 836 167 462 104 246 945 29 73 889 172 871 6 432 882 266 59 998 357 676 0 260 323 534 74 466 663 803 611 620 385 654 659 25 423 565 754 498 324 789 908 380 346 817 146 283 353 716 611 916 309 584 902 415 772	196
...
```

- The test files for SimulEval should be like:
  - `/XXX/cvss/cvss-c/fr-en/simuleval/test/wav_list.txt`:
  ```txt
  /XXX/covost2/fr/clips/common_voice_fr_17299399.mp3
  /XXX/covost2/fr/clips/common_voice_fr_17299400.mp3
  /XXX/covost2/fr/clips/common_voice_fr_17299401.mp3
  /XXX/covost2/fr/clips/common_voice_fr_17300796.mp3
  /XXX/covost2/fr/clips/common_voice_fr_17300797.mp3
  ...
  ```
  
  - `/XXX/cvss/cvss-c/fr-en/simuleval/test/target.txt`:
  
  ```txt
  really interesting work will finally be undertaken on that topic
  a profound reform is necessary
  not that many
  an inter ministerial committee on disability was held a few weeks back
  i shall give the floor to mr alain ramadier to support the amendment number one hundred twenty eight
  ...
  ```
  
  - `/XXX/cvss/cvss-c/fr-en/simuleval/test/src.txt`:
  
  ```txt
  un vrai travail intéressant va enfin être mené sur ce sujet
  une réforme profonde est nécessaire
  pas si nombreuses que ça
  un comité interministériel du handicap sest tenu il y a quelques semaines
  la parole est à monsieur alain ramadier pour soutenir lamendement numéro cent vingthuit
  ...
  ```
  
  
