# StreamSpeech Setup Complete! 🎉

## Virtual Environment Status
✅ **Virtual environment created**: `streamspeech_env`
✅ **All dependencies installed**
✅ **Fairseq configured** (via Python path)
✅ **SimulEval installed** (editable mode)

## Installed Packages
- **PyTorch 2.0.1** with CUDA 11.8 support
- **TorchVision & TorchAudio** (compatible versions)
- **Fairseq** (custom version from local directory)
- **SimulEval 1.1.0** (for evaluation)
- **Flask** (for web demo)
- **Audio processing**: soundfile, librosa, pydub
- **ML utilities**: numpy, pandas, scipy, scikit-learn
- **Configuration**: PyYAML, omegaconf, hydra-core
- **Other tools**: tensorboardX, sacrebleu, tqdm, and more

## CUDA Status
✅ **CUDA is available** on your system - GPU acceleration is ready!

---

## 📥 Required Models to Download

You need to download the following pre-trained models to use StreamSpeech:

### Option 1: Quick Download (Recommended)
**All models are available on Hugging Face:**
https://huggingface.co/ICTNLP/StreamSpeech_Models

### Option 2: Download Individual Models

#### 1️⃣ **StreamSpeech Models** (Choose your language pair)

**French → English:**
- **Simultaneous**: [streamspeech.simultaneous.fr-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.fr-en.pt) (~1.2 GB)
- **Offline**: [streamspeech.offline.fr-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.fr-en.pt) (~1.2 GB)
- **Unity baseline**: [unity.fr-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.fr-en.pt) (~1.2 GB)

**Spanish → English:**
- **Simultaneous**: [streamspeech.simultaneous.es-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.es-en.pt) (~1.2 GB)
- **Offline**: [streamspeech.offline.es-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.es-en.pt) (~1.2 GB)
- **Unity baseline**: [unity.es-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.es-en.pt) (~1.2 GB)

**German → English:**
- **Simultaneous**: [streamspeech.simultaneous.de-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.de-en.pt) (~1.2 GB)
- **Offline**: [streamspeech.offline.de-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.de-en.pt) (~1.2 GB)
- **Unity baseline**: [unity.de-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/unity.de-en.pt) (~1.2 GB)

#### 2️⃣ **Unit-based HiFi-GAN Vocoder** (Required for speech synthesis)

**For English output:**
- **Checkpoint**: [g_00500000](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000) (~55 MB)
- **Config**: [config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json) (~1 KB)

**For Spanish output (if needed):**
- **Checkpoint**: [g_00500000](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000)
- **Config**: [config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)

**For French output (if needed):**
- **Checkpoint**: [g_00500000](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000)
- **Config**: [config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)

#### 3️⃣ **mHuBERT Model** (For unit extraction)
- **Model**: [mhubert_base_vp_en_es_fr_it3.pt](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) (~316 MB)
- **K-means**: [mhubert_base_vp_en_es_fr_it3_L11_km1000.bin](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin) (~4 MB)

---

## 📁 Recommended Directory Structure

After downloading, organize your models like this:

```
D:\StreamSpeech\
├── pretrain_models\
│   ├── streamspeech.simultaneous.fr-en.pt
│   ├── streamspeech.offline.fr-en.pt
│   ├── unit-based_HiFi-GAN_vocoder\
│   │   ├── mHuBERT.layer11.km1000.en\
│   │   │   ├── g_00500000
│   │   │   └── config.json
│   │   ├── mHuBERT.layer11.km1000.es\
│   │   │   ├── g_00500000
│   │   │   └── config.json
│   │   └── mHuBERT.layer11.km1000.fr\
│   │       ├── g_00500000
│   │       └── config.json
│   └── mHuBERT\
│       ├── mhubert_base_vp_en_es_fr_it3.pt
│       └── mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
└── ... (other project files)
```

**Create the directories:**
```powershell
mkdir pretrain_models
mkdir pretrain_models\unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.en
mkdir pretrain_models\unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.es
mkdir pretrain_models\unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.fr
mkdir pretrain_models\mHuBERT
```

Then download the models into their respective directories.

---

## 🚀 Quick Start Guide

### 1. Activate the Environment
```powershell
.\streamspeech_env\Scripts\Activate.ps1
```

### 2. Test the Installation
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 3. Run Example Inference (after downloading models)

**Simultaneous Speech-to-Speech Translation:**
```powershell
$env:CUDA_VISIBLE_DEVICES="0"
$ROOT="D:\StreamSpeech"
$PRETRAIN_ROOT="D:\StreamSpeech\pretrain_models"
$LANG="fr"

$env:PYTHONPATH="$ROOT\fairseq"
simuleval --data-bin "$ROOT\configs\$LANG-en" `
    --user-dir "$ROOT\researches\ctc_unity" `
    --agent-dir "$ROOT\agent" `
    --source "$ROOT\example\wav_list.txt" `
    --target "$ROOT\example\target.txt" `
    --model-path "$PRETRAIN_ROOT\streamspeech.simultaneous.$LANG-en.pt" `
    --config-yaml config_gcmvn.yaml `
    --multitask-config-yaml config_mtl_asr_st_ctcst.yaml `
    --agent "$ROOT\agent\speech_to_speech.streamspeech.agent.py" `
    --vocoder "$PRETRAIN_ROOT\unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.en\g_00500000" `
    --vocoder-cfg "$PRETRAIN_ROOT\unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.en\config.json" `
    --dur-prediction `
    --source-segment-size 320 `
    --device gpu `
    --computation-aware `
    --output-asr-translation True
```

### 4. Run Web Demo (after downloading models)
```powershell
cd demo
python app.py
```
Then open your browser to `http://localhost:7860`

---

## 📋 Summary of What You Need

**For basic S2ST (French→English):**
1. ✅ Environment (already set up)
2. ⬇️ `streamspeech.simultaneous.fr-en.pt` (~1.2 GB)
3. ⬇️ HiFi-GAN vocoder for English (`g_00500000` + `config.json`) (~55 MB)
4. ⬇️ mHuBERT model (`.pt` file) (~316 MB)
5. ⬇️ mHuBERT k-means (`.bin` file) (~4 MB)

**Total download size: ~1.6 GB**

---

## 💡 Next Steps

1. **Download Models**: Start with French→English simultaneous model and English vocoder
2. **Update Config Files**: Edit paths in `configs/fr-en/config_gcmvn.yaml` and `config_mtl_asr_st_ctcst.yaml`
3. **Test with Examples**: Use the provided example audio files in `example/wavs/`
4. **Explore Features**: Try different tasks (ASR, S2TT, S2ST) with different latency settings

---

## 🔧 Troubleshooting

**Issue**: ImportError for fairseq
**Solution**: Make sure the virtual environment is activated. The `.pth` file automatically adds fairseq to the path.

**Issue**: CUDA out of memory
**Solution**: Use CPU mode by setting `--device cpu` or reduce batch size

**Issue**: Module not found
**Solution**: Ensure PYTHONPATH includes the fairseq directory:
```powershell
$env:PYTHONPATH="D:\StreamSpeech\fairseq"
```

---

## 📚 Resources

- **Paper**: https://arxiv.org/abs/2406.03049
- **Demo Site**: https://ictnlp.github.io/StreamSpeech-site/
- **Model Hub**: https://huggingface.co/ICTNLP/StreamSpeech_Models
- **GitHub**: https://github.com/ictnlp/StreamSpeech

---

**Environment created on**: November 6, 2025
**Python version**: 3.10
**PyTorch version**: 2.0.1 + CUDA 11.8
**GPU Support**: ✅ Enabled

