# StreamSpeech Setup Instructions

This guide will help you set up StreamSpeech for simultaneous speech-to-speech translation on Windows.

## Prerequisites

- **Python 3.10** (required - other versions may not work)
- **CUDA-capable GPU** (recommended for optimal performance)
- **Windows 10/11** (tested on Windows 10.0.19045)
- **Git** (for cloning repositories)

## Quick Setup

### 1. Install Python 3.10

If you don't have Python 3.10, install it using Windows Package Manager:

```powershell
winget install Python.Python.3.10
```

Verify installation:
```powershell
py -3.10 --version
```

### 2. Clone and Setup Environment

```powershell
# Navigate to your desired directory
cd D:\StreamSpeech

# Create virtual environment with Python 3.10
py -3.10 -m venv streamspeech_env

# Activate virtual environment
streamspeech_env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies

#### Option A: Install from requirements.txt (Recommended)
```powershell
pip install -r requirements.txt
```

#### Option B: Manual installation
```powershell
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install fairseq
pip install fairseq

# Install SimulEval (editable mode)
cd SimulEval
pip install --editable ./
cd ..

# Install Flask for web demo
pip install flask
```

### 4. Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import fairseq; print('Fairseq: OK')"
python -c "import simuleval; print('SimulEval: OK')"
python -c "import flask; print('Flask: OK')"
```

## Model Setup

### 1. Download Pre-trained Models

**🚀 Fast Download Option (Recommended):**

Download all pre-trained models from Google Drive for faster speeds:

**[📁 Download Pre-trained Models from Google Drive](https://drive.google.com/drive/folders/1C4Y0sq_-tSRSbbu8dt0QGRQsk4h-9v5m?usp=drive_link)**

1. Click the link above to access the Google Drive folder
2. Download the entire `pretrain_models` folder
3. Extract it to your StreamSpeech root directory

The folder contains:
- **StreamSpeech Models**: All language pairs (French-English, Spanish-English, German-English)
  - `streamspeech.simultaneous.[lang]-en.pt` (simultaneous translation)
  - `streamspeech.offline.[lang]-en.pt` (offline translation)
- **HiFi-GAN Vocoder**: Complete unit-based vocoder with config
  - `unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000`
  - `unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json`

**Alternative Download (Original Sources):**

If you prefer to download from original sources:

```powershell
mkdir pretrain_models
cd pretrain_models
```

#### StreamSpeech Models (choose one language pair):

**French-English:**
- Simultaneous: [streamspeech.simultaneous.fr-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.fr-en.pt)
- Offline: [streamspeech.offline.fr-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.fr-en.pt)

**Spanish-English:**
- Simultaneous: [streamspeech.simultaneous.es-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.es-en.pt)
- Offline: [streamspeech.offline.es-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.es-en.pt)

**German-English:**
- Simultaneous: [streamspeech.simultaneous.de-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.simultaneous.de-en.pt)
- Offline: [streamspeech.offline.de-en.pt](https://huggingface.co/ICTNLP/StreamSpeech_Models/blob/main/streamspeech.offline.de-en.pt)

#### HiFi-GAN Vocoder:
- Model: [g_00500000](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000)
- Config: [config.json](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)

Create the vocoder directory structure:
```powershell
mkdir unit-based_HiFi-GAN_vocoder\mHuBERT.layer11.km1000.en
# Place g_00500000 and config.json in this directory
```

### 2. Configure Paths

#### Option A: Automatic Setup (Recommended)
```powershell
cd demo
python setup_paths.py
```

#### Option B: Manual Setup
1. Copy the template: `cp paths_config_template.json paths_config.json`
2. Edit `paths_config.json` with your actual paths:

```json
{
    "streamspeech_root": "D:/StreamSpeech",
    "pretrain_models_root": "D:/StreamSpeech/pretrain_models",
    "language_pair": "es-en",
    "models": {
        "simultaneous": "D:/StreamSpeech/pretrain_models/streamspeech.simultaneous.es-en.pt",
        "offline": "D:/StreamSpeech/pretrain_models/streamspeech.offline.es-en.pt"
    },
    "vocoder": {
        "checkpoint": "D:/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000",
        "config": "D:/StreamSpeech/pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json"
    },
    "configs": {
        "data_bin": "D:/StreamSpeech/configs/es-en",
        "user_dir": "D:/StreamSpeech/researches/ctc_unity",
        "agent_dir": "D:/StreamSpeech/agent"
    }
}
```

#### Update Language Config Files
Update config files in `configs/es-en/`:
- Replace `/data/zhangshaolei/StreamSpeech` with your actual StreamSpeech path in:
  - `config_gcmvn.yaml`
  - `config_mtl_asr_st_ctcst.yaml`

## Running the Application

### 1. Command Line Interface

```powershell
# Activate environment
streamspeech_env\Scripts\activate

# Set CUDA device
$env:CUDA_VISIBLE_DEVICES="0"

# Run inference
cd demo
python infer.py --data-bin ../configs/fr-en --user-dir ../researches/ctc_unity --agent-dir ../agent --model-path ../pretrain_models/streamspeech.simultaneous.fr-en.pt --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml --segment-size 320 --vocoder ../pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000 --vocoder-cfg ../pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json --dur-prediction
```

### 2. Web Demo

```powershell
# Activate environment
streamspeech_env\Scripts\activate

# Start web server
cd demo
python app.py
```

Open your browser to `http://localhost:7860`

## Features

- **Streaming ASR**: Real-time speech recognition
- **Simultaneous S2TT**: Speech-to-text translation
- **Simultaneous S2ST**: Speech-to-speech translation
- **Adjustable Latency**: 320ms to 5000ms
- **Real-time Results**: Live updates during playback

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model loading errors**: Check file paths in config.json
3. **Audio format issues**: Ensure audio is in supported format (WAV, MP3)
4. **Permission errors**: Run PowerShell as Administrator
5. **Python version issues**: Ensure Python 3.10 is used

### Performance Tips:

- **GPU Recommended**: Significant speedup with CUDA
- **Memory Requirements**: ~8GB+ GPU memory for optimal performance
- **Latency**: Lower values (320ms) = faster response, higher values = better quality

## Paths Configuration System

StreamSpeech uses a flexible paths configuration system that makes it easy to deploy across different environments:

### Files:
- **`demo/paths_config.json`**: Your actual paths (not in git)
- **`demo/paths_config_template.json`**: Template for paths (in git)
- **`demo/setup_paths.py`**: Automatic setup script
- **`demo/.gitignore`**: Excludes local paths from git

### Benefits:
- ✅ Easy to change paths without modifying code
- ✅ Git-friendly (local paths not committed)
- ✅ Environment-specific configurations
- ✅ Automatic path validation

## Directory Structure

```
StreamSpeech/
├── configs/
│   └── [lang]-en/          # Language-specific configs
├── pretrain_models/        # Downloaded models
│   └── unit-based_HiFi-GAN_vocoder/
├── demo/
│   ├── config.json         # Main configuration
│   ├── paths_config.json   # Your paths (auto-generated)
│   ├── paths_config_template.json # Template
│   ├── setup_paths.py      # Setup script
│   ├── app.py             # Flask web app
│   └── templates/
│       └── index.html     # Web interface
├── requirements.txt        # Dependencies
└── SETUP_INSTRUCTIONS.md  # This file
```

## Supported Languages

- French → English
- Spanish → English  
- German → English

## Citation

If you use StreamSpeech in your research, please cite:

```bibtex
@inproceedings{streamspeech,
    title={StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning}, 
    author={Shaolei Zhang and Qingkai Fang and Shoutao Guo and Zhengrui Ma and Min Zhang and Yang Feng},
    year={2024},
    booktitle = {Proceedings of the 62th Annual Meeting of the Association for Computational Linguistics (Long Papers)},
    publisher = {Association for Computational Linguistics}
}
```

## Links

- **Paper**: [arXiv:2406.03049](https://arxiv.org/abs/2406.03049)
- **Demo**: [StreamSpeech Demo](https://ictnlp.github.io/StreamSpeech-site/)
- **Models**: [Hugging Face](https://huggingface.co/ICTNLP/StreamSpeech_Models/tree/main)
- **GitHub**: [StreamSpeech Repository](https://github.com/ictnlp/StreamSpeech)
