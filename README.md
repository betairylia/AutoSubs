*This project was developed with extensive assistance from Claude, Anthropic's AI assistant.*  
⬆ Including this claude-preferred disclaimer! 🤗🤖  
⬇ And this README.

# AutoSubs: ML-Based Auto Timing for Subtitling

AutoSubs is a machine learning system designed specifically for automatic subtitle timing prediction. This project focuses exclusively on timing generation and is **not intended for transcription or auto-translation**. Instead of manually timing each subtitle line by watching videos and listening to audio, AutoSubs uses deep learning to predict precise start and end timestamps from audio spectrograms.

**Note**: This is a research project for timing prediction only. Heavily WIP.

## 🏗️ System Overview

AutoSubs treats subtitle timing as an audio-based object detection problem, analyzing audio spectrograms to identify optimal timing points for subtitle placement and removal. AutoSubs is highly inspired by CornerNet.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone --recurse-submodules git@github.com:betairylia/AutoSubs.git
cd AutoSubs

# Install dependencies
pip install -r requirements.txt

# Or for development (includes testing and debugging tools)
pip install -r requirements-dev.txt
```

**System Requirements:**
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for training)
- GPU optional but recommended for training (CUDA or Apple Silicon)

### 1. Create Dataset

```bash
python scripts/create_dataset.py \
    --data_dir /path/to/audio_subtitle_pairs \
    --output dataset.h5 \
    --format hdf5 \
    --config configs/default.yaml
```

Your data directory should contain paired audio and subtitle files:
```
data/
├── episode1.mp3
├── episode1.ass
├── episode2.wav
├── episode2.ass
└── ...
```

### 2. Train Model

```bash
python scripts/train.py \
    --dataset dataset.h5 \
    --config configs/default.yaml \
    --experiment_name my_experiment
```

### 3. Run Inference

```bash
python scripts/inference.py \
    --model logs/my_experiment/checkpoints/best_model.pt \
    --input audio_file.wav \
    --output subtitles.ass \
    --config configs/default.yaml
```

### 4. Monitor Training (Optional)

```bash
# View training progress in TensorBoard
tensorboard --logdir logs/
```

## 📊 Dataset Requirements

AutoSubs requires paired audio and subtitle files for training:
- **Audio formats**: MP3, WAV, OGG, M4A, AAC, FLAC
- **Subtitle format**: ASS (Advanced SubStation Alpha)
- **Structure**: Files should be paired by filename (e.g., `episode1.mp3` + `episode1.ass`)

The system processes long audio files by dividing them into manageable chunks while preserving timing accuracy.

## 📁 Project Structure

```
AutoSubs/
├── config/          # Configuration system
│   ├── base.py      # Base config classes
│   ├── model.py     # Model configurations  
│   ├── data.py      # Data processing configs
│   ├── train.py     # Training configurations
│   └── config.py    # Main config class
├── data/            # Data processing pipeline
│   ├── utils.py     # File discovery utilities
│   ├── subtitle.py  # ASS file processing
│   ├── chunking.py  # Audio/subtitle chunking
│   └── dataset.py   # Dataset storage & loading
├── models/          # Neural network models
│   ├── backbone.py  # Conv1D & Transformer backbones
│   ├── heads.py     # Detection heads
│   └── network.py   # Complete model
├── training/        # Training system
│   ├── losses.py    # Loss functions
│   └── trainer.py   # Training loop
├── inference/       # Inference pipeline
│   ├── postprocessing.py  # NMS & feature matching
│   └── predictor.py       # Main predictor
├── utils/           # Utilities
│   ├── device.py    # Device detection
│   └── audio.py     # Audio processing
├── scripts/         # Command-line tools
│   ├── create_dataset.py  # Dataset creation
│   ├── train.py          # Model training
│   ├── inference.py      # Run inference
│   └── validate_dataset.py  # Dataset validation
└── configs/         # Configuration files
    └── default.yaml  # Default configuration
```

## ⚙️ Configuration

AutoSubs uses YAML configuration files for customization. Key settings include:

- **Model architecture**: Choice of backbone network and model size
- **Audio processing**: Sample rates, spectral analysis parameters
- **Training parameters**: Batch sizes, learning rates, optimization settings
- **Inference settings**: Confidence thresholds, timing precision

See `configs/default.yaml` for the complete configuration options.

## 🔧 Command-Line Tools

### Dataset Creation
```bash
# Create dataset with statistics
python scripts/create_dataset.py --data_dir /data --output dataset.h5 --stats_only --config configs/default.yaml

# Create full dataset
python scripts/create_dataset.py --data_dir /data --output dataset.h5 --format hdf5 --config configs/default.yaml
```

### Dataset Validation  
```bash
# Validate dataset integrity
python scripts/validate_dataset.py --dataset dataset.h5 --show_splits --test_loading --config configs/default.yaml

# Quick validation
python scripts/validate_dataset.py --dataset dataset.h5 --config configs/default.yaml
```

### Training
```bash  
# Train with custom config
python scripts/train.py --dataset dataset.h5 --config my_config.yaml

# Resume training
python scripts/train.py --dataset dataset.h5 --config configs/default.yaml --resume checkpoint.pt

# Override config parameters
python scripts/train.py --dataset dataset.h5 --config configs/default.yaml --batch_size 32 --lr 0.0005
```

### Inference
```bash
# Basic inference
python scripts/inference.py --model model.pt --input audio.wav --output subs.ass --config configs/default.yaml

# With audio analysis
python scripts/inference.py --model model.pt --input audio.wav --output subs.ass --config configs/default.yaml --stats

# Batch processing with custom overlap
python scripts/inference.py --model model.pt --input audio.wav --output subs.ass --config configs/default.yaml --batch_size 8 --overlap 0.2
```

## 🤝 Contributing

Contributions are welcome! This project is actively developed and we encourage community involvement.

## 📄 License

MIT (*Excluding Datasets*)

## 🙏 Acknowledgments

This project utilizes several open-source libraries and research contributions:
- PyTorch for deep learning framework
- Librosa for audio processing
- Various research papers on object detection and audio analysis

*This project was developed with assistance from Claude AI.*

---

**AutoSubs** - Automated timing for subtitling workflows
