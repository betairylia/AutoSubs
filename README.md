*This project was developed with extensive assistance from Claude, Anthropic's AI assistant.*  
^ Including this claude-preferred disclaimer! 🤗🤖

# AutoSubs: ML-Based Auto Timing for Subtitling

AutoSubs is a machine learning system for automatic subtitle timing prediction. Instead of manually timing each subtitle line by watching videos and listening to audio, AutoSubs uses deep learning to predict start and end timestamps from audio spectrograms.

## 🎯 Key Features

- **CornerNet-style Architecture**: Separate detection of subtitle start and end points with feature matching
- **Handles Overlapping Subtitles**: Supports intensive subtitle overlaps common in fansub content
- **60 FPS Timing Precision**: 16ms timing resolution as required by professional subtitling
- **Smart Audio Chunking**: 30-second overlapping blocks with proper padding and re-timing
- **Multiple Backbone Support**: Conv1D and Transformer architectures
- **Cross-Platform**: CUDA/MPS/CPU support with automatic device detection
- **Production Ready**: Complete pipeline from data preprocessing to inference

## 🏗️ Architecture Overview

The system treats subtitle timing as a 1D object detection problem:

1. **Audio Processing**: Convert audio to mel spectrograms (128 bands, 22050Hz)
2. **Backbone Network**: Extract temporal features using Conv1D or Transformer
3. **Dual Detection Heads**: Separate networks for start and end point detection
   - Classification head outputs confidence heatmaps
   - Feature head outputs vectors for start-end pairing
4. **Post-Processing**: Non-maximum suppression + feature matching to create subtitle pairs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
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

## 📊 Dataset Format

AutoSubs processes data in chunks:
- **Audio chunks**: 30-second segments with 10% overlap
- **Subtitle chunks**: ASS events re-timed to match audio chunks
- **Labels**: Gaussian-smoothed timing heatmaps + positive start-end pairs
- **Storage**: Efficient HDF5 or PyTorch format with metadata

## 🧠 Model Architecture

### Backbone Options

**Conv1D Backbone** (default):
- 4-layer CNN: 64→128→256→512 channels
- Kernel sizes: [3, 3, 3, 3]
- Strides: [1, 2, 2, 2] for temporal downsampling
- ~428k parameters

**Transformer Backbone**:
- 6 layers, 8 attention heads, 256 d_model
- Positional encoding for temporal awareness
- ~2.4M parameters

### Loss Functions

**Focal Loss** for heatmap prediction:
- Addresses class imbalance (most timepoints have no subtitles)
- α=0.25, γ=2.0 focusing on hard negatives

**Contrastive Loss** for feature matching:
- Pulls together features from true start-end pairs
- Pushes apart features from negative pairs
- Alternative InfoNCE loss implementation available

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

AutoSubs uses hierarchical YAML configuration:

```yaml
# Model configuration
model:
  backbone:
    type: "conv1d"  # or "transformer"
    channels: [64, 128, 256, 512]
    input_dim: 128
    output_dim: 256
  head:
    hidden_dims: [256, 128]
    feature_dim: 128
    dropout: 0.1

# Data configuration  
data:
  audio:
    sample_rate: 22050
    n_mels: 128
    n_fft: 2048
  chunking:
    chunk_duration: 30.0
    overlap_ratio: 0.1
    timing_fps: 60

# Training configuration
training:
  epochs: 100
  batch_size: 16
  optimizer:
    type: "adam"
    lr: 0.001
  loss:
    focal_alpha: 0.25
    focal_gamma: 2.0
```

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

## 📈 Training Details

### Data Processing Pipeline

1. **File Discovery**: Automatically pair audio/subtitle files by filename similarity
2. **Audio Chunking**: Split into 30s segments with configurable overlap
3. **Subtitle Re-timing**: Adjust ASS events to match chunk boundaries  
4. **Label Generation**: Create Gaussian-smoothed heatmaps from discrete timings
5. **Positive Pair Tracking**: Maintain start-end relationships for feature loss

### Training Loop

- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing, step, or plateau schedulers
- **Early Stopping**: Configurable patience and minimum delta
- **Checkpointing**: Save top-k models + regular intervals
- **TensorBoard Logging**: Loss curves, metrics, and hyperparameters

### Post-Processing Pipeline

1. **Peak Detection**: Non-maximum suppression or scipy peak finding
2. **Feature Matching**: Pair starts/ends using cosine similarity  
3. **Temporal Constraints**: Ensure logical start→end ordering
4. **Duration Filtering**: Remove very short/long subtitle candidates
5. **Overlap Resolution**: Handle competing subtitle timing predictions
6. **Confidence Filtering**: Remove low-confidence predictions

## 🎨 Advanced Usage

### Custom Model Architecture

```python
from config.model import ModelConfig, BackboneConfig, HeadConfig

config = ModelConfig(
    backbone=BackboneConfig(
        type="transformer",
        d_model=512,
        n_heads=16,
        n_layers=8
    ),
    head=HeadConfig(
        hidden_dims=[512, 256, 128],
        feature_dim=256
    )
)
```

### Custom Loss Configuration

```python  
from config.train import LossConfig

loss_config = LossConfig(
    focal_alpha=0.5,     # Higher weight on positive examples
    focal_gamma=3.0,     # Stronger focusing on hard examples  
    feature_weight=1.0,  # Equal weight to feature matching
    temperature=0.05     # Sharper feature similarities
)
```

### Inference with Raw Outputs

```python
from inference.predictor import create_predictor

predictor = create_predictor("model.pt", device="cuda")

# Get raw heatmaps and features
raw_outputs = predictor.predict_chunk(spectrogram, return_raw=True)
start_heatmap = raw_outputs["start_heatmap"]  
end_heatmap = raw_outputs["end_heatmap"]
start_features = raw_outputs["start_features"]
end_features = raw_outputs["end_features"]

# Custom post-processing
from inference.postprocessing import InferencePostProcessor
processor = InferencePostProcessor(config.model)  
events = processor.process_predictions(start_heatmap, end_heatmap, start_features, end_features)
```

## 📝 File Format Support

**Supported Audio Formats**: MP3, WAV, OGG, M4A, FLAC (via librosa)

**Subtitle Format**: ASS (Advanced SubStation Alpha)
- Full ASS parsing with timing extraction
- Preserves original formatting tags and styles
- Supports overlapping dialogue and complex timing

## 🛠️ Troubleshooting

### Common Issues

**"Timing FPS mismatch" warning**:
- Audio processing FPS ≠ timing FPS (60)
- Handled automatically by temporal adaptation layers
- Consider adjusting `hop_length` for exact matching

**Out of memory during training**:
- Reduce `batch_size` in config or command line
- Use gradient accumulation: `accumulate_grad_batches: 2`
- Switch to smaller backbone architecture

**No audio-subtitle pairs found**:
- Check filename matching between audio/subtitle files
- Ensure files have supported extensions (.mp3/.wav + .ass)
- Use `--stats_only` to debug file discovery

**Poor prediction accuracy**:  
- Increase training data (more diverse audio types)
- Adjust confidence thresholds in model config
- Try different backbone architecture (Conv1D ↔ Transformer)
- Increase model size or training epochs

### Device-Specific Notes

**Apple Silicon (MPS)**:
- Automatic mixed precision disabled (MPS limitations)
- Consider batch sizes 4-16 for optimal memory usage

**CUDA**:
- Enable mixed precision for faster training  
- Use larger batch sizes (16-32) if memory allows

**CPU Only**:
- Expect 5-10x slower training/inference
- Reduce model size for reasonable performance

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Additional Backbones**: ResNet, EfficientNet, ViT adaptations
- **Data Augmentation**: Pitch shifting, noise injection, speed changes  
- **Multi-language Support**: Language-specific timing patterns
- **Evaluation Metrics**: Precision/recall at different IoU thresholds
- **Export Formats**: SRT, VTT, other subtitle formats

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **CornerNet**: Inspiration for dual detection head architecture
- **Focal Loss**: Addressing class imbalance in detection tasks  
- **Librosa**: Excellent audio processing library
- **PyTorch**: Deep learning framework
- **ASS Library**: Python ASS file handling

## 📚 References

TODO!

---

**AutoSubs** - Bringing ML automation to the fansub community! 🎬✨