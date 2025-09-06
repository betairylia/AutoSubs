from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from .base import BaseConfig


@dataclass
class AudioConfig(BaseConfig):
    """Configuration for audio processing."""
    
    # Sampling and spectral parameters
    sample_rate: int = 22050  # target sample rate
    n_mels: int = 128  # number of mel bands
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # hop length for STFT
    win_length: int = 2048  # window length for STFT
    
    # Normalization
    normalize: bool = True
    
    # Whisper preprocessing option
    use_whisper_preprocessing: bool = False
    
    def validate(self):
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.n_mels > 0, "Number of mel bands must be positive"
        assert self.n_fft > 0, "FFT size must be positive"
        assert self.hop_length > 0, "Hop length must be positive"


@dataclass
class ChunkingConfig(BaseConfig):
    """Configuration for audio chunking."""
    
    chunk_duration: float = 30.0  # chunk duration in seconds
    overlap_ratio: float = 0.1  # overlap between chunks (0.1 = 10%)
    padding_duration: float = 2.0  # padding duration in seconds
    
    # Timing resolution (60 FPS as specified)
    timing_fps: int = 60  # frames per second for timing
    
    def validate(self):
        assert self.chunk_duration > 0, "Chunk duration must be positive"
        assert 0 <= self.overlap_ratio < 1, "Overlap ratio must be in [0, 1)"
        assert self.padding_duration >= 0, "Padding duration must be non-negative"
        assert self.timing_fps > 0, "Timing FPS must be positive"
    
    @property
    def chunk_samples(self) -> int:
        """Number of samples per chunk at 22050 Hz."""
        return int(self.chunk_duration * 22050)
    
    @property
    def overlap_samples(self) -> int:
        """Number of overlapping samples."""
        return int(self.chunk_samples * self.overlap_ratio)
    
    @property
    def padding_samples(self) -> int:
        """Number of padding samples."""
        return int(self.padding_duration * 22050)


@dataclass
class LabelConfig(BaseConfig):
    """Configuration for label processing."""
    
    # Gaussian kernel for soft labels
    gaussian_sigma: float = 0.5  # sigma for gaussian kernel (in seconds)
    
    # Label handling
    merge_same_timestamp: bool = True  # merge multiple events at same timestamp
    timestamp_tolerance: float = 0.1  # tolerance for timestamp matching (in seconds)
    
    def validate(self):
        assert self.gaussian_sigma > 0, "Gaussian sigma must be positive"
        assert self.timestamp_tolerance >= 0, "Timestamp tolerance must be non-negative"


@dataclass
class DataConfig(BaseConfig):
    """Overall data processing configuration."""
    
    # File discovery
    audio_extensions: List[str] = None  # [".mp3", ".wav", ".ogg", ".m4a"]
    subtitle_extensions: List[str] = None  # [".ass"]
    
    # Processing configs
    audio: AudioConfig = None
    chunking: ChunkingConfig = None
    labels: LabelConfig = None
    
    # Dataset parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Storage
    dataset_format: str = "hdf5"  # "hdf5", "pytorch"
    cache_dir: str = "./cache"
    
    def __post_init__(self):
        if self.audio_extensions is None:
            self.audio_extensions = [".mp3", ".wav", ".ogg", ".m4a", ".aac"]
        if self.subtitle_extensions is None:
            self.subtitle_extensions = [".ass"]
        if self.audio is None:
            self.audio = AudioConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.labels is None:
            self.labels = LabelConfig()
        super().__post_init__()
    
    def validate(self):
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Data splits must sum to 1.0"
        assert all(split >= 0 for split in [self.train_split, self.val_split, self.test_split]), \
            "Data splits must be non-negative"
        assert self.dataset_format in ["hdf5", "pytorch"], "Invalid dataset format"
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary with proper nested object handling."""
        processed_dict = config_dict.copy()
        
        if 'audio' in processed_dict and isinstance(processed_dict['audio'], dict):
            processed_dict['audio'] = AudioConfig.from_dict(processed_dict['audio'])
        
        if 'chunking' in processed_dict and isinstance(processed_dict['chunking'], dict):
            processed_dict['chunking'] = ChunkingConfig.from_dict(processed_dict['chunking'])
            
        if 'labels' in processed_dict and isinstance(processed_dict['labels'], dict):
            processed_dict['labels'] = LabelConfig.from_dict(processed_dict['labels'])
        
        return cls(**processed_dict)