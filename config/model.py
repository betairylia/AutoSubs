from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from .base import BaseConfig


@dataclass
class BackboneConfig(BaseConfig):
    """Configuration for audio backbone network."""
    
    type: str = "conv1d"  # "conv1d", "transformer", "whisper"
    
    # Conv1D specific parameters
    channels: List[int] = None  # [64, 128, 256, 512]
    kernel_sizes: List[int] = None  # [3, 3, 3, 3]
    strides: List[int] = None  # [1, 2, 2, 2]
    
    # Transformer specific parameters
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    
    # Whisper specific parameters
    whisper_model_size: str = "base"  # "tiny", "base", "small", "medium", "large"
    
    # Common parameters
    input_dim: int = 128  # mel spectrogram dimension
    output_dim: int = 256  # feature dimension
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]
        if self.strides is None:
            self.strides = [1, 2, 2, 2]
        super().__post_init__()
    
    def validate(self):
        if self.type == "conv1d":
            assert len(self.channels) == len(self.kernel_sizes) == len(self.strides), \
                "Conv1D parameters must have same length"
        elif self.type == "whisper":
            valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "turbo"]
            assert self.whisper_model_size in valid_sizes, \
                f"Whisper model size must be one of {valid_sizes}"
        assert self.input_dim > 0, "Input dimension must be positive"
        assert self.output_dim > 0, "Output dimension must be positive"


@dataclass  
class HeadConfig(BaseConfig):
    """Configuration for starting/ending network heads."""
    
    hidden_dims: List[int] = None  # [256, 128]
    dropout: float = 0.1
    activation: str = "relu"  # "relu", "gelu", "swish"
    feature_dim: int = 128  # dimension of feature vectors for pairing
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
        super().__post_init__()
    
    def validate(self):
        assert self.feature_dim > 0, "Feature dimension must be positive"
        assert 0 <= self.dropout <= 1, "Dropout must be between 0 and 1"
        assert self.activation in ["relu", "gelu", "swish"], "Invalid activation function"


@dataclass
class ModelConfig(BaseConfig):
    """Overall model configuration."""
    
    backbone: BackboneConfig = None
    head: HeadConfig = None
    
    # NMP filtering parameters
    nmp_window_size: int = 5  # window size for non-maximum suppression
    confidence_threshold: float = 0.5  # minimum confidence for detection
    
    # Post-processing parameters
    feature_distance_threshold: float = 2.0  # max distance for start-end pairing
    max_subtitle_length: float = 10.0  # max subtitle duration in seconds
    
    def __post_init__(self):
        if self.backbone is None:
            self.backbone = BackboneConfig()
        if self.head is None:
            self.head = HeadConfig()
        super().__post_init__()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary with proper nested object handling."""
        processed_dict = config_dict.copy()
        
        if 'backbone' in processed_dict and isinstance(processed_dict['backbone'], dict):
            processed_dict['backbone'] = BackboneConfig.from_dict(processed_dict['backbone'])
        
        if 'head' in processed_dict and isinstance(processed_dict['head'], dict):
            processed_dict['head'] = HeadConfig.from_dict(processed_dict['head'])
        
        return cls(**processed_dict)
    
    def validate(self):
        assert self.nmp_window_size > 0, "NMP window size must be positive"
        assert 0 <= self.confidence_threshold <= 1, "Confidence threshold must be between 0 and 1"
        assert self.feature_distance_threshold > 0, "Feature distance threshold must be positive"
        assert self.max_subtitle_length > 0, "Max subtitle length must be positive"