from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseConfig
from .model import ModelConfig
from .data import DataConfig
from .train import TrainingConfig


@dataclass
class Config(BaseConfig):
    """Main configuration class combining all sub-configurations."""
    
    # Project metadata
    project_name: str = "AutoSubs"
    version: str = "1.0.0"
    description: str = "ML-based auto timing for subtitling"
    
    # Sub-configurations
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    # Global settings
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        super().__post_init__()
    
    def validate(self):
        """Validate entire configuration."""
        # Validate sub-configurations
        self.model.validate()
        self.data.validate()
        self.training.validate()
        
        # Cross-validation between configs
        self._validate_cross_config()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary with proper nested object handling."""
        # Handle nested configurations
        processed_dict = config_dict.copy()
        
        if 'model' in processed_dict and isinstance(processed_dict['model'], dict):
            processed_dict['model'] = ModelConfig.from_dict(processed_dict['model'])
        
        if 'data' in processed_dict and isinstance(processed_dict['data'], dict):
            processed_dict['data'] = DataConfig.from_dict(processed_dict['data'])
            
        if 'training' in processed_dict and isinstance(processed_dict['training'], dict):
            processed_dict['training'] = TrainingConfig.from_dict(processed_dict['training'])
        
        return cls(**processed_dict)
    
    def _validate_cross_config(self):
        """Validate consistency between different config sections."""
        # Ensure backbone output dim matches head input requirements
        backbone_out = self.model.backbone.output_dim
        # Head should accept backbone output (no explicit input_dim in head config, 
        # assuming it's handled in model construction)
        
        # Ensure timing resolution consistency
        timing_fps = self.data.chunking.timing_fps
        sample_rate = self.data.audio.sample_rate
        hop_length = self.data.audio.hop_length
        
        # Frame rate from audio processing
        audio_fps = sample_rate / hop_length
        
        # Warn if there's a significant mismatch
        if abs(timing_fps - audio_fps) > 5:  # Allow some tolerance
            print(f"Warning: Timing FPS ({timing_fps}) and audio FPS ({audio_fps:.1f}) mismatch")


# Convenience function for loading config
def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file or return default config."""
    if config_path is None:
        return Config()  # Default configuration
    return Config.from_yaml(config_path)