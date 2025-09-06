from dataclasses import dataclass
from typing import Dict, Any, Optional
from .base import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizer."""
    
    type: str = "adam"  # "adam", "adamw", "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # Adam/AdamW specific
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # SGD specific
    momentum: float = 0.9
    nesterov: bool = False
    
    def validate(self):
        assert self.type in ["adam", "adamw", "sgd"], "Invalid optimizer type"
        assert self.lr > 0, "Learning rate must be positive"
        assert self.weight_decay >= 0, "Weight decay must be non-negative"


@dataclass
class SchedulerConfig(BaseConfig):
    """Configuration for learning rate scheduler."""
    
    type: str = "cosine"  # "cosine", "step", "plateau", "none"
    
    # Cosine annealing
    T_max: int = 100
    eta_min: float = 1e-6
    
    # Step scheduler
    step_size: int = 30
    gamma: float = 0.1
    
    # Plateau scheduler
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    
    def validate(self):
        assert self.type in ["cosine", "step", "plateau", "none"], "Invalid scheduler type"


@dataclass
class LossConfig(BaseConfig):
    """Configuration for loss functions."""
    
    # Focal loss parameters
    focal_alpha: float = 0.25  # weighting factor for rare class
    focal_gamma: float = 2.0  # focusing parameter
    
    # Feature loss parameters
    feature_margin: float = 1.0  # margin for contrastive loss
    temperature: float = 0.1  # temperature for similarity
    
    # Loss weighting
    heatmap_weight: float = 1.0  # weight for heatmap loss
    feature_weight: float = 0.5  # weight for feature loss
    
    def validate(self):
        assert self.focal_alpha > 0, "Focal alpha must be positive"
        assert self.focal_gamma > 0, "Focal gamma must be positive"
        assert self.feature_margin > 0, "Feature margin must be positive"
        assert self.temperature > 0, "Temperature must be positive"
        assert self.heatmap_weight >= 0, "Heatmap weight must be non-negative"
        assert self.feature_weight >= 0, "Feature weight must be non-negative"


@dataclass
class TrainingConfig(BaseConfig):
    """Overall training configuration."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    grad_clip_norm: float = 1.0
    
    # Device and performance
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Validation and checkpointing
    val_check_interval: int = 1000  # steps between validation
    save_top_k: int = 3  # number of best models to keep
    save_every_n_epochs: int = 10  # checkpoint frequency
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Logging
    log_every_n_steps: int = 50
    log_dir: str = "./logs"
    experiment_name: str = "autosubs"
    
    # Components
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None
    loss: LossConfig = None
    
    # Mixed precision
    use_amp: bool = True  # automatic mixed precision
    
    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()
        if self.loss is None:
            self.loss = LossConfig()
        super().__post_init__()
    
    def validate(self):
        assert self.epochs > 0, "Number of epochs must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.accumulate_grad_batches > 0, "Gradient accumulation batches must be positive"
        assert self.grad_clip_norm > 0, "Gradient clip norm must be positive"
        assert self.device in ["auto", "cuda", "mps", "cpu"], "Invalid device"
        assert self.num_workers >= 0, "Number of workers must be non-negative"
        assert self.val_check_interval > 0, "Validation check interval must be positive"
        assert self.save_top_k >= 0, "Save top k must be non-negative"
        assert self.early_stopping_patience > 0, "Early stopping patience must be positive"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary with proper nested object handling."""
        processed_dict = config_dict.copy()
        
        if 'optimizer' in processed_dict and isinstance(processed_dict['optimizer'], dict):
            processed_dict['optimizer'] = OptimizerConfig.from_dict(processed_dict['optimizer'])
        
        if 'scheduler' in processed_dict and isinstance(processed_dict['scheduler'], dict):
            processed_dict['scheduler'] = SchedulerConfig.from_dict(processed_dict['scheduler'])
            
        if 'loss' in processed_dict and isinstance(processed_dict['loss'], dict):
            processed_dict['loss'] = LossConfig.from_dict(processed_dict['loss'])
        
        return cls(**processed_dict)