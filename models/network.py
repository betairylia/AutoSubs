import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.model import ModelConfig
from models.backbone import create_backbone
from models.heads import create_head_network, TemporalAdaptationLayer


class AutoSubsNetwork(nn.Module):
    """
    Complete AutoSubs network combining backbone and dual heads.
    Implements the CornerNet-like approach for 1D audio subtitle timing detection.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Create backbone
        self.backbone = create_backbone(config.backbone)
        
        # Get backbone output dimension
        if config.backbone.type == "conv1d":
            backbone_output_dim = config.backbone.output_dim
        elif config.backbone.type == "transformer":
            backbone_output_dim = config.backbone.output_dim
        elif config.backbone.type == "whisper":
            backbone_output_dim = config.backbone.output_dim
        else:
            raise ValueError(f"Unknown backbone type: {config.backbone.type}")
        
        # Create temporal adaptation layer
        # This handles the mismatch between audio frames and timing frames
        self.temporal_adapter = TemporalAdaptationLayer(
            input_dim=backbone_output_dim,
            method="interpolate"
        )
        
        # Create dual heads
        self.dual_heads = create_head_network(
            backbone_output_dim=backbone_output_dim,
            config=config.head
        )
    
    def forward(
        self, 
        spectrogram: torch.Tensor, 
        target_timing_length: Optional[int] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete network.
        
        Args:
            spectrogram: Input mel spectrogram (batch_size, n_mels, n_frames)
            target_timing_length: Target length for timing predictions
            
        Returns:
            Dictionary with 'start' and 'end' predictions, each containing 'heatmap' and 'features'
        """
        # Backbone processing
        backbone_features = self.backbone(spectrogram)
        
        # Handle different backbone output formats
        if self.config.backbone.type == "conv1d":
            # Conv1D output: (batch_size, feature_dim, n_frames)
            # Transpose for temporal adaptation and heads
            backbone_features = backbone_features.transpose(1, 2)  # (batch_size, n_frames, feature_dim)
        elif self.config.backbone.type == "transformer":
            # Transformer output: (batch_size, n_frames, feature_dim)
            # Already in correct format
            pass
        elif self.config.backbone.type == "whisper":
            # Whisper encoder output: (batch_size, n_frames, feature_dim)
            # Already in correct format
            pass
        
        # Temporal adaptation to match timing resolution
        if target_timing_length is not None:
            adapted_features = self.temporal_adapter(backbone_features, target_timing_length)
        else:
            adapted_features = backbone_features
        
        # Dual head processing
        outputs = self.dual_heads(adapted_features)
        
        return outputs
    
    def get_timing_predictions(
        self, 
        spectrogram: torch.Tensor,
        timing_fps: int = None,
        chunk_duration: float = 30.0
    ) -> Dict[str, torch.Tensor]:
        """
        Get timing predictions with proper temporal resolution.
        
        Args:
            spectrogram: Input mel spectrogram
            timing_fps: Target timing FPS (frames per second). If None, uses backbone's natural resolution.
            chunk_duration: Chunk duration in seconds
            
        Returns:
            Dictionary with timing predictions
        """
        # Use backbone's natural temporal resolution if not specified
        if timing_fps is None:
            timing_fps = int(self.backbone.get_temporal_resolution())
        
        # Calculate target timing frames
        target_timing_frames = int(chunk_duration * timing_fps)
        
        # Forward pass
        outputs = self.forward(spectrogram, target_timing_frames)
        
        return {
            "start_heatmap": outputs["start"]["heatmap"],
            "end_heatmap": outputs["end"]["heatmap"],
            "start_features": outputs["start"]["features"],
            "end_features": outputs["end"]["features"]
        }
    
    def extract_features_at_times(
        self,
        spectrogram: torch.Tensor,
        time_indices: torch.Tensor,
        timing_fps: int = None,
        chunk_duration: float = 30.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract start and end features at specific time indices.
        Useful for computing feature loss during training.
        
        Args:
            spectrogram: Input mel spectrogram
            time_indices: Time indices (batch_size, n_events, 2) where [..., 0] is start, [..., 1] is end
            timing_fps: Timing FPS. If None, uses backbone's natural resolution.
            chunk_duration: Chunk duration
            
        Returns:
            Tuple of (start_features, end_features) at specified times
        """
        # Use backbone's natural temporal resolution if not specified
        if timing_fps is None:
            timing_fps = int(self.backbone.get_temporal_resolution())
        
        predictions = self.get_timing_predictions(spectrogram, timing_fps, chunk_duration)
        
        batch_size = time_indices.size(0)
        n_events = time_indices.size(1)
        feature_dim = predictions["start_features"].size(-1)
        
        # Extract features at specified indices
        start_features = torch.zeros(batch_size, n_events, feature_dim, device=spectrogram.device)
        end_features = torch.zeros(batch_size, n_events, feature_dim, device=spectrogram.device)
        
        for b in range(batch_size):
            for e in range(n_events):
                start_idx = time_indices[b, e, 0].long()
                end_idx = time_indices[b, e, 1].long()
                
                # Clamp indices to valid range
                start_idx = torch.clamp(start_idx, 0, predictions["start_features"].size(1) - 1)
                end_idx = torch.clamp(end_idx, 0, predictions["end_features"].size(1) - 1)
                
                start_features[b, e] = predictions["start_features"][b, start_idx]
                end_features[b, e] = predictions["end_features"][b, end_idx]
        
        return start_features, end_features


def create_model(config: ModelConfig) -> AutoSubsNetwork:
    """Factory function to create the complete model."""
    return AutoSubsNetwork(config)


class ModelSummary:
    """Utility class for model summary and statistics."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Component breakdown
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        head_params = sum(p.numel() for p in model.dual_heads.parameters())
        adapter_params = sum(p.numel() for p in model.temporal_adapter.parameters())
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "backbone": backbone_params,
            "heads": head_params,
            "adapter": adapter_params
        }
    
    @staticmethod
    def get_model_info(model: AutoSubsNetwork) -> Dict:
        """Get comprehensive model information."""
        param_counts = ModelSummary.count_parameters(model)
        
        return {
            "backbone_type": model.config.backbone.type,
            "backbone_config": model.config.backbone.__dict__,
            "head_config": model.config.head.__dict__,
            "parameter_counts": param_counts,
            "model_size_mb": param_counts["total"] * 4 / (1024 * 1024)  # Assuming float32
        }


if __name__ == "__main__":
    # Test complete network
    print("Testing complete AutoSubs network...")
    
    from config.model import ModelConfig, BackboneConfig, HeadConfig
    
    # Test Conv1D configuration
    conv_config = ModelConfig(
        backbone=BackboneConfig(
            type="conv1d",
            channels=[64, 128, 256],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            input_dim=128,
            output_dim=256
        ),
        head=HeadConfig(
            hidden_dims=[256, 128],
            dropout=0.1,
            activation="relu", 
            feature_dim=128
        )
    )
    
    print("\n1. Testing Conv1D network:")
    conv_model = create_model(conv_config)
    
    # Test input
    test_spectrogram = torch.randn(2, 128, 431)  # (batch, n_mels, n_frames)
    
    with torch.no_grad():
        # Test basic forward pass
        conv_output = conv_model(test_spectrogram)
        
        # Test timing predictions (use backbone's natural resolution)
        timing_predictions = conv_model.get_timing_predictions(
            test_spectrogram,
            chunk_duration=30.0
        )
    
    print(f"Input shape: {test_spectrogram.shape}")
    print(f"Start heatmap shape: {conv_output['start']['heatmap'].shape}")
    print(f"Timing predictions - start heatmap: {timing_predictions['start_heatmap'].shape}")
    
    # Test Transformer configuration
    transformer_config = ModelConfig(
        backbone=BackboneConfig(
            type="transformer",
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            input_dim=128,
            output_dim=256
        ),
        head=HeadConfig(
            hidden_dims=[256, 128],
            dropout=0.1,
            activation="relu",
            feature_dim=128
        )
    )
    
    print("\n2. Testing Transformer network:")
    transformer_model = create_model(transformer_config)
    
    with torch.no_grad():
        transformer_output = transformer_model(test_spectrogram)
        transformer_timing = transformer_model.get_timing_predictions(
            test_spectrogram,
            chunk_duration=30.0
        )
    
    print(f"Transformer start heatmap shape: {transformer_output['start']['heatmap'].shape}")
    print(f"Transformer timing predictions: {transformer_timing['start_heatmap'].shape}")
    
    # Model statistics
    print("\n3. Model statistics:")
    conv_info = ModelSummary.get_model_info(conv_model)
    transformer_info = ModelSummary.get_model_info(transformer_model)
    
    print(f"Conv1D model parameters: {conv_info['parameter_counts']['total']:,}")
    print(f"Conv1D model size: {conv_info['model_size_mb']:.2f} MB")
    print(f"Transformer model parameters: {transformer_info['parameter_counts']['total']:,}")
    print(f"Transformer model size: {transformer_info['model_size_mb']:.2f} MB")
    
    # Test feature extraction at specific times
    print("\n4. Testing feature extraction at times:")
    time_indices = torch.tensor([[[30, 150], [300, 450]]])  # Two events: (start, end) indices
    
    with torch.no_grad():
        start_feats, end_feats = conv_model.extract_features_at_times(
            test_spectrogram[:1],  # Single batch
            time_indices,
            chunk_duration=30.0
        )
    
    print(f"Extracted start features shape: {start_feats.shape}")
    print(f"Extracted end features shape: {end_feats.shape}")
    
    print("\nComplete network test passed!")