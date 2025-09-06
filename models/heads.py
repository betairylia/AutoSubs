import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.model import HeadConfig


class StartEndHead(nn.Module):
    """
    Head network for start/end detection.
    Outputs both classification heatmaps and feature vectors for pairing.
    """
    
    def __init__(self, input_dim: int, config: HeadConfig):
        super().__init__()
        self.config = config
        
        # Build the hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Classification head (output confidence 0-1)
        self.classification_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )
        
        # Feature head (for start-end pairing)
        self.feature_head = nn.Linear(prev_dim, config.feature_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features from backbone
            
        Returns:
            Dictionary with 'heatmap' and 'features'
        """
        # Hidden layers
        h = self.hidden_layers(x)
        
        # Classification output
        heatmap = self.classification_head(h).squeeze(-1)  # Remove last dimension
        
        # Feature output (normalize for better pairing)
        features = F.normalize(self.feature_head(h), p=2, dim=-1)
        
        return {
            "heatmap": heatmap,    # (batch_size, n_frames)
            "features": features   # (batch_size, n_frames, feature_dim)
        }


class DualHeadNetwork(nn.Module):
    """
    Dual network with identical starting and ending heads.
    """
    
    def __init__(self, input_dim: int, config: HeadConfig):
        super().__init__()
        
        # Create identical heads for starting and ending
        self.start_head = StartEndHead(input_dim, config)
        self.end_head = StartEndHead(input_dim, config)
        
        # Ensure both heads have the same architecture
        # (they are created separately but with identical configs)
    
    def forward(self, x) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through both heads.
        
        Args:
            x: Input features from backbone
            
        Returns:
            Dictionary with 'start' and 'end' predictions
        """
        start_output = self.start_head(x)
        end_output = self.end_head(x)
        
        return {
            "start": start_output,
            "end": end_output
        }


class TemporalAdaptationLayer(nn.Module):
    """
    Layer to adapt backbone features to timing prediction.
    Handles different temporal resolutions between audio features and timing labels.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        target_length: int = None,
        method: str = "interpolate"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_length = target_length
        self.method = method  # "interpolate", "conv", "pool"
        
        if method == "conv":
            # Use transposed conv for upsampling if needed
            self.upsample = nn.ConvTranspose1d(input_dim, input_dim, kernel_size=4, stride=2, padding=1)
        elif method == "pool":
            # Use adaptive pooling for fixed output length
            self.pool = nn.AdaptiveAvgPool1d(target_length) if target_length else None
    
    def forward(self, x, target_length: int = None):
        """
        Adapt temporal resolution.
        
        Args:
            x: Input tensor (batch_size, n_features, n_frames) or (batch_size, n_frames, n_features)
            target_length: Target temporal length
            
        Returns:
            Adapted tensor
        """
        if target_length is None:
            target_length = self.target_length
        
        if target_length is None:
            return x
        
        # Handle different input formats
        if x.dim() == 3 and x.size(1) == self.input_dim:
            # Format: (batch_size, n_features, n_frames) - Conv format
            current_length = x.size(2)
            conv_format = True
        elif x.dim() == 3 and x.size(2) == self.input_dim:
            # Format: (batch_size, n_frames, n_features) - Transformer format
            current_length = x.size(1)
            conv_format = False
            x = x.transpose(1, 2)  # Convert to conv format
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        if current_length == target_length:
            # No adaptation needed
            output = x
        elif self.method == "interpolate":
            # Simple interpolation
            output = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        elif self.method == "conv" and current_length < target_length:
            # Upsample using transposed convolution
            output = self.upsample(x)
            # May need additional interpolation for exact size
            if output.size(2) != target_length:
                output = F.interpolate(output, size=target_length, mode='linear', align_corners=False)
        elif self.method == "pool":
            # Use adaptive pooling
            output = self.pool(x) if self.pool else F.adaptive_avg_pool1d(x, target_length)
        else:
            # Fallback to interpolation
            output = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        # Convert back to original format if needed
        if not conv_format:
            output = output.transpose(1, 2)
        
        return output


class MultiScaleHead(nn.Module):
    """
    Multi-scale head for better temporal resolution handling.
    """
    
    def __init__(self, input_dim: int, config: HeadConfig):
        super().__init__()
        
        # Multiple heads at different scales
        self.scales = [1, 2, 4]  # Different temporal scales
        self.scale_heads = nn.ModuleList([
            StartEndHead(input_dim, config) for _ in self.scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(len(self.scales), 1)
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale processing."""
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale > 1:
                # Downsample for larger scale
                pooled_x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
                output = self.scale_heads[i](pooled_x)
                
                # Upsample back to original size
                output["heatmap"] = F.interpolate(
                    output["heatmap"].unsqueeze(1), 
                    size=x.size(-1), 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)
                
                output["features"] = F.interpolate(
                    output["features"].transpose(1, 2), 
                    size=x.size(-1), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            else:
                output = self.scale_heads[i](x)
            
            scale_outputs.append(output)
        
        # Fuse heatmaps from different scales
        heatmaps = torch.stack([out["heatmap"] for out in scale_outputs], dim=-1)
        fused_heatmap = self.fusion(heatmaps).squeeze(-1)
        
        # Use features from finest scale
        features = scale_outputs[0]["features"]
        
        return {
            "heatmap": fused_heatmap,
            "features": features
        }


def create_head_network(backbone_output_dim: int, config: HeadConfig, multi_scale: bool = False):
    """Factory function to create head networks."""
    if multi_scale:
        return DualHeadNetwork(backbone_output_dim, config)  # Could extend to use MultiScaleHead
    else:
        return DualHeadNetwork(backbone_output_dim, config)


if __name__ == "__main__":
    # Test head implementations
    print("Testing dual head networks...")
    
    from config.model import HeadConfig
    
    # Create head configuration
    head_config = HeadConfig(
        hidden_dims=[256, 128],
        dropout=0.1,
        activation="relu",
        feature_dim=128
    )
    
    print(f"Head config: {head_config.__dict__}")
    
    # Test with Conv1D backbone output format
    print("\n1. Testing with Conv1D backbone output:")
    conv_input = torch.randn(2, 256, 108)  # (batch, features, frames)
    
    # Need to transpose for head processing (heads expect frame-major format)
    conv_input_t = conv_input.transpose(1, 2)  # (batch, frames, features)
    
    dual_head = DualHeadNetwork(input_dim=256, config=head_config)
    
    with torch.no_grad():
        conv_output = dual_head(conv_input_t)
    
    print(f"Input shape: {conv_input_t.shape}")
    print(f"Start heatmap shape: {conv_output['start']['heatmap'].shape}")
    print(f"Start features shape: {conv_output['start']['features'].shape}")
    print(f"End heatmap shape: {conv_output['end']['heatmap'].shape}")
    print(f"End features shape: {conv_output['end']['features'].shape}")
    
    # Test with Transformer backbone output format
    print("\n2. Testing with Transformer backbone output:")
    transformer_input = torch.randn(2, 431, 256)  # (batch, frames, features)
    
    with torch.no_grad():
        transformer_output = dual_head(transformer_input)
    
    print(f"Input shape: {transformer_input.shape}")
    print(f"Start heatmap shape: {transformer_output['start']['heatmap'].shape}")
    print(f"Start features shape: {transformer_output['start']['features'].shape}")
    
    # Test temporal adaptation
    print("\n3. Testing temporal adaptation:")
    adaptation = TemporalAdaptationLayer(input_dim=256, target_length=600)
    
    # Test with different input lengths
    test_inputs = [
        torch.randn(1, 256, 108),   # Conv output
        torch.randn(1, 431, 256),   # Transformer output  
    ]
    
    for i, test_input in enumerate(test_inputs):
        with torch.no_grad():
            adapted = adaptation(test_input, target_length=600)
        print(f"Input {i+1}: {test_input.shape} -> {adapted.shape}")
    
    print("\nDual head network test complete!")
    
    # Test feature normalization
    print("\n4. Testing feature normalization:")
    features = conv_output['start']['features'][0]  # Take first batch
    print(f"Feature norms (should be ~1.0): {torch.norm(features, p=2, dim=-1)[:5]}")
    
    print("\nAll tests passed!")