import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.model import BackboneConfig


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv1DBlock(nn.Module):
    """A single Conv1D block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Conv1DBackbone(nn.Module):
    """Conv1D backbone for audio processing."""
    
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        
        # Input projection layer
        self.input_projection = nn.Conv1d(
            config.input_dim, 
            config.channels[0], 
            kernel_size=1
        )
        
        # Conv1D layers
        layers = []
        for i in range(len(config.channels)):
            in_channels = config.channels[i]
            out_channels = config.channels[i] if i == len(config.channels) - 1 else config.channels[i + 1]
            
            # Don't create layer for the last iteration
            if i < len(config.channels) - 1:
                layers.append(Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_sizes[i],
                    stride=config.strides[i],
                    padding=config.kernel_sizes[i] // 2  # Same padding
                ))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Output projection
        final_channels = config.channels[-1] if len(config.channels) > 1 else config.channels[0]
        self.output_projection = nn.Conv1d(
            final_channels,
            config.output_dim,
            kernel_size=1
        )
        
        # Global average pooling along frequency dimension if needed
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_mels, n_frames)
            
        Returns:
            Output tensor (batch_size, output_dim, n_frames')
        """
        # Input projection
        x = self.input_projection(x)
        
        # Conv1D layers
        x = self.conv_layers(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class TransformerBackbone(nn.Module):
    """Transformer backbone for audio processing."""
    
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.output_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, n_mels, n_frames)
            
        Returns:
            Output tensor (batch_size, n_frames, output_dim)
        """
        batch_size, n_mels, n_frames = x.shape
        
        # Transpose to (batch_size, n_frames, n_mels)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (n_frames, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # Back to (batch_size, n_frames, d_model)
        
        # Transformer layers
        x = self.transformer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class AudioBackbone(nn.Module):
    """Factory class for audio backbone networks."""
    
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        
        if config.type == "conv1d":
            self.backbone = Conv1DBackbone(config)
        elif config.type == "transformer":
            self.backbone = TransformerBackbone(config)
        else:
            raise ValueError(f"Unknown backbone type: {config.type}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_output_size(self, input_size: tuple) -> tuple:
        """Get output size given input size."""
        dummy_input = torch.randn(1, *input_size)
        with torch.no_grad():
            output = self.forward(dummy_input)
        return output.shape[1:]


class ResidualConv1DBlock(nn.Module):
    """Residual Conv1D block for more powerful conv backbones."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual  # Residual connection
        x = self.activation(x)
        
        return x


class AttentionPool1D(nn.Module):
    """Attention-based pooling for 1D sequences."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, length)
        Returns:
            Pooled tensor: (batch_size, channels)
        """
        # Transpose to (batch_size, length, channels)
        x = x.transpose(1, 2)
        
        # Compute attention weights
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (batch_size, length, 1)
        
        # Apply attention pooling
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, channels)
        
        return pooled


def create_backbone(config: BackboneConfig) -> AudioBackbone:
    """Factory function to create audio backbone."""
    return AudioBackbone(config)


if __name__ == "__main__":
    # Test backbone implementations
    print("Testing audio backbone implementations...")
    
    # Test Conv1D backbone
    print("\n1. Testing Conv1D backbone:")
    conv_config = BackboneConfig(
        type="conv1d",
        channels=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        input_dim=128,
        output_dim=256
    )
    
    conv_backbone = create_backbone(conv_config)
    dummy_input = torch.randn(2, 128, 431)  # (batch, n_mels, n_frames)
    
    with torch.no_grad():
        conv_output = conv_backbone(dummy_input)
    
    print(f"Conv1D input shape: {dummy_input.shape}")
    print(f"Conv1D output shape: {conv_output.shape}")
    
    # Test Transformer backbone
    print("\n2. Testing Transformer backbone:")
    transformer_config = BackboneConfig(
        type="transformer",
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        input_dim=128,
        output_dim=256
    )
    
    transformer_backbone = create_backbone(transformer_config)
    
    with torch.no_grad():
        transformer_output = transformer_backbone(dummy_input)
    
    print(f"Transformer input shape: {dummy_input.shape}")
    print(f"Transformer output shape: {transformer_output.shape}")
    
    print("\nBackbone implementations test complete!")
    
    # Test with different input sizes (keep n_mels=128 to match config)
    print("\n3. Testing with different frame lengths:")
    test_frame_lengths = [200, 431, 1000]
    
    for n_frames in test_frame_lengths:
        test_input = torch.randn(1, 128, n_frames)  # Keep n_mels=128 to match config
        with torch.no_grad():
            conv_out = conv_backbone(test_input)
            transformer_out = transformer_backbone(test_input)
        
        print(f"Input frames {n_frames} -> Conv1D: {conv_out.shape[1:]}, Transformer: {transformer_out.shape[1:]}")
    
    print("\nSize compatibility test complete!")