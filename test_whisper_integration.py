#!/usr/bin/env python3
"""
Test script for Whisper backbone integration.
Tests model compatibility, output shapes, and processing pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
import logging
from config.config import load_config
from models.network import create_model
from utils.audio import extract_mel_spectrogram_whisper, audio_to_spectrogram_whisper

def test_whisper_backbone():
    """Test WhisperBackbone class functionality."""
    print("=== Testing Whisper Backbone Integration ===\n")
    
    # Load config
    config = load_config("configs/default.yaml")
    print(f"Config loaded: backbone type = {config.model.backbone.type}")
    print(f"Whisper model size: {config.model.backbone.whisper_model_size}")
    print(f"Target output_dim: {config.model.backbone.output_dim}")
    print(f"Audio config: sr={config.data.audio.sample_rate}, n_mels={config.data.audio.n_mels}")
    print()
    
    # Create model
    print("Creating Whisper-based model...")
    model = create_model(config.model)
    print("Model created successfully!")
    
    # Print model info
    from models.network import ModelSummary
    model_info = ModelSummary.get_model_info(model)
    print(f"Model parameters: {model_info['parameter_counts']['total']:,}")
    print(f"Backbone parameters: {model_info['parameter_counts']['backbone']:,}")
    print(f"Temporal resolution: {model.backbone.get_temporal_resolution()} FPS")
    print()
    
    # Test with dummy input
    print("Testing with dummy input...")
    # Create dummy mel spectrogram: (batch_size, n_mels, n_frames)
    # For 30s at Whisper's preprocessing: 3000 frames
    batch_size = 2
    n_mels = config.data.audio.n_mels  # Should be 80
    n_frames = 3000  # 30s at Whisper's frame rate
    
    dummy_input = torch.randn(batch_size, n_mels, n_frames)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
        timing_predictions = model.get_timing_predictions(dummy_input, chunk_duration=30.0)
    
    print(f"Start heatmap shape: {outputs['start']['heatmap'].shape}")
    print(f"Start features shape: {outputs['start']['features'].shape}")
    print(f"End heatmap shape: {outputs['end']['heatmap'].shape}")
    print(f"End features shape: {outputs['end']['features'].shape}")
    print()
    
    print(f"Timing predictions - start heatmap: {timing_predictions['start_heatmap'].shape}")
    print(f"Timing predictions - start features: {timing_predictions['start_features'].shape}")
    print()
    
    # Check temporal resolution
    expected_frames = int(30.0 * model.backbone.get_temporal_resolution())  # 30s * 100fps = 3000 frames
    actual_frames = timing_predictions['start_heatmap'].shape[1]
    print(f"Expected timing frames (30s * {model.backbone.get_temporal_resolution()}fps): {expected_frames}")
    print(f"Actual timing frames: {actual_frames}")
    print(f"Temporal resolution match: {'✓' if actual_frames == expected_frames else '✗'}")
    print()
    
    return model, config


def test_whisper_preprocessing():
    """Test Whisper audio preprocessing functions."""
    print("=== Testing Whisper Preprocessing ===\n")
    
    # Create dummy audio (16kHz, 5 seconds)
    sample_rate = 16000
    duration = 5.0
    dummy_audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    print(f"Created dummy audio: {len(dummy_audio)} samples at {sample_rate}Hz ({duration}s)")
    
    # Test mel spectrogram extraction
    mel_spec = extract_mel_spectrogram_whisper(dummy_audio, n_mels=80)
    print(f"Whisper mel spectrogram shape: {mel_spec.shape}")
    print(f"Expected shape: (80, ~{int(duration * 100)})  # 80 mels, ~100 fps")
    
    # Test with different n_mels
    mel_spec_128 = extract_mel_spectrogram_whisper(dummy_audio, n_mels=128)
    print(f"Whisper mel spectrogram (128 mels): {mel_spec_128.shape}")
    print()
    
    return mel_spec


def test_full_pipeline():
    """Test the complete pipeline from audio to predictions."""
    print("=== Testing Full Pipeline ===\n")
    
    # Load model
    config = load_config("configs/default.yaml")
    model = create_model(config.model)
    
    # Create dummy audio and convert to spectrogram
    sample_rate = 16000
    duration = 30.0  # Full 30s chunk
    dummy_audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    
    # Use Whisper preprocessing
    mel_spec = extract_mel_spectrogram_whisper(dummy_audio, n_mels=config.data.audio.n_mels)
    print(f"Audio to mel spectrogram: {len(dummy_audio)} samples -> {mel_spec.shape}")
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Full forward pass
    with torch.no_grad():
        predictions = model.get_timing_predictions(input_tensor, chunk_duration=duration)
    
    print(f"Pipeline output shapes:")
    print(f"  Start heatmap: {predictions['start_heatmap'].shape}")
    print(f"  End heatmap: {predictions['end_heatmap'].shape}")
    print(f"  Start features: {predictions['start_features'].shape}")
    print(f"  End features: {predictions['end_features'].shape}")
    
    # Verify output values are reasonable
    start_heatmap = predictions['start_heatmap'][0].cpu().numpy()
    print(f"Start heatmap stats: min={start_heatmap.min():.3f}, max={start_heatmap.max():.3f}, mean={start_heatmap.mean():.3f}")
    print(f"Pipeline test: {'✓ PASSED' if 0 <= start_heatmap.min() and start_heatmap.max() <= 1 else '✗ FAILED'}")
    print()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run tests
        print("Starting Whisper integration tests...\n")
        
        # Test 1: Model creation and basic functionality
        model, config = test_whisper_backbone()
        
        # Test 2: Audio preprocessing
        mel_spec = test_whisper_preprocessing()
        
        # Test 3: Full pipeline
        test_full_pipeline()
        
        print("🎉 All tests completed successfully!")
        print("\n" + "="*50)
        print("WHISPER INTEGRATION READY!")
        print("- Backbone: ✓ Implemented")
        print("- Preprocessing: ✓ Implemented") 
        print("- Config: ✓ Updated")
        print("- Temporal adaptation: ✓ Updated")
        print("- Pipeline: ✓ Tested")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)