import librosa
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.data import AudioConfig


def load_audio(
    file_path: Union[str, Path], 
    config: Optional[AudioConfig] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to audio file
        config: Audio configuration (optional)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if config is None:
        config = AudioConfig()
    
    try:
        # Load audio with target sample rate
        audio, sr = librosa.load(
            file_path if isinstance(file_path, str) else file_path.as_posix(), 
            sr=config.sample_rate, 
            mono=True  # Convert to mono
        )
        
        logging.debug(f"Loaded audio: {file_path}, duration: {len(audio)/sr:.2f}s")
        return audio, sr
        
    except Exception as e:
        logging.error(f"Failed to load audio file {file_path}: {e}")
        raise


def extract_mel_spectrogram(
    audio: np.ndarray, 
    sr: int, 
    config: Optional[AudioConfig] = None
) -> np.ndarray:
    """
    Extract mel spectrogram from audio.
    
    Args:
        audio: Audio time series
        sr: Sample rate
        config: Audio configuration (optional)
        
    Returns:
        Mel spectrogram (n_mels, n_frames)
    """
    if config is None:
        config = AudioConfig()
    
    try:
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            power=2.0
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize if requested
        if config.normalize:
            mel_spec_db = normalize_spectrogram(mel_spec_db)
        
        logging.debug(f"Extracted mel spectrogram: {mel_spec_db.shape}")
        return mel_spec_db
        
    except Exception as e:
        logging.error(f"Failed to extract mel spectrogram: {e}")
        raise


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to [-1, 1] range.
    
    Args:
        spectrogram: Input spectrogram
        
    Returns:
        Normalized spectrogram
    """
    # Normalize to [-1, 1] range
    spec_min = np.min(spectrogram)
    spec_max = np.max(spectrogram)
    
    if spec_max - spec_min > 0:
        normalized = 2 * (spectrogram - spec_min) / (spec_max - spec_min) - 1
    else:
        normalized = np.zeros_like(spectrogram)
    
    return normalized


def audio_to_spectrogram(
    file_path: Union[str, Path], 
    config: Optional[AudioConfig] = None
) -> np.ndarray:
    """
    Complete pipeline: load audio file and convert to mel spectrogram.
    
    Args:
        file_path: Path to audio file
        config: Audio configuration (optional)
        
    Returns:
        Mel spectrogram (n_mels, n_frames)
    """
    # Load audio
    audio, sr = load_audio(file_path, config)
    
    # Extract spectrogram
    spectrogram = extract_mel_spectrogram(audio, sr, config)
    
    return spectrogram


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get duration of audio file in seconds without loading the entire file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        logging.error(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0


def resample_audio(
    audio: np.ndarray, 
    orig_sr: int, 
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Audio time series
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    try:
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        logging.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
        return resampled
    except Exception as e:
        logging.error(f"Failed to resample audio: {e}")
        raise


def pad_audio(
    audio: np.ndarray, 
    target_length: int, 
    mode: str = "constant"
) -> np.ndarray:
    """
    Pad audio to target length.
    
    Args:
        audio: Input audio
        target_length: Target length in samples
        mode: Padding mode ("constant", "edge", "reflect")
        
    Returns:
        Padded audio
    """
    if len(audio) >= target_length:
        return audio[:target_length]
    
    pad_length = target_length - len(audio)
    
    if mode == "constant":
        # Pad with zeros (silence)
        padded = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
    elif mode == "edge":
        # Pad with edge values
        padded = np.pad(audio, (0, pad_length), mode='edge')
    elif mode == "reflect":
        # Pad with reflection
        padded = np.pad(audio, (0, pad_length), mode='reflect')
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
    
    return padded


def trim_audio(
    audio: np.ndarray, 
    start_sample: int, 
    end_sample: Optional[int] = None
) -> np.ndarray:
    """
    Trim audio to specified sample range.
    
    Args:
        audio: Input audio
        start_sample: Start sample index
        end_sample: End sample index (None for end of audio)
        
    Returns:
        Trimmed audio
    """
    if end_sample is None:
        end_sample = len(audio)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample]


def time_to_samples(time_seconds: float, sample_rate: int) -> int:
    """
    Convert time in seconds to sample index.
    
    Args:
        time_seconds: Time in seconds
        sample_rate: Audio sample rate
        
    Returns:
        Sample index
    """
    return int(time_seconds * sample_rate)


def samples_to_time(samples: int, sample_rate: int) -> float:
    """
    Convert sample index to time in seconds.
    
    Args:
        samples: Sample index
        sample_rate: Audio sample rate
        
    Returns:
        Time in seconds
    """
    return samples / sample_rate


def spectrogram_to_tensor(spectrogram: np.ndarray) -> torch.Tensor:
    """
    Convert numpy spectrogram to PyTorch tensor.
    
    Args:
        spectrogram: Numpy spectrogram (n_mels, n_frames)
        
    Returns:
        PyTorch tensor (1, n_mels, n_frames)
    """
    # Add channel dimension and convert to tensor
    tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
    return tensor


def get_audio_stats(file_path: Union[str, Path]) -> dict:
    """
    Get comprehensive statistics about an audio file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio statistics
    """
    try:
        # Load audio for analysis
        audio, sr = librosa.load(file_path, sr=None)
        
        stats = {
            "file_path": str(file_path),
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "n_samples": len(audio),
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "max_amplitude": float(np.max(np.abs(audio))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        }
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        stats.update({
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        })
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to get audio stats for {file_path}: {e}")
        return {"file_path": str(file_path), "error": str(e)}


if __name__ == "__main__":
    # Test audio processing
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy audio for testing
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print("Testing audio processing...")
    
    # Test spectrogram extraction
    config = AudioConfig()
    mel_spec = extract_mel_spectrogram(dummy_audio, sr, config)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Test tensor conversion
    tensor = spectrogram_to_tensor(mel_spec)
    print(f"Tensor shape: {tensor.shape}")
    
    # Test padding and trimming
    padded = pad_audio(dummy_audio, int(sr * 10))  # Pad to 10 seconds
    print(f"Padded audio length: {len(padded)} samples ({len(padded)/sr:.1f}s)")
    
    trimmed = trim_audio(dummy_audio, 0, int(sr * 2))  # Trim to 2 seconds
    print(f"Trimmed audio length: {len(trimmed)} samples ({len(trimmed)/sr:.1f}s)")
    
    print("Audio processing test complete!")