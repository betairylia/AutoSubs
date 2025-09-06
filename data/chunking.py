import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.data import ChunkingConfig, AudioConfig
sys.path.append(str(Path(__file__).parent.parent / "utils"))
import audio as audio_utils
from data.subtitle import SubtitleEvent, retime_subtitles, filter_events_in_timerange, extract_timing_labels


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with associated metadata."""
    chunk_id: int
    audio_data: np.ndarray  # Raw audio samples
    spectrogram: np.ndarray  # Mel spectrogram
    start_time: float  # Original start time in full audio
    end_time: float    # Original end time in full audio
    duration: float    # Chunk duration
    sample_rate: int
    is_padded: bool = False
    padding_info: Optional[Dict] = None


@dataclass  
class SubtitleChunk:
    """Represents subtitle events within a chunk."""
    chunk_id: int
    events: List[SubtitleEvent]  # Retimed events for this chunk
    start_times: List[float]     # Start timestamps within chunk
    end_times: List[float]       # End timestamps within chunk
    positive_pairs: List[Tuple[float, float]]  # Start-end pairs


@dataclass
class DataChunk:
    """Combined audio and subtitle chunk for training."""
    chunk_id: int
    audio_chunk: AudioChunk
    subtitle_chunk: SubtitleChunk
    original_audio_file: str
    original_subtitle_file: str


def chunk_audio_file(
    audio_file: Union[str, Path],
    audio_config: Optional[AudioConfig] = None,
    chunking_config: Optional[ChunkingConfig] = None
) -> List[AudioChunk]:
    """
    Split audio file into overlapping chunks.
    
    Args:
        audio_file: Path to audio file
        audio_config: Audio processing configuration
        chunking_config: Chunking configuration
        
    Returns:
        List of audio chunks
    """
    if audio_config is None:
        audio_config = AudioConfig()
    if chunking_config is None:
        chunking_config = ChunkingConfig()
    
    # Load full audio - use Whisper loading if configured
    if audio_config.use_whisper_preprocessing:
        audio_data, sr = audio_utils.load_audio_whisper(audio_file, audio_config)
    else:
        audio_data, sr = audio_utils.load_audio(audio_file, audio_config)
    total_duration = len(audio_data) / sr
    
    logging.info(f"Chunking audio file {audio_file} (duration: {total_duration:.2f}s)")
    
    # Calculate chunk parameters
    chunk_samples = int(chunking_config.chunk_duration * sr)
    overlap_samples = int(chunk_samples * chunking_config.overlap_ratio)
    step_samples = chunk_samples - overlap_samples
    padding_samples = int(chunking_config.padding_duration * sr)
    
    chunks = []
    chunk_id = 0
    
    # Generate chunks
    start_sample = 0
    while start_sample < len(audio_data):
        end_sample = min(start_sample + chunk_samples, len(audio_data))
        
        # Extract chunk
        chunk_audio = audio_data[start_sample:end_sample]
        
        # Handle padding if needed
        padding_info = None
        is_padded = False
        
        if len(chunk_audio) < chunk_samples:
            # Need padding
            is_padded = True
            
            # Try to pad with surrounding audio first
            if start_sample > 0:
                # Can pad from before
                pre_pad_start = max(0, start_sample - padding_samples)
                pre_pad_audio = audio_data[pre_pad_start:start_sample]
            else:
                pre_pad_audio = np.array([])
            
            if end_sample < len(audio_data):
                # Can pad from after  
                post_pad_end = min(len(audio_data), end_sample + padding_samples)
                post_pad_audio = audio_data[end_sample:post_pad_end]
            else:
                post_pad_audio = np.array([])
            
            # Combine with padding
            padded_chunk = np.concatenate([pre_pad_audio, chunk_audio, post_pad_audio])
            
            # Final padding with silence if still not enough
            if len(padded_chunk) < chunk_samples:
                padded_chunk = audio_utils.pad_audio(padded_chunk, chunk_samples, mode="constant")
            elif len(padded_chunk) > chunk_samples:
                # Trim if too long
                padded_chunk = padded_chunk[:chunk_samples]
            
            chunk_audio = padded_chunk
            padding_info = {
                "pre_pad_samples": len(pre_pad_audio),
                "post_pad_samples": len(post_pad_audio),
                "silence_pad_samples": chunk_samples - len(padded_chunk) + len(pre_pad_audio) + len(post_pad_audio)
            }
        
        # Extract spectrogram - use Whisper preprocessing if configured
        if audio_config.use_whisper_preprocessing:
            spectrogram = audio_utils.extract_mel_spectrogram_whisper(
                chunk_audio, 
                config=audio_config, 
                n_mels=audio_config.n_mels
            )
        else:
            spectrogram = audio_utils.extract_mel_spectrogram(chunk_audio, sr, audio_config)
        
        # Create chunk
        chunk = AudioChunk(
            chunk_id=chunk_id,
            audio_data=chunk_audio,
            spectrogram=spectrogram,
            start_time=start_sample / sr,
            end_time=min(end_sample / sr, total_duration),
            duration=len(chunk_audio) / sr,
            sample_rate=sr,
            is_padded=is_padded,
            padding_info=padding_info
        )
        
        chunks.append(chunk)
        chunk_id += 1
        
        # Move to next chunk
        start_sample += step_samples
        
        # Break if we've covered the entire audio
        if end_sample >= len(audio_data):
            break
    
    logging.info(f"Created {len(chunks)} audio chunks")
    return chunks


def chunk_subtitles(
    subtitle_events: List[SubtitleEvent],
    audio_chunks: List[AudioChunk],
    chunking_config: Optional[ChunkingConfig] = None,
    label_config = None
) -> List[SubtitleChunk]:
    """
    Create subtitle chunks corresponding to audio chunks.
    
    Args:
        subtitle_events: List of subtitle events
        audio_chunks: List of audio chunks
        chunking_config: Chunking configuration
        label_config: Label configuration
        
    Returns:
        List of subtitle chunks
    """
    if chunking_config is None:
        chunking_config = ChunkingConfig()
    
    subtitle_chunks = []
    
    for audio_chunk in audio_chunks:
        # Find events that overlap with this chunk
        chunk_events = filter_events_in_timerange(
            subtitle_events, 
            audio_chunk.start_time, 
            audio_chunk.end_time
        )
        
        # Retime events to be relative to chunk start
        retimed_events = retime_subtitles(chunk_events, audio_chunk.start_time)
        
        # Extract timing labels
        start_times, end_times = extract_timing_labels(
            retimed_events, 
            audio_chunk.duration,
            label_config
        )
        
        # Create positive pairs
        positive_pairs = [(event.start_time, event.end_time) for event in retimed_events]
        
        subtitle_chunk = SubtitleChunk(
            chunk_id=audio_chunk.chunk_id,
            events=retimed_events,
            start_times=start_times,
            end_times=end_times,
            positive_pairs=positive_pairs
        )
        
        subtitle_chunks.append(subtitle_chunk)
    
    logging.info(f"Created {len(subtitle_chunks)} subtitle chunks")
    return subtitle_chunks


def create_data_chunks(
    audio_file: Union[str, Path],
    subtitle_file: Union[str, Path],
    audio_config: Optional[AudioConfig] = None,
    chunking_config: Optional[ChunkingConfig] = None,
    label_config = None
) -> List[DataChunk]:
    """
    Create combined data chunks from audio and subtitle files.
    
    Args:
        audio_file: Path to audio file
        subtitle_file: Path to subtitle file
        audio_config: Audio configuration
        chunking_config: Chunking configuration
        label_config: Label configuration
        
    Returns:
        List of data chunks
    """
    from data.subtitle import parse_ass_file
    
    # Parse subtitle file
    subtitle_events = parse_ass_file(subtitle_file, label_config)
    
    # Create audio chunks
    audio_chunks = chunk_audio_file(audio_file, audio_config, chunking_config)
    
    # Create subtitle chunks
    subtitle_chunks = chunk_subtitles(subtitle_events, audio_chunks, chunking_config, label_config)
    
    # Combine into data chunks
    data_chunks = []
    for audio_chunk, subtitle_chunk in zip(audio_chunks, subtitle_chunks):
        data_chunk = DataChunk(
            chunk_id=audio_chunk.chunk_id,
            audio_chunk=audio_chunk,
            subtitle_chunk=subtitle_chunk,
            original_audio_file=str(audio_file),
            original_subtitle_file=str(subtitle_file)
        )
        data_chunks.append(data_chunk)
    
    logging.info(f"Created {len(data_chunks)} combined data chunks")
    return data_chunks


def validate_chunks(chunks: List[DataChunk]) -> Dict[str, any]:
    """
    Validate data chunks and identify potential issues.
    
    Args:
        chunks: List of data chunks
        
    Returns:
        Validation report
    """
    report = {
        "total_chunks": len(chunks),
        "valid_chunks": 0,
        "empty_subtitle_chunks": 0,
        "padded_chunks": 0,
        "avg_events_per_chunk": 0,
        "avg_spectogram_frames": 0,
        "issues": []
    }
    
    total_events = 0
    total_frames = 0
    
    for chunk in chunks:
        is_valid = True
        
        # Check audio chunk
        if chunk.audio_chunk.spectrogram is None or chunk.audio_chunk.spectrogram.size == 0:
            report["issues"].append(f"Chunk {chunk.chunk_id}: Empty spectrogram")
            is_valid = False
        
        # Check subtitle chunk
        if len(chunk.subtitle_chunk.events) == 0:
            report["empty_subtitle_chunks"] += 1
        
        if chunk.audio_chunk.is_padded:
            report["padded_chunks"] += 1
        
        if is_valid:
            report["valid_chunks"] += 1
            
        total_events += len(chunk.subtitle_chunk.events)
        if chunk.audio_chunk.spectrogram is not None:
            total_frames += chunk.audio_chunk.spectrogram.shape[1]
    
    if len(chunks) > 0:
        report["avg_events_per_chunk"] = total_events / len(chunks)
        report["avg_spectogram_frames"] = total_frames / len(chunks)
    
    return report


def get_chunk_statistics(chunks: List[DataChunk]) -> Dict[str, any]:
    """
    Get comprehensive statistics about data chunks.
    
    Args:
        chunks: List of data chunks
        
    Returns:
        Statistics dictionary
    """
    if not chunks:
        return {"error": "No chunks provided"}
    
    import numpy as np
    
    # Collect metrics
    durations = [chunk.audio_chunk.duration for chunk in chunks]
    event_counts = [len(chunk.subtitle_chunk.events) for chunk in chunks]
    start_time_counts = [len(chunk.subtitle_chunk.start_times) for chunk in chunks]
    end_time_counts = [len(chunk.subtitle_chunk.end_times) for chunk in chunks]
    pair_counts = [len(chunk.subtitle_chunk.positive_pairs) for chunk in chunks]
    
    spectrogram_shapes = [chunk.audio_chunk.spectrogram.shape for chunk in chunks]
    frame_counts = [shape[1] for shape in spectrogram_shapes]
    
    stats = {
        "total_chunks": len(chunks),
        "total_duration": sum(durations),
        "avg_chunk_duration": np.mean(durations),
        "min_chunk_duration": np.min(durations),
        "max_chunk_duration": np.max(durations),
        "avg_events_per_chunk": np.mean(event_counts),
        "avg_start_times_per_chunk": np.mean(start_time_counts),
        "avg_end_times_per_chunk": np.mean(end_time_counts),
        "avg_pairs_per_chunk": np.mean(pair_counts),
        "avg_spectrogram_frames": np.mean(frame_counts),
        "padded_chunks": sum(1 for c in chunks if c.audio_chunk.is_padded),
        "empty_subtitle_chunks": sum(1 for c in chunks if len(c.subtitle_chunk.events) == 0),
    }
    
    return stats


if __name__ == "__main__":
    # Test chunking system
    logging.basicConfig(level=logging.INFO)
    
    print("Testing chunking system...")
    
    # Create dummy data for testing
    sr = 22050
    duration = 30.0  # 30 seconds
    dummy_audio = 0.1 * np.random.randn(int(sr * duration))  # Random noise
    
    dummy_events = [
        SubtitleEvent(2.0, 5.0, "First subtitle"),
        SubtitleEvent(8.0, 12.0, "Second subtitle"),  
        SubtitleEvent(15.0, 18.0, "Third subtitle"),
        SubtitleEvent(22.0, 26.0, "Fourth subtitle"),
    ]
    
    # Test audio chunking (simulate by creating chunks manually)
    from config.data import AudioConfig, ChunkingConfig
    
    audio_config = AudioConfig()
    chunking_config = ChunkingConfig(chunk_duration=10.0, overlap_ratio=0.2)
    
    # Create mock audio chunks
    chunk_duration_samples = int(chunking_config.chunk_duration * sr)
    overlap_samples = int(chunk_duration_samples * chunking_config.overlap_ratio)
    step_samples = chunk_duration_samples - overlap_samples
    
    audio_chunks = []
    chunk_id = 0
    
    for start_sample in range(0, len(dummy_audio), step_samples):
        end_sample = min(start_sample + chunk_duration_samples, len(dummy_audio))
        chunk_audio = dummy_audio[start_sample:end_sample]
        
        if len(chunk_audio) < chunk_duration_samples:
            chunk_audio = audio_utils.pad_audio(chunk_audio, chunk_duration_samples)
        
        spectrogram = audio_utils.extract_mel_spectrogram(chunk_audio, sr, audio_config)
        
        chunk = AudioChunk(
            chunk_id=chunk_id,
            audio_data=chunk_audio,
            spectrogram=spectrogram,
            start_time=start_sample / sr,
            end_time=end_sample / sr,
            duration=len(chunk_audio) / sr,
            sample_rate=sr,
            is_padded=(len(dummy_audio[start_sample:end_sample]) < chunk_duration_samples)
        )
        audio_chunks.append(chunk)
        chunk_id += 1
        
        if end_sample >= len(dummy_audio):
            break
    
    print(f"Created {len(audio_chunks)} audio chunks")
    
    # Test subtitle chunking
    subtitle_chunks = chunk_subtitles(dummy_events, audio_chunks, chunking_config)
    print(f"Created {len(subtitle_chunks)} subtitle chunks")
    
    # Create data chunks
    data_chunks = []
    for ac, sc in zip(audio_chunks, subtitle_chunks):
        dc = DataChunk(
            chunk_id=ac.chunk_id,
            audio_chunk=ac,
            subtitle_chunk=sc,
            original_audio_file="test_audio.wav",
            original_subtitle_file="test_subtitle.ass"
        )
        data_chunks.append(dc)
    
    # Validate and get statistics
    validation = validate_chunks(data_chunks)
    print(f"Validation: {validation}")
    
    stats = get_chunk_statistics(data_chunks)
    print(f"Statistics: {stats}")
    
    print("Chunking system test complete!")