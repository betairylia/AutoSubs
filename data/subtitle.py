import ass
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import re
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.data import LabelConfig


@dataclass
class SubtitleEvent:
    """Represents a single subtitle event."""
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    style: str = "Default"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def parse_ass_file(
    file_path: Union[str, Path], 
    config: Optional[LabelConfig] = None
) -> List[SubtitleEvent]:
    """
    Parse ASS subtitle file and extract timing information.
    
    Args:
        file_path: Path to ASS file
        config: Label configuration (optional)
        
    Returns:
        List of subtitle events
    """
    if config is None:
        config = LabelConfig()
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")
    
    try:
        # Load ASS file
        with open(file_path, encoding='utf-8-sig') as f:
            doc = ass.parse(f)
        
        events = []
        for event in doc.events:
            # Convert timedelta to seconds
            start_seconds = event.start.total_seconds()
            end_seconds = event.end.total_seconds()
            
            # Clean text (remove ASS formatting tags)
            clean_text = clean_ass_text(event.text)
            
            # Skip empty events
            if not clean_text.strip():
                continue
            
            subtitle_event = SubtitleEvent(
                start_time=start_seconds,
                end_time=end_seconds,
                text=clean_text,
                style=event.style
            )
            events.append(subtitle_event)
        
        logging.info(f"Parsed {len(events)} subtitle events from {file_path}")
        return events
        
    except Exception as e:
        logging.error(f"Failed to parse ASS file {file_path}: {e}")
        raise


def clean_ass_text(text: str) -> str:
    """
    Clean ASS text by removing formatting tags.
    
    Args:
        text: Raw ASS text with formatting
        
    Returns:
        Clean text
    """
    # Remove ASS override codes (e.g., {\b1}, {\i1}, etc.)
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Handle line breaks (convert \N to space)
    text = text.replace('\\N', ' ').replace('\\n', ' ')
    
    return text.strip()


def extract_timing_labels(
    events: List[SubtitleEvent],
    audio_duration: float,
    config: Optional[LabelConfig] = None
) -> Tuple[List[float], List[float]]:
    """
    Extract start and end timing labels from subtitle events.
    
    Args:
        events: List of subtitle events
        audio_duration: Total audio duration in seconds
        config: Label configuration (optional)
        
    Returns:
        Tuple of (start_times, end_times)
    """
    if config is None:
        config = LabelConfig()
    
    start_times = []
    end_times = []
    
    # Collect all unique start and end times
    for event in events:
        if 0 <= event.start_time <= audio_duration:
            start_times.append(event.start_time)
        
        if 0 <= event.end_time <= audio_duration:
            end_times.append(event.end_time)
    
    # Merge timestamps that are very close together
    if config.merge_same_timestamp:
        start_times = merge_close_timestamps(start_times, config.timestamp_tolerance)
        end_times = merge_close_timestamps(end_times, config.timestamp_tolerance)
    
    # Sort timestamps
    start_times.sort()
    end_times.sort()
    
    logging.info(f"Extracted {len(start_times)} start times and {len(end_times)} end times")
    return start_times, end_times


def merge_close_timestamps(timestamps: List[float], tolerance: float) -> List[float]:
    """
    Merge timestamps that are within tolerance of each other.
    
    Args:
        timestamps: List of timestamps
        tolerance: Tolerance in seconds
        
    Returns:
        List of merged timestamps
    """
    if not timestamps:
        return []
    
    timestamps = sorted(timestamps)
    merged = [timestamps[0]]
    
    for ts in timestamps[1:]:
        if ts - merged[-1] > tolerance:
            merged.append(ts)
        # If within tolerance, keep the existing timestamp (no change needed)
    
    return merged


def create_positive_pairs(
    events: List[SubtitleEvent],
    config: Optional[LabelConfig] = None
) -> List[Tuple[float, float]]:
    """
    Create positive start-end pairs from subtitle events.
    This is important for training the feature matching component.
    
    Args:
        events: List of subtitle events
        config: Label configuration (optional)
        
    Returns:
        List of (start_time, end_time) pairs
    """
    if config is None:
        config = LabelConfig()
    
    pairs = []
    for event in events:
        pairs.append((event.start_time, event.end_time))
    
    logging.info(f"Created {len(pairs)} positive start-end pairs")
    return pairs


def validate_subtitle_timing(events: List[SubtitleEvent]) -> Dict[str, any]:
    """
    Validate subtitle timing and detect potential issues.
    
    Args:
        events: List of subtitle events
        
    Returns:
        Dictionary with validation results
    """
    issues = {
        "negative_duration": [],
        "overlapping_events": [],
        "very_short_events": [],
        "very_long_events": [],
        "total_events": len(events),
        "valid_events": 0,
    }
    
    # Check each event
    for i, event in enumerate(events):
        is_valid = True
        
        # Check for negative duration
        if event.duration <= 0:
            issues["negative_duration"].append(i)
            is_valid = False
        
        # Check for very short events (< 0.1s)
        elif event.duration < 0.1:
            issues["very_short_events"].append(i)
        
        # Check for very long events (> 30s)
        elif event.duration > 30.0:
            issues["very_long_events"].append(i)
        
        if is_valid:
            issues["valid_events"] += 1
    
    # Check for overlapping events
    sorted_events = sorted(enumerate(events), key=lambda x: x[1].start_time)
    for i in range(len(sorted_events) - 1):
        idx1, event1 = sorted_events[i]
        idx2, event2 = sorted_events[i + 1]
        
        if event1.end_time > event2.start_time:
            issues["overlapping_events"].append((idx1, idx2))
    
    return issues


def get_subtitle_stats(events: List[SubtitleEvent]) -> Dict[str, float]:
    """
    Get statistics about subtitle events.
    
    Args:
        events: List of subtitle events
        
    Returns:
        Dictionary with statistics
    """
    if not events:
        return {"error": "No events provided"}
    
    durations = [event.duration for event in events]
    start_times = [event.start_time for event in events]
    end_times = [event.end_time for event in events]
    
    import numpy as np
    
    stats = {
        "total_events": len(events),
        "total_duration": max(end_times) - min(start_times),
        "subtitle_coverage": sum(durations),
        "avg_event_duration": np.mean(durations),
        "min_event_duration": np.min(durations),
        "max_event_duration": np.max(durations),
        "median_event_duration": np.median(durations),
        "std_event_duration": np.std(durations),
        "first_event_start": min(start_times),
        "last_event_end": max(end_times),
    }
    
    # Calculate coverage percentage
    if stats["total_duration"] > 0:
        stats["coverage_percentage"] = (stats["subtitle_coverage"] / stats["total_duration"]) * 100
    else:
        stats["coverage_percentage"] = 0.0
    
    return stats


def retime_subtitles(
    events: List[SubtitleEvent], 
    time_offset: float
) -> List[SubtitleEvent]:
    """
    Adjust subtitle timing by adding an offset.
    Useful for chunking audio and maintaining correct relative timing.
    
    Args:
        events: List of subtitle events
        time_offset: Offset to add/subtract (in seconds)
        
    Returns:
        List of retimed subtitle events
    """
    retimed_events = []
    
    for event in events:
        new_start = event.start_time - time_offset
        new_end = event.end_time - time_offset
        
        # Only keep events that are still in valid time range (>= 0)
        if new_end > 0:
            retimed_event = SubtitleEvent(
                start_time=max(0, new_start),
                end_time=new_end,
                text=event.text,
                style=event.style
            )
            retimed_events.append(retimed_event)
    
    return retimed_events


def filter_events_in_timerange(
    events: List[SubtitleEvent],
    start_time: float,
    end_time: float
) -> List[SubtitleEvent]:
    """
    Filter subtitle events that overlap with a given time range.
    
    Args:
        events: List of subtitle events
        start_time: Range start time (seconds)
        end_time: Range end time (seconds)
        
    Returns:
        List of events that overlap with the time range
    """
    filtered_events = []
    
    for event in events:
        # Check if event overlaps with the time range
        if (event.start_time < end_time and event.end_time > start_time):
            filtered_events.append(event)
    
    return filtered_events


if __name__ == "__main__":
    # Test subtitle processing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing subtitle processing...")
    
    # Create dummy subtitle events for testing
    dummy_events = [
        SubtitleEvent(0.0, 2.5, "Hello world"),
        SubtitleEvent(3.0, 5.5, "This is a test"),
        SubtitleEvent(6.0, 8.0, "Subtitle processing"),
        SubtitleEvent(9.5, 12.0, "Working correctly"),
    ]
    
    print(f"Created {len(dummy_events)} dummy events")
    
    # Test timing extraction
    config = LabelConfig()
    start_times, end_times = extract_timing_labels(dummy_events, 15.0, config)
    print(f"Extracted start times: {start_times}")
    print(f"Extracted end times: {end_times}")
    
    # Test positive pairs
    pairs = create_positive_pairs(dummy_events, config)
    print(f"Positive pairs: {pairs}")
    
    # Test statistics
    stats = get_subtitle_stats(dummy_events)
    print(f"Statistics: {stats}")
    
    # Test retiming
    retimed = retime_subtitles(dummy_events, 1.0)  # Shift by -1 second
    print(f"Retimed events: {[(e.start_time, e.end_time) for e in retimed]}")
    
    print("Subtitle processing test complete!")