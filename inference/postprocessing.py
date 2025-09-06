import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.model import ModelConfig


def non_maximum_suppression_1d(
    heatmap: torch.Tensor, 
    window_size: int = 5,
    threshold: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Apply Non-Maximum Suppression to 1D heatmap to find local peaks.
    
    Args:
        heatmap: 1D tensor of confidence values (n_frames,)
        window_size: Window size for NMS
        threshold: Minimum confidence threshold
        
    Returns:
        List of (frame_index, confidence) tuples for detected peaks
    """
    if heatmap.dim() != 1:
        raise ValueError("Expected 1D heatmap")
    
    # Apply threshold
    above_threshold = heatmap > threshold
    if not above_threshold.any():
        return []
    
    # Apply max pooling for NMS
    padded_heatmap = F.pad(heatmap.unsqueeze(0).unsqueeze(0), 
                          (window_size//2, window_size//2), mode='constant', value=0)
    max_pooled = F.max_pool1d(padded_heatmap, kernel_size=window_size, stride=1, padding=0)
    max_pooled = max_pooled.squeeze()
    
    # Find local maxima
    is_peak = (heatmap == max_pooled) & above_threshold
    peak_indices = torch.nonzero(is_peak, as_tuple=True)[0]
    peak_confidences = heatmap[peak_indices]
    
    # Convert to list of tuples
    peaks = [(idx.item(), conf.item()) for idx, conf in zip(peak_indices, peak_confidences)]
    
    # Sort by confidence (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks


def scipy_peak_detection(
    heatmap: np.ndarray,
    height: float = 0.5,
    distance: int = 5,
    prominence: float = 0.1
) -> List[Tuple[int, float]]:
    """
    Use scipy's find_peaks for more sophisticated peak detection.
    
    Args:
        heatmap: 1D numpy array of confidence values
        height: Minimum height of peaks
        distance: Minimum distance between peaks
        prominence: Minimum prominence of peaks
        
    Returns:
        List of (frame_index, confidence) tuples
    """
    peaks, properties = find_peaks(
        heatmap, 
        height=height, 
        distance=distance,
        prominence=prominence
    )
    
    peak_confidences = heatmap[peaks]
    peak_list = [(int(idx), float(conf)) for idx, conf in zip(peaks, peak_confidences)]
    
    # Sort by confidence (descending)
    peak_list.sort(key=lambda x: x[1], reverse=True)
    
    return peak_list


def match_start_end_pairs(
    start_peaks: List[Tuple[int, float]],
    end_peaks: List[Tuple[int, float]],
    start_features: torch.Tensor,
    end_features: torch.Tensor,
    config: ModelConfig,
    timing_fps: int = 60
) -> List[Dict]:
    """
    Match start and end peaks using feature similarity and temporal constraints.
    
    Args:
        start_peaks: List of (frame_idx, confidence) for start events
        end_peaks: List of (frame_idx, confidence) for end events  
        start_features: Feature vectors at all frames (n_frames, feature_dim)
        end_features: Feature vectors at all frames (n_frames, feature_dim)
        config: Model configuration
        timing_fps: Timing frames per second
        
    Returns:
        List of matched subtitle events
    """
    if not start_peaks or not end_peaks:
        return []
    
    matched_pairs = []
    used_starts = set()
    used_ends = set()
    
    # Convert max subtitle length to frames
    max_length_frames = int(config.max_subtitle_length * timing_fps)
    
    # Try to match each start with the best end
    for start_idx, start_conf in start_peaks:
        if start_idx in used_starts:
            continue
            
        best_end = None
        best_score = float('inf')
        
        start_feature = start_features[start_idx]
        
        for end_idx, end_conf in end_peaks:
            if end_idx in used_ends:
                continue
            
            # Temporal constraint: end must come after start
            if end_idx <= start_idx:
                continue
            
            # Duration constraint
            duration_frames = end_idx - start_idx
            if duration_frames > max_length_frames:
                continue
            
            # Feature similarity (lower distance = better match)
            end_feature = end_features[end_idx]
            feature_distance = F.pairwise_distance(
                start_feature.unsqueeze(0), 
                end_feature.unsqueeze(0), 
                p=2
            ).item()
            
            # Combined score (lower is better)
            # Weight by confidence and feature similarity
            confidence_score = 2.0 - (start_conf + end_conf)  # Lower confidence = higher score
            total_score = feature_distance + 0.1 * confidence_score
            
            if total_score < best_score and feature_distance < config.feature_distance_threshold:
                best_score = total_score
                best_end = (end_idx, end_conf, feature_distance)
        
        # If we found a good match, add it
        if best_end is not None:
            end_idx, end_conf, feature_dist = best_end
            
            # Convert frame indices to time
            start_time = start_idx / timing_fps
            end_time = end_idx / timing_fps
            
            matched_pairs.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "start_confidence": start_conf,
                "end_confidence": end_conf,
                "feature_distance": feature_dist,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "match_score": best_score
            })
            
            used_starts.add(start_idx)
            used_ends.add(end_idx)
    
    # Sort by confidence (average of start and end)
    matched_pairs.sort(key=lambda x: (x["start_confidence"] + x["end_confidence"]) / 2, reverse=True)
    
    return matched_pairs


def filter_overlapping_pairs(pairs: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
    """
    Filter out overlapping subtitle pairs, keeping higher confidence ones.
    
    Args:
        pairs: List of matched pairs
        overlap_threshold: Maximum allowed overlap ratio
        
    Returns:
        Filtered list of pairs
    """
    if len(pairs) <= 1:
        return pairs
    
    # Sort by average confidence (descending)
    sorted_pairs = sorted(pairs, key=lambda x: (x["start_confidence"] + x["end_confidence"]) / 2, reverse=True)
    
    filtered_pairs = []
    
    for current_pair in sorted_pairs:
        current_start = current_pair["start_time"]
        current_end = current_pair["end_time"]
        current_duration = current_pair["duration"]
        
        # Check overlap with already accepted pairs
        is_overlapping = False
        
        for accepted_pair in filtered_pairs:
            accepted_start = accepted_pair["start_time"]
            accepted_end = accepted_pair["end_time"]
            
            # Calculate overlap
            overlap_start = max(current_start, accepted_start)
            overlap_end = min(current_end, accepted_end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlap_ratio = overlap_duration / min(current_duration, accepted_pair["duration"])
                
                if overlap_ratio > overlap_threshold:
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            filtered_pairs.append(current_pair)
    
    # Sort by start time
    filtered_pairs.sort(key=lambda x: x["start_time"])
    
    return filtered_pairs


def merge_nearby_events(pairs: List[Dict], merge_threshold: float = 0.5) -> List[Dict]:
    """
    Merge subtitle events that are very close to each other.
    
    Args:
        pairs: List of matched pairs
        merge_threshold: Time threshold for merging (seconds)
        
    Returns:
        List with merged pairs
    """
    if len(pairs) <= 1:
        return pairs
    
    # Sort by start time
    sorted_pairs = sorted(pairs, key=lambda x: x["start_time"])
    merged_pairs = []
    
    current_pair = sorted_pairs[0].copy()
    
    for next_pair in sorted_pairs[1:]:
        # Check if next pair is close enough to merge
        gap = next_pair["start_time"] - current_pair["end_time"]
        
        if gap <= merge_threshold:
            # Merge pairs
            current_pair["end_time"] = next_pair["end_time"]
            current_pair["duration"] = current_pair["end_time"] - current_pair["start_time"]
            current_pair["end_confidence"] = max(current_pair["end_confidence"], next_pair["end_confidence"])
            current_pair["end_frame"] = next_pair["end_frame"]
        else:
            # Save current pair and start new one
            merged_pairs.append(current_pair)
            current_pair = next_pair.copy()
    
    # Add the last pair
    merged_pairs.append(current_pair)
    
    return merged_pairs


def apply_confidence_threshold(pairs: List[Dict], min_confidence: float = 0.3) -> List[Dict]:
    """
    Filter pairs based on minimum confidence threshold.
    
    Args:
        pairs: List of matched pairs
        min_confidence: Minimum average confidence
        
    Returns:
        Filtered pairs
    """
    return [
        pair for pair in pairs 
        if (pair["start_confidence"] + pair["end_confidence"]) / 2 >= min_confidence
    ]


class InferencePostProcessor:
    """Main post-processing class for inference."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def process_predictions(
        self,
        start_heatmap: torch.Tensor,
        end_heatmap: torch.Tensor,
        start_features: torch.Tensor,
        end_features: torch.Tensor,
        timing_fps: int = 60,
        use_scipy_peaks: bool = True
    ) -> List[Dict]:
        """
        Complete post-processing pipeline.
        
        Args:
            start_heatmap: Start confidence heatmap (n_frames,)
            end_heatmap: End confidence heatmap (n_frames,)
            start_features: Start features (n_frames, feature_dim)
            end_features: End features (n_frames, feature_dim)
            timing_fps: Timing frames per second
            use_scipy_peaks: Whether to use scipy peak detection
            
        Returns:
            List of processed subtitle events
        """
        # Convert to numpy if needed
        if isinstance(start_heatmap, torch.Tensor):
            start_heatmap_np = start_heatmap.cpu().numpy()
            end_heatmap_np = end_heatmap.cpu().numpy()
        else:
            start_heatmap_np = start_heatmap
            end_heatmap_np = end_heatmap
        
        # Peak detection
        if use_scipy_peaks:
            start_peaks = scipy_peak_detection(
                start_heatmap_np, 
                height=self.config.confidence_threshold,
                distance=self.config.nmp_window_size,
                prominence=0.1
            )
            end_peaks = scipy_peak_detection(
                end_heatmap_np,
                height=self.config.confidence_threshold, 
                distance=self.config.nmp_window_size,
                prominence=0.1
            )
        else:
            start_peaks = non_maximum_suppression_1d(
                start_heatmap if isinstance(start_heatmap, torch.Tensor) else torch.from_numpy(start_heatmap),
                window_size=self.config.nmp_window_size,
                threshold=self.config.confidence_threshold
            )
            end_peaks = non_maximum_suppression_1d(
                end_heatmap if isinstance(end_heatmap, torch.Tensor) else torch.from_numpy(end_heatmap),
                window_size=self.config.nmp_window_size,
                threshold=self.config.confidence_threshold
            )
        
        if not start_peaks or not end_peaks:
            return []
        
        # Feature matching
        matched_pairs = match_start_end_pairs(
            start_peaks, end_peaks,
            start_features, end_features,
            self.config, timing_fps
        )
        
        if not matched_pairs:
            return []
        
        # Post-processing filters
        filtered_pairs = apply_confidence_threshold(matched_pairs, min_confidence=0.2)
        filtered_pairs = filter_overlapping_pairs(filtered_pairs, overlap_threshold=0.3)
        filtered_pairs = merge_nearby_events(filtered_pairs, merge_threshold=0.5)
        
        return filtered_pairs


if __name__ == "__main__":
    # Test post-processing components
    print("Testing inference post-processing...")
    
    from config.model import ModelConfig, BackboneConfig, HeadConfig
    
    # Create test configuration
    config = ModelConfig(
        backbone=BackboneConfig(),
        head=HeadConfig(),
        nmp_window_size=5,
        confidence_threshold=0.5,
        feature_distance_threshold=2.0,
        max_subtitle_length=10.0
    )
    
    print(f"Config: NMS window={config.nmp_window_size}, threshold={config.confidence_threshold}")
    
    # Create test data
    n_frames = 1800  # 30 seconds at 60 FPS
    feature_dim = 128
    
    # Create synthetic heatmaps with some peaks
    start_heatmap = torch.zeros(n_frames)
    end_heatmap = torch.zeros(n_frames)
    
    # Add some synthetic events
    events = [
        (120, 300),   # 2-5 seconds
        (600, 900),   # 10-15 seconds  
        (1200, 1500), # 20-25 seconds
    ]
    
    for start_idx, end_idx in events:
        # Add Gaussian peaks
        start_peak = torch.exp(-((torch.arange(n_frames) - start_idx) ** 2) / (2 * 10 ** 2))
        end_peak = torch.exp(-((torch.arange(n_frames) - end_idx) ** 2) / (2 * 10 ** 2))
        
        start_heatmap += 0.8 * start_peak
        end_heatmap += 0.8 * end_peak
    
    # Add some noise
    start_heatmap += 0.1 * torch.randn(n_frames)
    end_heatmap += 0.1 * torch.randn(n_frames)
    start_heatmap = torch.clamp(start_heatmap, 0, 1)
    end_heatmap = torch.clamp(end_heatmap, 0, 1)
    
    # Create synthetic features (correlated for matching pairs)
    start_features = torch.randn(n_frames, feature_dim)
    end_features = torch.randn(n_frames, feature_dim)
    
    # Make features more similar for true pairs
    for start_idx, end_idx in events:
        shared_feature = torch.randn(feature_dim)
        start_features[start_idx] = 0.7 * shared_feature + 0.3 * start_features[start_idx]
        end_features[end_idx] = 0.7 * shared_feature + 0.3 * end_features[end_idx]
    
    # Normalize features
    start_features = F.normalize(start_features, p=2, dim=1)
    end_features = F.normalize(end_features, p=2, dim=1)
    
    print(f"Test data created: {n_frames} frames, {len(events)} ground truth events")
    
    # Test NMS
    print("\n1. Testing Non-Maximum Suppression:")
    start_peaks = non_maximum_suppression_1d(start_heatmap, window_size=5, threshold=0.5)
    end_peaks = non_maximum_suppression_1d(end_heatmap, window_size=5, threshold=0.5)
    
    print(f"Start peaks found: {len(start_peaks)}")
    print(f"End peaks found: {len(end_peaks)}")
    for i, (idx, conf) in enumerate(start_peaks[:3]):
        print(f"  Start peak {i}: frame {idx} ({idx/60:.1f}s), confidence {conf:.3f}")
    
    # Test scipy peaks
    print("\n2. Testing scipy peak detection:")
    start_peaks_scipy = scipy_peak_detection(start_heatmap.numpy(), height=0.5, distance=5)
    end_peaks_scipy = scipy_peak_detection(end_heatmap.numpy(), height=0.5, distance=5)
    
    print(f"Scipy - Start peaks: {len(start_peaks_scipy)}, End peaks: {len(end_peaks_scipy)}")
    
    # Test feature matching
    print("\n3. Testing feature matching:")
    matched_pairs = match_start_end_pairs(
        start_peaks, end_peaks,
        start_features, end_features,
        config, timing_fps=60
    )
    
    print(f"Matched pairs: {len(matched_pairs)}")
    for i, pair in enumerate(matched_pairs):
        print(f"  Pair {i}: {pair['start_time']:.1f}s - {pair['end_time']:.1f}s "
              f"(conf: {pair['start_confidence']:.3f}, {pair['end_confidence']:.3f}, "
              f"dist: {pair['feature_distance']:.3f})")
    
    # Test complete post-processor
    print("\n4. Testing complete post-processor:")
    processor = InferencePostProcessor(config)
    
    final_pairs = processor.process_predictions(
        start_heatmap, end_heatmap,
        start_features, end_features,
        timing_fps=60
    )
    
    print(f"Final processed pairs: {len(final_pairs)}")
    for i, pair in enumerate(final_pairs):
        print(f"  Final {i}: {pair['start_time']:.1f}s - {pair['end_time']:.1f}s "
              f"(duration: {pair['duration']:.1f}s)")
    
    print("\nPost-processing test completed successfully!")