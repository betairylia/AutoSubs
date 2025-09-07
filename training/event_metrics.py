import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def compute_event_iou_matrix(pred_events: torch.Tensor, gt_events: torch.Tensor) -> torch.Tensor:
    """
    Vectorized IoU computation using broadcasting.
    
    Args:
        pred_events: (N, 2) tensor of [start_time, end_time] for predicted events
        gt_events: (M, 2) tensor of [start_time, end_time] for ground truth events
        
    Returns:
        (N, M) IoU matrix where iou[i,j] is IoU between pred_events[i] and gt_events[j]
    """
    if pred_events.size(0) == 0 or gt_events.size(0) == 0:
        return torch.zeros(pred_events.size(0), gt_events.size(0), 
                          device=pred_events.device, dtype=pred_events.dtype)
    
    # Broadcasting: pred (N,1,2) vs gt (1,M,2) 
    pred_expanded = pred_events.unsqueeze(1)  # (N, 1, 2)
    gt_expanded = gt_events.unsqueeze(0)      # (1, M, 2)
    
    # Vectorized intersection calculation
    intersection_start = torch.max(pred_expanded[..., 0], gt_expanded[..., 0])  # (N, M)
    intersection_end = torch.min(pred_expanded[..., 1], gt_expanded[..., 1])    # (N, M)
    intersection = torch.clamp(intersection_end - intersection_start, min=0.0)  # (N, M)
    
    # Vectorized union calculation
    union_start = torch.min(pred_expanded[..., 0], gt_expanded[..., 0])  # (N, M)
    union_end = torch.max(pred_expanded[..., 1], gt_expanded[..., 1])    # (N, M)
    union = union_end - union_start  # (N, M)
    
    # Compute IoU with numerical stability
    iou_matrix = intersection / (union + 1e-7)
    
    return iou_matrix


def extract_gt_events_from_pairs(
    positive_pairs: torch.Tensor, 
    timing_fps: float
) -> torch.Tensor:
    """
    Extract ground truth events from positive_pairs tensor.
    
    Args:
        positive_pairs: (batch_size, max_pairs, 2) tensor of [start_frame, end_frame] indices
        timing_fps: Frames per second for timing conversion
        
    Returns:
        (total_events, 2) tensor of [start_time, end_time] for all valid GT events
    """
    # Flatten across batch dimension
    pairs_flat = positive_pairs.view(-1, 2)  # (batch_size * max_pairs, 2)
    
    # Filter out padding (negative indices)
    valid_mask = (pairs_flat[:, 0] >= 0) & (pairs_flat[:, 1] >= 0)
    valid_pairs = pairs_flat[valid_mask]  # (n_valid_pairs, 2)
    
    if valid_pairs.size(0) == 0:
        return torch.empty(0, 2, device=positive_pairs.device, dtype=torch.float32)
    
    # Convert frame indices to time (vectorized)
    gt_events = valid_pairs.float() / timing_fps  # (n_valid_pairs, 2)
    
    return gt_events


def extract_predicted_events_batch(
    outputs_batch: List[Dict[str, torch.Tensor]], 
    config, 
    timing_fps: float
) -> torch.Tensor:
    """
    Extract predicted events from batch of model outputs using post-processing.
    
    Args:
        outputs_batch: List of model outputs for each item in batch
        config: Model configuration  
        timing_fps: Frames per second for timing
        
    Returns:
        (total_pred_events, 2) tensor of [start_time, end_time] for all predicted events
    """
    from inference.postprocessing import InferencePostProcessor
    
    processor = InferencePostProcessor(config)
    all_pred_events = []
    
    for outputs in outputs_batch:
        # Extract predictions from single item
        start_heatmap = outputs["start_heatmap"].squeeze(0)  # Remove batch dim
        end_heatmap = outputs["end_heatmap"].squeeze(0)
        start_features = outputs["start_features"].squeeze(0) 
        end_features = outputs["end_features"].squeeze(0)
        
        # Process predictions
        events = processor.process_predictions(
            start_heatmap, end_heatmap, start_features, end_features, timing_fps
        )
        
        # Convert to tensor format
        for event in events:
            all_pred_events.append([event["start_time"], event["end_time"]])
    
    if not all_pred_events:
        return torch.empty(0, 2, device=outputs_batch[0]["start_heatmap"].device, dtype=torch.float32)
    
    return torch.tensor(all_pred_events, device=outputs_batch[0]["start_heatmap"].device, dtype=torch.float32)


def event_level_iou_metric(
    pred_events: torch.Tensor, 
    gt_events: torch.Tensor
) -> float:
    """
    Compute event-level IoU metric using optimal Hungarian matching.
    
    Args:
        pred_events: (N, 2) tensor of predicted [start_time, end_time]
        gt_events: (M, 2) tensor of ground truth [start_time, end_time]
        
    Returns:
        Event-level IoU score (0.0 to 1.0)
    """
    if pred_events.size(0) == 0 and gt_events.size(0) == 0:
        return 1.0  # Perfect match when both empty
    
    if pred_events.size(0) == 0 or gt_events.size(0) == 0:
        return 0.0  # No match when one is empty
    
    # Compute pairwise IoU matrix
    iou_matrix = compute_event_iou_matrix(pred_events, gt_events)  # (N, M)
    
    # Convert to cost matrix for Hungarian algorithm (minimize cost = maximize IoU)
    cost_matrix = 1.0 - iou_matrix.cpu().numpy()  # (N, M)
    
    # Find optimal assignment using Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Compute total IoU for matched pairs
    matched_ious = iou_matrix[pred_indices, gt_indices]  # (min(N,M),)
    total_matched_iou = matched_ious.sum().item()
    
    # Account for unmatched events (they contribute 0 IoU)
    total_events = max(pred_events.size(0), gt_events.size(0))
    
    # Average IoU across all events (matched + unmatched)
    event_iou = total_matched_iou / total_events
    
    return event_iou


def compute_event_iou_for_batch(
    outputs_batch: List[Dict[str, torch.Tensor]],
    batch_data: Dict[str, torch.Tensor],
    config,
    timing_fps: float
) -> float:
    """
    Compute event-level IoU for a complete batch.
    
    Args:
        outputs_batch: List of model outputs (one per batch item)
        batch_data: Batch data containing positive_pairs
        config: Model configuration
        timing_fps: Frames per second for timing
        
    Returns:
        Average event-level IoU for the batch
    """
    # Extract ground truth events from batch
    gt_events = extract_gt_events_from_pairs(
        batch_data["positive_pairs"], timing_fps
    )
    
    # Extract predicted events from batch outputs
    pred_events = extract_predicted_events_batch(
        outputs_batch, config, timing_fps
    )
    
    # Compute event-level IoU
    return event_level_iou_metric(pred_events, gt_events)


if __name__ == "__main__":
    # Test the event IoU metric
    print("Testing event-level IoU metric...")
    
    # Test IoU matrix computation
    print("\n1. Testing IoU matrix computation:")
    pred_events = torch.tensor([
        [1.0, 3.0],  # Event 1: 1-3s
        [5.0, 8.0],  # Event 2: 5-8s  
        [10.0, 12.0] # Event 3: 10-12s
    ])
    
    gt_events = torch.tensor([
        [0.5, 2.5],  # GT 1: 0.5-2.5s (partial overlap with pred 1)
        [5.5, 7.5],  # GT 2: 5.5-7.5s (good overlap with pred 2)
        [15.0, 17.0] # GT 3: 15-17s (no overlap)
    ])
    
    iou_matrix = compute_event_iou_matrix(pred_events, gt_events)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU matrix:\n{iou_matrix}")
    
    # Expected: 
    # [0,0] should be ~0.5 (1-2.5 intersect, 0.5-3 union)
    # [1,1] should be ~0.67 (5.5-7.5 intersect, 5-8 union) 
    # [2,2] should be 0 (no overlap)
    
    # Test event-level IoU
    print("\n2. Testing event-level IoU:")
    event_iou = event_level_iou_metric(pred_events, gt_events)
    print(f"Event-level IoU: {event_iou:.4f}")
    
    # Test with perfect match
    print("\n3. Testing perfect match:")
    perfect_iou = event_level_iou_metric(gt_events, gt_events)
    print(f"Perfect match IoU: {perfect_iou:.4f}")
    
    # Test with no predictions
    print("\n4. Testing empty predictions:")
    empty_pred = torch.empty(0, 2)
    empty_iou = event_level_iou_metric(empty_pred, gt_events)
    print(f"Empty predictions IoU: {empty_iou:.4f}")
    
    # Test GT extraction
    print("\n5. Testing GT extraction:")
    positive_pairs = torch.tensor([
        [[60, 180], [300, 480]],   # Batch 0: 2 events
        [[120, 240], [-1, -1]]     # Batch 1: 1 event + padding
    ])  # Shape: (2, 2, 2)
    
    extracted_gt = extract_gt_events_from_pairs(positive_pairs, timing_fps=60.0)
    print(f"Extracted GT events:\n{extracted_gt}")
    print(f"Expected: [[1.0, 3.0], [5.0, 8.0], [2.0, 4.0]]")
    
    print("\nEvent IoU metric tests completed successfully!")