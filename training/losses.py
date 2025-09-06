import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.train import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in heatmap prediction.
    
    The focal loss is designed to down-weight easy examples and focus on hard negatives.
    This is crucial for subtitle timing where most time points are negative (no subtitle events).
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Predicted probabilities (batch_size, n_frames)
            targets: Target labels (batch_size, n_frames) with values in [0, 1]
            
        Returns:
            Focal loss value
        """
        # Clamp predictions to avoid numerical instability
        predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)
        
        # Compute cross entropy
        ce_loss = -targets * torch.log(predictions) - (1 - targets) * torch.log(1 - predictions)
        
        # Compute focal weights
        pt = targets * predictions + (1 - targets) * (1 - predictions)  # pt = probability of correct class
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting (weight positive examples)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveFeatureLoss(nn.Module):
    """
    Contrastive loss for feature matching between start-end pairs.
    
    This loss pulls together features from positive start-end pairs and pushes apart
    features from negative pairs (mismatched start-end combinations).
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        start_features: torch.Tensor,
        end_features: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive feature loss.
        
        Args:
            start_features: Start features (batch_size, n_frames, feature_dim)
            end_features: End features (batch_size, n_frames, feature_dim)
            positive_pairs: Positive pairs indices (batch_size, n_pairs, 2)
            negative_pairs: Negative pairs indices (optional, will be generated if None)
            
        Returns:
            Contrastive loss value
        """
        batch_size, n_frames, feature_dim = start_features.shape
        device = start_features.device
        
        # Create mask for valid pairs (not -1 padding)
        valid_mask = (positive_pairs[:, :, 0] >= 0) & (positive_pairs[:, :, 1] >= 0)
        
        if not valid_mask.any():
            return torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        # Clamp indices to valid range
        start_indices = torch.clamp(positive_pairs[:, :, 0], 0, n_frames - 1)
        end_indices = torch.clamp(positive_pairs[:, :, 1], 0, n_frames - 1)
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, positive_pairs.size(1))
        
        # Gather features using advanced indexing - only for valid pairs
        # Flatten for gathering
        flat_batch_indices = batch_indices[valid_mask]
        flat_start_indices = start_indices[valid_mask]
        flat_end_indices = end_indices[valid_mask]
        
        if len(flat_batch_indices) == 0:
            return torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        pos_start_feats = start_features[flat_batch_indices, flat_start_indices]  # (n_valid_pairs, feature_dim)
        pos_end_feats = end_features[flat_batch_indices, flat_end_indices]
        
        # Positive loss (pull together) - L2 distance
        pos_distances = torch.norm(pos_start_feats - pos_end_feats, p=2, dim=1)
        pos_loss = pos_distances.mean()
        
        # Generate negative pairs efficiently using broadcasting
        neg_loss = torch.tensor(0.0, device=device)
        n_pos = len(pos_start_feats)
        
        if n_pos > 1:
            # Create all pairwise combinations efficiently
            # pos_start_feats: (n_pos, feature_dim) -> (n_pos, 1, feature_dim)
            # pos_end_feats: (n_pos, feature_dim) -> (1, n_pos, feature_dim)
            start_expanded = pos_start_feats.unsqueeze(1)  # (n_pos, 1, feature_dim)
            end_expanded = pos_end_feats.unsqueeze(0)      # (1, n_pos, feature_dim)
            
            # Compute all pairwise distances at once
            neg_distances = torch.norm(start_expanded - end_expanded, p=2, dim=2)  # (n_pos, n_pos)
            
            # Create mask to exclude diagonal (i != j)
            mask = ~torch.eye(n_pos, device=device, dtype=torch.bool)
            
            # Apply margin and clamp
            valid_neg_distances = neg_distances[mask]
            neg_loss = torch.clamp(self.margin - valid_neg_distances, min=0.0).mean()
        
        return pos_loss + neg_loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for feature matching (alternative to contrastive loss).
    Uses temperature scaling for better training dynamics.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        start_features: torch.Tensor,
        end_features: torch.Tensor, 
        positive_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for feature matching.
        
        Args:
            start_features: Start features (batch_size, n_frames, feature_dim)
            end_features: End features (batch_size, n_frames, feature_dim)
            positive_pairs: Positive pairs indices (batch_size, n_pairs, 2)
            
        Returns:
            InfoNCE loss value
        """
        batch_size, n_frames, feature_dim = start_features.shape
        device = start_features.device
        
        # Create mask for valid pairs (not -1 padding)
        valid_mask = (positive_pairs[:, :, 0] >= 0) & (positive_pairs[:, :, 1] >= 0)
        
        if not valid_mask.any():
            return torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        # Clamp indices to valid range
        start_indices = torch.clamp(positive_pairs[:, :, 0], 0, n_frames - 1)
        end_indices = torch.clamp(positive_pairs[:, :, 1], 0, n_frames - 1)
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, positive_pairs.size(1))
        
        # Gather features using advanced indexing - only for valid pairs
        flat_batch_indices = batch_indices[valid_mask]
        flat_start_indices = start_indices[valid_mask]
        flat_end_indices = end_indices[valid_mask]
        
        if len(flat_batch_indices) == 0:
            return torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        anchor_features = start_features[flat_batch_indices, flat_start_indices]  # (n_valid_pairs, feature_dim)
        positive_features = end_features[flat_batch_indices, flat_end_indices]   # (n_valid_pairs, feature_dim)
        
        n_valid = len(anchor_features)
        
        if n_valid == 0:
            return torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        # Normalize features for cosine similarity
        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        positive_features = F.normalize(positive_features, p=2, dim=1)
        
        # Compute positive similarities (anchor-positive pairs)
        pos_similarities = torch.sum(anchor_features * positive_features, dim=1)  # (n_valid,)
        pos_similarities = pos_similarities / self.temperature
        
        # For negatives, use all other positive features as negatives for each anchor
        if n_valid > 1:
            # Compute similarity matrix: anchor_features @ positive_features.T
            sim_matrix = torch.matmul(anchor_features, positive_features.t()) / self.temperature  # (n_valid, n_valid)
            
            # Create targets (diagonal elements are positive pairs)
            targets = torch.arange(n_valid, device=device)
            
            # InfoNCE loss using cross-entropy
            loss = F.cross_entropy(sim_matrix, targets)
        else:
            # If only one valid pair, return zero loss (no contrastive learning possible)
            loss = torch.zeros(1, device=device, dtype=start_features.dtype, requires_grad=start_features.requires_grad).squeeze()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for AutoSubs training.
    
    Combines focal loss for heatmaps and contrastive loss for feature matching.
    """
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # Initialize loss components
        self.focal_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )
        
        self.feature_loss = ContrastiveFeatureLoss(
            margin=config.feature_margin,
            temperature=config.temperature
        )
        
        # Alternative feature loss
        self.infonce_loss = InfoNCELoss(temperature=config.temperature)
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        use_infonce: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions with 'start' and 'end' keys
            targets: Ground truth targets
            use_infonce: Whether to use InfoNCE instead of contrastive loss
            
        Returns:
            Dictionary with loss components and total loss
        """
        # Extract predictions
        start_heatmap = predictions["start"]["heatmap"]
        end_heatmap = predictions["end"]["heatmap"]
        start_features = predictions["start"]["features"]
        end_features = predictions["end"]["features"]
        
        # Extract targets
        start_targets = targets["start_heatmap"]
        end_targets = targets["end_heatmap"]
        positive_pairs = targets["positive_pairs"]
        
        # Heatmap losses
        start_focal_loss = self.focal_loss(start_heatmap, start_targets)
        end_focal_loss = self.focal_loss(end_heatmap, end_targets)
        heatmap_loss = (start_focal_loss + end_focal_loss) / 2
        
        # Feature matching loss
        if use_infonce:
            feature_loss = self.infonce_loss(start_features, end_features, positive_pairs)
        else:
            feature_loss = self.feature_loss(start_features, end_features, positive_pairs)
        
        # Weighted combination
        total_loss = (self.config.heatmap_weight * heatmap_loss + 
                     self.config.feature_weight * feature_loss)
        
        return {
            "total_loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "feature_loss": feature_loss,
            "start_focal_loss": start_focal_loss,
            "end_focal_loss": end_focal_loss
        }


class LossMetrics:
    """Utility class for computing loss-related metrics."""
    
    @staticmethod
    def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """Compute binary accuracy for heatmaps."""
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        accuracy = (pred_binary == target_binary).float().mean()
        return accuracy.item()
    
    @staticmethod
    def compute_precision_recall(
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        threshold: float = 0.5
    ) -> Tuple[float, float]:
        """Compute precision and recall for heatmaps."""
        pred_binary = (predictions > threshold).float()
        target_binary = (targets > threshold).float()
        
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        return precision.item(), recall.item()
    
    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """Compute F1 score from precision and recall."""
        return 2 * precision * recall / (precision + recall + 1e-7)


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    from config.train import LossConfig
    
    # Create loss configuration
    loss_config = LossConfig(
        focal_alpha=0.25,
        focal_gamma=2.0,
        feature_margin=1.0,
        temperature=0.1,
        heatmap_weight=1.0,
        feature_weight=0.5
    )
    
    print(f"Loss config: {loss_config.__dict__}")
    
    # Create dummy data
    batch_size = 2
    n_frames = 1800  # 30 seconds * 60 FPS
    feature_dim = 128
    
    # Dummy predictions
    predictions = {
        "start": {
            "heatmap": torch.sigmoid(torch.randn(batch_size, n_frames)),
            "features": F.normalize(torch.randn(batch_size, n_frames, feature_dim), p=2, dim=-1)
        },
        "end": {
            "heatmap": torch.sigmoid(torch.randn(batch_size, n_frames)),
            "features": F.normalize(torch.randn(batch_size, n_frames, feature_dim), p=2, dim=-1)
        }
    }
    
    # Dummy targets
    start_targets = torch.zeros(batch_size, n_frames)
    end_targets = torch.zeros(batch_size, n_frames)
    
    # Add some positive samples
    start_targets[0, 100:110] = 1.0  # Event 1
    end_targets[0, 200:210] = 1.0
    start_targets[0, 500:510] = 1.0  # Event 2  
    end_targets[0, 800:810] = 1.0
    
    start_targets[1, 300:310] = 1.0  # Event in second batch
    end_targets[1, 600:610] = 1.0
    
    # Positive pairs (center indices of events)
    positive_pairs = torch.zeros(batch_size, 2, 2, dtype=torch.long)
    positive_pairs[0, 0] = torch.tensor([105, 205])  # First event in batch 0
    positive_pairs[0, 1] = torch.tensor([505, 805])  # Second event in batch 0
    positive_pairs[1, 0] = torch.tensor([305, 605])  # Event in batch 1
    positive_pairs[1, 1] = torch.tensor([0, 0])     # Padding (no second event)
    
    targets = {
        "start_heatmap": start_targets,
        "end_heatmap": end_targets,
        "positive_pairs": positive_pairs
    }
    
    print("\n1. Testing individual loss components:")
    
    # Test focal loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    focal_result = focal_loss(predictions["start"]["heatmap"], start_targets)
    print(f"Focal loss: {focal_result.item():.4f}")
    
    # Test contrastive loss
    contrastive_loss = ContrastiveFeatureLoss(margin=1.0, temperature=0.1)
    contrastive_result = contrastive_loss(
        predictions["start"]["features"],
        predictions["end"]["features"], 
        positive_pairs
    )
    print(f"Contrastive loss: {contrastive_result.item():.4f}")
    
    # Test InfoNCE loss
    infonce_loss = InfoNCELoss(temperature=0.1)
    infonce_result = infonce_loss(
        predictions["start"]["features"],
        predictions["end"]["features"],
        positive_pairs
    )
    print(f"InfoNCE loss: {infonce_result.item():.4f}")
    
    print("\n2. Testing combined loss:")
    combined_loss = CombinedLoss(loss_config)
    
    # Test with contrastive loss
    loss_dict = combined_loss(predictions, targets, use_infonce=False)
    print(f"Total loss (contrastive): {loss_dict['total_loss'].item():.4f}")
    print(f"Heatmap loss: {loss_dict['heatmap_loss'].item():.4f}")
    print(f"Feature loss: {loss_dict['feature_loss'].item():.4f}")
    
    # Test with InfoNCE loss
    loss_dict_infonce = combined_loss(predictions, targets, use_infonce=True)
    print(f"Total loss (InfoNCE): {loss_dict_infonce['total_loss'].item():.4f}")
    
    print("\n3. Testing metrics:")
    metrics = LossMetrics()
    
    accuracy = metrics.compute_accuracy(predictions["start"]["heatmap"], start_targets)
    precision, recall = metrics.compute_precision_recall(predictions["start"]["heatmap"], start_targets)
    f1 = metrics.compute_f1_score(precision, recall)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nLoss function tests completed successfully!")