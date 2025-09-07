import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import time
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import asdict
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.train import TrainingConfig
from config.config import Config
from models.network import create_model, AutoSubsNetwork
from training.losses import CombinedLoss, LossMetrics
from training.event_metrics import compute_event_iou_for_batch
from utils.device import get_available_device, optimize_device_settings
from data.dataset import AutoSubsDataset


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == "max"
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class AutoSubsTrainer:
    """Main trainer class for AutoSubs model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.training_config = config.training
        self.device = get_available_device(config.training.device)
        
        # Setup logging
        self._setup_logging()
        
        # Create model
        self.model = create_model(config.model).to(self.device)
        optimize_device_settings(self.device)
        
        # Create loss function
        self.criterion = CombinedLoss(config.training.loss)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler  
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None
        
        # Metrics tracker
        self.metrics = LossMetrics()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode="min"
        )
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup directories
        self.log_dir = Path(config.training.log_dir) / config.training.experiment_name
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        logging.info(f"Trainer initialized with device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.training_config.optimizer
        
        if opt_config.type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps
            )
        elif opt_config.type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas,
                eps=opt_config.eps
            )
        elif opt_config.type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
                nesterov=opt_config.nesterov
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        sched_config = self.training_config.scheduler
        
        if sched_config.type == "none":
            return None
        elif sched_config.type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.T_max,
                eta_min=sched_config.eta_min
            )
        elif sched_config.type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
        elif sched_config.type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=sched_config.patience,
                factor=sched_config.factor,
                threshold=sched_config.threshold,
                mode="min"
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_config.type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "total_loss": 0.0,
            "heatmap_loss": 0.0,
            "feature_loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast(str(self.device)):
                    outputs = self._forward_pass(batch)
                    loss_dict = self._compute_loss(outputs, batch)
                
                # Backward pass
                self.scaler.scale(loss_dict["total_loss"]).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_pass(batch)
                loss_dict = self._compute_loss(outputs, batch)
                
                # Backward pass
                loss_dict["total_loss"].backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip_norm)
                
                self.optimizer.step()
            
            # Accumulate metrics
            batch_metrics = self._compute_metrics(outputs, batch, loss_dict)
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
            
            epoch_losses.append(loss_dict["total_loss"].item())
            
            # Logging
            if batch_idx % self.training_config.log_every_n_steps == 0:
                logging.info(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss_dict['total_loss'].item():.4f}")
            
            # Tensorboard logging
            if self.current_step % self.training_config.log_every_n_steps == 0:
                self._log_to_tensorboard("train", batch_metrics, self.current_step)
            
            self.current_step += 1
            
            # Validation check
            if self.current_step % self.training_config.val_check_interval == 0:
                # This would be called from the main training loop
                pass
        
        # Average metrics over epoch
        n_batches = len(train_loader)
        epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = []
        val_metrics = {
            "total_loss": 0.0,
            "heatmap_loss": 0.0,
            "feature_loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "event_iou": 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self._forward_pass(batch)
                loss_dict = self._compute_loss(outputs, batch)
                
                # Accumulate metrics
                batch_metrics = self._compute_metrics(outputs, batch, loss_dict)
                for key, value in batch_metrics.items():
                    val_metrics[key] += value
                
                # Compute event-level IoU
                try:
                    batch_size = outputs["start_heatmap"].size(0)
                    outputs_batch = []
                    for i in range(batch_size):
                        item_outputs = {k: v[i:i+1] for k, v in outputs.items()}
                        outputs_batch.append(item_outputs)
                    
                    event_iou = compute_event_iou_for_batch(
                        outputs_batch, batch, self.config.model,
                        self.config.data.chunking.timing_fps
                    )
                    val_metrics["event_iou"] += event_iou
                except Exception as e:
                    # If event IoU computation fails, just log and continue
                    logging.warning(f"Event IoU computation failed: {e}")
                    val_metrics["event_iou"] += 0.0
                
                val_losses.append(loss_dict["total_loss"].item())
        
        # Average metrics
        n_batches = len(val_loader)
        val_metrics = {k: v / n_batches for k, v in val_metrics.items()}
        
        return val_metrics
    
    def _forward_pass(self, batch: Dict) -> Dict:
        """Forward pass through the model."""
        spectrogram = batch["spectrogram"]
        n_frames = batch["n_frames"][0] if isinstance(batch["n_frames"], list) else batch["n_frames"][0].item()  # Handle both list and tensor format
        
        outputs = self.model.get_timing_predictions(
            spectrogram,
            timing_fps=self.config.data.chunking.timing_fps,
            chunk_duration=self.config.data.chunking.chunk_duration
        )
        
        return outputs
    
    def _compute_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute loss from outputs and targets."""
        predictions = {
            "start": {
                "heatmap": outputs["start_heatmap"],
                "features": outputs["start_features"]
            },
            "end": {
                "heatmap": outputs["end_heatmap"], 
                "features": outputs["end_features"]
            }
        }
        
        targets = {
            "start_heatmap": batch["start_heatmap"],
            "end_heatmap": batch["end_heatmap"],
            "positive_pairs": batch["positive_pairs"]
        }
        
        return self.criterion(predictions, targets)
    
    def _compute_metrics(
        self, 
        outputs: Dict, 
        batch: Dict, 
        loss_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute training metrics."""
        # Basic loss metrics
        metrics = {
            "total_loss": loss_dict["total_loss"].item(),
            "heatmap_loss": loss_dict["heatmap_loss"].item(),
            "feature_loss": loss_dict["feature_loss"].item()
        }
        
        # Classification metrics for start heatmap
        start_acc = self.metrics.compute_accuracy(outputs["start_heatmap"], batch["start_heatmap"])
        start_prec, start_rec = self.metrics.compute_precision_recall(
            outputs["start_heatmap"], batch["start_heatmap"]
        )
        start_f1 = self.metrics.compute_f1_score(start_prec, start_rec)
        
        metrics.update({
            "accuracy": start_acc,
            "precision": start_prec,
            "recall": start_rec,
            "f1": start_f1
        })
        
        return metrics
    
    def _log_to_tensorboard(self, phase: str, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, step)
        
        # Log learning rate
        if self.scheduler:
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.training_config.optimizer.lr
            self.writer.add_scalar("training/learning_rate", current_lr, step)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "loss_state_dict": self.criterion.state_dict(),  # Save loss function state (EMA)
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        
        # Save regular checkpoint
        if epoch % self.training_config.save_every_n_epochs == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model: {best_path}")
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load loss function state (EMA) if available
        if "loss_state_dict" in checkpoint:
            self.criterion.load_state_dict(checkpoint["loss_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.current_step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        logging.info("Starting training...")
        logging.info(f"Training for {self.training_config.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.training_config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics["total_loss"])
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics["total_loss"])
                
                # Early stopping check
                if self.early_stopping(val_metrics["total_loss"]):
                    logging.info("Early stopping triggered")
                    break
                
                # Update best model
                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self.save_checkpoint(epoch, is_best=True)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("total_loss", train_metrics["total_loss"]))
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logging.info(f"Train loss: {train_metrics['total_loss']:.4f}")
            if val_metrics:
                logging.info(f"Val loss: {val_metrics['total_loss']:.4f}")
                logging.info(f"Val Event IoU: {val_metrics['event_iou']:.4f}")
            
            # Tensorboard logging
            self._log_to_tensorboard("train_epoch", train_metrics, epoch)
            if val_metrics:
                self._log_to_tensorboard("val_epoch", val_metrics, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch)
        
        logging.info("Training completed!")
        self.writer.close()


if __name__ == "__main__":
    # Test trainer setup (without actual training)
    print("Testing trainer setup...")
    
    from config.config import Config
    
    # Load default config
    config = Config()
    
    print(f"Config loaded: {config.project_name}")
    
    # Create trainer
    trainer = AutoSubsTrainer(config)
    
    print(f"Trainer initialized with device: {trainer.device}")
    print(f"Model created with {sum(p.numel() for p in trainer.model.parameters()):,} parameters")
    
    # Test checkpoint saving/loading
    trainer.save_checkpoint(0)
    print("Test checkpoint saved")
    
    checkpoint_path = trainer.checkpoint_dir / "latest.pt"
    if checkpoint_path.exists():
        trainer.load_checkpoint(str(checkpoint_path))
        print("Test checkpoint loaded")
    
    print("Trainer test completed successfully!")