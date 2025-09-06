#!/usr/bin/env python3
"""
Script to train AutoSubs model.

Usage:
    python scripts/train.py --dataset /path/to/dataset.h5 --config /path/to/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import load_config
from data.dataset import HDF5DatasetStorage, PyTorchDatasetStorage, AutoSubsDataset, split_dataset
from training.trainer import AutoSubsTrainer


def main():
    parser = argparse.ArgumentParser(description="Train AutoSubs model")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, type=str,
                       help="Path to dataset file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./logs",
                       help="Output directory for logs and checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name (overrides config)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use for training")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--no_validation", action="store_true",
                       help="Disable validation during training")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
    else:
        from config.config import Config
        config = Config()
        logging.info("Using default configuration")
    
    # Apply command line overrides
    if args.output_dir:
        config.training.log_dir = args.output_dir
    if args.experiment_name:
        config.training.experiment_name = args.experiment_name
    if args.device != "auto":
        config.training.device = args.device
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.optimizer.lr = args.lr
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logging.error(f"Dataset file not found: {dataset_path}")
        return 1
    
    # Load dataset
    logging.info(f"Loading dataset from: {dataset_path}")
    try:
        if dataset_path.suffix.lower() == '.h5':
            storage = HDF5DatasetStorage(dataset_path)
        else:
            storage = PyTorchDatasetStorage(dataset_path)
        
        chunks, metadata = storage.load_dataset()
        logging.info(f"Loaded {len(chunks)} chunks from dataset")
        
        if metadata:
            logging.info(f"Dataset metadata: {metadata.get('total_chunks', 'unknown')} chunks")
    
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return 1
    
    if not chunks:
        logging.error("Dataset is empty")
        return 1
    
    # Split dataset
    if not args.no_validation:
        train_chunks, val_chunks, test_chunks = split_dataset(chunks, config.data)
        logging.info(f"Dataset split: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test")
    else:
        train_chunks = chunks
        val_chunks = []
        test_chunks = []
        logging.info(f"Using all {len(train_chunks)} chunks for training (no validation)")
    
    # Create datasets
    train_dataset = AutoSubsDataset(train_chunks, config.data)
    val_dataset = AutoSubsDataset(val_chunks, config.data) if val_chunks else None
    
    logging.info(f"Created training dataset with {len(train_dataset)} samples")
    if val_dataset:
        logging.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )
    
    # Create trainer
    logging.info("Initializing trainer...")
    trainer = AutoSubsTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logging.error(f"Resume checkpoint not found: {resume_path}")
            return 1
        
        logging.info(f"Resuming training from: {resume_path}")
        trainer.load_checkpoint(str(resume_path))
    
    # Print training information
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {config.model.backbone.type} backbone")
    print(f"Device: {trainer.device}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.optimizer.lr}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Output directory: {trainer.log_dir}")
    
    # Start training
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    try:
        trainer.train(train_loader, val_loader)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
        print(f"Logs saved to: {trainer.log_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        logging.info("Saved checkpoint before exiting")
        return 1
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())