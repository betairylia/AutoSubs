#!/usr/bin/env python3
"""
Script to validate and analyze datasets.

Usage:
    python scripts/validate_dataset.py --dataset /path/to/dataset.h5
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.dataset import HDF5DatasetStorage, PyTorchDatasetStorage, AutoSubsDataset, split_dataset
from data.chunking import validate_chunks, get_chunk_statistics
from config.config import Config


def main():
    parser = argparse.ArgumentParser(description="Validate and analyze AutoSubs dataset")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, type=str,
                       help="Path to dataset file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="Number of samples to analyze in detail")
    parser.add_argument("--show_splits", action="store_true",
                       help="Show train/val/test split information")
    parser.add_argument("--test_loading", action="store_true",
                       help="Test dataset loading with DataLoader")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    if args.config:
        from config.config import load_config
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
    else:
        config = Config()
        logging.info("Using default configuration")
    
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
        
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Basic information
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Dataset file: {dataset_path}")
    print(f"Storage format: {dataset_path.suffix}")
    print(f"Total chunks: {len(chunks)}")
    
    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if isinstance(value, dict) and 'processing_stats' in str(key):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    if not chunks:
        logging.error("Dataset is empty")
        return 1
    
    # Validation
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    validation_report = validate_chunks(chunks)
    print(f"Total chunks: {validation_report['total_chunks']}")
    print(f"Valid chunks: {validation_report['valid_chunks']}")
    print(f"Empty subtitle chunks: {validation_report['empty_subtitle_chunks']}")
    print(f"Padded chunks: {validation_report['padded_chunks']}")
    print(f"Average events per chunk: {validation_report['avg_events_per_chunk']:.2f}")
    print(f"Average spectrogram frames: {validation_report['avg_spectogram_frames']:.1f}")
    
    if validation_report['issues']:
        print(f"\nValidation issues found:")
        for issue in validation_report['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(validation_report['issues']) > 5:
            print(f"  ... and {len(validation_report['issues']) - 5} more issues")
    else:
        print("✓ No validation issues found")
    
    # Statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    stats = get_chunk_statistics(chunks)
    print(f"Total duration: {stats['total_duration']:.1f} seconds ({stats['total_duration']/3600:.2f} hours)")
    print(f"Average chunk duration: {stats['avg_chunk_duration']:.2f}s")
    print(f"Duration range: {stats['min_chunk_duration']:.2f}s - {stats['max_chunk_duration']:.2f}s")
    print(f"Average events per chunk: {stats['avg_events_per_chunk']:.2f}")
    print(f"Average start times per chunk: {stats['avg_start_times_per_chunk']:.2f}")
    print(f"Average end times per chunk: {stats['avg_end_times_per_chunk']:.2f}")
    print(f"Average positive pairs per chunk: {stats['avg_pairs_per_chunk']:.2f}")
    print(f"Average spectrogram frames: {stats['avg_spectrogram_frames']:.1f}")
    print(f"Padded chunks: {stats['padded_chunks']} ({stats['padded_chunks']/len(chunks)*100:.1f}%)")
    print(f"Empty subtitle chunks: {stats['empty_subtitle_chunks']} ({stats['empty_subtitle_chunks']/len(chunks)*100:.1f}%)")
    
    # Sample analysis
    if args.sample_size > 0:
        print(f"\n" + "="*60)
        print(f"SAMPLE ANALYSIS ({min(args.sample_size, len(chunks))} chunks)")
        print("="*60)
        
        import random
        random.seed(42)  # For reproducible sampling
        sample_chunks = random.sample(chunks, min(args.sample_size, len(chunks)))
        
        for i, chunk in enumerate(sample_chunks):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Duration: {chunk.audio_chunk.duration:.2f}s")
            print(f"  Spectrogram shape: {chunk.audio_chunk.spectrogram.shape}")
            print(f"  Sample rate: {chunk.audio_chunk.sample_rate}")
            print(f"  Is padded: {chunk.audio_chunk.is_padded}")
            print(f"  Subtitle events: {len(chunk.subtitle_chunk.events)}")
            print(f"  Start times: {len(chunk.subtitle_chunk.start_times)}")
            print(f"  End times: {len(chunk.subtitle_chunk.end_times)}")
            print(f"  Positive pairs: {len(chunk.subtitle_chunk.positive_pairs)}")
            
            if chunk.subtitle_chunk.events:
                print(f"  Event example: {chunk.subtitle_chunk.events[0].start_time:.2f}s - "
                      f"{chunk.subtitle_chunk.events[0].end_time:.2f}s: "
                      f"'{chunk.subtitle_chunk.events[0].text[:50]}{'...' if len(chunk.subtitle_chunk.events[0].text) > 50 else ''}'")
    
    # Split analysis
    if args.show_splits:
        print(f"\n" + "="*60)
        print("DATASET SPLITS")
        print("="*60)
        
        train_chunks, val_chunks, test_chunks = split_dataset(chunks, config.data)
        
        print(f"Training set: {len(train_chunks)} chunks ({len(train_chunks)/len(chunks)*100:.1f}%)")
        print(f"Validation set: {len(val_chunks)} chunks ({len(val_chunks)/len(chunks)*100:.1f}%)")
        print(f"Test set: {len(test_chunks)} chunks ({len(test_chunks)/len(chunks)*100:.1f}%)")
        
        # Statistics for each split
        for split_name, split_chunks in [("Train", train_chunks), ("Validation", val_chunks), ("Test", test_chunks)]:
            if split_chunks:
                split_stats = get_chunk_statistics(split_chunks)
                print(f"\n{split_name} split statistics:")
                print(f"  Duration: {split_stats['total_duration']:.1f}s")
                print(f"  Events per chunk: {split_stats['avg_events_per_chunk']:.2f}")
                print(f"  Empty chunks: {split_stats['empty_subtitle_chunks']}")
    
    # DataLoader test
    if args.test_loading:
        print(f"\n" + "="*60)
        print("DATALOADER TEST")
        print("="*60)
        
        try:
            # Create dataset
            dataset = AutoSubsDataset(chunks[:min(100, len(chunks))], config.data)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            
            print(f"Created dataset with {len(dataset)} samples")
            print(f"Testing DataLoader...")
            
            # Test first batch
            batch = next(iter(dataloader))
            
            print(f"✓ DataLoader working successfully")
            print(f"  Batch size: {len(batch['spectrogram'])}")
            print(f"  Spectrogram shape: {batch['spectrogram'].shape}")
            print(f"  Start heatmap shape: {batch['start_heatmap'].shape}")
            print(f"  End heatmap shape: {batch['end_heatmap'].shape}")
            print(f"  Positive pairs shape: {batch['positive_pairs'].shape}")
            print(f"  Data types: {batch['spectrogram'].dtype}")
            
        except Exception as e:
            logging.error(f"DataLoader test failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    
    if validation_report['valid_chunks'] == validation_report['total_chunks'] and not validation_report['issues']:
        print("✓ Dataset validation passed - ready for training!")
    else:
        print("⚠ Dataset has some issues - review the validation report above")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())