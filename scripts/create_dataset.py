#!/usr/bin/env python3
"""
Script to create dataset from directory of audio-subtitle pairs.

Usage:
    python scripts/create_dataset.py --data_dir /path/to/data --output /path/to/dataset.h5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import load_config
from data.dataset import create_dataset_from_directory
from data.utils import get_directory_stats


def main():
    parser = argparse.ArgumentParser(description="Create AutoSubs dataset from audio-subtitle pairs")
    
    # Required arguments
    parser.add_argument("--data_dir", required=True, type=str,
                       help="Directory containing audio and subtitle files")
    parser.add_argument("--output", required=True, type=str,
                       help="Output path for dataset file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file (default: use default config)")
    parser.add_argument("--format", choices=["hdf5", "pytorch"], default="hdf5",
                       help="Dataset storage format (default: hdf5)")
    parser.add_argument("--stats_only", action="store_true",
                       help="Only show directory statistics without creating dataset")
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
    
    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logging.error(f"Data directory not found: {data_dir}")
        return 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Show directory statistics
    logging.info("Analyzing data directory...")
    stats = get_directory_stats(str(data_dir), config.data)
    
    print("\n" + "="*50)
    print("DIRECTORY STATISTICS")
    print("="*50)
    print(f"Directory: {data_dir}")
    print(f"Total files: {stats['total_files']}")
    print(f"Audio files: {stats['audio_files']}")
    print(f"Subtitle files: {stats['subtitle_files']}")
    print(f"Other files: {stats['other_files']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print(f"Audio-subtitle pairs found: {stats['pairs_found']}")
    print(f"Unpaired audio files: {stats['unpaired_audio']}")
    print(f"Unpaired subtitle files: {stats['unpaired_subtitles']}")
    
    if stats['file_extensions']:
        print(f"\nFile extensions:")
        for ext, count in sorted(stats['file_extensions'].items()):
            print(f"  {ext}: {count}")
    
    if args.stats_only:
        print("\nStats-only mode - exiting without creating dataset")
        return 0
    
    if stats['pairs_found'] == 0:
        logging.error("No audio-subtitle pairs found. Cannot create dataset.")
        return 1
    
    # Create dataset
    print("\n" + "="*50)
    print("DATASET CREATION")
    print("="*50)
    
    try:
        creation_stats = create_dataset_from_directory(
            str(data_dir),
            str(output_path),
            config.data,
            args.format
        )
        
        print(f"\nDataset created successfully: {output_path}")
        print(f"Storage format: {args.format}")
        print(f"Total pairs processed: {creation_stats['processed_pairs']}")
        print(f"Failed pairs: {creation_stats['failed_pairs']}")
        print(f"Total chunks created: {creation_stats['total_chunks']}")
        
        if creation_stats['processing_errors']:
            print(f"\nProcessing errors:")
            for error in creation_stats['processing_errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(creation_stats['processing_errors']) > 5:
                print(f"  ... and {len(creation_stats['processing_errors']) - 5} more")
        
        return 0
        
    except Exception as e:
        logging.error(f"Dataset creation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())