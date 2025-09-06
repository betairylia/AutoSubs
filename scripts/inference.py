#!/usr/bin/env python3
"""
Script to run inference on audio files.

Usage:
    python scripts/inference.py --model /path/to/model.pt --input /path/to/audio.wav --output /path/to/output.ass
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.predictor import create_predictor
from utils.audio import get_audio_duration, get_audio_stats


def main():
    parser = argparse.ArgumentParser(description="Run AutoSubs inference on audio files")
    
    # Required arguments
    parser.add_argument("--model", required=True, type=str,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input", required=True, type=str,
                       help="Path to input audio file")
    parser.add_argument("--output", required=True, type=str,
                       help="Path to output ASS subtitle file")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=None,
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="Device to use for inference")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing chunks")
    parser.add_argument("--overlap", type=float, default=0.1,
                       help="Overlap ratio between chunks (0.0-0.5)")
    parser.add_argument("--style", type=str, default="Default",
                       help="ASS style name for subtitles")
    parser.add_argument("--stats", action="store_true",
                       help="Show audio file statistics")
    parser.add_argument("--dry_run", action="store_true",
                       help="Load model and analyze audio without running inference")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        logging.error(f"Model checkpoint not found: {model_path}")
        return 1
    
    audio_path = Path(args.input)
    if not audio_path.exists():
        logging.error(f"Input audio file not found: {audio_path}")
        return 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate arguments
    if not 0.0 <= args.overlap <= 0.5:
        logging.error("Overlap ratio must be between 0.0 and 0.5")
        return 1
    
    # Show audio statistics if requested
    if args.stats:
        print("\n" + "="*50)
        print("AUDIO FILE STATISTICS")
        print("="*50)
        
        duration = get_audio_duration(str(audio_path))
        stats = get_audio_stats(str(audio_path))
        
        print(f"File: {audio_path}")
        print(f"Duration: {duration:.2f} seconds")
        if 'error' not in stats:
            print(f"Sample rate: {stats['sample_rate']} Hz")
            print(f"Samples: {stats['n_samples']:,}")
            print(f"RMS energy: {stats['rms_energy']:.4f}")
            print(f"Max amplitude: {stats['max_amplitude']:.4f}")
            print(f"Zero crossing rate: {stats['zero_crossing_rate']:.4f}")
            print(f"Spectral centroid: {stats['spectral_centroid_mean']:.1f} Hz")
            print(f"Spectral rolloff: {stats['spectral_rolloff_mean']:.1f} Hz")
        else:
            print(f"Error getting stats: {stats['error']}")
    
    # Load model
    print("\n" + "="*50)
    print("LOADING MODEL")
    print("="*50)
    
    try:
        start_time = time.time()
        predictor = create_predictor(str(model_path), args.config, args.device)
        load_time = time.time() - start_time
        
        model_info = predictor.get_model_info()
        print(f"Model loaded successfully in {load_time:.2f}s")
        print(f"Device: {model_info['device']}")
        print(f"Parameters: {model_info['parameters']:,}")
        print(f"Backbone: {model_info['backbone_type']}")
        print(f"Input resolution: {model_info['input_resolution']}")
        print(f"Timing FPS: {model_info['timing_fps']}")
        print(f"Chunk duration: {model_info['chunk_duration']}s")
        print(f"Confidence threshold: {model_info['confidence_threshold']}")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    if args.dry_run:
        print("\nDry run mode - exiting without running inference")
        return 0
    
    # Run inference
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    
    try:
        start_time = time.time()
        
        events = predictor.predict_audio_file(
            str(audio_path),
            overlap_ratio=args.overlap,
            batch_size=args.batch_size
        )
        
        inference_time = time.time() - start_time
        audio_duration = get_audio_duration(str(audio_path))
        
        print(f"Inference completed in {inference_time:.2f}s")
        print(f"Processing speed: {audio_duration/inference_time:.1f}x realtime")
        print(f"Found {len(events)} subtitle events")
        
        if events:
            print(f"\nPredicted events:")
            for i, event in enumerate(events):
                duration = event["end_time"] - event["start_time"]
                avg_conf = (event["start_confidence"] + event["end_confidence"]) / 2
                print(f"  {i+1:2d}: {event['start_time']:6.2f}s - {event['end_time']:6.2f}s "
                      f"({duration:5.2f}s, conf: {avg_conf:.3f})")
        
        # Save results
        predictor.save_ass_file(events, str(output_path), args.style)
        
        print(f"\nResults saved to: {output_path}")
        
        # Summary statistics
        if events:
            total_subtitle_time = sum(e["end_time"] - e["start_time"] for e in events)
            coverage = (total_subtitle_time / audio_duration) * 100
            avg_duration = total_subtitle_time / len(events)
            avg_confidence = sum((e["start_confidence"] + e["end_confidence"]) / 2 for e in events) / len(events)
            
            print(f"\nSummary:")
            print(f"  Subtitle coverage: {coverage:.1f}%")
            print(f"  Average event duration: {avg_duration:.2f}s")
            print(f"  Average confidence: {avg_confidence:.3f}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())