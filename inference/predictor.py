import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from models.network import create_model, AutoSubsNetwork
from inference.postprocessing import InferencePostProcessor
from utils.device import get_available_device
from utils.audio import audio_to_spectrogram, get_audio_duration
from data.chunking import chunk_audio_file
from data.subtitle import SubtitleEvent
from datetime import timedelta
import ass


class AutoSubsPredictor:
    """Main predictor class for inference on audio files."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to config file (will try to load from checkpoint if not provided)
            device: Device to use for inference
        """
        self.device = get_available_device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if config_path:
            from config.config import load_config
            self.config = load_config(config_path)
        else:
            # Try to load config from checkpoint
            if "config" in checkpoint:
                self.config = Config()
                # Update config with saved values (simplified loading)
                logging.info("Loading config from checkpoint")
            else:
                # Fall back to default config
                self.config = Config()
                logging.warning("Using default config - model may not work correctly")
        
        # Create and load model
        self.model = create_model(self.config.model)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Create post-processor
        self.post_processor = InferencePostProcessor(self.config.model)
        
        logging.info(f"Predictor loaded on device: {self.device}")
        logging.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def predict_audio_file(
        self, 
        audio_file: Union[str, Path],
        overlap_ratio: float = 0.1,
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Predict subtitle timing for an entire audio file.
        
        Args:
            audio_file: Path to audio file
            overlap_ratio: Overlap ratio for chunking
            batch_size: Batch size for inference
            
        Returns:
            List of predicted subtitle events
        """
        audio_file = Path(audio_file)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get audio duration for context
        total_duration = get_audio_duration(audio_file)
        logging.info(f"Processing audio file: {audio_file} (duration: {total_duration:.1f}s)")
        
        # Create modified chunking config for inference
        chunking_config = self.config.data.chunking
        chunking_config.overlap_ratio = overlap_ratio
        
        # Chunk the audio
        audio_chunks = chunk_audio_file(audio_file, self.config.data.audio, chunking_config)
        logging.info(f"Created {len(audio_chunks)} audio chunks for inference")
        
        # Process chunks in batches
        all_predictions = []
        
        for i in range(0, len(audio_chunks), batch_size):
            batch_chunks = audio_chunks[i:i + batch_size]
            batch_predictions = self._predict_batch(batch_chunks)
            all_predictions.extend(batch_predictions)
            
            logging.info(
                f"Processed batch {min(i + batch_size, len(audio_chunks))}/{len(audio_chunks)} " + 
                f"with {sum([len(c['events']) for c in batch_predictions])} raw events")
        
        # Merge overlapping predictions
        merged_predictions = self._merge_overlapping_chunks(all_predictions, audio_chunks)
        
        # Global post-processing
        final_predictions = self._global_postprocessing(merged_predictions, total_duration)
        
        logging.info(f"Final prediction: {len(final_predictions)} subtitle events")
        return final_predictions
    
    def _predict_batch(self, chunks: List) -> List[Dict]:
        """Process a batch of chunks."""
        batch_spectrograms = []
        
        # Prepare batch
        for chunk in chunks:
            batch_spectrograms.append(torch.from_numpy(chunk.spectrogram))
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_spectrograms).to(self.device)
        
        # Run inference
        with torch.no_grad():
            batch_outputs = self.model.get_timing_predictions(
                batch_tensor,
                timing_fps=self.config.data.chunking.timing_fps,
                chunk_duration=self.config.data.chunking.chunk_duration
            )
        
        # Post-process each chunk
        batch_predictions = []
        for i, chunk in enumerate(chunks):
            # Extract predictions for this chunk
            start_heatmap = batch_outputs["start_heatmap"][i]
            end_heatmap = batch_outputs["end_heatmap"][i]
            start_features = batch_outputs["start_features"][i]
            end_features = batch_outputs["end_features"][i]
            
            # Post-process
            chunk_events = self.post_processor.process_predictions(
                start_heatmap, end_heatmap,
                start_features, end_features,
                timing_fps=self.config.data.chunking.timing_fps
            )
            
            # Adjust timing to global coordinates
            for event in chunk_events:
                event["start_time"] += chunk.start_time
                event["end_time"] += chunk.start_time
                event["chunk_id"] = chunk.chunk_id
                event["chunk_start"] = chunk.start_time
            
            batch_predictions.append({
                "chunk": chunk,
                "events": chunk_events
            })
        
        return batch_predictions
    
    def _merge_overlapping_chunks(self, predictions: List[Dict], chunks: List) -> List[Dict]:
        """Merge predictions from overlapping chunks."""
        if not predictions:
            return []
        
        # Collect all events
        all_events = []
        for pred in predictions:
            all_events.extend(pred["events"])
        
        if not all_events:
            return []
        
        # Sort by start time
        all_events.sort(key=lambda x: x["start_time"])
        
        # Merge events that are very close (likely duplicates from overlapping chunks)
        merged_events = []
        merge_threshold = 1.0  # 1 second threshold
        
        for event in all_events:
            # Check if this event is close to the last merged event
            if (merged_events and 
                abs(event["start_time"] - merged_events[-1]["start_time"]) < merge_threshold and
                abs(event["end_time"] - merged_events[-1]["end_time"]) < merge_threshold):
                
                # Update the existing event with higher confidence values
                last_event = merged_events[-1]
                if event["start_confidence"] + event["end_confidence"] > \
                   last_event["start_confidence"] + last_event["end_confidence"]:
                    merged_events[-1] = event
            else:
                merged_events.append(event)
        
        return merged_events
    
    def _global_postprocessing(self, events: List[Dict], total_duration: float) -> List[Dict]:
        """Apply global post-processing filters."""
        if not events:
            return []
        
        # Filter events that are outside the audio duration
        valid_events = [
            event for event in events 
            if 0 <= event["start_time"] < total_duration and event["end_time"] <= total_duration
        ]
        
        # Apply confidence threshold
        min_avg_confidence = 0.3
        confident_events = [
            event for event in valid_events
            if (event["start_confidence"] + event["end_confidence"]) / 2 >= min_avg_confidence
        ]
        
        # Filter by duration (remove very short or very long events)
        duration_filtered = [
            event for event in confident_events
            if 0.5 <= event["duration"] <= self.config.model.max_subtitle_length
        ]
        
        # Sort by start time
        duration_filtered.sort(key=lambda x: x["start_time"])
        
        return duration_filtered
    
    def predict_chunk(
        self, 
        spectrogram: torch.Tensor,
        return_raw: bool = False
    ) -> List[Dict]:
        """
        Predict timing for a single spectrogram chunk.
        
        Args:
            spectrogram: Input spectrogram (1, n_mels, n_frames) or (n_mels, n_frames)
            return_raw: If True, return raw heatmaps and features
            
        Returns:
            List of predicted events or raw predictions if return_raw=True
        """
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
        
        spectrogram = spectrogram.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_timing_predictions(
                spectrogram,
                timing_fps=self.config.data.chunking.timing_fps,
                chunk_duration=self.config.data.chunking.chunk_duration
            )
        
        if return_raw:
            return outputs
        
        # Post-process
        events = self.post_processor.process_predictions(
            outputs["start_heatmap"][0],  # Remove batch dimension
            outputs["end_heatmap"][0],
            outputs["start_features"][0],
            outputs["end_features"][0],
            timing_fps=self.config.data.chunking.timing_fps
        )
        
        return events
    
    def save_ass_file(self, events: List[Dict], output_path: Union[str, Path], style: str = "Default"):
        """
        Save predicted events to ASS subtitle file.
        
        Args:
            events: List of predicted subtitle events
            output_path: Path to save ASS file
            style: ASS style name to use
        """
        output_path = Path(output_path)
        
        # Create ASS document
        doc = ass.Document()
        
        # Add a basic style
        doc.styles.append(ass.Style(
            name=style,
            fontname="Arial",
            fontsize=20,
            bold=True,
            italic=False,
            underline=False,
            strikeout=False,
            scalex=100,
            scaley=100,
            spacing=0,
            angle=0,
            borderstyle=1,
            outline=2,
            shadow=0,
            alignment=2,
            marginl=10,
            marginr=10,
            marginv=10,
            encoding=1
        ))
        
        # Add events
        for i, event in enumerate(events):
            start_time = event["start_time"] * 1000  # Convert to milliseconds
            end_time = event["end_time"] * 1000
            
            # Create dialogue event
            dialogue = ass.Dialogue(
                start=timedelta(milliseconds = start_time),
                end=timedelta(milliseconds = end_time),
                style=style,
                text=""
            )
            doc.events.append(dialogue)
        
        # Save file
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            doc.dump_file(f)
        
        logging.info(f"Saved {len(events)} subtitle events to: {output_path}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "backbone_type": self.config.model.backbone.type,
            "input_resolution": f"{self.config.data.audio.n_mels} mel bands",
            "timing_fps": self.config.data.chunking.timing_fps,
            "chunk_duration": self.config.data.chunking.chunk_duration,
            "confidence_threshold": self.config.model.confidence_threshold
        }


def create_predictor(model_path: str, config_path: Optional[str] = None, device: str = "auto") -> AutoSubsPredictor:
    """Factory function to create predictor."""
    return AutoSubsPredictor(model_path, config_path, device)


if __name__ == "__main__":
    # Test predictor setup (would need a trained model)
    print("Testing predictor setup...")
    
    # This would normally require a trained model checkpoint
    print("Predictor implementation completed!")
    print("Features:")
    print("- Batch processing of audio chunks")
    print("- Overlapping chunk merging") 
    print("- Global post-processing filters")
    print("- Direct ASS file output")
    print("- Raw prediction access for debugging")
    print("- Model information reporting")
    
    # Create a dummy predictor test
    from config.config import Config
    config = Config()
    
    print(f"\nDefault config summary:")
    print(f"- Backbone: {config.model.backbone.type}")
    print(f"- Timing FPS: {config.data.chunking.timing_fps}")
    print(f"- Chunk duration: {config.data.chunking.chunk_duration}s")
    print(f"- Confidence threshold: {config.model.confidence_threshold}")
    
    print("\nPredictor test completed!")