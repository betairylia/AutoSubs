import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from config.data import DataConfig


def find_audio_subtitle_pairs(
    data_dir: str, 
    config: Optional[DataConfig] = None
) -> List[Tuple[Path, Path]]:
    """
    Find audio-subtitle pairs in a directory.
    
    Args:
        data_dir: Directory to search for files
        config: Data configuration (optional)
        
    Returns:
        List of (audio_file, subtitle_file) pairs
    """
    if config is None:
        from config.data import DataConfig
        config = DataConfig()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all audio and subtitle files
    audio_files = []
    subtitle_files = []
    
    for ext in config.audio_extensions:
        audio_files.extend(data_path.rglob(f"*{ext}"))
    
    for ext in config.subtitle_extensions:
        subtitle_files.extend(data_path.rglob(f"*{ext}"))
    
    logging.info(f"Found {len(audio_files)} audio files and {len(subtitle_files)} subtitle files")
    
    # Match files based on filename similarity
    pairs = []
    used_subtitles = set()
    
    for audio_file in audio_files:
        best_match = None
        best_score = 0
        
        audio_stem = audio_file.stem.lower()
        
        for subtitle_file in subtitle_files:
            if subtitle_file in used_subtitles:
                continue
                
            subtitle_stem = subtitle_file.stem.lower()
            
            # Calculate similarity score
            score = calculate_filename_similarity(audio_stem, subtitle_stem)
            
            if score > best_score and score > 0.5:  # Minimum similarity threshold
                best_match = subtitle_file
                best_score = score
        
        if best_match is not None:
            pairs.append((audio_file, best_match))
            used_subtitles.add(best_match)
            logging.debug(f"Paired: {audio_file.name} <-> {best_match.name} (score: {best_score:.3f})")
    
    logging.info(f"Found {len(pairs)} audio-subtitle pairs")
    return pairs


def calculate_filename_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two filenames.
    
    Args:
        name1, name2: Filenames to compare (without extension)
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize names
    name1 = normalize_filename(name1)
    name2 = normalize_filename(name2)
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Check if one is a substring of the other
    if name1 in name2 or name2 in name1:
        return 0.8
    
    # Calculate Jaccard similarity on word level
    words1 = set(name1.split())
    words2 = set(name2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for better matching.
    
    Args:
        filename: Input filename
        
    Returns:
        Normalized filename
    """
    # Convert to lowercase
    filename = filename.lower()
    
    # Remove common subtitle/audio markers
    patterns_to_remove = [
        r'\[.*?\]',  # Remove bracketed content
        r'\(.*?\)',  # Remove parenthetical content  
        r'\..*$',    # Remove extensions (should already be removed)
        r'_+',       # Replace multiple underscores with single
        r'-+',       # Replace multiple dashes with single
        r'\s+',      # Replace multiple spaces with single
    ]
    
    for pattern in patterns_to_remove:
        filename = re.sub(pattern, ' ', filename)
    
    # Clean up whitespace
    filename = filename.strip()
    
    return filename


def validate_audio_subtitle_pair(audio_file: Path, subtitle_file: Path) -> bool:
    """
    Validate that an audio-subtitle pair is valid.
    
    Args:
        audio_file: Path to audio file
        subtitle_file: Path to subtitle file
        
    Returns:
        True if pair is valid
    """
    # Check if both files exist
    if not audio_file.exists():
        logging.warning(f"Audio file not found: {audio_file}")
        return False
    
    if not subtitle_file.exists():
        logging.warning(f"Subtitle file not found: {subtitle_file}")
        return False
    
    # Check file sizes (basic sanity check)
    if audio_file.stat().st_size == 0:
        logging.warning(f"Audio file is empty: {audio_file}")
        return False
    
    if subtitle_file.stat().st_size == 0:
        logging.warning(f"Subtitle file is empty: {subtitle_file}")
        return False
    
    return True


def get_directory_stats(data_dir: str, config: Optional[DataConfig] = None) -> Dict:
    """
    Get statistics about files in a directory.
    
    Args:
        data_dir: Directory to analyze
        config: Data configuration (optional)
        
    Returns:
        Dictionary with statistics
    """
    if config is None:
        from config.data import DataConfig
        config = DataConfig()
    
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"error": f"Directory not found: {data_dir}"}
    
    stats = {
        "total_files": 0,
        "audio_files": 0,
        "subtitle_files": 0,
        "other_files": 0,
        "total_size_mb": 0,
        "pairs_found": 0,
        "unpaired_audio": 0,
        "unpaired_subtitles": 0,
        "file_extensions": {},
    }
    
    # Count all files
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            stats["total_files"] += 1
            stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
            
            ext = file_path.suffix.lower()
            stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1
            
            if ext in config.audio_extensions:
                stats["audio_files"] += 1
            elif ext in config.subtitle_extensions:
                stats["subtitle_files"] += 1
            else:
                stats["other_files"] += 1
    
    # Count pairs
    pairs = find_audio_subtitle_pairs(data_dir, config)
    stats["pairs_found"] = len(pairs)
    stats["unpaired_audio"] = stats["audio_files"] - stats["pairs_found"]
    stats["unpaired_subtitles"] = stats["subtitle_files"] - stats["pairs_found"]
    
    return stats


def create_file_manifest(data_dir: str, output_path: Optional[str] = None) -> Dict:
    """
    Create a manifest of all audio-subtitle pairs in a directory.
    
    Args:
        data_dir: Directory to scan
        output_path: Optional path to save manifest
        
    Returns:
        Manifest dictionary
    """
    pairs = find_audio_subtitle_pairs(data_dir)
    
    manifest = {
        "data_directory": str(Path(data_dir).absolute()),
        "created_at": str(Path().absolute()),
        "total_pairs": len(pairs),
        "pairs": []
    }
    
    for i, (audio_file, subtitle_file) in enumerate(pairs):
        pair_info = {
            "id": i,
            "audio_file": {
                "path": str(audio_file.absolute()),
                "relative_path": str(audio_file.relative_to(data_dir)),
                "size_mb": audio_file.stat().st_size / (1024 * 1024),
                "extension": audio_file.suffix.lower()
            },
            "subtitle_file": {
                "path": str(subtitle_file.absolute()),
                "relative_path": str(subtitle_file.relative_to(data_dir)),
                "size_mb": subtitle_file.stat().st_size / (1024 * 1024),
                "extension": subtitle_file.suffix.lower()
            },
            "valid": validate_audio_subtitle_pair(audio_file, subtitle_file)
        }
        manifest["pairs"].append(pair_info)
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Manifest saved to: {output_path}")
    
    return manifest