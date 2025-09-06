import torch
import platform
import logging
from typing import Optional


def get_available_device(device: str = "auto") -> torch.device:
    """
    Detect and return the best available device for PyTorch.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: Best available device
    """
    if device != "auto":
        # User specified a device, try to use it
        if device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logging.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        elif device == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                logging.warning("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")
        elif device == "cpu":
            return torch.device("cpu")
        else:
            logging.warning(f"Unknown device '{device}', falling back to auto-detection")
    
    # Auto-detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_name = torch.cuda.get_device_name(0)
        logging.info(f"Using CUDA device: {cuda_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    
    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed information about the device.
    
    Args:
        device: Device to get info for (default: auto-detect)
        
    Returns:
        dict: Device information
    """
    if device is None:
        device = get_available_device()
    
    info = {
        "device": str(device),
        "type": device.type,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(device),
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_cached": torch.cuda.memory_reserved(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })
    elif device.type == "cpu":
        info.update({
            "cpu_count": torch.get_num_threads(),
        })
    
    return info


def optimize_device_settings(device: torch.device) -> None:
    """
    Apply device-specific optimizations.
    
    Args:
        device: Device to optimize for
    """
    if device.type == "cuda":
        # Enable optimizations for CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logging.info("Enabled CUDA optimizations")
        
    elif device.type == "mps":
        # MPS-specific optimizations (if any)
        logging.info("Using MPS device optimizations")
        
    elif device.type == "cpu":
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        logging.info(f"Using {torch.get_num_threads()} CPU threads")


def move_to_device(obj, device: torch.device):
    """
    Safely move tensors or models to device.
    
    Args:
        obj: Object to move (tensor, model, etc.)
        device: Target device
        
    Returns:
        Object moved to device
    """
    try:
        return obj.to(device)
    except Exception as e:
        logging.error(f"Failed to move object to {device}: {e}")
        return obj


def check_device_compatibility() -> dict:
    """
    Check compatibility and performance characteristics of available devices.
    
    Returns:
        dict: Compatibility report
    """
    report = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_available": True,
        "recommended_device": None,
        "warnings": [],
    }
    
    if report["cuda_available"]:
        try:
            # Test CUDA functionality
            test_tensor = torch.ones(100, 100, device="cuda")
            _ = torch.mm(test_tensor, test_tensor)
            report["cuda_functional"] = True
            report["recommended_device"] = "cuda"
        except Exception as e:
            report["cuda_functional"] = False
            report["warnings"].append(f"CUDA available but not functional: {e}")
    
    if report["mps_available"] and report["recommended_device"] is None:
        try:
            # Test MPS functionality
            test_tensor = torch.ones(100, 100, device="mps")
            _ = torch.mm(test_tensor, test_tensor)
            report["mps_functional"] = True
            report["recommended_device"] = "mps"
        except Exception as e:
            report["mps_functional"] = False
            report["warnings"].append(f"MPS available but not functional: {e}")
    
    if report["recommended_device"] is None:
        report["recommended_device"] = "cpu"
    
    return report


if __name__ == "__main__":
    # Test device detection
    logging.basicConfig(level=logging.INFO)
    
    print("Device Compatibility Report:")
    print("=" * 40)
    
    compatibility = check_device_compatibility()
    for key, value in compatibility.items():
        print(f"{key}: {value}")
    
    print("\nDevice Information:")
    print("=" * 40)
    device = get_available_device()
    info = get_device_info(device)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"\nOptimizing for {device}...")
    optimize_device_settings(device)