import torch
from typing import Optional, Union


class EMA:
    """
    Exponential Moving Average utility class.
    
    Maintains an exponential moving average of a scalar or tensor value.
    Supports PyTorch checkpointing with state_dict/load_state_dict.
    """
    
    def __init__(self, momentum: float = 0.9, device: Optional[torch.device] = None):
        """
        Initialize EMA.
        
        Args:
            momentum: EMA momentum parameter (higher = smoother)
            device: Device to store the EMA value on
        """
        self.momentum = momentum
        self.device = device
        self.value = None
        self.initialized = False
    
    def update(self, new_value: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Update EMA with new value.
        
        Args:
            new_value: New value to incorporate into the EMA
            
        Returns:
            Current EMA value
        """
        if isinstance(new_value, torch.Tensor):
            if self.device is None:
                self.device = new_value.device
            new_value = new_value.to(self.device)
        
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * new_value
        
        return self.value
    
    def get(self) -> Optional[Union[float, torch.Tensor]]:
        """Get current EMA value."""
        return self.value
    
    def reset(self):
        """Reset EMA state."""
        self.value = None
        self.initialized = False
    
    def to(self, device: torch.device):
        """Move EMA to device."""
        self.device = device
        if self.value is not None and isinstance(self.value, torch.Tensor):
            self.value = self.value.to(device)
        return self
    
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'momentum': self.momentum,
            'value': self.value,
            'initialized': self.initialized,
            'device': str(self.device) if self.device else None
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict from checkpoint."""
        self.momentum = state_dict['momentum']
        self.value = state_dict['value']
        self.initialized = state_dict['initialized']
        
        if state_dict.get('device') is not None:
            self.device = torch.device(state_dict['device'])
            if self.value is not None and isinstance(self.value, torch.Tensor):
                self.value = self.value.to(self.device)


if __name__ == "__main__":
    # Test EMA utility
    print("Testing EMA utility...")
    
    # Test with float values
    ema_float = EMA(momentum=0.9)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    print("Float EMA test:")
    for val in values:
        result = ema_float.update(val)
        print(f"Value: {val}, EMA: {result:.4f}")
    
    # Test with tensor values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema_tensor = EMA(momentum=0.8, device=device)
    
    print(f"\nTensor EMA test (device: {device}):")
    for i, val in enumerate(values):
        tensor_val = torch.tensor(val, device=device)
        result = ema_tensor.update(tensor_val)
        print(f"Value: {val}, EMA: {result.item():.4f}")
    
    # Test state dict
    print("\nState dict test:")
    state = ema_float.state_dict()
    print(f"State dict: {state}")
    
    new_ema = EMA()
    new_ema.load_state_dict(state)
    print(f"Loaded EMA value: {new_ema.get()}")
    
    print("EMA utility test completed!")