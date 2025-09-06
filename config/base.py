import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class with YAML loading capabilities."""
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        pass
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary with type conversion."""
        # Get the expected types from the dataclass fields
        import dataclasses
        if dataclasses.is_dataclass(cls):
            processed_dict = {}
            for field in dataclasses.fields(cls):
                field_name = field.name
                if field_name in config_dict:
                    value = config_dict[field_name]
                    # Convert string values to expected types
                    if field.type == float and isinstance(value, str):
                        processed_dict[field_name] = float(value)
                    elif field.type == int and isinstance(value, str):
                        processed_dict[field_name] = int(value)
                    elif field.type == bool and isinstance(value, str):
                        processed_dict[field_name] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        processed_dict[field_name] = value
                elif field.default != dataclasses.MISSING:
                    # Use default value if not provided
                    processed_dict[field_name] = field.default
                elif field.default_factory != dataclasses.MISSING:
                    # Use default factory if not provided
                    processed_dict[field_name] = field.default_factory()
            # Add any extra keys that aren't fields
            for key, value in config_dict.items():
                if key not in processed_dict:
                    processed_dict[key] = value
            return cls(**processed_dict)
        else:
            return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        self.validate()


def load_config(config_path: str) -> 'Config':
    """Load full configuration from YAML file."""
    from .config import Config
    return Config.from_yaml(config_path)