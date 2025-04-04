"""
Configuration module for CleanSlate.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, validator


class CleaningConfig(BaseModel):
    """Configuration for data cleaning operations."""
    remove_outliers: bool = True
    fill_missing_values: bool = True
    standardize_columns: bool = True
    drop_duplicate_rows: bool = True
    max_outlier_percentage: float = 0.05
    missing_value_strategy: str = "auto"  # auto, mean, median, mode, constant

    @validator("missing_value_strategy")
    def validate_missing_strategy(cls, v):
        valid_strategies = ["auto", "mean", "median", "mode", "constant", "knn", "regression"]
        if v not in valid_strategies:
            raise ValueError(f"Missing value strategy must be one of {valid_strategies}")
        return v


class PipelineConfig(BaseModel):
    """Configuration for data cleaning pipelines."""
    auto_save: bool = True
    version_control: bool = True
    max_pipeline_steps: int = 20
    parallel_execution: bool = True
    pipeline_timeout_seconds: int = 3600


class UIConfig(BaseModel):
    """Configuration for the user interface."""
    theme: str = "light"
    show_advanced_options: bool = False
    max_rows_preview: int = 100
    enable_realtime_updates: bool = True


class Config:
    """Main configuration class for CleanSlate."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        self.cleaning = CleaningConfig()
        self.pipeline = PipelineConfig()
        self.ui = UIConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Update configuration from loaded data
        if "cleaning" in config_data:
            self.cleaning = CleaningConfig(**config_data["cleaning"])
        
        if "pipeline" in config_data:
            self.pipeline = PipelineConfig(**config_data["pipeline"])
        
        if "ui" in config_data:
            self.ui = UIConfig(**config_data["ui"])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "cleaning": self.cleaning.dict(),
            "pipeline": self.pipeline.dict(),
            "ui": self.ui.dict()
        }
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save the configuration file.
        """
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False) 