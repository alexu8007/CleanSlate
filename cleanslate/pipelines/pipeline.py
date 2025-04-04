"""
Pipeline module for creating and managing data cleaning workflows.
"""

import uuid
import copy
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
import logging
import json

from cleanslate.core.config import Config
from cleanslate.core.dataset import Dataset

logger = logging.getLogger("cleanslate.pipelines")


class PipelineStep:
    """
    Class representing a single step in a cleaning pipeline.
    """
    
    def __init__(self, 
                 name: str, 
                 transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 description: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline step.
        
        Args:
            name: Name of the step.
            transform_fn: Function that transforms a DataFrame.
            description: Description of the step.
            params: Parameters for the transform function.
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.transform_fn = transform_fn
        self.description = description or f"Apply {name} transformation"
        self.params = params or {}
        self.created_at = datetime.now()
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to a DataFrame.
        
        Args:
            data: DataFrame to transform.
        
        Returns:
            Transformed DataFrame.
        """
        return self.transform_fn(data, **self.params)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline step to a dictionary.
        
        Returns:
            Dictionary representation of the pipeline step.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], transform_registry: Dict[str, Callable]) -> 'PipelineStep':
        """
        Create a pipeline step from a dictionary.
        
        Args:
            data: Dictionary containing pipeline step data.
            transform_registry: Registry of transform functions.
        
        Returns:
            PipelineStep object.
        """
        if data["name"] not in transform_registry:
            raise ValueError(f"Transform function '{data['name']}' not found in registry.")
        
        step = cls(
            name=data["name"],
            transform_fn=transform_registry[data["name"]],
            description=data.get("description"),
            params=data.get("params", {})
        )
        
        step.id = data["id"]
        step.created_at = datetime.fromisoformat(data["created_at"])
        
        return step


class Pipeline:
    """
    Class for creating and managing data cleaning pipelines.
    """
    
    # Registry of built-in transform functions
    TRANSFORM_REGISTRY = {}
    
    @classmethod
    def register_transform(cls, name: str, transform_fn: Callable):
        """
        Register a transform function in the registry.
        
        Args:
            name: Name of the transform function.
            transform_fn: Transform function.
        """
        cls.TRANSFORM_REGISTRY[name] = transform_fn
    
    def __init__(self, config: Config, name: Optional[str] = None):
        """
        Initialize a pipeline.
        
        Args:
            config: Configuration object.
            name: Name of the pipeline. If not provided, a default name will be generated.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"pipeline_{self.id[:8]}"
        self.config = config
        self.steps = []
        self.created_at = datetime.now()
        self.modified_at = self.created_at
        self.description = ""
        self.metadata = {}
    
    def add_step(self, 
                 name: str, 
                 transform_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 description: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None,
                 position: Optional[int] = None) -> PipelineStep:
        """
        Add a step to the pipeline.
        
        Args:
            name: Name of the step.
            transform_fn: Function that transforms a DataFrame. If not provided, the function will be looked up in the registry.
            description: Description of the step.
            params: Parameters for the transform function.
            position: Position to insert the step at. If not provided, the step will be added at the end.
        
        Returns:
            Added pipeline step.
        """
        # Look up transform function in registry if not provided
        if transform_fn is None:
            if name not in self.TRANSFORM_REGISTRY:
                raise ValueError(f"Transform function '{name}' not found in registry.")
            transform_fn = self.TRANSFORM_REGISTRY[name]
        
        # Create pipeline step
        step = PipelineStep(name, transform_fn, description, params)
        
        # Add step to pipeline
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        # Update modified timestamp
        self.modified_at = datetime.now()
        
        logger.info("Added step '%s' to pipeline '%s'", step.name, self.name)
        return step
    
    def remove_step(self, step_id: str) -> None:
        """
        Remove a step from the pipeline.
        
        Args:
            step_id: ID of the step to remove.
        """
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                del self.steps[i]
                self.modified_at = datetime.now()
                logger.info("Removed step '%s' from pipeline '%s'", step.name, self.name)
                return
        
        raise ValueError(f"Step with ID '{step_id}' not found in pipeline.")
    
    def move_step(self, step_id: str, new_position: int) -> None:
        """
        Move a step to a new position in the pipeline.
        
        Args:
            step_id: ID of the step to move.
            new_position: New position for the step.
        """
        if new_position < 0 or new_position >= len(self.steps):
            raise ValueError(f"Invalid position: {new_position}. Must be between 0 and {len(self.steps) - 1}.")
        
        # Find the step
        step_idx = None
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                step_idx = i
                break
        
        if step_idx is None:
            raise ValueError(f"Step with ID '{step_id}' not found in pipeline.")
        
        # Move the step
        step = self.steps.pop(step_idx)
        self.steps.insert(new_position, step)
        
        # Update modified timestamp
        self.modified_at = datetime.now()
        
        logger.info("Moved step '%s' to position %d in pipeline '%s'", step.name, new_position, self.name)
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the pipeline to a DataFrame.
        
        Args:
            data: DataFrame to transform.
        
        Returns:
            Transformed DataFrame.
        """
        if not self.steps:
            logger.warning("Pipeline '%s' has no steps. Returning original data.", self.name)
            return data
        
        # Make a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Apply each step in sequence
        for step in self.steps:
            try:
                result = step.apply(result)
                logger.info("Applied step '%s' in pipeline '%s'", step.name, self.name)
            except Exception as e:
                logger.error("Error applying step '%s' in pipeline '%s': %s", step.name, self.name, str(e))
                raise RuntimeError(f"Error in pipeline step '{step.name}': {str(e)}") from e
        
        return result
    
    def build_auto_pipeline(self, dataset: Dataset) -> None:
        """
        Build an automatic cleaning pipeline based on the dataset.
        
        Args:
            dataset: Dataset to build the pipeline for.
        """
        # Clear existing steps
        self.steps = []
        
        # 1. Drop columns with too many missing values
        self.add_step(
            name="drop_high_missing_columns",
            description="Drop columns with high missing rate",
            params={"threshold": 0.7}  # Drop columns with >70% missing values
        )
        
        # 2. Fill missing values
        self.add_step(
            name="fill_missing_values",
            description="Fill missing values",
            params={"strategy": "auto"}  # Auto-detect strategy based on column type
        )
        
        # 3. Remove outliers
        self.add_step(
            name="remove_outliers",
            description="Remove or cap outliers",
            params={"method": "cap", "threshold": 3}  # Cap at 3 std
        )
        
        # 4. Fix data types
        self.add_step(
            name="fix_data_types",
            description="Fix inconsistent data types"
        )
        
        # 5. Drop duplicate rows
        self.add_step(
            name="drop_duplicates",
            description="Remove duplicate rows"
        )
        
        # Update pipeline metadata
        self.name = f"auto_pipeline_{dataset.name}"
        self.description = f"Automatic cleaning pipeline for dataset '{dataset.name}'"
        self.modified_at = datetime.now()
        
        logger.info("Built automatic pipeline '%s' for dataset '%s'", self.name, dataset.name)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline to a dictionary.
        
        Returns:
            Dictionary representation of the pipeline.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Config) -> 'Pipeline':
        """
        Create a pipeline from a dictionary.
        
        Args:
            data: Dictionary containing pipeline data.
            config: Configuration object.
        
        Returns:
            Pipeline object.
        """
        pipeline = cls(config, name=data["name"])
        
        pipeline.id = data["id"]
        pipeline.description = data.get("description", "")
        pipeline.created_at = datetime.fromisoformat(data["created_at"])
        pipeline.modified_at = datetime.fromisoformat(data["modified_at"])
        pipeline.metadata = data.get("metadata", {})
        
        # Add steps
        for step_data in data.get("steps", []):
            step = PipelineStep.from_dict(step_data, cls.TRANSFORM_REGISTRY)
            pipeline.steps.append(step)
        
        return pipeline
    
    def save(self, file_path: str) -> None:
        """
        Save the pipeline to a file.
        
        Args:
            file_path: Path to save the pipeline to.
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        
        logger.info("Saved pipeline '%s' to %s", self.name, file_path)
    
    @classmethod
    def load(cls, file_path: str, config: Config) -> 'Pipeline':
        """
        Load a pipeline from a file.
        
        Args:
            file_path: Path to load the pipeline from.
            config: Configuration object.
        
        Returns:
            Pipeline object.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        pipeline = cls.from_dict(data, config)
        
        logger.info("Loaded pipeline '%s' from %s", pipeline.name, file_path)
        return pipeline


# Register built-in transforms

def drop_high_missing_columns(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns with high missing rate.
    
    Args:
        data: DataFrame to transform.
        threshold: Missing rate threshold. Columns with missing rate > threshold will be dropped.
    
    Returns:
        Transformed DataFrame.
    """
    missing_rates = data.isna().mean()
    cols_to_drop = missing_rates[missing_rates > threshold].index
    return data.drop(columns=cols_to_drop)

Pipeline.register_transform("drop_high_missing_columns", drop_high_missing_columns)


def fill_missing_values(data: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Fill missing values in a DataFrame.
    
    Args:
        data: DataFrame to transform.
        strategy: Strategy for filling missing values.
            - "auto": Automatically choose strategy based on column type.
            - "mean": Fill with column mean (numeric).
            - "median": Fill with column median (numeric).
            - "mode": Fill with column mode (categorical).
            - "zero": Fill with 0 (numeric).
            - "constant": Fill with a constant value.
    
    Returns:
        Transformed DataFrame.
    """
    result = data.copy()
    
    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue
        
        if strategy == "auto":
            # Choose strategy based on column type
            if pd.api.types.is_numeric_dtype(result[col]):
                # Use median for numeric columns
                result[col] = result[col].fillna(result[col].median())
            else:
                # Use mode for categorical columns
                mode_value = result[col].mode().iloc[0] if not result[col].mode().empty else None
                result[col] = result[col].fillna(mode_value)
        elif strategy == "mean":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(result[col].median())
        elif strategy == "mode":
            mode_value = result[col].mode().iloc[0] if not result[col].mode().empty else None
            result[col] = result[col].fillna(mode_value)
        elif strategy == "zero":
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
    
    return result

Pipeline.register_transform("fill_missing_values", fill_missing_values)


def remove_outliers(data: pd.DataFrame, method: str = "cap", threshold: float = 3.0, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove or cap outliers in a DataFrame.
    
    Args:
        data: DataFrame to transform.
        method: Method for handling outliers.
            - "cap": Cap outliers at threshold * std from mean.
            - "remove": Remove rows with outliers.
            - "iqr": Use IQR method to identify outliers.
        threshold: Threshold for outlier detection.
        cols: Columns to process. If None, all numeric columns will be processed.
    
    Returns:
        Transformed DataFrame.
    """
    result = data.copy()
    
    # If no columns specified, use all numeric columns
    if cols is None:
        cols = result.select_dtypes(include=np.number).columns.tolist()
    
    # Process each column
    for col in cols:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
            continue
        
        if method == "cap":
            # Cap outliers at threshold * std from mean
            mean = result[col].mean()
            std = result[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == "iqr":
            # Use IQR method
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == "remove":
            # Identify outliers
            mean = result[col].mean()
            std = result[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Create outlier mask
            outlier_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
            
            # Remove rows with outliers
            result = result[~outlier_mask]
    
    return result

Pipeline.register_transform("remove_outliers", remove_outliers)


def fix_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix inconsistent data types in a DataFrame.
    
    Args:
        data: DataFrame to transform.
    
    Returns:
        Transformed DataFrame.
    """
    result = data.copy()
    
    for col in result.columns:
        # Try to convert string columns to numeric if they contain numeric strings
        if pd.api.types.is_object_dtype(result[col]):
            # Check if column contains numeric values
            try:
                pd.to_numeric(result[col], errors='raise')
                # If no error, convert to numeric
                result[col] = pd.to_numeric(result[col])
            except:
                pass
            
            # Check if column contains boolean values
            if set(result[col].dropna().unique()).issubset(set(['True', 'False', True, False, 0, 1, '0', '1', 'yes', 'no', 'y', 'n'])):
                # Convert to boolean
                bool_map = {
                    'True': True, 'False': False,
                    True: True, False: False,
                    1: True, 0: False,
                    '1': True, '0': False,
                    'yes': True, 'no': False,
                    'y': True, 'n': False
                }
                result[col] = result[col].map(bool_map)
            
            # Check if column contains date values
            try:
                pd.to_datetime(result[col], errors='raise')
                # If no error, convert to datetime
                result[col] = pd.to_datetime(result[col])
            except:
                pass
    
    return result

Pipeline.register_transform("fix_data_types", fix_data_types)


def drop_duplicates(data: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        data: DataFrame to transform.
        subset: Columns to consider for identifying duplicates. If None, all columns are used.
        keep: Which duplicates to keep.
            - "first": Keep first occurrence.
            - "last": Keep last occurrence.
            - False: Drop all duplicates.
    
    Returns:
        Transformed DataFrame with duplicates removed.
    """
    return data.drop_duplicates(subset=subset, keep=keep)

Pipeline.register_transform("drop_duplicates", drop_duplicates)


def standardize_column_names(data: pd.DataFrame, case: str = 'lower', replace_spaces: bool = True) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame.
    
    Args:
        data: DataFrame to transform.
        case: Case to convert column names to.
            - "lower": Convert to lowercase.
            - "upper": Convert to uppercase.
            - "title": Convert to title case.
        replace_spaces: Whether to replace spaces with underscores.
    
    Returns:
        Transformed DataFrame with standardized column names.
    """
    result = data.copy()
    
    new_columns = {}
    for col in result.columns:
        new_col = col
        
        # Convert case
        if case == 'lower':
            new_col = new_col.lower()
        elif case == 'upper':
            new_col = new_col.upper()
        elif case == 'title':
            new_col = new_col.title()
        
        # Replace spaces
        if replace_spaces:
            new_col = new_col.replace(' ', '_')
        
        new_columns[col] = new_col
    
    result = result.rename(columns=new_columns)
    return result

Pipeline.register_transform("standardize_column_names", standardize_column_names) 