"""
Dataset module for handling data in CleanSlate.
"""

import os
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime


class Dataset:
    """Class for handling datasets in CleanSlate."""
    
    def __init__(self, data: Optional[Union[pd.DataFrame, str]] = None, name: Optional[str] = None):
        """
        Initialize a dataset.
        
        Args:
            data: Data to initialize the dataset with. Can be a pandas DataFrame or a path to a file.
            name: Name of the dataset. If not provided, a UUID will be generated.
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"dataset_{self.id[:8]}"
        self.created_at = datetime.now()
        self.modified_at = self.created_at
        self.metadata = {}
        self.history = []
        
        # Initialize data
        if data is None:
            self.data = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str):
            self.load_from_file(data)
        else:
            raise TypeError("Data must be a pandas DataFrame or a path to a file.")
        
        # Initialize basic metadata
        self._update_metadata()
    
    def _update_metadata(self) -> None:
        """Update dataset metadata."""
        self.metadata["num_rows"] = len(self.data)
        self.metadata["num_columns"] = len(self.data.columns)
        self.metadata["column_types"] = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        self.metadata["modified_at"] = datetime.now()
        self.modified_at = self.metadata["modified_at"]
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the file to load data from.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[-1].lower()
        
        if file_ext == ".csv":
            self.data = pd.read_csv(file_path)
        elif file_ext == ".parquet":
            self.data = pd.read_parquet(file_path)
        elif file_ext in [".xls", ".xlsx"]:
            self.data = pd.read_excel(file_path)
        elif file_ext == ".json":
            self.data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        self.metadata["source_file"] = file_path
        self._update_metadata()
        self._log_operation(f"Loaded data from {file_path}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save data to a file.
        
        Args:
            file_path: Path to save the data to.
        """
        file_ext = os.path.splitext(file_path)[-1].lower()
        
        if file_ext == ".csv":
            self.data.to_csv(file_path, index=False)
        elif file_ext == ".parquet":
            self.data.to_parquet(file_path, index=False)
        elif file_ext in [".xls", ".xlsx"]:
            self.data.to_excel(file_path, index=False)
        elif file_ext == ".json":
            self.data.to_json(file_path, orient="records")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        self._log_operation(f"Saved data to {file_path}")
    
    def _log_operation(self, operation: str) -> None:
        """
        Log an operation on the dataset.
        
        Args:
            operation: Description of the operation.
        """
        self.history.append({
            "timestamp": datetime.now(),
            "operation": operation
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary containing summary information.
        """
        summary = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "num_rows": self.metadata["num_rows"],
            "num_columns": self.metadata["num_columns"],
            "column_types": self.metadata["column_types"],
            "columns": list(self.data.columns),
            "missing_values": self.data.isna().sum().to_dict(),
            "operations_count": len(self.history)
        }
        
        # Add numerical column statistics if data is not empty
        if not self.data.empty:
            numeric_cols = self.data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                summary["numeric_stats"] = self.data[numeric_cols].describe().to_dict()
        
        return summary
    
    def get_profile(self) -> Dict[str, Any]:
        """
        Get a detailed profile of the dataset.
        
        Returns:
            Dictionary containing detailed profile information.
        """
        profile = self.get_summary()
        
        # Add more detailed information
        if not self.data.empty:
            # Column-wise information
            column_info = {}
            for col in self.data.columns:
                col_data = self.data[col]
                col_info = {
                    "dtype": str(col_data.dtype),
                    "missing_count": col_data.isna().sum(),
                    "missing_percentage": (col_data.isna().sum() / len(col_data)) * 100,
                    "unique_count": col_data.nunique()
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    col_info.update({
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "mean": col_data.mean(),
                        "median": col_data.median(),
                        "std": col_data.std()
                    })
                
                if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                    # Get value counts for categorical data (top 10)
                    value_counts = col_data.value_counts().head(10).to_dict()
                    col_info["top_values"] = value_counts
                
                column_info[col] = col_info
            
            profile["column_details"] = column_info
            
            # Correlation matrix for numeric columns
            numeric_cols = self.data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 1:
                profile["correlation_matrix"] = self.data[numeric_cols].corr().to_dict()
        
        return profile
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Get the first n rows of the dataset.
        
        Args:
            n: Number of rows to return.
        
        Returns:
            DataFrame containing the first n rows.
        """
        return self.data.head(n)
    
    def sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get a random sample of n rows from the dataset.
        
        Args:
            n: Number of rows to sample.
        
        Returns:
            DataFrame containing the sampled rows.
        """
        return self.data.sample(min(n, len(self.data))) 