"""
Version control module for tracking dataset changes and managing rollbacks.
"""

import os
import shutil
import json
import uuid
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging

from cleanslate.core.config import Config
from cleanslate.core.dataset import Dataset
from cleanslate.pipelines.pipeline import Pipeline

logger = logging.getLogger("cleanslate.versioning")

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types."""
    
    def default(self, obj):
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle UUID objects
        if isinstance(obj, uuid.UUID):
            return str(obj)
        
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return "DataFrame: rows={}, cols={}".format(len(obj), len(obj.columns))
            
        # Handle objects with to_dict method (like Dataset or Pipeline)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
            
        # Let the base class handle it or raise TypeError
        return super().default(obj)

class VersionControl:
    """Class for managing dataset versions and rollbacks."""
    
    def __init__(self, config: Config, storage_dir: Optional[str] = None):
        """
        Initialize version control.
        
        Args:
            config: Configuration object.
            storage_dir: Directory to store version data. If not provided, a default directory will be used.
        """
        self.config = config
        self.storage_dir = storage_dir or os.path.expanduser("~/.cleanslate/versions")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize versions index
        self.versions_index = self._load_versions_index()
        
        logger.info("Initialized version control with storage directory: %s", self.storage_dir)
    
    def _load_versions_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load versions index from file.
        
        Returns:
            Dictionary mapping dataset IDs to version information.
        """
        index_file = os.path.join(self.storage_dir, "versions_index.json")
        
        if not os.path.exists(index_file):
            return {}
        
        try:
            with open(index_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to load versions index. Creating new index.")
            return {}
    
    def _save_versions_index(self) -> None:
        """Save versions index to file."""
        index_file = os.path.join(self.storage_dir, "versions_index.json")
        
        with open(index_file, "w") as f:
            json.dump(self.versions_index, f, indent=4)
    
    def save_version(self, dataset: Dataset, pipeline: Pipeline) -> str:
        """
        Save a version of a dataset.
        
        Args:
            dataset: Dataset to save.
            pipeline: Pipeline used to create the dataset.
        
        Returns:
            Version ID.
        """
        # Create version metadata
        version_metadata = {
            "id": str(uuid.uuid4()),
            "dataset_id": dataset.id,
            "pipeline_id": pipeline.id,
            "created_at": datetime.now(),
            "dataset_metadata": dataset.metadata,
            "pipeline_metadata": pipeline.to_dict()
        }
        
        # Save metadata to JSON file
        version_dir = os.path.join(self.storage_dir, version_metadata["id"])
        os.makedirs(version_dir, exist_ok=True)
        
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(version_metadata, f, indent=4, cls=CustomJSONEncoder)
        
        return version_metadata["id"]
    
    def get_versions(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Get all versions of a dataset.
        
        Args:
            dataset: Dataset to get versions for.
        
        Returns:
            List of version information dictionaries.
        """
        if dataset.id not in self.versions_index:
            return []
        
        # Sort versions by creation time (newest first)
        versions = sorted(
            self.versions_index[dataset.id],
            key=lambda v: v["created_at"],
            reverse=True
        )
        
        return versions
    
    def get_version_metadata(self, version_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific version.
        
        Args:
            version_id: ID of the version.
        
        Returns:
            Dictionary containing version metadata.
        """
        metadata_file = os.path.join(self.storage_dir, version_id, "metadata.json")
        
        if not os.path.exists(metadata_file):
            raise ValueError(f"Version with ID '{version_id}' not found.")
        
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def load_version_data(self, version_id: str) -> pd.DataFrame:
        """
        Load data for a specific version.
        
        Args:
            version_id: ID of the version.
        
        Returns:
            DataFrame containing version data.
        """
        # Get version metadata
        metadata = self.get_version_metadata(version_id)
        
        # Load data
        data_file = os.path.join(self.storage_dir, version_id, metadata["data_file"])
        
        if not os.path.exists(data_file):
            raise ValueError(f"Data file for version '{version_id}' not found.")
        
        return pd.read_parquet(data_file)
    
    def restore_version(self, dataset: Dataset, version_id: str) -> Dataset:
        """
        Restore a dataset to a previous version.
        
        Args:
            dataset: Dataset to restore.
            version_id: ID of the version to restore.
        
        Returns:
            Restored dataset.
        """
        # Get version metadata
        metadata = self.get_version_metadata(version_id)
        
        # Load version data
        data = self.load_version_data(version_id)
        
        # Create new dataset with version data
        restored_dataset = Dataset(data, f"{dataset.name}_restored")
        
        # Add metadata
        restored_dataset.metadata = metadata["metadata"]
        restored_dataset._log_operation(f"Restored from version {version_id}")
        
        logger.info("Restored dataset '%s' to version '%s'", dataset.name, version_id)
        return restored_dataset
    
    def delete_version(self, version_id: str) -> None:
        """
        Delete a specific version.
        
        Args:
            version_id: ID of the version to delete.
        """
        # Get version metadata
        try:
            metadata = self.get_version_metadata(version_id)
        except ValueError:
            logger.warning("Version '%s' not found", version_id)
            return
        
        # Remove version from index
        dataset_id = metadata["dataset_id"]
        if dataset_id in self.versions_index:
            self.versions_index[dataset_id] = [
                v for v in self.versions_index[dataset_id] if v["id"] != version_id
            ]
            
            # Remove dataset from index if no versions remain
            if not self.versions_index[dataset_id]:
                del self.versions_index[dataset_id]
            
            # Save versions index
            self._save_versions_index()
        
        # Remove version directory
        version_dir = os.path.join(self.storage_dir, version_id)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        
        logger.info("Deleted version '%s'", version_id)
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.
        
        Args:
            version_id1: ID of the first version.
            version_id2: ID of the second version.
        
        Returns:
            Dictionary containing comparison results.
        """
        # Get version metadata
        metadata1 = self.get_version_metadata(version_id1)
        metadata2 = self.get_version_metadata(version_id2)
        
        # Load version data
        data1 = self.load_version_data(version_id1)
        data2 = self.load_version_data(version_id2)
        
        # Compare datasets
        comparison = {
            "version1": {
                "id": version_id1,
                "dataset_id": metadata1["dataset_id"],
                "dataset_name": metadata1["dataset_name"],
                "created_at": metadata1["created_at"],
                "pipeline_id": metadata1["pipeline_id"],
                "pipeline_name": metadata1["pipeline_name"],
                "shape": data1.shape
            },
            "version2": {
                "id": version_id2,
                "dataset_id": metadata2["dataset_id"],
                "dataset_name": metadata2["dataset_name"],
                "created_at": metadata2["created_at"],
                "pipeline_id": metadata2["pipeline_id"],
                "pipeline_name": metadata2["pipeline_name"],
                "shape": data2.shape
            },
            "differences": {},
            "summary": {}
        }
        
        # Shape comparison
        comparison["summary"]["shape_diff"] = {
            "rows": data2.shape[0] - data1.shape[0],
            "columns": data2.shape[1] - data1.shape[1]
        }
        
        # Column comparison
        comparison["summary"]["common_columns"] = list(set(data1.columns) & set(data2.columns))
        comparison["summary"]["only_in_version1"] = list(set(data1.columns) - set(data2.columns))
        comparison["summary"]["only_in_version2"] = list(set(data2.columns) - set(data1.columns))
        
        # Statistical differences for common columns
        diff_stats = {}
        for col in comparison["summary"]["common_columns"]:
            if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col]):
                diff_stats[col] = {
                    "mean_diff": data2[col].mean() - data1[col].mean(),
                    "std_diff": data2[col].std() - data1[col].std(),
                    "min_diff": data2[col].min() - data1[col].min(),
                    "max_diff": data2[col].max() - data1[col].max()
                }
        
        comparison["differences"]["statistics"] = diff_stats
        
        logger.info("Compared versions '%s' and '%s'", version_id1, version_id2)
        return comparison
    
    def get_version_history(self, dataset: Dataset, include_data: bool = False) -> List[Dict[str, Any]]:
        """
        Get the complete version history for a dataset.
        
        Args:
            dataset: Dataset to get history for.
            include_data: Whether to include dataset data in the history.
        
        Returns:
            List of version information dictionaries.
        """
        versions = self.get_versions(dataset)
        
        if not versions:
            return []
        
        # Add detailed metadata
        for version in versions:
            # Get version metadata
            metadata = self.get_version_metadata(version["id"])
            version.update(metadata)
            
            # Load data if requested
            if include_data:
                data = self.load_version_data(version["id"])
                version["data"] = data
        
        return versions
    
    def create_branch(self, dataset: Dataset, version_id: str, branch_name: str) -> Dataset:
        """
        Create a new branch from a specific version.
        
        Args:
            dataset: Original dataset.
            version_id: ID of the version to branch from.
            branch_name: Name of the new branch.
        
        Returns:
            New dataset representing the branch.
        """
        # Restore the version
        branched_dataset = self.restore_version(dataset, version_id)
        
        # Update metadata
        branched_dataset.name = f"{dataset.name}_{branch_name}"
        branched_dataset.metadata["branch"] = {
            "parent_dataset_id": dataset.id,
            "parent_dataset_name": dataset.name,
            "parent_version_id": version_id,
            "branch_name": branch_name,
            "created_at": datetime.now().isoformat()
        }
        
        # Log operation
        branched_dataset._log_operation(f"Created branch '{branch_name}' from version {version_id}")
        
        logger.info("Created branch '%s' from dataset '%s' version '%s'", branch_name, dataset.name, version_id)
        return branched_dataset 