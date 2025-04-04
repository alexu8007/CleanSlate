"""
Main CleanSlate class for data cleaning operations.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import json
import logging

from cleanslate.core.config import Config
from cleanslate.core.dataset import Dataset
from cleanslate.anomaly.detector import AnomalyDetector
from cleanslate.scoring.quality import QualityScorer
from cleanslate.pipelines.pipeline import Pipeline
from cleanslate.versioning.version_control import VersionControl


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cleanslate")


class CleanSlate:
    """
    Main class for CleanSlate data cleaning operations.
    
    This class provides a high-level interface for all CleanSlate operations,
    including loading data, detecting anomalies, cleaning data, and more.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize CleanSlate.
        
        Args:
            config_path: Path to configuration file. If not provided, default configuration is used.
        """
        self.config = Config(config_path)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.quality_scorer = QualityScorer(self.config)
        self.version_control = VersionControl(self.config)
        self.datasets = {}
        self.pipelines = {}
        
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        logger.info("CleanSlate initialized with ID: %s", self.id)
    
    def load_data(self, data_source: Union[str, pd.DataFrame], name: Optional[str] = None) -> Dataset:
        """
        Load data from a file or DataFrame.
        
        Args:
            data_source: Path to data file or pandas DataFrame.
            name: Name of the dataset. If not provided, a default name will be generated.
        
        Returns:
            Dataset object containing the loaded data.
        """
        dataset = Dataset(data_source, name)
        self.datasets[dataset.id] = dataset
        logger.info("Loaded dataset '%s' with ID: %s", dataset.name, dataset.id)
        return dataset
    
    def clean(self, dataset: Union[Dataset, str], pipeline_id: Optional[str] = None) -> Dataset:
        """
        Clean a dataset using a specified pipeline or automatic cleaning.
        
        Args:
            dataset: Dataset object or dataset ID to clean.
            pipeline_id: ID of the pipeline to use. If not provided, automatic cleaning is performed.
        
        Returns:
            Cleaned dataset.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Create a copy of the dataset for cleaning
        cleaned_dataset = Dataset(dataset.data.copy(), f"{dataset.name}_cleaned")
        
        # Use existing pipeline if provided, otherwise create a new pipeline for automatic cleaning
        if pipeline_id:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline with ID '{pipeline_id}' not found.")
            pipeline = self.pipelines[pipeline_id]
        else:
            # Create a pipeline for automatic cleaning
            pipeline = Pipeline(self.config)
            pipeline.build_auto_pipeline(dataset)
            self.pipelines[pipeline.id] = pipeline
        
        # Apply the pipeline to the dataset
        cleaned_data = pipeline.apply(cleaned_dataset.data)
        cleaned_dataset.data = cleaned_data
        
        # Store in version control if enabled
        if self.config.pipeline.version_control:
            self.version_control.save_version(cleaned_dataset, pipeline)
        
        # Log operations
        cleaned_dataset._log_operation(f"Applied pipeline '{pipeline.name}' (ID: {pipeline.id})")
        cleaned_dataset._update_metadata()
        
        # Store the cleaned dataset
        self.datasets[cleaned_dataset.id] = cleaned_dataset
        
        logger.info("Cleaned dataset '%s' with pipeline '%s'", dataset.name, pipeline.name)
        return cleaned_dataset
    
    def detect_anomalies(self, dataset: Union[Dataset, str], columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in a dataset.
        
        Args:
            dataset: Dataset object or dataset ID to analyze.
            columns: Columns to analyze. If not provided, all columns are analyzed.
        
        Returns:
            Dictionary containing anomaly detection results.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect(dataset.data, columns)
        
        # Log operation
        dataset._log_operation(f"Detected anomalies in {len(anomalies['columns'])} columns")
        
        logger.info("Detected anomalies in dataset '%s'", dataset.name)
        return anomalies
    
    def score(self, dataset: Union[Dataset, str]) -> Dict[str, float]:
        """
        Score the quality of a dataset.
        
        Args:
            dataset: Dataset object or dataset ID to score.
        
        Returns:
            Dictionary containing quality scores.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Score the dataset
        scores = self.quality_scorer.score(dataset.data)
        
        # Log operation
        dataset._log_operation(f"Scored dataset quality (overall: {scores['overall']:.2f})")
        
        logger.info("Scored dataset '%s' with overall quality: %.2f", dataset.name, scores["overall"])
        return scores
    
    def create_pipeline(self, name: Optional[str] = None) -> Pipeline:
        """
        Create a new pipeline.
        
        Args:
            name: Name of the pipeline. If not provided, a default name will be generated.
        
        Returns:
            New pipeline object.
        """
        pipeline = Pipeline(self.config, name=name)
        self.pipelines[pipeline.id] = pipeline
        
        logger.info("Created pipeline '%s' with ID: %s", pipeline.name, pipeline.id)
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Pipeline:
        """
        Get a pipeline by ID.
        
        Args:
            pipeline_id: ID of the pipeline.
        
        Returns:
            Pipeline object.
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline with ID '{pipeline_id}' not found.")
        return self.pipelines[pipeline_id]
    
    def get_dataset(self, dataset_id: str) -> Dataset:
        """
        Get a dataset by ID.
        
        Args:
            dataset_id: ID of the dataset.
        
        Returns:
            Dataset object.
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset with ID '{dataset_id}' not found.")
        return self.datasets[dataset_id]
    
    def compare_datasets(self, dataset1: Union[Dataset, str], dataset2: Union[Dataset, str]) -> Dict[str, Any]:
        """
        Compare two datasets.
        
        Args:
            dataset1: First dataset object or dataset ID.
            dataset2: Second dataset object or dataset ID.
        
        Returns:
            Dictionary containing comparison results.
        """
        # Get the datasets if dataset IDs were provided
        if isinstance(dataset1, str):
            if dataset1 not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset1}' not found.")
            dataset1 = self.datasets[dataset1]
        
        if isinstance(dataset2, str):
            if dataset2 not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset2}' not found.")
            dataset2 = self.datasets[dataset2]
        
        # Compare datasets
        comparison = {
            "dataset1": {
                "id": dataset1.id,
                "name": dataset1.name,
                "shape": dataset1.data.shape
            },
            "dataset2": {
                "id": dataset2.id,
                "name": dataset2.name,
                "shape": dataset2.data.shape
            },
            "differences": {},
            "summary": {}
        }
        
        # Shape comparison
        comparison["summary"]["shape_diff"] = {
            "rows": dataset2.data.shape[0] - dataset1.data.shape[0],
            "columns": dataset2.data.shape[1] - dataset1.data.shape[1]
        }
        
        # Column comparison
        comparison["summary"]["common_columns"] = list(set(dataset1.data.columns) & set(dataset2.data.columns))
        comparison["summary"]["only_in_dataset1"] = list(set(dataset1.data.columns) - set(dataset2.data.columns))
        comparison["summary"]["only_in_dataset2"] = list(set(dataset2.data.columns) - set(dataset1.data.columns))
        
        # Statistical differences for common columns
        diff_stats = {}
        for col in comparison["summary"]["common_columns"]:
            if pd.api.types.is_numeric_dtype(dataset1.data[col]) and pd.api.types.is_numeric_dtype(dataset2.data[col]):
                diff_stats[col] = {
                    "mean_diff": dataset2.data[col].mean() - dataset1.data[col].mean(),
                    "std_diff": dataset2.data[col].std() - dataset1.data[col].std(),
                    "min_diff": dataset2.data[col].min() - dataset1.data[col].min(),
                    "max_diff": dataset2.data[col].max() - dataset1.data[col].max()
                }
        
        comparison["differences"]["statistics"] = diff_stats
        
        # Quality score comparison
        score1 = self.quality_scorer.score(dataset1.data)
        score2 = self.quality_scorer.score(dataset2.data)
        
        comparison["differences"]["quality_scores"] = {
            "dataset1": score1,
            "dataset2": score2,
            "improvement": {k: score2[k] - score1[k] for k in score1}
        }
        
        logger.info("Compared datasets '%s' and '%s'", dataset1.name, dataset2.name)
        return comparison
    
    def export(self, dataset: Union[Dataset, str], file_path: str) -> None:
        """
        Export a dataset to a file.
        
        Args:
            dataset: Dataset object or dataset ID to export.
            file_path: Path to save the dataset to.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Export the dataset
        dataset.save_to_file(file_path)
        
        logger.info("Exported dataset '%s' to %s", dataset.name, file_path)
    
    def get_versions(self, dataset: Union[Dataset, str]) -> List[Dict[str, Any]]:
        """
        Get all versions of a dataset.
        
        Args:
            dataset: Dataset object or dataset ID.
        
        Returns:
            List of version information dictionaries.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Get versions
        versions = self.version_control.get_versions(dataset)
        
        logger.info("Retrieved %d versions for dataset '%s'", len(versions), dataset.name)
        return versions
    
    def restore_version(self, dataset: Union[Dataset, str], version_id: str) -> Dataset:
        """
        Restore a dataset to a previous version.
        
        Args:
            dataset: Dataset object or dataset ID.
            version_id: ID of the version to restore.
        
        Returns:
            Restored dataset.
        """
        # Get the dataset if a dataset ID was provided
        if isinstance(dataset, str):
            if dataset not in self.datasets:
                raise ValueError(f"Dataset with ID '{dataset}' not found.")
            dataset = self.datasets[dataset]
        
        # Restore version
        restored_dataset = self.version_control.restore_version(dataset, version_id)
        
        # Store the restored dataset
        self.datasets[restored_dataset.id] = restored_dataset
        
        logger.info("Restored dataset '%s' to version %s", dataset.name, version_id)
        return restored_dataset
    
    def save_state(self, file_path: str) -> None:
        """
        Save the current state of CleanSlate to a file.
        
        Args:
            file_path: Path to save the state to.
        """
        state = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "config": self.config.to_dict(),
            "datasets": {},
            "pipelines": {}
        }
        
        # Save datasets
        for dataset_id, dataset in self.datasets.items():
            dataset_state = {
                "id": dataset.id,
                "name": dataset.name,
                "created_at": dataset.created_at.isoformat(),
                "modified_at": dataset.modified_at.isoformat(),
                "metadata": dataset.metadata,
                "history": dataset.history,
                "data_file": f"{dataset.id}.parquet"  # Save data separately
            }
            state["datasets"][dataset_id] = dataset_state
            
            # Save dataset data
            dataset_dir = os.path.dirname(file_path)
            dataset_file = os.path.join(dataset_dir, f"{dataset.id}.parquet")
            dataset.data.to_parquet(dataset_file, index=False)
        
        # Save pipelines
        for pipeline_id, pipeline in self.pipelines.items():
            state["pipelines"][pipeline_id] = pipeline.to_dict()
        
        # Save state to file
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info("Saved CleanSlate state to %s", file_path)
    
    def load_state(self, file_path: str) -> None:
        """
        Load CleanSlate state from a file.
        
        Args:
            file_path: Path to load the state from.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"State file not found: {file_path}")
        
        # Load state from file
        with open(file_path, "r") as f:
            state = json.load(f)
        
        # Set properties
        self.id = state["id"]
        self.created_at = datetime.fromisoformat(state["created_at"])
        
        # Load configuration
        self.config = Config()
        for section, section_data in state["config"].items():
            setattr(self.config, section, section_data)
        
        # Load datasets
        self.datasets = {}
        for dataset_id, dataset_state in state["datasets"].items():
            # Load dataset data
            dataset_dir = os.path.dirname(file_path)
            dataset_file = os.path.join(dataset_dir, dataset_state["data_file"])
            data = pd.read_parquet(dataset_file)
            
            # Create dataset object
            dataset = Dataset(data, dataset_state["name"])
            dataset.id = dataset_state["id"]
            dataset.created_at = datetime.fromisoformat(dataset_state["created_at"])
            dataset.modified_at = datetime.fromisoformat(dataset_state["modified_at"])
            dataset.metadata = dataset_state["metadata"]
            dataset.history = dataset_state["history"]
            
            self.datasets[dataset_id] = dataset
        
        # Load pipelines
        self.pipelines = {}
        for pipeline_id, pipeline_state in state["pipelines"].items():
            pipeline = Pipeline.from_dict(pipeline_state, self.config)
            self.pipelines[pipeline_id] = pipeline
        
        logger.info("Loaded CleanSlate state from %s", file_path) 