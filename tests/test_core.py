"""
Tests for the core CleanSlate functionality.
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleanslate import CleanSlate, Config
from cleanslate.core.dataset import Dataset


class TestConfig(unittest.TestCase):
    """Tests for the Config class."""
    
    def test_default_config(self):
        """Test that the default configuration is created correctly."""
        config = Config()
        
        # Check that the default configuration is created
        self.assertIsNotNone(config.cleaning)
        self.assertIsNotNone(config.pipeline)
        self.assertIsNotNone(config.ui)
        
        # Check some default values
        self.assertTrue(config.cleaning.remove_outliers)
        self.assertTrue(config.cleaning.fill_missing_values)
        self.assertTrue(config.pipeline.version_control)
    
    def test_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
            cleaning:
              remove_outliers: false
              fill_missing_values: true
              missing_value_strategy: median
            pipeline:
              auto_save: false
              max_pipeline_steps: 10
            ui:
              theme: dark
            """)
            config_path = f.name
        
        try:
            # Load the configuration
            config = Config(config_path)
            
            # Check that the configuration was loaded correctly
            self.assertFalse(config.cleaning.remove_outliers)
            self.assertTrue(config.cleaning.fill_missing_values)
            self.assertEqual(config.cleaning.missing_value_strategy, "median")
            self.assertFalse(config.pipeline.auto_save)
            self.assertEqual(config.pipeline.max_pipeline_steps, 10)
            self.assertEqual(config.ui.theme, "dark")
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        config = Config()
        
        # Modify some configuration values
        config.cleaning.remove_outliers = False
        config.pipeline.auto_save = False
        config.ui.theme = "dark"
        
        # Save the configuration to a file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = f.name
        
        try:
            config.save_to_file(config_path)
            
            # Load the configuration from the file
            config2 = Config(config_path)
            
            # Check that the configuration was loaded correctly
            self.assertFalse(config2.cleaning.remove_outliers)
            self.assertFalse(config2.pipeline.auto_save)
            self.assertEqual(config2.ui.theme, "dark")
        finally:
            # Clean up
            os.unlink(config_path)


class TestDataset(unittest.TestCase):
    """Tests for the Dataset class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame
        self.df = pd.DataFrame({
            "id": range(1, 101),
            "value": np.random.normal(10, 2, 100),
            "category": np.random.choice(["A", "B", "C"], 100)
        })
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            self.csv_path = f.name
            self.df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.csv_path)
    
    def test_create_from_dataframe(self):
        """Test creating a dataset from a DataFrame."""
        dataset = Dataset(self.df, name="test_dataset")
        
        # Check that the dataset was created correctly
        self.assertEqual(dataset.name, "test_dataset")
        self.assertEqual(len(dataset.data), 100)
        self.assertEqual(list(dataset.data.columns), ["id", "value", "category"])
    
    def test_create_from_file(self):
        """Test creating a dataset from a file."""
        dataset = Dataset(self.csv_path, name="test_dataset")
        
        # Check that the dataset was created correctly
        self.assertEqual(dataset.name, "test_dataset")
        self.assertEqual(len(dataset.data), 100)
        self.assertEqual(list(dataset.data.columns), ["id", "value", "category"])
    
    def test_save_to_file(self):
        """Test saving a dataset to a file."""
        dataset = Dataset(self.df, name="test_dataset")
        
        # Save the dataset to a file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            dataset.save_to_file(output_path)
            
            # Check that the file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Load the dataset from the file
            loaded_df = pd.read_csv(output_path)
            
            # Check that the data was saved correctly
            self.assertEqual(len(loaded_df), 100)
            self.assertEqual(list(loaded_df.columns), ["id", "value", "category"])
        finally:
            # Clean up
            os.unlink(output_path)
    
    def test_get_summary(self):
        """Test getting a summary of a dataset."""
        dataset = Dataset(self.df, name="test_dataset")
        
        # Get the summary
        summary = dataset.get_summary()
        
        # Check that the summary contains the expected information
        self.assertEqual(summary["name"], "test_dataset")
        self.assertEqual(summary["num_rows"], 100)
        self.assertEqual(summary["num_columns"], 3)
        self.assertEqual(summary["columns"], ["id", "value", "category"])
    
    def test_get_profile(self):
        """Test getting a detailed profile of a dataset."""
        dataset = Dataset(self.df, name="test_dataset")
        
        # Get the profile
        profile = dataset.get_profile()
        
        # Check that the profile contains the expected information
        self.assertEqual(profile["name"], "test_dataset")
        self.assertEqual(profile["num_rows"], 100)
        self.assertEqual(profile["num_columns"], 3)
        self.assertIn("column_details", profile)
        self.assertIn("id", profile["column_details"])
        self.assertIn("value", profile["column_details"])
        self.assertIn("category", profile["column_details"])
        self.assertIn("correlation_matrix", profile)


class TestCleanSlate(unittest.TestCase):
    """Tests for the CleanSlate class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with issues
        np.random.seed(42)
        self.df = pd.DataFrame({
            "id": range(1, 101),
            "value": np.random.normal(10, 2, 100),
            "category": np.random.choice(["A", "B", "C"], 100)
        })
        
        # Add some missing values
        self.df.loc[np.random.choice(100, 10), "value"] = np.nan
        
        # Add some outliers
        self.df.loc[np.random.choice(100, 5), "value"] = 100
        
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            self.csv_path = f.name
            self.df.to_csv(self.csv_path, index=False)
        
        # Create a CleanSlate instance
        self.cleaner = CleanSlate()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.csv_path)
    
    def test_load_data(self):
        """Test loading data."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Check that the dataset was loaded correctly
        self.assertEqual(dataset.name, "test_dataset")
        self.assertEqual(len(dataset.data), 100)
        self.assertEqual(list(dataset.data.columns), ["id", "value", "category"])
    
    def test_clean_data(self):
        """Test cleaning data."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Clean the data
        cleaned_dataset = self.cleaner.clean(dataset)
        
        # Check that the cleaning was successful
        self.assertEqual(len(cleaned_dataset.data), 100)  # No rows should be removed
        self.assertEqual(list(cleaned_dataset.data.columns), ["id", "value", "category"])  # No columns should be removed
        self.assertEqual(cleaned_dataset.data["value"].isna().sum(), 0)  # Missing values should be filled
    
    def test_detect_anomalies(self):
        """Test detecting anomalies."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Detect anomalies
        anomalies = self.cleaner.detect_anomalies(dataset)
        
        # Check that anomalies were detected
        self.assertIn("columns", anomalies)
        self.assertIn("missing_values", anomalies)
        self.assertIn("summary", anomalies)
        
        # Check that value column has anomalies
        self.assertIn("value", anomalies["columns"])
        
        # Check that missing values were detected
        self.assertEqual(anomalies["missing_values"]["total_missing"], 10)
    
    def test_score_quality(self):
        """Test scoring data quality."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Score the data
        scores = self.cleaner.score(dataset)
        
        # Check that scores were calculated
        self.assertIn("completeness", scores)
        self.assertIn("consistency", scores)
        self.assertIn("validity", scores)
        self.assertIn("uniqueness", scores)
        self.assertIn("overall", scores)
        
        # Check that completeness score reflects missing values
        self.assertLess(scores["completeness"], 100)  # Should be less than 100 due to missing values
    
    def test_create_pipeline(self):
        """Test creating a pipeline."""
        # Create a pipeline
        pipeline = self.cleaner.create_pipeline(name="test_pipeline")
        
        # Check that the pipeline was created correctly
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertEqual(len(pipeline.steps), 0)
        
        # Add a step to the pipeline
        pipeline.add_step(
            name="fill_missing_values",
            description="Fill missing values",
            params={"strategy": "median"}
        )
        
        # Check that the step was added
        self.assertEqual(len(pipeline.steps), 1)
        self.assertEqual(pipeline.steps[0].name, "fill_missing_values")
    
    def test_compare_datasets(self):
        """Test comparing datasets."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Clean the data
        cleaned_dataset = self.cleaner.clean(dataset)
        
        # Compare the datasets
        comparison = self.cleaner.compare_datasets(dataset, cleaned_dataset)
        
        # Check that the comparison contains the expected information
        self.assertIn("dataset1", comparison)
        self.assertIn("dataset2", comparison)
        self.assertIn("differences", comparison)
        self.assertIn("summary", comparison)
        
        # Check that the summary contains the expected information
        self.assertIn("shape_diff", comparison["summary"])
        self.assertIn("common_columns", comparison["summary"])
    
    def test_export(self):
        """Test exporting a dataset."""
        # Load the data
        dataset = self.cleaner.load_data(self.csv_path, name="test_dataset")
        
        # Export the dataset
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            self.cleaner.export(dataset, output_path)
            
            # Check that the file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Load the dataset from the file
            loaded_df = pd.read_csv(output_path)
            
            # Check that the data was exported correctly
            self.assertEqual(len(loaded_df), 100)
            self.assertEqual(list(loaded_df.columns), ["id", "value", "category"])
        finally:
            # Clean up
            os.unlink(output_path)


if __name__ == "__main__":
    unittest.main() 