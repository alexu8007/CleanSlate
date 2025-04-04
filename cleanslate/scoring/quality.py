"""
Quality scoring module for quantifying the cleanliness of datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging

from cleanslate.core.config import Config

logger = logging.getLogger("cleanslate.scoring")


class QualityScorer:
    """Class for scoring the quality of datasets."""
    
    def __init__(self, config: Config):
        """
        Initialize quality scorer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
    
    def score(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Score the quality of a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Dictionary containing quality scores.
        """
        # Empty DataFrame case
        if data.empty:
            return {
                "completeness": 0.0,
                "consistency": 0.0,
                "validity": 0.0,
                "uniqueness": 0.0,
                "overall": 0.0
            }
        
        # Calculate individual scores
        completeness_score = self._score_completeness(data)
        consistency_score = self._score_consistency(data)
        validity_score = self._score_validity(data)
        uniqueness_score = self._score_uniqueness(data)
        
        # Calculate overall score (weighted average)
        overall_score = (
            completeness_score * 0.3 +
            consistency_score * 0.25 +
            validity_score * 0.25 +
            uniqueness_score * 0.2
        )
        
        return {
            "completeness": completeness_score,
            "consistency": consistency_score,
            "validity": validity_score,
            "uniqueness": uniqueness_score,
            "overall": overall_score
        }
    
    def _score_completeness(self, data: pd.DataFrame) -> float:
        """
        Score the completeness of a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Completeness score (0-100).
        """
        # Calculate percentage of non-missing values
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isna().sum().sum()
        
        if total_cells == 0:
            return 0.0
        
        completeness = 100 * (1 - missing_cells / total_cells)
        return completeness
    
    def _score_consistency(self, data: pd.DataFrame) -> float:
        """
        Score the consistency of a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Consistency score (0-100).
        """
        # Initialize score components
        scores = []
        
        # Check data type consistency
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Check if numeric column contains only numeric values
            if pd.api.types.is_numeric_dtype(col_data):
                # All numeric values by definition
                scores.append(100.0)
            
            # Check if string column contains mixed types
            elif pd.api.types.is_string_dtype(col_data):
                # Check for potential numeric values stored as strings
                numeric_strings = 0
                for val in col_data.sample(min(100, len(col_data))):
                    if isinstance(val, str) and val.replace('.', '', 1).isdigit():
                        numeric_strings += 1
                
                # If more than 10% sampled values are numeric strings, penalize
                if numeric_strings > 0:
                    inconsistency = numeric_strings / min(100, len(col_data))
                    scores.append(100 * (1 - inconsistency))
                else:
                    scores.append(100.0)
            
            # For other types, assume consistent
            else:
                scores.append(100.0)
        
        # Check value range consistency for numeric columns
        for col in data.select_dtypes(include=np.number).columns:
            col_data = data[col].dropna()
            
            if len(col_data) < 2:
                continue
                
            # Check for extreme values using IQR
            q1, q3 = np.percentile(col_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (3 * iqr)
            upper_bound = q3 + (3 * iqr)
            
            extreme_values = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            if len(col_data) > 0:
                range_consistency = 100 * (1 - extreme_values / len(col_data))
                scores.append(range_consistency)
        
        # Calculate overall consistency score
        if not scores:
            return 100.0  # No issues found
            
        return sum(scores) / len(scores)
    
    def _score_validity(self, data: pd.DataFrame) -> float:
        """
        Score the validity of a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Validity score (0-100).
        """
        # Initialize validity issues counter
        validity_issues = 0
        total_checked = 0
        
        # Check for validity issues in each column
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Check numeric columns for invalid values (NaN, Inf)
            if pd.api.types.is_numeric_dtype(col_data):
                invalid_values = (~np.isfinite(col_data)).sum()
                validity_issues += invalid_values
                total_checked += len(col_data)
            
            # Check string columns for empty strings
            elif pd.api.types.is_string_dtype(col_data):
                empty_strings = (col_data == "").sum()
                validity_issues += empty_strings
                total_checked += len(col_data)
        
        # Calculate validity score
        if total_checked == 0:
            return 100.0  # No issues found
            
        validity = 100 * (1 - validity_issues / total_checked)
        return validity
    
    def _score_uniqueness(self, data: pd.DataFrame) -> float:
        """
        Score the uniqueness of a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Uniqueness score (0-100).
        """
        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        if len(data) == 0:
            return 100.0  # No duplicates in empty dataset
            
        row_uniqueness = 100 * (1 - duplicate_rows / len(data))
        
        # Check for duplicate values in columns (excluding categorical columns)
        column_uniqueness_scores = []
        
        for col in data.columns:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Skip columns that are likely categorical
            if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                if col_data.nunique() / len(col_data) < 0.5:  # Less than 50% unique values
                    continue
            
            # Calculate uniqueness for this column
            duplicates = col_data.duplicated().sum()
            if len(col_data) > 0:
                column_uniqueness = 100 * (1 - duplicates / len(col_data))
                column_uniqueness_scores.append(column_uniqueness)
        
        # Calculate overall uniqueness (weighing row uniqueness more heavily)
        if not column_uniqueness_scores:
            return row_uniqueness
            
        column_uniqueness_avg = sum(column_uniqueness_scores) / len(column_uniqueness_scores)
        overall_uniqueness = (row_uniqueness * 0.7) + (column_uniqueness_avg * 0.3)
        
        return overall_uniqueness
    
    def get_improvement_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """
        Get recommendations for improving data quality based on scores.
        
        Args:
            scores: Dictionary of quality scores.
        
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Check completeness
        if scores["completeness"] < 90:
            recommendations.append("Improve completeness by addressing missing values using imputation techniques.")
        elif scores["completeness"] < 98:
            recommendations.append("Consider filling remaining missing values to improve completeness score.")
        
        # Check consistency
        if scores["consistency"] < 85:
            recommendations.append("Address data consistency issues by standardizing formats and value representations.")
        elif scores["consistency"] < 95:
            recommendations.append("Review data types and extreme values to improve consistency score.")
        
        # Check validity
        if scores["validity"] < 90:
            recommendations.append("Fix data validity issues such as infinite values or empty strings.")
        elif scores["validity"] < 98:
            recommendations.append("Review remaining invalid data points to improve validity score.")
        
        # Check uniqueness
        if scores["uniqueness"] < 80:
            recommendations.append("Remove duplicate rows and address columns with high duplication.")
        elif scores["uniqueness"] < 95:
            recommendations.append("Review remaining duplicates to improve uniqueness score.")
        
        # Overall recommendations
        if scores["overall"] < 85:
            recommendations.append("Consider a thorough data cleaning process to address multiple quality issues.")
        
        return recommendations
    
    def detailed_score(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a detailed quality score breakdown for a dataset.
        
        Args:
            data: DataFrame to score.
        
        Returns:
            Dictionary containing detailed quality score information.
        """
        # Get overall scores
        scores = self.score(data)
        
        # Add detailed breakdown for each column
        column_scores = {}
        
        for col in data.columns:
            col_data = data[col]
            
            # Column completeness
            missing_rate = col_data.isna().mean() * 100
            completeness = 100 - missing_rate
            
            # Column validity (depends on data type)
            if pd.api.types.is_numeric_dtype(col_data):
                invalid = (~np.isfinite(col_data.dropna())).mean() * 100 if len(col_data.dropna()) > 0 else 0
                validity = 100 - invalid
            elif pd.api.types.is_string_dtype(col_data):
                invalid = (col_data.dropna() == "").mean() * 100 if len(col_data.dropna()) > 0 else 0
                validity = 100 - invalid
            else:
                validity = 100.0
            
            # Column uniqueness
            duplicates = col_data.dropna().duplicated().mean() * 100 if len(col_data.dropna()) > 0 else 0
            uniqueness = 100 - duplicates
            
            column_scores[col] = {
                "completeness": completeness,
                "validity": validity,
                "uniqueness": uniqueness
            }
        
        # Add recommendations
        recommendations = self.get_improvement_recommendations(scores)
        
        return {
            "overall_scores": scores,
            "column_scores": column_scores,
            "recommendations": recommendations
        } 