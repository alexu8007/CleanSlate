"""
Anomaly detector module for identifying outliers, inconsistencies, and missing values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from cleanslate.core.config import Config

logger = logging.getLogger("cleanslate.anomaly")


class AnomalyDetector:
    """Class for detecting anomalies in datasets."""
    
    def __init__(self, config: Config):
        """
        Initialize anomaly detector.
        
        Args:
            config: Configuration object.
        """
        self.config = config
    
    def detect(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect anomalies in a dataset.
        
        Args:
            data: DataFrame to analyze.
            columns: Columns to analyze. If not provided, all columns are analyzed.
        
        Returns:
            Dictionary containing anomaly detection results.
        """
        if columns is None:
            # Use all columns except those with too many unique values
            columns = []
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    columns.append(col)
                elif data[col].nunique() / len(data) < 0.5:  # Only include categorical columns with less than 50% unique values
                    columns.append(col)
        
        results = {
            "columns": {},
            "summary": {
                "total_anomalies": 0,
                "anomaly_percentage": 0.0,
                "analyzed_columns": len(columns)
            }
        }
        
        total_anomalies = 0
        
        for col in columns:
            # Skip if column is all null
            if data[col].isna().all():
                continue
                
            # Detect anomalies based on dtype
            if pd.api.types.is_numeric_dtype(data[col]):
                anomalies = self._detect_numeric_anomalies(data[col])
            elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                anomalies = self._detect_categorical_anomalies(data[col])
            else:
                # Skip other types
                continue
            
            if anomalies:
                results["columns"][col] = anomalies
                total_anomalies += len(anomalies["indices"])
        
        # Add missing values to results
        missing_values = self._detect_missing_values(data)
        results["missing_values"] = missing_values
        
        # Update summary
        results["summary"]["total_anomalies"] = total_anomalies
        results["summary"]["anomaly_percentage"] = (total_anomalies / (len(data) * len(columns))) * 100 if len(data) * len(columns) > 0 else 0
        results["summary"]["total_missing"] = missing_values["total_missing"]
        results["summary"]["missing_percentage"] = missing_values["missing_percentage"]
        
        return results
    
    def _detect_numeric_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect anomalies in a numeric column.
        
        Args:
            series: Series to analyze.
        
        Returns:
            Dictionary containing anomaly detection results.
        """
        # Remove missing values for analysis
        data_clean = series.dropna().values.reshape(-1, 1)
        
        if len(data_clean) < 10:
            # Not enough data for meaningful anomaly detection
            return {}
        
        # Z-score method (for normally distributed data)
        z_scores = stats.zscore(data_clean, nan_policy='omit').flatten()
        z_score_outliers = np.where(np.abs(z_scores) > 3)[0]
        
        # IQR method (robust to non-normal distributions)
        q1, q3 = np.percentile(data_clean, [25, 75])
        iqr = q3 - q1
        iqr_lower_bound = q1 - (1.5 * iqr)
        iqr_upper_bound = q3 + (1.5 * iqr)
        iqr_outliers = np.where((data_clean < iqr_lower_bound) | (data_clean > iqr_upper_bound))[0]
        
        # Isolation Forest (machine learning approach)
        if len(data_clean) >= 100:  # Only use for larger datasets
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            predictions = iso_forest.fit_predict(data_clean)
            iso_outliers = np.where(predictions == -1)[0]
        else:
            iso_outliers = np.array([])
        
        # Combine results (get indices in original series)
        all_outliers = set()
        clean_indices = series.dropna().index
        
        for outlier_idx in np.concatenate([z_score_outliers, iqr_outliers, iso_outliers]):
            if outlier_idx < len(clean_indices):
                all_outliers.add(clean_indices[outlier_idx])
        
        # Convert to list and sort
        outlier_indices = sorted(list(all_outliers))
        
        if not outlier_indices:
            return {}
        
        # Get outlier values
        outlier_values = series.loc[outlier_indices].tolist()
        
        return {
            "indices": outlier_indices,
            "values": outlier_values,
            "count": len(outlier_indices),
            "percentage": (len(outlier_indices) / len(series)) * 100,
            "stats": {
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std()
            }
        }
    
    def _detect_categorical_anomalies(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect anomalies in a categorical column.
        
        Args:
            series: Series to analyze.
        
        Returns:
            Dictionary containing anomaly detection results.
        """
        # Remove missing values for analysis
        data_clean = series.dropna()
        
        if len(data_clean) < 10 or data_clean.nunique() < 2:
            # Not enough data or variability for meaningful anomaly detection
            return {}
        
        # Frequency-based approach
        value_counts = data_clean.value_counts()
        total_count = len(data_clean)
        
        # Identify rare categories (less than 1% occurrence)
        rare_values = value_counts[value_counts / total_count < 0.01]
        
        if rare_values.empty:
            return {}
        
        # Get indices of rows with rare values
        outlier_indices = []
        for value in rare_values.index:
            indices = data_clean[data_clean == value].index.tolist()
            outlier_indices.extend(indices)
        
        # Sort indices
        outlier_indices = sorted(outlier_indices)
        
        if not outlier_indices:
            return {}
        
        # Get outlier values
        outlier_values = series.loc[outlier_indices].tolist()
        
        return {
            "indices": outlier_indices,
            "values": outlier_values,
            "count": len(outlier_indices),
            "percentage": (len(outlier_indices) / len(series)) * 100,
            "rare_categories": rare_values.index.tolist(),
            "value_counts": value_counts.to_dict()
        }
    
    def _detect_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect missing values in a dataset.
        
        Args:
            data: DataFrame to analyze.
        
        Returns:
            Dictionary containing missing value detection results.
        """
        missing_counts = data.isna().sum()
        total_cells = data.shape[0] * data.shape[1]
        total_missing = missing_counts.sum()
        
        # Calculate missing value patterns
        missing_patterns = []
        for index, row in data.isna().iterrows():
            pattern = tuple(row)
            missing_patterns.append(pattern)
        
        # Count unique patterns
        pattern_counts = {}
        for pattern in missing_patterns:
            pattern_str = str(pattern)
            if pattern_str in pattern_counts:
                pattern_counts[pattern_str] += 1
            else:
                pattern_counts[pattern_str] = 1
        
        return {
            "column_missing": missing_counts.to_dict(),
            "column_missing_percentage": (missing_counts / len(data) * 100).to_dict(),
            "total_missing": int(total_missing),
            "missing_percentage": (total_missing / total_cells) * 100 if total_cells > 0 else 0,
            "pattern_count": len(pattern_counts),
            "top_patterns": {k: v for k, v in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]}
        }
    
    def get_recommendations(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Get recommendations for dealing with anomalies.
        
        Args:
            data: DataFrame containing the data.
            anomaly_results: Results from anomaly detection.
        
        Returns:
            Dictionary containing recommendations for each column.
        """
        recommendations = {}
        
        # Process numeric columns with anomalies
        for col in anomaly_results["columns"]:
            if col not in data.columns:
                continue
                
            col_anomalies = anomaly_results["columns"][col]
            col_recs = []
            
            if pd.api.types.is_numeric_dtype(data[col]):
                # Numeric column recommendations
                if col_anomalies.get("percentage", 0) > 10:
                    col_recs.append(f"High percentage of outliers ({col_anomalies['percentage']:.1f}%). Consider transforming this column (e.g., log transform).")
                
                if col_anomalies.get("count", 0) > 0:
                    col_recs.append(f"Consider capping outliers at {col_anomalies['stats']['mean'] + 3 * col_anomalies['stats']['std']:.2f} (3 std from mean).")
                    col_recs.append("Consider using robust statistics (median, IQR) instead of mean/std for this column.")
            
            elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                # Categorical column recommendations
                if "rare_categories" in col_anomalies and len(col_anomalies["rare_categories"]) > 0:
                    if len(col_anomalies["rare_categories"]) > 5:
                        col_recs.append(f"Consider grouping {len(col_anomalies['rare_categories'])} rare categories into an 'Other' category.")
                    else:
                        rare_cats = ", ".join([f"'{v}'" for v in col_anomalies["rare_categories"][:5]])
                        col_recs.append(f"Consider grouping rare categories ({rare_cats}) into an 'Other' category.")
            
            if col_recs:
                recommendations[col] = col_recs
        
        # Process missing values
        missing_recs = []
        high_missing_cols = []
        
        for col, missing_pct in anomaly_results["missing_values"]["column_missing_percentage"].items():
            if missing_pct > 0:
                if missing_pct > 50:
                    high_missing_cols.append(col)
                elif missing_pct > 20:
                    recommendations[col] = recommendations.get(col, []) + [f"High missing rate ({missing_pct:.1f}%). Consider using multiple imputation or dropping this column."]
                elif missing_pct > 5:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        recommendations[col] = recommendations.get(col, []) + [f"Consider imputing missing values using KNN or regression-based methods."]
                    else:
                        recommendations[col] = recommendations.get(col, []) + [f"Consider imputing missing values using mode or creating a 'Missing' category."]
        
        if high_missing_cols:
            if len(high_missing_cols) > 3:
                missing_recs.append(f"{len(high_missing_cols)} columns have >50% missing values. Consider dropping these columns.")
            else:
                cols_str = ", ".join([f"'{col}'" for col in high_missing_cols[:3]])
                missing_recs.append(f"Columns {cols_str} have >50% missing values. Consider dropping these columns.")
        
        if anomaly_results["missing_values"]["pattern_count"] > 1:
            missing_recs.append(f"Detected {anomaly_results['missing_values']['pattern_count']} missing value patterns. Consider using multiple imputation methods.")
        
        if missing_recs:
            recommendations["missing_values"] = missing_recs
        
        return recommendations 