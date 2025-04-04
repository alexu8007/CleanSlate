"""
Streamlit-based web UI for CleanSlate.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import json
import time
from datetime import datetime
import base64

# Add parent directory to path to import cleanslate
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from cleanslate import CleanSlate, Config
from cleanslate.core.dataset import Dataset
from cleanslate.pipelines.pipeline import Pipeline

# Set page configuration
st.set_page_config(
    page_title="CleanSlate - Data Cleaning Tool",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "cleaner" not in st.session_state:
    st.session_state.cleaner = CleanSlate()
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
if "cleaned_dataset" not in st.session_state:
    st.session_state.cleaned_dataset = None
if "current_pipeline" not in st.session_state:
    st.session_state.current_pipeline = None
if "anomalies" not in st.session_state:
    st.session_state.anomalies = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "cleaning_done" not in st.session_state:
    st.session_state.cleaning_done = False


def load_data(file):
    """Load data from an uploaded file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
    
    # Load the data
    try:
        dataset = st.session_state.cleaner.load_data(tmp_path, name=Path(file.name).stem)
        st.session_state.current_dataset = dataset
        st.session_state.file_uploaded = True
        return dataset
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass


def detect_anomalies(dataset):
    """Detect anomalies in the dataset."""
    with st.spinner("Detecting anomalies..."):
        anomalies = st.session_state.cleaner.detect_anomalies(dataset)
        st.session_state.anomalies = anomalies
        return anomalies


def clean_data(dataset, pipeline_id=None):
    """Clean the dataset using the specified pipeline."""
    with st.spinner("Cleaning data..."):
        cleaned_dataset = st.session_state.cleaner.clean(dataset, pipeline_id)
        st.session_state.cleaned_dataset = cleaned_dataset
        st.session_state.cleaning_done = True
        return cleaned_dataset


def create_custom_pipeline():
    """Create a custom pipeline with user-specified steps."""
    pipeline = st.session_state.cleaner.create_pipeline(name="custom_pipeline")
    st.session_state.current_pipeline = pipeline
    return pipeline


def add_step_to_pipeline(pipeline, step_name, step_params):
    """Add a step to the pipeline."""
    pipeline.add_step(name=step_name, params=step_params)


def render_sidebar():
    """Render the sidebar with navigation and options."""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/yourusername/cleanslate/main/docs/logo.png", width=200)
        st.title("CleanSlate")
        st.caption("AI-powered data cleaning")
        
        # Navigation
        st.header("Navigation")
        page = st.radio("Go to", ["Upload Data", "Analyze & Detect", "Clean Data", "Compare & Export"])
        
        # Dataset info
        if st.session_state.current_dataset is not None:
            st.header("Current Dataset")
            st.text(f"Name: {st.session_state.current_dataset.name}")
            st.text(f"Rows: {len(st.session_state.current_dataset.data)}")
            st.text(f"Columns: {len(st.session_state.current_dataset.data.columns)}")
        
        # Reset button
        if st.button("Start New Session"):
            st.session_state.cleaner = CleanSlate()
            st.session_state.current_dataset = None
            st.session_state.cleaned_dataset = None
            st.session_state.current_pipeline = None
            st.session_state.anomalies = None
            st.session_state.file_uploaded = False
            st.session_state.cleaning_done = False
            st.experimental_rerun()
        
        st.divider()
        st.caption("© 2023 CleanSlate")
    
    return page


def render_upload_page():
    """Render the data upload page."""
    st.header("Upload Your Data")
    st.write("Upload a CSV, Excel, or Parquet file to begin cleaning your data.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "parquet"])
    
    if uploaded_file is not None:
        dataset = load_data(uploaded_file)
        
        if dataset is not None:
            st.success(f"Successfully loaded {dataset.name} with {len(dataset.data)} rows and {len(dataset.data.columns)} columns.")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(dataset.data.head(10))
            
            # Display data types
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame({"Column": dataset.data.columns, "Type": dataset.data.dtypes.astype(str)})
            st.dataframe(dtypes_df)
            
            # Display missing values
            st.subheader("Missing Values")
            missing_df = pd.DataFrame({
                "Column": dataset.data.columns,
                "Missing Count": dataset.data.isna().sum().values,
                "Missing Percentage": (dataset.data.isna().sum() / len(dataset.data) * 100).values
            }).sort_values("Missing Count", ascending=False)
            
            st.dataframe(missing_df)
            
            # Plot missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(dataset.data.isna(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("Navigate to 'Analyze & Detect' to discover anomalies and issues in your data.")
    else:
        st.info("Please upload a file to continue.")


def render_analyze_page():
    """Render the data analysis and anomaly detection page."""
    st.header("Analyze Data & Detect Anomalies")
    
    if st.session_state.current_dataset is None:
        st.warning("Please upload a dataset first.")
        return
    
    dataset = st.session_state.current_dataset
    
    # Display dataset summary
    st.subheader("Dataset Summary")
    summary = dataset.get_summary()
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", summary["num_rows"])
    with col2:
        st.metric("Columns", summary["num_columns"])
    with col3:
        missing_count = sum(summary["missing_values"].values())
        missing_percent = missing_count / (summary["num_rows"] * summary["num_columns"]) * 100
        st.metric("Missing Values", f"{missing_count} ({missing_percent:.2f}%)")
    
    # Quality scoring
    st.subheader("Data Quality Score")
    
    if st.button("Calculate Quality Score"):
        with st.spinner("Calculating quality score..."):
            quality_scores = st.session_state.cleaner.score(dataset)
            
            # Display quality scores
            score_cols = st.columns(5)
            with score_cols[0]:
                st.metric("Overall", f"{quality_scores['overall']:.2f}")
            with score_cols[1]:
                st.metric("Completeness", f"{quality_scores['completeness']:.2f}")
            with score_cols[2]:
                st.metric("Consistency", f"{quality_scores['consistency']:.2f}")
            with score_cols[3]:
                st.metric("Validity", f"{quality_scores['validity']:.2f}")
            with score_cols[4]:
                st.metric("Uniqueness", f"{quality_scores['uniqueness']:.2f}")
            
            # Plot quality scores
            fig = go.Figure(data=[
                go.Bar(
                    x=["Overall", "Completeness", "Consistency", "Validity", "Uniqueness"],
                    y=[quality_scores["overall"], quality_scores["completeness"], 
                       quality_scores["consistency"], quality_scores["validity"], 
                       quality_scores["uniqueness"]],
                    marker_color=["blue", "green", "orange", "red", "purple"]
                )
            ])
            fig.update_layout(title="Data Quality Scores", yaxis_title="Score (0-100)")
            st.plotly_chart(fig)
    
    # Anomaly detection
    st.subheader("Anomaly Detection")
    
    if st.button("Detect Anomalies"):
        anomalies = detect_anomalies(dataset)
        
        # Display anomaly summary
        st.write(f"Found {anomalies['summary']['total_anomalies']} anomalies in the data")
        st.write(f"Missing values: {anomalies['summary']['total_missing']} ({anomalies['summary']['missing_percentage']:.2f}%)")
        
        # Display anomalies by column
        if anomalies["columns"]:
            st.write("### Anomalies by Column")
            
            # Create tabs for each column with anomalies
            tabs = st.tabs(list(anomalies["columns"].keys()))
            
            for i, col in enumerate(anomalies["columns"]):
                with tabs[i]:
                    col_anomalies = anomalies["columns"][col]
                    st.write(f"**{col}**")
                    st.write(f"Anomalies detected: {col_anomalies['count']} ({col_anomalies['percentage']:.2f}%)")
                    
                    # Display anomaly details
                    if "values" in col_anomalies and col_anomalies["values"]:
                        st.write("Anomalous values:")
                        
                        # Create a DataFrame for display
                        anomaly_df = pd.DataFrame({
                            "Index": col_anomalies["indices"],
                            "Value": col_anomalies["values"]
                        })
                        st.dataframe(anomaly_df)
                        
                        # Plot histogram if numeric
                        if pd.api.types.is_numeric_dtype(dataset.data[col]):
                            fig = px.histogram(dataset.data, x=col)
                            # Add vertical lines for anomalies
                            for value in col_anomalies["values"]:
                                fig.add_vline(x=value, line_dash="dash", line_color="red")
                            st.plotly_chart(fig)
        
        # Missing values pattern visualization
        st.write("### Missing Values Patterns")
        
        # Display top missing patterns
        if "top_patterns" in anomalies["missing_values"]:
            st.write("Top missing value patterns:")
            pattern_data = []
            
            for pattern, count in anomalies["missing_values"]["top_patterns"].items():
                pattern_data.append({"Pattern": pattern, "Count": count})
            
            st.dataframe(pd.DataFrame(pattern_data))
        
        # Column-wise missing values
        missing_cols = pd.DataFrame({
            "Column": list(anomalies["missing_values"]["column_missing"].keys()),
            "Missing Count": list(anomalies["missing_values"]["column_missing"].values()),
            "Missing Percentage": list(anomalies["missing_values"]["column_missing_percentage"].values())
        }).sort_values("Missing Count", ascending=False)
        
        st.write("Missing values by column:")
        st.dataframe(missing_cols)
        
        # Plot missing values by column
        fig = px.bar(missing_cols, x="Column", y="Missing Percentage", 
                    title="Missing Values by Column (%)")
        st.plotly_chart(fig)


def render_clean_page():
    """Render the data cleaning page."""
    st.header("Clean Your Data")
    
    if st.session_state.current_dataset is None:
        st.warning("Please upload a dataset first.")
        return
    
    dataset = st.session_state.current_dataset
    
    # Cleaning options
    st.subheader("Cleaning Options")
    
    cleaning_type = st.radio("Choose cleaning approach:", 
                            ["Automatic Cleaning", "Custom Pipeline"])
    
    if cleaning_type == "Automatic Cleaning":
        st.write("Automatic cleaning will apply a pre-configured pipeline that works well for most datasets.")
        
        if st.button("Clean Data Automatically"):
            cleaned_dataset = clean_data(dataset)
            
            st.success(f"Data cleaned successfully! Rows: {len(cleaned_dataset.data)}, Columns: {len(cleaned_dataset.data.columns)}")
            
            # Preview cleaned data
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_dataset.data.head(10))
            
            # Score improvement
            st.subheader("Quality Score Improvement")
            
            with st.spinner("Calculating quality scores..."):
                original_scores = st.session_state.cleaner.score(dataset)
                cleaned_scores = st.session_state.cleaner.score(cleaned_dataset)
                
                # Display score comparisons
                score_metrics = ["overall", "completeness", "consistency", "validity", "uniqueness"]
                score_cols = st.columns(len(score_metrics))
                
                for i, metric in enumerate(score_metrics):
                    with score_cols[i]:
                        improvement = cleaned_scores[metric] - original_scores[metric]
                        st.metric(
                            metric.capitalize(),
                            f"{cleaned_scores[metric]:.2f}",
                            f"{improvement:+.2f}"
                        )
    
    else:  # Custom Pipeline
        st.write("Build a custom cleaning pipeline by selecting steps and parameters.")
        
        # Create a new pipeline if not exists
        if st.session_state.current_pipeline is None:
            pipeline = create_custom_pipeline()
        else:
            pipeline = st.session_state.current_pipeline
        
        # Pipeline steps
        st.subheader("Pipeline Steps")
        
        # Display current steps
        if pipeline.steps:
            st.write("Current pipeline steps:")
            for i, step in enumerate(pipeline.steps):
                st.write(f"{i+1}. **{step.name}**: {step.description}")
        else:
            st.info("No steps added yet. Add steps below.")
        
        # Add new step
        st.write("### Add New Step")
        
        # Step selection
        step_options = {
            "standardize_column_names": "Standardize column names",
            "drop_high_missing_columns": "Drop columns with high missing rate",
            "fill_missing_values": "Fill missing values",
            "remove_outliers": "Remove/cap outliers",
            "fix_data_types": "Fix data types",
            "drop_duplicates": "Remove duplicate rows"
        }
        
        selected_step = st.selectbox("Select step to add:", list(step_options.keys()),
                                    format_func=lambda x: step_options[x])
        
        # Step parameters
        st.write("Step parameters:")
        
        step_params = {}
        
        if selected_step == "standardize_column_names":
            case = st.radio("Convert case to:", ["lower", "upper", "title"])
            replace_spaces = st.checkbox("Replace spaces with underscores", value=True)
            step_params = {"case": case, "replace_spaces": replace_spaces}
        
        elif selected_step == "drop_high_missing_columns":
            threshold = st.slider("Missing threshold (drop if % missing > threshold):", 
                                 0.0, 1.0, 0.5, 0.05)
            step_params = {"threshold": threshold}
        
        elif selected_step == "fill_missing_values":
            strategy = st.selectbox("Strategy:", 
                                  ["auto", "mean", "median", "mode", "zero"])
            step_params = {"strategy": strategy}
        
        elif selected_step == "remove_outliers":
            method = st.selectbox("Method:", ["cap", "remove", "iqr"])
            threshold = st.slider("Threshold:", 1.0, 5.0, 3.0, 0.1)
            step_params = {"method": method, "threshold": threshold}
        
        elif selected_step == "fix_data_types":
            # No parameters needed
            pass
        
        elif selected_step == "drop_duplicates":
            keep = st.selectbox("Which duplicates to keep:", ["first", "last", "False"],
                               format_func=lambda x: "None (drop all)" if x == "False" else x)
            if keep == "False":
                keep = False
            step_params = {"keep": keep}
        
        # Add step button
        if st.button("Add Step to Pipeline"):
            add_step_to_pipeline(pipeline, selected_step, step_params)
            st.success(f"Added {step_options[selected_step]} step to pipeline.")
            st.experimental_rerun()
        
        # Apply pipeline button
        if pipeline.steps and st.button("Apply Custom Pipeline"):
            cleaned_dataset = clean_data(dataset, pipeline.id)
            
            st.success(f"Data cleaned successfully! Rows: {len(cleaned_dataset.data)}, Columns: {len(cleaned_dataset.data.columns)}")
            
            # Preview cleaned data
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_dataset.data.head(10))
            
            # Score improvement
            st.subheader("Quality Score Improvement")
            
            with st.spinner("Calculating quality scores..."):
                original_scores = st.session_state.cleaner.score(dataset)
                cleaned_scores = st.session_state.cleaner.score(cleaned_dataset)
                
                # Display score comparisons
                score_metrics = ["overall", "completeness", "consistency", "validity", "uniqueness"]
                score_cols = st.columns(len(score_metrics))
                
                for i, metric in enumerate(score_metrics):
                    with score_cols[i]:
                        improvement = cleaned_scores[metric] - original_scores[metric]
                        st.metric(
                            metric.capitalize(),
                            f"{cleaned_scores[metric]:.2f}",
                            f"{improvement:+.2f}"
                        )


def render_compare_page():
    """Render the compare and export page."""
    st.header("Compare & Export Data")
    
    if st.session_state.current_dataset is None:
        st.warning("Please upload a dataset first.")
        return
    
    if not st.session_state.cleaning_done:
        st.warning("Please clean your data first.")
        return
    
    dataset = st.session_state.current_dataset
    cleaned_dataset = st.session_state.cleaned_dataset
    
    # Compare datasets
    st.subheader("Dataset Comparison")
    
    # Get comparison results
    comparison = st.session_state.cleaner.compare_datasets(dataset, cleaned_dataset)
    
    # Display shape changes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", comparison["dataset1"]["shape"][0])
        st.metric("Original Columns", comparison["dataset1"]["shape"][1])
    with col2:
        st.metric("Cleaned Rows", comparison["dataset2"]["shape"][0])
        st.metric("Cleaned Columns", comparison["dataset2"]["shape"][1])
    with col3:
        row_diff = comparison["summary"]["shape_diff"]["rows"]
        col_diff = comparison["summary"]["shape_diff"]["columns"]
        st.metric("Row Difference", row_diff, str(row_diff))
        st.metric("Column Difference", col_diff, str(col_diff))
    
    # Dropped columns
    if comparison["summary"]["only_in_dataset1"]:
        st.write("### Dropped Columns")
        st.write(", ".join(comparison["summary"]["only_in_dataset1"]))
    
    # Statistical differences
    if comparison["differences"]["statistics"]:
        st.write("### Statistical Differences")
        
        # Create tabs for each column with statistics
        stat_cols = list(comparison["differences"]["statistics"].keys())
        if stat_cols:
            stat_tabs = st.tabs(stat_cols)
            
            for i, col in enumerate(stat_cols):
                with stat_tabs[i]:
                    col_stats = comparison["differences"]["statistics"][col]
                    
                    # Display metrics
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Mean Difference", f"{col_stats['mean_diff']:.2f}")
                    with metrics_cols[1]:
                        st.metric("Std Difference", f"{col_stats['std_diff']:.2f}")
                    with metrics_cols[2]:
                        st.metric("Min Difference", f"{col_stats['min_diff']:.2f}")
                    with metrics_cols[3]:
                        st.metric("Max Difference", f"{col_stats['max_diff']:.2f}")
                    
                    # Compare distributions
                    st.write("#### Distribution Comparison")
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=dataset.data[col], name="Original", opacity=0.7))
                    fig.add_trace(go.Histogram(x=cleaned_dataset.data[col], name="Cleaned", opacity=0.7))
                    fig.update_layout(barmode="overlay", title=f"{col} Distribution")
                    st.plotly_chart(fig)
    
    # Quality score comparison
    if "quality_scores" in comparison["differences"]:
        st.write("### Quality Score Comparison")
        
        quality_diffs = comparison["differences"]["quality_scores"]["improvement"]
        metrics = list(quality_diffs.keys())
        values = list(quality_diffs.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=["blue" if v >= 0 else "red" for v in values]
            )
        ])
        fig.update_layout(title="Quality Score Improvement", yaxis_title="Change")
        st.plotly_chart(fig)
    
    # Export cleaned data
    st.subheader("Export Cleaned Data")
    export_format = st.selectbox("Export format:", ["CSV", "Excel", "Parquet"])
    
    if st.button("Export Cleaned Data"):
        with st.spinner("Preparing export..."):
            if export_format == "CSV":
                file_ext = "csv"
                mime_type = "text/csv"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                cleaned_dataset.data.to_csv(temp_file.name, index=False)
            elif export_format == "Excel":
                file_ext = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                cleaned_dataset.data.to_excel(temp_file.name, index=False)
            else:  # Parquet
                file_ext = "parquet"
                mime_type = "application/octet-stream"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                cleaned_dataset.data.to_parquet(temp_file.name, index=False)
            
            # Read file data
            with open(temp_file.name, "rb") as f:
                file_data = f.read()
            
            # Encode file data
            b64 = base64.b64encode(file_data).decode()
            
            # Create download link
            href = f'<a href="data:{mime_type};base64,{b64}" download="cleaned_data.{file_ext}">Click here to download cleaned data</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    # Save cleaning pipeline
    st.subheader("Save Cleaning Pipeline")
    pipeline_name = st.text_input("Pipeline name:", value="my_cleaning_pipeline")
    
    if st.button("Save Pipeline"):
        with st.spinner("Saving pipeline..."):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            
            # Get the pipeline used for cleaning
            if st.session_state.current_pipeline:
                pipeline = st.session_state.current_pipeline
            else:
                # Find the automatically created pipeline
                for p_id, p in st.session_state.cleaner.pipelines.items():
                    if p.name.startswith("auto_pipeline"):
                        pipeline = p
                        break
            
            # Save pipeline
            pipeline.save(temp_file.name)
            
            # Read file data
            with open(temp_file.name, "rb") as f:
                file_data = f.read()
            
            # Encode file data
            b64 = base64.b64encode(file_data).decode()
            
            # Create download link
            href = f'<a href="data:application/json;base64,{b64}" download="{pipeline_name}.json">Click here to download pipeline</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass


def main():
    """Main function for the Streamlit app."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.5rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "Upload Data":
        render_upload_page()
    elif page == "Analyze & Detect":
        render_analyze_page()
    elif page == "Clean Data":
        render_clean_page()
    elif page == "Compare & Export":
        render_compare_page()


if __name__ == "__main__":
    main() 