# Quick Start Guide

This guide will help you get started with CleanSlate to clean your data.

## Basic Usage

Here's a minimal example of using CleanSlate to clean a dataset:

```python
from cleanslate import CleanSlate

# Initialize CleanSlate
cleaner = CleanSlate()

# Load data
dataset = cleaner.load_data("path/to/your/data.csv")

# Clean data (automatic pipeline)
cleaned_dataset = cleaner.clean(dataset)

# Export cleaned data
cleaner.export(cleaned_dataset, "path/to/cleaned_data.csv")
```

## Web UI

If you prefer a graphical interface, you can use the CleanSlate web UI:

```bash
# Start the UI
cleanslate ui
```

This will launch a web interface at http://localhost:8501 where you can:
1. Upload data files
2. Analyze and detect anomalies
3. Clean your data with automatic or custom pipelines
4. Compare and export results

## Command Line Interface

CleanSlate provides a command-line interface for basic operations:

```bash
# Clean a dataset
cleanslate clean data.csv -o cleaned_data.csv

# Analyze a dataset
cleanslate analyze data.csv -o analysis.json

# Score data quality
cleanslate score data.csv
```

## Common Tasks

### Detecting Anomalies

```python
from cleanslate import CleanSlate

cleaner = CleanSlate()
dataset = cleaner.load_data("data.csv")

# Detect anomalies
anomalies = cleaner.detect_anomalies(dataset)

# Print anomaly summary
print(f"Found {anomalies['summary']['total_anomalies']} anomalies")
print(f"Missing values: {anomalies['summary']['total_missing']}")
```

### Scoring Data Quality

```python
from cleanslate import CleanSlate

cleaner = CleanSlate()
dataset = cleaner.load_data("data.csv")

# Score data quality
scores = cleaner.score(dataset)

# Print scores
print(f"Overall quality: {scores['overall']:.2f}")
print(f"Completeness: {scores['completeness']:.2f}")
print(f"Consistency: {scores['consistency']:.2f}")
print(f"Validity: {scores['validity']:.2f}")
print(f"Uniqueness: {scores['uniqueness']:.2f}")
```

### Creating Custom Cleaning Pipelines

```python
from cleanslate import CleanSlate

cleaner = CleanSlate()
dataset = cleaner.load_data("data.csv")

# Create a custom pipeline
pipeline = cleaner.create_pipeline(name="my_pipeline")

# Add steps to the pipeline
pipeline.add_step(
    name="drop_high_missing_columns",
    params={"threshold": 0.5}
)

pipeline.add_step(
    name="fill_missing_values",
    params={"strategy": "median"}
)

pipeline.add_step(
    name="remove_outliers",
    params={"method": "iqr", "threshold": 1.5}
)

# Apply the pipeline
cleaned_dataset = cleaner.clean(dataset, pipeline.id)

# Export the cleaned dataset
cleaner.export(cleaned_dataset, "cleaned_data.csv")

# Save the pipeline for future use
pipeline.save("my_pipeline.json")
```

### Comparing Datasets

```python
from cleanslate import CleanSlate

cleaner = CleanSlate()
original = cleaner.load_data("data.csv")
cleaned = cleaner.clean(original)

# Compare datasets
comparison = cleaner.compare_datasets(original, cleaned)

# Print comparison summary
print(f"Rows before: {comparison['dataset1']['shape'][0]}, after: {comparison['dataset2']['shape'][0]}")
print(f"Columns before: {comparison['dataset1']['shape'][1]}, after: {comparison['dataset2']['shape'][1]}")

if comparison['summary']['only_in_dataset1']:
    print(f"Dropped columns: {', '.join(comparison['summary']['only_in_dataset1'])}")
```

## Next Steps

- Learn about [configuration options](configuration.md)
- See more examples in the [examples directory](../../examples/)
- Read the [API reference](../api_reference/core.md) for detailed documentation 