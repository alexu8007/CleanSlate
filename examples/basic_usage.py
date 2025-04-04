"""
Basic example of using CleanSlate to clean a dataset.
"""

import os
import pandas as pd
import numpy as np
from cleanslate import CleanSlate

# Create sample data with common issues
def create_sample_data():
    """Create a sample dataset with common data quality issues."""
    # Create a DataFrame with some issues
    np.random.seed(42)
    
    # Create base data
    n_samples = 1000
    data = {
        "id": range(1, n_samples + 1),
        "age": np.random.normal(35, 12, n_samples).astype(int),
        "income": np.random.normal(50000, 15000, n_samples),
        "education_years": np.random.normal(13, 3, n_samples).astype(int),
        "customer_since": pd.date_range(start='2015-01-01', periods=n_samples),
        "purchased": np.random.choice([True, False], n_samples),
        "satisfaction": np.random.choice([1, 2, 3, 4, 5], n_samples),
        "category": np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values (about 5% of the data)
    for col in df.columns:
        if col != 'id':  # Keep IDs intact
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan
    
    # Introduce outliers in numeric columns
    # Age outliers (some very old people)
    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.randint(95, 120, 10)
    
    # Income outliers (some extremely high incomes)
    outlier_indices = np.random.choice(n_samples, 8, replace=False)
    df.loc[outlier_indices, 'income'] = np.random.uniform(200000, 1000000, 8)
    
    # Add a column with mixed data types (strings that should be numeric)
    mixed_strings = []
    for _ in range(n_samples):
        if np.random.random() < 0.1:
            mixed_strings.append("unknown")
        else:
            mixed_strings.append(str(np.random.uniform(1, 100)))
    
    df['score'] = mixed_strings
    
    # Add a column with rare categories
    # Add a column with rare categories
    # Calculate exact proportions to match n_samples (1000)
    categories = ['category_1'] * int(n_samples * 0.6) + \
                ['category_2'] * int(n_samples * 0.3) + \
                ['category_3'] * int(n_samples * 0.05)

    # Add some rare categories (5% of total)
    rare_count = n_samples - len(categories)
    rare_categories = []
    for i in range(1, rare_count + 1):
        rare_categories.append(f'rare_{i}')

    categories.extend(rare_categories)

    # Shuffle and assign to dataframe
    np.random.shuffle(categories)
    df['group'] = categories
    
    # Add duplicate rows (about 2% of the data)
    duplicate_count = int(n_samples * 0.02)
    duplicate_indices = np.random.choice(n_samples, duplicate_count, replace=False)
    for idx in duplicate_indices:
        # Choose a random position to insert the duplicate
        insert_pos = np.random.randint(0, len(df))
        df = pd.concat([df.iloc[:insert_pos], df.iloc[[idx]], df.iloc[insert_pos:]])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Add a mostly empty column (>70% missing)
    mostly_empty = np.empty(len(df))
    mostly_empty.fill(np.nan)
    # Fill about 25% of the values
    fill_indices = np.random.choice(len(df), int(len(df) * 0.25), replace=False)
    mostly_empty[fill_indices] = np.random.normal(0, 1, len(fill_indices))
    df['barely_used_feature'] = mostly_empty
    
    return df

def main():
    """Run a basic CleanSlate example."""
    # Create sample data
    print("Creating sample data with quality issues...")
    sample_data = create_sample_data()
    
    # Save sample data to file
    if not os.path.exists("data"):
        os.makedirs("data")
    sample_data.to_csv("data/sample_data.csv", index=False)
    
    print(f"Sample data shape: {sample_data.shape}")
    print("\nSample of the data:")
    print(sample_data.head())
    
    # Initialize CleanSlate
    print("\nInitializing CleanSlate...")
    cleaner = CleanSlate()
    
    # Load the data
    print("Loading data...")
    dataset = cleaner.load_data("data/sample_data.csv", name="sample_dataset")
    
    # Get dataset profile
    print("\nData profile before cleaning:")
    profile = dataset.get_profile()
    print(f"Rows: {profile['num_rows']}, Columns: {profile['num_columns']}")
    print(f"Missing values: {sum(profile['missing_values'].values())}")
    
    # Score data quality before cleaning
    print("\nData quality scores before cleaning:")
    scores_before = cleaner.score(dataset)
    for metric, score in scores_before.items():
        print(f"  {metric}: {score:.2f}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    anomalies = cleaner.detect_anomalies(dataset)
    print(f"Found {anomalies['summary']['total_anomalies']} anomalies in the data")
    print(f"Missing values: {anomalies['summary']['total_missing']} ({anomalies['summary']['missing_percentage']:.2f}%)")
    
    # Some examples of detected anomalies in specific columns
    if 'age' in anomalies['columns']:
        print(f"\nAge anomalies: {len(anomalies['columns']['age']['indices'])} outliers detected")
        print(f"Anomalous ages: {anomalies['columns']['age']['values']}")
    
    if 'income' in anomalies['columns']:
        print(f"\nIncome anomalies: {len(anomalies['columns']['income']['indices'])} outliers detected")
        print(f"Income range: ${anomalies['columns']['income']['stats']['min']:.2f} - ${anomalies['columns']['income']['stats']['max']:.2f}")
        print(f"Mean income: ${anomalies['columns']['income']['stats']['mean']:.2f}")
    
    # Clean the data (automatic pipeline)
    print("\nCleaning data with automatic pipeline...")
    cleaned_dataset = cleaner.clean(dataset)
    
    # Get some stats after cleaning
    print("\nData profile after cleaning:")
    cleaned_profile = cleaned_dataset.get_profile()
    print(f"Rows: {cleaned_profile['num_rows']}, Columns: {cleaned_profile['num_columns']}")
    print(f"Missing values: {sum(cleaned_profile['missing_values'].values())}")
    
    # Check data quality after cleaning
    print("\nData quality scores after cleaning:")
    scores_after = cleaner.score(cleaned_dataset)
    for metric, score in scores_after.items():
        print(f"  {metric}: {score:.2f}")
        diff = score - scores_before[metric]
        print(f"  Change: {diff:+.2f}")
    
    # Compare original and cleaned datasets
    print("\nComparing original and cleaned datasets:")
    comparison = cleaner.compare_datasets(dataset, cleaned_dataset)
    print(f"Row difference: {comparison['summary']['shape_diff']['rows']}")
    print(f"Column difference: {comparison['summary']['shape_diff']['columns']}")
    
    if comparison['summary']['only_in_dataset1']:
        print(f"Columns dropped: {comparison['summary']['only_in_dataset1']}")
    
    # Save cleaned data
    cleaner.export(cleaned_dataset, "data/cleaned_data.csv")
    print("\nCleaned data saved to 'data/cleaned_data.csv'")
    
    # Create a custom pipeline
    print("\nCreating a custom cleaning pipeline...")
    pipeline = cleaner.create_pipeline(name="custom_cleaning")
    
    # Add steps to the pipeline
    print("Adding steps to pipeline...")
    pipeline.add_step(
        name="standardize_column_names",
        description="Standardize column names to lowercase",
        params={"case": "lower", "replace_spaces": True}
    )
    
    pipeline.add_step(
        name="drop_high_missing_columns",
        description="Drop columns with >50% missing values",
        params={"threshold": 0.5}
    )
    
    pipeline.add_step(
        name="fill_missing_values",
        description="Fill missing values using the median for numeric columns",
        params={"strategy": "median"}
    )
    
    pipeline.add_step(
        name="remove_outliers",
        description="Cap outliers using the IQR method",
        params={"method": "iqr", "threshold": 1.5}
    )
    
    pipeline.add_step(
        name="fix_data_types",
        description="Fix inconsistent data types"
    )
    
    pipeline.add_step(
        name="drop_duplicates",
        description="Remove duplicate rows",
        params={"keep": "first"}
    )
    
    # Apply the custom pipeline
    print("Applying custom pipeline...")
    custom_cleaned = cleaner.clean(dataset, pipeline.id)
    
    # Score custom cleaned data
    custom_scores = cleaner.score(custom_cleaned)
    print("\nData quality scores after custom cleaning:")
    for metric, score in custom_scores.items():
        print(f"  {metric}: {score:.2f}")
        diff = score - scores_before[metric]
        print(f"  Change: {diff:+.2f}")
    
    # Save custom cleaned data
    cleaner.export(custom_cleaned, "data/custom_cleaned_data.csv")
    print("\nCustom cleaned data saved to 'data/custom_cleaned_data.csv'")
    
    # Save the CleanSlate state
    cleaner.save_state("data/cleanslate_state.json")
    print("\nCleanSlate state saved to 'data/cleanslate_state.json'")
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 