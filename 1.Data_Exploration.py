"""
Data Exploration：

This Python script performs comprehensive exploratory data analysis (EDA) on a given dataset. It includes:

Data loading with basic validation

Structural inspection and summary statistics

Attribute analysis including data types, uniqueness, missing rates, and value ranges

Visualizations of numeric and categorical fields using matplotlib and seaborn, along with a correlation heatmap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load CSV data and identify potential mixed-type columns"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded data with {len(df)} rows.")

        mixed_cols = [col for col in df.columns if df[col].dtype == object]
        if mixed_cols:
            print(f"\n[Warning] The following columns may contain mixed data types: {mixed_cols}")
            print("Please verify the consistency of these columns.")

        return df
    except Exception as e:
        print(f"[Error] Failed to load file: {e}")
        return None

def explore_data(df):
    """Display basic information and statistics of the dataset"""
    print("\n--- Basic Data Information ---")
    print(f"Dataset shape: {df.shape} (rows, columns)")

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("\n--- Column Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- Missing Value Summary ---")
    print(df.isnull().sum().to_string())

    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all').to_string())

def analyze_attributes(df):
    """Analyze attributes: data types, unique values, missing values, and statistics for numeric columns"""
    print("\n--- Attribute Analysis ---")
    analysis_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Unique Values': df.nunique(),
        'Missing Values': df.isnull().sum(),
        'Missing Rate (%)': (df.isnull().mean() * 100).round(2)
    })
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        analysis_df.loc[col, 'Min'] = df[col].min()
        analysis_df.loc[col, 'Max'] = df[col].max()
        analysis_df.loc[col, 'Mean'] = df[col].mean()
        analysis_df.loc[col, 'Median'] = df[col].median()
    print(analysis_df.to_string())

def visualize_data(df):
    """Visualize numerical and categorical data, including distributions and correlation heatmap"""
    print("\n--- Data Visualization ---")
    plt.style.use('seaborn-v0_8-muted')

    exclude_cols = [col for col in df.columns if df[col].nunique() >= len(df) * 0.9]
    print(f"Excluded columns (likely IDs or unique values): {exclude_cols}")

    numeric_cols = df.select_dtypes(include=np.number).columns.difference(exclude_cols)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)

    #Numerical columns visualization
    for col in numeric_cols:
        print(f"\nVisualizing numeric column: {col}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df[col].dropna(), bins=20, ax=axes[0], kde=True, color='skyblue', edgecolor='black')
        axes[0].set_title(f'{col} - Histogram')

        sns.boxplot(x=df[col], ax=axes[1], color='lightcoral')
        axes[1].set_title(f'{col} - Boxplot')

        plt.tight_layout()
        plt.show()

    #Categorical columns visualization
    for col in categorical_cols:
        print(f"\nVisualizing categorical column: {col}")
        plt.figure(figsize=(6, 4))
        df[col].value_counts().head(10).plot(kind='bar', color='mediumseagreen', edgecolor='black')
        plt.title(f'{col} - Top 10 Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    #Correlation heatmap
    if len(numeric_cols) > 1:
        print("\nCorrelation Heatmap:")
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': 0.75})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

def main(file_path):
    df = load_data(file_path)
    if df is None:
        return
    explore_data(df)
    analyze_attributes(df)
    visualize_data(df)

if __name__ == "__main__":
    file_path = "data/Original_record.csv"
    main(file_path)
