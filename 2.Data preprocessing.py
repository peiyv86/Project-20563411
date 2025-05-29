"""
Data preprocessing pipeline for credit risk analysis
Overview
This Python script preprocesses the data for the credit risk analysis dataset.
It obtains raw financial data containing customer information and loan details,
performs extensive cleaning and transformation, and outputs a processed dataset to prepare for machine learning modeling.

Key Features

Data Cleaning:
Handles missing values by removing records with excessive missing attributes
Cleans numeric fields containing special characters (currency symbols, etc.)
Standardizes inconsistent categorical values

Feature Engineering:
Discretizes continuous variables into meaningful bins
Creates derived features (e.g., converting days to months/years)
Performs one-hot encoding for categorical variables

Data Transformation:
Standardizes numeric features using z-score normalization
Converts all data to appropriate numeric types
Handles special cases for binary and ordinal variables

Exploratory Analysis:
Generates descriptive statistics for each feature
Provides visualizations of feature distributions
Includes a pie chart showing the target variable (Default) distribution
Technical Implementation

The pipeline is built using:
Pandas for data manipulation
Scikit-learn for preprocessing (StandardScaler, KBinsDiscretizer)
Seaborn and Matplotlib for visualization
Imbalanced-learn pipeline structure (though the full pipeline isn't shown)

Usage
Input: CSV file containing raw credit data ("data/Original_record.csv")
Output: Cleaned and processed CSV file ("data/Cleaned_data.csv")

The script automatically:
Analyzes each feature
Performs appropriate transformations
Generates visualizations
Saves the processed data

Benefits
Comprehensive Processing: Handles all common data quality issues in financial datasets
Modular Design: Easy to extend or modify specific processing steps
Transparent Analysis: Provides detailed output at each processing stage

Reproducible: Standardized transformations ensure consistent results
This pipeline is particularly valuable for credit risk modeling, fraud detection,
and other financial machine learning applications where data quality is critical.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
sns.set_style("white")
plt.rcParams['axes.unicode_minus'] = False

# File paths
file_path = r"data/Original_record.csv"
save_path = r"data/Train_data_clean.csv"

def plot_distribution(data, col, plot_type='hist', figsize=(8, 5)):
    """Visualize distribution of a column with different plot types.
    Args:
        data: DataFrame containing the data
        col: Column name to visualize
        plot_type: Type of plot ('hist' for histogram, 'count' for count plot)
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize, facecolor='white')
    ax = plt.gca()
    if plot_type == 'hist':
        sns.histplot(data[col], kde=True, bins=30, color='#3498db', edgecolor='none')
        plt.ylabel('Frequency', fontsize=12)
    elif plot_type == 'count':
        sns.countplot(x=data[col], palette='pastel', edgecolor='.6')
        plt.xticks(rotation=45, ha='right')

    plt.title(f'{col} Distribution', fontsize=14, pad=20)
    plt.xlabel(col, fontsize=12)
    sns.despine()  # Remove top and right borders
    plt.tight_layout()
    plt.show()

def plot_default_pie(data):
    """Plot a pie chart showing the distribution of Default values.
    Args:
        data: DataFrame containing the 'Default' column
    """
    default_counts = data['Default'].value_counts()
    plt.figure(figsize=(8, 6), facecolor='white')
    plt.pie(default_counts,
            labels=default_counts.index,
            autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    plt.title('Default Attribute Distribution', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def analyze_column(df, col, plot_type='hist'):
    """Analyze and visualize a column with summary statistics.
    Args:
        df: DataFrame containing the data
        col: Column name to analyze
        plot_type: Type of visualization
    """
    print(f"\n=== {col} ===")
    print("Descriptive Statistics:")
    print(df[col].describe())
    print("\nValue Distribution:")
    print(df[col].value_counts(dropna=False).head(10))
    plot_distribution(df, col, plot_type)

def process_numeric(col, df, money=False, bins=None, labels=None, drop_original=False):
    """Process numeric columns with cleaning, binning, and discretization.
    Args:
        col: Column name to process
        df: DataFrame containing the data
        money: Whether to clean currency symbols
        bins: Bin edges for discretization
        labels: Labels for bins
        drop_original: Whether to drop original column after processing
    Returns:
        Processed DataFrame
    """
    if money:
        df[col] = df[col].astype(str).str.replace(r'[$,#@x]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

    if bins and labels:
        group_col = f'{col}_Group'
        df[group_col] = pd.cut(df[col], bins=bins, labels=labels)
        # One-hot encoding with 0 and 1
        dummies = pd.get_dummies(df[group_col], prefix=col).astype(np.uint8)
        df = pd.concat([df, dummies], axis=1)
        df.drop(group_col, axis=1, inplace=True)  # Remove temporary grouping column

    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    level_col = f'{col}_Level'
    df[level_col] = est.fit_transform(df[[col]]).astype(np.uint8)

    if drop_original:
        df.drop(col, axis=1, inplace=True)

    return df

def process_categorical(col, df, fillna=0, drop_original=True):
    """Process categorical columns with encoding.
    Args:
        col: Column name to process
        df: DataFrame containing the data
        fillna: Value to fill missing values with
        drop_original: Whether to drop original column after processing
    Returns:
        Processed DataFrame
    """
    unique_vals = df[col].dropna().unique()
    val_map = {v: i + 1 for i, v in enumerate(unique_vals)}
    code_col = f'{col}_Code'
    df[code_col] = df[col].map(val_map).fillna(fillna).astype(np.uint8)

    if drop_original:
        df.drop(col, axis=1, inplace=True)

    return df

#Main data processing pipeline
def main():
    # Load data
    df = pd.read_csv(file_path)
    print("\nOriginal data shape:", df.shape)

    # Handle missing values
    print("\nProcessing missing values...")
    df['missing_count'] = df.isnull().sum(axis=1)
    rows_before = len(df)
    df = df[df['missing_count'] <= 5].copy()
    rows_removed = rows_before - len(df)
    df.drop('missing_count', axis=1, inplace=True)
    print(f"Removed {rows_removed} rows (with >5 missing attributes)")
    print("Cleaned data shape:", df.shape)
    print("\nProcessing individual columns...")
    #Client_Income
    print("\nProcessing Client_Income...")
    df = process_numeric('Client_Income', df, money=True,
                         bins=[0, 10000, 20000, 30000, np.inf],
                         labels=['<10k', '10k-20k', '20k-30k', '>30k'],
                         drop_original=False)
    analyze_column(df, 'Client_Income')

    #Binary/count variables
    binary_cols = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count']
    for col in binary_cols:
        print(f"\nProcessing {col}...")
        df[col].fillna(0, inplace=True)
        df[col] = df[col].astype(np.uint8)
        analyze_column(df, col, 'count')

    #Special handling for Child_Count
    print("\nProcessing Child_Count binning...")
    df = process_numeric('Child_Count', df, bins=[0, 2, 3, np.inf],
                         labels=['nochild', 'lesschild', 'manychild'],
                         drop_original=True)

    #Monetary variables
    money_cols = ['Credit_Amount', 'Loan_Annuity']
    for col in money_cols:
        print(f"\nProcessing {col}...")
        df = process_numeric(col, df, money=True, drop_original=True)
        analyze_column(df, f'{col}_Level')

    #Categorical variables
    cat_cols = ['Accompany_Client', 'Client_Income_Type', 'Client_Education']
    for col in cat_cols:
        print(f"\nProcessing {col}...")
        df = process_categorical(col, df)
        analyze_column(df, f'{col}_Code', 'count')

    #Other categorical variables
    other_cat_cols = ['Client_Marital_Status', 'Client_Housing_Type']
    for col in other_cat_cols:
        print(f"\nProcessing {col}...")
        df = process_categorical(col, df)
        analyze_column(df, f'{col}_Code', 'count')

    #Client_Gender
    print("\nProcessing Client_Gender...")
    gender_map = {'Male': 0, 'Female': 1, 'XNA': 2}
    df['Client_Gender'] = df['Client_Gender'].map(gender_map).fillna(3).astype(np.uint8)
    analyze_column(df, 'Client_Gender', 'count')

    #Loan_Contract_Type
    print("\nProcessing Loan_Contract_Type...")
    df['Loan_Contract_Type'] = df['Loan_Contract_Type'].fillna(df['Loan_Contract_Type'].mode()[0])
    df['Loan_Contract_Type'] = df['Loan_Contract_Type'].map({'CL': 0, 'RL': 1}).astype(np.uint8)
    analyze_column(df, 'Loan_Contract_Type', 'count')

    #Population_Region_Relative
    print("\nProcessing Population_Region_Relative...")
    df = process_numeric('Population_Region_Relative', df, money=True)

    #Age_Days
    print("\nProcessing Age_Days...")
    df = process_numeric('Age_Days', df, money=True)
    df['Age'] = df['Age_Days'] / 365
    df = process_numeric('Age', df, bins=[0, 25, 35, 55, np.inf],
                         labels=['<25', '25-35', '35-55', '>55'],
                         drop_original=True)
    df.drop('Age_Days', axis=1, inplace=True)

    #Employed_Days
    df['Employed_Days'] = df['Employed_Days'].astype(str).str.replace('x', '').str.replace('@', '')
    df['Employed_Days'] = pd.to_numeric(df['Employed_Days'], errors='coerce')
    employed_median = df['Employed_Days'].median()# Fill with median
    df['Employed_Days'].fillna(employed_median, inplace=True)
    df['Employed_Months'] = df['Employed_Days'] / 30#Convert to months

    #Days-type variables
    days_cols = ['Registration_Days', 'ID_Days', 'Own_House_Age']
    for col in days_cols:
        print(f"\nProcessing {col}...")
        df = process_numeric(col, df, money=True)
        analyze_column(df, col)

    #Tag variables
    df['Mobile_Tag'].fillna(0, inplace=True)
    df['Mobile_Tag'] = df['Mobile_Tag'].astype(int)
    tag_cols = ['Homephone_Tag', 'Workphone_Working']
    for col in tag_cols:
        print(f"\nProcessing {col}...")
        df[col].fillna(0, inplace=True)
        df[col] = df[col].astype(np.uint8)
        analyze_column(df, col, 'count')

    #Client_Occupation
    occupation_list = df['Client_Occupation'].dropna().unique()
    occupation_map = {v: i + 1 for i, v in enumerate(occupation_list)}  # Start numbering from 1
    df['Occupation_Code'] = df['Client_Occupation'].map(occupation_map)
    df['Occupation_Code'] = df['Occupation_Code'].fillna(0).astype(int)

    #Other processing
    print("\nProcessing Client_Family_Members...")
    df['Client_Family_Members'].fillna(1, inplace=True)
    df['Client_Family_Members'] = df['Client_Family_Members'].astype(np.uint8)
    analyze_column(df, 'Client_Family_Members', 'count')

    rating_cols = ['Cleint_City_Rating', 'Application_Process_Day', 'Application_Process_Hour']
    for col in rating_cols:
        print(f"\nProcessing {col}...")
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = df[col].astype(np.uint8)
        analyze_column(df, col, 'count')

    binary_tag_cols = ['Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag']
    for col in binary_tag_cols:
        print(f"\nProcessing {col}...")
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(np.uint8)
        analyze_column(df, col, 'count')

    #Type_Organization
    print("\nProcessing Type_Organization...")
    df['Type_Organization'].fillna('Other', inplace=True)
    analyze_column(df, 'Type_Organization', 'count')

    #Score sources
    score_cols = ['Score_Source_1', 'Score_Source_2', 'Score_Source_3']
    for col in score_cols:
        print(f"\nProcessing {col}...")
        df = process_numeric(col, df, money=True)
        analyze_column(df, col)

    #Phone_Change
    phone_mean = df['Phone_Change'].dropna().mean()
    df.loc[:, 'Phone_Change'] = df['Phone_Change'].fillna(phone_mean)
    analyze_column(df, 'Phone_Change', plot_type='hist')
    bins = [0, 200, 1000, np.inf]#Binning and discretization
    labels = ['low', 'medium', 'high']
    df = df.assign(Phone_Change_Group=pd.cut(df['Phone_Change'], bins=bins, labels=labels))
    phone_dummies = pd.get_dummies(df['Phone_Change_Group'], prefix='Phone_Change').astype(np.uint8)#One-hot encoding (ensuring 0/1 values)
    df = pd.concat([df, phone_dummies], axis=1)#Merge and remove temporary columns
    df = df.drop(columns=['Phone_Change_Group', 'Phone_Change'])

    #Other numerical variables
    other_num_cols = ['Social_Circle_Default', 'Credit_Bureau']
    for col in other_num_cols:
        print(f"\nProcessing {col}...")
        df[col].fillna(df[col].median() if df[col].dtype == 'float' else df[col].mode()[0], inplace=True)
        analyze_column(df, col)

    columns_to_drop = [
        'Client_Occupation', 'Type_Organization'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

    #Standardization
    print("\nPerforming standardization...")
    scaler = StandardScaler()
    cols_to_scale = ['Client_Income', 'Registration_Days', 'ID_Days', 'Employed_Months']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    #Save results
    print("\nSaving processed data...")
    df.to_csv(save_path, index=False)

    #Plot Default attribute distribution
    print("\nPlotting Default attribute distribution...")
    plot_default_pie(df)

    print("\n=== Processing complete ===")
    print("Processed data shape:", df.shape)
    print("First 3 rows sample:")
    print(df.head(3))

if __name__ == "__main__":
    main()