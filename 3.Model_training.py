"""
Credit Default Prediction Model Evaluation

Overview
This Python script provides a framework for evaluating machine learning models for credit default prediction.
It includes data preprocessing, feature selection, model training with nested cross-validation, and performance evaluation on test data.

Key Features
Data Processing:
Automated cleaning of numeric columns
Handling of missing values with median imputation
Correlation analysis to identify key predictive features

Model Evaluation:
Three model types: Logistic Regression, Random Forest, and Neural Network
Nested cross-validation for robust performance estimation
Comprehensive metrics including Accuracy, F1, Precision, Recall, and AUC-ROC

Enhanced Visualization:
Combined confusion matrices for easy model comparison
Beautifully styled heatmaps for correlation analysis
Clear labeling of all visualizations

Test Prediction:
Consistent preprocessing for test data
Performance evaluation when labels are available
Prediction output when labels are unavailable
Technical Implementation

The solution leverages:
Scikit-learn for machine learning pipelines
Matplotlib and Seaborn for professional visualizations
Pandas for efficient data manipulation
Stratified sampling to maintain class distribution

Usage Instructions
Place your training data in data/Cleaned_data.csv
Place test data (optional) in data/Text_data_clean.csv

Run the script to:
Analyze feature correlations
Train and evaluate models
Generate prediction results
Produce publication-quality visualizations
The script automatically handles all preprocessing steps and provides clear performance metrics for each model, making it easy to compare different approaches for credit default prediction.

Output
Feature correlation heatmap
Combined confusion matrices for cross-validation results
Combined confusion matrices for test predictions
Detailed performance metrics in the console

All visualizations are properly labeled and styled for professional presentation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer)

#Data Loading and Preprocessing
data_path = r"data/Train_data_clean.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=['Default', 'ID'])
y = df['Default'].astype(int)

def clean_column(series):
    series = series.astype(str).str.replace('[^\d.-]', '', regex=True)
    series = series.replace('', np.nan)
    return pd.to_numeric(series, errors='coerce')

X_cleaned = X.apply(clean_column)
X_cleaned = X_cleaned.apply(lambda col: col.fillna(col.median()) if col.notna().any() else 0)

#Correlation Analysis
corr_matrix = X_cleaned.copy()
corr_matrix['Default'] = y
correlations = corr_matrix.corr()['Default'].drop('Default').sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[correlations.index[:10].tolist() + ['Default']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
plt.title("Top 10 Features Most Correlated with Default", pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Feature Selection
selector = RandomForestClassifier(n_estimators=100, random_state=42)
selector.fit(X_cleaned, y)
importances = selector.feature_importances_
top_features = X_cleaned.columns[np.argsort(importances)[-20:][::-1]]
X_selected = X_cleaned[top_features]

#Model Dictionary
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1500, random_state=33))
    ]),
    "Random Forest": Pipeline([
        ('classifier', RandomForestClassifier(class_weight='balanced_subsample', n_estimators=100, random_state=33, n_jobs=-1))
    ]),
    "Neural Network": Pipeline([
        ('scaler', StandardScaler()),  # Neural networks are sensitive to feature scaling
        ('classifier', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                     solver='adam', alpha=1e-4, max_iter=500, random_state=33))
    ])
}

#Nested Cross-Validation Evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'roc_auc': make_scorer(roc_auc_score)
}

#Create a figure for combined confusion matrices
plt.figure(figsize=(15, 5))
plt.suptitle("Model Performance Comparison (Cross-Validation)", y=1.05, fontsize=16)

for idx, (name, model) in enumerate(models.items()):
    print(f"\n{'='*50}\nModel Evaluation (Nested CV): {name}\n{'='*50}")
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_selected, y)):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        #Fit model
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        #Collect true and predicted values
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    #Calculate metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    print(f"Accuracy          : {accuracy_score(all_y_true, all_y_pred):.4f}")
    print(f"F1 (macro)        : {f1_score(all_y_true, all_y_pred, average='macro'):.4f}")
    print(f"Precision (macro) : {precision_score(all_y_true, all_y_pred, average='macro'):.4f}")
    print(f"Recall (macro)    : {recall_score(all_y_true, all_y_pred, average='macro'):.4f}")
    print(f"AUC-ROC           : {roc_auc_score(all_y_true, all_y_pred):.4f}")

    #Plot confusion matrix in subplot
    plt.subplot(1, 3, idx+1)
    cm = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'],
                cbar=False)
    plt.title(f'{name}\nAccuracy: {accuracy_score(all_y_true, all_y_pred):.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

#Prediction on Test Data
test_path = r"data/Test_data_clean.csv"
test_df = pd.read_csv(test_path)

#Keep same columns as training data and apply same cleaning
test_X = test_df[X.columns]  # Ensure consistent column order
test_X_cleaned = test_X.apply(clean_column)
test_X_cleaned = test_X_cleaned.apply(lambda col: col.fillna(col.median()) if col.notna().any() else 0)

#Select features used in training
test_X_selected = test_X_cleaned[top_features]

#Extract true labels if available
if 'Default' in test_df.columns:
    test_y = test_df['Default'].astype(int)
else:
    test_y = None

#Create a figure for test confusion matrices
if test_y is not None:
    plt.figure(figsize=(15, 5))
    plt.suptitle("Model Performance on Test Data", y=1.05, fontsize=16)

#Predict with trained models
for idx, (name, model) in enumerate(models.items()):
    print(f"\n{'='*50}\nModel Prediction: {name}\n{'='*50}")
    model.fit(X_selected, y)  # Retrain with all training data
    test_pred_proba = model.predict_proba(test_X_selected)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)

    if test_y is not None:
        print(f"Test Accuracy          : {accuracy_score(test_y, test_pred):.4f}")
        print(f"Test F1 (macro)        : {f1_score(test_y, test_pred, average='macro'):.4f}")
        print(f"Test Precision (macro) : {precision_score(test_y, test_pred, average='macro'):.4f}")
        print(f"Test Recall (macro)    : {recall_score(test_y, test_pred, average='macro'):.4f}")
        print(f"Test AUC-ROC           : {roc_auc_score(test_y, test_pred):.4f}")

        # Plot test confusion matrix in subplot
        plt.subplot(1, 3, idx+1)
        cm_test = confusion_matrix(test_y, test_pred)
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Non-Default', 'Default'],
                    yticklabels=['Non-Default', 'Default'],
                    cbar=False)
        plt.title(f'{name}\nAccuracy: {accuracy_score(test_y, test_pred):.2f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    else:
        print("No true labels in test data, showing first 10 predictions:")
        print(test_pred[:10])

if test_y is not None:
    plt.tight_layout()
    plt.show()