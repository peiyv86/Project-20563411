## Data Set Information

The dataset contains customer and loan information from a financial institution. It includes anonymized customer demographics, credit history, loan characteristics, and repayment behaviors. The objective is to predict the likelihood of loan default.

---

### Business Understanding

#### What is Loan Default Prediction?

In the financial industry, credit risk is crucial to profitability. When the borrower fails to perform the statutory debt repayment obligations, the loan default occurs. Timely and accurate prediction of loan default enables institutions to:

- Reduce credit losses  
- Improve customer segmentation  
- Strengthen loan strategy

Machine learning can automatically detect the risk patterns in a large number of historical financial data, and provide predictions for whether new applicants may default.

#### Why Machine Learning?

Traditional credit scoring depends on manually defined rules. However, financial risk in the real world is affected by the nonlinear interaction between various characteristics. ML models (such as random forest and neural network) can:

- Learn complex patterns from historical defaults  
- Automatically sort element importance

In this project, we establish and evaluate multiple classification models to improve the accuracy of default risk prediction.

---

### Source

The dataset is publicly available or synthetically generated to reflect real-world financial data. Personally identifiable information has been removed.

---

## Data Understanding

```
Data Set Characteristics:  Tabular  
Area: Finance / Credit Risk  
Attribute Characteristics: Mixed (Categorical, Numerical)  
Missing Values? Yes (handled during preprocessing)
```

### Attribute Description

The dataset contains 40+ attributes. Below are key categories:

**Identification**  
- `ID`: Loan application ID (unique identifier, not used for modeling)

**Demographic Information**  
- `Age_Days`: Age in days (negative value indicates past from reference date)  
- `Client_Gender`: Gender of the client (Male/Female)  
- `Child_Count`: Number of children  
- `Client_Family_Members`: Total family members  
- `Cleint_City_Rating`: City rating (1 = low, 3 = high)

**Employment & Housing**  
- `Employed_Days`: Days employed (negative means days before application)  
- `Client_Education`: Highest education level  
- `Client_Occupation`: Occupation type  
- `Client_Housing_Type`: Housing type  
- `Own_House_Age`: Age of owned house in years

**Financial Status**  
- `Client_Income`: Monthly income in USD  
- `Credit_Amount`: Loan amount requested  
- `Loan_Annuity`: Loan annuity in USD  
- `Client_Income_Type`: Income type (e.g., working, pensioner)

**Credit & History**  
- `Active_Loan`: Existing open loans (0 = No, 1 = Yes)  
- `Score_Source_1/2/3`: Standardized credit scores (nulls filled with median)  
- `Social_Circle_Default`: Defaults in last 60 days (median filled)  
- `Credit_Bureau`: Number of credit inquiries in the last year

**Contact Information**  
- `Mobile_Tag`, `Homephone_Tag`, `Workphone_Working`: Contact status (1 = available)  
- `Phone_Change`: Days since phone number changed

**Application Metadata**  
- `Application_Process_Day`: Day of the week loan was applied  
- `Application_Process_Hour`: Hour of day loan was applied  
- `Client_Permanent_Match_Tag`: Permanent address matches contact address (1/0)  
- `Client_Contact_Work_Tag`: Contact address matches work address (1/0)

**Assets Ownership**  
- `Car_Owned`, `Bike_Owned`, `House_Own`: Ownership indicators (0/1)

**Administrative Fields**  
- `Registration_Days`, `ID_Days`: Days since registration / ID change

**Company Profile**  
- `Type_Organization`: Organization type (rare categories merged as 'Other')

**Target**  
- `Default`: Loan default status (1 = defaulted, 0 = non-default)

---

### Attribute Information

The data includes:

- Demographics: Age, Gender, Occupation, City score  
- Financials: Income, Loan Amount, Interest Rate, Tenure  
- History: Prior defaults, Credit score  
- Target: Default (1 = Yes, 0 = No)

---

## Team Collaboration - Directory Structure

#### Instructions

```
- Clone the GitHub repository  
- Please run the notebooks in sequence  
- At the end of the step 3, there will be 2 model files created.

├── data  
│    ├── Original_record.csv  
│    ├── Test_data.csv  
│    ├── Test_data_clean.csv  
│    ├── Train_data_clean.csv  
├── images  
│    ├── Confusion matrix - test set.png  
│    ├── Confusion matrix - test set.png  
│    ├── Correlation thermogram (Pearson Correlation Coefficient).png  
├── 1.Data_Exploration.py  
├── 2.Data_preprocessing.py  
├── 3.Model_training.py  
├── presentation  
│    ├── Predicting loan default using machine learning.pptx  
```

---

## Data Preparation and Visualization

```
Code Used: Python  
Packages: pandas, numpy, matplotlib, seaborn  

Steps:  
- Missing value imputation  
- Categorical encoding (one-hot & label)  
- Outlier detection  
- Feature normalization and transformation  
```

---

## Modeling and Evaluation

```
Code Used: Python  
Packages: scikit-learn, xgboost, keras, imbalanced-learn  

Models:  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Neural Network (MLP)  

Metrics:  
- Accuracy, Precision, Recall, F1-score  
- AUC-ROC Curve  
```

---

## Outcome Summary

**By building robust prediction models, this project aims to optimize:**

- Credit risk profiling  
- Approval decision automation  
- Risk-based pricing  
- Capital provisioning and reserve planning  
- Fraud detection and regulatory compliance
