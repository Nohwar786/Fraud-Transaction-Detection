# Fraud Transaction Detection

## Project Overview

Fraud Transaction Detection is a Machine Learning project that predicts whether a financial transaction is potentially fraudulent or legitimate.

The project uses transaction information such as transaction type, amount, account balances, and balance changes to identify suspicious transactions.

## Dataset

The dataset contains financial transaction records with the following important features:

- Step
- Transaction Type
- Amount
- Old Balance of Origin
- New Balance of Origin
- Old Balance of Destination
- New Balance of Destination
- Is Flagged Fraud
- Is Fraud

## Machine Learning

The project includes:

- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Class Imbalance Handling
- Logistic Regression
- Random Forest Classifier
- Model Evaluation

### Engineered Features

Two important balance-change features were created:

- `balance_change_orig`
- `balance_change_dest`

Additional error features were created:

- `errorBalanceOrig`
- `errorBalanceDest`

## Model

The final application uses a trained Random Forest classifier to predict fraudulent transactions.

The model is saved as:

```text
fraud_detection_model.pkl