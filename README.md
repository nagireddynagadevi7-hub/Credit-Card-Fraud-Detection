# Credit Card Fraud Detection System

## About the Project
This project is a machine learning-based Credit Card Fraud Detection System developed using Python and Streamlit.

The system analyses credit card transaction data and predicts whether a transaction is genuine or potentially fraudulent.

## Objectives
- Identify potentially fraudulent credit card transactions.
- Analyse transaction data using graphs.
- Predict individual transactions.
- Perform predictions on multiple transactions using CSV files.
- Evaluate the performance of the machine learning model.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Main Features
1. **Dashboard:** Displays transaction statistics and visualisations.
2. **Fraud Detection:** Predicts whether an individual transaction is genuine or fraudulent.
3. **Batch Prediction:** Processes multiple transactions from an uploaded CSV file.
4. **Model Performance:** Displays accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.

## Dataset
The project uses the Credit Card Fraud Detection dataset containing 284,807 transactions, including genuine and fraudulent transactions.

## Model Results
- Accuracy: 99.94%
- Precision: 84.21%
- Recall: 81.63%
- F1-score: 0.83
- ROC-AUC: 0.975

These results are from the model's test data evaluation.

## How to Run
1. Install Python.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Keep the dataset (`creditcard.csv`) and trained model (`fraud_model.pkl`) in the project folder.
4. Run the application:

```bash
streamlit run app.py
```

## Project Limitation
This is an academic demonstration using historical transaction data. It is not connected to a live banking or payment system and does not automatically block transactions.
