
import pandas as pd

# Load Kaggle dataset
df = pd.read_csv("creditcard.csv")

# Display first five transactions
print("\nFIRST FIVE TRANSACTIONS")
print(df.head())

# Display dataset size
print("\nDATASET SHAPE")
print(df.shape)

# Count genuine and fraudulent transactions
print("\nTRANSACTION COUNTS")
print(df["Class"].value_counts())

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum().sum())

print("\nDataset loaded successfully!")
