
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

# 2. Separate input and output
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 5. Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 7. Evaluate model
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC AUC SCORE:", roc_auc_score(y_test, y_prob))

# 8. Save trained model
joblib.dump(model, "fraud_model.pkl")

print("\nModel saved successfully!")
