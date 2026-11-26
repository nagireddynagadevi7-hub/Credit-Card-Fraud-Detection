
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD DATA AND MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

model = load_model()
df = load_data()

features = df.drop("Class", axis=1)
target = df["Class"]

# Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- SIDEBAR ----------------

st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fraud Detection",
        "Batch Prediction",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Powered by Random Forest ML")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("💳 Credit Card Fraud Detection System")
    st.write("Real-time style transaction analysis using Machine Learning")

    total = len(df)
    fraud = int(df["Class"].sum())
    genuine = total - fraud
    fraud_percent = fraud / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Genuine Transactions", f"{genuine:,}")
    c3.metric("Fraud Transactions", f"{fraud:,}")
    c4.metric("Fraud Percentage", f"{fraud_percent:.3f}%")

    st.subheader("Transaction Distribution")

    fig, ax = plt.subplots()
    ax.pie(
        [genuine, fraud],
        labels=["Genuine", "Fraud"],
        autopct="%1.2f%%",
        startangle=90
    )
    ax.set_title("Genuine vs Fraud Transactions")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Transaction Amount Analysis")

    fig, ax = plt.subplots()
    sns.histplot(
        data=df[df["Class"] == 0],
        x="Amount",
        bins=40,
        color="green",
        label="Genuine",
        ax=ax
    )
    sns.histplot(
        data=df[df["Class"] == 1],
        x="Amount",
        bins=40,
        color="red",
        label="Fraud",
        ax=ax
    )
    ax.set_xlim(0, 500)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- FRAUD DETECTION ----------------

elif page == "Fraud Detection":

    st.title("🔍 Individual Transaction Detection")

    st.write(
        "Select a real transaction from the Kaggle dataset "
        "to demonstrate the trained model."
    )

    choice = st.radio(
        "Choose transaction type",
        ["Genuine Transaction", "Fraudulent Transaction"]
    )

    if choice == "Genuine Transaction":
        available = df.index[df["Class"] == 0].tolist()
    else:
        available = df.index[df["Class"] == 1].tolist()

    selected = st.selectbox(
        "Select transaction record",
        available[:1000]
    )

    transaction = df.loc[[selected], features.columns]

    st.write("Selected transaction details")
    st.dataframe(transaction, use_container_width=True)

    if st.button("Analyze Transaction", type="primary"):

        prediction = model.predict(transaction)[0]
        probability = model.predict_proba(transaction)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ FRAUDULENT TRANSACTION DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        st.metric(
            "Predicted Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "Dataset's recorded class: "
            + ("Fraud" if df.loc[selected, "Class"] == 1 else "Genuine")
        )

        st.progress(float(probability))

# ---------------- BATCH PREDICTION ----------------

elif page == "Batch Prediction":

    st.title("📂 Batch Transaction Prediction")

    st.write(
        "Upload a CSV containing transaction features "
        "to predict multiple transactions."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        if "Class" in new_data.columns:
            new_data = new_data.drop(columns=["Class"])

        missing = set(features.columns) - set(new_data.columns)
        extra = set(new_data.columns) - set(features.columns)

        if missing or extra:
            st.error(
                "Incorrect columns. Please upload a CSV with "
                "the same 30 feature columns as the Kaggle dataset."
            )
        else:
            new_data = new_data[features.columns]

            if not np.isfinite(new_data.to_numpy(dtype=float)).all():
                st.error("The CSV contains missing or invalid numeric values.")
            else:
                if st.button("Predict All Transactions", type="primary"):

                    predictions = model.predict(new_data)
                    probabilities = model.predict_proba(new_data)[:, 1]

                    results = new_data.copy()
                    results["Prediction"] = np.where(
                        predictions == 1, "Fraud", "Genuine"
                    )
                    results["Fraud_Probability"] = probabilities

                    st.subheader("Prediction Results")
                    st.dataframe(results, use_container_width=True)

                    st.metric(
                        "Fraud Transactions Detected",
                        int((predictions == 1).sum())
                    )

                    csv = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        csv,
                        "fraud_predictions.csv",
                        "text/csv"
                    )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("📊 Machine Learning Model Performance")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{report['accuracy'] * 100:.2f}%")
    c2.metric("Fraud Precision", f"{report['1']['precision'] * 100:.2f}%")
    c3.metric("Fraud Recall", f"{report['1']['recall'] * 100:.2f}%")

    c4, c5 = st.columns(2)

    c4.metric("Fraud F1-Score", f"{report['1']['f1-score']:.2f}")
    c5.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.caption(
        "Evaluation is performed on the held-out 20% test dataset."
    )

# ---------------- FOOTER ----------------

st.sidebar.markdown("---")
st.sidebar.caption("Credit Card Fraud Detection | ML Project")
