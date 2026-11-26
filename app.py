
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
    roc_auc_score,
    roc_curve
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="FraudGuard | Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. LIGHT THEME DESIGN
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F5F7FA;
    color: #25344A;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #172B4D;
    border-right: 1px solid #243B5D;
}

[data-testid="stSidebar"] * {
    color: #F5F7FA;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #E1EAF5;
}

h1 {
    color: #172B4D !important;
    font-weight: 750 !important;
    letter-spacing: -0.7px;
}

h2, h3 {
    color: #243B5D !important;
    font-weight: 650 !important;
}

p, label {
    color: #66758A;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3E8EF;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 7px rgba(25, 43, 69, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #66758A;
}

div[data-testid="stMetricValue"] {
    color: #172B4D;
    font-weight: 750;
}

div[data-testid="stButton"] button {
    background: #168C83;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 18px;
}

div[data-testid="stButton"] button:hover {
    background: #11766F;
    color: white;
    border: none;
}

div[data-testid="stDownloadButton"] button {
    background: #FFFFFF;
    color: #168C83;
    border: 1px solid #168C83;
    border-radius: 8px;
    font-weight: 600;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #AAB9C9;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E3E8EF;
    border-radius: 9px;
}

hr {
    border-color: #E1E7EF;
}

.small-note {
    color: #718096;
    font-size: 13px;
}

.section-label {
    color: #168C83;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.3px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E3E8EF;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATA AND MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")


model = load_model()
df = load_data()

features = df.drop(columns=["Class"])
target = df["Class"]


# ==================================================
# 4. COMMON FUNCTIONS
# ==================================================

def page_heading(title, description):

    st.markdown(
        '<div class="section-label">FRAUDGUARD ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.title(title)

    st.markdown(
        f'<p class="small-note">{description}</p>',
        unsafe_allow_html=True
    )


def show_chart_style(ax):

    ax.set_facecolor("#FFFFFF")

    ax.tick_params(colors="#66758A")

    ax.xaxis.label.set_color("#66758A")
    ax.yaxis.label.set_color("#66758A")

    ax.title.set_color("#172B4D")

    for spine in ax.spines.values():
        spine.set_color("#E3E8EF")


def get_evaluation():

    if "evaluation_results" not in st.session_state:

        with st.spinner("Evaluating model performance..."):

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=0.2,
                random_state=42,
                stratify=target
            )

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )

            st.session_state["evaluation_results"] = {
                "y_test": y_test,
                "predictions": predictions,
                "probabilities": probabilities,
                "report": report
            }

    return st.session_state["evaluation_results"]


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.markdown("# 💳 FraudGuard")

    st.caption("CREDIT CARD SECURITY SYSTEM")

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Fraud Detection",
            "Batch Prediction",
            "Model Performance"
        ]
    )

    st.markdown("---")

    st.markdown("**MODEL INFORMATION**")

    st.caption("Algorithm: Random Forest")
    st.caption("Dataset: Credit Card Transactions")
    st.caption("Mode: Academic ML Prototype")

    st.markdown("---")

    st.caption("Machine Learning Project")
    st.caption("Computer Science and Business Systems")


# ==================================================
# 6. DASHBOARD
# ==================================================

if page == "Dashboard":

    page_heading(
        "Credit Card Fraud Detection",
        "An overview of transaction data and fraud patterns."
    )

    total = len(df)

    fraud = int(target.sum())

    genuine = total - fraud

    fraud_percent = fraud / total * 100

    st.markdown("### Transaction Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")

    c2.metric("Genuine Transactions", f"{genuine:,}")

    c3.metric("Fraud Transactions", f"{fraud:,}")

    c4.metric("Fraud Rate", f"{fraud_percent:.3f}%")

    st.markdown("---")

    left, right = st.columns(2)

    # Transaction distribution chart

    with left:

        st.markdown("### Transaction Distribution")

        fig, ax = plt.subplots(figsize=(5, 3))

        fig.patch.set_facecolor("#FFFFFF")

        ax.pie(
            [genuine, fraud],
            labels=["Genuine", "Fraud"],
            autopct="%1.2f%%",
            startangle=90,
            colors=["#168C83", "#DC6B70"],
            textprops={"color": "#25344A", "fontsize": 9},
            wedgeprops={
                "width": 0.42,
                "edgecolor": "#FFFFFF"
            }
        )

        ax.set_title(
            "Genuine vs Fraud",
            color="#172B4D",
            fontsize=12
        )

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # Transaction amount chart

    with right:

        st.markdown("### Transaction Amount Analysis")

        fig, ax = plt.subplots(figsize=(5, 3))

        fig.patch.set_facecolor("#FFFFFF")

        genuine_amounts = df.loc[
            df["Class"] == 0, "Amount"
        ]

        fraud_amounts = df.loc[
            df["Class"] == 1, "Amount"
        ]

        ax.hist(
            genuine_amounts,
            bins=35,
            range=(0, 500),
            alpha=0.75,
            color="#168C83",
            label="Genuine"
        )

        ax.hist(
            fraud_amounts,
            bins=35,
            range=(0, 500),
            alpha=0.85,
            color="#DC6B70",
            label="Fraud"
        )

        ax.set_xlabel("Transaction Amount")

        ax.set_ylabel("Number of Transactions")

        ax.set_title(
            "Amount Distribution",
            fontsize=12
        )

        ax.legend(
            facecolor="#FFFFFF",
            edgecolor="#E3E8EF",
            labelcolor="#25344A"
        )

        show_chart_style(ax)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    st.markdown("---")

    st.markdown("### Recent Dataset Records")

    st.dataframe(
        df.head(8),
        use_container_width=True,
        height=240,
        hide_index=True
    )

    st.caption("Showing 8 sample records from the original dataset.")


# ==================================================
# 7. INDIVIDUAL FRAUD DETECTION
# ==================================================

elif page == "Fraud Detection":

    page_heading(
        "Individual Transaction Analysis",
        "Select a transaction and check the prediction made by the trained model."
    )

    st.markdown("### Select Transaction")

    choice = st.radio(
        "Choose transaction category",
        [
            "Genuine Transaction",
            "Fraudulent Transaction"
        ],
        horizontal=True
    )

    if choice == "Genuine Transaction":

        available = df.index[
            df["Class"] == 0
        ].tolist()

    else:

        available = df.index[
            df["Class"] == 1
        ].tolist()

    selected = st.selectbox(
        "Transaction Record ID",
        available[:1000]
    )

    transaction = df.loc[
        [selected],
        features.columns
    ]

    with st.expander("View Transaction Details"):

        st.dataframe(
            transaction,
            use_container_width=True,
            height=180
        )

    if st.button("Analyze Transaction", type="primary"):

        prediction = model.predict(transaction)[0]

        probability = model.predict_proba(
            transaction
        )[0][1]

        st.markdown("---")

        st.markdown("### Analysis Result")

        if prediction == 1:

            st.error("⚠ FRAUDULENT TRANSACTION DETECTED")

        else:

            st.success("✓ GENUINE TRANSACTION")

        c1, c2 = st.columns(2)

        c1.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Predicted Category",
            "Fraud" if prediction == 1 else "Genuine"
        )

        st.progress(float(probability))

        actual = df.loc[selected, "Class"]

        st.caption(
            "Recorded dataset label: "
            + ("Fraud" if actual == 1 else "Genuine")
        )

        st.caption(
            "The displayed probability is the model's estimated "
            "fraud probability, not a guarantee of fraud."
        )


# ==================================================
# 8. BATCH PREDICTION
# ==================================================

elif page == "Batch Prediction":

    page_heading(
        "Batch Transaction Analysis",
        "Upload a CSV file to analyze multiple transactions together."
    )

    st.markdown("### Upload Transaction File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            new_data = pd.read_csv(uploaded_file)

            if "Class" in new_data.columns:

                new_data = new_data.drop(
                    columns=["Class"]
                )

            missing = (
                set(features.columns) - set(new_data.columns)
            )

            extra = (
                set(new_data.columns) - set(features.columns)
            )

            if missing or extra:

                st.error(
                    "The uploaded file does not contain "
                    "the required 30 feature columns."
                )

                if missing:

                    st.write(
                        "Missing columns:",
                        sorted(missing)
                    )

                if extra:

                    st.write(
                        "Unexpected columns:",
                        sorted(extra)
                    )

            elif new_data.empty:

                st.warning("The uploaded CSV file is empty.")

            else:

                new_data = new_data[features.columns]

                new_data = new_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if not np.isfinite(
                    new_data.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "The file contains missing or invalid numeric values."
                    )

                else:

                    st.success(
                        f"File uploaded successfully: "
                        f"{len(new_data):,} transactions"
                    )

                    if st.button(
                        "Predict All Transactions",
                        type="primary"
                    ):

                        with st.spinner("Analyzing transactions..."):

                            predictions = model.predict(new_data)

                            probabilities = model.predict_proba(
                                new_data
                            )[:, 1]

                            results = new_data.copy()

                            results["Prediction"] = np.where(
                                predictions == 1,
                                "Fraud",
                                "Genuine"
                            )

                            results["Fraud_Probability"] = probabilities

                            st.session_state["batch_results"] = results

                    if "batch_results" in st.session_state:

                        results = st.session_state["batch_results"]

                        fraud_count = int(
                            (results["Prediction"] == "Fraud").sum()
                        )

                        genuine_count = (
                            len(results) - fraud_count
                        )

                        st.markdown("---")

                        st.markdown("### Prediction Summary")

                        a, b, c = st.columns(3)

                        a.metric(
                            "Total Analyzed",
                            f"{len(results):,}"
                        )

                        b.metric(
                            "Fraud Predicted",
                            f"{fraud_count:,}"
                        )

                        c.metric(
                            "Genuine Predicted",
                            f"{genuine_count:,}"
                        )

                        st.markdown("### Sample Prediction Results")

                        st.dataframe(
                            results.head(10),
                            use_container_width=True,
                            height=280
                        )

                        csv = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Complete Results",
                            csv,
                            "fraud_predictions.csv",
                            "text/csv"
                        )

                        st.caption(
                            "These categories are model predictions, "
                            "not confirmed fraud labels."
                        )

        except Exception as error:

            st.error(
                f"Unable to read the uploaded file: {error}"
            )


# ==================================================
# 9. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    page_heading(
        "Model Performance",
        "Evaluation of the Random Forest model on the held-out test dataset."
    )

    evaluation = get_evaluation()

    y_test = evaluation["y_test"]

    y_pred = evaluation["predictions"]

    y_prob = evaluation["probabilities"]

    report = evaluation["report"]

    accuracy = report["accuracy"]

    precision = report["1"]["precision"]

    recall = report["1"]["recall"]

    f1 = report["1"]["f1-score"]

    auc = roc_auc_score(y_test, y_prob)

    st.markdown("### Performance Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    c2.metric(
        "Fraud Precision",
        f"{precision * 100:.2f}%"
    )

    c3.metric(
        "Fraud Recall",
        f"{recall * 100:.2f}%"
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Fraud F1-Score",
        f"{f1:.2f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{auc:.4f}"
    )

    st.markdown("---")

    left, right = st.columns(2)

    # Confusion matrix

    with left:

        st.markdown("### Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 3.5))

        fig.patch.set_facecolor("#FFFFFF")

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="GnBu",
            cbar=False,
            linewidths=1,
            linecolor="#FFFFFF",
            xticklabels=["Genuine", "Fraud"],
            yticklabels=["Genuine", "Fraud"],
            annot_kws={"size": 11},
            ax=ax
        )

        ax.set_xlabel(
            "Predicted Class",
            color="#25344A"
        )

        ax.set_ylabel(
            "Actual Class",
            color="#25344A"
        )

        ax.set_title(
            "Actual vs Predicted",
            color="#172B4D"
        )

        ax.tick_params(colors="#66758A")

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # ROC curve

    with right:

        st.markdown("### ROC Curve")

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_prob
        )

        fig, ax = plt.subplots(figsize=(5, 3.5))

        fig.patch.set_facecolor("#FFFFFF")

        ax.plot(
            fpr,
            tpr,
            color="#168C83",
            linewidth=2,
            label=f"Random Forest (AUC = {auc:.4f})"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#AAB9C9",
            label="Random Classifier"
        )

        ax.set_xlabel("False Positive Rate")

        ax.set_ylabel("True Positive Rate")

        ax.set_title(
            "ROC Curve",
            color="#172B4D"
        )

        ax.legend(
            facecolor="#FFFFFF",
            edgecolor="#E3E8EF",
            labelcolor="#25344A",
            fontsize=8
        )

        show_chart_style(ax)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    st.markdown("---")

    with st.expander("View Detailed Classification Report"):

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    st.caption(
        "Evaluation uses the stratified 20% test split. "
        "Accuracy alone is not sufficient for judging fraud detection "
        "because fraudulent transactions are rare."
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    '<p style="text-align:center;color:#718096;font-size:12px;">'
    'FRAUDGUARD | Credit Card Fraud Detection | Machine Learning Project'
    '</p>',
    unsafe_allow_html=True
)
