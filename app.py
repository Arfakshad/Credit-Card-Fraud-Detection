
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f2027,
        #203a43,
        #2c5364
    );
}

h1, h2, h3 {
    color: white !important;
}

[data-testid="stMetricValue"] {
    color: #00ff99;
}

[data-testid="stMetricLabel"] {
    color: white;
}

p, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("credit_card_fraud_model.pkl")

# --------------------------------------------------
# LOAD DATASET (OPTIONAL)
# --------------------------------------------------
df = None

if os.path.exists("creditcard.csv"):
    df = pd.read_csv("creditcard.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("💳 Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Analysis",
        "Fraud Prediction",
        "Model Performance"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":

    st.title("💳 Credit Card Fraud Detection System")

    st.markdown("""
    ### Machine Learning Project

    Detect fraudulent credit card transactions using a trained machine learning model.

    #### Features
    - Fraud Detection
    - Dataset Analytics
    - Interactive Dashboard
    - CSV Upload
    - Prediction Download
    """)

# --------------------------------------------------
# DATASET ANALYSIS
# --------------------------------------------------
elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    if df is not None:

        total = len(df)
        fraud = len(df[df["Class"] == 1])
        legit = len(df[df["Class"] == 0])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", f"{total:,}")
        col2.metric("Fraud Transactions", f"{fraud:,}")
        col3.metric("Legitimate", f"{legit:,}")

        st.subheader("Fraud Distribution")

        pie_df = pd.DataFrame({
            "Category": ["Legitimate", "Fraud"],
            "Count": [legit, fraud]
        })

        fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    else:
        st.warning("creditcard.csv not found.")

# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------
elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction")

    st.write(
        "Upload a CSV file containing transactions."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(input_df.head())

        try:

            predictions = model.predict(input_df)

            input_df["Prediction"] = predictions

            input_df["Prediction"] = input_df[
                "Prediction"
            ].map({
                0: "Legitimate",
                1: "Fraud"
            })

            st.subheader("Prediction Results")
            st.dataframe(input_df.head())

            csv = input_df.to_csv(index=False)

            st.download_button(
                label="⬇ Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.metric(
        label="Training Accuracy",
        value="95.81%"
    )

    st.metric(
        label="Testing Accuracy",
        value="92.89%"
    )

    st.success(
        "The model performs well on unseen data and can identify fraudulent transactions effectively."
    )
