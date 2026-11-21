import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ChurnLens | Churn Analytics & Risk Predictor", layout="wide")

DB_PATH = Path("data/churn.db")
MODEL_PATH = Path("models/churn_pipeline.pkl")


@st.cache_data
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM customers", conn)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


df = load_data()
artifact = load_artifact()
pipeline = artifact["pipeline"]
defaults = artifact["defaults"]

st.title("ChurnLens: Customer Churn Analytics & Risk Simulator")
st.caption(f"Active Model: {artifact['model_name']} | Test ROC-AUC: {artifact['roc_auc']:.3f}")

tab1, tab2 = st.tabs(["Executive KPIs & Churn Drivers", "What-If Risk Simulator"])

with tab1:
    total_cust = len(df)
    churn_rate = df["churn_flag"].mean() * 100
    rev_at_risk = df.loc[df["churn_flag"] == 1, "MonthlyCharges"].sum()
    avg_churn_tenure = df.loc[df["churn_flag"] == 1, "tenure"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_cust:,}")
    c2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    c3.metric("Monthly Revenue Lost", f"${rev_at_risk:,.0f}")
    c4.metric("Avg Tenure (Churned)", f"{avg_churn_tenure:.1f} mos")

    col_left, col_right = st.columns(2)

    with col_left:
        contract_df = (
            df.groupby("Contract", as_index=False)["churn_flag"]
            .mean()
            .assign(churn_pct=lambda d: d["churn_flag"] * 100)
        )
        fig1 = px.bar(
            contract_df,
            x="Contract",
            y="churn_pct",
            text_auto=".1f",
            title="Churn Rate (%) by Contract Type",
            labels={"churn_pct": "Churn Rate (%)"},
            color="Contract",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        fig2 = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            barmode="overlay",
            nbins=30,
            title="Monthly Charges Distribution: Churned vs. Retained",
            opacity=0.65,
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Live Customer Risk Simulator")
    st.write("Adjust customer attributes below to simulate real-time churn probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (Months)", 0, 72, 6)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0, step=1.0)

    with col2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with col3:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Build full feature row using model defaults + user overrides
    input_data = defaults.copy()
    input_data.update(
        {
            "Contract": contract,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": round(tenure * monthly_charges, 2),
            "InternetService": internet,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaymentMethod": payment,
            "PaperlessBilling": paperless,
            "SeniorCitizen": senior,
        }
    )

    input_df = pd.DataFrame([input_data])
    prob = pipeline.predict_proba(input_df)[0, 1]

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted Churn Probability", f"{prob * 100:.1f}%")
    with r2:
        if prob >= 0.65:
            st.error("High Risk: Offer a 1-year contract discount + bundled Tech Support immediately.")
        elif prob >= 0.35:
            st.warning("Moderate Risk: Trigger proactive check-in or autopay enrollment incentive.")
        else:
            st.success("Low Risk: Customer profile is stable; maintain standard engagement.")
