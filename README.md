# ChurnLens: End-to-End Customer Churn Analytics & Risk Predictor

An end-to-end analytics and machine learning pipeline that identifies telecom churn drivers, quantifies monthly revenue at risk via SQL, and serves a live "What-If" retention simulator in Streamlit.

## Architecture
`Raw CSV` → `SQLite ETL (src/etl_and_sql.py)` → `Business SQL (sql/query.sql)` → `Scikit-learn Pipeline (src/trainmodel.py)` → `Streamlit Dashboard (app.py)`

## Key Business Findings
- **Revenue Exposure:** ~26.5% overall churn rate, representing **$139K+ in monthly recurring revenue lost**.
- **Contract Vulnerability:** Month-to-month customers churn at **42.7%**, compared to **2.8%** for two-year contracts.
- **Early-Tenure Risk:** Customers in their first 12 months account for the highest churn concentration, especially those on Fiber Optic plans paying via Electronic Check.

## Model Performance
- Evaluated **Logistic Regression** vs. **Random Forest** using `class_weight="balanced"` to address the 73/27 class imbalance.
- Prioritized **ROC-AUC (~0.84)** and **Churn Recall (~0.79)** over raw accuracy so retention teams catch high-risk accounts instead of missing them.

## Dashboard Preview
![Executive KPIs](assets/dashboard_kpis.png)
![What-If Risk Simulator](assets/risk_simulator.png)

## How to Run Locally
```bash
# 1. Install dependencies
uv sync

# 2. Run ETL & SQL business queries
uv run python src/etl_and_sql.py

# 3. Train and serialize the ML pipeline
uv run python src/trainmodel.py

# 4. Launch the Streamlit app
uv run streamlit run app.py
