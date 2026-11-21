import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = Path("data/churn.db")
MODEL_PATH = Path("models/churn_pipeline.pkl")


def train():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM customers", conn)

    X = df.drop(columns=["customerID", "Churn", "churn_flag"])
    y = df["churn_flag"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Stratify preserves the exact churn ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ]
    )

    candidates = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
    }

    best_name, best_pipeline, best_auc = None, None, -1.0

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        rec = recall_score(y_test, y_pred)

        print(f"=== {name} ===")
        print(f"ROC-AUC: {auc:.4f} | Churn Recall: {rec:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        if auc > best_auc:
            best_auc, best_name, best_pipeline = auc, name, pipe

    # Extract Top 5 Churn Drivers from the winning model
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    clf = best_pipeline.named_steps["classifier"]

    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = clf.feature_importances_

    top_drivers = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(5)
    )
    print(f"Winner: {best_name} (ROC-AUC: {best_auc:.4f})")
    print("\nTop 5 Churn Drivers:")
    print(top_drivers.to_string())

    # Save pipeline + default baseline values for the Streamlit simulator
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "roc_auc": best_auc,
        "defaults": X.mode().iloc[0].to_dict(),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[Saved] {MODEL_PATH}")


if __name__ == "__main__":
    train()
