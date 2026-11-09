"""Streamlit app: credit card fraud detection with XGBoost.

Run locally:  streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
VCOLS = [f"V{i}" for i in range(1, 29)]


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model(os.path.join(HERE, "models", "xgb_fraud.json"))
    scaler = joblib.load(os.path.join(HERE, "models", "amount_scaler.joblib"))
    with open(os.path.join(HERE, "outputs", "results.json")) as f:
        meta = json.load(f)
    return model, scaler, meta


@st.cache_data
def load_samples():
    return pd.read_csv(os.path.join(HERE, "data", "sample_transactions.csv"))


model, scaler, meta = load_artifacts()
samples = load_samples()
THRESHOLD = meta["threshold"]


def fraud_probability(df):
    """df needs columns V1..V28 and Amount (raw dollars). Returns P(fraud) per row."""
    X = df[VCOLS].astype(float).copy()
    amt = pd.DataFrame({"Amount_log": np.log1p(df["Amount"].astype(float).clip(lower=0))})
    X["Amount_log"] = scaler.transform(amt)[:, 0]
    return model.predict_proba(X[meta["features"]])[:, 1]


st.title("Credit Card Fraud Detection")
st.caption("XGBoost classifier trained on the Kaggle credit card fraud dataset. Educational project, not a production fraud system.")

tab1, tab2, tab3 = st.tabs(["Check one transaction", "Score a CSV file", "Model performance"])

# ---------------------------------------------------------------- single transaction
with tab1:
    st.write(
        "The dataset's features **V1-V28** are anonymized PCA components, so they cannot be typed in by hand in a meaningful way. "
        "Load a real example from the held-out test set, then edit any values and see how the prediction changes."
    )
    legit = samples[samples["Class"] == 0].head(5).reset_index(drop=True)
    fraud = samples[samples["Class"] == 1].head(5).reset_index(drop=True)
    options = {"Blank transaction (all zeros)": None}
    options.update({f"Example legitimate transaction {i + 1}": legit.iloc[i] for i in range(len(legit))})
    options.update({f"Example fraudulent transaction {i + 1}": fraud.iloc[i] for i in range(len(fraud))})
    choice = st.selectbox("Start from", list(options))
    st.caption("At the chosen threshold the model catches roughly two thirds of fraud cases, so some fraud examples are "
               "correctly shown as missed. See the Model performance tab for the full numbers.")
    row = options[choice]
    k = choice.replace(" ", "_")  # new widget keys per example, so values reset when the example changes

    amount = st.number_input("Amount (USD)", min_value=0.0, value=float(row["Amount"]) if row is not None else 0.0,
                             step=1.0, key=f"{k}_amount")
    with st.expander("Anonymized features V1-V28", expanded=False):
        cols = st.columns(4)
        vals = {}
        for i, c in enumerate(VCOLS):
            vals[c] = cols[i % 4].number_input(c, value=float(row[c]) if row is not None else 0.0,
                                               format="%.4f", key=f"{k}_{c}")

    if st.button("Predict", type="primary"):
        one = pd.DataFrame([{**vals, "Amount": amount}])
        p = float(fraud_probability(one)[0])
        c1, c2 = st.columns([1, 2])
        c1.metric("Fraud probability", f"{p * 100:.1f}%")
        if p >= THRESHOLD:
            c2.error(f"FRAUD ALERT: probability is above the decision threshold ({THRESHOLD:.2f}). Block and review this transaction.")
        else:
            c2.success(f"Looks legitimate: probability is below the decision threshold ({THRESHOLD:.2f}).")
        st.progress(min(max(p, 0.0), 1.0))
        if row is not None:
            st.caption(f"True label of this example: {'FRAUD' if int(row['Class']) == 1 else 'legitimate'}.")

# ---------------------------------------------------------------- batch scoring
with tab2:
    st.write("Upload a CSV with columns `V1`...`V28` and `Amount` (an optional `Class` column enables an accuracy check).")
    up = st.file_uploader("CSV file", type="csv")
    use_demo = st.checkbox("Use the built-in demo file instead (200 test-set transactions, 20% fraud)", value=up is None)
    data = pd.read_csv(up) if (up is not None and not use_demo) else (samples if use_demo else None)
    if data is not None:
        missing = [c for c in VCOLS + ["Amount"] if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            out = data.copy()
            out["fraud_probability"] = fraud_probability(data)
            out["flagged_as_fraud"] = (out["fraud_probability"] >= THRESHOLD).astype(int)
            a, b = st.columns(2)
            a.metric("Transactions scored", f"{len(out):,}")
            b.metric("Flagged as fraud", f"{int(out['flagged_as_fraud'].sum()):,}")
            if "Class" in out.columns:
                ct = pd.crosstab(out["Class"].map({0: "Actually legitimate", 1: "Actually fraud"}),
                                 out["flagged_as_fraud"].map({0: "Predicted legitimate", 1: "Predicted fraud"}))
                st.write("Confusion matrix on this file")
                st.dataframe(ct)
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50), width="stretch")
            st.download_button("Download all results (CSV)", out.to_csv(index=False).encode(), "fraud_scores.csv", "text/csv")

# ---------------------------------------------------------------- model performance
with tab3:
    st.write(
        f"Evaluated on **{meta['test_rows']:,}** held-out transactions, of which only **{meta['test_fraud']}** are fraud. "
        f"The decision threshold ({THRESHOLD:.2f}) was chosen on a separate validation set."
    )
    mt = pd.DataFrame(meta["metrics"]).set_index("Model").round(3)
    st.dataframe(mt, width="stretch")
    ct = meta["confusion_tuned"]
    st.write(f"At the chosen threshold the model caught **{ct['TP']} of {ct['TP'] + ct['FN']}** frauds, "
             f"with **{ct['FP']}** false alarms among {ct['TN'] + ct['FP']:,} legitimate transactions.")
    left, right = st.columns(2)
    left.image(os.path.join(HERE, "outputs", "02_roc_curves.png"), caption="ROC curves")
    right.image(os.path.join(HERE, "outputs", "04_confusion_matrix.png"), caption="Confusion matrices")
    left.image(os.path.join(HERE, "outputs", "03_precision_recall.png"), caption="Precision-recall curves")
    right.image(os.path.join(HERE, "outputs", "05_feature_importance.png"), caption="Feature importance")
