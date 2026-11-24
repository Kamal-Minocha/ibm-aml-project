"""
IBM AML Risk Scoring - interactive demo
=========================================

A Streamlit front end for the trained XGBoost money-laundering risk model.

Important honesty note for anyone reading this code:
The model was trained on 40 engineered features that include each account's
recent transaction history (how many recipients it paid in the last 24h, how
quickly it forwarded received money, etc.). A single raw transaction typed
into a form is NOT enough information to reproduce those features correctly.
This app therefore offers three honest ways to interact with the model:

  1. Sample Explorer   - real, fully engineered transactions (ground truth
                          known) that you pick from a dropdown.
  2. Batch Scoring      - upload a CSV/Parquet file that already has the
                          40 engineered columns (same format predict_batch.py
                          expects) and get ranked alerts back.
  3. Feature Playground - a "what-if" sandbox where a fixed baseline
                          transaction's interpretable fields can be dragged,
                          to show the model's sensitivity to each feature.
                          This is for teaching intuition, not a real scoring
                          tool for a brand-new transaction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_DIR = BASE_DIR / "sample_data"

MODEL_PATH = MODEL_DIR / "xgboost_graph_model.json"
CONFIG_PATH = MODEL_DIR / "final_xgboost_config.json"
MAPPINGS_PATH = MODEL_DIR / "category_mappings.json"
STATS_PATH = MODEL_DIR / "feature_stats.json"

OPTIONAL_COLUMNS = [
    "transaction_id",
    "timestamp",
    "from_account",
    "to_account",
    "is_laundering",
]

st.set_page_config(
    page_title="AML Risk Scoring Demo",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_mappings() -> dict:
    return json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_stats() -> dict:
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / name)


CONFIG = load_config()
FEATURES: list[str] = CONFIG["features"]
CATEGORICAL_FEATURES: list[str] = CONFIG["categorical_features"]
NUMERIC_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
MAPPINGS = load_mappings()
FEATURE_STATS = load_feature_stats()
MODEL = load_model()
BOOSTER = MODEL.get_booster()

FEATURE_DESCRIPTIONS = {
    "amount_paid": "Amount sent by the sender",
    "amount_received": "Amount received by the receiver (currency may differ)",
    "transaction_hour": "Hour of day (0-23) the transaction occurred",
    "transaction_weekday": "Day of week (1=Mon .. 7=Sun)",
    "transaction_day": "Day of the month",
    "transaction_month": "Month (always September in this dataset)",
    "same_bank": "Sender and receiver use the same bank",
    "same_account": "Sender and receiver are the same account",
    "same_currency": "Payment and receiving currency match",
    "received_amount_nonpositive": "Received amount was zero or negative",
    "paid_amount_nonpositive": "Paid amount was zero or negative",
    "sender_metadata_matched": "Sender account was found in the accounts file",
    "receiver_metadata_matched": "Receiver account was found in the accounts file",
    "sender_tx_count_1h": "Sender's transactions in the previous hour",
    "sender_outgoing_count_24h": "Sender's outgoing transactions in the previous 24h",
    "sender_incoming_count_24h": "Sender's incoming transactions in the previous 24h",
    "sender_outgoing_amount_24h": "Total sent by sender in the previous 24h",
    "sender_incoming_amount_24h": "Total received by sender in the previous 24h",
    "sender_unique_recipients_24h": "Distinct accounts the sender paid in the previous 24h",
    "sender_unique_senders_24h": "Distinct accounts that paid the sender in the previous 24h",
    "sender_seconds_since_activity": "Seconds since the sender's previous activity",
    "receiver_tx_count_1h": "Receiver's transactions in the previous hour",
    "receiver_outgoing_count_24h": "Receiver's outgoing transactions in the previous 24h",
    "receiver_incoming_count_24h": "Receiver's incoming transactions in the previous 24h",
    "receiver_outgoing_amount_24h": "Total sent by receiver in the previous 24h",
    "receiver_incoming_amount_24h": "Total received by receiver in the previous 24h",
    "receiver_unique_recipients_24h": "Distinct accounts the receiver paid in the previous 24h",
    "receiver_unique_senders_24h": "Distinct accounts that paid the receiver in the previous 24h",
    "receiver_seconds_since_activity": "Seconds since the receiver's previous activity",
    "sender_historical_fan_out_24h": "Distinct accounts the sender paid, previous 24h (network view)",
    "sender_historical_edge_count_24h": "Sender's previous outgoing transaction count (network view)",
    "receiver_historical_fan_in_24h": "Distinct accounts that paid the receiver, previous 24h",
    "receiver_historical_edge_count_24h": "Receiver's previous incoming transaction count",
    "sender_received_money_24h": "Sender had received money in the previous 24h",
    "rapid_forwarding_indicator": "Sender recently received money and is now forwarding it onward",
    "payment_format": "Payment rail (ACH, Wire, Cheque, Bitcoin, ...)",
    "payment_currency": "Currency the sender paid in",
    "receiving_currency": "Currency the receiver received in",
    "sender_entity_type": "Sender's business/entity category",
    "receiver_entity_type": "Receiver's business/entity category",
}

ALERT_BUDGET = CONFIG.get("alert_budget", 1000)
VALIDATION_THRESHOLD = CONFIG.get("threshold", 0.979)


# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------


def encode_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact training-time category -> integer code mapping."""
    frame = frame.copy()
    for column in CATEGORICAL_FEATURES:
        mapping = MAPPINGS[column]
        frame[column] = (
            frame[column]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(np.int32)
        )
    return frame


def score_frame(frame: pd.DataFrame) -> np.ndarray:
    """Return laundering probabilities for a model-ready dataframe."""
    encoded = encode_categoricals(frame)[FEATURES].to_numpy()
    return MODEL.predict_proba(encoded)[:, 1]


def shap_for_row(row: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Per-feature contribution to this row's prediction.

    Uses XGBoost's own built-in TreeSHAP implementation
    (``pred_contribs=True``), which produces the same values as the
    separate ``shap`` library for tree models, with no extra dependency.
    """
    encoded = encode_categoricals(row)[FEATURES].to_numpy()
    dmatrix = xgb.DMatrix(encoded, feature_names=FEATURES)
    contributions = BOOSTER.predict(dmatrix, pred_contribs=True)[0]
    feature_contributions = contributions[:-1]
    base_value = float(contributions[-1])
    return feature_contributions, base_value


def risk_badge(probability: float) -> str:
    if probability >= 0.90:
        return "🔴 Very high risk"
    if probability >= 0.5:
        return "🟠 High risk"
    if probability >= 0.1:
        return "🟡 Elevated risk"
    return "🟢 Low risk"


def plot_shap_bar(shap_values: np.ndarray, row_values: pd.Series, top_n: int = 10):
    order = np.argsort(-np.abs(shap_values))[:top_n]
    labels = []
    values = []
    colors = []
    for index in order[::-1]:
        feature_name = FEATURES[index]
        value = shap_values[index]
        raw_value = row_values[feature_name]
        labels.append(f"{feature_name} = {raw_value}")
        values.append(value)
        colors.append("#d62728" if value > 0 else "#1f77b4")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="What pushed this prediction up (red) or down (blue)",
        xaxis_title="Contribution to risk score (log-odds, TreeSHAP)",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def probability_gauge(probability: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Predicted laundering probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#333333"},
                "steps": [
                    {"range": [0, 10], "color": "#d4edda"},
                    {"range": [10, 50], "color": "#fff3cd"},
                    {"range": [50, 90], "color": "#ffe0b2"},
                    {"range": [90, 100], "color": "#f8d7da"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🔎 AML Risk Scoring")
st.sidebar.caption(
    "Interactive demo of the IBM Anti-Money-Laundering project's "
    "final XGBoost model."
)
page = st.sidebar.radio(
    "Choose a mode",
    [
        "🏠 Overview",
        "🧾 Sample Explorer",
        "📂 Batch Scoring",
        "🎛️ Feature Playground",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** XGBoost, 40 engineered features\n\n"
    f"**Alert policy:** top {ALERT_BUDGET:,} ranked transactions\n\n"
    "**Test-set result:** 63.5% precision / 40.7% recall at that budget"
)
st.sidebar.markdown(
    "⚠️ Output is a **risk score for investigation**, "
    "not proof of wrongdoing."
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "🏠 Overview":
    st.title("Anti-Money-Laundering Risk Scoring — Interactive Demo")
    st.markdown(
        """
This app demonstrates the final model trained in the **IBM AML project**:
an **XGBoost classifier** that scores transactions in the IBM synthetic
anti-money-laundering dataset by laundering probability, built on top of:

- basic transaction fields (amount, currency, payment format, timing)
- static sender/receiver account metadata
- **behavioral features**: how active each account was in the previous 1h/24h
- **network features**: fan-in/fan-out counts and rapid-forwarding behaviour

Because the model depends on each account's recent history, it cannot score
a brand-new transaction typed in with no context. This demo works around
that honestly with three modes:
"""
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🧾 Sample Explorer")
        st.write(
            "Pick a real transaction (with known ground truth) from the "
            "held-out test period or from the unseen Medium dataset, and "
            "see the model's score and explanation for it."
        )
    with col2:
        st.subheader("📂 Batch Scoring")
        st.write(
            "Upload a CSV/Parquet file of already-engineered transactions "
            "(same format as the project's `predict_batch.py`) and get "
            "ranked, downloadable alerts."
        )
    with col3:
        st.subheader("🎛️ Feature Playground")
        st.write(
            "Start from a typical transaction and drag sliders for the "
            "interpretable features to see how the model's risk score "
            "reacts — a sandbox for intuition, not real scoring."
        )

    st.markdown("---")
    st.subheader("Pipeline this model came from")
    st.code(
        "HI-Small_Trans.csv\n"
        "  -> preprocess_data.py           (clean + basic features)\n"
        "  -> join_account_metadata.py     (sender/receiver account info)\n"
        "  -> add_behavior_features.py     (rolling 1h/24h activity)\n"
        "  -> add_graph_features.py        (fan-in/fan-out, rapid forwarding)\n"
        "  -> train_xgboost.py             (40-feature XGBoost model)\n"
        "  -> finalize_xgboost.py          (top-1,000-alert policy)",
        language="text",
    )

    st.subheader("Headline results (held-out test set, 762,103 transactions)")
    metrics = {
        "Metric": ["PR-AUC", "ROC-AUC", "Precision @ 1,000 alerts", "Recall @ 1,000 alerts"],
        "Value": ["0.468", "0.983", "63.5%", "40.7%"],
    }
    st.table(pd.DataFrame(metrics))
    st.caption(
        "Of every 1,000 transactions the model flags, about 635 are "
        "genuinely laundering according to the dataset's labels, and the "
        "model finds about 41% of all laundering transactions in that "
        "test period."
    )

# ---------------------------------------------------------------------------
# Page: Sample Explorer
# ---------------------------------------------------------------------------

elif page == "🧾 Sample Explorer":
    st.title("🧾 Sample Explorer")
    st.write(
        "These are **real, fully engineered transactions** pulled from the "
        "project's held-out data, with the ground-truth label known. Pick "
        "one to see what the model predicts and why."
    )

    dataset_choice = st.selectbox(
        "Transaction source",
        [
            "Small dataset — held-out test period (model's own test set)",
            "Medium dataset — unseen data from Sep 19 onward (generalisation check)",
        ],
    )

    if dataset_choice.startswith("Small"):
        laundering_df = load_sample("small_laundering_examples.csv")
        normal_df = load_sample("small_not_laundering_examples.csv")
    else:
        combined = load_sample("medium_mixed_examples.csv")
        laundering_df = combined[combined["is_laundering"] == 1].reset_index(drop=True)
        normal_df = combined[combined["is_laundering"] == 0].reset_index(drop=True)

    label_choice = st.radio(
        "Ground-truth label to browse",
        ["Known laundering", "Known normal"],
        horizontal=True,
    )
    pool = laundering_df if label_choice == "Known laundering" else normal_df

    def row_label(i: int) -> str:
        r = pool.iloc[i]
        return (
            f"#{i+1} — {r.get('timestamp', '')} — "
            f"paid {r['amount_paid']:,.2f} {r.get('payment_currency', '')} "
            f"via {r.get('payment_format', '')}"
        )

    row_index = st.selectbox(
        f"Pick a transaction ({len(pool)} available)",
        range(len(pool)),
        format_func=row_label,
    )
    row = pool.iloc[[row_index]]
    probability = float(score_frame(row)[0])
    actual_label = int(row.iloc[0]["is_laundering"])

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(probability_gauge(probability), width='stretch')
        st.metric("Model's risk badge", risk_badge(probability))
        actual_text = "Laundering" if actual_label == 1 else "Normal"
        match = (probability >= 0.5) == bool(actual_label)
        st.metric(
            "Actual dataset label",
            actual_text,
            delta="model agrees" if match else "model disagrees",
            delta_color="normal" if match else "inverse",
        )
        st.caption(
            f"Reference: the validation-selected threshold for the "
            f"production top-{ALERT_BUDGET:,} policy was "
            f"{VALIDATION_THRESHOLD:.4f}. In production the decision is "
            "made by rank among all scored transactions, not a fixed cutoff."
        )

    with right:
        shap_values, base_value = shap_for_row(row)
        st.plotly_chart(
            plot_shap_bar(shap_values, row.iloc[0]), width='stretch'
        )

    with st.expander("See all 40 feature values for this transaction"):
        display_df = row[OPTIONAL_COLUMNS[1:-1] + FEATURES].T.astype(str)
        display_df.columns = ["value"]
        st.dataframe(display_df, width='stretch')

# ---------------------------------------------------------------------------
# Page: Batch Scoring
# ---------------------------------------------------------------------------

elif page == "📂 Batch Scoring":
    st.title("📂 Batch Scoring")
    st.write(
        "Upload a CSV or Parquet file that already contains the 40 "
        "engineered feature columns (the same format the project's "
        "`predict_batch.py` expects). The app will score every row, rank "
        "them by laundering probability, and let you pick an alert budget."
    )
    st.info(
        "Don't have a file handy? Use one of the ready-made files in "
        "`webapp/sample_data/` from the repository, or download a worked "
        "example below."
    )

    example_choice = st.selectbox(
        "Or load a bundled example instead of uploading",
        [
            "— none, I will upload my own file —",
            "small_laundering_examples.csv",
            "small_not_laundering_examples.csv",
            "medium_mixed_examples.csv",
        ],
    )

    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])

    frame = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".parquet"):
            frame = pd.read_parquet(uploaded)
        else:
            frame = pd.read_csv(uploaded)
    elif example_choice != "— none, I will upload my own file —":
        frame = load_sample(example_choice)

    if frame is not None:
        missing = [c for c in FEATURES if c not in frame.columns]
        if missing:
            st.error(
                "This file is not model-ready. It is missing the following "
                "engineered columns, so it cannot be scored directly:\n\n"
                + ", ".join(missing)
                + "\n\nRun the project's preprocessing, account-join, "
                "behavior, and graph-feature stages first, or use one of "
                "the bundled sample files."
            )
        else:
            probabilities = score_frame(frame)
            ranking = (-probabilities).argsort().argsort() + 1  # 1 = highest risk

            max_budget = len(frame)
            default_budget = min(ALERT_BUDGET, max_budget)
            budget = st.slider(
                "Alert budget (how many top-ranked transactions to flag)",
                min_value=1,
                max_value=max_budget,
                value=default_budget,
            )

            results = frame.copy()
            results["laundering_probability"] = probabilities
            results["alert_rank"] = ranking
            results["is_alert"] = results["alert_rank"] <= budget
            results = results.sort_values(
                "laundering_probability", ascending=False
            ).reset_index(drop=True)

            cols_to_show = [
                c
                for c in OPTIONAL_COLUMNS
                if c in results.columns and c != "is_laundering"
            ] + ["laundering_probability", "alert_rank", "is_alert"]

            st.subheader(f"Ranked results — top {budget:,} flagged as alerts")

            if "is_laundering" in results.columns:
                alerts = results[results["is_alert"]]
                true_positives = int((alerts["is_laundering"] == 1).sum())
                false_positives = int((alerts["is_laundering"] == 0).sum())
                total_positive = int((results["is_laundering"] == 1).sum())
                precision = true_positives / max(len(alerts), 1)
                recall = true_positives / max(total_positive, 1)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Alerts", f"{len(alerts):,}")
                m2.metric("True positives", f"{true_positives:,}")
                m3.metric("Precision", f"{precision:.1%}")
                m4.metric("Recall", f"{recall:.1%}")
                st.caption(
                    "Precision/recall are only computable here because this "
                    "file has known `is_laundering` labels. Real unseen "
                    "transactions would not have that column."
                )
                cols_to_show.insert(-3, "is_laundering")

            st.dataframe(
                results[cols_to_show].style.apply(
                    lambda s: [
                        "background-color: #f8d7da" if alert else ""
                        for alert in results["is_alert"]
                    ],
                    axis=0,
                    subset=["is_alert"],
                ),
                width='stretch',
                height=500,
            )

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download full ranked results as CSV",
                data=csv_bytes,
                file_name="aml_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Page: Feature Playground
# ---------------------------------------------------------------------------

elif page == "🎛️ Feature Playground":
    st.title("🎛️ Feature Playground")
    st.write(
        "Start from a typical transaction and change a few interpretable "
        "features to see how the model's risk score reacts. Every other "
        "feature stays fixed at its training-data median or most common "
        "value. **This is a teaching sandbox, not a scoring tool for a "
        "real new transaction** — the model was trained on 40 features "
        "together, and this view only exposes the easiest ones to reason "
        "about."
    )

    baseline = {}
    for feature in NUMERIC_FEATURES:
        baseline[feature] = FEATURE_STATS.get(feature, {}).get("median", 0.0)
    baseline["payment_format"] = "ACH"
    baseline["payment_currency"] = "US Dollar"
    baseline["receiving_currency"] = "US Dollar"
    baseline["sender_entity_type"] = "Corporation"
    baseline["receiver_entity_type"] = "Corporation"

    st.subheader("Adjustable fields")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Transaction**")
        amount = st.slider(
            "Amount paid ($)", 0.0, 50000.0, float(min(baseline["amount_paid"] * 5, 5000)), step=50.0
        )
        baseline["amount_paid"] = amount
        baseline["amount_received"] = amount
        baseline["payment_format"] = st.selectbox(
            "Payment format", list(MAPPINGS["payment_format"].keys()), index=0
        )
        baseline["same_bank"] = int(
            st.checkbox("Same bank for sender & receiver", value=False)
        )
        baseline["same_currency"] = int(
            st.checkbox("Same currency on both sides", value=True)
        )
        baseline["transaction_hour"] = st.slider("Hour of day", 0, 23, 10)

    with c2:
        st.markdown("**Sender's recent behaviour**")
        baseline["sender_tx_count_1h"] = st.slider(
            "Sender's transactions in the last hour", 0, 50, 1
        )
        baseline["sender_outgoing_count_24h"] = st.slider(
            "Sender's outgoing transactions, last 24h", 0, 200, 3
        )
        baseline["sender_unique_recipients_24h"] = st.slider(
            "Distinct recipients the sender paid, last 24h", 0, 100, 1
        )
        baseline["sender_historical_fan_out_24h"] = baseline[
            "sender_unique_recipients_24h"
        ]
        baseline["sender_received_money_24h"] = int(
            st.checkbox("Sender received money in the last 24h", value=False)
        )
        baseline["rapid_forwarding_indicator"] = int(
            st.checkbox(
                "Rapid forwarding: sender just received money and is sending it on",
                value=False,
            )
        )

    with c3:
        st.markdown("**Receiver's recent behaviour**")
        baseline["receiver_tx_count_1h"] = st.slider(
            "Receiver's transactions in the last hour", 0, 50, 1
        )
        baseline["receiver_unique_senders_24h"] = st.slider(
            "Distinct senders who paid the receiver, last 24h", 0, 100, 1
        )
        baseline["receiver_historical_fan_in_24h"] = baseline[
            "receiver_unique_senders_24h"
        ]
        baseline["receiver_entity_type"] = st.selectbox(
            "Receiver entity type",
            list(MAPPINGS["sender_entity_type"].keys()),
            index=0,
        )

    row = pd.DataFrame([baseline])[FEATURES]
    probability = float(score_frame(row)[0])

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(probability_gauge(probability), width='stretch')
        st.metric("Risk badge", risk_badge(probability))
    with right:
        shap_values, base_value = shap_for_row(row)
        st.plotly_chart(
            plot_shap_bar(shap_values, row.iloc[0]), width='stretch'
        )

    st.caption(
        "Try this: set 'rapid forwarding' on, raise the sender's distinct "
        "recipients and the receiver's distinct senders, and watch the "
        "risk score climb — that combination mimics a fan-out/fan-in "
        "laundering pattern."
    )
