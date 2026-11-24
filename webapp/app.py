"""
AML risk-scoring demo: Streamlit front end for the project's final
XGBoost model. All model logic lives in ``scoring.py``; this file is UI only.

Run from the project root:
    .venv\\Scripts\\python.exe -m streamlit run webapp/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

WEBAPP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBAPP_DIR))

import scoring  # noqa: E402

SAMPLE_DIR = WEBAPP_DIR / "sample_data"
SAMPLE_FILES = {
    "quick_demo_mixed_20.csv": "20 rows: 10 laundering + 10 normal (HI-Small test period)",
    "small_laundering_examples.csv": "25 known laundering rows (HI-Small test period)",
    "small_not_laundering_examples.csv": "25 known normal rows (HI-Small test period)",
    "medium_mixed_examples.csv": "200 rows: 100 + 100 from HI-Medium, Sep 19+ (never seen by the model)",
}
TABLE_PREVIEW_ROWS = 2_000

FEATURE_DESCRIPTIONS = {
    "amount_paid": "Amount sent",
    "amount_received": "Amount received",
    "transaction_hour": "Hour of day",
    "transaction_weekday": "Day of week (1 = Monday)",
    "transaction_day": "Day of month",
    "transaction_month": "Month",
    "same_bank": "Same bank on both sides",
    "same_account": "Self-transfer",
    "same_currency": "Same currency on both sides",
    "received_amount_nonpositive": "Received amount <= 0",
    "paid_amount_nonpositive": "Paid amount <= 0",
    "sender_metadata_matched": "Sender found in accounts file",
    "receiver_metadata_matched": "Receiver found in accounts file",
    "sender_tx_count_1h": "Sender transactions, last 1h",
    "sender_outgoing_count_24h": "Sender payments made, last 24h",
    "sender_incoming_count_24h": "Sender payments received, last 24h",
    "sender_outgoing_amount_24h": "Sender amount sent, last 24h",
    "sender_incoming_amount_24h": "Sender amount received, last 24h",
    "sender_unique_recipients_24h": "Sender distinct recipients, last 24h",
    "sender_unique_senders_24h": "Sender distinct payers, last 24h",
    "sender_seconds_since_activity": "Seconds since sender's last activity",
    "receiver_tx_count_1h": "Receiver transactions, last 1h",
    "receiver_outgoing_count_24h": "Receiver payments made, last 24h",
    "receiver_incoming_count_24h": "Receiver payments received, last 24h",
    "receiver_outgoing_amount_24h": "Receiver amount sent, last 24h",
    "receiver_incoming_amount_24h": "Receiver amount received, last 24h",
    "receiver_unique_recipients_24h": "Receiver distinct recipients, last 24h",
    "receiver_unique_senders_24h": "Receiver distinct payers, last 24h",
    "receiver_seconds_since_activity": "Seconds since receiver's last activity",
    "sender_historical_fan_out_24h": "Sender fan-out, last 24h",
    "sender_historical_edge_count_24h": "Sender outgoing edges, last 24h",
    "receiver_historical_fan_in_24h": "Receiver fan-in, last 24h",
    "receiver_historical_edge_count_24h": "Receiver incoming edges, last 24h",
    "sender_received_money_24h": "Sender received money, last 24h",
    "rapid_forwarding_indicator": "Rapid forwarding (received then sent on)",
    "payment_format": "Payment format",
    "payment_currency": "Payment currency",
    "receiving_currency": "Receiving currency",
    "sender_entity_type": "Sender entity type",
    "receiver_entity_type": "Receiver entity type",
}

st.set_page_config(page_title="AML Risk Scoring Demo", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_bundle() -> scoring.ModelBundle:
    return scoring.load_bundle()


@st.cache_data(show_spinner=False)
def read_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / name)


@st.cache_data(show_spinner="Scoring transactions...", max_entries=4)
def score_table(data: bytes, name: str):
    """Parse and score a file once; reruns (e.g. budget changes) reuse it."""
    frame = scoring.read_table(data, name)
    prepared = scoring.prepare_batch(frame, BUNDLE)
    scores = scoring.score(BUNDLE, prepared.matrix)
    keep = [c for c in scoring.ID_COLUMNS if c in frame.columns]
    result = frame[keep].copy()
    if prepared.labels is not None:
        result[scoring.LABEL_COLUMN] = prepared.labels
    result["risk_score"] = scores
    return result, prepared.warnings


BUNDLE = get_bundle()
METRICS = BUNDLE.reference["metrics"]
BUDGET = BUNDLE.alert_budget


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------

def display_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "missing"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def show_score(score_value: float) -> None:
    percentile = BUNDLE.percentile(score_value)
    in_top_k = score_value >= BUNDLE.top_k_cutoff
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{score_value:.4f}")
    c2.metric("Higher than", f"{percentile:.2%}", help=(
        "Share of the 762,103 held-out test-period transactions that "
        "received a lower score."))
    c3.metric(f"Top-{BUDGET:,} alert?", "Yes" if in_top_k else "No", help=(
        f"Production policy: investigators review the {BUDGET:,} highest "
        f"scores per period. In the test period that meant a score of at "
        f"least {BUNDLE.top_k_cutoff:.4f}."))
    st.progress(min(max(score_value, 0.0), 1.0))


def explanation_chart(display_row: pd.Series, matrix_row: np.ndarray, top_n: int = 10):
    contrib, bias = scoring.contributions(BUNDLE, matrix_row.reshape(1, -1))
    contrib = contrib[0]
    order = np.argsort(-np.abs(contrib))[:top_n]
    data = pd.DataFrame({
        "feature": [
            f"{FEATURE_DESCRIPTIONS.get(BUNDLE.features[i], BUNDLE.features[i])}"
            f" = {display_value(display_row[BUNDLE.features[i]])}"
            for i in order
        ],
        "contribution": contrib[order],
        "direction": np.where(contrib[order] > 0, "raises risk", "lowers risk"),
    })
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution (log-odds)"),
            y=alt.Y("feature:N", sort=alt.SortField("contribution", order="descending"), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["raises risk", "lowers risk"], range=["#d62728", "#1f77b4"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["feature", alt.Tooltip("contribution:Q", format=".3f")],
        )
        .properties(height=34 * len(data) + 40)
    )
    st.markdown("**Why this score** — the 10 features that moved it most")
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"Starting point (average log-odds) {bias[0]:+.3f}; adding all 40 "
        f"contributions gives {bias[0] + contrib.sum():+.3f}, which is exactly "
        "this transaction's score on the log-odds scale (TreeSHAP)."
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("Anti-Money-Laundering Risk Scoring")
    st.markdown(
        "This demo serves the final model of the IBM AML project: an "
        "**XGBoost classifier over 40 engineered features** that ranks "
        "transactions so investigators can review the riskiest "
        f"**{BUDGET:,}** per period."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", f"{METRICS['pr_auc']:.3f}")
    c2.metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
    c3.metric(f"Precision @ {BUDGET:,}", f"{METRICS['precision_at_budget']:.1%}")
    c4.metric(f"Recall @ {BUDGET:,}", f"{METRICS['recall_at_budget']:.1%}")
    st.caption(
        f"Held-out HI-Small test period: {BUNDLE.reference['test_rows']:,} "
        f"transactions, {BUNDLE.reference['test_positives']:,} labelled "
        f"laundering. Of the top {BUDGET:,} scores, "
        f"{METRICS['true_positives_at_budget']:,} were laundering."
    )

    st.subheader("Pages")
    st.markdown(
        "- **Sample Explorer** — pick a real labelled transaction and see its "
        "score and which features drove it.\n"
        "- **Batch Scoring** — upload a model-ready CSV/Parquet file (or use a "
        "bundled one) and get a ranked alert list.\n"
        "- **Feature Playground** — build a hypothetical transaction and watch "
        "how the score responds."
    )
    st.subheader("Where the 40 features come from")
    st.code(
        "raw transactions -> preprocess_data.py        (clean, time/amount flags)\n"
        "                 -> join_account_metadata.py  (sender/receiver entity type)\n"
        "                 -> add_behavior_features.py  (1h/24h activity, prior rows only)\n"
        "                 -> add_graph_features.py     (fan-in/out, rapid forwarding)\n"
        "                 -> XGBoost risk score",
        language="text",
    )
    st.info(
        "The model needs each account's recent history, so it cannot score a "
        "single raw transaction typed in without context. Scores are rankings "
        "from a heavily class-weighted model, not calibrated probabilities, and "
        "a high score is a reason to investigate, not proof of laundering."
    )


def page_sample_explorer() -> None:
    st.title("Sample Explorer")
    st.write("Real, fully engineered transactions with known labels.")
    source = st.radio(
        "Source",
        ["HI-Small held-out test period", "HI-Medium, Sep 19+ (unseen data)"],
        horizontal=True,
    )
    if source.startswith("HI-Small"):
        pool = pd.concat(
            [read_sample("small_laundering_examples.csv"),
             read_sample("small_not_laundering_examples.csv")],
            ignore_index=True,
        )
    else:
        pool = read_sample("medium_mixed_examples.csv")

    label = st.radio("Show", ["Laundering", "Normal"], horizontal=True)
    pool = pool[pool[scoring.LABEL_COLUMN] == (1 if label == "Laundering" else 0)]
    pool = pool.reset_index(drop=True)

    index = st.selectbox(
        f"Transaction ({len(pool)} available)",
        range(len(pool)),
        format_func=lambda i: (
            f"#{i + 1}  {pool.at[i, 'timestamp']}  "
            f"{pool.at[i, 'amount_paid']:,.2f} {pool.at[i, 'payment_currency']} "
            f"via {pool.at[i, 'payment_format']}"
        ),
    )
    row = pool.iloc[[index]]
    prepared = scoring.prepare_batch(row, BUNDLE)
    score_value = float(scoring.score(BUNDLE, prepared.matrix)[0])

    left, right = st.columns([2, 3])
    with left:
        show_score(score_value)
        st.metric("Dataset label", label)
        st.caption(
            "Ranking-based: 'Top-1,000 alert' is how this transaction would have "
            "fared in the test period's review queue."
        )
    with right:
        explanation_chart(row.iloc[0], prepared.matrix[0])

    with st.expander("All 40 feature values"):
        st.dataframe(
            pd.DataFrame({
                "feature": BUNDLE.features,
                "meaning": [FEATURE_DESCRIPTIONS.get(f, "") for f in BUNDLE.features],
                "value": [display_value(row.iloc[0][f]) for f in BUNDLE.features],
            }),
            hide_index=True,
            width="stretch",
        )


def page_batch_scoring() -> None:
    st.title("Batch Scoring")
    st.write(
        "Score a file that already has the 40 engineered columns (the output "
        "format of the project's feature pipeline). Rows are ranked by risk "
        "score and the top ones become alerts."
    )
    uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    bundled = st.selectbox(
        "...or use a bundled sample file",
        ["(none)", *SAMPLE_FILES],
        format_func=lambda n: n if n == "(none)" else f"{n} — {SAMPLE_FILES[n]}",
        disabled=uploaded is not None,
    )

    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
    elif bundled != "(none)":
        data, name = (SAMPLE_DIR / bundled).read_bytes(), bundled
    else:
        st.info("Upload a file or pick a bundled sample to begin.")
        return

    try:
        results, warnings = score_table(data, name)
    except scoring.BatchValidationError as error:  # unreadable or invalid file
        st.error(f"This file can't be scored. {error}")
        return
    except Exception as error:  # anything unforeseen: report, don't crash
        st.error(f"Unexpected error while scoring `{name}`: {error}")
        return

    for warning in warnings:
        st.warning(warning)

    rows = len(results)
    budget = int(st.number_input(
        "Alert budget (how many top-ranked rows to flag)",
        min_value=1, max_value=rows, value=min(BUDGET, rows), step=1,
    ))
    rank, is_alert = scoring.rank_alerts(results["risk_score"].to_numpy(), budget)
    table = results.assign(alert_rank=rank, is_alert=is_alert).sort_values("alert_rank")

    st.subheader(f"{rows:,} rows scored, top {budget:,} flagged")
    if scoring.LABEL_COLUMN in table.columns:
        m = scoring.alert_metrics(table[scoring.LABEL_COLUMN].to_numpy(), table["is_alert"].to_numpy())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Alerts", f"{m['alerts']:,}")
        c2.metric("Laundering among alerts", f"{m['true_positives']:,}")
        c3.metric("Precision", f"{m['precision']:.1%}")
        c4.metric("Recall", f"{m['recall']:.1%}",
                  help=f"{m['true_positives']:,} of {m['positives']:,} labelled laundering rows flagged")
        st.caption(
            "Shown because the file has an `is_laundering` column. Bundled "
            "samples are far richer in laundering than real traffic (~0.1%), "
            "so these numbers are much higher than real-world precision."
        )

    shown = table.head(TABLE_PREVIEW_ROWS)
    if rows > TABLE_PREVIEW_ROWS:
        st.caption(f"Showing the top {TABLE_PREVIEW_ROWS:,} rows; the download has all {rows:,}.")
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "risk_score", min_value=0.0, max_value=1.0, format="%.4f"),
            "is_alert": st.column_config.CheckboxColumn("is_alert"),
        },
    )
    st.download_button(
        "Download ranked results (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="aml_ranked_alerts.csv",
        mime="text/csv",
    )


PRESETS = {
    "Typical transaction": {},
    "Fan-out with rapid forwarding": {
        "pg_payment_format": "ACH",
        "pg_amount_paid": 9_500.0,
        "pg_sender_incoming_count_24h": 3,
        "pg_sender_unique_senders_24h": 3,
        "pg_sender_outgoing_count_24h": 40,
        "pg_sender_unique_recipients_24h": 40,
        "pg_sender_tx_count_1h": 25,
        "pg_sender_minutes_since": 1.0,
    },
}
MAX_AMOUNT = 1e13  # above every amount in the data; keeps 24h totals finite
COUNT_FIELDS = ("outgoing_count_24h", "unique_recipients_24h",
                "incoming_count_24h", "unique_senders_24h", "tx_count_1h")
SAVED_KEY = "pg_saved"  # survives page switches, unlike widget state


def playground_defaults() -> dict:
    d = BUNDLE.defaults
    return {
        "pg_amount_paid": float(d["amount_paid"]),
        "pg_amount_received": float(d["amount_received"]),
        "pg_payment_format": d["payment_format"],
        "pg_payment_currency": d["payment_currency"],
        "pg_receiving_currency": d["receiving_currency"],
        "pg_sender_entity_type": d["sender_entity_type"],
        "pg_receiver_entity_type": d["receiver_entity_type"],
        "pg_same_bank": bool(d["same_bank"]),
        "pg_same_account": bool(d["same_account"]),
        "pg_hour": int(d["transaction_hour"]),
        **{
            f"pg_{side}_{name}": int(d[f"{side}_{name}"])
            for side in ("sender", "receiver")
            for name in COUNT_FIELDS
        },
        "pg_sender_minutes_since": float(d["sender_seconds_since_activity"]) / 60,
        "pg_receiver_minutes_since": float(d["receiver_seconds_since_activity"]) / 60,
    }


def apply_preset(name: str) -> None:
    values = {**playground_defaults(), **PRESETS[name]}
    st.session_state.update(values)
    st.session_state["pg_received_kept"] = values["pg_amount_received"]


def minutes_bounds(side: str) -> tuple[float, float]:
    """Allowed 'minutes since previous activity' for the current counts."""
    ss = st.session_state
    total = ss[f"pg_{side}_outgoing_count_24h"] + ss[f"pg_{side}_incoming_count_24h"]
    low, high = scoring.seconds_since_range(ss[f"pg_{side}_tx_count_1h"], total)
    return math.ceil(low / 60 * 100) / 100, (high / 60 if high is not None else 1e6)


def sync_playground_state() -> None:
    """Restore values lost on page switches, then make every input
    consistent before any widget is drawn (Streamlit only allows changing
    a widget's value before it is created in a run)."""
    ss = st.session_state
    saved = ss.get(SAVED_KEY, {})
    for key, value in playground_defaults().items():
        if key not in ss:
            ss[key] = saved.get(key, value)

    if ss.pg_same_account:  # one account: the receiver *is* the sender
        ss.pg_same_bank = True
        ss.pg_receiver_entity_type = ss.pg_sender_entity_type
        for name in (*COUNT_FIELDS, "minutes_since"):
            ss[f"pg_receiver_{name}"] = ss[f"pg_sender_{name}"]

    for side in ("sender", "receiver"):
        p = f"pg_{side}_"
        ss[p + "unique_recipients_24h"] = min(ss[p + "unique_recipients_24h"],
                                              ss[p + "outgoing_count_24h"])
        ss[p + "unique_senders_24h"] = min(ss[p + "unique_senders_24h"],
                                           ss[p + "incoming_count_24h"])
        ss[p + "tx_count_1h"] = min(ss[p + "tx_count_1h"],
                                    ss[p + "outgoing_count_24h"] + ss[p + "incoming_count_24h"])
        low, high = minutes_bounds(side)
        ss[p + "minutes_since"] = float(min(max(ss[p + "minutes_since"], low), high))


def page_playground() -> None:
    st.title("Feature Playground")
    st.write(
        "Build a hypothetical transaction and watch the score respond. Inputs "
        "start at training-data medians. Fields the pipeline derives (rapid "
        "forwarding, fan-in/out, 24h amounts, weekday) are computed for you, "
        "and inputs are kept within the relationships that always hold in the "
        "real feature data."
    )
    sync_playground_state()
    ss = st.session_state

    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS, strict=True):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    categories = {
        c: [v for v in BUNDLE.mappings[c] if v != scoring.MISSING_CATEGORY]
        for c in BUNDLE.categorical_features
    }
    self_transfer = ss.pg_same_account
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**This transaction**")
        st.number_input("Amount paid", 0.0, MAX_AMOUNT, step=100.0, key="pg_amount_paid")
        st.selectbox("Payment format", categories["payment_format"], key="pg_payment_format")
        st.selectbox("Payment currency", categories["payment_currency"], key="pg_payment_currency")
        st.selectbox("Receiving currency", categories["receiving_currency"], key="pg_receiving_currency")
        currencies_differ = ss.pg_payment_currency != ss.pg_receiving_currency
        if currencies_differ:
            # Shown only when currencies differ. Streamlit forgets the value of
            # a hidden widget, so keep a copy and restore it on reappearance.
            if not ss.get("pg_received_shown", False):
                ss.pg_amount_received = ss.get("pg_received_kept", ss.pg_amount_received)
            st.number_input("Amount received (in receiving currency)", 0.0, MAX_AMOUNT,
                            step=100.0, key="pg_amount_received")
            ss["pg_received_kept"] = ss.pg_amount_received
        ss["pg_received_shown"] = currencies_differ
        st.slider("Hour of day", 0, 23, key="pg_hour")
        st.checkbox("Self-transfer (same account)", key="pg_same_account")
        st.checkbox("Same bank", key="pg_same_bank", disabled=self_transfer)
        st.selectbox("Sender entity type", categories["sender_entity_type"], key="pg_sender_entity_type")
        st.selectbox("Receiver entity type", categories["receiver_entity_type"],
                     key="pg_receiver_entity_type", disabled=self_transfer)

    def history(side: str, title: str, disabled: bool = False) -> None:
        st.markdown(f"**{title}, previous 24 hours**")
        if disabled:
            st.caption("Self-transfer: same account as the sender, so same history.")
        p = f"pg_{side}_"
        out, inc = ss[p + "outgoing_count_24h"], ss[p + "incoming_count_24h"]
        low, high = minutes_bounds(side)
        st.number_input("Payments made", 0, 5_000, step=1, key=p + "outgoing_count_24h",
                        disabled=disabled)
        st.number_input("...to distinct accounts", 0, out, step=1,
                        key=p + "unique_recipients_24h", disabled=disabled)
        st.number_input("Payments received", 0, 5_000, step=1, key=p + "incoming_count_24h",
                        disabled=disabled)
        st.number_input("...from distinct accounts", 0, inc, step=1,
                        key=p + "unique_senders_24h", disabled=disabled)
        st.number_input("Transactions in the last hour", 0, out + inc, step=1,
                        key=p + "tx_count_1h", disabled=disabled)
        st.number_input("Minutes since previous activity", low, high, step=1.0,
                        key=p + "minutes_since", disabled=disabled,
                        help="Limited by the counts above: activity in the last hour "
                             "means at most 60 minutes; none in 24h means over 1,440.")

    with c2:
        history("sender", "Sender")
    with c3:
        history("receiver", "Receiver", disabled=self_transfer)

    ss[SAVED_KEY] = {key: ss[key] for key in playground_defaults()}

    overrides = {
        "amount_paid": ss.pg_amount_paid,
        "amount_received": ss.pg_amount_received,
        "payment_format": ss.pg_payment_format,
        "payment_currency": ss.pg_payment_currency,
        "receiving_currency": ss.pg_receiving_currency,
        "sender_entity_type": ss.pg_sender_entity_type,
        "receiver_entity_type": ss.pg_receiver_entity_type,
        "same_account": int(ss.pg_same_account),
        "same_bank": int(ss.pg_same_bank),
        "transaction_hour": ss.pg_hour,
    }
    for side in ("sender", "receiver"):
        for name in COUNT_FIELDS:
            overrides[f"{side}_{name}"] = ss[f"pg_{side}_{name}"]
        overrides[f"{side}_seconds_since_activity"] = ss[f"pg_{side}_minutes_since"] * 60

    row = scoring.build_playground_row(BUNDLE, overrides)
    prepared = scoring.prepare_batch(row, BUNDLE)
    score_value = float(scoring.score(BUNDLE, prepared.matrix)[0])

    st.divider()
    derived = row.iloc[0]
    st.markdown(
        f"Derived by the pipeline rules: rapid forwarding = "
        f"**{'yes' if derived['rapid_forwarding_indicator'] else 'no'}**, "
        f"sender received money in 24h = "
        f"**{'yes' if derived['sender_received_money_24h'] else 'no'}**, "
        f"same currency = **{'yes' if derived['same_currency'] else 'no'}**."
    )
    left, right = st.columns([2, 3])
    with left:
        show_score(score_value)
        st.caption(
            "A what-if sandbox. Features not shown (day of month, account "
            "metadata) stay at training medians, and 24h amounts assume earlier "
            "transactions were the same size as this one."
        )
    with right:
        explanation_chart(row.iloc[0], prepared.matrix[0])


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Sample Explorer": page_sample_explorer,
    "Batch Scoring": page_batch_scoring,
    "Feature Playground": page_playground,
}

st.sidebar.title("🔎 AML Risk Scoring")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.divider()
st.sidebar.caption(
    f"XGBoost · 40 features · top-{BUDGET:,} alert policy\n\n"
    f"Test set: {METRICS['precision_at_budget']:.1%} precision, "
    f"{METRICS['recall_at_budget']:.1%} recall at {BUDGET:,} alerts.\n\n"
    "Scores are investigation priorities, not proof of wrongdoing."
)
PAGES[choice]()
