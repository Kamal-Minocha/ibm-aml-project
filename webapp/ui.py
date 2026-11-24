"""
Shared building blocks for the demo's pages: cached data access, colours,
page chrome and the charts. Model logic stays in ``scoring.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
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

# Chart colours per theme. Diverging pair (raises/lowers risk) and the two
# label colours were run through a colour-blind separation and contrast check
# for each mode; text stays in ink colours, never series colours.
PALETTE = {
    "light": {"raises": "#e34948", "lowers": "#2a78d6", "normal": "#2a78d6",
              "laundering": "#eb6834", "ink": "#0b0b0b", "muted": "#898781",
              "track": "#e1e0d9"},
    "dark": {"raises": "#e66767", "lowers": "#3987e5", "normal": "#3987e5",
             "laundering": "#d95926", "ink": "#f0efec", "muted": "#898781",
             "track": "#383835"},
}

CSS = """
<style>
.aml-hero {
  padding: 1.75rem 2rem 1.5rem;
  border-radius: 1rem;
  border: 1px solid rgba(127, 127, 127, 0.18);
  background: linear-gradient(135deg, rgba(42, 120, 214, 0.16) 0%,
              rgba(42, 120, 214, 0.04) 55%, rgba(227, 73, 72, 0.08) 100%);
  margin-bottom: 0.75rem;
}
.aml-eyebrow {
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; opacity: 0.65; margin: 0 0 0.35rem;
}
.aml-hero h1 { font-size: 2.15rem; line-height: 1.15; margin: 0; padding: 0; }
.aml-hero p { font-size: 1.05rem; opacity: 0.82; margin: 0.6rem 0 0; max-width: 54rem; }
.aml-subtitle { font-size: 1.02rem; opacity: 0.75; margin: -0.6rem 0 1.1rem; max-width: 60rem; }
.aml-step { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
            text-transform: uppercase; opacity: 0.6; margin-bottom: 0.15rem; }
.aml-step-title { font-weight: 650; margin-bottom: 0.25rem; }
.aml-step-body { font-size: 0.9rem; opacity: 0.8; }
</style>
"""


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
    bundle = get_bundle()
    frame = scoring.read_table(data, name)
    prepared = scoring.prepare_batch(frame, bundle)
    scores = scoring.score(bundle, prepared.matrix)
    keep = [c for c in scoring.ID_COLUMNS if c in frame.columns]
    result = frame[keep].copy()
    if prepared.labels is not None:
        result[scoring.LABEL_COLUMN] = prepared.labels
    result["risk_score"] = scores
    return result, prepared.warnings


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

def colors() -> dict[str, str]:
    mode = getattr(st.context.theme, "type", None) or "light"
    return PALETTE.get(mode, PALETTE["light"])


def inject_css() -> None:
    st.html(CSS)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.html(f'<p class="aml-subtitle">{subtitle}</p>')


def footer() -> None:
    bundle = get_bundle()
    metrics = bundle.reference["metrics"]
    st.divider()
    st.caption(
        f"XGBoost model · 40 engineered features · top-{bundle.alert_budget:,} "
        f"alert policy · test set: {metrics['precision_at_budget']:.1%} precision, "
        f"{metrics['recall_at_budget']:.1%} recall at {bundle.alert_budget:,} alerts. "
        "Scores are investigation priorities, not proof of wrongdoing."
    )


def display_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "missing"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


# ---------------------------------------------------------------------------
# Score panel and charts
# ---------------------------------------------------------------------------

def score_panel(score_value: float) -> None:
    """Score, its rank in the test period, and the alert decision."""
    bundle = get_bundle()
    budget = bundle.alert_budget
    cutoff = bundle.top_k_cutoff
    percentile = bundle.percentile(score_value)
    in_top_k = score_value >= cutoff

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{score_value:.4f}", border=True, help=(
        "The model's output between 0 and 1. It ranks transactions; it is not "
        "a calibrated probability."))
    c2.metric("Higher than", f"{percentile:.2%}", border=True, help=(
        f"Share of the {bundle.reference['test_rows']:,} held-out test-period "
        "transactions that received a lower score."))
    c3.metric(f"Top-{budget:,} alert?", "Yes" if in_top_k else "No", border=True, help=(
        f"Investigators review the {budget:,} highest scores per period. In the "
        f"test period that meant a score of at least {cutoff:.4f}."))

    if in_top_k:
        st.badge("Would be flagged for investigation", icon=":material/flag:", color="red")
    else:
        st.badge("Below the alert cut-off", icon=":material/remove_done:", color="gray")
    score_bullet(score_value, cutoff)


def score_bullet(score_value: float, cutoff: float) -> None:
    """Horizontal bar: the score against the top-1,000 cut-off on a 0-1 scale."""
    c = colors()
    axis = alt.Axis(title=None, values=[0, 0.25, 0.5, 0.75, 1], format=".2f",
                    grid=False, domain=False, ticks=False, labelColor=c["muted"])
    scale = alt.Scale(domain=[0, 1], nice=False)
    track = alt.Chart(pd.DataFrame({"a": [0.0], "b": [1.0]})).mark_bar(
        height=12, cornerRadius=6, color=c["track"]
    ).encode(x=alt.X("a:Q", scale=scale, axis=axis), x2="b:Q")
    bar = alt.Chart(pd.DataFrame({"a": [0.0], "score": [score_value]})).mark_bar(
        height=12, cornerRadius=6, color=c["lowers"]
    ).encode(x="a:Q", x2="score:Q",
             tooltip=[alt.Tooltip("score:Q", title="Risk score", format=".4f")])
    rule = alt.Chart(pd.DataFrame({"cutoff": [cutoff]})).mark_tick(
        thickness=3, size=24, color=c["ink"]
    ).encode(x="cutoff:Q",
             tooltip=[alt.Tooltip("cutoff:Q", title="Top-1,000 cut-off", format=".4f")])
    st.altair_chart(
        (track + bar + rule).properties(height=46).configure_view(stroke=None),
        width="stretch",
    )
    st.caption(f"Bar: this score. Tick: the top-1,000 alert cut-off ({cutoff:.4f}).")


def explanation_chart(display_row: pd.Series, matrix_row: np.ndarray, top_n: int = 10) -> None:
    """Diverging bars: the features that moved this score most."""
    bundle = get_bundle()
    c = colors()
    contrib, bias = scoring.contributions(bundle, matrix_row.reshape(1, -1))
    contrib = contrib[0]
    order = np.argsort(-np.abs(contrib))[:top_n]
    names = [bundle.features[i] for i in order]
    data = pd.DataFrame({
        "feature": [
            f"{FEATURE_DESCRIPTIONS.get(n, n)} = {display_value(display_row[n])}" for n in names
        ],
        "contribution": contrib[order],
        "magnitude": np.abs(contrib[order]),
        "direction": np.where(contrib[order] > 0, "Raises risk", "Lowers risk"),
    })
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        x=alt.X("contribution:Q", title="Contribution to the score (log-odds)",
                axis=alt.Axis(gridColor=c["track"], gridOpacity=0.6, labelColor=c["muted"],
                              titleColor=c["muted"], domain=False, ticks=False)),
        y=alt.Y("feature:N", sort=alt.EncodingSortField("magnitude", order="descending"),
                title=None, axis=alt.Axis(labelLimit=320, domain=False, ticks=False)),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                            range=[c["raises"], c["lowers"]]),
            legend=alt.Legend(orient="top", title=None, symbolType="square"),
        ),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("contribution:Q", title="Contribution", format="+.3f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=c["muted"], strokeWidth=1).encode(x="x:Q")
    st.markdown("**Why this score**: the 10 features that moved it most")
    st.altair_chart(
        (bars + zero).properties(height=30 * len(data) + 40).configure_view(stroke=None),
        width="stretch",
    )
    st.caption(
        f"Starting point (average log-odds) {bias[0]:+.3f}; adding all 40 "
        f"contributions gives {bias[0] + contrib.sum():+.3f}, which is exactly "
        "this transaction's score on the log-odds scale (TreeSHAP)."
    )


def score_histogram(table: pd.DataFrame) -> None:
    """How a batch's scores are spread, split by label when labels exist."""
    c = colors()
    frame = pd.DataFrame({"bin": np.minimum((table["risk_score"] * 20).astype(int), 19)})
    labelled = scoring.LABEL_COLUMN in table.columns
    frame["group"] = (
        np.where(table[scoring.LABEL_COLUMN].to_numpy() == 1, "Laundering", "Normal")
        if labelled else "All rows"
    )
    counts = frame.groupby(["bin", "group"]).size().reset_index(name="rows")
    counts["range"] = [f"{b / 20:.2f}–{(b + 1) / 20:.2f}" for b in counts["bin"]]
    encoding = {
        "x": alt.X("bin:O", scale=alt.Scale(domain=list(range(20))), title="Risk score",
                   axis=alt.Axis(labelAngle=0, labelColor=c["muted"], titleColor=c["muted"],
                                 labelExpr="datum.value % 5 == 0 ? format(datum.value / 20, '.2f') : ''",
                                 domain=False, ticks=False)),
        "y": alt.Y("rows:Q", title="Rows", axis=alt.Axis(gridColor=c["track"], gridOpacity=0.6,
                                                        labelColor=c["muted"], titleColor=c["muted"],
                                                        domain=False, ticks=False)),
        "tooltip": [alt.Tooltip("group:N", title="Group"), alt.Tooltip("range:N", title="Score"),
                    alt.Tooltip("rows:Q", title="Rows")],
    }
    if labelled:
        encoding["color"] = alt.Color(
            "group:N", scale=alt.Scale(domain=["Normal", "Laundering"],
                                       range=[c["normal"], c["laundering"]]),
            legend=alt.Legend(orient="top", title=None, symbolType="square"))
        encoding["xOffset"] = alt.XOffset("group:N", sort=["Normal", "Laundering"])
        mark = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    else:
        mark = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                          color=c["normal"])
    st.markdown("**How the scores are spread**")
    st.altair_chart(mark.encode(**encoding).properties(height=260).configure_view(stroke=None),
                    width="stretch", height=330)
