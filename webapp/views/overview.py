"""Overview page: what the model is, how well it did, where to go next."""

import streamlit as st

import ui

bundle = ui.get_bundle()
metrics = bundle.reference["metrics"]
budget = bundle.alert_budget

st.html(
    '<div class="aml-hero">'
    '<p class="aml-eyebrow">IBM AML project · final model</p>'
    "<h1>Anti-Money-Laundering Risk Scoring</h1>"
    "<p>An XGBoost model over 40 engineered features ranks every transaction by "
    f"risk, so investigators can review the riskiest {budget:,} first. Explore "
    "real examples, score a batch of your own, or build a hypothetical "
    "transaction and watch the score respond.</p>"
    "</div>"
)

st.subheader("How well it works")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", border=True, help=(
    "Area under the precision-recall curve: how well laundering is ranked above "
    "normal traffic across all thresholds. A random model would score about "
    "0.002 here, the share of laundering in the test period."))
c2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", border=True, help=(
    "Chance that a random laundering transaction scores higher than a random "
    "normal one. Looks high on rare-event data, so PR-AUC is the stricter number."))
c3.metric(f"Precision @ {budget:,}", f"{metrics['precision_at_budget']:.1%}", border=True,
          help=f"Of the {budget:,} highest-scored test transactions, the share that were laundering.")
c4.metric(f"Recall @ {budget:,}", f"{metrics['recall_at_budget']:.1%}", border=True,
          help=f"Share of all laundering transactions in the test period found within the top {budget:,}.")
st.caption(
    f"Held-out HI-Small test period: {bundle.reference['test_rows']:,} transactions, "
    f"{bundle.reference['test_positives']:,} labelled laundering. Of the top {budget:,} "
    f"scores, {metrics['true_positives_at_budget']:,} were laundering."
)

st.subheader("Explore")
pages = [
    ("views/sample_explorer.py", "Sample Explorer", ":material/search:",
     "Pick a real labelled transaction and see its score and the features that drove it."),
    ("views/batch_scoring.py", "Batch Scoring", ":material/table_view:",
     "Upload a model-ready file, or use a bundled one, and get a ranked alert list."),
    ("views/playground.py", "Feature Playground", ":material/tune:",
     "Build a hypothetical transaction and watch how the score responds."),
]
for column, (path, title, icon, text) in zip(st.columns(3), pages, strict=True):
    with column.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        st.caption(text)
        st.page_link(path, label=f"Open {title}", icon=icon)

st.subheader("Where the 40 features come from")
steps = [
    ("Clean", "preprocess_data.py", "Timestamps, amounts, currency and same-bank flags."),
    ("Accounts", "join_account_metadata.py", "Sender and receiver entity types."),
    ("Behaviour", "add_behavior_features.py", "Each account's activity in the previous 1h and 24h."),
    ("Network", "add_graph_features.py", "Fan-in, fan-out and rapid forwarding of money."),
    ("Score", "XGBoost", "One risk score per transaction; the top ones become alerts."),
]
for number, (column, (title, script, text)) in enumerate(zip(st.columns(5), steps, strict=True), 1):
    with column.container(border=True, height="stretch"):
        st.html(
            f'<div class="aml-step">Step {number}</div>'
            f'<div class="aml-step-title">{title}</div>'
            f'<div class="aml-step-body">{text}</div>'
        )
        st.caption(f"`{script}`")

st.info(
    "The model needs each account's recent history, so it cannot score a single "
    "raw transaction typed in without context. Scores are rankings from a heavily "
    "class-weighted model, not calibrated probabilities, and a high score is a "
    "reason to investigate, not proof of laundering.",
    icon=":material/info:",
)
