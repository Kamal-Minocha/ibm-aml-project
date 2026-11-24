"""Sample Explorer: one real labelled transaction, its score and why."""

import pandas as pd
import streamlit as st

import scoring
import ui

bundle = ui.get_bundle()
ui.page_header(
    "Sample Explorer",
    "Real transactions that already went through the full feature pipeline, "
    "with their true labels. Pick one to see its score and what drove it.",
)

with st.container(border=True):
    left, middle = st.columns(2)
    source = left.segmented_control(
        "Source",
        ["HI-Small held-out test period", "HI-Medium, Sep 19+ (unseen data)"],
        default="HI-Small held-out test period",
        required=True,
        key="se_source",
    )
    label = middle.segmented_control(
        "Show", ["Laundering", "Normal"], default="Laundering", required=True, key="se_show",
    )
    if source.startswith("HI-Small"):
        pool = pd.concat(
            [ui.read_sample("small_laundering_examples.csv"),
             ui.read_sample("small_not_laundering_examples.csv")],
            ignore_index=True,
        )
    else:
        pool = ui.read_sample("medium_mixed_examples.csv")
    pool = pool[pool[scoring.LABEL_COLUMN] == (1 if label == "Laundering" else 0)]
    pool = pool.reset_index(drop=True)

    index = st.selectbox(
        f"Transaction ({len(pool)} available)",
        range(len(pool)),
        format_func=lambda i: (
            f"#{i + 1}  ·  {pool.at[i, 'timestamp']}  ·  "
            f"{pool.at[i, 'amount_paid']:,.2f} {pool.at[i, 'payment_currency']}  ·  "
            f"{pool.at[i, 'payment_format']}"
        ),
    )

row = pool.iloc[[index]]
prepared = scoring.prepare_batch(row, bundle)
score_value = float(scoring.score(bundle, prepared.matrix)[0])

left, right = st.columns([5, 6], gap="large")
with left:
    st.markdown("**The model's verdict**")
    ui.score_panel(score_value)
    st.metric("Dataset label", label, border=True,
              help="The ground truth from the IBM dataset, for comparison with the score.")
    st.caption(
        f"'Top-{bundle.alert_budget:,} alert' is how this transaction would have fared "
        "in the test period's review queue."
    )
with right:
    ui.explanation_chart(row.iloc[0], prepared.matrix[0])

with st.expander("All 40 feature values", icon=":material/list:"):
    st.dataframe(
        pd.DataFrame({
            "feature": bundle.features,
            "meaning": [ui.FEATURE_DESCRIPTIONS.get(f, "") for f in bundle.features],
            "value": [ui.display_value(row.iloc[0][f]) for f in bundle.features],
        }),
        hide_index=True,
        width="stretch",
    )
