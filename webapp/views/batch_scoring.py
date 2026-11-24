"""Batch Scoring: rank a whole file and flag the top rows as alerts."""

import streamlit as st

import scoring
import ui

bundle = ui.get_bundle()
ui.page_header(
    "Batch Scoring",
    "Score a file that already has the 40 engineered columns (the output of the "
    "project's feature pipeline). Every row is ranked by risk and the top ones "
    "become alerts.",
)

with st.container(border=True):
    left, right = st.columns(2, gap="large")
    uploaded = left.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    with right:
        bundled = st.selectbox(
            "...or use a bundled sample file",
            ["(none)", *ui.SAMPLE_FILES],
            format_func=lambda n: n if n == "(none)" else f"{n}: {ui.SAMPLE_FILES[n]}",
            disabled=uploaded is not None,
        )
        with st.popover("What should the file contain?", icon=":material/help:"):
            st.markdown(
                "All **40 feature columns** by name, in any order; extra columns are "
                "ignored. Optional columns kept in the output: `transaction_id`, "
                "`timestamp`, `from_account`, `to_account`, and `is_laundering` "
                "(0/1, enables precision and recall). CSV (comma or semicolon, "
                "UTF-8 or Windows-1252) or Parquet, up to 200,000 rows."
            )
            st.code("\n".join(bundle.features), language=None)

if uploaded is not None:
    data, name = uploaded.getvalue(), uploaded.name
elif bundled != "(none)":
    data, name = (ui.SAMPLE_DIR / bundled).read_bytes(), bundled
else:
    st.info("Upload a file or pick a bundled sample to begin.", icon=":material/upload_file:")
    st.stop()

try:
    results, warnings = ui.score_table(data, name)
except scoring.BatchValidationError as error:  # unreadable or invalid file
    st.error(f"This file can't be scored. {error}", icon=":material/error:")
    st.stop()
except Exception as error:  # anything unforeseen: report, don't crash
    st.error(f"Unexpected error while scoring `{name}`: {error}", icon=":material/error:")
    st.stop()

for warning in warnings:
    st.warning(warning, icon=":material/warning:")

rows = len(results)
control, _ = st.columns([1, 2])
budget = int(control.number_input(
    "Alert budget (how many top-ranked rows to flag)",
    min_value=1, max_value=rows, value=min(bundle.alert_budget, rows), step=1,
))
rank, is_alert = scoring.rank_alerts(results["risk_score"].to_numpy(), budget)
table = results.assign(alert_rank=rank, is_alert=is_alert).sort_values("alert_rank")
lowest_alert = float(table.loc[table["is_alert"], "risk_score"].min())

st.subheader(f"{rows:,} rows scored, top {budget:,} flagged")
labelled = scoring.LABEL_COLUMN in table.columns
cards = st.columns(5 if labelled else 2)
cards[0].metric("Rows scored", f"{rows:,}", border=True)
cards[1].metric("Alerts", f"{budget:,}", border=True,
                help=f"The {budget:,} highest scores; the lowest alerted score is {lowest_alert:.4f}.")
if labelled:
    m = scoring.alert_metrics(table[scoring.LABEL_COLUMN].to_numpy(), table["is_alert"].to_numpy())
    cards[2].metric("Laundering among alerts", f"{m['true_positives']:,}", border=True)
    cards[3].metric("Precision", f"{m['precision']:.1%}", border=True,
                    help="Share of the alerts that are labelled laundering.")
    cards[4].metric("Recall", f"{m['recall']:.1%}", border=True,
                    help=f"{m['true_positives']:,} of {m['positives']:,} labelled laundering rows flagged.")
    st.caption(
        "Precision and recall are shown because the file has an `is_laundering` "
        "column. Bundled samples are far richer in laundering than real traffic "
        "(~0.1%), so these numbers are much higher than real-world precision."
    )

ui.score_histogram(table)

st.markdown("**Ranked transactions**")
shown = table.head(ui.TABLE_PREVIEW_ROWS)
if rows > ui.TABLE_PREVIEW_ROWS:
    st.caption(f"Showing the top {ui.TABLE_PREVIEW_ROWS:,} rows; the download has all {rows:,}.")
st.dataframe(
    shown,
    hide_index=True,
    width="stretch",
    column_config={
        "risk_score": st.column_config.ProgressColumn(
            "risk_score", min_value=0.0, max_value=1.0, format="%.4f"),
        "is_alert": st.column_config.CheckboxColumn("is_alert"),
        "alert_rank": st.column_config.NumberColumn("alert_rank", format="%d"),
    },
)
st.download_button(
    "Download ranked results (CSV)",
    table.to_csv(index=False).encode("utf-8"),
    file_name="aml_ranked_alerts.csv",
    mime="text/csv",
    icon=":material/download:",
    type="primary",
)
