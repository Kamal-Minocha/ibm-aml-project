"""Headless UI tests for the web app (webapp/app.py + webapp/views/) using
Streamlit's AppTest.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest webapp/tests
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

WEBAPP = Path(__file__).resolve().parents[1]
APP = str(WEBAPP / "app.py")
SAMPLES = WEBAPP / "sample_data"
PAGE_FILES = {
    "Overview": "views/overview.py",
    "Sample Explorer": "views/sample_explorer.py",
    "Batch Scoring": "views/batch_scoring.py",
    "Feature Playground": "views/playground.py",
}


def go_to(at: AppTest, page: str) -> AppTest:
    at.switch_page(PAGE_FILES[page]).run()
    assert not at.exception, at.exception
    return at


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return go_to(at, page)


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def upload(at: AppTest, name: str, content: bytes, mime: str = "text/csv") -> AppTest:
    at.file_uploader[0].set_value((name, content, mime)).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize(
    "page", ["Overview", "Sample Explorer", "Batch Scoring", "Feature Playground"]
)
def test_every_page_renders(page):
    open_page(page)


def test_overview_shows_recomputed_metrics():
    at = open_page("Overview")
    assert metric(at, "PR-AUC") == "0.468"
    assert metric(at, "Precision @ 1,000") == "63.5%"


def test_sample_explorer_medium_normal_row():
    at = open_page("Sample Explorer")
    at.button_group(key="se_source").set_value("HI-Medium, Sep 19+ (unseen data)").run()
    at.button_group(key="se_show").set_value("Normal").run()
    at.selectbox[0].set_value(5).run()
    assert not at.exception
    assert metric(at, "Dataset label") == "Normal"
    assert metric(at, "Top-1,000 alert?") in {"Yes", "No"}


@pytest.mark.parametrize(
    "sample",
    ["quick_demo_mixed_20.csv", "small_laundering_examples.csv",
     "small_not_laundering_examples.csv", "medium_mixed_examples.csv"],
)
def test_batch_bundled_samples(sample):
    at = open_page("Batch Scoring")
    at.selectbox[0].set_value(sample).run()
    assert not at.exception
    assert not at.error
    rows = len(pd.read_csv(SAMPLES / sample))
    assert metric(at, "Alerts") == f"{rows:,}"  # default budget covers all rows
    # Narrowing the budget reuses the cached scores and updates the metrics.
    at.number_input[0].set_value(1).run()
    assert metric(at, "Alerts") == "1"


def test_batch_single_row_upload():
    """A one-row file must not break the budget widget (min == max == 1)."""
    row = pd.read_csv(SAMPLES / "quick_demo_mixed_20.csv").head(1)
    at = upload(open_page("Batch Scoring"), "one.csv", row.to_csv(index=False).encode())
    assert at.number_input[0].value == 1
    assert metric(at, "Alerts") == "1"


def test_batch_parquet_upload():
    frame = pd.read_csv(SAMPLES / "medium_mixed_examples.csv")
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    at = upload(open_page("Batch Scoring"), "batch.parquet", buffer.getvalue(),
                "application/octet-stream")
    assert metric(at, "Alerts") == "200"


def test_batch_rejects_raw_transactions():
    raw = b"Timestamp,From Bank,Account,Amount Paid\n2022/09/01 00:20,10,8000EBD30,3697.34\n"
    at = upload(open_page("Batch Scoring"), "raw.csv", raw)
    assert at.error and "required feature columns" in at.error[0].value


def test_batch_rejects_unreadable_file():
    at = upload(open_page("Batch Scoring"), "broken.parquet", b"not a parquet file",
                "application/octet-stream")
    assert at.error and "Could not read" in at.error[0].value


def test_playground_preset_raises_score_and_derives_flags():
    at = open_page("Feature Playground")
    typical = float(metric(at, "Risk score"))
    preset = next(b for b in at.button if b.label == "Fan-out with rapid forwarding")
    preset.click().run()
    assert not at.exception
    assert float(metric(at, "Risk score")) > typical
    assert any("rapid forwarding = **yes**" in m.value for m in at.markdown)


def test_playground_clamps_dependent_counts():
    at = open_page("Feature Playground")
    outgoing = at.number_input(key="pg_sender_outgoing_count_24h")
    outgoing.set_value(0).run()
    assert not at.exception
    assert at.number_input(key="pg_sender_unique_recipients_24h").value == 0
    assert any("rapid forwarding = **no**" in m.value for m in at.markdown)


def test_playground_values_survive_page_switch():
    at = open_page("Feature Playground")
    at.number_input(key="pg_sender_outgoing_count_24h").set_value(17).run()
    go_to(at, "Overview")
    go_to(at, "Feature Playground")
    assert not at.exception
    assert at.number_input(key="pg_sender_outgoing_count_24h").value == 17


def test_amount_received_survives_currency_toggle():
    at = open_page("Feature Playground")
    at.selectbox(key="pg_receiving_currency").set_value("Euro").run()
    at.number_input(key="pg_amount_received").set_value(5.0).run()
    at.selectbox(key="pg_receiving_currency").set_value("US Dollar").run()
    at.selectbox(key="pg_receiving_currency").set_value("Euro").run()
    assert not at.exception
    assert at.number_input(key="pg_amount_received").value == 5.0


def test_self_transfer_mirrors_sender():
    at = open_page("Feature Playground")
    at.number_input(key="pg_sender_outgoing_count_24h").set_value(12).run()
    at.checkbox(key="pg_same_account").check().run()
    assert not at.exception
    receiver = at.number_input(key="pg_receiver_outgoing_count_24h")
    assert receiver.value == 12 and receiver.disabled
    assert at.checkbox(key="pg_same_bank").value is True
    assert (at.selectbox(key="pg_receiver_entity_type").value
            == at.selectbox(key="pg_sender_entity_type").value)


def test_minutes_since_bounded_by_counts():
    at = open_page("Feature Playground")
    at.number_input(key="pg_sender_minutes_since").set_value(30.0).run()
    for key in ("pg_sender_outgoing_count_24h", "pg_sender_incoming_count_24h"):
        at.number_input(key=key).set_value(0).run()
    assert not at.exception
    # No activity in 24h: the previous activity must be over a day ago.
    assert at.number_input(key="pg_sender_minutes_since").value > 1_440


def test_default_page_is_overview():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert metric(at, "PR-AUC") == "0.468"


def test_batch_without_labels_shows_counts_only():
    frame = pd.read_csv(SAMPLES / "quick_demo_mixed_20.csv").drop(columns=["is_laundering"])
    at = upload(open_page("Batch Scoring"), "unlabelled.csv", frame.to_csv(index=False).encode())
    labels = {m.label for m in at.metric}
    assert {"Rows scored", "Alerts"} <= labels and "Precision" not in labels
