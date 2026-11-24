"""Unit tests for webapp/scoring.py.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest webapp/tests
"""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

WEBAPP = Path(__file__).resolve().parents[1]
ROOT = WEBAPP.parent
sys.path.insert(0, str(WEBAPP))

import scoring  # noqa: E402

SAMPLES = WEBAPP / "sample_data"


@pytest.fixture(scope="module")
def bundle() -> scoring.ModelBundle:
    return scoring.load_bundle()


@pytest.fixture(scope="module")
def mixed() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "quick_demo_mixed_20.csv")


def scores_for(bundle, frame):
    return scoring.score(bundle, scoring.prepare_batch(frame, bundle).matrix)


# --- loading and parity ----------------------------------------------------

def test_bundle_shape(bundle):
    assert len(bundle.features) == 40
    assert bundle.booster.num_features() == 40
    assert set(bundle.categorical_features) <= set(bundle.features)
    # Model was early-stopped: only the first best_iteration + 1 trees count.
    assert bundle.iteration_range[1] < bundle.booster.num_boosted_rounds()


def test_matches_sklearn_predict_proba(bundle, mixed):
    pytest.importorskip("sklearn")  # XGBClassifier needs it; the app does not
    classifier = xgb.XGBClassifier()
    classifier.load_model(WEBAPP / "model" / "xgboost_graph_model.json")
    matrix = scoring.prepare_batch(mixed, bundle).matrix
    expected = classifier.predict_proba(matrix)[:, 1]
    np.testing.assert_allclose(scoring.score(bundle, matrix), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "inputs, predictions",
    [
        ("processed/sample_laundering.csv", "processed/sample_laundering_predictions.csv"),
        ("processed/sample_not_laundering.csv", "processed/sample_not_laundering_predictions.csv"),
        ("webapp/sample_data/medium_mixed_examples.csv", "processed_medium/medium_predictions.csv"),
    ],
)
def test_matches_original_predict_batch_output(bundle, inputs, predictions):
    """Scores equal those saved by the pipeline's predict_batch.py.

    Both files hold the same rows in different orders, and several rows share
    a timestamp and account pair, so the score sets are compared sorted.
    """
    if not (ROOT / inputs).exists() or not (ROOT / predictions).exists():
        pytest.skip("pipeline output not available (git-ignored)")
    frame = pd.read_csv(ROOT / inputs)
    expected = pd.read_csv(ROOT / predictions)
    assert len(frame) == len(expected)
    assert frame["is_laundering"].sum() == expected["is_laundering"].sum()
    np.testing.assert_allclose(
        np.sort(scores_for(bundle, frame)),
        np.sort(expected["laundering_probability"].to_numpy()),
        atol=1e-6,
    )


def test_reference_metrics_match_pipeline(bundle):
    metrics = bundle.reference["metrics"]
    assert metrics["pr_auc"] == pytest.approx(0.468044, abs=1e-6)
    assert metrics["true_positives_at_budget"] == 635
    assert bundle.reference["test_rows"] == 762_103


# --- explanations ----------------------------------------------------------

def test_contributions_add_up_to_score(bundle, mixed):
    matrix = scoring.prepare_batch(mixed, bundle).matrix
    contrib, bias = scoring.contributions(bundle, matrix)
    assert contrib.shape == matrix.shape
    logit = np.log(scoring.score(bundle, matrix) / (1 - scoring.score(bundle, matrix)))
    np.testing.assert_allclose(bias + contrib.sum(axis=1), logit, atol=1e-4)


# --- input validation ------------------------------------------------------

def pipeline_encode(frame: pd.DataFrame, bundle) -> np.ndarray:
    """Independent copy of train_xgboost.py / predict_batch.py encoding."""
    pl = pytest.importorskip("polars")
    table = pl.from_pandas(frame[bundle.features])
    table = table.with_columns([
        pl.col(c).cast(pl.String).replace_strict(bundle.mappings[c], default=-1)
        .cast(pl.Int32).alias(c)
        for c in bundle.categorical_features
    ])
    return table.select(bundle.features).to_numpy().astype(np.float32)


def test_encoding_matches_pipeline_including_blank_and_unseen(bundle, mixed):
    frame = mixed.head(4).copy()
    frame["sender_entity_type"] = frame["sender_entity_type"].astype(object)
    frame.loc[0, "sender_entity_type"] = None            # blank (unmatched account)
    frame.loc[1, "sender_entity_type"] = "Alien Corp"    # never seen in training
    frame.loc[2, "receiving_currency"] = None
    np.testing.assert_array_equal(
        scoring.prepare_batch(frame, bundle).matrix, pipeline_encode(frame, bundle)
    )
    column = bundle.features.index("sender_entity_type")
    assert scoring.prepare_batch(frame, bundle).matrix[0, column] == -1


def test_real_rows_with_blank_entity_types_match_pipeline(bundle):
    """The 56 HI-Small rows with a blank entity type score exactly as the
    pipeline scores them (encoded independently, via predict_proba)."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("sklearn")
    table_path = ROOT / "processed" / "transactions_with_graph_features.parquet"
    if not table_path.exists():
        pytest.skip("pipeline output not available (git-ignored)")
    rows = (
        pl.scan_parquet(table_path)
        .filter(pl.col("sender_entity_type").is_null()
                | pl.col("receiver_entity_type").is_null())
        .select(bundle.features)
        .collect()
        .to_pandas()
    )
    assert len(rows) > 0
    classifier = xgb.XGBClassifier()
    classifier.load_model(WEBAPP / "model" / "xgboost_graph_model.json")
    expected = classifier.predict_proba(pipeline_encode(rows, bundle))[:, 1]
    np.testing.assert_allclose(scores_for(bundle, rows), expected, rtol=1e-6)


def test_unseen_category_is_unknown_with_warning(bundle, mixed):
    row = mixed.head(1).copy()
    row["payment_format"] = "Carrier Pigeon"
    prepared = scoring.prepare_batch(row, bundle)
    assert prepared.matrix[0, bundle.features.index("payment_format")] == -1
    assert any("payment_format" in w for w in prepared.warnings)


def test_missing_columns_rejected(bundle, mixed):
    with pytest.raises(scoring.BatchValidationError, match="amount_paid"):
        scoring.prepare_batch(mixed.drop(columns=["amount_paid"]), bundle)


def test_raw_transactions_rejected(bundle):
    raw = pd.DataFrame({"amount_paid": [100.0], "payment_format": ["ACH"]})
    with pytest.raises(scoring.BatchValidationError, match="missing 38 of the 40"):
        scoring.prepare_batch(raw, bundle)


def test_empty_and_oversized_rejected(bundle, mixed):
    with pytest.raises(scoring.BatchValidationError, match="no rows"):
        scoring.prepare_batch(mixed.head(0), bundle)
    with pytest.raises(scoring.BatchValidationError, match="limit"):
        scoring.prepare_batch(mixed, bundle, max_rows=5)


@pytest.mark.parametrize("bad", ["lots", np.inf, 1e39])  # 1e39 overflows float32
def test_non_numeric_values_rejected(bundle, mixed, bad):
    frame = mixed.copy()
    frame["amount_paid"] = frame["amount_paid"].astype(object)
    frame.loc[0, "amount_paid"] = bad
    with pytest.raises(scoring.BatchValidationError, match="amount_paid"):
        scoring.prepare_batch(frame, bundle)


def test_missing_numeric_values_allowed(bundle, mixed):
    frame = mixed.copy()
    frame.loc[0, "sender_seconds_since_activity"] = np.nan  # first-ever activity
    assert np.isfinite(scores_for(bundle, frame)).all()


def test_invalid_labels_ignored_with_warning(bundle, mixed):
    frame = mixed.copy()
    frame.loc[0, "is_laundering"] = 7
    prepared = scoring.prepare_batch(frame, bundle)
    assert prepared.labels is None
    assert any("is_laundering" in w for w in prepared.warnings)


# --- file parsing ----------------------------------------------------------

def test_read_table_formats(mixed):
    comma = mixed.to_csv(index=False)
    expected = scoring.read_table(comma.encode("utf-8"), "a.csv")
    assert expected.shape == mixed.shape
    semicolon = mixed.to_csv(index=False, sep=";").encode("utf-8")
    assert scoring.read_table(semicolon, "b.csv").shape == mixed.shape
    bom = b"\xef\xbb\xbf" + comma.encode("utf-8")
    assert list(scoring.read_table(bom, "c.csv").columns) == list(mixed.columns)
    windows = mixed.assign(from_account="Café").to_csv(index=False).encode("cp1252")
    assert scoring.read_table(windows, "d.csv")["from_account"].iloc[0] == "Café"
    buffer = io.BytesIO()
    mixed.to_parquet(buffer, index=False)
    assert scoring.read_table(buffer.getvalue(), "e.parquet").shape == mixed.shape


def test_read_table_unreadable():
    with pytest.raises(scoring.BatchValidationError, match="Could not read"):
        scoring.read_table(b"definitely not parquet", "x.parquet")


# --- ranking ---------------------------------------------------------------

def test_rank_alerts():
    rank, alert = scoring.rank_alerts(np.array([0.2, 0.9, 0.9, 0.1]), 2)
    assert rank.tolist() == [3, 1, 2, 4]  # ties keep input order
    assert alert.tolist() == [False, True, True, False]
    _, alert = scoring.rank_alerts(np.array([0.5]), 1000)  # budget > rows
    assert alert.tolist() == [True]
    with pytest.raises(ValueError):
        scoring.rank_alerts(np.array([0.5]), 0)


def test_alert_metrics():
    m = scoring.alert_metrics(np.array([1, 0, 1, 0]), np.array([True, True, False, False]))
    assert (m["true_positives"], m["false_positives"], m["positives"]) == (1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_percentile_and_cutoff(bundle):
    share_above = bundle.reference["alert_budget"] / bundle.reference["test_rows"]
    assert bundle.percentile(bundle.top_k_cutoff) == pytest.approx(1 - share_above, abs=2e-4)
    assert bundle.percentile(0.0) == 0.0 and bundle.percentile(1.0) == 1.0


# --- playground ------------------------------------------------------------

def test_playground_rows_respect_pipeline_identities(bundle):
    rng = np.random.default_rng(0)
    currencies = list(bundle.mappings["payment_currency"])
    for _ in range(200):
        overrides = {
            "amount_paid": float(rng.uniform(0, 50_000)),
            "amount_received": float(rng.uniform(0, 50_000)),
            "payment_currency": rng.choice(currencies),
            "receiving_currency": rng.choice(currencies),
            "same_account": int(rng.integers(0, 2)),
            "same_bank": int(rng.integers(0, 2)),
            "transaction_day": int(rng.integers(1, 31)),
        }
        for side in ("sender", "receiver"):
            for name in ("outgoing_count_24h", "unique_recipients_24h",
                         "incoming_count_24h", "unique_senders_24h", "tx_count_1h"):
                overrides[f"{side}_{name}"] = int(rng.integers(0, 4) * rng.integers(0, 10))
            overrides[f"{side}_seconds_since_activity"] = float(rng.uniform(0, 200_000))
        r = scoring.build_playground_row(bundle, overrides).iloc[0]

        assert list(r.index) == bundle.features
        assert r.sender_historical_fan_out_24h == r.sender_unique_recipients_24h
        assert r.sender_historical_edge_count_24h == r.sender_outgoing_count_24h
        assert r.receiver_historical_fan_in_24h == r.receiver_unique_senders_24h
        assert r.receiver_historical_edge_count_24h == r.receiver_incoming_count_24h
        assert r.sender_received_money_24h == int(r.sender_incoming_count_24h > 0)
        assert r.rapid_forwarding_indicator == int(
            r.sender_received_money_24h == 1 and r.sender_historical_edge_count_24h > 0)
        for side in ("sender", "receiver"):
            assert r[f"{side}_unique_recipients_24h"] <= r[f"{side}_outgoing_count_24h"]
            assert r[f"{side}_unique_senders_24h"] <= r[f"{side}_incoming_count_24h"]
            assert r[f"{side}_tx_count_1h"] <= (
                r[f"{side}_outgoing_count_24h"] + r[f"{side}_incoming_count_24h"])
            # seconds-since-activity agrees with the activity counts
            seconds = r[f"{side}_seconds_since_activity"]
            if r[f"{side}_tx_count_1h"] > 0:
                assert 0 <= seconds <= 3_600
            elif r[f"{side}_outgoing_count_24h"] + r[f"{side}_incoming_count_24h"] > 0:
                assert 3_600 < seconds <= 86_400
            else:
                assert seconds > 86_400
        assert r.same_currency == int(r.payment_currency == r.receiving_currency)
        if r.same_currency:
            assert r.amount_received == r.amount_paid
        if r.same_account:
            assert r.same_bank == 1
            assert r.receiver_entity_type == r.sender_entity_type
            for name in scoring.HISTORY_FIELDS:
                assert r[f"receiver_{name}"] == r[f"sender_{name}"]
        assert r.transaction_weekday == date(2022, 9, int(r.transaction_day)).isoweekday()
        assert np.isfinite(scores_for(bundle, pd.DataFrame([r]))).all()
