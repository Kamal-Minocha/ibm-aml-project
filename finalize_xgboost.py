"""
Finalize the XGBoost model for a 1,000-alert investigation budget.

The threshold is selected from validation scores only. The saved threshold is
then evaluated once on the chronological test set.
"""

from pathlib import Path
import json

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "processed" / "transactions_with_graph_features.parquet"
)
MODEL_PATH = BASE_DIR / "processed" / "xgboost_graph_model.json"
CONFIG_PATH = BASE_DIR / "processed" / "final_xgboost_config.json"
METRICS_PATH = BASE_DIR / "processed" / "final_xgboost_metrics.txt"

NUMERIC_FEATURES = [
    "amount_paid", "amount_received", "transaction_hour",
    "transaction_weekday", "transaction_day", "transaction_month",
    "same_bank", "same_account", "same_currency",
    "received_amount_nonpositive", "paid_amount_nonpositive",
    "sender_metadata_matched", "receiver_metadata_matched",
    "sender_tx_count_1h", "sender_outgoing_count_24h",
    "sender_incoming_count_24h", "sender_outgoing_amount_24h",
    "sender_incoming_amount_24h", "sender_unique_recipients_24h",
    "sender_unique_senders_24h", "sender_seconds_since_activity",
    "receiver_tx_count_1h", "receiver_outgoing_count_24h",
    "receiver_incoming_count_24h", "receiver_outgoing_amount_24h",
    "receiver_incoming_amount_24h", "receiver_unique_recipients_24h",
    "receiver_unique_senders_24h", "receiver_seconds_since_activity",
    "sender_historical_fan_out_24h",
    "sender_historical_edge_count_24h",
    "receiver_historical_fan_in_24h",
    "receiver_historical_edge_count_24h",
    "sender_received_money_24h",
    "rapid_forwarding_indicator",
]
CATEGORICAL_FEATURES = [
    "payment_format",
    "payment_currency",
    "receiving_currency",
    "sender_entity_type",
    "receiver_entity_type",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_laundering"
ALERT_BUDGET = 1_000


def build_mappings(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    mappings = {}
    for column in CATEGORICAL_FEATURES:
        values = (
            frame.select(
                pl.col(column).cast(pl.String).fill_null("__MISSING__")
            )
            .unique()
            .sort(column)
            .get_column(column)
            .to_list()
        )
        mappings[column] = {value: index for index, value in enumerate(values)}
    return mappings


def encode(
    frame: pl.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> pl.DataFrame:
    return frame.with_columns([
        pl.col(column)
        .cast(pl.String)
        .replace_strict(mappings[column], default=-1)
        .cast(pl.Int32)
        .alias(column)
        for column in CATEGORICAL_FEATURES
    ])


def arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.select(FEATURES).to_numpy(),
        frame.get_column(TARGET).to_numpy().astype(np.int8),
    )


def top_k_threshold(
    probabilities: np.ndarray,
    budget: int,
) -> float:
    order = np.argsort(-probabilities, kind="stable")
    return float(probabilities[order[budget - 1]])


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": threshold,
        "alerts": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "pr_auc": average_precision_score(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


def top_k_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    budget: int,
) -> dict[str, float | int | list[list[int]]]:
    order = np.argsort(-probabilities, kind="stable")[:budget]
    predictions = np.zeros(len(labels), dtype=np.int8)
    predictions[order] = 1
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "alerts": budget,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "pr_auc": average_precision_score(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


for path in [DATA_PATH, MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

frame = (
    pl.scan_parquet(DATA_PATH)
    .select(["timestamp", TARGET, *FEATURES])
    .filter(pl.col(TARGET).is_in([0, 1]))
    .collect(engine="streaming")
)

train_end, validation_end = (
    frame
    .select([
        pl.col("timestamp").quantile(0.70).alias("train_end"),
        pl.col("timestamp").quantile(0.85).alias("validation_end"),
    ])
    .row(0)
)
train = frame.filter(pl.col("timestamp") < train_end)
validation = frame.filter(
    (pl.col("timestamp") >= train_end)
    & (pl.col("timestamp") < validation_end)
)
test = frame.filter(pl.col("timestamp") >= validation_end)

mappings = build_mappings(train)
validation = encode(validation, mappings)
test = encode(test, mappings)
x_validation, y_validation = arrays(validation)
x_test, y_test = arrays(test)

model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))
validation_probabilities = model.predict_proba(x_validation)[:, 1]
test_probabilities = model.predict_proba(x_test)[:, 1]

threshold = top_k_threshold(validation_probabilities, ALERT_BUDGET)
validation_metrics = metrics(
    y_validation,
    validation_probabilities,
    threshold,
)
test_metrics = metrics(y_test, test_probabilities, threshold)
validation_budget_metrics = top_k_metrics(
    y_validation,
    validation_probabilities,
    ALERT_BUDGET,
)
test_budget_metrics = top_k_metrics(
    y_test,
    test_probabilities,
    ALERT_BUDGET,
)

config = {
    "model": "XGBoost",
    "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
    "feature_data": str(DATA_PATH.relative_to(BASE_DIR)),
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "alert_budget": ALERT_BUDGET,
    "decision_policy": "top_k_scores_per_evaluation_period",
    "threshold_selected_on": "validation",
    "threshold": threshold,
    "train_end": str(train_end),
    "validation_end": str(validation_end),
}
CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

report = [
    "Final XGBoost configuration",
    "=" * 45,
    f"Alert budget: {ALERT_BUDGET:,}",
    f"Threshold selected from validation: {threshold:.12f}",
    "Final decision policy: top 1,000 XGBoost scores per evaluation period",
    "",
    "VALIDATION RESULTS",
    "-" * 25,
    *(f"{key}: {value}" for key, value in validation_metrics.items()),
    "",
    "VALIDATION TOP-K RESULTS",
    "-" * 25,
    *(
        f"{key}: {value}"
        for key, value in validation_budget_metrics.items()
    ),
    "",
    "TEST RESULTS",
    "-" * 25,
    *(f"{key}: {value}" for key, value in test_metrics.items()),
    "",
    "TEST TOP-K RESULTS (FINAL POLICY)",
    "-" * 25,
    *(f"{key}: {value}" for key, value in test_budget_metrics.items()),
]
METRICS_PATH.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"Saved configuration: {CONFIG_PATH}")
print(f"Saved metrics: {METRICS_PATH}")
