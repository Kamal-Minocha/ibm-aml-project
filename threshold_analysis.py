"""
Compare saved LightGBM and XGBoost models at equivalent recall levels.

Thresholds are selected on validation data only and then evaluated once on
the chronological test data.
"""

from pathlib import Path

import lightgbm as lgb
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
LIGHTGBM_PATH = BASE_DIR / "processed" / "graph_model.txt"
XGBOOST_PATH = (
    BASE_DIR / "processed" / "xgboost_graph_model.json"
)
OUTPUT_PATH = BASE_DIR / "processed" / "threshold_analysis.csv"
REPORT_PATH = BASE_DIR / "processed" / "threshold_analysis.txt"

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
TARGET_RECALLS = [0.40, 0.60, 0.70, 0.80, 0.90]


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


def select_threshold_for_recall(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
) -> float:
    order = np.argsort(-probabilities, kind="stable")
    ordered_labels = labels[order]
    cumulative_true_positives = np.cumsum(ordered_labels)
    total_positives = int(labels.sum())

    recalls = cumulative_true_positives / total_positives
    eligible = np.flatnonzero(recalls >= target_recall)
    if len(eligible) == 0:
        raise RuntimeError(
            f"Cannot reach target recall {target_recall:.0%}."
        )

    # Select the smallest alert set that reaches the requested recall.
    index = int(eligible[0])
    return float(probabilities[order[index]])


def evaluate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "alerts": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "pr_auc": average_precision_score(labels, probabilities),
    }


for path in [DATA_PATH, LIGHTGBM_PATH, XGBOOST_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Loading graph-enhanced data...")
lazy = pl.scan_parquet(DATA_PATH)
frame = (
    lazy
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
train = encode(train, mappings)
validation = encode(validation, mappings)
test = encode(test, mappings)

def arrays(data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        data.select(FEATURES).to_numpy(),
        data.get_column(TARGET).to_numpy().astype(np.int8),
    )


x_train, y_train = arrays(train)
x_validation, y_validation = arrays(validation)
x_test, y_test = arrays(test)

print("Loading saved models and generating probabilities...")
lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_PATH))
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model(str(XGBOOST_PATH))

model_probabilities = {
    "LightGBM": (
        lightgbm_model.predict(x_validation),
        lightgbm_model.predict(x_test),
    ),
    "XGBoost": (
        xgboost_model.predict_proba(x_validation)[:, 1],
        xgboost_model.predict_proba(x_test)[:, 1],
    ),
}

rows = []
for model_name, (validation_probabilities, test_probabilities) in (
    model_probabilities.items()
):
    for target_recall in TARGET_RECALLS:
        threshold = select_threshold_for_recall(
            y_validation,
            validation_probabilities,
            target_recall,
        )
        test_metrics = evaluate(
            y_test,
            test_probabilities,
            threshold,
        )
        rows.append({
            "model": model_name,
            "target_validation_recall": target_recall,
            "validation_threshold": threshold,
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_alerts": test_metrics["alerts"],
            "test_true_positives": test_metrics["true_positives"],
            "test_false_positives": test_metrics["false_positives"],
            "test_false_negatives": test_metrics["false_negatives"],
        })

results = pl.DataFrame(rows)
results.write_csv(OUTPUT_PATH)

report_lines = [
    "Threshold analysis: LightGBM versus XGBoost",
    "=" * 55,
    "Thresholds selected on validation data only.",
    "Metrics below are measured on the untouched chronological test set.",
    f"Train end: {train_end}",
    f"Validation end: {validation_end}",
    "",
]
report_lines.extend(
    "\t".join(str(value) for value in row)
    for row in results.iter_rows()
)
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print(results.write_csv())
print(f"Saved table: {OUTPUT_PATH}")
print(f"Saved report: {REPORT_PATH}")
