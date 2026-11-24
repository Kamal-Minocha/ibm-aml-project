"""
Compare LightGBM and XGBoost at fixed investigation alert budgets.

The models rank the same chronological test transactions. For each budget,
only the top-k scores are treated as alerts.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "processed" / "transactions_with_graph_features.parquet"
)
LIGHTGBM_PATH = BASE_DIR / "processed" / "graph_model.txt"
XGBOOST_PATH = BASE_DIR / "processed" / "xgboost_graph_model.json"
OUTPUT_PATH = BASE_DIR / "processed" / "alert_budget_analysis.csv"
REPORT_PATH = BASE_DIR / "processed" / "alert_budget_analysis.txt"

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
ALERT_BUDGETS = [100, 500, 1_000, 5_000, 10_000]


def build_mappings(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    result = {}
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
        result[column] = {value: index for index, value in enumerate(values)}
    return result


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


def to_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.select(FEATURES).to_numpy(),
        frame.get_column(TARGET).to_numpy().astype(np.int8),
    )


def evaluate_budget(
    labels: np.ndarray,
    probabilities: np.ndarray,
    budget: int,
) -> dict[str, float | int]:
    order = np.argsort(-probabilities, kind="stable")[:budget]
    selected_labels = labels[order]
    true_positives = int(selected_labels.sum())
    total_positives = int(labels.sum())
    return {
        "alert_budget": budget,
        "true_positives": true_positives,
        "false_positives": budget - true_positives,
        "precision": true_positives / budget,
        "recall": true_positives / total_positives,
        "pr_auc": average_precision_score(labels, probabilities),
    }


for path in [DATA_PATH, LIGHTGBM_PATH, XGBOOST_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Loading test data...")
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
test = frame.filter(pl.col("timestamp") >= validation_end)

mappings = build_mappings(train)
test = encode(test, mappings)
x_test, y_test = to_arrays(test)

print("Generating model scores...")
lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_PATH))
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model(str(XGBOOST_PATH))

probabilities = {
    "LightGBM": lightgbm_model.predict(x_test),
    "XGBoost": xgboost_model.predict_proba(x_test)[:, 1],
}

rows = []
for model_name, scores in probabilities.items():
    for budget in ALERT_BUDGETS:
        rows.append({
            "model": model_name,
            **evaluate_budget(y_test, scores, budget),
        })

results = pl.DataFrame(rows)
results.write_csv(OUTPUT_PATH)

report_lines = [
    "Fixed alert-budget analysis: LightGBM versus XGBoost",
    "=" * 60,
    "The same chronological test transactions were ranked by both models.",
    "Only the top-k scores were treated as investigation alerts.",
    f"Test positives: {int(y_test.sum()):,}",
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
