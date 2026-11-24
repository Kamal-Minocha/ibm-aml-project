"""
Train an XGBoost CPU model on the graph-enhanced feature table.

The split and features match train_baseline.py --graph so the model
comparison is controlled.
"""

from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR / "processed" / "transactions_with_graph_features.parquet"
)
MODEL_PATH = BASE_DIR / "processed" / "xgboost_graph_model.json"
METRICS_PATH = BASE_DIR / "processed" / "xgboost_graph_metrics.txt"
IMPORTANCE_PATH = (
    BASE_DIR / "processed" / "xgboost_graph_feature_importance.csv"
)

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
TARGET = "is_laundering"
REQUIRED_COLUMNS = [
    "timestamp", TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES
]


def category_mappings(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
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


def arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return (
        frame.select(features).to_numpy(),
        frame.get_column(TARGET).to_numpy().astype(np.int8),
    )


def report(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> list[str]:
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return [
        f"{name} rows: {len(labels):,}",
        f"{name} positives: {int(labels.sum()):,}",
        f"{name} PR-AUC: {average_precision_score(labels, probabilities):.6f}",
        f"{name} ROC-AUC: {roc_auc_score(labels, probabilities):.6f}",
        f"{name} precision at threshold {threshold:.6f}: "
        f"{precision_score(labels, predictions, zero_division=0):.6f}",
        f"{name} recall at threshold {threshold:.6f}: "
        f"{recall_score(labels, predictions, zero_division=0):.6f}",
        f"{name} F1 at threshold {threshold:.6f}: "
        f"{f1_score(labels, predictions, zero_division=0):.6f}",
        f"{name} confusion matrix [[TN, FP], [FN, TP]]: {matrix.tolist()}",
    ]


if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

print("Loading graph-enhanced features...")
lazy = pl.scan_parquet(INPUT_PATH)
missing = [
    column for column in REQUIRED_COLUMNS
    if column not in lazy.collect_schema().names()
]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

frame = (
    lazy.select(REQUIRED_COLUMNS)
    .filter(pl.col(TARGET).is_in([0, 1]))
    .collect(engine="streaming")
)

boundaries = (
    frame.select([
        pl.col("timestamp").quantile(0.70).alias("train_end"),
        pl.col("timestamp").quantile(0.85).alias("validation_end"),
    ])
    .row(0)
)
train_end, validation_end = boundaries
train = frame.filter(pl.col("timestamp") < train_end)
validation = frame.filter(
    (pl.col("timestamp") >= train_end)
    & (pl.col("timestamp") < validation_end)
)
test = frame.filter(pl.col("timestamp") >= validation_end)

print(
    f"Split sizes: train={train.height:,}, "
    f"validation={validation.height:,}, test={test.height:,}"
)
print(
    f"Positive counts: train={train[TARGET].sum():,}, "
    f"validation={validation[TARGET].sum():,}, "
    f"test={test[TARGET].sum():,}"
)

mappings = category_mappings(train)
train = encode(train, mappings)
validation = encode(validation, mappings)
test = encode(test, mappings)
x_train, y_train = arrays(train)
x_validation, y_validation = arrays(validation)
x_test, y_test = arrays(test)

positive_count = int(y_train.sum())
negative_count = len(y_train) - positive_count

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    device="cpu",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=8,
    early_stopping_rounds=50,
)

print("Training CPU XGBoost...")
model.fit(
    x_train,
    y_train,
    eval_set=[(x_validation, y_validation)],
    verbose=False,
)

validation_probabilities = model.predict_proba(x_validation)[:, 1]
test_probabilities = model.predict_proba(x_test)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
validation_f1 = [
    f1_score(
        y_validation,
        (validation_probabilities >= threshold).astype(np.int8),
        zero_division=0,
    )
    for threshold in thresholds
]
selected_threshold = float(thresholds[int(np.argmax(validation_f1))])

metrics = [
    "HI-Small graph-enhanced CPU XGBoost",
    "=" * 55,
    "Features: transaction, account behavior, and network features",
    "Split: chronological 70% train, 15% validation, 15% test",
    f"Best iteration: {model.best_iteration}",
    f"Selected validation threshold: {selected_threshold:.6f}",
    "",
]
metrics.extend(
    report(
        "Validation",
        y_validation,
        validation_probabilities,
        selected_threshold,
    )
)
metrics.append("")
metrics.extend(
    report("Test", y_test, test_probabilities, selected_threshold)
)

model.save_model(MODEL_PATH)
METRICS_PATH.write_text("\n".join(metrics), encoding="utf-8")
importance = pl.DataFrame({
    "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    "importance": model.feature_importances_,
}).sort("importance", descending=True)
importance.write_csv(IMPORTANCE_PATH)

print("\n".join(metrics))
print(f"Saved model: {MODEL_PATH}")
print(f"Saved metrics: {METRICS_PATH}")
print(f"Saved feature importance: {IMPORTANCE_PATH}")
