"""
Stage 3: Train and evaluate a basic transaction-level LightGBM baseline.

Input:
    processed/transactions_with_accounts.parquet

Outputs:
    processed/baseline_model.txt
    processed/baseline_metrics.txt
    processed/baseline_feature_importance.csv

This baseline deliberately uses only transaction-level features and static
account metadata. Historical and graph features are added in later stages.
"""

from pathlib import Path
import argparse

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--behavior",
    action="store_true",
    help="Train using the historical behavior feature table.",
)
parser.add_argument(
    "--graph",
    action="store_true",
    help="Train using historical behavior and network features.",
)
args = parser.parse_args()
if args.graph and not args.behavior:
    args.behavior = True

BASE_DIR = Path(__file__).resolve().parent
if args.graph:
    INPUT_PATH = (
        BASE_DIR / "processed" / "transactions_with_graph_features.parquet"
    )
    MODEL_PATH = BASE_DIR / "processed" / "graph_model.txt"
    METRICS_PATH = BASE_DIR / "processed" / "graph_metrics.txt"
    IMPORTANCE_PATH = (
        BASE_DIR / "processed" / "graph_feature_importance.csv"
    )
elif args.behavior:
    INPUT_PATH = (
        BASE_DIR / "processed" / "transactions_with_behavior.parquet"
    )
    MODEL_PATH = BASE_DIR / "processed" / "behavior_model.txt"
    METRICS_PATH = BASE_DIR / "processed" / "behavior_metrics.txt"
    IMPORTANCE_PATH = (
        BASE_DIR / "processed" / "behavior_feature_importance.csv"
    )
else:
    INPUT_PATH = (
        BASE_DIR / "processed" / "transactions_with_accounts.parquet"
    )
    MODEL_PATH = BASE_DIR / "processed" / "baseline_model.txt"
    METRICS_PATH = BASE_DIR / "processed" / "baseline_metrics.txt"
    IMPORTANCE_PATH = (
        BASE_DIR / "processed" / "baseline_feature_importance.csv"
    )

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")


NUMERIC_FEATURES = [
    "amount_paid",
    "amount_received",
    "transaction_hour",
    "transaction_weekday",
    "transaction_day",
    "transaction_month",
    "same_bank",
    "same_account",
    "same_currency",
    "received_amount_nonpositive",
    "paid_amount_nonpositive",
    "sender_metadata_matched",
    "receiver_metadata_matched",
]

BEHAVIOR_FEATURES = [
    "sender_tx_count_1h",
    "sender_outgoing_count_24h",
    "sender_incoming_count_24h",
    "sender_outgoing_amount_24h",
    "sender_incoming_amount_24h",
    "sender_unique_recipients_24h",
    "sender_unique_senders_24h",
    "sender_seconds_since_activity",
    "receiver_tx_count_1h",
    "receiver_outgoing_count_24h",
    "receiver_incoming_count_24h",
    "receiver_outgoing_amount_24h",
    "receiver_incoming_amount_24h",
    "receiver_unique_recipients_24h",
    "receiver_unique_senders_24h",
    "receiver_seconds_since_activity",
]

if args.behavior:
    NUMERIC_FEATURES.extend(BEHAVIOR_FEATURES)

GRAPH_FEATURES = [
    "sender_historical_fan_out_24h",
    "sender_historical_edge_count_24h",
    "receiver_historical_fan_in_24h",
    "receiver_historical_edge_count_24h",
    "sender_received_money_24h",
    "rapid_forwarding_indicator",
]

if args.graph:
    NUMERIC_FEATURES.extend(GRAPH_FEATURES)

CATEGORICAL_FEATURES = [
    "payment_format",
    "payment_currency",
    "receiving_currency",
    "sender_entity_type",
    "receiver_entity_type",
]

TARGET = "is_laundering"
REQUIRED_COLUMNS = [
    "timestamp",
    TARGET,
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]


def get_time_boundaries(frame: pl.LazyFrame) -> tuple[object, object]:
    boundaries = (
        frame
        .select([
            pl.col("timestamp").quantile(0.70).alias("train_end"),
            pl.col("timestamp").quantile(0.85).alias("validation_end"),
        ])
        .collect()
        .row(0)
    )
    return boundaries[0], boundaries[1]


def encode_categories(
    frame: pl.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> pl.DataFrame:
    expressions = []
    for column in CATEGORICAL_FEATURES:
        expressions.append(
            pl.col(column)
            .cast(pl.String)
            .replace_strict(
                mappings[column],
                default=-1,
            )
            .cast(pl.Int32)
            .alias(column)
        )
    return frame.with_columns(expressions)


def make_category_mappings(
    frame: pl.DataFrame,
) -> dict[str, dict[str, int]]:
    mappings = {}
    for column in CATEGORICAL_FEATURES:
        values = (
            frame
            .select(pl.col(column).cast(pl.String).fill_null("__MISSING__"))
            .unique()
            .sort(column)
            .get_column(column)
            .to_list()
        )
        mappings[column] = {
            value: index for index, value in enumerate(values)
        }
    return mappings


def evaluate(
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


print("Loading baseline features...")
lazy_frame = pl.scan_parquet(INPUT_PATH)
available_columns = lazy_frame.collect_schema().names()
missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in available_columns
]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

frame = (
    lazy_frame
    .select(REQUIRED_COLUMNS)
    .filter(pl.col(TARGET).is_in([0, 1]))
    .collect(engine="streaming")
)

train_end, validation_end = get_time_boundaries(frame.lazy())
print(f"Training ends before: {train_end}")
print(f"Validation ends before: {validation_end}")

train = frame.filter(pl.col("timestamp") < train_end)
validation = frame.filter(
    (pl.col("timestamp") >= train_end)
    & (pl.col("timestamp") < validation_end)
)
test = frame.filter(pl.col("timestamp") >= validation_end)

if train.is_empty() or validation.is_empty() or test.is_empty():
    raise RuntimeError("One chronological split is empty.")

print(
    f"Split sizes: train={train.height:,}, "
    f"validation={validation.height:,}, test={test.height:,}"
)
print(
    f"Positive counts: train={train[TARGET].sum():,}, "
    f"validation={validation[TARGET].sum():,}, "
    f"test={test[TARGET].sum():,}"
)

mappings = make_category_mappings(train)
train = encode_categories(train, mappings)
validation = encode_categories(validation, mappings)
test = encode_categories(test, mappings)

feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
categorical_indices = [
    feature_columns.index(column)
    for column in CATEGORICAL_FEATURES
]

def arrays(data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = data.select(feature_columns).to_numpy()
    y = data.get_column(TARGET).to_numpy().astype(np.int8)
    return x, y


x_train, y_train = arrays(train)
x_validation, y_validation = arrays(validation)
x_test, y_test = arrays(test)

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)
if positive_count == 0:
    raise RuntimeError("The training split contains no positive labels.")

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    random_state=42,
    n_jobs=-1,
)

print("Training LightGBM baseline...")
model.fit(
    x_train,
    y_train,
    eval_set=[(x_validation, y_validation)],
    eval_metric="average_precision",
    categorical_feature=categorical_indices,
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100),
    ],
)

validation_probabilities = model.predict_proba(x_validation)[:, 1]
test_probabilities = model.predict_proba(x_test)[:, 1]

# Choose a validation threshold that maximizes F1, then apply it unchanged
# to the test set. This keeps test evaluation independent of threshold tuning.
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
    (
        "HI-Small transaction-level LightGBM with historical behavior "
        "and network features"
        if args.graph
        else (
            "HI-Small transaction-level LightGBM with historical behavior"
            if args.behavior
            else "HI-Small transaction-level LightGBM baseline"
        )
    ),
    "=" * 55,
    (
        (
            "Features: transaction fields, static account metadata, "
            "historical behavior, and network features"
            if args.graph
            else (
                "Features: transaction fields, static account metadata, "
                "and historical behavior"
                if args.behavior
                else (
                    "Features: transaction fields and static account "
                    "metadata only"
                )
            )
        )
    ),
    "Split: chronological 70% train, 15% validation, 15% test",
    f"Selected validation threshold: {selected_threshold:.6f}",
    "",
]
metrics.extend(
    evaluate(
        "Validation",
        y_validation,
        validation_probabilities,
        selected_threshold,
    )
)
metrics.append("")
metrics.extend(
    evaluate("Test", y_test, test_probabilities, selected_threshold)
)

MODEL_PATH.write_text(model.booster_.model_to_string(), encoding="utf-8")
METRICS_PATH.write_text("\n".join(metrics), encoding="utf-8")

importance = pl.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_,
}).sort("importance", descending=True)
importance.write_csv(IMPORTANCE_PATH)

print("\n".join(metrics))
print(f"Saved model: {MODEL_PATH}")
print(f"Saved metrics: {METRICS_PATH}")
print(f"Saved feature importance: {IMPORTANCE_PATH}")
