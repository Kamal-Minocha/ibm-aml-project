"""
Rebuild everything the web app needs from the main pipeline's outputs.

Run from the project root after ``finalize_xgboost.py`` has been run:

    .venv\\Scripts\\python.exe webapp\\build_assets.py

Reads (all git-ignored, produced by the pipeline):
    processed/final_xgboost_config.json
    processed/xgboost_graph_model.json
    processed/transactions_with_graph_features.parquet
    processed_medium/medium_model_ready_combined.parquet   (optional)

Writes (small, committed so the hosted app can run without the big data):
    webapp/model/xgboost_graph_model.json   copy of the trained model
    webapp/model/model_config.json          feature list, alert budget, split
    webapp/model/category_mappings.json     training-split category codes
    webapp/model/feature_defaults.json      training-split medians / modes
    webapp/model/score_reference.json       test-period score distribution
                                            and recomputed headline metrics
    webapp/sample_data/*.csv                labelled, model-ready test batches

Needs polars and scikit-learn (already in the project's root requirements).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
sys.path.insert(0, str(WEBAPP_DIR))

import scoring  # noqa: E402

PROCESSED = ROOT / "processed"
SOURCE_CONFIG = PROCESSED / "final_xgboost_config.json"
SOURCE_MODEL = PROCESSED / "xgboost_graph_model.json"
FEATURE_TABLE = PROCESSED / "transactions_with_graph_features.parquet"
MEDIUM_SAMPLE = ROOT / "processed_medium" / "medium_model_ready_combined.parquet"

MODEL_DIR = WEBAPP_DIR / "model"
SAMPLE_DIR = WEBAPP_DIR / "sample_data"

SAMPLE_SIZE = 25
QUICK_DEMO_PER_CLASS = 10
CSV_DATETIME = "%Y-%m-%d %H:%M:%S"

# Dense quantile grid in the extreme tail, where alerts live.
QUANTILE_LEVELS = np.unique(np.concatenate([
    np.linspace(0.0, 0.99, 991),
    np.linspace(0.99, 0.999, 901),
    np.linspace(0.999, 1.0, 1001),
]))


def write_json(path: Path, payload: dict) -> None:
    """Every output goes via a temp file, so a failed build never leaves a
    half-written file behind."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"  wrote {path.relative_to(ROOT)}")


def write_csv(frame: pl.DataFrame, name: str) -> None:
    path = SAMPLE_DIR / name
    temporary = path.with_suffix(".tmp")
    frame.write_csv(temporary, datetime_format=CSV_DATETIME)
    temporary.replace(path)
    print(f"  wrote {path.relative_to(ROOT)} ({frame.height} rows)")


def main() -> None:
    for required in (SOURCE_CONFIG, SOURCE_MODEL, FEATURE_TABLE):
        if not required.exists():
            raise SystemExit(f"Missing pipeline output: {required}")

    MODEL_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)

    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    features: list[str] = config["features"]
    categorical: list[str] = config["categorical_features"]
    numeric = [f for f in features if f not in categorical]
    train_end = datetime.fromisoformat(config["train_end"])
    validation_end = datetime.fromisoformat(config["validation_end"])
    budget = int(config["alert_budget"])

    table = pl.scan_parquet(FEATURE_TABLE)
    train = table.filter(pl.col("timestamp") < train_end)
    test = table.filter(pl.col("timestamp") >= validation_end)

    print("Model files")
    model_copy = MODEL_DIR / "xgboost_graph_model.json"
    shutil.copyfile(SOURCE_MODEL, model_copy.with_suffix(".tmp"))
    model_copy.with_suffix(".tmp").replace(model_copy)
    print("  copied xgboost_graph_model.json")
    write_json(MODEL_DIR / "model_config.json", {
        "model": config["model"],
        "features": features,
        "categorical_features": categorical,
        "alert_budget": budget,
        "train_end": config["train_end"],
        "validation_end": config["validation_end"],
    })

    print("Training-split encodings and defaults")
    train_categories = train.select(categorical).collect(engine="streaming")
    mappings = {}
    modes = {}
    for column in categorical:
        # Same construction as train_xgboost.category_mappings(): nulls get a
        # "__MISSING__" slot, though encoding still sends real nulls to -1.
        values = train_categories.get_column(column).cast(pl.String).fill_null(
            scoring.MISSING_CATEGORY
        )
        ordered = values.unique().sort().to_list()
        mappings[column] = {value: index for index, value in enumerate(ordered)}
        modes[column] = values.drop_nulls().filter(
            values != scoring.MISSING_CATEGORY
        ).mode().sort().item(0)
    write_json(MODEL_DIR / "category_mappings.json", mappings)

    medians = train.select(
        [pl.col(c).cast(pl.Float64).median() for c in numeric]
    ).collect(engine="streaming").row(0, named=True)
    write_json(MODEL_DIR / "feature_defaults.json", {**medians, **modes})

    print("Scoring the held-out test period")
    # Scoring doesn't use score_reference.json, so the existing one (if any)
    # stays in place until the new one is written.
    bundle = scoring.load_bundle(MODEL_DIR)
    test_frame = test.select([*features, "is_laundering"]).collect(
        engine="streaming"
    ).to_pandas()
    prepared = scoring.prepare_batch(test_frame, bundle, max_rows=None)
    scores = scoring.score(bundle, prepared.matrix)
    labels = prepared.labels
    _, is_alert = scoring.rank_alerts(scores, budget)
    at_budget = scoring.alert_metrics(labels, is_alert)
    write_json(MODEL_DIR / "score_reference.json", {
        "description": "Scores of the held-out HI-Small test period "
                       f"(timestamp >= {config['validation_end']}).",
        "test_rows": int(len(scores)),
        "test_positives": int(labels.sum()),
        "alert_budget": budget,
        "top_k_cutoff": float(np.sort(scores)[::-1][budget - 1]),
        "metrics": {
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "precision_at_budget": at_budget["precision"],
            "recall_at_budget": at_budget["recall"],
            "true_positives_at_budget": at_budget["true_positives"],
        },
        "quantile_levels": [round(float(x), 6) for x in QUANTILE_LEVELS],
        "quantile_values": [
            float(x) for x in np.quantile(scores, QUANTILE_LEVELS)
        ],
    })
    print(f"  PR-AUC {average_precision_score(labels, scores):.6f}, "
          f"precision@{budget} {at_budget['precision']:.4f}, "
          f"recall@{budget} {at_budget['recall']:.4f}")

    print("Sample datasets")
    ordered_test = test.sort(["timestamp", "from_account", "to_account"])
    laundering = ordered_test.filter(pl.col("is_laundering") == 1).head(
        SAMPLE_SIZE
    ).collect(engine="streaming")
    normal = ordered_test.filter(pl.col("is_laundering") == 0).head(
        SAMPLE_SIZE
    ).collect(engine="streaming")
    write_csv(laundering, "small_laundering_examples.csv")
    write_csv(normal, "small_not_laundering_examples.csv")
    quick = pl.concat([
        laundering.head(QUICK_DEMO_PER_CLASS), normal.head(QUICK_DEMO_PER_CLASS)
    ]).sample(fraction=1.0, shuffle=True, seed=42)
    write_csv(quick, "quick_demo_mixed_20.csv")

    if MEDIUM_SAMPLE.exists():
        write_csv(pl.read_parquet(MEDIUM_SAMPLE), "medium_mixed_examples.csv")
    else:
        print(f"  skipped medium sample ({MEDIUM_SAMPLE} not found)")

    print("Done.")


if __name__ == "__main__":
    main()
