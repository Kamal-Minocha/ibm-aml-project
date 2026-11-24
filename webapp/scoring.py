"""
Scoring logic for the AML demo web app.

This module has no Streamlit imports so it can be unit-tested and reused.
It reproduces the exact inference path of the training pipeline
(``train_xgboost.py`` / ``predict_batch.py``):

* the same 40 features, in the same order;
* categorical values mapped to the integer codes learned on the training
  split; nulls and unseen values become ``-1``, exactly what the pipeline's
  ``replace_strict(mapping, default=-1)`` does. (``"__MISSING__"`` is only
  used when the mapping is built, so its code never reaches the model.);
* predictions limited to the trees kept by early stopping
  (``best_iteration``), so scores match ``XGBClassifier.predict_proba``.

All artifacts it reads live in ``webapp/model`` and are produced by
``webapp/build_assets.py``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = Path(__file__).resolve().parent / "model"

MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY_CODE = -1
LABEL_COLUMN = "is_laundering"
ID_COLUMNS = ("transaction_id", "timestamp", "from_account", "to_account")

# Hard cap on uploaded batch size. Keeps memory bounded on a small hosted
# instance (Streamlit Community Cloud gives roughly 1 GB of RAM).
MAX_BATCH_ROWS = 200_000


class BatchValidationError(ValueError):
    """Raised when an input batch cannot be scored."""


@dataclass(frozen=True)
class ModelBundle:
    booster: xgb.Booster
    iteration_range: tuple[int, int]
    features: list[str]
    categorical_features: list[str]
    mappings: dict[str, dict[str, int]]
    alert_budget: int
    defaults: dict[str, object]
    reference: dict[str, object]

    @property
    def numeric_features(self) -> list[str]:
        return [f for f in self.features if f not in self.categorical_features]

    def percentile(self, score: float) -> float:
        """Share (0-1) of held-out test transactions that scored lower."""
        levels = np.asarray(self.reference["quantile_levels"], dtype=float)
        values = np.asarray(self.reference["quantile_values"], dtype=float)
        return float(np.interp(score, values, levels))

    @property
    def top_k_cutoff(self) -> float:
        """Score of the k-th highest test transaction (k = alert budget)."""
        return float(self.reference["top_k_cutoff"])


@dataclass
class PreparedBatch:
    matrix: np.ndarray
    labels: np.ndarray | None
    warnings: list[str] = field(default_factory=list)


def read_table(data: bytes, name: str) -> pd.DataFrame:
    """Parse an uploaded CSV or Parquet file.

    CSVs are read as UTF-8 (with or without BOM), falling back to Windows-1252.
    Semicolon-separated files, as some regional Excel settings export, are
    detected. Any parse failure becomes a BatchValidationError.
    """
    try:
        if name.lower().endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                frame = pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        if frame.shape[1] == 1 and ";" in str(frame.columns[0]):
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=";",
                                low_memory=False)
        return frame
    except Exception as error:  # pandas/pyarrow raise many unrelated types
        raise BatchValidationError(f"Could not read `{name}` as a table: {error}") from error


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(model_dir: Path = MODEL_DIR) -> ModelBundle:
    config = _read_json(model_dir / "model_config.json")
    mappings = _read_json(model_dir / "category_mappings.json")
    defaults = _read_json(model_dir / "feature_defaults.json")
    reference_path = model_dir / "score_reference.json"
    # Absent only while build_assets.py is generating it.
    reference = _read_json(reference_path) if reference_path.exists() else {}

    booster = xgb.Booster()
    booster.load_model(model_dir / "xgboost_graph_model.json")

    features = list(config["features"])
    categorical = list(config["categorical_features"])

    if booster.num_features() != len(features):
        raise ValueError(
            f"Model expects {booster.num_features()} features but the config "
            f"lists {len(features)}."
        )
    missing_maps = [c for c in categorical if c not in mappings]
    if missing_maps:
        raise ValueError(f"No category mapping for: {missing_maps}")

    # Early stopping keeps every tree in the file but records the best one.
    # predict_proba() honours it; Booster.predict() must be told explicitly.
    best = booster.attr("best_iteration")
    last = int(best) + 1 if best is not None else booster.num_boosted_rounds()

    return ModelBundle(
        booster=booster,
        iteration_range=(0, last),
        features=features,
        categorical_features=categorical,
        mappings=mappings,
        alert_budget=int(config["alert_budget"]),
        defaults=defaults,
        reference=reference,
    )


def prepare_batch(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    max_rows: int | None = MAX_BATCH_ROWS,
) -> PreparedBatch:
    """Validate a model-ready table and convert it to the model's matrix.

    ``max_rows=None`` disables the size cap (used for offline batch jobs).
    """
    if len(frame) == 0:
        raise BatchValidationError("The file contains no rows.")
    if max_rows is not None and len(frame) > max_rows:
        raise BatchValidationError(
            f"The file has {len(frame):,} rows; the limit is "
            f"{max_rows:,}. Split it into smaller files."
        )

    missing = [c for c in bundle.features if c not in frame.columns]
    if missing:
        raise BatchValidationError(
            f"The file is missing {len(missing)} of the {len(bundle.features)} "
            "required feature columns: " + ", ".join(missing)
        )

    warnings: list[str] = []
    columns: dict[str, np.ndarray] = {}
    bad_numeric: list[str] = []

    for column in bundle.features:
        source = frame[column]
        if column in bundle.categorical_features:
            text = source.astype("string")
            codes = text.map(bundle.mappings[column])
            unseen = codes.isna() & text.notna()
            if unseen.any():
                examples = sorted(set(text[unseen].head(50)))[:3]
                warnings.append(
                    f"{int(unseen.sum()):,} row(s) have a `{column}` value not "
                    f"seen in training (e.g. {examples}); they were encoded as "
                    "'unknown', like blank values."
                )
            columns[column] = (
                codes.fillna(UNKNOWN_CATEGORY_CODE).to_numpy(dtype=np.float32)
            )
        else:
            numeric = pd.to_numeric(source, errors="coerce")
            # Values beyond float32 range (~3.4e38) overflow to inf in the cast.
            with np.errstate(over="ignore"):
                values = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
            invalid = (numeric.isna() & source.notna()).to_numpy() | np.isinf(values)
            if invalid.any():
                bad_numeric.append(f"{column} ({int(invalid.sum()):,} rows)")
            columns[column] = values

    if bad_numeric:
        raise BatchValidationError(
            "These columns must hold ordinary numbers but contain text, "
            "infinite or out-of-range values: " + ", ".join(bad_numeric)
        )

    matrix = np.column_stack([columns[c] for c in bundle.features])

    labels = None
    if LABEL_COLUMN in frame.columns:
        raw = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
        if raw.isin([0, 1]).all():
            labels = raw.to_numpy(dtype=np.int8)
        else:
            warnings.append(
                f"`{LABEL_COLUMN}` is present but not entirely 0/1, so "
                "precision and recall are not shown."
            )

    return PreparedBatch(matrix=matrix, labels=labels, warnings=warnings)


def score(bundle: ModelBundle, matrix: np.ndarray) -> np.ndarray:
    """Model risk score in [0, 1] for each row."""
    return bundle.booster.predict(
        xgb.DMatrix(matrix, missing=np.nan),
        iteration_range=bundle.iteration_range,
    )


def contributions(
    bundle: ModelBundle, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature TreeSHAP contributions (log-odds) and the bias term.

    For each row ``bias + contributions.sum()`` equals the log-odds of
    ``score()``, so the explanation accounts for the score exactly.
    """
    values = bundle.booster.predict(
        xgb.DMatrix(matrix, missing=np.nan),
        pred_contribs=True,
        iteration_range=bundle.iteration_range,
    )
    return values[:, :-1], values[:, -1]


def rank_alerts(scores: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rank, is_alert): rank 1 is the highest score; the top
    ``budget`` rows are alerts. Ties keep input order."""
    if budget < 1:
        raise ValueError("budget must be at least 1")
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(scores), dtype=np.int64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank, rank <= budget


def alert_metrics(labels: np.ndarray, is_alert: np.ndarray) -> dict[str, float]:
    true_positives = int(((labels == 1) & is_alert).sum())
    alerts = int(is_alert.sum())
    positives = int((labels == 1).sum())
    return {
        "alerts": alerts,
        "true_positives": true_positives,
        "false_positives": alerts - true_positives,
        "positives": positives,
        "precision": true_positives / alerts if alerts else 0.0,
        "recall": true_positives / positives if positives else 0.0,
    }


# ---------------------------------------------------------------------------
# Feature playground
# ---------------------------------------------------------------------------

HISTORY_FIELDS = (
    "outgoing_count_24h",
    "unique_recipients_24h",
    "incoming_count_24h",
    "unique_senders_24h",
    "tx_count_1h",
    "seconds_since_activity",
)
DATASET_YEAR, DATASET_MONTH = 2022, 9


def seconds_since_range(tx_count_1h: float, count_24h: float) -> tuple[int, int | None]:
    """Allowed ``*_seconds_since_activity`` range given the activity counts.

    In the pipeline output, activity in the last hour means the previous
    event was at most 3,600 s ago; activity only earlier in the 24 h window
    means 3,600-86,400 s; no 24 h activity means more than a day ago. (Exact
    zero, a same-timestamp event, also occurs; the playground doesn't model it.)
    """
    if tx_count_1h > 0:
        return 0, 3_600
    if count_24h > 0:
        return 3_601, 86_400
    return 86_401, None


def build_playground_row(
    bundle: ModelBundle, overrides: dict[str, object]
) -> pd.DataFrame:
    """Build one feature row from training defaults plus user overrides,
    then enforce relationships that hold in the real pipeline output, so
    the model isn't asked about a transaction the pipeline could not produce.

    Enforced (each holds for 100% of the 5.08M HI-Small rows):
    * graph fan-out / fan-in / edge counts equal their behaviour-feature
      counterparts;
    * ``sender_received_money_24h`` = sender received anything in 24h, and
      ``rapid_forwarding_indicator`` = received money AND sent earlier in 24h;
    * distinct counterparties <= transaction counts; 1h activity <= 24h;
    * seconds since previous activity consistent with the counts
      (``seconds_since_range``);
    * a self-transfer is same-bank, same entity type, and the sender's and
      receiver's histories are identical (they are the same account);
    * same currency implies equal amounts; weekday matches day of month.
    """
    row: dict[str, object] = dict(bundle.defaults)
    row.update(overrides)

    def num(name: str) -> float:
        return float(row[name])

    row["same_currency"] = int(row["payment_currency"] == row["receiving_currency"])
    if row["same_currency"]:
        row["amount_received"] = row["amount_paid"]
    if int(row["same_account"]):
        row["same_bank"] = 1
        row["receiver_entity_type"] = row["sender_entity_type"]
        for name in HISTORY_FIELDS:
            row[f"receiver_{name}"] = row[f"sender_{name}"]
    row["transaction_weekday"] = date(
        DATASET_YEAR, DATASET_MONTH, int(num("transaction_day"))
    ).isoweekday()

    for side in ("sender", "receiver"):
        out_count = max(0.0, num(f"{side}_outgoing_count_24h"))
        in_count = max(0.0, num(f"{side}_incoming_count_24h"))
        row[f"{side}_outgoing_count_24h"] = out_count
        row[f"{side}_incoming_count_24h"] = in_count
        row[f"{side}_unique_recipients_24h"] = min(
            num(f"{side}_unique_recipients_24h"), out_count
        )
        row[f"{side}_unique_senders_24h"] = min(
            num(f"{side}_unique_senders_24h"), in_count
        )
        row[f"{side}_tx_count_1h"] = min(
            num(f"{side}_tx_count_1h"), out_count + in_count
        )
        low, high = seconds_since_range(
            row[f"{side}_tx_count_1h"], out_count + in_count
        )
        seconds = max(num(f"{side}_seconds_since_activity"), low)
        row[f"{side}_seconds_since_activity"] = (
            seconds if high is None else min(seconds, high)
        )
        # Earlier transactions are assumed to be about the size of this one.
        typical = num("amount_paid")
        row[f"{side}_outgoing_amount_24h"] = out_count * typical
        row[f"{side}_incoming_amount_24h"] = in_count * typical

    row["sender_historical_fan_out_24h"] = row["sender_unique_recipients_24h"]
    row["sender_historical_edge_count_24h"] = row["sender_outgoing_count_24h"]
    row["receiver_historical_fan_in_24h"] = row["receiver_unique_senders_24h"]
    row["receiver_historical_edge_count_24h"] = row["receiver_incoming_count_24h"]
    row["sender_received_money_24h"] = int(num("sender_incoming_count_24h") > 0)
    row["rapid_forwarding_indicator"] = int(
        row["sender_received_money_24h"] == 1
        and num("sender_outgoing_count_24h") > 0
    )
    row["paid_amount_nonpositive"] = int(num("amount_paid") <= 0)
    row["received_amount_nonpositive"] = int(num("amount_received") <= 0)
    row["sender_metadata_matched"] = 1
    row["receiver_metadata_matched"] = 1

    return pd.DataFrame([{name: row[name] for name in bundle.features}])
