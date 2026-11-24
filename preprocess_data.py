"""
HI-Small Money Laundering Dataset Preprocessing
================================================

Designed for Windows and large CSV datasets.

Features:
- Multithreaded CPU processing using Polars
- Memory-efficient CSV processing
- Preserves bank/account IDs as strings
- Renames duplicate Account columns
- Parses timestamps
- Handles invalid timestamps and missing values
- Creates basic transaction features
- Saves cleaned data as compressed Parquet
- Generates class counts and an audit report

Install:
    py -m pip install -U polars pyarrow

Run:
    py preprocess_data.py

Custom paths:
    py preprocess_data.py --input "D:\\data\\HI-Small_Trans.csv"
"""

import os
import time
import argparse
from pathlib import Path

# Allow Polars to use all logical CPU cores available to it.
# Set this before importing Polars.
os.environ.setdefault(
    "POLARS_MAX_THREADS",
    str(os.cpu_count() or 1)
)

import polars as pl


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Preprocess the HI-Small transaction dataset"
)

parser.add_argument(
    "--input",
    type=str,
    default=str(BASE_DIR / "HI-Small_Trans.csv"),
    help="Path to the transaction CSV"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default=str(BASE_DIR / "processed"),
    help="Directory for processed files"
)

args = parser.parse_args()

INPUT_FILE = Path(args.input).resolve()
OUTPUT_DIR = Path(args.output_dir).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "transactions_clean.parquet"
AUDIT_FILE = OUTPUT_DIR / "audit_report.txt"
CLASS_FILE = OUTPUT_DIR / "class_counts.csv"


# ---------------------------------------------------------
# 2. CHECK INPUT
# ---------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Place HI-Small_Trans.csv beside this script "
        "or provide the correct --input path."
    )

print("=" * 65)
print("HI-SMALL DATASET PREPROCESSING")
print("=" * 65)

print(f"Input file: {INPUT_FILE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Available logical CPU cores: {os.cpu_count()}")
print(f"Polars version: {pl.__version__}")

print(
    "\nProcessing will use Polars' multithreaded CPU engine."
    "\nGPU acceleration is not normally beneficial for CSV cleaning."
)

start_time = time.perf_counter()


# ---------------------------------------------------------
# 3. DEFINE COLUMN NAMES AND DATA TYPES
# ---------------------------------------------------------

# The source CSV has two columns named "Account".
# We rename them by position to distinguish sender and receiver.

COLUMN_NAMES = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# Keep IDs as strings so leading zeros are preserved.
# Numeric transaction amounts remain numeric.

SCHEMA = {
    "timestamp": pl.String,
    "from_bank": pl.String,
    "from_account": pl.String,
    "to_bank": pl.String,
    "to_account": pl.String,
    "amount_received": pl.Float64,
    "receiving_currency": pl.String,
    "amount_paid": pl.Float64,
    "payment_currency": pl.String,
    "payment_format": pl.String,
    "is_laundering": pl.Int8,
}


# ---------------------------------------------------------
# 4. SCAN CSV WITHOUT LOADING IT ALL INTO MEMORY
# ---------------------------------------------------------

print("\n[1/5] Scanning CSV...")

df = pl.scan_csv(
    INPUT_FILE,
    has_header=True,
    new_columns=COLUMN_NAMES,
    schema_overrides=SCHEMA,
    infer_schema_length=10000,
    null_values=["", "NA", "N/A", "null", "NULL"],
    ignore_errors=False,
    low_memory=True,
)

print("CSV scan configured.")


# ---------------------------------------------------------
# 5. CLEAN STRINGS AND PARSE TIMESTAMPS
# ---------------------------------------------------------

print("\n[2/5] Cleaning columns...")

string_columns = [
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]

df = df.with_columns([
    pl.col(column)
    .str.strip_chars()
    .alias(column)
    for column in string_columns
])

# Expected timestamp format:
# 2022/09/01 00:20

df = df.with_columns([
    pl.col("timestamp")
    .str.strptime(
        pl.Datetime,
        format="%Y/%m/%d %H:%M",
        strict=False,
    )
    .alias("timestamp"),

    pl.col("is_laundering")
    .cast(pl.Int8, strict=False)
    .alias("is_laundering"),
])


# ---------------------------------------------------------
# 6. FEATURE ENGINEERING
# ---------------------------------------------------------

print("\n[3/5] Creating basic features...")

df = df.with_columns([
    # Time-based features
    pl.col("timestamp")
    .dt.hour()
    .cast(pl.Int8)
    .alias("transaction_hour"),

    pl.col("timestamp")
    .dt.weekday()
    .cast(pl.Int8)
    .alias("transaction_weekday"),

    pl.col("timestamp")
    .dt.day()
    .cast(pl.Int8)
    .alias("transaction_day"),

    pl.col("timestamp")
    .dt.month()
    .cast(pl.Int8)
    .alias("transaction_month"),

    # Bank and account relationships
    (
        pl.col("from_bank") == pl.col("to_bank")
    )
    .fill_null(False)
    .cast(pl.Int8)
    .alias("same_bank"),

    (
        pl.col("from_account") == pl.col("to_account")
    )
    .fill_null(False)
    .cast(pl.Int8)
    .alias("same_account"),

    (
        pl.col("receiving_currency")
        == pl.col("payment_currency")
    )
    .fill_null(False)
    .cast(pl.Int8)
    .alias("same_currency"),

    # Basic amount validation indicators
    (
        pl.col("amount_received") <= 0
    )
    .fill_null(False)
    .cast(pl.Int8)
    .alias("received_amount_nonpositive"),

    (
        pl.col("amount_paid") <= 0
    )
    .fill_null(False)
    .cast(pl.Int8)
    .alias("paid_amount_nonpositive"),
])

# Important:
# We are NOT using is_laundering to create features.
# It remains the target variable for supervised learning.


# ---------------------------------------------------------
# 7. SAVE PROCESSED DATASET
# ---------------------------------------------------------

print("\n[4/5] Saving compressed Parquet dataset...")
print("This step may take some time for a 5-million-row CSV.")

df.sink_parquet(
    OUTPUT_FILE,
    compression="zstd",
    compression_level=3,
    statistics=True,
    maintain_order=True,
    engine="streaming",
)

print(f"Saved: {OUTPUT_FILE}")


# ---------------------------------------------------------
# 8. DATASET AUDIT
# ---------------------------------------------------------

print("\n[5/5] Generating dataset audit...")

# Read the saved Parquet lazily for the audit.
# This avoids creating a large in-memory DataFrame.

clean = pl.scan_parquet(OUTPUT_FILE)

summary = clean.select([
    pl.len().alias("total_rows"),

    pl.col("timestamp")
    .null_count()
    .alias("null_timestamps"),

    pl.col("from_account")
    .null_count()
    .alias("null_sender_accounts"),

    pl.col("to_account")
    .null_count()
    .alias("null_receiver_accounts"),

    pl.col("amount_received")
    .null_count()
    .alias("null_amount_received"),

    pl.col("amount_paid")
    .null_count()
    .alias("null_amount_paid"),

    pl.col("payment_format")
    .null_count()
    .alias("null_payment_formats"),

    pl.col("is_laundering")
    .null_count()
    .alias("null_labels"),

    pl.col("amount_received")
    .min()
    .alias("minimum_amount_received"),

    pl.col("amount_received")
    .max()
    .alias("maximum_amount_received"),

    pl.col("timestamp")
    .min()
    .alias("first_timestamp"),

    pl.col("timestamp")
    .max()
    .alias("last_timestamp"),

    pl.col("from_account")
    .n_unique()
    .alias("unique_sender_accounts"),

    pl.col("to_account")
    .n_unique()
    .alias("unique_receiver_accounts"),
]).collect(engine="streaming")


# Count normal and laundering transactions.

class_counts = (
    clean
    .group_by("is_laundering")
    .agg(
        pl.len().alias("transaction_count")
    )
    .sort("is_laundering")
    .collect(engine="streaming")
)

class_counts.write_csv(CLASS_FILE)

class_distribution = {
    str(row["is_laundering"]): row["transaction_count"]
    for row in class_counts.to_dicts()
}

normal_count = class_distribution.get("0", 0)
laundering_count = class_distribution.get("1", 0)

known_labels = normal_count + laundering_count

laundering_percentage = (
    100 * laundering_count / known_labels
    if known_labels
    else 0
)

summary_dict = summary.to_dicts()[0]

report_lines = [
    "HI-Small Dataset Audit Report",
    "=" * 45,
    f"Input: {INPUT_FILE}",
    f"Output: {OUTPUT_FILE}",
    f"Total rows: {summary_dict['total_rows']:,}",
    f"Output size: {OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MiB",
    f"Processing time: {time.perf_counter() - start_time:.2f} seconds",
    "",
    "CLASS DISTRIBUTION",
    "-" * 30,
    f"Normal transactions (0): {normal_count:,}",
    f"Laundering transactions (1): {laundering_count:,}",
    f"Known labels: {known_labels:,}",
    f"Laundering percentage among known labels: {laundering_percentage:.6f}%",
    "",
    "DATA QUALITY",
    "-" * 30,
]

for column, value in summary_dict.items():
    if column.startswith("null_"):
        report_lines.append(f"{column}: {value:,}")

report_lines.extend([
    "",
    "ADDITIONAL SUMMARY",
    "-" * 30,
])

for column, value in summary_dict.items():
    if not column.startswith("null_"):
        report_lines.append(f"{column}: {value}")

report_lines.extend([
    "",
    "IMPORTANT NOTES",
    "-" * 30,
    "1. Account and bank IDs are stored as strings.",
    "2. The target column is is_laundering.",
    "3. Do not include is_laundering among model input features.",
    "4. The account metadata file was not joined.",
    "5. The patterns file was not used as a feature source.",
    "6. Investigate rows with missing timestamps or labels.",
    "7. Use a chronological train/validation/test split.",
    "8. These labels are dataset labels, not independent proof of crime.",
])

AUDIT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ---------------------------------------------------------
# 9. FINISH
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETED")
print("=" * 65)

print(f"Clean dataset: {OUTPUT_FILE}")
print(f"Audit report:  {AUDIT_FILE}")
print(f"Class counts:  {CLASS_FILE}")

print(f"\nTotal rows: {summary_dict['total_rows']:,}")
print(f"Normal transactions: {normal_count:,}")
print(f"Laundering transactions: {laundering_count:,}")
print(f"Laundering percentage: {laundering_percentage:.6f}%")

print(f"\nTotal time: {time.perf_counter() - start_time:.2f} seconds")
print("\nReview audit_report.txt before training your model.")