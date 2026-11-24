"""
Export temporally distinct labeled Medium-dataset sample batches.

Only transactions from 2022-09-19 onward are used because the Small dataset
ends on 2022-09-18. This guarantees the exported rows are not from the
Small test batches.
"""

from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "HI-Medium_Trans.csv"
OUTPUT_DIR = BASE_DIR / "processed"
POSITIVE_OUTPUT = OUTPUT_DIR / "medium_sample_laundering.csv"
NEGATIVE_OUTPUT = OUTPUT_DIR / "medium_sample_not_laundering.csv"
SAMPLE_SIZE = 100
START_DATE = "2022-09-19 00:00:00"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Medium transaction file not found: {INPUT_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
input_path = INPUT_PATH.as_posix()

def export_sample(label: int, output_path: Path) -> None:
    query = f"""
    COPY (
        SELECT
            "Timestamp",
            "From Bank",
            "Account" AS from_account,
            "To Bank",
            "Account_1" AS to_account,
            "Amount Received",
            "Receiving Currency",
            "Amount Paid",
            "Payment Currency",
            "Payment Format",
            "Is Laundering"
        FROM read_csv(
            '{input_path}',
            header=true,
            columns={{
                'Timestamp': 'TIMESTAMP',
                'From Bank': 'VARCHAR',
                'Account': 'VARCHAR',
                'To Bank': 'VARCHAR',
                'Account_1': 'VARCHAR',
                'Amount Received': 'DOUBLE',
                'Receiving Currency': 'VARCHAR',
                'Amount Paid': 'DOUBLE',
                'Payment Currency': 'VARCHAR',
                'Payment Format': 'VARCHAR',
                'Is Laundering': 'INTEGER'
            }},
            sample_size=-1
        )
        WHERE "Timestamp" >= TIMESTAMP '{START_DATE}'
          AND "Is Laundering" = {label}
        ORDER BY "Timestamp", from_account, to_account
        LIMIT {SAMPLE_SIZE}
    ) TO '{output_path.as_posix()}' WITH (HEADER, DELIMITER ',')
    """
    duckdb.sql(query)


export_sample(1, POSITIVE_OUTPUT)
export_sample(0, NEGATIVE_OUTPUT)

print(f"Saved laundering batch: {POSITIVE_OUTPUT}")
print(f"Saved non-laundering batch: {NEGATIVE_OUTPUT}")
print(
    f"Each batch contains up to {SAMPLE_SIZE} rows from "
    f"{START_DATE} onward."
)
