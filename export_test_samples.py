"""
Export labeled positive and negative sample batches from the held-out test
period for manual model testing.
"""

from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    BASE_DIR / "processed" / "transactions_with_graph_features.parquet"
)
POSITIVE_OUTPUT = BASE_DIR / "processed" / "sample_laundering.csv"
NEGATIVE_OUTPUT = BASE_DIR / "processed" / "sample_not_laundering.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

input_path = INPUT_PATH.as_posix()
positive_path = POSITIVE_OUTPUT.as_posix()
negative_path = NEGATIVE_OUTPUT.as_posix()

query = f"""
COPY (
    SELECT *
    FROM read_parquet('{input_path}')
    WHERE timestamp >= TIMESTAMP '2022-09-09 03:16:00'
      AND is_laundering = 1
    ORDER BY timestamp
    LIMIT 25
) TO '{positive_path}' WITH (HEADER, DELIMITER ',')
"""
duckdb.sql(query)

query = f"""
COPY (
    SELECT *
    FROM read_parquet('{input_path}')
    WHERE timestamp >= TIMESTAMP '2022-09-09 03:16:00'
      AND is_laundering = 0
    ORDER BY timestamp
    LIMIT 25
) TO '{negative_path}' WITH (HEADER, DELIMITER ',')
"""
duckdb.sql(query)

print(f"Saved positive sample: {POSITIVE_OUTPUT}")
print(f"Saved negative sample: {NEGATIVE_OUTPUT}")
