"""
Stage 5: Add leakage-safe transaction-network features.

The features describe the network visible before each transaction:
historical fan-in, historical fan-out, and rapid forwarding.
Pattern names from hi-small_patterns.txt are intentionally not copied into
the feature table because those annotations can leak the target.
"""

from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "processed_medium" / "transactions_with_behavior.parquet"
OUTPUT_PATH = (
    BASE_DIR / "processed_medium" / "transactions_with_graph_features.parquet"
)
TEMP_DIR = Path("D:/duckdb_temp_graph_medium")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

input_path = INPUT_PATH.as_posix()
output_path = OUTPUT_PATH.as_posix()

query = f"""
COPY (
    WITH base AS (
        SELECT
            row_number() OVER () - 1 AS transaction_id,
            t.*
        FROM read_parquet('{input_path}') AS t
    ),
    account_events AS (
        SELECT
            transaction_id,
            timestamp,
            from_account AS account_id,
            to_account AS counterparty_id,
            1 AS outgoing_event,
            0 AS incoming_event
        FROM base
        UNION ALL
        SELECT
            transaction_id,
            timestamp,
            to_account AS account_id,
            from_account AS counterparty_id,
            0 AS outgoing_event,
            1 AS incoming_event
        FROM base
    ),
    event_features AS (
        SELECT
            transaction_id,
            outgoing_event,
            incoming_event,
            COUNT(DISTINCT counterparty_id) FILTER (
                WHERE outgoing_event = 1
            ) OVER prior_window AS historical_fan_out_24h,
            COUNT(*) FILTER (
                WHERE outgoing_event = 1
            ) OVER prior_window AS historical_outgoing_edge_count_24h,
            COUNT(DISTINCT counterparty_id) FILTER (
                WHERE incoming_event = 1
            ) OVER prior_window AS historical_fan_in_24h,
            COUNT(*) FILTER (
                WHERE incoming_event = 1
            ) OVER prior_window AS historical_incoming_edge_count_24h,
            COUNT(*) FILTER (
                WHERE incoming_event = 1
            ) OVER prior_window AS prior_incoming_count
        FROM account_events
        WINDOW prior_window AS (
            PARTITION BY account_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING
            AND INTERVAL '1 microsecond' PRECEDING
        )
    ),
    graph_features AS (
        SELECT
            transaction_id,
            MAX(
                CASE WHEN outgoing_event = 1
                THEN historical_fan_out_24h ELSE 0 END
            ) AS sender_historical_fan_out_24h,
            MAX(
                CASE WHEN outgoing_event = 1
                THEN historical_outgoing_edge_count_24h ELSE 0 END
            ) AS sender_historical_edge_count_24h,
            MAX(
                CASE WHEN incoming_event = 1
                THEN historical_fan_in_24h ELSE 0 END
            ) AS receiver_historical_fan_in_24h,
            MAX(
                CASE WHEN incoming_event = 1
                THEN historical_incoming_edge_count_24h ELSE 0 END
            ) AS receiver_historical_edge_count_24h,
            MAX(
                CASE WHEN outgoing_event = 1
                THEN prior_incoming_count ELSE 0 END
            ) AS sender_prior_incoming_count
        FROM event_features
        GROUP BY transaction_id
    )
    SELECT
        b.* EXCLUDE (transaction_id),
        g.sender_historical_fan_out_24h,
        g.sender_historical_edge_count_24h,
        g.receiver_historical_fan_in_24h,
        g.receiver_historical_edge_count_24h,
        CAST(g.sender_prior_incoming_count > 0 AS INTEGER)
            AS sender_received_money_24h,
        CAST(
            g.sender_prior_incoming_count > 0
            AND g.sender_historical_edge_count_24h > 0
            AS INTEGER
        ) AS rapid_forwarding_indicator
    FROM base b
    INNER JOIN graph_features g
        USING (transaction_id)
)
TO '{output_path}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating leakage-safe network features...")
connection = duckdb.connect()
connection.execute("SET memory_limit = '6GB'")
connection.execute("SET threads = 4")
connection.execute("SET preserve_insertion_order = false")
connection.execute(f"SET temp_directory = '{TEMP_DIR.as_posix()}'")
connection.execute("SET max_temp_directory_size = '100GB'")
try:
    connection.execute(query)
finally:
    connection.close()
print(f"Saved: {OUTPUT_PATH}")
