"""
Stage 4: Add leakage-safe historical account behavior features.

All rolling values exclude the current transaction and transactions at the
same timestamp. This conservative rule prevents future information from
entering a transaction's features.
"""

from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "processed_medium" / "transactions_with_accounts.parquet"
OUTPUT_PATH = BASE_DIR / "processed_medium" / "transactions_with_behavior.parquet"
TEMP_DIR = Path("D:/duckdb_temp_medium")

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
    events AS (
        SELECT
            transaction_id,
            timestamp,
            from_account AS account_id,
            to_account AS counterparty_id,
            'sender' AS role,
            1 AS outgoing_count,
            0 AS incoming_count,
            amount_paid AS outgoing_amount,
            CAST(0 AS DOUBLE) AS incoming_amount
        FROM base
        UNION ALL
        SELECT
            transaction_id,
            timestamp,
            to_account AS account_id,
            from_account AS counterparty_id,
            'receiver' AS role,
            0 AS outgoing_count,
            1 AS incoming_count,
            CAST(0 AS DOUBLE) AS outgoing_amount,
            amount_received AS incoming_amount
        FROM base
    ),
    behavior AS (
        SELECT
            transaction_id,
            role,
            COALESCE(
                COUNT(*) OVER activity_1h,
                0
            ) AS tx_count_1h,
            COALESCE(
                COUNT(*) FILTER (
                    WHERE outgoing_count = 1
                ) OVER activity_24h,
                0
            ) AS outgoing_count_24h,
            COALESCE(
                COUNT(*) FILTER (
                    WHERE incoming_count = 1
                ) OVER activity_24h,
                0
            ) AS incoming_count_24h,
            COALESCE(
                SUM(outgoing_amount) OVER activity_24h,
                0
            ) AS outgoing_amount_24h,
            COALESCE(
                SUM(incoming_amount) OVER activity_24h,
                0
            ) AS incoming_amount_24h,
            COALESCE(
                COUNT(DISTINCT counterparty_id) FILTER (
                    WHERE outgoing_count = 1
                ) OVER activity_24h,
                0
            ) AS unique_outgoing_counterparties_24h,
            COALESCE(
                COUNT(DISTINCT counterparty_id) FILTER (
                    WHERE incoming_count = 1
                ) OVER activity_24h,
                0
            ) AS unique_incoming_counterparties_24h,
            EXTRACT(
                EPOCH FROM (
                    timestamp - LAG(timestamp) OVER (
                        PARTITION BY account_id
                        ORDER BY timestamp
                    )
                )
            ) AS seconds_since_account_activity
        FROM events
        WINDOW
            activity_1h AS (
                PARTITION BY account_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ),
            activity_24h AS (
                PARTITION BY account_id
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            )
    ),
    behavior_by_transaction AS (
        SELECT
            transaction_id,
            MAX(
                CASE WHEN role = 'sender'
                THEN tx_count_1h ELSE 0 END
            ) AS sender_tx_count_1h,
            MAX(
                CASE WHEN role = 'sender'
                THEN outgoing_count_24h ELSE 0 END
            ) AS sender_outgoing_count_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN incoming_count_24h ELSE 0 END
            ) AS sender_incoming_count_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN outgoing_amount_24h ELSE 0 END
            ) AS sender_outgoing_amount_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN incoming_amount_24h ELSE 0 END
            ) AS sender_incoming_amount_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN unique_outgoing_counterparties_24h ELSE 0 END
            ) AS sender_unique_recipients_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN unique_incoming_counterparties_24h ELSE 0 END
            ) AS sender_unique_senders_24h,
            MAX(
                CASE WHEN role = 'sender'
                THEN seconds_since_account_activity END
            ) AS sender_seconds_since_activity,
            MAX(
                CASE WHEN role = 'receiver'
                THEN tx_count_1h ELSE 0 END
            ) AS receiver_tx_count_1h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN outgoing_count_24h ELSE 0 END
            ) AS receiver_outgoing_count_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN incoming_count_24h ELSE 0 END
            ) AS receiver_incoming_count_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN outgoing_amount_24h ELSE 0 END
            ) AS receiver_outgoing_amount_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN incoming_amount_24h ELSE 0 END
            ) AS receiver_incoming_amount_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN unique_outgoing_counterparties_24h ELSE 0 END
            ) AS receiver_unique_recipients_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN unique_incoming_counterparties_24h ELSE 0 END
            ) AS receiver_unique_senders_24h,
            MAX(
                CASE WHEN role = 'receiver'
                THEN seconds_since_account_activity END
            ) AS receiver_seconds_since_activity
        FROM behavior
        GROUP BY transaction_id
    )
    SELECT
        b.* EXCLUDE (transaction_id),
        f.* EXCLUDE (transaction_id)
    FROM base AS b
    INNER JOIN behavior_by_transaction AS f
        USING (transaction_id)
)
TO '{output_path}'
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

print("Creating leakage-safe historical features...")
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
