"""
Stage 2: Join account metadata to cleaned transactions.

Inputs:
    processed/transactions_clean.parquet
    HI-Small_accounts.csv

Output:
    processed/transactions_with_accounts.parquet

The script:
1. Loads the cleaned transaction Parquet file.
2. Loads account metadata.
3. Normalizes identifiers as strings.
4. Checks possible join keys.
5. Joins account metadata for senders and receivers.
6. Verifies that the transaction row count is preserved.
7. Reports account-match rates.
"""

from pathlib import Path

import polars as pl


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRANSACTIONS_PATH = (
    BASE_DIR / "processed_medium" / "transactions_clean.parquet"
)

ACCOUNTS_PATH = BASE_DIR / "HI-Medium_accounts.csv"

OUTPUT_PATH = (
    BASE_DIR / "processed_medium" / "transactions_with_accounts.parquet"
)


# ---------------------------------------------------------
# 2. Validate input files
# ---------------------------------------------------------

if not TRANSACTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Transaction file not found:\n{TRANSACTIONS_PATH}"
    )

if not ACCOUNTS_PATH.exists():
    raise FileNotFoundError(
        f"Account metadata file not found:\n{ACCOUNTS_PATH}"
    )


# ---------------------------------------------------------
# 3. Load transactions and account metadata
# ---------------------------------------------------------

print("Loading cleaned transactions...")

transactions = pl.scan_parquet(TRANSACTIONS_PATH)

print("Loading account metadata...")

accounts = pl.read_csv(
    ACCOUNTS_PATH,
    has_header=True,
    infer_schema=False,
    null_values=["", "NA", "N/A", "null", "NULL"],
)


# ---------------------------------------------------------
# 4. Rename account metadata columns
# ---------------------------------------------------------

accounts = accounts.rename({
    "Bank Name": "bank_name",
    "Bank ID": "bank_id",
    "Account Number": "account_number",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
})


required_account_columns = [
    "bank_name",
    "bank_id",
    "account_number",
    "entity_id",
    "entity_name",
]

missing_account_columns = [
    column
    for column in required_account_columns
    if column not in accounts.columns
]

if missing_account_columns:
    raise ValueError(
        "The account metadata file is missing required columns: "
        f"{missing_account_columns}"
    )


# ---------------------------------------------------------
# 5. Normalize identifiers and text
# ---------------------------------------------------------

transaction_columns = [
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
]

transactions = transactions.with_columns([
    pl.col(column)
    .cast(pl.String)
    .str.strip_chars()
    .alias(column)
    for column in transaction_columns
])

accounts = accounts.with_columns([
    pl.col(column)
    .cast(pl.String)
    .str.strip_chars()
    .alias(column)
    for column in [
        "bank_name",
        "bank_id",
        "account_number",
        "entity_id",
        "entity_name",
    ]
])


# Extract a general entity category from names such as:
# "Sole Proprietorship #50438"
# "Corporation #33520"
# "Partnership #35397"
accounts = accounts.with_columns(
    pl.col("entity_name")
    .str.replace(r"\s+#\d+$", "")
    .str.strip_chars()
    .alias("entity_type")
)


# ---------------------------------------------------------
# 6. Check duplicate account records
# ---------------------------------------------------------

duplicate_account_rows = (
    accounts
    .group_by("account_number")
    .len()
    .filter(pl.col("len") > 1)
)

duplicate_composite_rows = (
    accounts
    .group_by(["bank_id", "account_number"])
    .len()
    .filter(pl.col("len") > 1)
)

duplicate_account_count = len(duplicate_account_rows)
duplicate_composite_count = len(duplicate_composite_rows)

print(
    f"Duplicate account-number keys: {duplicate_account_count:,}"
)
print(
    "Duplicate bank/account keys: "
    f"{duplicate_composite_count:,}"
)


# ---------------------------------------------------------
# 7. Build possible lookup tables
# ---------------------------------------------------------

# The composite key is semantically preferable when transaction
# bank IDs use the same identifier system as account bank IDs.
composite_lookup = (
    accounts
    .select([
        "bank_id",
        "account_number",
        "bank_name",
        "entity_id",
        "entity_name",
        "entity_type",
    ])
    .with_columns(
        pl.concat_str([
            pl.col("bank_id"),
            pl.lit("|"),
            pl.col("account_number"),
        ]).alias("composite_key")
    )
    .lazy()
)

# Account-only lookup is used only if it produces a better and
# unambiguous match. The script reports which key it selected.
unique_account_numbers = (
    accounts
    .group_by("account_number")
    .agg(pl.len().alias("_account_record_count"))
    .filter(pl.col("_account_record_count") == 1)
    .select("account_number")
)

account_lookup = (
    accounts
    .select([
        "account_number",
        "bank_name",
        "entity_id",
        "entity_name",
        "entity_type",
    ])
    .join(
        unique_account_numbers,
        on="account_number",
        how="semi",
    )
    .lazy()
)


# ---------------------------------------------------------
# 8. Measure candidate join match rates
# ---------------------------------------------------------

transaction_keys = transactions.select([
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
])


def calculate_match_rate(
    key_mode: str,
) -> tuple[float, float]:
    """
    Return sender and receiver metadata match rates.
    """

    if key_mode == "composite":
        lookup_keys = composite_lookup.select(
            [
                "composite_key",
                pl.lit(1).alias("account_metadata_match"),
            ]
        ).unique()

        keys = transaction_keys.with_columns([
            pl.concat_str([
                pl.col("from_bank"),
                pl.lit("|"),
                pl.col("from_account"),
            ]).alias("sender_key"),

            pl.concat_str([
                pl.col("to_bank"),
                pl.lit("|"),
                pl.col("to_account"),
            ]).alias("receiver_key"),
        ])

        sender_matches = (
            keys
            .select("sender_key")
            .join(
                lookup_keys,
                left_on="sender_key",
                right_on="composite_key",
                how="left",
            )
            .select(
                pl.col("account_metadata_match")
                .is_not_null()
                .mean()
                .alias("rate")
            )
            .collect(engine="streaming")
            .item()
        )

        receiver_matches = (
            keys
            .select("receiver_key")
            .join(
                lookup_keys,
                left_on="receiver_key",
                right_on="composite_key",
                how="left",
            )
            .select(
                pl.col("account_metadata_match")
                .is_not_null()
                .mean()
                .alias("rate")
            )
            .collect(engine="streaming")
            .item()
        )

    elif key_mode == "account":
        lookup_keys = account_lookup.select(
            [
                "account_number",
                pl.lit(1).alias("account_metadata_match"),
            ]
        ).unique()

        sender_matches = (
            transaction_keys
            .select(pl.col("from_account").alias("account_number"))
            .join(
                lookup_keys,
                on="account_number",
                how="left",
            )
            .select(
                pl.col("account_metadata_match")
                .is_not_null()
                .mean()
                .alias("rate")
            )
            .collect(engine="streaming")
            .item()
        )

        receiver_matches = (
            transaction_keys
            .select(pl.col("to_account").alias("account_number"))
            .join(
                lookup_keys,
                on="account_number",
                how="left",
            )
            .select(
                pl.col("account_metadata_match")
                .is_not_null()
                .mean()
                .alias("rate")
            )
            .collect(engine="streaming")
            .item()
        )

    else:
        raise ValueError(f"Unknown key mode: {key_mode}")

    return float(sender_matches), float(receiver_matches)


composite_sender_rate, composite_receiver_rate = (
    calculate_match_rate("composite")
)

account_sender_rate, account_receiver_rate = (
    calculate_match_rate("account")
)

print(
    "Composite sender match rate: "
    f"{composite_sender_rate:.2%}"
)
print(
    "Composite receiver match rate: "
    f"{composite_receiver_rate:.2%}"
)
print(
    "Account-only sender match rate: "
    f"{account_sender_rate:.2%}"
)
print(
    "Account-only receiver match rate: "
    f"{account_receiver_rate:.2%}"
)


# Prefer the composite key when it gives strong matches.
# Otherwise use account-only matching if it gives strong matches.
minimum_acceptable_rate = 0.90

composite_average = (
    composite_sender_rate + composite_receiver_rate
) / 2

account_average = (
    account_sender_rate + account_receiver_rate
) / 2

if (
    composite_average >= minimum_acceptable_rate
    and duplicate_composite_count == 0
):
    join_mode = "composite"

elif (
    account_average >= minimum_acceptable_rate
):
    join_mode = "account"
    print(
        "Using account-only matching. "
        f"{duplicate_account_count:,} ambiguous account keys "
        "will remain without metadata."
    )

else:
    raise RuntimeError(
        "No safe join key was found.\n"
        "The account metadata does not match the transaction "
        "identifiers strongly enough, or the metadata keys are "
        "not unique.\n"
        "Inspect the reported match rates before continuing."
    )

print(f"Selected join mode: {join_mode}")


# ---------------------------------------------------------
# 9. Prepare sender and receiver lookup tables
# ---------------------------------------------------------

metadata_columns = [
    "bank_name",
    "entity_id",
    "entity_name",
    "entity_type",
]


if join_mode == "composite":
    sender_lookup = (
        composite_lookup
        .select([
            "composite_key",
            *metadata_columns,
        ])
        .rename({
            column: f"sender_{column}"
            for column in metadata_columns
        })
    )

    receiver_lookup = (
        composite_lookup
        .select([
            "composite_key",
            *metadata_columns,
        ])
        .rename({
            column: f"receiver_{column}"
            for column in metadata_columns
        })
    )

    enriched = transactions.with_columns([
        pl.concat_str([
            pl.col("from_bank"),
            pl.lit("|"),
            pl.col("from_account"),
        ]).alias("sender_join_key"),

        pl.concat_str([
            pl.col("to_bank"),
            pl.lit("|"),
            pl.col("to_account"),
        ]).alias("receiver_join_key"),
    ])

    enriched = enriched.join(
        sender_lookup,
        left_on="sender_join_key",
        right_on="composite_key",
        how="left",
    )

    enriched = enriched.join(
        receiver_lookup,
        left_on="receiver_join_key",
        right_on="composite_key",
        how="left",
    )

else:
    sender_lookup = (
        account_lookup
        .select([
            "account_number",
            *metadata_columns,
        ])
        .rename({
            column: f"sender_{column}"
            for column in metadata_columns
        })
    )

    receiver_lookup = (
        account_lookup
        .select([
            "account_number",
            *metadata_columns,
        ])
        .rename({
            column: f"receiver_{column}"
            for column in metadata_columns
        })
    )

    enriched = transactions.join(
        sender_lookup,
        left_on="from_account",
        right_on="account_number",
        how="left",
    )

    enriched = enriched.join(
        receiver_lookup,
        left_on="to_account",
        right_on="account_number",
        how="left",
    )


# ---------------------------------------------------------
# 10. Add explicit match indicators
# ---------------------------------------------------------

enriched = enriched.with_columns([
    pl.col("sender_entity_id")
    .is_not_null()
    .cast(pl.Int8)
    .alias("sender_metadata_matched"),

    pl.col("receiver_entity_id")
    .is_not_null()
    .cast(pl.Int8)
    .alias("receiver_metadata_matched"),
])


# Remove temporary join keys and duplicate lookup keys.
enriched = enriched.drop([
    column
    for column in [
        "sender_join_key",
        "receiver_join_key",
        "composite_key",
        "account_number",
    ]
    if column in enriched.collect_schema().names()
])


# ---------------------------------------------------------
# 11. Verify row count and save
# ---------------------------------------------------------

original_count = (
    transactions
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)

enriched_count = (
    enriched
    .select(pl.len())
    .collect(engine="streaming")
    .item()
)

if original_count != enriched_count:
    raise RuntimeError(
        "Row count changed during metadata joins:\n"
        f"Original rows: {original_count:,}\n"
        f"Enriched rows: {enriched_count:,}"
    )


print(f"Original transaction rows: {original_count:,}")
print(f"Enriched transaction rows: {enriched_count:,}")

print("Writing enriched Parquet file...")

enriched.sink_parquet(
    OUTPUT_PATH,
    compression="zstd",
    compression_level=3,
    statistics=True,
    maintain_order=True,
    engine="streaming",
)

print(f"Saved output to:\n{OUTPUT_PATH}")
print("Stage 2 account enrichment completed successfully.")