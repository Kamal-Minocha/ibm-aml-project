import duckdb

input_path = r"processed\transactions_with_accounts.parquet"
output_path = r"processed\transactions_sample.csv"

query = f"""
COPY (
    SELECT *
    FROM '{input_path}'
    LIMIT 100
) TO '{output_path}'
WITH (HEADER, DELIMITER ',')
"""

duckdb.sql(query)

print(f"Saved 100 actual rows to: {output_path}")
input("Press Enter to exit...")