# IBM AML Project: Summary

Detects money-laundering transactions in the IBM synthetic AML dataset (`HI-Small`). The final model is XGBoost trained on transaction, account, behaviour and network features. It ranks transactions by laundering probability, and investigators review the **top 1,000 alerts**.

---

## 1. Data

| File | Role |
|---|---|
| `HI-Small_Trans.csv` | Main dataset: 5,078,345 transactions, Sep 1–18 2022. Target = `Is Laundering` |
| `HI-Small_accounts.csv` | Account metadata (bank name, entity ID, entity name) |
| `hi-small_patterns.txt` | Known laundering patterns (fan-in, cycle, etc.). Used only as a reference and **not** as a feature, because that would leak the label |
| `HI-Medium_*` | Larger dataset (31.9M rows, Sep 1–28). Used only to test the trained model on unseen data |

Class imbalance: only **0.102%** of transactions are laundering (5,177 of 5.08M). That is why accuracy is ignored and the project uses **precision, recall and PR-AUC** instead.

---

## 2. Pipeline

Each stage reads the previous Parquet file and writes a new one. The row count stays the same at every stage.

```text
HI-Small_Trans.csv
  │  preprocess_data.py         rename columns, IDs kept as strings, parse timestamps,
  ▼                             basic features (hour, weekday, same_bank, same_currency…)
processed/transactions_clean.parquet
  │  join_account_metadata.py   join accounts twice (sender + receiver) on account number,
  ▼                             add entity_type (Corporation / Sole Proprietorship / Partnership)
processed/transactions_with_accounts.parquet
  │  add_behavior_features.py   1h / 24h rolling counts, amounts, unique counterparties,
  ▼                             time since last activity (past transactions only)
processed/transactions_with_behavior.parquet
  │  add_graph_features.py      fan-in, fan-out, edge counts, rapid-forwarding indicator
  ▼                             (past 24h only)
processed/transactions_with_graph_features.parquet   ← 37 model features
  │
  ├─ train_baseline.py [--behavior | --graph]   LightGBM models
  ├─ train_xgboost.py                           XGBoost model (final)
  ├─ threshold_analysis.py                      compare models at equal recall
  ├─ alert_budget_analysis.py                   compare models at top-N alerts
  ├─ finalize_xgboost.py                        lock in the 1,000-alert policy
  └─ predict_xgboost.py / predict_batch.py      score transactions
```

**Key design decisions**
- **Chronological split** instead of a random one: train = first 70% (to Sep 7 14:55), validation = next 15% (to Sep 9 03:16), test = last 15%.
- **No future leakage:** behaviour and graph features only look at transactions *before* the current one. Transactions with the same timestamp are excluded too.
- **Raw IDs are not model inputs.** Account numbers and entity names are used to build features and are then dropped. Categoricals (payment format, currencies, entity types) use native categorical encoding.
- **Class weighting** handles the imbalance (no SMOTE).
- **Account join quirk:** bank IDs in the two files use different systems (0% match), so the join uses the account number alone (≈100% match). 8 ambiguous account numbers are excluded. 50 senders and 13 receivers are left unmatched and flagged.

---

## 3. Results (Small test set, 762,103 rows, 1,561 laundering)

### Model progression

| Model | Features | PR-AUC | ROC-AUC | Precision | Recall | Alerts |
|---|---|---:|---:|---:|---:|---:|
| LightGBM baseline | transaction + account | 0.018 | 0.924 | 0.33% | 98.7% | 471,291 |
| LightGBM | + behaviour | 0.014 | 0.920 | 1.07% | 98.4% | 143,975 |
| LightGBM | + behaviour + graph | 0.020 | 0.944 | 1.24% | 98.6% | 123,699 |
| **XGBoost** | + behaviour + graph | **0.468** | **0.983** | **62.2%** | 40.9% | **1,027** |

The LightGBM models catch almost everything but bury the hits in false alerts. The cause is that their scores bunch up near 1.0, so the threshold has no fine control. XGBoost ranks far better.

### Fixed alert budget (top-N alerts reviewed)

| Top N | LightGBM hits | XGBoost hits | XGBoost precision | XGBoost recall |
|---:|---:|---:|---:|---:|
| 100 | 4 | **98** | 98.0% | 6.3% |
| 500 | 13 | **412** | 82.4% | 26.4% |
| 1,000 | 34 | **635** | 63.5% | 40.7% |
| 5,000 | 120 | **967** | 19.3% | 62.0% |
| 10,000 | 216 | **1,127** | 11.3% | 72.2% |

### Final model: XGBoost with the top 1,000 alerts per period

| | Value |
|---|---|
| Alerts | 1,000 |
| True / false positives | 635 / 365 |
| Precision | **63.5%** |
| Recall | **40.7%** |
| F1 | 0.496 |
| PR-AUC | 0.468 |

The policy is "top 1,000" rather than a fixed probability threshold. The validation threshold (0.979) produced 2,769 alerts on the test set, so a fixed threshold can't hold the alert count steady.

Files: `processed/xgboost_graph_model.json`, `processed/final_xgboost_config.json`, `processed/final_xgboost_metrics.txt`.

### Generalisation check on HI-Medium (unseen data)

The full Medium dataset was run through the same pipeline (`processed_medium/`). Then 100 laundering and 100 normal transactions from **Sep 19 onward** (after the Small data ends) were scored with the Small-trained model:

| | Value |
|---|---|
| Top-100 alerts → true positives | **89 / 100** (11 false positives) |
| Precision = Recall (top 100) | 89% |
| ROC-AUC / Avg. precision | 0.946 / 0.922 |
| Median probability (laundering vs normal) | 0.980 vs 0.007 |

The model carries over to later, unseen transactions. One caveat: this batch is 50% laundering, while real data is about 0.1%, so these numbers are much more optimistic than real-world precision would be.

---

## 4. How to use it interactively

Run all commands from the project folder in PowerShell:

```powershell
cd "D:\Sem 5\machine learning\lab\ibm aml project"
```

### A. Score a ready-made batch (quickest)

`predict_batch.py` accepts any `.csv` or `.parquet` file that already has the 37 engineered columns:

```powershell
.\.venv\Scripts\python.exe predict_batch.py `
  --input  "processed\sample_laundering.csv" `
  --output "processed\my_predictions.csv" `
  --alerts 25
```

The output is sorted by `laundering_probability` and adds `alert_rank` and `is_alert`, with the top N set to `true`.

Ready-to-use batches:
| File | Content |
|---|---|
| `processed/sample_laundering.csv` | 25 laundering rows from the Small test period |
| `processed/sample_not_laundering.csv` | 25 normal rows from the Small test period |
| `processed_medium/medium_model_ready_combined.parquet` | 100 laundering + 100 normal rows from Medium (Sep 19+) |

(`processed/medium_sample_*.csv` are **raw** rows and can't be scored directly.)

### B. Score new or raw transactions

The model **cannot** score raw transactions on their own. It needs each account's recent history to build the behaviour and graph features. A new dataset must go through the full pipeline first:

```powershell
# 1. Clean (accepts arguments)
.\.venv\Scripts\python.exe preprocess_data.py --input ".\NEW_Trans.csv" --output-dir ".\processed_new"
# 2–4. Join accounts, behaviour, graph: these scripts have HARD-CODED paths.
#      Edit the paths at the top of each file first (they currently point at processed_medium / HI-Medium).
.\.venv\Scripts\python.exe join_account_metadata.py
.\.venv\Scripts\python.exe add_behavior_features.py
.\.venv\Scripts\python.exe add_graph_features.py
# 5. Score
.\.venv\Scripts\python.exe predict_batch.py --input ".\processed_new\transactions_with_graph_features.parquet" --output ".\processed_new\predictions.csv" --alerts 1000
```

Run steps 1–4 on the **full** transaction history, not on a small sample, or the history-based features will be wrong. Large files such as Medium need lots of disk: DuckDB spill files go to `D:/` (about 9 GB for Medium), and the behaviour step takes a long time.

### C. Look inside Parquet files

`view_parquet.py` (DuckDB) exports 100 rows to `processed/transactions_sample.csv`, which you can open in Excel or VS Code.

### D. Retrain or re-run the analyses (Small data)

```powershell
.\.venv\Scripts\python.exe train_baseline.py            # LightGBM baseline
.\.venv\Scripts\python.exe train_baseline.py --behavior # + behaviour
.\.venv\Scripts\python.exe train_baseline.py --graph    # + behaviour + graph
.\.venv\Scripts\python.exe train_xgboost.py
.\.venv\Scripts\python.exe threshold_analysis.py
.\.venv\Scripts\python.exe alert_budget_analysis.py
.\.venv\Scripts\python.exe finalize_xgboost.py
```

---

## 5. Limitations

- The labels come from a synthetic dataset. A high score is a **risk flag for investigation**, not proof of laundering.
- At the 1,000-alert budget about 59% of laundering transactions are still missed. Raising the budget trades precision for recall (see the table above).
- The pipeline is batch-only, and the stage scripts have hard-coded input and output paths.
- Possible next steps: score the full Medium dataset at its real prevalence, tune XGBoost hyperparameters, and add SHAP explanations per alert.
