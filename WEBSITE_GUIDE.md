# AML Risk Scoring Demo: Website Guide

A Streamlit web app that serves the project's final XGBoost model. You can explore
real labelled transactions, score uploaded batches, and experiment with a
hypothetical transaction. Everything here refers to files in this repository.

---

## 1. Overview

**What it does.** The app loads the trained model (`webapp/model/xgboost_graph_model.json`)
and offers four pages:

| Page | Purpose |
|---|---|
| **Overview** | What the model is, its held-out test results, and where its features come from |
| **Sample Explorer** | Pick one real labelled transaction; see its risk score, how it ranks, and which features drove it |
| **Batch Scoring** | Upload a CSV/Parquet file (or pick a bundled one); get every row ranked, with the top ones flagged as alerts |
| **Feature Playground** | Build a hypothetical transaction from a few inputs and watch the score respond |

**What problem it solves.** The pipeline's results otherwise live in scripts and text
files. The app makes them inspectable and demonstrable in a browser: per-transaction
explanations, alert-budget trade-offs and model behaviour, without running the pipeline.

**How it fits the project.** It only reads the pipeline's output and never
retrains anything. Its scoring reproduces the pipeline's own inference (`train_xgboost.py`
encoding, `predict_batch.py` scoring) exactly. The automated tests check this
against saved `predict_batch.py` output and against an independent copy of the
pipeline's encoding.

```text
pipeline (root scripts) --> processed/ (large, git-ignored)
                                 |
                  webapp/build_assets.py   (run by hand when the model changes)
                                 v
          webapp/model/*.json + webapp/sample_data/*.csv  (small, committed)
                                 v
                 webapp/app.py  -->  browser / Streamlit Community Cloud
```

### Two things to understand before using it

1. **The model can't score a raw transaction on its own.** 22 of its 40 features
   describe each account's activity in the previous 1-24 hours (counts, distinct
   counterparties, fan-in/fan-out, rapid forwarding). Those come from the full
   transaction history, so a single raw row (amount, currency, format) is not enough.
   Batch Scoring therefore needs files that already went through the pipeline, and the
   Playground builds the history fields from what you enter.
2. **Scores are rankings, not probabilities.** Training weighted each laundering
   example about 1,240x, so a score of 0.6 does not mean a 60% chance. The app shows
   how a score ranks against all 762,103 held-out test transactions, and whether it
   would have reached the **top 1,000**, which is the project's alert policy (the
   cut-off was a score of 0.9903). A high score means "investigate", not proof.

---

## 2. Setup and configuration

### Prerequisites
- Python 3.10 or newer. Tested on 3.14 (this project's venv) and 3.12 (Streamlit
  Cloud's default).
- This repository. The app needs only the committed files in `webapp/`. The large
  datasets and `processed/` folders are **not** needed to run it.

### Install

From the project root (`D:\Sem 5\machine learning\lab\ibm aml project`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r webapp\requirements.txt
```

`webapp/requirements.txt` lists only what the app needs:
- streamlit
- XGBoost 3.4+, the version that saved the model: the `xgboost-cpu` build on Linux
  and the normal `xgboost` package on Windows/macOS. On Linux this avoids a large
  CUDA download, and on Windows it won't disturb the pipeline's install.
- pandas and numpy
- pyarrow, for Parquet uploads

The root `requirements.txt` stays the pipeline's dependency list.

### Configuration
- `.streamlit/config.toml` (repo root) sets the upload limit to 100 MB and turns off
  Streamlit's usage statistics. It must stay at the root: Streamlit Cloud only reads
  it there, even though the app is in `webapp/`.
- `webapp/scoring.py`: `MAX_BATCH_ROWS = 200_000` caps uploaded rows to keep memory
  inside Streamlit Cloud's ~1 GB.
- No environment variables, secrets, database or migrations are needed.

---

## 3. Running and deploying

### Run locally

```powershell
cd "D:\Sem 5\machine learning\lab\ibm aml project"
.\.venv\Scripts\python.exe -m streamlit run webapp\app.py
```

Your browser opens `http://localhost:8501`. Stop it with `Ctrl+C`. Run it from the
project root so `.streamlit/config.toml` is picked up.

### Deploy to Streamlit Community Cloud (free)

Streamlit Cloud builds the app from GitHub, so **all** the new files must be
committed. The app fails to start if `webapp/scoring.py` or the new
`webapp/model/*.json` files are missing from the repository.

```powershell
git add .gitignore .streamlit README.md PROJECT_SUMMARY.md WEBSITE_GUIDE.md webapp
git status            # check: scoring.py, build_assets.py, tests/, model/*.json, .streamlit/
git commit -m "Improve AML demo web app"
git push
```

The first deployment needs your own GitHub login in a browser, once:

1. Open **https://share.streamlit.io** and sign in with the GitHub account that owns
   `Kamal-Minocha/ibm-aml-project`.
2. Click **Create app** → **Deploy a public app from GitHub** (wording may vary).
3. Repository `Kamal-Minocha/ibm-aml-project`, branch `main`, main file path
   **`webapp/app.py`**.
4. Optional: under *Advanced settings*, pick Python 3.12 (the default).
5. Click **Deploy**. The first build takes a few minutes. You get a public
   `https://<name>.streamlit.app` URL to share.

Afterwards, every push to `main` redeploys automatically. If you already deployed an
earlier version, pushing these changes is enough; Cloud reinstalls from the new
`webapp/requirements.txt`. Free apps go to sleep when unused and take a few seconds
to wake on the next visit.

---

## 4. Step-by-step usage

### Overview page
Read the headline numbers: PR-AUC 0.468, ROC-AUC 0.983, and 63.5% precision / 40.7%
recall at 1,000 alerts on the held-out test period. These are not typed in:
`build_assets.py` recomputes them by re-scoring the test period, and they match the
pipeline's `final_xgboost_metrics.txt`.

### Sample Explorer
1. Choose **Source**: *HI-Small held-out test period*, or *HI-Medium, Sep 19+*
   (dates after all the training data, never seen by the model).
2. Choose **Show**: *Laundering* or *Normal*.
3. Pick a transaction. Each is listed by time, amount, currency and payment format.
4. Read the result:
   - **Risk score**, and **Higher than X%**: the share of the 762,103 test-period
     transactions that scored lower.
   - **Top-1,000 alert?**: would it have been in the test period's review queue?
   - **Dataset label**: the ground truth.
   - **Why this score**: the 10 features that moved the score most (red raises
     risk, blue lowers it), on the log-odds scale. The caption shows that all 40
     contributions plus the starting point add up exactly to the score.
   - **All 40 feature values**: an expandable table of the raw inputs.

What to expect: laundering rows usually score far above normal ones (HI-Medium: mean
0.91 vs 0.13), but only some laundering rows reach the top 1,000. That matches the
model's ~41% recall at that budget. For example, 3 of the 25 HI-Small laundering
examples would have been alerted.

### Batch Scoring
1. **Upload** a `.csv` or `.parquet` file, **or** pick a bundled sample from the
   dropdown (no upload needed).
2. Set the **Alert budget**, the number of top-ranked rows to flag. It starts at
   1,000, or the row count if the file is smaller. Changing it is instant because
   scores are cached per file.
3. Read the ranked table, highest risk first. `risk_score` is shown as a bar and
   `is_alert` as a checkbox. The first 2,000 rows are displayed.
4. If the file has an `is_laundering` column of 0/1 values, you also get
   **precision** and **recall** at your budget.
5. **Download ranked results (CSV)** saves every row with `risk_score`,
   `alert_rank` and `is_alert`.

**Input format.**
- The file needs all 40 feature columns by name, in any order. Extra columns are ignored.
- These optional columns are carried into the output: `transaction_id`, `timestamp`,
  `from_account`, `to_account`, `is_laundering`.
- CSVs can be comma- or semicolon-separated, UTF-8 or Windows-1252.
- Blank numeric values are allowed and treated as missing, as in training. For
  example, `sender_seconds_since_activity` is blank when an account had no earlier
  activity.

| Situation | Behaviour |
|---|---|
| Raw transaction file / any missing feature column | Red error listing the missing columns; nothing scored |
| Text, infinite, or out-of-range (> 3.4e38) value in a numeric column | Red error naming the column and number of bad rows |
| Category not seen in training (e.g. a new currency) | Scored as "unknown" (code -1), with a yellow warning |
| Blank category (e.g. unmatched account entity type) | Scored with code -1, exactly as training encoded blanks |
| `is_laundering` present but not all 0/1 | Scored normally; precision/recall hidden, with a warning |
| Empty file, or more than 200,000 rows | Red error |
| Corrupt or unreadable file | Red error "Could not read ..." |

### Feature Playground
1. Start from **Typical transaction** (training-data medians and most common values),
   or click **Fan-out with rapid forwarding**. That preset is an account that just
   received money and is paying 40 distinct accounts.
2. Change the fields:
   - amount, payment format, currencies and hour
   - self-transfer, same bank, and entity types
   - for sender and receiver separately: payments made and received in the last 24h,
     how many distinct accounts, transactions in the last hour, and minutes since
     the previous activity
3. The line under the inputs shows the **derived** flags (rapid forwarding, received
   money, same currency). You can't set these directly: the pipeline computes them
   from the counts, and the app does the same.
4. Watch the score, its rank and the explanation chart update. Your inputs are kept
   when you visit another page and come back.

Built-in consistency rules (each holds for 100% of the 5.08M training rows):
- Distinct counterparties can't exceed the payment count, and last-hour activity
  can't exceed 24h activity. Each input's allowed maximum follows from the others.
- **Minutes since previous activity** is limited by the counts:
  - activity in the last hour → at most 60 minutes
  - activity only earlier in the 24h window → 60-1,440 minutes
  - no 24h activity → more than 1,440 minutes
- The four graph "fan" features equal their behaviour counterparts.
- Rapid forwarding = received money in 24h **and** made an earlier payment in 24h.
- **Self-transfer:** sender and receiver are one account, so the receiver's history,
  entity type and bank mirror the sender's. Those inputs are shown greyed out.
- Matching currencies mean matching amounts, and the weekday follows the day of month.

What to expect: the typical transaction scores about 0.0015 (52nd percentile). The
fan-out preset scores about 0.82 (97th percentile), which is a big rise but still
below the 0.9903 needed for the top 1,000.

Limits: day-of-month and account-metadata fields stay at training medians, and 24h
amount totals assume earlier transactions were the same size as this one.

### Bundled test datasets (`webapp/sample_data/`)

All are real rows from the pipeline's output files, with all 40 features and the
`is_laundering` label.

| File | Rows | Content | Use it to |
|---|---:|---|---|
| `quick_demo_mixed_20.csv` | 20 | 10 laundering + 10 normal, shuffled, HI-Small test period | Quickest Batch Scoring demo (8 of the top 10 are laundering) |
| `small_laundering_examples.csv` | 25 | First 25 laundering rows of the HI-Small test period | Sample Explorer; checking catches |
| `small_not_laundering_examples.csv` | 25 | First 25 normal rows of the same period | Checking false alarms (none reach the top 1,000) |
| `medium_mixed_examples.csv` | 200 | 100 + 100 from HI-Medium, Sep 19 onward | Generalisation check: 89 of the top 100 are laundering |

These are about 50% laundering, while real traffic is about 0.1%, so their
precision is far higher than real-world performance. To make your own test file,
delete or combine rows from these CSVs, or filter more rows out of
`processed/transactions_with_graph_features.parquet`.

---

## 5. Technical implementation

### Files

| File | Responsibility |
|---|---|
| `webapp/app.py` | UI only: pages, widgets, charts, caching, playground state. No model logic |
| `webapp/scoring.py` | Pure logic, no Streamlit: read files, load the model files, validate and encode input, score, explain, rank, alert metrics, build consistent playground rows |
| `webapp/build_assets.py` | Regenerates everything in `webapp/model/` and `webapp/sample_data/` from the pipeline output |
| `webapp/model/xgboost_graph_model.json` | Copy of the trained model (`processed/xgboost_graph_model.json`) |
| `webapp/model/model_config.json` | Feature order, categorical columns, alert budget, train/validation cut-offs |
| `webapp/model/category_mappings.json` | Category-to-integer codes learned on the training split |
| `webapp/model/feature_defaults.json` | Training-split medians/modes: the Playground's starting point |
| `webapp/model/score_reference.json` | Test-period score quantiles, top-1,000 cut-off, recomputed metrics |
| `webapp/tests/` | Unit and UI tests (section 7) |
| `.streamlit/config.toml` | Upload limit, telemetry off |

### Scoring path (`scoring.py`)
1. `read_table()` parses the upload (Parquet, or CSV with encoding and delimiter
   fallbacks). Any parse failure becomes a clear validation error.
2. `prepare_batch()`:
   - checks the row count and the required columns
   - coerces numeric columns to float32, rejecting text, infinity and values beyond
     float32's range
   - maps categories with the training codes: blanks and unseen values become -1.
     That matches the pipeline's `replace_strict(mapping, default=-1)`. (The
     `__MISSING__` entry in the mapping exists only because the pipeline builds the
     mapping that way; no row is ever encoded with it.)
3. `score()` calls `Booster.predict` limited to `iteration_range=(0, best_iteration+1)`.
   Training used early stopping, which keeps all 451 trees in the file but marks
   tree 401 as the best. `predict_proba` honours that mark automatically;
   `Booster.predict` doesn't unless told.
4. `contributions()` uses XGBoost's built-in TreeSHAP (`pred_contribs=True`) over the
   same tree range, so the contributions always add up to the score.
5. `rank_alerts()` gives rank 1 to the top score (ties keep file order) and flags
   the top *budget* rows, the same policy as `finalize_xgboost.py`.
6. `build_playground_row()` starts from the training defaults, applies your
   inputs, then enforces the consistency rules from section 4.

### Key design decisions
- **Streamlit, single process.** One Python file serving one model. It needs no
  separate API, frontend or database, and deploys free from GitHub.
- **Logic separated from UI** so it can be unit-tested and reused, and so the UI
  file stays readable.
- **Precomputed, committed model files** instead of reading the 300 MB Parquet at
  start-up: the hosted app starts in seconds and needs no private data.
- **XGBoost's built-in TreeSHAP instead of the `shap` package**: same values, no
  numba/llvmlite dependency.
- **Percentile and top-1,000 test** instead of fixed "high/low risk" bands, because
  the scores aren't calibrated probabilities.
- **Caching:** the model is loaded once (`st.cache_resource`), and each uploaded
  file is parsed and scored once (`st.cache_data`, keyed by file content).
- **Playground state:** Streamlit forgets values of widgets that aren't on screen.
  The page keeps its own copy, so inputs survive page switches and hidden fields.

### Updating the model
After retraining with `train_xgboost.py` and `finalize_xgboost.py`:

```powershell
.\.venv\Scripts\python.exe webapp\build_assets.py
.\.venv\Scripts\python.exe -m pytest webapp\tests -q -p no:cacheprovider
```

The build needs `processed/` (and optionally
`processed_medium/medium_model_ready_combined.parquet` for the Medium sample), plus
polars and scikit-learn from the root requirements. Each output file is replaced only
once it is complete, so a failed build leaves the previous files working. Some tests
have fixed expected values (e.g. PR-AUC 0.468044); update those deliberately if the
model changes.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: streamlit` | Web app dependencies not installed in the active environment | `.\.venv\Scripts\python.exe -m pip install -r webapp\requirements.txt` |
| `streamlit` is not recognised | Scripts folder not on PATH | Use `.\.venv\Scripts\python.exe -m streamlit run webapp\app.py` |
| `ModuleNotFoundError: scoring` on Streamlit Cloud | `webapp/scoring.py` wasn't committed | `git add webapp` and push (section 3) |
| "The file is missing N of the 40 required feature columns" | Raw or partially processed data | Run the pipeline up to `add_graph_features.py`, or use a bundled sample |
| "must hold ordinary numbers but contain text..." | e.g. an Excel export with `$1,000` or `N/A` | Export plain numbers; leave blanks empty |
| Upload refused above 100 MB | `server.maxUploadSize` | Use Parquet (much smaller) or split the file |
| Port 8501 in use | Another Streamlit app is running | Close it, or add `--server.port 8502` |
| Cloud build fails | Unsupported Python version picked | Choose Python 3.12 in Advanced settings |
| Cloud app slow on first visit | Free apps sleep when idle | Wait for it to wake; later loads are fast |
| `build_assets.py`: "Missing pipeline output" | `processed/` not generated on this machine | Run the pipeline (see README) first |

On Linux, don't install `webapp/requirements.txt` into an environment that already
has the regular `xgboost` package: both packages provide the `xgboost` module and
would conflict. A fresh environment, like Streamlit Cloud's, is fine.

**Known limitations:**
- It is a single-model demo, not a production service. There is no authentication,
  so a public Cloud app is open to anyone with the link. Only bundled synthetic data
  ships with it, and uploads are processed in memory and not stored.
- It scores only pipeline-processed data.
- Playground results are illustrative. It doesn't model same-timestamp activity
  (a minutes-since value of exactly 0) or "no activity ever" (blank).

---

## 7. Testing

```powershell
.\.venv\Scripts\python.exe -m pip install pytest scikit-learn   # test-only
.\.venv\Scripts\python.exe -m pytest webapp\tests -q -p no:cacheprovider
```

44 tests:
- **`test_scoring.py` (24)**:
  - Scores match `XGBClassifier.predict_proba`, and match the saved `predict_batch.py`
    outputs for the two HI-Small sample files and the 200-row HI-Medium batch.
  - Encoding matches an independent copy of the pipeline's Polars encoding, with
    blank and unseen categories. The 56 real HI-Small rows with blank entity types
    score exactly as the pipeline scores them.
  - The recomputed metrics match the pipeline's (PR-AUC 0.468044, 635 true positives).
  - Explanations add up to the score; percentile and cut-off are consistent.
  - File parsing: comma, semicolon, BOM, Windows-1252, Parquet and corrupt files.
  - Rejection of missing columns, raw data, empty or oversized files, and text,
    infinite or float32-overflowing numbers; blank numbers allowed; invalid labels
    ignored.
  - Ranking ties and budget edge cases.
  - 200 random Playground inputs all satisfy every consistency rule.
- **`test_app.py` (20)**, run headless via Streamlit's `AppTest`:
  - Every page renders, and the Overview metrics are correct.
  - Sample Explorer on Medium data.
  - Each bundled sample in Batch Scoring, including budget changes.
  - Uploads: a one-row file, Parquet, a raw file (rejected) and a corrupt file
    (rejected).
  - Playground: the preset raises the score, dependent counts clamp, minutes-since
    follows the counts, self-transfer mirrors the sender, and values survive page
    switches and currency toggles.

The comparisons against pipeline output skip automatically when the git-ignored
`processed/` files aren't present. The scikit-learn-based checks skip without
scikit-learn.

Verified:
- On Python 3.14 (project venv).
- On clean Python 3.12 installs, with both `xgboost` and `xgboost-cpu`.
- Running the app server locally.

Not verified from this machine:
- The actual Streamlit Cloud deployment, which needs your login.
- Running on Linux itself.
- How the pages look in a real browser.

---

## 8. Changes compared to the first (Sonnet) version

| Area | First version | Now | Why |
|---|---|---|---|
| Explanation chart | Summed all 451 trees while the score used 401 | Same tree range for both | The "why" chart didn't explain the score shown (e.g. score 0.53, explanation 0.40) |
| Decision display | "Model agrees" if score ≥ 0.5; risk bands at 0.1/0.5/0.9; called a "probability" | Percentile vs test period + "top-1,000 alert?" | Scores aren't calibrated; 0.5 had no meaning in this project |
| Playground | Inputs could contradict each other: rapid forwarding with no money received; fan-out not equal to recipients; minutes-since unrelated to the counts; self-transfers with two different histories; even the default was impossible | Derived fields computed by the pipeline's rules; dependent limits; mirrored self-transfers | The model was being asked about transactions that can't exist |
| Batch scoring | Re-scored the file on every click; `st.slider` crashed on 1-row files; Styler failed on large files | Cached scoring; number input; column config; 200k-row cap | Speed and crash fixes |
| Validation and parsing | Any text in a numeric column crashed XGBoost; semicolon or non-UTF-8 CSVs failed | Clear errors and warnings (table in section 4); delimiter and encoding fallbacks | Usable messages, common Excel exports work |
| Structure | 666-line `app.py` with logic and UI mixed; no tests | `scoring.py` + `app.py` + `build_assets.py`; 44 tests | Testable and maintainable |
| Model files | Hand-copied with one-off commands; config with Windows paths; defaults from all data, including the test period | Regenerated by `build_assets.py` from the training split; metrics recomputed | Reproducible; no test-period data in the defaults |
| Dependencies | plotly; unpinned `xgboost` (pulls a large CUDA package on Linux) | Altair (bundled with Streamlit); `xgboost-cpu` on Linux, xgboost 3.4+ | Smaller, faster Cloud build; model loads in the installed version |
| Config | none | Root `.streamlit/config.toml` (100 MB uploads, telemetry off) | Cloud reads config only from the root |
| Docs | Said deployment was "actually used" and guessed a URL; generic expectations | This guide, with measured behaviour | Accuracy |

**Kept:**
- Streamlit and Community Cloud, and the four-page layout
- XGBoost's built-in TreeSHAP and the bundled-sample approach
- the `.gitignore` rules
- the 37 → 40 feature-count correction, now also fixed in `README.md` and
  `PROJECT_SUMMARY.md`
- the first version's effective handling of blank categories (code -1). It turned a
  blank into the text `"nan"`, which isn't in the mapping, so it fell through to -1.
  That happens to match training. The behaviour is now deliberate, and it is
  tested against the pipeline's encoding.

**Removed:** `webapp/model/feature_stats.json` and `webapp/model/final_xgboost_config.json`,
replaced by `feature_defaults.json` and `model_config.json`, plus the plotly dependency.
