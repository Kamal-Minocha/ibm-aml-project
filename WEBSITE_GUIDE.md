# AML Risk Scoring Demo — Website Guide

This guide explains the interactive website built for the IBM Anti-Money-Laundering
project, how it was deployed, and exactly how to use it — step by step — including
where to get test data to feed into it.

The app lives in [`webapp/`](webapp/) and is built with **Streamlit**, a Python
framework that turns a script into a web app with no separate frontend/backend.

> **What the website is for:** it is a teaching/demo front end for the already-trained
> final XGBoost model (`webapp/model/xgboost_graph_model.json`). It does **not** retrain
> anything and it does **not** connect to a live bank feed — every prediction it shows
> comes from that one saved model file.

---

## 1. What's on the website

The site has four pages, picked from the sidebar:

| Page | What it does |
|---|---|
| 🏠 **Overview** | Explains the project, the pipeline, and the headline test-set results |
| 🧾 **Sample Explorer** | Pick one real, already-labeled transaction and see the model's score + explanation |
| 📂 **Batch Scoring** | Upload a file of many transactions and get them ranked by risk |
| 🎛️ **Feature Playground** | Drag sliders on a hypothetical transaction and watch the risk score react live |

---

## 2. Step-by-step: using the live website

### Step 1 — Open the site

Go to the URL you get after deployment (see Part 3 below), for example:

```text
https://ibm-aml-project.streamlit.app
```

It may take 10-30 seconds to "wake up" the first time if nobody has visited recently
(Streamlit Cloud's free tier sleeps idle apps).

### Step 2 — Read the Overview page

This is the landing page. It shows the pipeline, the model used, and the key numbers:
PR-AUC 0.468, precision 63.5% and recall 40.7% at the production alert budget of 1,000
alerts. Point your teacher here first — it is the "executive summary" of the whole project.

### Step 3 — Try the Sample Explorer

1. In the sidebar, click **🧾 Sample Explorer**.
2. Choose a **transaction source**:
   - *Small dataset* — from the model's own held-out test period (Sep 9–18, 2022).
   - *Medium dataset* — from Sep 19 onward, data the model never saw at all during
     training or testing. This is the stronger "does it generalise?" demonstration.
3. Choose **Known laundering** or **Known normal** to filter the dropdown by ground truth.
4. Pick any transaction from the dropdown (it's labeled with its amount, currency and
   payment method so you can pick an interesting one).
5. You'll see:
   - A **gauge** showing the model's predicted laundering probability.
   - A **risk badge** (🟢 low → 🔴 very high).
   - The **actual label** from the dataset, with "model agrees / disagrees".
   - A **bar chart (SHAP explanation)**: which of the 40 features pushed the score up
     (red) or down (blue) for this specific transaction. This is the most convincing
     part to show a teacher — it answers "why did the model think this?"
   - An expandable table of all 37 raw feature values, for anyone who wants to inspect
     the exact numbers.

**Suggested demo flow:** show one known-laundering transaction with a high score and
its SHAP chart (point out `rapid_forwarding_indicator`, fan-in/fan-out features, or
unusual transaction counts lighting up red), then show one known-normal transaction
with a low score, then switch to the Medium dataset and repeat — showing the model
still works on data it never trained on.

### Step 4 — Try Batch Scoring

1. Click **📂 Batch Scoring** in the sidebar.
2. Either:
   - Upload one of the CSV files from `webapp/sample_data/` (see Part 4 below), **or**
   - Use the dropdown "Or load a bundled example instead of uploading" to pick one of
     the same files directly inside the app (no download/upload round-trip needed).
3. The app scores every row and shows a **ranked table**, highest risk first, with the
   top rows (up to your alert budget) highlighted red as "alerts".
4. Drag the **alert budget slider** to see how precision and recall trade off as you
   review more or fewer alerts (this mirrors `alert_budget_analysis.py` from the
   original pipeline, live).
5. Because the sample files include the true `is_laundering` label, the app also shows
   **precision and recall** for your chosen budget — real unseen transactions wouldn't
   have that column, so this comparison only works because these are labeled test
   examples.
6. Click **⬇ Download full ranked results as CSV** to save the output.

### Step 5 — Try the Feature Playground

1. Click **🎛️ Feature Playground**.
2. Everything starts at a "typical" transaction (median values from the training data).
3. Adjust the sliders/checkboxes — amount, payment format, how many recipients the
   sender paid in the last 24h, whether the sender just received money and is now
   forwarding it ("rapid forwarding"), how many distinct senders paid the receiver, etc.
4. Watch the gauge and SHAP chart update live as you change each field.

**Suggested demo:** start at the default (low risk), then turn on "rapid forwarding",
raise "distinct recipients the sender paid" and "distinct senders who paid the
receiver" to high values — this mimics a fan-out/fan-in laundering pattern — and watch
the risk score climb from green to red. This is the clearest way to show a teacher
*what the model actually learned*, without needing a real suspicious transaction on hand.

> This page is explicitly a teaching sandbox. A single typed-in transaction is missing
> most of the real account history the model was trained on, so treat the numbers here
> as "what-if" illustrations, not real predictions on a new transaction.

---

## 3. How it was built and deployed

- **Framework:** Streamlit (pure Python, no separate frontend).
- **Model:** the same `xgboost_graph_model.json` produced by `train_xgboost.py` /
  `finalize_xgboost.py` in the main pipeline — copied into `webapp/model/` so the web
  app is self-contained.
- **Explanations:** SHAP's `TreeExplainer`, which is exact and fast for tree models
  like XGBoost.
- **Hosting:** Streamlit Community Cloud — free, builds directly from the GitHub repo,
  and redeploys automatically whenever `main` is pushed.

### Deployment steps actually used

1. All the code (`webapp/app.py`, `webapp/requirements.txt`), the model file, its
   config, and small sample datasets were committed to the `main` branch of
   `https://github.com/Kamal-Minocha/ibm-aml-project`.
2. The large raw datasets and pipeline output folders stay out of GitHub (see the
   `.gitignore` section below) — the web app only needs the small files in `webapp/`.
3. **To finish deployment (one-time, needs your own GitHub login):**
   1. Go to **https://share.streamlit.io**.
   2. Sign in with your GitHub account (the same one that owns `ibm-aml-project`).
   3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
   4. Repository: `Kamal-Minocha/ibm-aml-project`. Branch: `main`. Main file path:
      `webapp/app.py`.
   5. Click **Deploy**. The first build takes 2-5 minutes (it installs the packages
      in `webapp/requirements.txt`).
   6. You'll get a public URL like `https://ibm-aml-project.streamlit.app` (or a
      variant if that name is taken) — share that link with your teacher.

This last step needs your own GitHub/Streamlit login through the browser, so it has to
be done by you once, interactively. Every future `git push` to `main` will automatically
redeploy the same app at the same URL — no need to repeat these steps.

---

## 4. Test datasets to use with the website

All of these already contain the full 40 engineered features the model needs, so you
can feed them straight into **Batch Scoring**, or they're what powers **Sample Explorer**.
They live in `webapp/sample_data/` in the repository.

| File | Rows | Content | Best used for |
|---|---:|---|---|
| `small_laundering_examples.csv` | 25 | Known laundering transactions, from the model's held-out **Small** test period | Sample Explorer "known laundering"; a quick "does it catch obvious cases?" batch upload |
| `small_not_laundering_examples.csv` | 25 | Known normal transactions, same Small test period | Sample Explorer "known normal"; checking false-positive behaviour |
| `quick_demo_mixed_20.csv` | 20 | 10 laundering + 10 normal, shuffled together, from Small | The fastest single-file demo for Batch Scoring — one upload, mixed ground truth, instant precision/recall |
| `medium_mixed_examples.csv` | 200 | 100 laundering + 100 normal, from the **Medium** dataset, **Sep 19 onward** — dates the model never trained or was tested on | The strongest "generalises to unseen data" demonstration; shows ~89/100 laundering transactions correctly ranked in the top 100 alerts |

**Where these came from:** they're real rows from `HI-Small_Trans.csv` and
`HI-Medium_Trans.csv`, pushed through the full project pipeline (cleaning → account
join → behaviour features → graph features), so every column the model expects is
already present and correct — no extra preprocessing needed before uploading them.

**To test with your own subset:** open any of these CSVs, delete some rows, or mix
rows from both files, and re-upload — the app only needs the 40 feature columns plus
(optionally) `is_laundering` for the precision/recall comparison.

**Important:** don't try to build a new test row by hand with just amount/currency/payment
format — the app will reject it with a list of missing columns, because the model needs
the behavioural and network history columns too. Use the **Feature Playground** instead
for "what if I change this one thing" exploration.

---

## 5. What's public on GitHub vs. what's not

The repository is public (so Streamlit Cloud can build from it and your teacher can
browse the code), but the `.gitignore` keeps the following out of GitHub:

- The multi-gigabyte raw CSVs (`HI-Small_Trans.csv`, `HI-Medium_Trans.csv`, etc.) and
  every generated pipeline file (`processed/`, `processed_medium/`) — too large for
  GitHub and easy to regenerate from the scripts.
- `chat-readable.md`, `starting.md`, `chat.json` — private planning/chat exports, not
  meant for a public audience.
- Any `.env`, `credentials.json`, `secrets.json`, or `.streamlit/secrets.toml` — in case
  any are ever added; none are needed by this app today, but the rule is there as a
  safety net.
- Python/IDE junk (`__pycache__/`, `.venv/`, `.vscode/`, etc.).

Everything under `webapp/` — the app code, the model file, its config, and the sample
CSVs — **is** committed, because Streamlit Cloud needs them to build and run the site.
