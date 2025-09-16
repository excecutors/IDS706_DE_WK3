# IDS706 Week 3 — Reproducible Data Analysis (Polars + ML + Tests + Docker)

This repo contains a **reproducible** and **testable** data analysis workflow using:

* **Local CSV** by default (no cloud creds needed)
* Optional **AWS S3** fallback (owner only) if you export an env var
* **Polars** for data wrangling
* **Matplotlib/Seaborn** for plots
* **scikit‑learn & XGBoost** for ML
* **pytest** for tests
* **Docker** for fully pinned runtime

Dataset: daily gold & market features (**SPX, GLD, USO, SLV, EUR/USD**) for **2015–2025**.

---

## Repo Structure

```
IDS706_DE_WK3/
├─ analysis.py          # Main pipeline; prints stats, trains models, saves plots
├─ requirements.txt     # Python deps
├─ Dockerfile           # Container recipe
├─ Makefile             # Local + Docker shortcuts
├─ data/
│  └─ gold_data_2015_25.csv  # Local copy of dataset (used by default)
├─ outputs/             # Plots written here (created on first run)
├─ tests/
│  └─ test_analysis.py  # Unit + system tests
└─ README.md
```

### Key files (what they do)

* **`analysis.py`** – Loads the CSV, cleans/types, filters & groups, builds plots, trains **LinearRegression** and **XGBRegressor**, prints metrics, saves plots to `outputs/`.
* **`Makefile`** – One‑liners for installing deps, running tests, building/running Docker, etc.
* **`Dockerfile`** – Minimal Python 3.11 slim image + project deps.
* **`tests/test_analysis.py`** – Verifies the pipeline (data shape/columns, filtering, grouping, ML metrics, and that plots are produced).

---

## How `analysis.py` finds data (no code changes needed)

`analysis.py` supports **three** sources, in this order:

1. If `USE_S3=="1"` → load from **S3** (owner only) using `BUCKET/KEY/REGION` env vars.
2. Else if `DATA_PATH` exists → load the **local CSV** (default: `data/gold_data_2015_25.csv`).
3. Else → fallback to **S3** (you may remove this fallback if you want to forbid S3 for graders).

Environment variables you can use (all optional):

```bash
# Defaults used by the script if not provided
DATA_PATH=data/gold_data_2015_25.csv
USE_S3=0                 # set to 1 to force S3
BUCKET=kaggle-gold-dataset
KEY=gold_data_2015_25.csv
REGION=us-east-2
```

> **For graders**: You do **not** need S3. The included `data/gold_data_2015_25.csv` will be used automatically (no env vars required).

---

## Run Locally (recommended for quick checks)

```bash
# 1) Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the pipeline
python analysis.py
```

Outputs:

* Plots saved to `outputs/gld_hist.png` and `outputs/spx_vs_gld.png`
* Printed schema/describe/head
* Train/Test metrics for Linear Regression & XGBoost

### Makefile shortcuts (local)

```bash
make install     # create venv + install deps
make format      # black *.py
make lint        # flake8 *.py
make test        # pytest -vv
make run-local   # python analysis.py
make clean       # remove caches + outputs
```

---

## Docker (fully reproducible)

Build the image (one time or after changes):

```bash
docker build -t ids706-wk3 .
```

Run the container and save plots back to your host:

```bash
docker run -it --name ids706-wk3 \
  -v $(pwd)/outputs:/app/outputs \
  ids706-wk3:latest
```

> The container defaults to **local CSV**. If you want to force S3 (owner only):

```bash
docker run -it --rm \
  -e USE_S3=1 -e BUCKET=kaggle-gold-dataset -e KEY=gold_data_2015_25.csv -e REGION=us-east-2 \
  ids706-wk3:latest
```

Helpful Docker commands:

```bash
docker ps -a            # list containers (running + stopped)
docker images           # list images
docker rm -f ids706-wk3 # remove container
docker rmi -f ids706-wk3# remove image
```

### Makefile shortcuts (Docker)

```bash
make docker-build     # docker build -t ids706-wk3 .
make docker-run       # run with outputs mounted
make image_show       # docker images
make container_show   # docker ps -a
make docker-clean     # remove container + image
```

---

## Tests (what we verify)

We use `pytest` to check:

* **DataFrame integrity** – correct columns & reasonable row count
* **Filtering** – `high_gold` truly satisfies `GLD > 180`
* **Grouping** – yearly averages exist and are numeric
* **ML sanity** – Linear Regression & XGBoost run and produce reasonable R²/RMSE
* **Artifacts** – plots are written to `outputs/`

Run tests locally:

```bash
pytest -q
```

Run tests inside Docker:

```bash
docker run -it --rm ids706-wk3:latest pytest -q
```

---

## Example Results (typical)

* **Linear Regression**: R² ≈ 0.92, RMSE ≈ 12.8
* **XGBoost**: R² ≈ 0.99, RMSE ≈ 3.8 (excellent fit, no major overfitting)
* **Plots**:

  * Histogram of gold prices → `outputs/gld_hist.png`
  * SPX vs GLD scatter plot → `outputs/spx_vs_gld.png`

---

## Key Takeaways

* **Docker** guarantees reproducibility → same results across machines.
* **Testing** validates correctness of data pipeline and models.
* **Polars** provides fast, efficient data handling.
* **XGBoost** outperforms Linear Regression significantly.

## Run in GitHub Codespaces (Step‑by‑Step)

> This repo is already set up to run from the **local CSV** in `data/`. You don’t need AWS.

### 1) Open the repo in a Codespace

* Go to your GitHub repository page.
* Click **Code ▸ Codespaces** tab ▸ **Create codespace on main** (or your branch).
* Wait for the container to build and Github Codespace to open in the browser.

### 2) Confirm you’re inside the dev container

```bash
[ -f /.dockerenv ] && echo "Inside container" || echo "Not in container"
echo $CODESPACES   # should print: true
python --version
```

### 3) Install Python dependencies

```bash
pip install -r requirements.txt
```

> If `pip` is not found, try `pip3`. If you prefer isolation: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

### 4) Run the tests

```bash
pytest -q
```

* You should see all tests pass.

### 5) Run the analysis script

```bash
python analysis.py
```

* The script prints dataset diagnostics to the terminal and saves plots into **`outputs/`**:

  * `outputs/gld_hist.png`
  * `outputs/spx_vs_gld.png`