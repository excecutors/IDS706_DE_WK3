[![CI](https://github.com/excecutors/IDS706_DE_WK3/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/excecutors/IDS706_DE_WK3/actions/workflows/ci.yml)

## Note for WK5 Graders
Scroll down to see the Week 5 updates. Everything above reflects the Week 3 submission (kept unchanged for ease of Week 3 grading as of September 2025). The new Week 5 Commentary begins below the Week 5 header.


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

`analysis.py` supports **two** sources, in this order:

1. `DATA_PATH` exists → load the **local CSV** (default: `data/gold_data_2015_25.csv`).
2. Else → fallback to **S3**

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

## Run Locally

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

## Run in GitHub Codespaces (recommended for quick checks)

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
  * 

---

## Week 5 Addition (Refactoring + CI polish)

In this iteration, the project was made more **professional and reproducible**:

* **Refactoring**
  * Renaming (for example, we renamed `df` → `prices` for clarity)
  * Extract method/variable (for example, we extracted plotting code into clean functions)

* **Code Quality**
  * Added **Black** for auto-formatting.
  * Added **Flake8** for linting.
  * Fixed style issues (line length, blank lines, newline at EOF) so CI passes.

* **Continuous Integration**
  * Configured GitHub Actions to run `black --check`, `flake8`, and `pytest`.
  * Added a CI badge to the README (see top of file).

These changes improve readability and maintainability, making sure that every push is automatically tested and style-checked.
