# IDS706 Week 3 Data Engineering Mini Project

## Overview

This project demonstrates a full **data engineering workflow** that is both **reproducible** and **testable**. It integrates:

* **Local CSV / AWS S3 (owner only)** for dataset loading
* **Polars** for data processing
* **Matplotlib/Seaborn** for visualization
* **Scikit-Learn & XGBoost** for machine learning
* **Pytest** for automated testing
* **Docker** for reproducibility and environment setup

The dataset contains gold prices and related market data (**SPX, GLD, USO, SLV, EUR/USD**) from **2015–2025**. The CSV file is included in this repo under `data/`, but the project also supports loading via a public HTTPS URL or AWS S3 bucket.

---

## Repository Structure

```
IDS706_DE_WK3/
│── analysis.py         # Core Python script (runs full pipeline)
│── requirements.txt    # Python dependencies
│── Dockerfile          # Docker environment definition
│── Makefile            # Helper shortcuts for Docker commands
│── data/               # Contains gold_data_2015_25.csv (local dataset)
│── outputs/            # Saved plots and results
│── tests/              # Pytest test suite
│── README.md           # Project documentation
```

### File Purpose

* **analysis.py** → Main script. Loads data (local `data/` by default), filters/groups, runs ML, saves plots.
* **requirements.txt** → Lists Python libraries to install.
* **Dockerfile** → Instructions to build a Docker image with all dependencies.
* **Makefile** → Shortcuts to build, run, and clean Docker containers.
* **tests/** → Contains `test_analysis.py` with unit/system tests.
* **outputs/** → Where plots (`gld_hist.png`, `spx_vs_gld.png`) are saved.

---

## Running the Project

### **1. Local (without Docker)**

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python analysis.py
```

Plots will be saved in the `outputs/` folder.

---

### **2. With Docker** (Reproducible setup)

#### Step 1: Build the Docker Image

```bash
docker build -t ids706-wk3 .
```

* `ids706-wk3` is the name of the image
* Dockerfile ensures dependencies are installed in a clean environment

#### Step 2: Run the Container (default = local CSV)

```bash
docker run -it \
  -v $(pwd)/outputs:/app/outputs \
  ids706-wk3:latest
```

* `-v $(pwd)/outputs:/app/outputs` → saves plots locally to `outputs/`

#### Step 3: Inspect Containers & Images

```bash
docker ps -a              # list all containers
docker images             # list all built images
docker rm <container_id>  # remove container
docker rmi <image_id>     # remove image
```

#### Step 4: Using the Makefile (shortcuts)

```bash
make build        # builds the Docker image
make run          # runs the container
make clean        # removes the image
make image_show   # shows images
make container_show # shows containers
```

---

## Testing

### **Why Testing?**

We test to:

* Confirm data loads correctly
* Validate filtering & grouping logic
* Ensure ML models run and produce reasonable metrics
* Guarantee plots are saved

### **Run Tests Locally**

```bash
# Inside virtual environment
pytest -q
```

### **Run Tests in Docker**

```bash
docker run -it ids706-wk3:latest pytest -q
```

### **Test Coverage**

* **End-to-End Execution** → Does `analysis.py` run without errors?
* **DataFrame Integrity** → Correct columns + sufficient rows
* **Filtering & Grouping** → `high_gold` has only GLD > 180; yearly averages computed
* **ML Metrics** → Linear Regression + XGBoost R² & RMSE are within reasonable range
* **Idempotency** → Running multiple times gives same outputs

---

## Results

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

## Docker App
<img width="2520" height="1432" alt="image" src="https://github.com/user-attachments/assets/4bf57b4d-ec1d-4994-972e-f7109a4e2992" />
