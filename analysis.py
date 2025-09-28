#!/usr/bin/env python
# coding: utf-8

from io import StringIO
import os

import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# ## Pull from AWS S3

# In[4]:

# -----------------------------
# Config (defaults = local CSV)
# -----------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/gold_data_2015_25.csv")
USE_S3 = os.getenv("USE_S3", "0")  # "1" to force S3, otherwise local

BUCKET = os.getenv("BUCKET", "kaggle-gold-dataset")
KEY = os.getenv("KEY", "gold_data_2015_25.csv")
REGION = os.getenv("REGION", "us-east-2")

os.makedirs("outputs", exist_ok=True)


def _load_from_s3(bucket: str, key: str, region: str) -> pl.DataFrame:
    import boto3

    s3 = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    csv_text = obj["Body"].read().decode("utf-8")
    return pl.read_csv(StringIO(csv_text))


# ---------------------------------------
# Resolution order:
#   1) If USE_S3 == "1" -> use S3
#   2) Else try local CSV (DATA_PATH)
#   3) Else (optional) fall back to S3
# ---------------------------------------
if USE_S3 == "1":
    prices = _load_from_s3(BUCKET, KEY, REGION)
elif os.path.exists(DATA_PATH):
    prices = pl.read_csv(DATA_PATH)
else:
    # last-resort fallback
    prices = _load_from_s3(BUCKET, KEY, REGION)

# ## Inspect the Data

# In[5]:


print(prices.head())


# In[6]:


print(prices.schema)


# In[7]:


print(prices.describe())


# In[8]:


print(prices.shape)


# ## Basic Filtering & Grouping

# In[9]:


# filter: GLD > 180
high_gold = prices.filter(pl.col("GLD") > 180)
print(high_gold.shape)


# In[10]:


# Normalize Date dtype (works whether it's already date or a string)
prices = prices.with_columns(pl.col("Date").cast(pl.Date))

# Add Year column
prices = prices.with_columns(pl.col("Date").dt.year().alias("Year"))

# Group by Year and compute average GLD
yearly_avg_gld = prices.group_by("Year").agg(pl.col("GLD").mean().alias("avg_GLD"))
print(yearly_avg_gld)


# ## Visualization

# In[11]:


def plot_gld_hist(prices: pl.DataFrame, outpath: str = "outputs/gld_hist.png") -> None:
    gld_values = prices["GLD"].to_numpy()
    sns.histplot(gld_values, bins=30, kde=True)
    plt.title("Distribution of Gold Prices (2015–2025)")
    plt.xlabel("GLD Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.clf()


def plot_spx_vs_gld(
    prices: pl.DataFrame, outpath: str = "outputs/spx_vs_gld.png"
) -> None:
    spx = prices["SPX"].to_numpy()
    gld = prices["GLD"].to_numpy()
    sns.scatterplot(x=spx, y=gld)
    plt.title("SPX vs Gold (GLD)")
    plt.xlabel("S&P 500 (SPX)")
    plt.ylabel("Gold (GLD)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.clf()


os.makedirs("outputs", exist_ok=True)
plot_gld_hist(prices)
plot_spx_vs_gld(prices)


# ## ML Exploration

# In[12]:

# Convert Polars → NumPy for sklearn
features = prices.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
target = prices["GLD"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Fit linreg_model
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)

# Predictions
y_pred = linreg_model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R² score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


# In[13]:

# Features/target from Polars to NumPy
features = prices.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
target = prices["GLD"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Fit XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=500,  # number of trees
    learning_rate=0.05,  # step size shrinkage
    max_depth=4,  # complexity of trees
    subsample=0.8,  # row sampling (regularization)
    colsample_bytree=0.8,  # feature sampling (regularization)
    random_state=42,
    reg_lambda=1,  # L2 regularization
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train)

# Predictions
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)


# Metrics
def eval_metrics(y_true, y_pred, label="Test"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{label} R²: {r2:.4f}")
    print(f"{label} MSE: {mse:.4f}")
    print(f"{label} RMSE: {rmse:.4f}")
    print("-" * 30)


eval_metrics(y_train, y_train_pred, "Train")
eval_metrics(y_test, y_test_pred, "Test")
