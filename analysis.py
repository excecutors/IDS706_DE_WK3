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
    df = _load_from_s3(BUCKET, KEY, REGION)
elif os.path.exists(DATA_PATH):
    df = pl.read_csv(DATA_PATH)
else:
    # last-resort fallback
    df = _load_from_s3(BUCKET, KEY, REGION)

# ## Inspect the Data

# In[5]:


print(df.head())


# In[6]:


print(df.schema)


# In[7]:


print(df.describe())


# In[8]:


print(df.shape)


# ## Basic Filtering & Grouping

# In[9]:


# filter: GLD > 180
high_gold = df.filter(pl.col("GLD") > 180)
print(high_gold.shape)


# In[10]:


# Normalize Date dtype (works whether it's already date or a string)
df = df.with_columns(pl.col("Date").cast(pl.Date))

# Add Year column
df = df.with_columns(pl.col("Date").dt.year().alias("Year"))

# Group by Year and compute average GLD
yearly_avg = df.group_by("Year").agg(pl.col("GLD").mean().alias("avg_GLD"))
print(yearly_avg)


# ## Visualization

# In[11]:


# Histogram -- save plots as files
gld = df["GLD"].to_numpy()
sns.histplot(gld, bins=30, kde=True)
plt.title("Distribution of Gold Prices (2015–2025)")
plt.xlabel("GLD Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/gld_hist.png", dpi=150)
plt.clf()  # clear the figure for the next plot

# Scatter: SPX vs GLD -- save plots as files
spx = df["SPX"].to_numpy()
gld = df["GLD"].to_numpy()
sns.scatterplot(x=spx, y=gld)
plt.title("SPX vs Gold (GLD)")
plt.xlabel("S&P 500 (SPX)")
plt.ylabel("Gold (GLD)")
plt.tight_layout()
plt.savefig("outputs/spx_vs_gld.png", dpi=150)
plt.clf()


# ## ML Exploration

# In[12]:

# Convert Polars → NumPy for sklearn
X = df.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
y = df["GLD"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R² score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


# In[13]:

# Features/target from Polars to NumPy
X = df.select(["SPX", "USO", "SLV", "EUR/USD"]).to_numpy()
y = df["GLD"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit XGBoost Regressor
model = XGBRegressor(
    n_estimators=500,  # number of trees
    learning_rate=0.05,  # step size shrinkage
    max_depth=4,  # complexity of trees
    subsample=0.8,  # row sampling (regularization)
    colsample_bytree=0.8,  # feature sampling (regularization)
    random_state=42,
    reg_lambda=1,  # L2 regularization
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


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
