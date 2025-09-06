"""
Sales Data Wrangling & Predictive Modeling — Single-File Pipeline
-----------------------------------------------------------------
Run:
  python sales_pipeline.py

Optional:
  Place your own CSV at ./sales_raw.csv with similar columns to skip synthetic generation.

Outputs:
  - sales_clean.parquet
  - model.pkl
  - metrics.json
  - heatmap.png
  - actual_vs_pred.png
  - residuals_vs_fitted.png
  - qq_plot.png
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Plotting
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Modeling
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------
# Config
# -------------------------------
RAW_CSV = Path("sales_raw.csv")
CLEAN_PARQUET = Path("sales_clean.parquet")
MODEL_PKL = Path("model.pkl")
METRICS_JSON = Path("metrics.json")

TARGET = "revenue"
NUMERIC = ["ad_spend","customer_age","median_income","site_visits","prior_month_sales","month"]
CATEG = ["region","channel","customer_segment"]

# -------------------------------
# Synthetic data generator
# -------------------------------
def make_synthetic(n_rows=50000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2021, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365*3, size=n_rows)]
    months = np.array([d.month for d in dates])

    regions = rng.choice(["north","south","east","west"], size=n_rows, p=[0.25,0.25,0.25,0.25])
    channels = rng.choice(["online","retail","partner","phone"], size=n_rows, p=[0.5,0.25,0.2,0.05])
    segments = rng.choice(["consumer","smb","enterprise"], size=n_rows, p=[0.6,0.3,0.1])

    ad_spend = rng.gamma(shape=2.5, scale=500.0, size=n_rows)  # skewed
    customer_age = rng.normal(38, 10, size=n_rows).clip(18, 80)
    median_income = rng.normal(70000, 15000, size=n_rows).clip(20000, 200000)
    site_visits = rng.poisson(20, size=n_rows)
    prior_month_sales = rng.normal(2000, 800, size=n_rows).clip(0, None)

    # Seasonality by month (e.g., Nov/Dec stronger)
    month_factor = np.array([
        0.9, 0.92, 0.95, 1.0, 1.02, 1.05,
        1.08, 1.06, 1.03, 1.1, 1.3, 1.4
    ])[months-1]

    # Region / channel / segment effects
    region_effect = pd.Series({"north":1.05,"south":0.98,"east":1.0,"west":1.02})[regions].to_numpy()
    channel_effect = pd.Series({"online":1.06,"retail":0.95,"partner":1.0,"phone":0.9})[channels].to_numpy()
    seg_effect = pd.Series({"consumer":0.95,"smb":1.05,"enterprise":1.2})[segments].to_numpy()

    # True revenue signal
    base = 1500
    coef_ad = 2.8
    coef_age = 3.0
    coef_inc = 0.015
    coef_visits = 8.0
    coef_lag = 0.35

    noise = rng.normal(0, 250, size=n_rows)
    revenue = (
        base
        + coef_ad*np.sqrt(ad_spend)
        + coef_age*(customer_age - 35)
        + coef_inc*(median_income/1000)
        + coef_visits*np.log1p(site_visits)
        + coef_lag*np.sqrt(prior_month_sales+1)
    ) * month_factor * region_effect * channel_effect * seg_effect + noise

    df = pd.DataFrame({
        "order_id": rng.integers(10_000_000, 99_999_999, size=n_rows),
        "customer_id": rng.integers(1_000, 99_999, size=n_rows),
        "channel": channels,
        "region": regions,
        "customer_segment": segments,
        "ad_spend": ad_spend.round(2),
        "customer_age": customer_age.round(1),
        "median_income": median_income.round(0),
        "site_visits": site_visits,
        "prior_month_sales": prior_month_sales.round(2),
        "order_date": pd.to_datetime(dates),
        "revenue": revenue.round(2),
    })

    # Inject duplicates & missingness
    dup_sample = df.sample(frac=0.02, random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)
    for col in ["customer_age","median_income","ad_spend"]:
        idx = rng.choice(df.index, size=int(0.03*len(df)), replace=False)
        df.loc[idx, col] = np.nan

    return df

# -------------------------------
# Wrangling
# -------------------------------
def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df2 = df.drop_duplicates(subset=["order_id"], keep="last")
    print(f"[dedupe] {before:,} -> {len(df2):,}")
    return df2

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize categorical text
    for col in ["channel","region","customer_segment"]:
        df[col] = (df[col].astype(str).str.strip().str.lower())

    # Impute numerics
    for col in ["customer_age","median_income","ad_spend","site_visits","prior_month_sales"]:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    # Basic sanity filters
    df = df[df["ad_spend"] >= 0]
    df = df[df["customer_age"].between(15, 100)]
    df = df[df["median_income"] > 0]

    # Calendar features
    df["month"] = df["order_date"].dt.month.astype(int)
    df["year"] = df["order_date"].dt.year.astype(int)

    # Ensure target present
    df = df[df["revenue"].notna()]
    return df

# -------------------------------
# Modeling
# -------------------------------
def build_pipeline():
    num = Pipeline(steps=[("scaler", StandardScaler())])
    cat = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEG)])
    model = LinearRegression()
    pipe = Pipeline([("pre", pre), ("lr", model)])
    return pipe

def train_and_eval(df: pd.DataFrame):
    X = df[NUMERIC + CATEG]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    r2 = float(r2_score(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    mae = float(mean_absolute_error(y_test, preds))
    metrics = {"r2": r2, "rmse": rmse, "mae": mae}
    print("[metrics]", metrics)

    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipe, f)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipe, metrics

# -------------------------------
# Visualization
# -------------------------------
def plot_heatmap(df: pd.DataFrame) -> Path:
    corr = df[NUMERIC + [TARGET]].corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    out = Path("heatmap.png")
    plt.savefig(out, dpi=140); plt.close()
    return out

def plot_actual_vs_pred(df: pd.DataFrame, model) -> Path:
    X = df[NUMERIC + CATEG]
    y = df[TARGET]
    yhat = model.predict(X)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y, y=yhat, s=10, alpha=0.4)
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Actual Revenue"); plt.ylabel("Predicted Revenue")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    out = Path("actual_vs_pred.png")
    plt.savefig(out, dpi=140); plt.close()
    return out

def plot_residuals(df: pd.DataFrame, model) -> Path:
    X = df[NUMERIC + CATEG]
    y = df[TARGET]
    yhat = model.predict(X)
    resid = y - yhat
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=yhat, y=resid, s=8, alpha=0.35)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Fitted"); plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    out = Path("residuals_vs_fitted.png")
    plt.savefig(out, dpi=140); plt.close()
    return out

def plot_qq(df: pd.DataFrame, model) -> Path:
    X = df[NUMERIC + CATEG]
    y = df[TARGET]
    yhat = model.predict(X)
    resid = (y - yhat)
    plt.figure(figsize=(6,6))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Residuals Q-Q Plot")
    plt.tight_layout()
    out = Path("qq_plot.png")
    plt.savefig(out, dpi=140); plt.close()
    return out

# -------------------------------
# Main Orchestration
# -------------------------------
def main():
    # 1) Load or generate raw
    if RAW_CSV.exists():
        print(f"[data] Using existing CSV: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, parse_dates=["order_date"])
    else:
        print("[data] Generating synthetic dataset (50k+) ...")
        df = make_synthetic()
        df.to_csv(RAW_CSV, index=False)
        print(f"[data] Wrote {len(df):,} rows to {RAW_CSV}")

    # 2) Wrangle
    df = dedupe(df)
    df = basic_clean(df)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"[data] Clean saved -> {CLEAN_PARQUET} ({len(df):,} rows)")

    # 3) Model
    model, metrics = train_and_eval(df)

    # 4) Visualizations
    outs = [
        plot_heatmap(df),
        plot_actual_vs_pred(df, model),
        plot_residuals(df, model),
        plot_qq(df, model),
    ]
    print("[plots]", ", ".join(str(p) for p in outs))

if __name__ == "__main__":
    main()
