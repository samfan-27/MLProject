import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

TOP10 = [
    "bpv_5",
    "rv_roll5",
    "VIX",
    "rv_lag1",
    "vov_bad",
    "VIX_lag1",
    "bpv",
    "bad_5",
    "upside_ratio",
    "jump_relative",
]

def add_Y_reg(df):
    df = df.copy()
    df["Y_reg"] = df.groupby("Stock")["rv"].shift(-1)
    df = df.dropna(subset=["Y_reg"])
    return df


def run_phase2(df):
    print("=== PHASE 2: HAR baseline model ===")
    df = add_Y_reg(df)
    print(f"Added Y_reg. Shape: {df.shape}")

    har_feats = ["rv_lag1", "rv_roll5", "rv_roll22"]
    df = df.dropna(subset=har_feats)

    X = df[har_feats]
    y = df["Y_reg"]

    har_model = LinearRegression()
    har_model.fit(X, y)

    df["har_pred"] = har_model.predict(X)
    df["har_error"] = df["Y_reg"] - df["har_pred"]

    mse = np.mean(df["har_error"] ** 2)
    print(f"HAR MSE: {mse:.6f}")
    print("HAR model complete. Residuals computed.\n")

    print("HAR error summary:")
    print(df[["Y_reg", "har_error"]].describe(), "\n")

    return df


def run_phase3(df):
    print("=== PHASE 3: Isolation Forest → anomaly labels ===")
    df = df.copy()

    df["error_scaled"] = df.groupby("Stock")["har_error"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    print("Residuals standardized per stock.")
    print(df[["har_error", "error_scaled"]].head(), "\n")

    iso = IsolationForest(contamination=0.05, random_state=42)

    df["is_abnormal"] = iso.fit_predict(df[["error_scaled"]])
    df["is_abnormal"] = (df["is_abnormal"] == -1).astype(int)

    total_abn = df["is_abnormal"].sum()
    pct = df["is_abnormal"].mean() * 100
    print(f"Total anomalies detected: {total_abn}")
    print(f"Percentage of anomalies: {pct:.2f}%")
    print(df[["har_error", "error_scaled", "is_abnormal"]].head(), "\n")

    return df


def run_phase4(df, features):
    print("=== PHASE 4: Build Y_class ===")

    df = df.copy()
    df["Y_class"] = df.groupby("Stock")["is_abnormal"].shift(-1)
    df = df.dropna(subset=["Y_class"])
    df["Y_class"] = df["Y_class"].astype(int)

    print(f"Original data size: {df.shape}")
    print(f"Final data size for Phase 4: {df.shape}\n")

    keep = features + ["Y_reg", "Y_class"]
    df_small = df[keep]

    print("Final 5 rows:")
    print(df_small.head(), "\n")

    os.makedirs("data", exist_ok=True)
    df_small.to_parquet("data/p4_clean.parquet")

    return df_small


if __name__ == "__main__":

    print("Loading feature panel...")
    df = pd.read_parquet("data/feature_panel.parquet")
    print(f"Initial shape: {df.shape}\n")

    df = run_phase2(df)
    df = run_phase3(df)
    df = run_phase4(df, TOP10)
