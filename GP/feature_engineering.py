import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def load_volatility_data(data_path="../data", frequency="5_min"):
    sheet_map = {
        "dates": "Dates",
        "companies": "Companies",
        "rv":  "RV_5" if frequency=="5_min" else "RV",
        "bpv": "BPV_5" if frequency=="5_min" else "BPV",
        "good":"Good_5" if frequency=="5_min" else "Good",
        "bad": "Bad_5" if frequency=="5_min" else "Bad",
        "rq":  "RQ_5" if frequency=="5_min" else "RQ"
    }

    file = os.path.join(data_path,"RV_March2024.xlsx")

    df_dates = pd.read_excel(file, sheet_name=sheet_map["dates"], header=None)
    df_companies = pd.read_excel(file, sheet_name=sheet_map["companies"], header=None)

    dates = pd.to_datetime(df_dates[0])
    companies = df_companies[0].tolist()

    frames = {}
    for k in ["rv","bpv","good","bad","rq"]:
        df = pd.read_excel(file, sheet_name=sheet_map[k], header=None)
        df.columns = companies
        df.index = dates
        df.replace(0,np.nan,inplace=True)
        frames[k] = df

    return frames, dates, companies


def load_vix(path="../data/VIX_History.csv"):
    vix = pd.read_csv(path)
    vix["DATE"] = pd.to_datetime(vix["DATE"])
    vix = vix.rename(columns={"DATE":"Date","CLOSE":"VIX"})
    vix = vix[["Date","VIX"]].set_index("Date").sort_index()
    return vix


def add_microstructure(df):
    df = df.copy()
    df["jump"] = df["rv"] - df["bpv"]
    df["jump_relative"] = df["jump"] / (df["rv"] + 1e-12)
    df["asymmetry"] = df["good"] - df["bad"]
    df["asymmetry_ratio"] = df["good"] / (df["bad"] + 1e-12)
    df["downside_ratio"] = df["bad"] / (df["rv"] + 1e-12)
    df["upside_ratio"] = df["good"] / (df["rv"] + 1e-12)
    return df


def add_har_features(df):
    df = df.copy()
    g = df.groupby("Stock")["rv"]
    df["rv_lag1"] = g.shift(1)
    df["rv_roll5"] = g.rolling(5).mean().shift(1).reset_index(0,drop=True)
    df["rv_roll22"] = g.rolling(22).mean().shift(1).reset_index(0,drop=True)
    return df


def add_vol_of_vol(df):
    df = df.copy()
    for col in ["rv","good","bad","rq"]:
        df[f"vov_{col}"] = (
            df.groupby("Stock")[col]
            .rolling(5).std().shift(1)
            .reset_index(0,drop=True)
        )
    return df


def add_shock_precursors(df):
    df = df.copy()
    df["pre_jump_max_3d"] = (
        df.groupby("Stock")["jump"]
        .rolling(3).max().shift(1)
        .reset_index(0,drop=True)
    )
    df["pre_bad_mean_3d"] = (
        df.groupby("Stock")["bad"]
        .rolling(3).mean().shift(1)
        .reset_index(0,drop=True)
    )
    df["pre_asym_diff"] = df.groupby("Stock")["asymmetry"].diff().shift(1)
    return df


def add_clustering(df, n_clusters=5):
    df = df.copy()
    cols = ["rv","bpv","good","bad","jump"]
    df["cluster"] = np.nan

    for date, sub in df.groupby(level="Date"):
        X = sub[cols].fillna(sub[cols].median())
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = km.fit_predict(X)
        except:
            labels = np.zeros(len(sub))
        df.loc[(date,), "cluster"] = labels

    df["cluster_lag1"] = df.groupby("Stock")["cluster"].shift(1)
    df["cluster_change"] = (df["cluster"] != df["cluster_lag1"]).astype(int)
    return df.drop(columns=["cluster"])


def build_features(df, vix_df):
    df = df.join(vix_df, on="Date")
    df["VIX_lag1"] = df.groupby("Stock")["VIX"].shift(1)

    df = add_microstructure(df)
    df = add_har_features(df)
    df = add_vol_of_vol(df)
    df = add_shock_precursors(df)
    df = add_clustering(df)

    return df


if __name__ == "__main__":
    print("Loading volatility data...")
    vol_5, dates, companies = load_volatility_data("..//data", frequency="5_min")
    vol_1, _, _ = load_volatility_data("..//data", frequency="1_min")

    df = pd.DataFrame(
        index=pd.MultiIndex.from_product([dates, companies], names=["Date","Stock"])
    )

    for k, v in vol_1.items():
        df[k] = v.stack().sort_index()

    rename_5 = {"rv":"rv_5","bpv":"bpv_5","good":"good_5","bad":"bad_5","rq":"rq_5"}
    for k, df5 in vol_5.items():
        df[rename_5[k]] = df5.stack().sort_index()

    df = df.dropna(subset=["rv"])
    print("Raw panel:", df.shape)

    print("Loading VIX...")
    vix_df = load_vix()

    print("Building features...")
    df_feat = build_features(df, vix_df)

    df_feat = df_feat.dropna()
    print("Final panel:", df_feat.shape)

    os.makedirs("data", exist_ok=True)
    df_feat.to_parquet("data/feature_panel.parquet")
    print("Saved to data/feature_panel.parquet")
