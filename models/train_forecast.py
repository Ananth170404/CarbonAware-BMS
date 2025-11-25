#!/usr/bin/env python3
"""
train_forecast.py

Usage:
  (venv) avinash@Avinash:/opt/electricity-pipeline/models$ python3 train_forecast.py

Reads parquet files under:
  /opt/electricity-pipeline/raw_local/kafka_parquet/

Trains a LightGBM regressor per node and forecasts next 7 days (15-min intervals).
Writes single CSV:
  /opt/electricity-pipeline/output_local/predictions_next_7days.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from tqdm import tqdm
import os

RAW_DIR = Path("/opt/electricity-pipeline/raw_local/kafka_parquet")
OUTPUT_CSV = Path("/opt/electricity-pipeline/output_local/predictions_next_7days.csv")
MODEL_SAVE_DIR = Path("/opt/electricity-pipeline/models/saved_models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
N_DAYS = 7
FREQ = "15T"  # 15 minutes
HORIZON = N_DAYS * 24 * 4  # 7 days * 24 hours * 4 intervals per hour = 672

# Feature generation parameters
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 96, 192]  # recent lags + 1 day lag (96 intervals)
ROLL_WINDOWS = [4, 96]  # 1 hour rolling (4*15min), 1 day rolling

RANDOM_STATE = 42

def read_all_parquets(root_dir: Path) -> pd.DataFrame:
    """Recursively read all parquet files under root_dir into a single DataFrame."""
    # Collect files
    files = list(root_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root_dir}")

    dfs = []
    for f in tqdm(files, desc="Reading parquet files"):
        try:
            df = pd.read_parquet(f)
            # If node_id or date folder not in file, try to infer from parent path
            # we expect columns: ts, consumption_kw, hour and maybe node_id
            if "node_id" not in df.columns:
                # try to infer from path parts containing 'node_id=' or node_id folder
                parts = f.parts
                node = None
                for p in parts:
                    if p.startswith("node_id="):
                        node = p.split("node_id=")[1]
                        break
                if node is None:
                    # fallback: maybe parent folder name is node id
                    node = f.parent.name
                df["node_id"] = node
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")

    # concat
    big = pd.concat(dfs, ignore_index=True, sort=False)
    return big

def prepare_node_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and sort a node-level DataFrame."""
    # Ensure required columns
    if "ts" not in df.columns:
        raise ValueError("Input data must have 'ts' column")

    # parse ts, ensure timezone-naive
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # keep required columns
    if "consumption_kw" not in df.columns:
        # attempt common name alternatives
        for alt in ["consumption", "consumption_kW", "consumption_kw/h"]:
            if alt in df.columns:
                df.rename(columns={alt: "consumption_kw"}, inplace=True)
                break
        else:
            raise ValueError("No consumption column found (expected 'consumption_kw')")

    # reindex to uniform 15-min frequency
    df = df.set_index("ts")
    # If there are small gaps, resample/interpolate. We will upsample to desired freq and forward-fill/backfill small gaps.
    # create continuous index
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ)
    df = df.reindex(idx)
    # keep node_id column propagation
    if "node_id" in df.columns:
        df["node_id"] = df["node_id"].ffill().bfill()
    # fill missing consumption: prefer interpolation, fallback to forward-fill
    df["consumption_kw"] = df["consumption_kw"].interpolate(limit=8).ffill().bfill()
    df = df.reset_index().rename(columns={"index": "ts"})
    df["hour"] = df["ts"].dt.strftime("%H")
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given node-level df with ts and consumption_kw, create lag & rolling features."""
    df = df.copy().set_index("ts")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["consumption_kw"].shift(lag)
    for win in ROLL_WINDOWS:
        df[f"roll_mean_{win}"] = df["consumption_kw"].shift(1).rolling(window=win, min_periods=1).mean()
        df[f"roll_std_{win}"] = df["consumption_kw"].shift(1).rolling(window=win, min_periods=1).std().fillna(0)
    # time features
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df = df.dropna(subset=["lag_1"])  # drop rows where lag_1 missing
    return df.reset_index()

def train_model(X_train, y_train, X_valid=None, y_valid=None):
    """Train a LightGBM regressor and return it."""
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse"
        )
    else:
        model.fit(X_train, y_train)
    return model

def recursive_forecast(model, last_known_df, fh, feature_cols):
    """
    Recursive multi-step forecasting:
    - last_known_df: dataframe containing last rows up to the latest known timestamp (columns: ts, consumption_kw)
    - fh: horizon (number of steps)
    - feature_cols: the columns the model expects (ordering must match training)
    Returns list of (ts, pred_value)
    """
    # ensure index is datetime and sorted
    last = last_known_df.copy()
    last["ts"] = pd.to_datetime(last["ts"])
    last = last.set_index("ts").sort_index()
    preds = []

    # work on a copy we can append to
    current = last.copy()

    # precompute fallback mean for speed
    fallback_mean = float(current["consumption_kw"].tail(96).mean() if len(current) > 0 else 0.0)

    for step in range(fh):
        next_ts = current.index[-1] + pd.Timedelta(FREQ)

        row = {}
        # build lag features by looking back into current['consumption_kw']
        for lag in LAGS:
            lag_ts = next_ts - pd.Timedelta(minutes=15 * lag)
            # use get with nearest exact match; if not present, fallback to last 96 mean
            if lag_ts in current.index:
                row[f"lag_{lag}"] = float(current.at[lag_ts, "consumption_kw"])
            else:
                row[f"lag_{lag}"] = fallback_mean

        # rolling features
        for win in ROLL_WINDOWS:
            window_start = next_ts - pd.Timedelta(minutes=15 * win)
            window_vals = current.loc[window_start:current.index[-1], "consumption_kw"]
            if len(window_vals) == 0:
                row[f"roll_mean_{win}"] = fallback_mean
                row[f"roll_std_{win}"] = float(current["consumption_kw"].tail(96).std() if len(current) > 0 else 0.0)
            else:
                row[f"roll_mean_{win}"] = float(window_vals.mean())
                row[f"roll_std_{win}"] = float(window_vals.std() if window_vals.std() == window_vals.std() else 0.0)

        # time features
        row["minute"] = int(next_ts.minute)
        row["hour"] = int(next_ts.hour)
        row["dayofweek"] = int(next_ts.dayofweek)
        row["day"] = int(next_ts.day)
        row["month"] = int(next_ts.month)
        row["is_weekend"] = int(next_ts.dayofweek in (5, 6))

        # build X_row in correct column order; if any feature missing, fill with fallback_mean or zero
        X_row = pd.DataFrame([row])
        # ensure all feature_cols exist in X_row (fill missing)
        for c in feature_cols:
            if c not in X_row.columns:
                # if it's a lag or roll that we didn't compute, fallback to mean/0
                if c.startswith("lag_") or c.startswith("roll_"):
                    X_row[c] = fallback_mean
                else:
                    X_row[c] = 0

        X_row = X_row[feature_cols]
        pred = model.predict(X_row)[0]
        pred = float(pred)

        # append predicted value to current using pd.concat (append removed in pandas)
        new_row_df = pd.DataFrame({"consumption_kw": [pred]}, index=[next_ts])
        current = pd.concat([current, new_row_df], verify_integrity=False)

        preds.append((next_ts, pred))

    return preds



def false_recursive_forecast(model, last_known_df, fh, feature_cols):
    """
    Recursive multi-step forecasting:
    - last_known_df: dataframe containing last rows up to the latest known timestamp (indexed by ts)
    - fh: horizon (number of steps)
    - feature_cols: the columns the model expects
    """
    # ensure index is datetime and sorted
    last = last_known_df.copy().set_index("ts").sort_index()
    preds = []
    current = last.copy()  # contains actual past values + features

    # We will append predicted rows one by one and compute new features from predicted consumption
    for step in range(fh):
        next_ts = current.index[-1] + pd.Timedelta(FREQ)
        # build one-row df for next_ts
        row = {}
        # build lag features by looking back into current['consumption_kw']
        for lag in LAGS:
            lag_ts = next_ts - pd.Timedelta(minutes=15 * lag)
            if lag_ts in current.index:
                row[f"lag_{lag}"] = current.at[lag_ts, "consumption_kw"]
            else:
                # if missing, use mean of last 96 values as fallback
                row[f"lag_{lag}"] = current["consumption_kw"].tail(96).mean()

        for win in ROLL_WINDOWS:
            window_start = next_ts - pd.Timedelta(minutes=15 * win)
            window_vals = current.loc[window_start:current.index[-1], "consumption_kw"]
            if len(window_vals) == 0:
                row[f"roll_mean_{win}"] = current["consumption_kw"].tail(96).mean()
                row[f"roll_std_{win}"] = current["consumption_kw"].tail(96).std()
            else:
                row[f"roll_mean_{win}"] = window_vals.mean()
                row[f"roll_std_{win}"] = window_vals.std() if window_vals.std() == window_vals.std() else 0.0

        # time features
        row["minute"] = next_ts.minute
        row["hour"] = next_ts.hour
        row["dayofweek"] = next_ts.dayofweek
        row["day"] = next_ts.day
        row["month"] = next_ts.month
        row["is_weekend"] = int(next_ts.dayofweek in (5,6))

        X_row = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X_row)[0]
        # append predicted value to current
        new_row = pd.Series({
            "consumption_kw": pred
        })
        # To preserve other fields used when computing lags, append full row with consumption_kw
        new_index = next_ts
        current = current.append(pd.DataFrame({"consumption_kw": [pred]}, index=[new_index]))
        preds.append((next_ts, float(pred)))
    return preds

def process_node(node_id: str, node_df: pd.DataFrame):
    """Train and forecast for a single node. Returns DataFrame with future predictions."""
    # Prepare node df
    df = prepare_node_df(node_df)
    df = df.set_index("ts")
    # If too few rows, skip
    if len(df) < 200:
        print(f"Skipping node {node_id}: insufficient data ({len(df)} rows).")
        return pd.DataFrame(columns=["node_id", "ts", "predicted_consumption_kw"])

    # create feature matrix
    feat_df = create_features(df.reset_index()[["ts", "consumption_kw"]])
    target_col = "consumption_kw"
    # define features
    feature_cols = [c for c in feat_df.columns if c not in ["ts", "consumption_kw"]]

    # split last chunk for validation (time-based)
    train_cut = int(len(feat_df) * 0.9)
    train_df = feat_df.iloc[:train_cut]
    valid_df = feat_df.iloc[train_cut:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    # train
    model = train_model(X_train, y_train, X_valid, y_valid)

    # save model
    model_file = MODEL_SAVE_DIR / f"lgb_node_{node_id}.pkl"
    try:
        joblib.dump(model, model_file)
    except Exception as e:
        print(f"Warning: could not save model for node {node_id}: {e}")

    # recursive forecast using the most recent raw series for lags
    # we need last_known_df containing raw consumption_kw indexed by ts
    last_known_df = df.reset_index()[["ts", "consumption_kw"]].set_index("ts")
    # ensure we have the latest continuous index (we filled earlier) so lags work
    last_known_df = last_known_df.copy()
    # call recursive_forecast — need feature_cols in same order used in training
    preds = recursive_forecast(model, last_known_df.reset_index()[["ts", "consumption_kw"]], HORIZON, feature_cols)

    # convert preds to dataframe
    pred_df = pd.DataFrame(preds, columns=["ts", "predicted_consumption_kw"])
    pred_df["node_id"] = node_id
    # order columns
    pred_df = pred_df[["node_id", "ts", "predicted_consumption_kw"]]
    return pred_df

def main():
    print("Start reading data...")
    big = read_all_parquets(RAW_DIR)

    # normalize column names (lowercase)
    big.columns = [c.lower() for c in big.columns]
    # expected columns: ts, consumption_kw, node_id
    if "ts" not in big.columns:
        # try alternatives
        if "timestamp" in big.columns:
            big = big.rename(columns={"timestamp": "ts"})
        else:
            raise ValueError("No timestamp column found in parquet files.")

    if "consumption_kw" not in big.columns:
        # try alternatives
        alt_found = False
        for alt in ["consumption", "consumption_kW", "value", "kw"]:
            if alt in big.columns:
                big = big.rename(columns={alt: "consumption_kw"})
                alt_found = True
                break
        if not alt_found:
            raise ValueError("No consumption column found in data. Expected 'consumption_kw' or similar.")

    if "node_id" not in big.columns:
        # try to infer node_id from directory structure: path was likely included earlier; otherwise fail
        raise ValueError("No node_id column found and couldn't infer.")

    # Ensure node_id is string
    big["node_id"] = big["node_id"].astype(str)

    # Process each node_id separately
    results = []
    nodes = big["node_id"].unique()
    print(f"Found {len(nodes)} nodes. Starting per-node processing...")

    for node in nodes:
        print(f"\nProcessing node: {node}")
        node_df = big[big["node_id"] == node][["ts", "consumption_kw"]].copy()
        try:
            pred_df = process_node(node, node_df)
            if not pred_df.empty:
                results.append(pred_df)
        except Exception as e:
            print(f"Error processing node {node}: {e}")

    if not results:
        print("No predictions generated.")
        return

    out = pd.concat(results, ignore_index=True)
    # order by node_id then ts
    out = out.sort_values(["node_id", "ts"]).reset_index(drop=True)
    # ensure output dir exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to: {OUTPUT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
