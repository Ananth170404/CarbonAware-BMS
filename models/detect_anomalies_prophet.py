#!/usr/bin/env python3
"""
detect_anomalies_prophet.py

Run from: (venv) avinash@Avinash:/opt/electricity-pipeline/models$
Produces:
  /opt/electricity-pipeline/output_local/anomalies_last_week.csv

Anomaly CSV columns:
  node_id, ts (ISO), date, time, anomaly_type (above|below),
  actual, yhat, yhat_lower, yhat_upper
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback
import sys

RAW_DIR = Path("/opt/electricity-pipeline/raw_local/kafka_parquet")
OUTPUT_CSV = Path("/opt/electricity-pipeline/output_local/anomalies_last_week.csv")

# Prophet import
try:
    from prophet import Prophet
except Exception as e:
    raise ImportError("prophet not installed or import failed. Install with: pip install prophet") from e

def list_parquet_files(root: Path):
    files = list(root.rglob("*.parquet"))
    # filter out zero-size files
    valid = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                # skip zero-size
                continue
            valid.append(f)
        except Exception:
            continue
    return valid

def infer_node_from_path(path: Path):
    for part in path.parts:
        if part.startswith("node_id="):
            return part.split("=", 1)[1]
    return path.parent.name

def collect_unique_dates(files):
    """First pass: collect unique calendar dates across all files by reading only ts column where possible."""
    dates = set()
    for f in tqdm(files, desc="Scanning dates (reading ts)"):
        try:
            # read only ts (faster)
            df = pd.read_parquet(f, columns=["ts"])
            if df is None or df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.dropna(subset=["ts"])
            dates.update(set(df["ts"].dt.date.unique().tolist()))
        except Exception:
            # fallback to full-read if needed
            try:
                df = pd.read_parquet(f)
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                    df = df.dropna(subset=["ts"])
                    dates.update(set(df["ts"].dt.date.unique().tolist()))
            except Exception:
                continue
    return sorted(dates)

def read_rows_for_dates(files, target_dates):
    """Second pass: read rows whose ts.date is in target_dates; return dataframe with node_id, ts, consumption_kw."""
    frames = []
    target_dates_set = set(target_dates)
    for f in tqdm(files, desc="Reading candidate rows"):
        try:
            # attempt to read only columns we need, fallback to full read
            try:
                df = pd.read_parquet(f, columns=["ts", "consumption_kw", "node_id"])
            except Exception:
                df = pd.read_parquet(f)

            if df is None or df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            if "ts" not in df.columns:
                continue
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.dropna(subset=["ts"])
            mask = df["ts"].dt.date.isin(target_dates_set)
            if not mask.any():
                continue
            sub = df.loc[mask].copy()
            # ensure consumption column
            if "consumption_kw" not in sub.columns:
                for alt in ["consumption", "value", "kw", "consumption_kW"]:
                    if alt in sub.columns:
                        sub = sub.rename(columns={alt: "consumption_kw"})
                        break
            if "consumption_kw" not in sub.columns:
                # skip these rows if no consumption column
                continue
            if "node_id" not in sub.columns:
                sub["node_id"] = infer_node_from_path(f)
            sub = sub[["node_id", "ts", "consumption_kw"]]
            frames.append(sub)
        except Exception:
            # skip problematic file
            continue

    if frames:
        big = pd.concat(frames, ignore_index=True, sort=False)
        # normalize types
        big["node_id"] = big["node_id"].astype(str)
        big["ts"] = pd.to_datetime(big["ts"])
        big = big.dropna(subset=["consumption_kw"])
        return big
    else:
        return pd.DataFrame(columns=["node_id", "ts", "consumption_kw"])

def run_for_node(node_id, node_df, second_week_dates, last_week_dates):
    """
    node_df: all rows for this node limited to the 14 target dates (ts & consumption_kw)
    second_week_dates: list of date objects (earlier 7)
    last_week_dates: list of date objects (later 7)
    Returns dataframe of anomalies (node_id, ts, date, time, anomaly_type, actual, yhat, yhat_lower, yhat_upper)
    """
    # prepare training (second last week) and testing (last week)
    node_df["date"] = node_df["ts"].dt.date
    train_df = node_df[node_df["date"].isin(second_week_dates)].copy()
    test_df = node_df[node_df["date"].isin(last_week_dates)].copy()

    # require minimum amount of training data to attempt fit - here at least 50 points (approx 12.5 hours)
    if len(train_df) < 50 or len(test_df) == 0:
        return pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ])

    # Prophet expects columns ds and y
    prophet_train = train_df[["ts", "consumption_kw"]].rename(columns={"ts": "ds", "consumption_kw": "y"})
    # Prophet works better when regular frequency; but we will not force resampling here - using raw 15-min points.
    # configure prophet
    model = Prophet(interval_width=0.95, daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    # add extra seasonality if desired; but keep model light
    try:
        model.fit(prophet_train)
    except Exception as e:
        # if fit fails, skip node
        return pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ])

    # prepare future dataframe: timestamps present in test_df (unique, sorted)
    future_ts = pd.DataFrame({"ds": sorted(test_df["ts"].unique())})
    # forecast
    forecast = model.predict(future_ts)  # has ds, yhat, yhat_lower, yhat_upper
    # merge test actuals with forecast
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    merged = pd.merge(test_df, forecast, left_on="ts", right_on="ds", how="left")
    merged = merged.dropna(subset=["yhat"])  # drop rows with no prediction
    if merged.empty:
        return pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ])

    # detect anomalies
    anomalies = []
    for _, row in merged.iterrows():
        actual = float(row["consumption_kw"])
        yhat = float(row["yhat"])
        ylow = float(row["yhat_lower"])
        yhigh = float(row["yhat_upper"])
        if pd.isna(ylow) or pd.isna(yhigh):
            continue
        if actual > yhigh:
            anomalies.append({
                "node_id": str(node_id),
                "ts": row["ts"],
                "date": row["ts"].date().isoformat(),
                "time": row["ts"].time().isoformat(),
                "anomaly_type": "above",
                "actual": actual,
                "yhat": yhat,
                "yhat_lower": ylow,
                "yhat_upper": yhigh
            })
        elif actual < ylow:
            anomalies.append({
                "node_id": str(node_id),
                "ts": row["ts"],
                "date": row["ts"].date().isoformat(),
                "time": row["ts"].time().isoformat(),
                "anomaly_type": "below",
                "actual": actual,
                "yhat": yhat,
                "yhat_lower": ylow,
                "yhat_upper": yhigh
            })
    if anomalies:
        return pd.DataFrame(anomalies)
    else:
        return pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ])

def main():
    files = list_parquet_files(RAW_DIR)
    if not files:
        print("No parquet files found. Exiting.")
        return

    # 1) collect unique calendar dates across data
    all_dates_sorted = collect_unique_dates(files)
    if not all_dates_sorted:
        print("No timestamps found across parquet files. Exiting.")
        return
    # take last 14 distinct dates
    last_14 = all_dates_sorted[-14:] if len(all_dates_sorted) >= 14 else all_dates_sorted
    if len(last_14) < 2:
        print("Not enough distinct dates to run 2-week analysis.")
        return
    # split into second_last_week (earlier 7) and last_week (later 7)
    if len(last_14) >= 14:
        second_last_week = last_14[:7]
        last_week = last_14[7:]
    else:
        # if less than 14 but >=2, split half
        mid = len(last_14) // 2
        second_last_week = last_14[:mid]
        last_week = last_14[mid:]

    print("Second-last-week dates (training):", second_last_week)
    print("Last-week dates (testing):", last_week)

    # 2) read rows for these 14 dates
    combined = read_rows_for_dates(files, second_last_week + last_week)
    if combined.empty:
        print("No data rows found for the selected 14 dates.")
        # write empty csv
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ]).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty CSV to {OUTPUT_CSV}")
        return

    # 3) run node-wise
    anomalies_list = []
    nodes = combined["node_id"].unique()
    print(f"Processing {len(nodes)} nodes...")
    for node in tqdm(nodes, desc="Nodes"):
        node_df = combined[combined["node_id"] == node][["ts", "consumption_kw"]].copy()
        try:
            result = run_for_node(node, node_df, second_last_week, last_week)
            if not result.empty:
                anomalies_list.append(result)
        except Exception:
            # continue on error for other nodes
            traceback.print_exc()
            continue

    if anomalies_list:
        out = pd.concat(anomalies_list, ignore_index=True, sort=False)
        out = out.sort_values(["node_id", "ts"]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "node_id","ts","date","time","anomaly_type","actual","yhat","yhat_lower","yhat_upper"
        ])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} anomaly rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

