#!/usr/bin/env python3
"""
export_last_7_dates.py

Extract all rows whose ts falls in the last 7 distinct calendar dates
found across all parquet files under RAW_DIR, and save a single CSV.

Run from:
(venv) avinash@Avinash:/opt/electricity-pipeline/models$ python3 export_last_7_dates.py
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
import traceback

RAW_DIR = Path("/opt/electricity-pipeline/raw_local/kafka_parquet")
OUTPUT_CSV = Path("/opt/electricity-pipeline/output_local/last_7_dates_history.csv")

# columns we want to read when collecting rows
DESIRED_COLS = ["ts", "consumption_kw", "node_id"]

def list_parquet_files(root: Path):
    files = list(root.rglob("*.parquet"))
    valid = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                # skip zero-size
                continue
            valid.append(f)
        except Exception:
            # skip inaccessible files
            continue
    return valid

def infer_node_from_path(path: Path):
    for part in path.parts:
        if part.startswith("node_id="):
            return part.split("=", 1)[1]
    return path.parent.name

def collect_unique_dates(files):
    """First pass: collect distinct calendar dates from ts column across all files."""
    dates = set()
    for f in tqdm(files, desc="Collecting dates (reading ts)"):
        try:
            # try to read only ts column (faster than full read)
            # pandas supports columns argument for read_parquet
            df = pd.read_parquet(f, columns=["ts"])
            if df is None or df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            # drop NA ts
            df = df.dropna(subset=["ts"])
            # convert to date objects for uniqueness (calendar date)
            # using .dt.date for python date objects
            file_dates = set(df["ts"].dt.date.unique().tolist())
            dates.update(file_dates)
        except Exception:
            # If reading only ts fails (some files have different schema), attempt full read fallback
            try:
                df = pd.read_parquet(f)
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                    df = df.dropna(subset=["ts"])
                    dates.update(set(df["ts"].dt.date.unique().tolist()))
            except Exception:
                # unreadable file: skip
                continue
    return sorted(dates)  # ascending

def collect_rows_for_dates(files, target_dates):
    """Second pass: read each file and collect rows whose ts.date is in target_dates."""
    frames = []
    target_dates_set = set(target_dates)
    for f in tqdm(files, desc="Collecting rows for selected dates"):
        try:
            # Try to read only necessary columns if available
            # If any column not present, pandas will raise — we catch and fallback to reading full file.
            try:
                df = pd.read_parquet(f, columns=["ts", "consumption_kw", "node_id"])
            except Exception:
                df = pd.read_parquet(f)  # fallback read all
            if df is None or df.empty:
                continue

            # normalize colnames to lowercase for robustness
            df.columns = [c.lower() for c in df.columns]

            # ensure ts present
            if "ts" not in df.columns:
                # skip file if no timestamp
                continue
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.dropna(subset=["ts"])
            # filter by target dates
            df_dates = df["ts"].dt.date
            mask = df_dates.isin(target_dates_set)
            if not mask.any():
                continue
            df = df.loc[mask, :].copy()

            # ensure consumption_kw exists; try common alternatives
            if "consumption_kw" not in df.columns:
                for alt in ["consumption", "value", "kw", "consumption_kW"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "consumption_kw"})
                        break
            if "consumption_kw" not in df.columns:
                # if still missing, skip these rows
                continue

            # ensure node_id exists
            if "node_id" not in df.columns:
                df["node_id"] = infer_node_from_path(f)

            # keep only desired columns (ensure order)
            df = df[["node_id", "ts", "consumption_kw"]]
            frames.append(df)
        except Exception:
            # skip problematic file but don't crash
            continue
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    else:
        return pd.DataFrame(columns=["node_id", "ts", "consumption_kw"])

def main():
    files = list_parquet_files(RAW_DIR)
    if not files:
        print(f"No parquet files found under {RAW_DIR}")
        return

    # 1) collect unique dates
    all_dates_sorted = collect_unique_dates(files)
    if not all_dates_sorted:
        print("No timestamps found in any parquet files.")
        # write empty csv with headers
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["node_id", "ts", "consumption_kw"]).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty CSV to {OUTPUT_CSV}")
        return

    # choose last 7 distinct dates
    last_7_dates = all_dates_sorted[-7:] if len(all_dates_sorted) >= 7 else all_dates_sorted
    print("Selected last dates (ascending):")
    for d in last_7_dates:
        print("  ", d)

    # 2) collect rows for those dates
    result = collect_rows_for_dates(files, last_7_dates)
    if result.empty:
        print("No rows matched the selected dates.")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["node_id", "ts", "consumption_kw"]).to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote empty CSV to {OUTPUT_CSV}")
        return

    # normalize types & dedupe/sort
    result["node_id"] = result["node_id"].astype(str)
    result["ts"] = pd.to_datetime(result["ts"], errors="coerce")
    result = result.dropna(subset=["ts"])
    # remove duplicates keeping last (if multiple)
    result = result.sort_values(["node_id", "ts"]).drop_duplicates(subset=["node_id", "ts"], keep="last")
    result = result.sort_values(["node_id", "ts"]).reset_index(drop=True)

    # write csv
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(result)} rows for last {len(last_7_dates)} dates to: {OUTPUT_CSV}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
