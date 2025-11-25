# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Electricity — Forecast & Anomalies Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path("/opt/electricity-pipeline/output_local")
ANOMALIES_FILE = ROOT / "anomalies_last_week.csv"
HISTORY_FILE = ROOT / "last_7_dates_history.csv"
PRED_FILE = ROOT / "predictions_next_7days.csv"

@st.cache_data(ttl=600)
def load_csv_safe(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["ts"])
        return df
    except Exception:
        # fallback: try reading without parse
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df

# Load datasets
anomalies = load_csv_safe(ANOMALIES_FILE)
history = load_csv_safe(HISTORY_FILE)
predictions = load_csv_safe(PRED_FILE)

# Basic checks & info
st.title("Electricity Monitoring — History · Forecast · Anomalies")
st.markdown("Interactive dashboard for node-wise electricity consumption, forecasts and anomaly analysis.")

col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.metric("Nodes (history)", int(history["node_id"].nunique() if not history.empty else 0))
with col2:
    st.metric("Records (history)", int(len(history)))
with col3:
    st.metric("Anomaly rows", int(len(anomalies)))

# Sidebar filters
st.sidebar.header("Filters & Controls")

# Node selector: union across all datasets
all_nodes = sorted(set(
    list(history["node_id"].unique()) if not history.empty else []
    + list(predictions["node_id"].unique()) if not predictions.empty else []
    + list(anomalies["node_id"].unique()) if not anomalies.empty else []
))
# if all_nodes is empty, fallback to a placeholder
if not all_nodes:
    all_nodes = ["(no nodes found)"]

node = st.sidebar.selectbox("Select node", options=["All"] + all_nodes, index=0)

# Date range filter based on history
if not history.empty:
    min_ts = history["ts"].min().date()
    max_ts = history["ts"].max().date()
    date_range = st.sidebar.date_input("History date range", value=(min_ts, max_ts), min_value=min_ts, max_value=max_ts)
else:
    date_range = None

# Chart choices
show_history = st.sidebar.checkbox("Show history (last 7 dates)", value=True)
show_predictions = st.sidebar.checkbox("Show predictions (7 days)", value=True)
show_anomalies = st.sidebar.checkbox("Highlight anomalies", value=True)

# Helper: filter function
def filter_df(df, node_sel, date_range_sel=None):
    if df.empty:
        return df
    out = df.copy()
    if node_sel and node_sel != "All":
        out = out[out["node_id"] == node_sel]
    if date_range_sel is not None and "ts" in out.columns:
        start, end = pd.to_datetime(date_range_sel[0]), pd.to_datetime(date_range_sel[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out["ts"] >= start) & (out["ts"] <= end)]
    return out

# Apply filters
hist_f = filter_df(history, node, date_range)
pred_f = filter_df(predictions, node, None)
anom_f = filter_df(anomalies, node, None)

# Panel: Main time series plot (history + predictions)
st.subheader("Time series: Consumption (15-min intervals)")
ts_col, info_col = st.columns([4,1])

with ts_col:
    fig = None
    if show_history and not hist_f.empty:
        fig = px.line(hist_f, x="ts", y="consumption_kw", labels={"ts":"Timestamp","consumption_kw":"kW"}, title=f"Consumption — {node}")
    if show_predictions and not pred_f.empty:
        # predictions may have duplicates; ensure ts is datetime
        pred_f = pred_f.copy()
        if "ts" in pred_f.columns:
            pred_f["ts"] = pd.to_datetime(pred_f["ts"])
        # overlay predictions
        if fig is None:
            fig = px.line(pred_f, x="ts", y="predicted_consumption_kw", labels={"predicted_consumption_kw":"Predicted kW"}, title=f"Consumption — {node}")
        else:
            fig.add_scatter(x=pred_f["ts"], y=pred_f["predicted_consumption_kw"], mode="lines", name="Predicted")
    if fig is None:
        st.info("No data to plot. Check CSV files or filters.")
    else:
        if show_anomalies and not anom_f.empty:
            # plot anomalies as markers
            anom_plot = anom_f.copy()
            anom_plot["ts"] = pd.to_datetime(anom_plot["ts"])
            fig.add_scatter(x=anom_plot["ts"], y=anom_plot["actual"], mode="markers", name="Anomaly",
                            marker=dict(size=6, symbol="x"), hovertext=anom_plot["anomaly_type"])
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True)

with info_col:
    st.write("Quick stats")
    if not hist_f.empty:
        st.metric("Avg kW (filtered)", f"{hist_f['consumption_kw'].mean():.2f}")
        st.metric("Max kW (filtered)", f"{hist_f['consumption_kw'].max():.2f}")
    else:
        st.write("No historical data for selected filters")

# Panel: Anomalies table and summary
st.subheader("Anomalies (last week)")
if anom_f.empty:
    st.info("No anomalies for selected node/filters.")
else:
    st.dataframe(anom_f.sort_values(["ts"]).reset_index(drop=True).astype({"ts":"datetime64[ns]"}))

    # anomalies counts by type
    anom_counts = anom_f["anomaly_type"].value_counts().reset_index()
    anom_counts.columns = ["anomaly_type","count"]
    fig2 = px.bar(anom_counts, x="anomaly_type", y="count", title="Anomaly counts by type")
    st.plotly_chart(fig2, use_container_width=True)

# Panel: Aggregations by day
st.subheader("Daily aggregation (sum of consumption per day)")
if hist_f.empty:
    st.info("No historical data for aggregation.")
else:
    agg = hist_f.set_index("ts").resample("D")["consumption_kw"].sum().reset_index()
    agg["date"] = agg["ts"].dt.date
    fig3 = px.bar(agg, x="date", y="consumption_kw", labels={"consumption_kw":"kW·day"}, title="Daily consumption (sum)")
    st.plotly_chart(fig3, use_container_width=True)

# Panel: Download filtered data
st.subheader("Download filtered data")
if not hist_f.empty:
    csv = hist_f.to_csv(index=False)
    st.download_button("Download history (CSV)", csv, file_name=f"history_filtered_{node}.csv", mime="text/csv")
if not pred_f.empty:
    csvp = pred_f.to_csv(index=False)
    st.download_button("Download predictions (CSV)", csvp, file_name=f"predictions_{node}.csv", mime="text/csv")
if not anom_f.empty:
    csva = anom_f.to_csv(index=False)
    st.download_button("Download anomalies (CSV)", csva, file_name=f"anomalies_{node}.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with Streamlit · Data loaded from /opt/electricity-pipeline/output_local")
