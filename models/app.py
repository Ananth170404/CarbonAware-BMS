# app.py
"""
Streamlit Dashboard — Electricity Monitoring (enhanced)

Place this file in:
 /opt/electricity-pipeline/models/streamlit_app/app.py

Run:
 (venv) $ streamlit run app.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# -----------------------
# Config & file paths
# -----------------------
st.set_page_config(
    page_title="Electricity — Forecast & Anomalies",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path("/opt/electricity-pipeline/output_local")
ANOMALIES_FILE = ROOT / "anomalies_last_week.csv"
HISTORY_FILE = ROOT / "last_7_dates_history.csv"
PRED_FILE = ROOT / "predictions_next_7days.csv"

# -----------------------
# Styling (fonts + CSS)
# -----------------------
# Google font + small style adjustments
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .big-title {
        font-size:28px;
        font-weight:800;
        letter-spacing:0.2px;
    }
    .subtitle {
        color: #9aa5b1;
        margin-bottom: 12px;
    }
    .kpi {
        font-size:22px;
        font-weight:700;
    }
    .card {
        padding: 12px;
        border-radius: 8px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
    }
    /* Hide Streamlit menu and footer for cleaner look */
    #MainMenu {visibility: visible;}
    footer {visibility: visible;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Utilities
# -----------------------
@st.cache_data(ttl=300)
def read_csv_safe(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        df = pd.read_csv(path, engine="python", low_memory=False)
    # normalize columns to lower-case for robustness
    df.columns = [c.lower() for c in df.columns]
    # ensure ts column parse
    if "ts" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        except Exception:
            df["ts"] = pd.to_datetime(df["ts"].astype(str), errors="coerce")
    return df

def ensure_consumption_col(df: pd.DataFrame):
    """Ensure df has 'consumption_kw' column; try common alternatives"""
    if "consumption_kw" in df.columns:
        return df
    for alt in ["consumption", "value", "kw", "consumption_kw", "consumption_kW"]:
        if alt in df.columns:
            df = df.rename(columns={alt: "consumption_kw"})
            return df
    # If none found but there's a numeric column other than ts, try picking it (last resort)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "ts"]
    if numeric_cols:
        df = df.rename(columns={numeric_cols[0]: "consumption_kw"})
    return df

def safe_unique_sorted(series):
    try:
        return sorted(pd.Series(series).dropna().unique().tolist())
    except Exception:
        return []

# -----------------------
# Load data
# -----------------------
history = read_csv_safe(HISTORY_FILE)
predictions = read_csv_safe(PRED_FILE)
anomalies = read_csv_safe(ANOMALIES_FILE)

history = ensure_consumption_col(history)
predictions = ensure_consumption_col(predictions)
anomalies = ensure_consumption_col(anomalies)

# unify column names for predictions (often predicted_consumption_kw)
if "predicted_consumption_kw" in predictions.columns and "predicted_consumption_kw" not in predictions.columns:
    pass
# try to rename predicted column if present as different name
for candidate in ["predicted_consumption_kw", "predicted_kw", "prediction", "yhat", "yhat_pred"]:
    if candidate in predictions.columns and "predicted_consumption_kw" not in predictions.columns:
        predictions = predictions.rename(columns={candidate: "predicted_consumption_kw"})

# fallback: if predictions have same 'consumption_kw' as actuals use that column name when plotting
if "predicted_consumption_kw" not in predictions.columns and "consumption_kw" in predictions.columns:
    predictions = predictions.rename(columns={"consumption_kw": "predicted_consumption_kw"})

# Ensure ts columns exist and are datetimes
for df in [history, predictions, anomalies]:
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        # create empty timestamp to avoid crashing
        df["ts"] = pd.NaT

# -----------------------
# Sidebar (global controls)
# -----------------------
st.sidebar.header("Filters & Controls")
# Provide global node list as union
nodes = safe_unique_sorted(
    list(history.get("node_id", [])) + list(predictions.get("node_id", [])) + list(anomalies.get("node_id", []))
)
nodes = ["All"] + nodes if nodes else ["All"]

selected_node = st.sidebar.selectbox("Select node", options=nodes, index=0)
date_min = history["ts"].min() if not history.empty else None
date_max = history["ts"].max() if not history.empty else None
if date_min is not None and date_max is not None:
    date_range = st.sidebar.date_input("History date range", value=(date_min.date(), date_max.date()), min_value=date_min.date(), max_value=date_max.date())
else:
    date_range = None

# smoothing and rolling
smoothing = st.sidebar.checkbox("Show rolling mean (1h)", value=False)
rolling_window = st.sidebar.selectbox("Rolling window (minutes)", options=[15, 30, 60, 120], index=2)
rolling_periods = int(rolling_window / 15) if rolling_window >= 15 else 1

# compare nodes controls (for Compare tab)
compare_nodes = st.sidebar.multiselect("Compare nodes (Compare tab)", options=nodes[1:][:20], default=nodes[1:3] if len(nodes) > 2 else [])

# theme options
st.sidebar.markdown("---")
show_conf_intervals = st.sidebar.checkbox("Show forecast confidence (if available)", value=True)

# -----------------------
# Helper filters
# -----------------------
def filter_by_node(df: pd.DataFrame, node_sel):
    if df.empty or node_sel == "All":
        return df
    if "node_id" not in df.columns:
        return df
    return df[df["node_id"].astype(str) == str(node_sel)]

def filter_by_date_range(df: pd.DataFrame, date_range_sel):
    if df.empty or date_range_sel is None or "ts" not in df.columns:
        return df
    start = pd.to_datetime(date_range_sel[0])
    end = pd.to_datetime(date_range_sel[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["ts"] >= start) & (df["ts"] <= end)]

# Apply filters
hist_f = filter_by_node(history, selected_node)
hist_f = filter_by_date_range(hist_f, date_range)
pred_f = filter_by_node(predictions, selected_node)
anom_f = filter_by_node(anomalies, selected_node)

# -----------------------
# Layout: Header + KPIs
# -----------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<div class="big-title">Electricity Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">History · Forecast · Anomalies — interactive analytics & exports</div>', unsafe_allow_html=True)
with header_col2:
    # KPI cards
    k1, k2, k3 = st.columns(3)
    k1.metric("Nodes (history)", int(history["node_id"].nunique() if "node_id" in history.columns else 0))
    k2.metric("Records (history)", int(len(history)))
    k3.metric("Anomaly rows", int(len(anomalies)))

st.markdown("---")

# -----------------------
# Tabs / Pages
# -----------------------
tabs = st.tabs(["Overview", "Node detail", "Compare nodes", "Anomalies & Export"])
tab_overview, tab_node, tab_compare, tab_anom = tabs

# -----------------------
# Overview tab
# -----------------------
with tab_overview:
    st.subheader("Overview — Trends & Daily Patterns")
    left, right = st.columns([3, 1])
    with left:
        # Combined time series (all nodes aggregated)
        if history.empty:
            st.info("No historical data available to render overview.")
        else:
            agg = history.copy()
            if "consumption_kw" not in agg.columns:
                st.error("Historical file missing consumption column.")
            else:
                agg = agg.set_index("ts").resample("1H")["consumption_kw"].sum().reset_index()
                fig = px.line(agg, x="ts", y="consumption_kw", title="Aggregated hourly consumption (all nodes)")
                st.plotly_chart(fig, use_container_width=True)
    with right:
        st.write("Overview metrics")
        if not history.empty:
            avg = history["consumption_kw"].mean()
            st.metric("Avg consumption (kW)", f"{avg:.3f}")
            st.markdown("**Daily total (last 7 days)**")
            day_sum = history.set_index("ts").resample("D")["consumption_kw"].sum().reset_index()
            st.dataframe(day_sum.rename(columns={"ts":"date"}).tail(7).assign(date=lambda d: d["ts"].dt.date).drop(columns=["ts"]).reset_index(drop=True), hide_index=True)

    st.markdown("### Hour-of-day heatmap (pattern across days)")
    if history.empty:
        st.info("No history for heatmap.")
    else:
        heat = history.copy()
        heat["date"] = heat["ts"].dt.date
        heat["hour"] = heat["ts"].dt.hour
        # pivot table: hour x date sum
        pv = heat.groupby(["date","hour"])["consumption_kw"].sum().reset_index()
        pivot = pv.pivot(index="hour", columns="date", values="consumption_kw").fillna(0)
        # convert to long form for plotly
        heat_long = pivot.reset_index().melt(id_vars="hour", var_name="date", value_name="kwh")
        fig_heat = px.imshow(pivot.values,
                             labels=dict(x="Date index", y="Hour of day", color="kW"),
                             x=[str(d) for d in pivot.columns],
                             y=pivot.index,
                             aspect="auto",
                             title="Heatmap: hour vs date (sum kW)")
        st.plotly_chart(fig_heat, use_container_width=True)

# -----------------------
# Node detail tab
# -----------------------
with tab_node:
    st.subheader("Node detail")
    st.markdown(f"Selected node: **{selected_node}**")
    col_time, col_info = st.columns([4, 1])
    # Time series for selected node
    with col_time:
        if hist_f.empty and pred_f.empty:
            st.info("No data for this node with current filters.")
        else:
            fig = go.Figure()
            if not hist_f.empty:
                tmp = hist_f.copy()
                tmp = tmp.sort_values("ts")
                fig.add_trace(go.Scatter(x=tmp["ts"], y=tmp["consumption_kw"], name="Actual", mode="lines", line=dict(width=1)))
                if smoothing:
                    tmp["rolling"] = tmp["consumption_kw"].rolling(window=max(1, rolling_periods)).mean()
                    fig.add_trace(go.Scatter(x=tmp["ts"], y=tmp["rolling"], name=f"Rolling ({rolling_window}m)"))
            if not pred_f.empty:
                pf = pred_f.copy()
                if "predicted_consumption_kw" in pf.columns:
                    fig.add_trace(go.Scatter(x=pf["ts"], y=pf["predicted_consumption_kw"], name="Predicted", mode="lines", line=dict(dash="dash")))
                elif "consumption_kw" in pf.columns:
                    fig.add_trace(go.Scatter(x=pf["ts"], y=pf["consumption_kw"], name="Predicted (consumption_kw)", mode="lines", line=dict(dash="dash")))
                # confidence
                if show_conf_intervals and "yhat_lower" in pf.columns and "yhat_upper" in pf.columns:
                    fig.add_traces([
                        go.Scatter(x=pf["ts"], y=pf["yhat_upper"], name="yhat_upper", line=dict(width=0), showlegend=False),
                        go.Scatter(x=pf["ts"], y=pf["yhat_lower"], fill='tonexty', name="Confidence interval", fillcolor='rgba(0,100,80,0.1)', line=dict(width=0), showlegend=True)
                    ])
            # anomalies overlay
            if not anom_f.empty:
                af = anom_f.copy()
                fig.add_trace(go.Scatter(x=af["ts"], y=af["actual"], name="Anomalies", mode="markers", marker=dict(size=6, symbol="x")))
            fig.update_layout(title=f"Consumption — {selected_node}", xaxis_title="Timestamp", yaxis_title="kW", legend_title="Series")
            st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.write("Quick stats")
        if not hist_f.empty:
            st.metric("Avg kW", f"{hist_f['consumption_kw'].mean():.2f}")
            st.metric("Max kW", f"{hist_f['consumption_kw'].max():.2f}")
            st.metric("Records", len(hist_f))
        else:
            st.write("No history data for this node.")

    # Boxplot by hour
    st.markdown("#### Distribution by hour (boxplot)")
    if hist_f.empty:
        st.info("No data for boxplot.")
    else:
        bp = hist_f.copy()
        bp["hour"] = bp["ts"].dt.hour
        fig_box = px.box(bp, x="hour", y="consumption_kw", points="outliers", title="Hourly distribution (boxplot)")
        st.plotly_chart(fig_box, use_container_width=True)

# -----------------------
# Compare nodes tab
# -----------------------
with tab_compare:
    st.subheader("Compare nodes (multi-node overlay)")
    if not compare_nodes:
        st.info("Select two or more nodes in the sidebar 'Compare nodes' to use this view (limit 20).")
    else:
        # read and combine for selected nodes
        comp_df = history[history["node_id"].isin(compare_nodes)].copy() if not history.empty else pd.DataFrame()
        if comp_df.empty:
            st.info("No historical data for selected nodes.")
        else:
            # resample to hourly sums for clarity
            comp_df = comp_df.set_index("ts").groupby("node_id")["consumption_kw"].resample("1H").sum().reset_index()
            figc = px.line(comp_df, x="ts", y="consumption_kw", color="node_id", title="Hourly consumption: compare nodes")
            st.plotly_chart(figc, use_container_width=True)

# -----------------------
# Anomalies & Export tab
# -----------------------
with tab_anom:
    st.subheader("Anomalies & Exports")
    st.markdown("This table shows the detected anomalies (last week). You can filter and download.")

    if anomalies.empty:
        st.info("No anomalies file detected.")
    else:
        # filtering
        anom_tbl = anomalies.copy()
        if selected_node != "All":
            anom_tbl = anom_tbl[anom_tbl["node_id"].astype(str) == str(selected_node)]
        anom_tbl = anom_tbl.sort_values("ts").reset_index(drop=True)
        st.dataframe(anom_tbl, use_container_width=True, height=300)

        # anomaly counts
        if "anomaly_type" in anom_tbl.columns:
            counts = anom_tbl["anomaly_type"].value_counts().rename_axis("type").reset_index(name="count")
            figbar = px.bar(counts, x="type", y="count", title="Anomaly counts")
            st.plotly_chart(figbar, use_container_width=True)

        # download buttons
        st.markdown("### Download data")
        csv_hist = hist_f.to_csv(index=False) if not hist_f.empty else ""
        csv_pred = pred_f.to_csv(index=False) if not pred_f.empty else ""
        csv_anom = anom_tbl.to_csv(index=False) if not anom_tbl.empty else ""
        if csv_hist:
            st.download_button("Download history (filtered)", csv_hist, file_name=f"history_filtered_{selected_node}.csv", mime="text/csv")
        if csv_pred:
            st.download_button("Download predictions (filtered)", csv_pred, file_name=f"predictions_{selected_node}.csv", mime="text/csv")
        if csv_anom:
            st.download_button("Download anomalies (filtered)", csv_anom, file_name=f"anomalies_{selected_node}.csv", mime="text/csv")

st.markdown("---")
st.caption("Dashboard generated by Streamlit — data loaded from /opt/electricity-pipeline/output_local. Customize further by editing app.py")

