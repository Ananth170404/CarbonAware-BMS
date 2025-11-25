# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# --- Custom CSS for professional style ---
st.markdown("""
    <style>
        .main {
            background-color: #f4f6fa;
        }
        .stApp {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .css-18e3th9 {
            padding-top: 0rem;
        }
        .metric-label > div {
            font-size:1.1em !important;
            color:#324259 !important;
            font-weight:bold !important;
        }
        .stSidebar {
            background: linear-gradient(180deg, #324259 80%, #f4f6fa 100%);
            color: #fff !important;
        }
        .block-container {
            padding-top: 2rem;
        }
        .reportview-container .main .block-container {
            max-width: 1200px;
        }
        .css-1d391kg { /* subtitle color */
            color: #324259 !important;
        }
        .css-1v3fvcr { /* caption color */
            color: #5c667a !important;
        }
        h1,h2,h3 { color:#1E2A38; }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1em !important;
            padding: 12px !important;
        }
        .stDownloadButton {
            background-color: #2c3762;
            color: #fff;
        }
        .stDownloadButton:hover {
            background-color: #4762ac;
        }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Electricity — Forecast & Anomalies Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
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
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        return df

anomalies = load_csv_safe(ANOMALIES_FILE)
history = load_csv_safe(HISTORY_FILE)
predictions = load_csv_safe(PRED_FILE)

# Sidebar — Company branding & Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Electricity_logo.png", width=160)
st.sidebar.title("Electricity Analytics ✔️")
st.sidebar.header("🔎  Filters & Controls")

# Node selector: union across all datasets
all_nodes = sorted(set(
    list(history["node_id"].unique()) if not history.empty else []
    + list(predictions["node_id"].unique()) if not predictions.empty else []
    + list(anomalies["node_id"].unique()) if not anomalies.empty else []
))
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

theme = st.sidebar.radio("Theme", options=["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp {
        background-color:#232629;
        color:#fff;
    }
    h1, h2, h3 { color:#daf4fa; }
    .css-1v3fvcr { color:#87a2b4 !important; }
    .block-container { background: #232629 !important; }
    </style>
    """, unsafe_allow_html=True)

show_history = st.sidebar.checkbox("Show history (last 7 dates)", value=True)
show_predictions = st.sidebar.checkbox("Show predictions (7 days)", value=True)
show_anomalies = st.sidebar.checkbox("Highlight anomalies", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Data loaded from `/opt/electricity-pipeline/output_local`.")

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

hist_f = filter_df(history, node, date_range)
pred_f = filter_df(predictions, node, None)
anom_f = filter_df(anomalies, node, None)

# --- Multitabs professional layout ---

st.title("Electricity Monitoring Dashboard ⚡️")
st.caption("Interactive MNC-grade dashboard for node-wise electricity consumption, forecasting, and anomaly analysis.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "🧮 Analytics",
    "⚠️ Anomalies"
])

# --- Overview Tab ---
with tab1:
    st.subheader("Dashboard — Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes (history)", int(history["node_id"].nunique() if not history.empty else 0))
    col2.metric("Records (history)", int(len(history)))
    col3.metric("Anomaly Rows", int(len(anomalies)))
    col4.metric("Predictions", int(len(predictions)))
    st.markdown("---")

    if not hist_f.empty:
        kpi1 = hist_f["consumption_kw"].mean()
        kpi2 = hist_f["consumption_kw"].max()
        kpi3 = hist_f["consumption_kw"].min()
        st.write(
            f"**Avg kW:** {kpi1:.2f} &nbsp;|&nbsp; **Max kW:** {kpi2:.2f} &nbsp;|&nbsp; **Min kW:** {kpi3:.2f}"
        )

    st.markdown("---")
    # Pie chart: Anomalies by type
    if not anom_f.empty:
        anom_counts = anom_f["anomaly_type"].value_counts().reset_index()
        anom_counts.columns = ["anomaly_type", "count"]
        fig_pie = px.pie(anom_counts, names="anomaly_type", values="count", title="Anomaly Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Table: Recent anomalies
    if not anom_f.empty:
        st.write("**Recent Anomaly Events:**")
        st.dataframe(anom_f.sort_values("ts", ascending=False).head(10).reset_index(drop=True))

# --- Time Series Tab ---
with tab2:
    st.subheader("Time Series — Consumption & Prediction")
    fig = None
    if show_history and not hist_f.empty:
        fig = px.line(hist_f, x="ts", y="consumption_kw", color="node_id", labels={"ts":"Timestamp","consumption_kw":"kW"}, title=f"Consumption — {node}")
    if show_predictions and not pred_f.empty:
        pred_f = pred_f.copy()
        if "ts" in pred_f.columns:
            pred_f["ts"] = pd.to_datetime(pred_f["ts"])
        if fig is None:
            fig = px.line(pred_f, x="ts", y="predicted_consumption_kw", color="node_id", labels={"predicted_consumption_kw":"Predicted kW"}, title=f"Predictions — {node}")
        else:
            fig.add_scatter(x=pred_f["ts"], y=pred_f["predicted_consumption_kw"], mode="lines", name="Predicted")
    if fig is not None:
        if show_anomalies and not anom_f.empty:
            anom_plot = anom_f.copy()
            anom_plot["ts"] = pd.to_datetime(anom_plot["ts"])
            fig.add_scatter(x=anom_plot["ts"], y=anom_plot["actual"], mode="markers", name="Anomaly",
                            marker=dict(size=7, symbol="cross"), hovertext=anom_plot["anomaly_type"])
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available.")

    st.markdown("---")
    st.write("**Download filtered data:**")
    if not hist_f.empty:
        csv = hist_f.to_csv(index=False)
        st.download_button("Download history (CSV)", csv, file_name=f"history_filtered_{node}.csv", mime="text/csv")
    if not pred_f.empty:
        csvp = pred_f.to_csv(index=False)
        st.download_button("Download predictions (CSV)", csvp, file_name=f"predictions_{node}.csv", mime="text/csv")
    if not anom_f.empty:
        csva = anom_f.to_csv(index=False)
        st.download_button("Download anomalies (CSV)", csva, file_name=f"anomalies_{node}.csv", mime="text/csv")

# --- Analytics Tab ---
with tab3:
    st.subheader("Advanced Analytics & Visualizations")
    if not hist_f.empty:

        st.markdown("**Correlation Heatmap:**")
        cols_for_corr = ["consumption_kw"]
        # Add prediction column for correlation
        if not pred_f.empty and "predicted_consumption_kw" in pred_f.columns:
            hist_pred = hist_f.merge(pred_f[["ts","predicted_consumption_kw"]], on="ts", how="left")
            cols_for_corr.append("predicted_consumption_kw")
            corr_df = hist_pred[cols_for_corr].corr()
        else:
            corr_df = hist_f[cols_for_corr].corr()
        fig_heat = px.imshow(corr_df, text_auto=True, color_continuous_scale="Viridis", title="Feature Correlation")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Distribution (Box Plot) by Node:**")
        fig_box = px.box(hist_f, x="node_id", y="consumption_kw", color="node_id", title="Consumption Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("**Daily Aggregation:**")
        agg = hist_f.set_index("ts").resample("D")["consumption_kw"].sum().reset_index()
        agg["date"] = agg["ts"].dt.date
        fig_day = px.bar(agg, x="date", y="consumption_kw", labels={"consumption_kw": "kW·day"}, title="Daily Consumption (Sum)", text_auto=True)
        st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.info("No historical data for analytics.")

# --- Anomalies Tab ---
with tab4:
    st.subheader("Anomalies — Last Week")
    if anom_f.empty:
        st.info("No anomalies for selected node/filters.")
    else:
        st.dataframe(anom_f.sort_values(["ts"]).reset_index(drop=True).astype({"ts": "datetime64[ns]"}))
        # Bar: Anomalies count by type
        anom_counts = anom_f["anomaly_type"].value_counts().reset_index()
        anom_counts.columns = ["anomaly_type","count"]
        fig2 = px.bar(anom_counts, x="anomaly_type", y="count", title="Anomaly counts by type", color="anomaly_type")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")
        st.write("**Recent anomaly details:**")
        st.dataframe(anom_f[["ts","node_id","actual","anomaly_type"]].sort_values("ts", ascending=False).head(10))

st.markdown("---")
st.caption("""
    Built with Streamlit · Data from /opt/electricity-pipeline/output_local · Professional MNC-grade dashboard
""")
