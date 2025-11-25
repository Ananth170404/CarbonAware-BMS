# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# --- Advanced Professional CSS with Brighter Colors ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto:wght@300;400;500;700&family=Orbitron:wght@400;700;900&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #1a1d2e 0%, #2d3561 50%, #1f2642 100%);
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }
        
        .main {
            background: transparent;
        }
        
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        
        /* Main Title Styling */
        .main-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 3.5em;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(90deg, #00e5ff 0%, #00b4d8 50%, #0077b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px rgba(0,229,255,0.4);
            margin-bottom: 0.2em;
            letter-spacing: 3px;
        }
        
        .main-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1.3em;
            font-weight: 300;
            text-align: center;
            color: #e0e0e0;
            margin-bottom: 2em;
            letter-spacing: 1px;
        }
        
        /* Section Headers */
        h2 {
            font-family: 'Poppins', sans-serif !important;
            font-size: 2em !important;
            font-weight: 600 !important;
            color: #00e5ff !important;
            border-bottom: 3px solid #00b4d8;
            padding-bottom: 0.5em;
            margin-top: 1.5em;
            margin-bottom: 1em;
            letter-spacing: 1.5px;
        }
        
        h3 {
            font-family: 'Poppins', sans-serif !important;
            font-size: 1.5em !important;
            font-weight: 500 !important;
            color: #48cae4 !important;
            margin-top: 1.2em;
            letter-spacing: 1px;
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2em !important;
            font-weight: 700 !important;
            color: #00e5ff !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1.1em !important;
            font-weight: 400 !important;
            color: #ffffff !important;
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Sidebar Styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2d3561 0%, #1a1d2e 100%);
        }
        
        .css-1d391kg .stSelectbox label, 
        .css-1d391kg .stCheckbox label,
        .css-1d391kg .stRadio label {
            color: #00e5ff !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 500 !important;
            font-size: 1.1em !important;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(45, 53, 97, 0.6);
            border-radius: 10px;
            padding: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Poppins', sans-serif;
            font-size: 1.15em;
            font-weight: 500;
            color: #e0e0e0;
            background-color: transparent;
            border-radius: 8px;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #00e5ff 0%, #0096c7 100%);
            color: #000000 !important;
            font-weight: 600;
        }
        
        /* DataFrames */
        .dataframe {
            font-family: 'Roboto', sans-serif !important;
            background-color: #2d3561 !important;
            color: #ffffff !important;
        }
        
        /* Buttons */
        .stDownloadButton button {
            background: linear-gradient(90deg, #00b4d8 0%, #00e5ff 100%);
            color: #000000;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.5em;
            transition: all 0.3s ease;
        }
        
        .stDownloadButton button:hover {
            background: linear-gradient(90deg, #00e5ff 0%, #48cae4 100%);
            box-shadow: 0 0 20px rgba(0,229,255,0.6);
        }
        
        /* Info boxes */
        .stAlert {
            background-color: rgba(0, 180, 216, 0.15);
            border-left: 4px solid #00b4d8;
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }
        
        /* Dividers */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00b4d8, transparent);
            margin: 2em 0;
        }
        
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Electricity Analytics Platform",
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

# --- Sidebar Configuration ---
st.sidebar.markdown("""
    <div style='text-align:center; padding:20px 0;'>
        <h1 style='font-family:Orbitron; color:#00e5ff; font-size:1.8em; letter-spacing:2px;'>ELECTRICITY</h1>
        <p style='font-family:Poppins; color:#48cae4; font-size:1em;'>Analytics Platform</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#00e5ff; font-family:Poppins;'>Filters & Controls</h3>", unsafe_allow_html=True)

all_nodes = sorted(set(
    list(history["node_id"].unique()) if not history.empty else []
    + list(predictions["node_id"].unique()) if not predictions.empty else []
    + list(anomalies["node_id"].unique()) if not anomalies.empty else []
))
if not all_nodes:
    all_nodes = ["(no nodes found)"]
node = st.sidebar.selectbox("Select Node", options=["All"] + all_nodes, index=0)

if not history.empty:
    min_ts = history["ts"].min().date()
    max_ts = history["ts"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_ts, max_ts), min_value=min_ts, max_value=max_ts)
else:
    date_range = None

show_history = st.sidebar.checkbox("Show Historical Data", value=True)
show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
show_anomalies = st.sidebar.checkbox("Highlight Anomalies", value=True)

st.sidebar.markdown("---")
chart_theme = st.sidebar.radio("Chart Theme", options=["plotly_dark", "plotly_white"], index=0)

st.sidebar.markdown("---")
st.sidebar.info("Data Source: /opt/electricity-pipeline/output_local")

# --- Helper Functions ---
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

# --- Main Title ---
st.markdown("<h1 class='main-title'>ELECTRICITY ANALYTICS PLATFORM</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Advanced Monitoring, Forecasting & Anomaly Detection System</p>", unsafe_allow_html=True)

# --- Multi-Tab Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Dashboard",
    "Time Series Analysis",
    "Advanced Analytics",
    "Anomaly Detection",
    "Data Export"
])

# ==================== TAB 1: EXECUTIVE DASHBOARD ====================
with tab1:
    st.markdown("## Executive Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Nodes", int(history["node_id"].nunique() if not history.empty else 0))
    col2.metric("Historical Records", f"{len(history):,}")
    col3.metric("Predictions", f"{len(predictions):,}")
    col4.metric("Anomalies Detected", int(len(anomalies)))
    col5.metric("Active Filters", "Yes" if node != "All" else "No")
    
    st.markdown("---")
    
    if not hist_f.empty:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### Key Performance Indicators")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric("Average Consumption", f"{hist_f['consumption_kw'].mean():.2f} kW")
            kpi_col2.metric("Peak Consumption", f"{hist_f['consumption_kw'].max():.2f} kW")
            kpi_col3.metric("Minimum Consumption", f"{hist_f['consumption_kw'].min():.2f} kW")
        
        with col_right:
            if not anom_f.empty:
                st.markdown("### Anomaly Breakdown")
                anom_counts = anom_f["anomaly_type"].value_counts().reset_index()
                anom_counts.columns = ["anomaly_type", "count"]
                fig_pie = px.pie(
                    anom_counts, 
                    names="anomaly_type", 
                    values="count",
                    color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#fb5607', '#8338ec'],
                    template=chart_theme
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig_pie.update_layout(
                    font=dict(color='#ffffff', size=13),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    if not hist_f.empty:
        st.markdown("### Consumption Trend Overview")
        fig_overview = px.area(
            hist_f, 
            x="ts", 
            y="consumption_kw",
            color="node_id",
            template=chart_theme,
            color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec', '#fb5607']
        )
        fig_overview.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Consumption (kW)",
            hovermode='x unified',
            height=400,
            font=dict(color='#ffffff', size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,53,97,0.3)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_overview, use_container_width=True)

# ==================== TAB 2: TIME SERIES ANALYSIS ====================
with tab2:
    st.markdown("## Time Series Analysis")
    
    fig = go.Figure()
    
    colors_hist = ['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec', '#fb5607']
    colors_pred = ['#48cae4', '#ff5d8f', '#ffd60a', '#40ffb3', '#a663ff', '#ff7f51']
    
    if show_history and not hist_f.empty:
        for idx, node_id in enumerate(hist_f["node_id"].unique()):
            node_data = hist_f[hist_f["node_id"] == node_id]
            fig.add_trace(go.Scatter(
                x=node_data["ts"],
                y=node_data["consumption_kw"],
                mode='lines',
                name=f"Node {node_id} (Historical)",
                line=dict(width=2.5, color=colors_hist[idx % len(colors_hist)])
            ))
    
    if show_predictions and not pred_f.empty:
        pred_f = pred_f.copy()
        if "ts" in pred_f.columns:
            pred_f["ts"] = pd.to_datetime(pred_f["ts"])
        for idx, node_id in enumerate(pred_f["node_id"].unique()):
            node_data = pred_f[pred_f["node_id"] == node_id]
            fig.add_trace(go.Scatter(
                x=node_data["ts"],
                y=node_data["predicted_consumption_kw"],
                mode='lines',
                name=f"Node {node_id} (Predicted)",
                line=dict(width=2.5, dash='dash', color=colors_pred[idx % len(colors_pred)])
            ))
    
    if show_anomalies and not anom_f.empty:
        anom_plot = anom_f.copy()
        anom_plot["ts"] = pd.to_datetime(anom_plot["ts"])
        fig.add_trace(go.Scatter(
            x=anom_plot["ts"],
            y=anom_plot["actual"],
            mode='markers',
            name="Anomaly",
            marker=dict(size=12, symbol='x', color='#ff006e', line=dict(width=2, color='#ffffff'))
        ))
    
    fig.update_layout(
        template=chart_theme,
        xaxis_title="Timestamp",
        yaxis_title="Consumption (kW)",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#ffffff', size=13),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(45,53,97,0.3)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if not hist_f.empty:
        st.markdown("### Hourly Consumption Pattern")
        hist_f_copy = hist_f.copy()
        hist_f_copy['hour'] = hist_f_copy['ts'].dt.hour
        hourly_avg = hist_f_copy.groupby('hour')['consumption_kw'].mean().reset_index()
        
        fig_hourly = px.bar(
            hourly_avg,
            x='hour',
            y='consumption_kw',
            template=chart_theme,
            color='consumption_kw',
            color_continuous_scale=['#0077b6', '#00b4d8', '#00e5ff', '#06ffa5', '#ffbe0b', '#ff006e']
        )
        fig_hourly.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Consumption (kW)",
            height=350,
            font=dict(color='#ffffff', size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,53,97,0.3)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

# ==================== TAB 3: ADVANCED ANALYTICS ====================
with tab3:
    st.markdown("## Advanced Analytics & Visualizations")
    
    if not hist_f.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Distribution Analysis")
            fig_box = px.box(
                hist_f, 
                x="node_id", 
                y="consumption_kw",
                color="node_id",
                template=chart_theme,
                color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec', '#fb5607']
            )
            fig_box.update_layout(
                xaxis_title="Node ID",
                yaxis_title="Consumption (kW)",
                showlegend=False,
                height=400,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45,53,97,0.3)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            st.markdown("### Violin Distribution")
            fig_violin = px.violin(
                hist_f,
                x="node_id",
                y="consumption_kw",
                color="node_id",
                box=True,
                template=chart_theme,
                color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec', '#fb5607']
            )
            fig_violin.update_layout(
                xaxis_title="Node ID",
                yaxis_title="Consumption (kW)",
                showlegend=False,
                height=400,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45,53,97,0.3)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Daily Aggregation Analysis")
        agg = hist_f.set_index("ts").resample("D")["consumption_kw"].agg(['sum', 'mean', 'max', 'min']).reset_index()
        agg["date"] = agg["ts"].dt.date
        
        fig_daily = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Daily Sum", "Daily Average", "Daily Maximum", "Daily Minimum")
        )
        
        fig_daily.add_trace(
            go.Bar(x=agg["date"], y=agg["sum"], name="Sum", marker_color='#00e5ff'),
            row=1, col=1
        )
        fig_daily.add_trace(
            go.Scatter(x=agg["date"], y=agg["mean"], name="Average", mode='lines+markers', 
                      line=dict(color='#06ffa5', width=3), marker=dict(size=8)),
            row=1, col=2
        )
        fig_daily.add_trace(
            go.Scatter(x=agg["date"], y=agg["max"], name="Maximum", mode='lines+markers', 
                      line=dict(color='#ff006e', width=3), marker=dict(size=8)),
            row=2, col=1
        )
        fig_daily.add_trace(
            go.Scatter(x=agg["date"], y=agg["min"], name="Minimum", mode='lines+markers', 
                      line=dict(color='#ffbe0b', width=3), marker=dict(size=8)),
            row=2, col=2
        )
        
        fig_daily.update_layout(
            template=chart_theme,
            height=600,
            showlegend=False,
            font=dict(color='#ffffff', size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,53,97,0.3)'
        )
        fig_daily.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_daily.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_daily, use_container_width=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### Statistical Summary")
            stats_df = hist_f.groupby('node_id')['consumption_kw'].describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        with col4:
            st.markdown("### Correlation Heatmap")
            if not pred_f.empty and "predicted_consumption_kw" in pred_f.columns:
                hist_pred = hist_f.merge(pred_f[["ts", "node_id", "predicted_consumption_kw"]], on=["ts", "node_id"], how="left")
                corr_cols = ["consumption_kw", "predicted_consumption_kw"]
                corr_df = hist_pred[corr_cols].corr()
            else:
                corr_df = hist_f[["consumption_kw"]].corr()
            
            fig_heat = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale=['#0077b6', '#00b4d8', '#00e5ff', '#06ffa5', '#ffbe0b', '#ff006e'],
                template=chart_theme
            )
            fig_heat.update_layout(
                height=300,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Consumption Histogram")
        fig_hist = px.histogram(
            hist_f,
            x="consumption_kw",
            nbins=50,
            color="node_id",
            template=chart_theme,
            marginal="violin",
            color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec', '#fb5607']
        )
        fig_hist.update_layout(
            xaxis_title="Consumption (kW)",
            yaxis_title="Frequency",
            height=400,
            font=dict(color='#ffffff', size=13),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(45,53,97,0.3)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ==================== TAB 4: ANOMALY DETECTION ====================
with tab4:
    st.markdown("## Anomaly Detection & Analysis")
    
    if anom_f.empty:
        st.info("No anomalies detected for the selected filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomalies", len(anom_f))
        col2.metric("Anomaly Types", anom_f["anomaly_type"].nunique())
        col3.metric("Affected Nodes", anom_f["node_id"].nunique())
        
        st.markdown("---")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### Anomaly Timeline")
            fig_anom_time = px.scatter(
                anom_f,
                x="ts",
                y="actual",
                color="anomaly_type",
                size="actual",
                template=chart_theme,
                color_discrete_sequence=['#ff006e', '#00e5ff', '#ffbe0b', '#06ffa5', '#8338ec'],
                hover_data=["node_id"]
            )
            fig_anom_time.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Actual Consumption (kW)",
                height=400,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45,53,97,0.3)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_anom_time, use_container_width=True)
        
        with col_right:
            st.markdown("### Anomaly Type Distribution")
            anom_counts = anom_f["anomaly_type"].value_counts().reset_index()
            anom_counts.columns = ["anomaly_type", "count"]
            fig_anom_bar = px.bar(
                anom_counts,
                x="anomaly_type",
                y="count",
                color="anomaly_type",
                template=chart_theme,
                color_discrete_sequence=['#00e5ff', '#ff006e', '#ffbe0b', '#06ffa5', '#8338ec']
            )
            fig_anom_bar.update_layout(
                xaxis_title="Anomaly Type",
                yaxis_title="Count",
                showlegend=False,
                height=400,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45,53,97,0.3)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_anom_bar, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Detailed Anomaly Records")
        display_cols = [col for col in ["ts", "node_id", "actual", "anomaly_type"] if col in anom_f.columns]
        st.dataframe(
            anom_f[display_cols].sort_values("ts", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        if "node_id" in anom_f.columns:
            st.markdown("### Anomalies by Node")
            node_anom = anom_f.groupby("node_id").size().reset_index(name="count")
            fig_node_anom = px.bar(
                node_anom,
                x="node_id",
                y="count",
                template=chart_theme,
                color="count",
                color_continuous_scale=['#0077b6', '#00b4d8', '#00e5ff', '#ff006e']
            )
            fig_node_anom.update_layout(
                xaxis_title="Node ID",
                yaxis_title="Anomaly Count",
                height=350,
                font=dict(color='#ffffff', size=13),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(45,53,97,0.3)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_node_anom, use_container_width=True)

# ==================== TAB 5: DATA EXPORT ====================
with tab5:
    st.markdown("## Data Export & Download")
    
    st.markdown("### Download Filtered Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Historical Data")
        if not hist_f.empty:
            csv_hist = hist_f.to_csv(index=False)
            st.download_button(
                "Download History CSV",
                csv_hist,
                file_name=f"history_filtered_{node}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.metric("Records", len(hist_f))
        else:
            st.info("No historical data available")
    
    with col2:
        st.markdown("#### Prediction Data")
        if not pred_f.empty:
            csv_pred = pred_f.to_csv(index=False)
            st.download_button(
                "Download Predictions CSV",
                csv_pred,
                file_name=f"predictions_{node}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.metric("Records", len(pred_f))
        else:
            st.info("No prediction data available")
    
    with col3:
        st.markdown("#### Anomaly Data")
        if not anom_f.empty:
            csv_anom = anom_f.to_csv(index=False)
            st.download_button(
                "Download Anomalies CSV",
                csv_anom,
                file_name=f"anomalies_{node}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.metric("Records", len(anom_f))
        else:
            st.info("No anomaly data available")
    
    st.markdown("---")
    
    st.markdown("### Data Preview")
    preview_choice = st.radio("Select dataset to preview", ["Historical", "Predictions", "Anomalies"], horizontal=True)
    
    if preview_choice == "Historical" and not hist_f.empty:
        st.dataframe(hist_f.head(100), use_container_width=True, height=400)
    elif preview_choice == "Predictions" and not pred_f.empty:
        st.dataframe(pred_f.head(100), use_container_width=True, height=400)
    elif preview_choice == "Anomalies" and not anom_f.empty:
        st.dataframe(anom_f.head(100), use_container_width=True, height=400)
    else:
        st.info("No data available for preview")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align:center; padding:20px; font-family:Roboto; color:#48cae4;'>
        <p>Electricity Analytics Platform | Data Pipeline: /opt/electricity-pipeline/output_local</p>
        <p style='font-size:0.9em; color:#e0e0e0;'>Professional Dashboard | Real-time Monitoring & Forecasting</p>
    </div>
""", unsafe_allow_html=True)
