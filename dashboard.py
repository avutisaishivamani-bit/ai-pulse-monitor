import streamlit as st                          # Interactive Web UI framework
import pandas as pd                             # Data manipulation and SQL query results formatting
import sqlite3                                  # SQLite connection
import plotly.express as px                     # High-level interactive charts
import plotly.graph_objects as go               # Low-level multi-trace charts (Raw vs Filtered)
import os                                       # File checking
from pulsemaintain.config import DB_PATH, DRIFT_HOUR, TOTAL_HOURS  # Configuration constants
import time                                     # UI delay handling

# Configure Streamlit browser tab title and wide screen layout
st.set_page_config(page_title="PulseMaintain AI", layout="wide")
st.title("PulseMaintain AI: Predictive Maintenance Dashboard")
st.markdown("**Powered by Machine Learning:** `RandomForestRegressor` (for RUL Prediction) & `RandomForestClassifier` (for Failure Probability)")

# Cached DB connection to maintain single persistent connection across widget reruns
@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        # Auto-initialize the DB so the user immediately sees the 100% health state
        from pulsemaintain.db import init_db
        init_db()
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_conn()

# Sidebar: Manual Refresh button to re-read updated SQLite data
if st.sidebar.button("Refresh Data"):
    st.rerun()
    
st.sidebar.markdown("### Controls")

# Sidebar Control 1: Reset database back to hour 0 with all machines at 100% health
if st.sidebar.button("1. Reset to Initial State"):
    st.cache_resource.clear()
    try:
        conn.close()
    except Exception:
        pass
    from pulsemaintain.db import reset_db, init_db
    reset_db()
    init_db()
    st.success("Factory reset to Day 0 (100% Health)")
    time.sleep(0.5)
    st.rerun()

# Sidebar Control 2: Trigger full 400-hour simulation run with fresh randomized data
if st.sidebar.button("2. Run 400-Hour Simulation"):
    with st.spinner("Running simulation for 400 hours with fresh dynamic data... please wait (~15 seconds)."):
        st.cache_resource.clear()
        try:
            conn.close()
        except Exception:
            pass
        from run_demo import run_simulation
        run_simulation()
    
    st.success("Simulation Complete! Fresh data generated.")
    time.sleep(0.5)
    st.rerun()

# Four navigation tabs for different aspects of the system
tab1, tab2, tab3, tab4 = st.tabs(["Asset Health", "Telemetry", "Alerts & Work Orders", "MLOps Monitoring"])

# ==================== TAB 1: ASSET HEALTH & PREDICTED RUL ====================
with tab1:
    st.header("Asset Health Overview")
    df_assets = pd.read_sql("SELECT * FROM Industrial_Asset", conn)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", len(df_assets))                                  # KPI: Total factory machines
    col2.metric("Average Health", f"{df_assets['health_score'].mean():.1f}%")    # KPI: Mean health score across fleet
    
    # Query latest ML predictions (RUL and failure probability) for each asset
    df_latest_pred = pd.read_sql('''
        SELECT p.asset_id, p.predicted_rul, p.failure_probability
        FROM Prediction_Log p
        INNER JOIN (
            SELECT asset_id, MAX(hour) as max_hour
            FROM Prediction_Log GROUP BY asset_id
        ) latest ON p.asset_id = latest.asset_id AND p.hour = latest.max_hour
    ''', conn)
    
    if not df_latest_pred.empty:
        # Merge physical asset info with ML predictions table
        df_joined = pd.merge(df_assets, df_latest_pred, on='asset_id', how='left')
        st.dataframe(df_joined[['asset_id', 'name', 'health_score', 'operating_hours', 'predicted_rul', 'failure_probability']], use_container_width=True)
        
        # Interactive selector for individual machine RUL history
        st.subheader("Historical Prediction Trend")
        asset_sel = st.selectbox("Select Asset for History", df_assets['asset_id'].unique())
        df_history = pd.read_sql(f"SELECT hour, predicted_rul, failure_probability FROM Prediction_Log WHERE asset_id = '{asset_sel}' ORDER BY hour DESC", conn)
        
        # Plot Remaining Useful Life (RUL) downward degradation line
        fig = px.line(df_history, x='hour', y='predicted_rul', title=f'RUL Trend for {asset_sel}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical prediction raw records
        st.subheader(f"Tabular Data: {asset_sel}")
        st.dataframe(df_history, use_container_width=True)

# ==================== TAB 2: TELEMETRY & SIGNAL FILTERING ====================
with tab2:
    st.header("Telemetry Data")
    df_assets = pd.read_sql("SELECT asset_id FROM Industrial_Asset", conn)
    if not df_assets.empty:
        asset_sel = st.selectbox("Select Asset for Telemetry", df_assets['asset_id'].unique())
        df_telemetry = pd.read_sql(f"SELECT * FROM Telemetry_Data WHERE asset_id = '{asset_sel}'", conn)
        
        if not df_telemetry.empty:
            st.subheader("Vibration (Raw vs Filtered)")
            fig = go.Figure()
            # Show noisy raw sensor signal in faint line
            fig.add_trace(go.Scatter(x=df_telemetry['hour'], y=df_telemetry['vibration_raw'], name='Raw', opacity=0.3))
            # Show cleaned Moving-Average filtered signal in solid line
            fig.add_trace(go.Scatter(x=df_telemetry['hour'], y=df_telemetry['vibration_filtered'], name='Filtered'))
            # Vertical red line marking calibration drift injection at hour 200
            fig.add_vline(x=DRIFT_HOUR, line_dash="dash", line_color="red", annotation_text="Calibration Drift")
            st.plotly_chart(fig, use_container_width=True)
            
            # Filtered temperature sensor trajectory
            st.subheader("Temperature (Filtered)")
            fig2 = px.line(df_telemetry, x='hour', y='temperature_filtered')
            fig2.add_vline(x=DRIFT_HOUR, line_dash="dash", line_color="red", annotation_text="Calibration Drift")
            st.plotly_chart(fig2, use_container_width=True)

# ==================== TAB 3: ALERTS & WORK ORDERS ====================
with tab3:
    st.header("Alerts & Work Orders")
    df_alerts = pd.read_sql("SELECT * FROM Maintenance_Alert ORDER BY hour DESC", conn)
    df_wo = pd.read_sql("SELECT * FROM Work_Order ORDER BY creation_hour DESC", conn)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Alerts")        # Shows critical automated alerts when risk threshold breached
        st.dataframe(df_alerts.head(10), use_container_width=True)
    with col2:
        st.subheader("Work Orders")          # Shows auto-dispatched maintenance work orders
        st.dataframe(df_wo.head(10), use_container_width=True)
        
    # Bar chart showing work order load balancing across technicians (Alice, Bob, Charlie)
    st.subheader("Technician Workload")
    df_tech = pd.read_sql('''
        SELECT t.name, COUNT(w.work_order_id) as open_orders
        FROM Technician t
        LEFT JOIN Work_Order w ON t.technician_id = w.technician_id AND w.status = 'OPEN'
        GROUP BY t.name
    ''', conn)
    fig3 = px.bar(df_tech, x='name', y='open_orders', title="Open Work Orders per Technician")
    st.plotly_chart(fig3, use_container_width=True)

# ==================== TAB 4: MLOPS MONITORING & MODEL DRIFT ====================
with tab4:
    st.header("MLOps Monitoring")
    
    # Model Registry: Displays registered versions (v1 active -> retired, v2 retrained)
    st.subheader("Model Registry")
    df_models = pd.read_sql("SELECT * FROM Model_Version ORDER BY version_id DESC", conn)
    st.dataframe(df_models[['model_name', 'version_number', 'training_hour', 'metrics', 'status']], use_container_width=True)
    
    # Population Stability Index (PSI) drift chart
    st.subheader("Model Drift (PSI & KS)")
    df_drift = pd.read_sql("SELECT * FROM Model_Drift_Metric ORDER BY hour", conn)
    if not df_drift.empty:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_drift['hour'], y=df_drift['psi_value'], name='PSI Value'))
        # Red horizontal dashed line indicating 0.25 threshold
        fig4.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Drift Threshold")
        st.plotly_chart(fig4, use_container_width=True)
        
        # Table of confirmed drift occurrences that triggered automated retraining
        st.dataframe(df_drift[df_drift['is_drift_detected'] == 1], use_container_width=True)

