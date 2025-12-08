"""
Monitoring Dashboard

Visualizes Evidently AI drift reports.
Run with: poetry run streamlit run scripts/monitoring_dashboard.py
"""
import streamlit as st
from pathlib import Path
import json
import glob
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="Data Drift Monitoring", layout="wide")

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "monitoring"
LOGS_DIR = PROJECT_ROOT / "logs"

st.title("🔎 Data Drift Monitoring Dashboard")

# Sidebar
st.sidebar.title("Controls")

# 1. Production Logs Stats
st.header("Production Statistics")
log_file = LOGS_DIR / "production_logs.csv"

if log_file.exists():
    try:
        df_logs = pd.read_csv(log_file)
        st.metric("Total Predictions Served", len(df_logs))
        
        if "timestamp" in df_logs.columns:
            st.text(f"Last prediction: {df_logs['timestamp'].iloc[-1]}")
            
        if "risk_level" in df_logs.columns:
            st.bar_chart(df_logs["risk_level"].value_counts())
    except Exception as e:
        st.error(f"Error reading logs: {e}")
else:
    st.warning("No production logs found yet.")

# 2. Drift Reports
st.header("Drift Reports")

reports = sorted(glob.glob(str(REPORTS_DIR / "*.html")), reverse=True)

if not reports:
    st.info("No drift reports generated yet. Run `scripts/monitor_drift.py`.")
else:
    selected_report = st.selectbox("Select Report", [Path(r).name for r in reports])
    
    if selected_report:
        report_path = REPORTS_DIR / selected_report
        
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        st.download_button(
            label="Download HTML Report",
            data=html_content,
            file_name=selected_report,
            mime="text/html"
        )
        
        # Display report in iframe
        components.html(html_content, height=1000, scrolling=True)
