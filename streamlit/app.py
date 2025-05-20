import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add the streamlit directory to path
streamlit_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(streamlit_dir)

from src import config, utils
from utils import st_utils

st.set_page_config(
    page_title="CS Tay Price Optimization",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
st_utils.initialize_session_state()

st.title("CS Tay Price Optimization Dashboard")
st.markdown("---")

# Home page content
st.write("""
## Welcome to the Price Optimization Dashboard

This tool allows you to:
1. **Preprocess** your sales data
2. Perform **Customer Segmentation** using RFM analysis and K-means clustering
3. Calculate **Price Elasticity** for your products
4. Run **Revenue Optimization** to find optimal pricing strategies

Use the sidebar to navigate between different sections.
""")

# Display information about current data state
st.sidebar.header("Data Status")
if st.session_state.raw_data is not None:
    st.sidebar.success(f"✓ Raw data loaded ({len(st.session_state.raw_data)} rows)")
else:
    st.sidebar.warning("⚠ Raw data not loaded")

if st.session_state.processed_data is not None:
    st.sidebar.success(f"✓ Processed data ready ({len(st.session_state.processed_data)} rows)")

if st.session_state.segmentation_results is not None:
    st.sidebar.success(f"✓ Customer segmentation complete ({len(st.session_state.segmentation_results)} customers)")

if st.session_state.elasticity_results is not None:
    st.sidebar.success("✓ Price elasticity calculated")

if st.session_state.optimization_results is not None:
    st.sidebar.success("✓ Price optimization complete")