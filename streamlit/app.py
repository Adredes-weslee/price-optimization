import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Add the parent directory to path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add the streamlit directory to path
streamlit_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(streamlit_dir)

from src import config, utils
from utils import st_utils

# Function to load sample data if available
def load_sample_data():
    # Load raw data
    raw_data_path = Path(parent_dir) / "data" / "raw" / "sales_data.csv"
    if raw_data_path.exists() and st.session_state.raw_data is None:
        try:
            df = pd.read_csv(raw_data_path)
            st.session_state.raw_data = df
        except Exception as e:
            st.error(f"Error loading raw sample data: {e}")
    
    # Load processed data
    processed_data_path = Path(parent_dir) / "data" / "processed" / "aggregated_df.csv"
    if processed_data_path.exists() and st.session_state.processed_data is None:
        try:
            df = pd.read_csv(processed_data_path)
            st.session_state.processed_data = df
        except Exception as e:
            st.error(f"Error loading processed sample data: {e}")
    
    # Load segmentation results
    segmentation_path = Path(parent_dir) / "data" / "segmentation" / "customer_segmentation_df.csv"
    if segmentation_path.exists() and st.session_state.segmentation_results is None:
        try:
            df = pd.read_csv(segmentation_path)
            st.session_state.segmentation_results = df
        except Exception as e:
            st.error(f"Error loading segmentation results: {e}")
    
    # Load elasticity results
    elasticity_path = Path(parent_dir) / "data" / "optimization" / "price_elasticities_df.csv"
    if elasticity_path.exists() and st.session_state.elasticity_results is None:
        try:
            df = pd.read_csv(elasticity_path)
            st.session_state.elasticity_results = df
        except Exception as e:
            st.error(f"Error loading elasticity results: {e}")
    
    # Load optimization results
    optimization_path = Path(parent_dir) / "data" / "optimization" / "revenue_optimization_results.csv"
    if optimization_path.exists() and st.session_state.optimization_results is None:
        try:
            df = pd.read_csv(optimization_path)
            st.session_state.optimization_results = df
        except Exception as e:
            st.error(f"Error loading optimization results: {e}")

st.set_page_config(
    page_title="Retail Price Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to override sidebar text formatting
st.markdown("""
    <style>
    /* Target the sidebar navigation items */
    section[data-testid="stSidebar"] .css-pkbazv, 
    section[data-testid="stSidebar"] .css-17lntkn,
    section[data-testid="stSidebar"] span.css-10trblm {
        text-transform: uppercase !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add this at the end of your app.py file
st.sidebar.markdown("""
<div class="footer">
Created with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)

# Initialize session state
st_utils.initialize_session_state()

# Load sample data if raw_data is not already loaded
load_sample_data()

st.title("Retail Price Optimization Dashboard")
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