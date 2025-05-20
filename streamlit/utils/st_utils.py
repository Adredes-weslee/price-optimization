"""
Utility functions for Streamlit app components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config

def initialize_session_state():
    """
    Initialize session state variables if they don't exist.
    """
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "segmentation_results" not in st.session_state:
        st.session_state.segmentation_results = None
    if "elasticity_results" not in st.session_state:
        st.session_state.elasticity_results = None
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None

def check_data_requirements(required_data_key, message=None):
    """
    Check if required data exists in session state and display warning if not.
    
    Args:
        required_data_key: Key in session_state to check
        message: Custom warning message (optional)
    
    Returns:
        bool: True if data exists, False otherwise
    """
    if required_data_key not in st.session_state or st.session_state[required_data_key] is None:
        if message:
            st.warning(message)
        else:
            st.warning(f"Required data '{required_data_key}' not found. Please complete previous steps first.")
        return False
    return True

def display_data_info(df, title="Data Overview"):
    """
    Display basic information about a DataFrame.
    """
    st.subheader(title)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {len(df)}")
        st.write(f"**Columns:** {len(df.columns)}")
        
    with col2:
        if config.COL_TRANSACTION_DATE in df.columns:
            min_date = df[config.COL_TRANSACTION_DATE].min()
            max_date = df[config.COL_TRANSACTION_DATE].max()
            st.write(f"**Date Range:** {min_date} to {max_date}")
        
        if config.COL_CUSTOMER_CODE in df.columns:
            st.write(f"**Unique Customers:** {df[config.COL_CUSTOMER_CODE].nunique()}")
            
        if config.COL_INVENTORY_CODE in df.columns:
            st.write(f"**Unique Products:** {df[config.COL_INVENTORY_CODE].nunique()}")

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percent(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def display_summary_metrics(df, metric_columns, title="Summary Metrics"):
    """
    Display summary metrics in a nice format
    
    Args:
        df: DataFrame containing metrics
        metric_columns: List of column names to summarize
        title: Section title
    """
    st.subheader(title)
    
    # Calculate metrics
    metrics = {}
    for col in metric_columns:
        metrics[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median()
        }
    
    # Display in columns
    cols = st.columns(len(metric_columns))
    for i, (col_name, values) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label=f"{col_name}", value=f"{values['mean']:.2f}")
            st.write(f"**Range:** {values['min']:.2f} to {values['max']:.2f}")
            st.write(f"**Median:** {values['median']:.2f}")

def load_latest_results():
    """
    Load the latest results from files if they exist.
    Useful when re-running the app without having to rerun all calculations.
    """
    # Try to load processed data
    if st.session_state.processed_data is None and config.AGGREGATED_DATA_PATH.exists():
        try:
            from src import utils
            st.session_state.processed_data = utils.load_csv_data(config.AGGREGATED_DATA_PATH)
        except Exception:
            pass
    
    # Try to load segmentation results
    if st.session_state.segmentation_results is None:
        segmentation_path = config.SEGMENTATION_OUTPUT_DIR / "customer_segments.csv"
        if segmentation_path.exists():
            try:
                from src import utils
                st.session_state.segmentation_results = utils.load_csv_data(segmentation_path)
            except Exception:
                pass
    
    # Try to load elasticity results
    if st.session_state.elasticity_results is None:
        elasticity_path = config.OPTIMIZATION_OUTPUT_DIR / "price_elasticities_calculated.csv"
        if elasticity_path.exists():
            try:
                from src import utils
                st.session_state.elasticity_results = utils.load_csv_data(elasticity_path)
            except Exception:
                pass
    
    # Try to load optimization results
    if st.session_state.optimization_results is None:
        optimization_path = config.OPTIMIZATION_OUTPUT_DIR / "optimized_prices.csv"
        if optimization_path.exists():
            try:
                from src import utils
                st.session_state.optimization_results = utils.load_csv_data(optimization_path)
            except Exception:
                pass