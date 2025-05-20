import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config, data_preprocessing, utils

st.title("Data Preprocessing")

# File upload
uploaded_file = st.file_uploader("Upload sales data CSV", type="csv")

if uploaded_file is not None:
    # Read the data
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_data = df
        st.success(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Display raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10))
        
        # Data preprocessing options
        st.subheader("Preprocessing Options")
        
        with st.expander("Preprocessing Configuration"):
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            reconcile_inv = st.checkbox("Reconcile Inventory Information", value=True)
            aggregate_trans = st.checkbox("Aggregate Transactions", value=True)
            
        if st.button("Run Preprocessing"):
            st.info("Running preprocessing...")
            
            # Custom preprocessing based on selections
            processed_df = df.copy()
            
            if reconcile_inv:
                processed_df = data_preprocessing.reconcile_inventory_info(processed_df)
            
            if handle_missing:
                processed_df = data_preprocessing.handle_missing_values(processed_df)
            
            if aggregate_trans:
                processed_df = data_preprocessing.aggregate_transactions(processed_df)
            
            # Type casting and feature engineering
            processed_df = data_preprocessing.cast_data_types(processed_df)
            processed_df = data_preprocessing.engineer_features(processed_df)
            
            st.session_state.processed_data = processed_df
            st.success("Preprocessing complete!")
            
            # Preview processed data
            st.subheader("Processed Data Preview")
            st.dataframe(processed_df.head(10))
            
    except Exception as e:
        st.error(f"Error processing data: {e}")