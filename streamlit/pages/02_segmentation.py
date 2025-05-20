import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config, customer_segmentation, utils

st.title("Customer Segmentation")

# Check if processed data exists
if st.session_state.processed_data is None:
    st.warning("Please complete the data preprocessing step first!")
    st.stop()

# Display data overview
st.subheader("Data Overview")
st.write(f"Number of transactions: {len(st.session_state.processed_data)}")
st.write(f"Number of unique customers: {st.session_state.processed_data[config.COL_CUSTOMER_CODE].nunique()}")
st.write(f"Date range: {st.session_state.processed_data[config.COL_TRANSACTION_DATE].min()} to {st.session_state.processed_data[config.COL_TRANSACTION_DATE].max()}")

# RFM Configuration
st.subheader("RFM Analysis Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    rfm_r_bins = st.slider("Recency Bins", min_value=5, max_value=20, value=10)
with col2:
    rfm_f_bins = st.slider("Frequency Bins", min_value=5, max_value=20, value=10)
with col3:
    rfm_m_bins = st.slider("Monetary Bins", min_value=5, max_value=20, value=10)

# K-means Configuration
st.subheader("K-Means Clustering Configuration")

col1, col2 = st.columns(2)
with col1:
    kmeans_n_clusters = st.slider("Number of Initial Clusters", min_value=2, max_value=10, value=3)
with col2:
    kmeans_n_subclusters = st.slider("Number of Sub-clusters", min_value=2, max_value=10, value=3)

# Run segmentation
if st.button("Run Customer Segmentation"):
    st.info("Running customer segmentation...")
    
    # Override config parameters
    config.RFM_R_BINS = rfm_r_bins
    config.RFM_F_BINS = rfm_f_bins
    config.RFM_M_BINS = rfm_m_bins
    config.KMEANS_N_CLUSTERS_RUN1 = kmeans_n_clusters
    config.KMEANS_N_CLUSTERS_RUN2 = kmeans_n_subclusters
    
    try:
        # Calculate RFM
        rfm_df = customer_segmentation.calculate_rfm(st.session_state.processed_data)
        
        # Score RFM and get segments
        rfm_scored_df = customer_segmentation.score_rfm(rfm_df)
        
        # Perform K-means clustering
        segmentation_results = customer_segmentation.perform_kmeans_clustering(rfm_scored_df)
        
        # Store results in session state
        st.session_state.segmentation_results = segmentation_results
        
        # Display results
        st.success("Customer segmentation complete!")
        
        # Show segmentation results
        st.subheader("Segmentation Results")
        st.dataframe(segmentation_results.head(10))
        
        # Visualization
        st.subheader("Segment Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_counts = segmentation_results['Cluster_Name'].value_counts()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
        plt.xticks(rotation=45)
        plt.title("Customer Segment Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error during segmentation: {e}")