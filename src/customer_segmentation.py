# customer_segmentation.py
"""
Performs customer segmentation using RFM analysis and K-Means clustering.
Corresponds to the logic in 'customer_segmentation_combined_original.md'.
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
import matplotlib.pyplot as plt # For visualizations if run directly
import seaborn as sns # For visualizations if run directly

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import project-specific modules
from . import config
from . import utils

# Configure logger
import logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, Monetary value, and Total Quantity for each customer.
    For anonymized data, we use Customer Code as the customer identifier.
    """
    logger.info("Calculating RFM metrics...")
    # Ensure Transaction Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[config.COL_TRANSACTION_DATE]):
        df[config.COL_TRANSACTION_DATE] = pd.to_datetime(df[config.COL_TRANSACTION_DATE], errors='coerce')

    # Monetary: Sum of Total Base Amt for each customer
    monetary_df = df.groupby(config.COL_CUSTOMER_CODE, observed=False)[config.COL_TOTAL_BASE_AMT].sum().reset_index()
    monetary_df.columns = [config.COL_CUSTOMER_CODE, 'Monetary']

    # Frequency: Count of unique Sales Order No. for each customer
    frequency_df = df.groupby(config.COL_CUSTOMER_CODE, observed=False)[config.COL_SALES_ORDER_NO].nunique().reset_index()
    frequency_df.columns = [config.COL_CUSTOMER_CODE, 'Frequency']

    # Recency: Days since last transaction
    snapshot_date = df[config.COL_TRANSACTION_DATE].max() + timedelta(days=config.RFM_SNAPSHOT_DAYS_OFFSET)
    logger.info(f"Snapshot date for Recency calculation: {snapshot_date}")
    
    recency_df = df.groupby(config.COL_CUSTOMER_CODE, observed=False)[config.COL_TRANSACTION_DATE].max().reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df[config.COL_TRANSACTION_DATE]).dt.days
    recency_df = recency_df[[config.COL_CUSTOMER_CODE, 'Recency']]

    # Total Quantity: Sum of Qty for each customer
    quantity_df = df.groupby(config.COL_CUSTOMER_CODE, observed=False)[config.COL_QTY].sum().reset_index()
    quantity_df.columns = [config.COL_CUSTOMER_CODE, 'Total Quantity']
    
    # Merge RFM metrics
    rfm_df = recency_df.merge(frequency_df, on=config.COL_CUSTOMER_CODE, how='inner')
    rfm_df = rfm_df.merge(monetary_df, on=config.COL_CUSTOMER_CODE, how='inner')
    rfm_df = rfm_df.merge(quantity_df, on=config.COL_CUSTOMER_CODE, how='inner')
    
    logger.info(f"RFM calculation complete. RFM DataFrame shape: {rfm_df.shape}")
    return rfm_df

def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns R, F, M scores (1-10 deciles) and a combined RFM segment.
    """
    logger.info("Scoring RFM values...")
    rfm_scored_df = rfm_df.copy()

    # Define labels for scoring (R is reversed: higher score for lower recency days)
    r_labels = range(config.RFM_R_BINS, 0, -1) 
    f_labels = range(1, config.RFM_F_BINS + 1)
    m_labels = range(1, config.RFM_M_BINS + 1)

    # Handle potential issues with pd.cut if all values are the same or too few unique values for bins
    try:
        rfm_scored_df['R_Score'] = pd.cut(rfm_scored_df['Recency'], bins=config.RFM_R_BINS, labels=r_labels, duplicates='drop').astype(int)
    except ValueError as e:
        logger.warning(f"Could not create {config.RFM_R_BINS} bins for Recency due to: {e}. Assigning default score 1.")
        rfm_scored_df['R_Score'] = 1
        
    try:
        rfm_scored_df['F_Score'] = pd.cut(rfm_scored_df['Frequency'], bins=config.RFM_F_BINS, labels=f_labels, duplicates='drop').astype(int)
    except ValueError as e:
        logger.warning(f"Could not create {config.RFM_F_BINS} bins for Frequency due to: {e}. Assigning default score 1.")
        rfm_scored_df['F_Score'] = 1
        
    try:
        rfm_scored_df['M_Score'] = pd.cut(rfm_scored_df['Monetary'], bins=config.RFM_M_BINS, labels=m_labels, duplicates='drop').astype(int)
    except ValueError as e:
        logger.warning(f"Could not create {config.RFM_M_BINS} bins for Monetary due to: {e}. Assigning default score 1.")
        rfm_scored_df['M_Score'] = 1
        
    rfm_scored_df['RFM_Score_Sum'] = rfm_scored_df['R_Score'] + rfm_scored_df['F_Score'] + rfm_scored_df['M_Score']
    
    # Define RFM segments based on score sum (as per notebook logic)
    def assign_rfm_segment(row):
        score_sum = row['RFM_Score_Sum']
        if score_sum >= 24:
            return 'Champions'
        elif score_sum >= 18:
            return 'Loyal Customers'
        elif score_sum >= 12:
            return 'Potential Loyalists'
        elif score_sum >= 9:
            return 'At Risk Customers'
        elif score_sum >= 6:
            return 'Hibernating'
        else:
            return 'Lost Customers'

    rfm_scored_df['RFM_Segment_Initial'] = rfm_scored_df.apply(assign_rfm_segment, axis=1)

    # Adjust segments (as per notebook logic section 4.12)
    def adjust_segment(row):
        segment = row['RFM_Segment_Initial']
        # Specific rules to refine segments based on individual R, F, and M scores
        if row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'Lost Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] > 2 and row['F_Score'] <= 4:
            return 'Hibernating'
        elif row['R_Score'] >= 4 and row['F_Score'] <= 2 and row['M_Score'] <= 3:
            return 'New Customers'
        elif row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 8:
            return 'Champions'
        return segment

    rfm_scored_df['RFM_Segment'] = rfm_scored_df.apply(adjust_segment, axis=1)
    rfm_scored_df.drop(columns=['RFM_Segment_Initial'], inplace=True) # Drop intermediate column
    
    logger.info("RFM scoring and initial segmentation complete.")
    return rfm_scored_df

def perform_kmeans_clustering(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs K-Means clustering on RFM features, including a two-step process
    if a dominant cluster is found.
    """
    logger.info("Performing K-Means clustering...")
    k_rfm_df = rfm_df[[config.COL_CUSTOMER_CODE, 'Recency', 'Frequency', 'Monetary', 'Total Quantity']].copy()
    features_for_clustering = ['Recency', 'Frequency', 'Monetary', 'Total Quantity']

    # Scale the data - using RobustScaler to handle outliers
    scaler = RobustScaler()
    k_rfm_scaled = scaler.fit_transform(k_rfm_df[features_for_clustering])
    k_rfm_scaled_df = pd.DataFrame(k_rfm_scaled, columns=features_for_clustering)
    
    # First K-Means run
    logger.info(f"Running K-Means with {config.KMEANS_N_CLUSTERS_RUN1} clusters...")
    kmeans_run1 = KMeans(n_clusters=config.KMEANS_N_CLUSTERS_RUN1, random_state=config.KMEANS_RANDOM_STATE)
    k_rfm_df['Cluster_ID'] = kmeans_run1.fit_predict(k_rfm_scaled)
    
    # Log silhouette score for first clustering
    sil_score_run1 = silhouette_score(k_rfm_scaled, k_rfm_df['Cluster_ID'])
    logger.info(f"First K-Means clustering silhouette score: {sil_score_run1:.4f}")
    
    # Check for dominant cluster
    cluster_counts = k_rfm_df['Cluster_ID'].value_counts()
    largest_cluster = cluster_counts.idxmax()
    largest_cluster_size = cluster_counts[largest_cluster]
    largest_cluster_pct = largest_cluster_size / k_rfm_df.shape[0] * 100
    
    logger.info(f"Largest cluster is {largest_cluster} with {largest_cluster_size} samples ({largest_cluster_pct:.2f}%)")
    
    # If the largest cluster contains more than 40% of the data, perform a second clustering on just that cluster
    if largest_cluster_pct > 40:
        logger.info(f"Dominant cluster found ({largest_cluster_pct:.2f}% of data). Performing second clustering...")
        
        # Filter for samples in the largest cluster
        dominant_cluster_mask = k_rfm_df['Cluster_ID'] == largest_cluster
        largest_cluster_idx = k_rfm_df.loc[dominant_cluster_mask].index
        largest_cluster_data = k_rfm_scaled[dominant_cluster_mask]
        
        # Second K-Means run on just the dominant cluster
        kmeans_run2 = KMeans(n_clusters=config.KMEANS_N_CLUSTERS_RUN2, random_state=config.KMEANS_RANDOM_STATE)
        sub_clusters = kmeans_run2.fit_predict(largest_cluster_data)
        # Add the sub-cluster ID to the original dataframe with an offset to avoid overlap with original IDs
        # e.g. if original clusters are 0,1,2 and largest is 0, new clusters would be 0+3=3, 1+3=4, 2+3=5
        sub_cluster_with_offset = sub_clusters + config.KMEANS_CLUSTER_ID_OFFSET_RUN2
        k_rfm_df.loc[largest_cluster_idx, 'Cluster_ID'] = sub_cluster_with_offset
        
        # Log silhouette score for second clustering
        if len(np.unique(sub_clusters)) > 1:  # Need at least 2 clusters for silhouette score
            sil_score_run2 = silhouette_score(largest_cluster_data, sub_clusters)
            logger.info(f"Second K-Means clustering silhouette score: {sil_score_run2:.4f}")
    
    # Replace the custom names with generic numbered labels
    unique_clusters = sorted(k_rfm_df['Cluster_ID'].unique())
    cluster_names = {}
    for i, cluster_id in enumerate(unique_clusters):
        cluster_names[cluster_id] = f"Segment {i+1}"
    
    # If we have more clusters from the second run, add more names
    for i in range(3, len(unique_clusters)):
        cluster_names[unique_clusters[i]] = f"Sub-Segment {i-2}"
    k_rfm_df['Cluster_Name'] = k_rfm_df['Cluster_ID'].map(cluster_names)
    logger.info("K-Means clustering complete.")
    
    # Merge clustering results back with the RFM scores and segments
    result_df = rfm_df.merge(k_rfm_df[[config.COL_CUSTOMER_CODE, 'Cluster_ID', 'Cluster_Name']], 
                           on=config.COL_CUSTOMER_CODE, how='inner')
    
    return result_df

def run_segmentation():
    """
    Main function to run the customer segmentation pipeline.
    """
    logger.info("Starting customer segmentation...")
    
    # 1. Load aggregated data
    aggregated_df = utils.load_csv_data(config.AGGREGATED_DATA_PATH)
    if aggregated_df is None:
        logger.error("Failed to load aggregated data. Segmentation halted.")
        return None
    
    logger.info(f"Loaded aggregated data. Shape: {aggregated_df.shape}")
    
    # 2. Calculate RFM metrics
    logger.info("Calculating RFM metrics...")
    rfm_df = calculate_rfm(aggregated_df)
    
    # 3. Score RFM and get initial segments
    logger.info("Scoring RFM and assigning segments...")
    rfm_scored_df = score_rfm(rfm_df)
    
    # 4. Perform K-Means clustering for advanced segmentation
    logger.info("Performing K-Means clustering for advanced segmentation...")
    final_segmentation_df = perform_kmeans_clustering(rfm_scored_df)
    
    # 5. Save segmentation results
    if utils.save_df_to_csv(final_segmentation_df, config.CUSTOMER_SEGMENTATION_OUTPUT_PATH):
        logger.info(f"Successfully saved customer segmentation results to {config.CUSTOMER_SEGMENTATION_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save customer segmentation results to {config.CUSTOMER_SEGMENTATION_OUTPUT_PATH}")
    
    logger.info("Customer segmentation completed successfully.")
    return final_segmentation_df

if __name__ == '__main__':
    # This allows the script to be run directly.
    if not config.AGGREGATED_DATA_PATH.exists():
        print(f"ERROR: Aggregated data file not found at {config.AGGREGATED_DATA_PATH}")
        print("Please run data_preprocessing.py first.")
    else:
        segmentation_results_df = run_segmentation()
        if segmentation_results_df is not None:
            print("\nCustomer segmentation complete. Sample of results:")
            print(segmentation_results_df.head())
            print(f"\nSegmentation data saved to {config.SEGMENTATION_OUTPUT_DIR}")

            # Visualization for cluster distribution
            if 'Cluster_ID' in segmentation_results_df.columns:  # Changed from 'KMeans_Segment' to 'Cluster_ID'
                plt.figure(figsize=(10, 7))
                kmeans_counts = segmentation_results_df['Cluster_ID'].value_counts().sort_index()
                sns.barplot(x=kmeans_counts.index, y=kmeans_counts.values)
                plt.title('K-Means Cluster Distribution')
                plt.xlabel('Cluster ID')
                plt.ylabel('Number of Customers')
                # plt.show()  # Uncomment if you want to display the plot interactively
                kmeans_dist_path = config.SEGMENTATION_OUTPUT_DIR / "kmeans_segment_distribution.png"
                plt.savefig(kmeans_dist_path)
                logger.info(f"Saved K-Means segment distribution to {kmeans_dist_path}")
                plt.close()