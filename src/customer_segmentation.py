# customer_segmentation.py
"""
Performs customer segmentation using RFM analysis and K-Means clustering.
Corresponds to the logic in 'customer_segmentation_combined_original.md'.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt # For visualizations if run directly
import seaborn as sns # For visualizations if run directly
import squarify # For treemap visualization

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
# from yellowbrick.cluster import KElbowVisualizer # Not strictly needed for script, but was in notebook
from sklearn.metrics import silhouette_score

# Import project-specific modules
import config
import utils

# Configure logger
import logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, Monetary value, and Total Quantity for each customer.
    """
    logger.info("Calculating RFM metrics...")
    # Ensure Transaction Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[config.COL_TRANSACTION_DATE]):
        df[config.COL_TRANSACTION_DATE] = pd.to_datetime(df[config.COL_TRANSACTION_DATE], errors='coerce')

    # Monetary: Sum of Total Base Amt for each customer
    monetary_df = df.groupby(config.COL_CUSTOMER_ID, observed=False)[config.COL_TOTAL_BASE_AMT].sum().reset_index()
    monetary_df.columns = [config.COL_CUSTOMER_ID, 'Monetary']

    # Frequency: Count of unique Sales Order No. for each customer
    frequency_df = df.groupby(config.COL_CUSTOMER_ID, observed=False)[config.COL_SALES_ORDER_NO].nunique().reset_index()
    frequency_df.columns = [config.COL_CUSTOMER_ID, 'Frequency']

    # Recency: Days since last transaction
    snapshot_date = df[config.COL_TRANSACTION_DATE].max() + timedelta(days=config.RFM_SNAPSHOT_DAYS_OFFSET)
    logger.info(f"Snapshot date for Recency calculation: {snapshot_date}")
    
    recency_df = df.groupby(config.COL_CUSTOMER_ID, observed=False)[config.COL_TRANSACTION_DATE].max().reset_index()
    recency_df['Recency'] = (snapshot_date - recency_df[config.COL_TRANSACTION_DATE]).dt.days
    recency_df = recency_df[[config.COL_CUSTOMER_ID, 'Recency']]

    # Total Quantity: Sum of Qty for each customer
    quantity_df = df.groupby(config.COL_CUSTOMER_ID, observed=False)[config.COL_QTY].sum().reset_index()
    quantity_df.columns = [config.COL_CUSTOMER_ID, 'Total Quantity']
    
    # Merge RFM metrics
    rfm_df = recency_df.merge(frequency_df, on=config.COL_CUSTOMER_ID, how='inner')
    rfm_df = rfm_df.merge(monetary_df, on=config.COL_CUSTOMER_ID, how='inner')
    rfm_df = rfm_df.merge(quantity_df, on=config.COL_CUSTOMER_ID, how='inner')
    
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
        score = row['RFM_Score_Sum']
        if score >= 27: return 'Champions'
        elif score >= 21: return 'Loyal_Customers'
        elif score >= 18: return 'Potential_Loyalists'
        elif score >= 15: return 'New_Customers' # Initial assignment
        elif score >= 12: return 'At_Risk' # Initial assignment
        elif score >= 6: return 'Hibernating' # Initial assignment
        else: return 'Lost'

    rfm_scored_df['RFM_Segment_Initial'] = rfm_scored_df.apply(assign_rfm_segment, axis=1)

    # Adjust segments (as per notebook logic section 4.12)
    def adjust_segment(row):
        if row['R_Score'] >= 8 and row['F_Score'] <= 3 and row['M_Score'] <= 3:
            return 'New_Customers'
        # The notebook logic for 'At Risk' or 'Hibernating' with R_Score >= 8 seems to re-label them as New_Customers
        elif row['R_Score'] >= 8 and row['RFM_Segment_Initial'] in ['At_Risk', 'Hibernating']:
             return 'New_Customers'
        return row['RFM_Segment_Initial']

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
    k_rfm_df = rfm_df[[config.COL_CUSTOMER_ID, 'Recency', 'Frequency', 'Monetary', 'Total Quantity']].copy()
    features_for_clustering = ['Recency', 'Frequency', 'Monetary', 'Total Quantity']
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(k_rfm_df[features_for_clustering])
    # scaled_df = pd.DataFrame(scaled_data, columns=features_for_clustering, index=k_rfm_df.index)

    # --- K-Means Run 1 ---
    logger.info("K-Means Run 1: Clustering all customers.")
    kmeans_run1 = KMeans(n_clusters=config.KMEANS_N_CLUSTERS_RUN1, 
                         random_state=config.KMEANS_RANDOM_STATE,
                         n_init='auto') # Suppress future warning
    k_rfm_df['KMeans_Segment_Run1'] = kmeans_run1.fit_predict(scaled_data)
    
    # Identify dominant cluster to sub-cluster (e.g., cluster 0 from notebook, which became segment 1)
    # The notebook logic seems to isolate the largest, less distinct cluster for further segmentation.
    # Here, we assume cluster 0 (label 0) is the one to be sub-clustered.
    # The notebook assigned cluster labels then +1. So cluster 0 became segment 1.
    # Then it isolated segment 1 for further clustering.
    
    # In the notebook, cluster labels were 0, 1, 2. Segment 1 (original label 0) was sub-clustered.
    # Segments 2 and 3 (original labels 1 and 2) were kept aside.
    
    # Find the cluster label that corresponds to the largest group, which was '1' after adding 1.
    # So, original label was 0.
    # The notebook logic was:
    # kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_data)
    # clusters = kmeans.labels_
    # k_rfm["KMeans_Segment"] = clusters
    # k_rfm["KMeans_Segment"] = k_rfm["KMeans_Segment"] + 1
    # cluster_2_and_3 = k_rfm[k_rfm['KMeans_Segment'].isin([2, 3])]
    # cluster_1 = k_rfm[~k_rfm['KMeans_Segment'].isin([2, 3])] -> This is the one to re-cluster
    
    # Replicate notebook logic for identifying sub-cluster target
    temp_segment_col = 'KMeans_Segment_Temp'
    k_rfm_df[temp_segment_col] = k_rfm_df['KMeans_Segment_Run1'] + 1 # Segments 1, 2, 3

    df_to_recluster = k_rfm_df[~k_rfm_df[temp_segment_col].isin([2,3])].copy() # This is segment 1 (original label 0)
    df_kept_aside = k_rfm_df[k_rfm_df[temp_segment_col].isin([2,3])].copy()
    
    k_rfm_df.drop(columns=[temp_segment_col, 'KMeans_Segment_Run1'], inplace=True) # Clean up

    if df_to_recluster.empty:
        logger.warning("No data to re-cluster after K-Means Run 1. Using Run 1 results directly.")
        k_rfm_df['KMeans_Segment'] = kmeans_run1.labels_
    else:
        logger.info(f"K-Means Run 2: Sub-clustering {len(df_to_recluster)} customers.")
        scaled_data_run2 = scaler.fit_transform(df_to_recluster[features_for_clustering])
        
        kmeans_run2 = KMeans(n_clusters=config.KMEANS_N_CLUSTERS_RUN2, 
                             random_state=config.KMEANS_RANDOM_STATE,
                             n_init='auto')
        df_to_recluster['KMeans_Segment'] = kmeans_run2.fit_predict(scaled_data_run2)
        # Offset these new cluster labels to avoid collision with df_kept_aside
        df_to_recluster['KMeans_Segment'] += config.KMEANS_CLUSTER_ID_OFFSET_RUN2 
        
        # Assign original cluster labels to df_kept_aside
        # df_kept_aside used KMeans_Segment_Run1 labels directly (0, 1, 2).
        # The notebook then remapped these. Let's use the original labels from run1 for those kept aside.
        df_kept_aside['KMeans_Segment'] = df_kept_aside['KMeans_Segment_Run1'] # Original labels 0,1,2

        # Combine
        final_clustered_df = pd.concat([df_to_recluster, df_kept_aside], axis=0)
        k_rfm_df = final_clustered_df.copy()


    # Remap all KMeans_Segment labels to be sequential starting from 1
    unique_kmeans_segments = sorted(k_rfm_df['KMeans_Segment'].unique())
    mapping = {old_val: new_val for new_val, old_val in enumerate(unique_kmeans_segments, start=1)}
    k_rfm_df['KMeans_Segment'] = k_rfm_df['KMeans_Segment'].map(mapping)
    
    logger.info("K-Means clustering complete.")
    return k_rfm_df[[config.COL_CUSTOMER_ID, 'KMeans_Segment']]


def run_segmentation():
    """
    Main function to run the customer segmentation pipeline.
    """
    logger.info("Starting customer segmentation...")

    # 1. Load Processed Data
    aggregated_df = utils.load_csv_data(config.AGGREGATED_DATA_PATH)
    if aggregated_df is None:
        logger.error("Failed to load aggregated data. Exiting segmentation.")
        return

    # 2. Calculate RFM metrics
    rfm_calculated_df = calculate_rfm(aggregated_df)

    # 3. Score RFM and assign initial segments
    rfm_scored_segmented_df = score_rfm(rfm_calculated_df)
    
    # 4. Perform K-Means Clustering
    # The K-Means in the notebook uses the base RFM values (Recency, Frequency, Monetary, Total Quantity)
    # not the R, F, M scores.
    kmeans_segments_df = perform_kmeans_clustering(rfm_calculated_df) # Pass the df before R,F,M scores

    # 5. Merge RFM Segments with K-Means Segments
    # Merge on Customer Code
    final_segmentation_df = pd.merge(rfm_scored_segmented_df, kmeans_segments_df, on=config.COL_CUSTOMER_ID, how='left')
    logger.info(f"RFM and K-Means segments merged. Shape: {final_segmentation_df.shape}")

    # 6. Merge with Customer Category Desc and Item Category for context
    # Need to get unique Customer Code to Category/Item mapping from aggregated_df
    cust_item_cat_df = aggregated_df[[
        config.COL_CUSTOMER_ID, 
        config.COL_CUSTOMER_CATEGORY_DESC, 
        config.COL_ITEM_CATEGORY
    ]].drop_duplicates(subset=[config.COL_CUSTOMER_ID], keep='first').copy()
    
    final_segmentation_df = pd.merge(final_segmentation_df, cust_item_cat_df, on=config.COL_CUSTOMER_ID, how='left')
    
    # Drop RFM_Concat if it exists (it was in the notebook but not essential for final output)
    if 'RFM_Concat' in final_segmentation_df.columns:
        final_segmentation_df = final_segmentation_df.drop(columns=['RFM_Concat'])
        
    # Select and reorder columns to match notebook output `cust_broad` approx.
    output_columns = [
        config.COL_CUSTOMER_ID, 'Recency', 'Frequency', 'Monetary', 'Total Quantity',
        'R_Score', 'F_Score', 'M_Score', 'RFM_Score_Sum', 'RFM_Segment', 'KMeans_Segment',
        config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY
    ]
    # Filter for columns that actually exist to prevent KeyErrors
    output_columns = [col for col in output_columns if col in final_segmentation_df.columns]
    final_segmentation_df = final_segmentation_df[output_columns]

    # 7. Save Segmentation Data
    if utils.save_df_to_csv(final_segmentation_df, config.CUSTOMER_SEGMENTATION_OUTPUT_PATH):
        logger.info(f"Successfully saved customer segmentation data to {config.CUSTOMER_SEGMENTATION_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save customer segmentation data.")
        
    logger.info("Customer segmentation finished.")
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

            # Optional: Display treemap (requires matplotlib and squarify)
            # if 'RFM_Segment' in segmentation_results_df.columns:
            #     plt.figure(figsize=(12, 8))
            #     segment_counts = segmentation_results_df['RFM_Segment'].value_counts()
            #     squarify.plot(sizes=segment_counts.values, label=segment_counts.index, alpha=0.7,
            #                   text_kwargs={'fontsize':10})
            #     plt.axis('off')
            #     plt.title('Customer Segments Treemap (RFM)')
            #     # plt.show() # Use plt.savefig() if running in a non-interactive script
            #     treemap_path = config.SEGMENTATION_OUTPUT_DIR / "rfm_segment_treemap.png"
            #     plt.savefig(treemap_path)
            #     logger.info(f"Saved RFM segment treemap to {treemap_path}")
            #     plt.close()

            # if 'KMeans_Segment' in segmentation_results_df.columns:
            #     plt.figure(figsize=(10, 7))
            #     kmeans_counts = segmentation_results_df['KMeans_Segment'].value_counts().sort_index()
            #     sns.barplot(x=kmeans_counts.index, y=kmeans_counts.values)
            #     plt.title('K-Means Segment Distribution')
            #     plt.xlabel('K-Means Segment')
            #     plt.ylabel('Number of Customers')
            #     # plt.show()
            #     kmeans_dist_path = config.SEGMENTATION_OUTPUT_DIR / "kmeans_segment_distribution.png"
            #     plt.savefig(kmeans_dist_path)
            #     logger.info(f"Saved K-Means segment distribution to {kmeans_dist_path}")
            #     plt.close()