# config.py
"""
Configuration file for the CS Tay Customer Segmentation and Price Optimization project.
Stores file paths, constants, and other configuration parameters.
"""
import os
import logging
from pathlib import Path

# --- Project Root ---
# Assuming this config.py file is in the 'src' directory of the project root
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
# Input data
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEGMENTATION_OUTPUT_DIR = PROJECT_ROOT / "data" / "segmentation"
OPTIMIZATION_OUTPUT_DIR = PROJECT_ROOT / "data" / "optimization"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- File Names ---
# Input
# IMPORTANT: User needs to place their original sales_data.csv in the RAW_DATA_DIR
ORIGINAL_SALES_DATA_FILE = "sales_data.csv" # Original sales data CSV

# Processed
AGGREGATED_DATA_FILE = "aggregated_df.csv"
NO_CUSTOMER_NAME_AGG_DATA_FILE = "no_customer_name_agg_df.csv" # Optional

# Segmentation
CUSTOMER_SEGMENTATION_FILE = "customer_segmentation_df.csv"

# Elasticity & Optimization
PRICE_ELASTICITIES_FILE = "price_elasticities_calculated.csv"
REVENUE_OPTIMIZATION_RESULTS_FILE = "revenue_optimization_results.csv"

# --- Full File Paths ---
# Input
RAW_SALES_DATA_PATH = RAW_DATA_DIR / ORIGINAL_SALES_DATA_FILE

# Processed
AGGREGATED_DATA_PATH = PROCESSED_DATA_DIR / AGGREGATED_DATA_FILE
NO_CUSTOMER_NAME_AGG_DATA_PATH = PROCESSED_DATA_DIR / NO_CUSTOMER_NAME_AGG_DATA_FILE

# Segmentation
CUSTOMER_SEGMENTATION_OUTPUT_PATH = SEGMENTATION_OUTPUT_DIR / CUSTOMER_SEGMENTATION_FILE

# Elasticity & Optimization
PRICE_ELASTICITIES_OUTPUT_PATH = OPTIMIZATION_OUTPUT_DIR / PRICE_ELASTICITIES_FILE
REVENUE_OPTIMIZATION_OUTPUT_PATH = OPTIMIZATION_OUTPUT_DIR / REVENUE_OPTIMIZATION_RESULTS_FILE


# --- Data Loading Parameters ---
DEFAULT_ENCODING = 'ISO-8859-1' # or 'latin1', 'cp1252' as seen in notebooks

# --- Customer Segmentation Parameters ---
RFM_SNAPSHOT_DAYS_OFFSET = 1 # Days to add to max transaction date for snapshot
RFM_R_BINS = 10
RFM_F_BINS = 10
RFM_M_BINS = 10
KMEANS_RANDOM_STATE = 42
KMEANS_N_CLUSTERS_RUN1 = 3 # Based on notebook, might need adjustment
KMEANS_N_CLUSTERS_RUN2 = 3 # Based on notebook, for sub-clustering
KMEANS_CLUSTER_ID_OFFSET_RUN2 = 4 # To ensure unique cluster IDs after combining

# --- Price Elasticity Parameters ---
PRICE_ELASTICITY_CUSTOMER_CATEGORIES = ['SUPERMARKET', 'RETAIL']
PRICE_ELASTICITY_TOP_N_SKUS = 15

# --- Revenue Optimization Parameters ---
# Bounds for price change percentages (e.g., -50% to +50%)
OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND = -0.50
OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND = 2.00

# Columns for RFM calculation
COL_TRANSACTION_DATE = 'Transaction Date'
COL_SALES_ORDER_NO = 'Sales Order No.'
COL_TOTAL_BASE_AMT = 'Total Base Amt'
COL_QTY = 'Qty'
COL_INVENTORY_CODE = 'Inventory Code'
COL_INVENTORY_DESC = 'Inventory Desc'
COL_CUSTOMER_CODE = 'Customer Code'
COL_CUSTOMER_NAME = 'Customer Name'
COL_CUSTOMER_CATEGORY_DESC = 'Customer Category Desc'
COL_PRICE_PER_QTY = 'Price per qty'
COL_ITEM_CATEGORY = 'Item Category'

# --- Logging Configuration (Example) ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if __name__ == '__main__':
    # This part allows you to run config.py to check paths (optional)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Sales Data Path: {RAW_SALES_DATA_PATH}")
    print(f"Aggregated Data Path: {AGGREGATED_DATA_PATH}")
    print(f"Customer Segmentation Output Path: {CUSTOMER_SEGMENTATION_OUTPUT_PATH}")
    print(f"Price Elasticities Output Path: {PRICE_ELASTICITIES_OUTPUT_PATH}")
    print(f"Revenue Optimization Output Path: {REVENUE_OPTIMIZATION_OUTPUT_PATH}")    # Verify that the raw data file exists
    if RAW_SALES_DATA_PATH.exists():
        print(f"Raw sales data file found at: {RAW_SALES_DATA_PATH}")
    else:
        print(f"⚠️ Raw sales data file NOT FOUND at: {RAW_SALES_DATA_PATH}")
        print(f"Please ensure '{ORIGINAL_SALES_DATA_FILE}' is placed in the 'data/raw/' directory.")
