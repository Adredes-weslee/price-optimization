# data_preprocessing.py
"""
Handles loading the raw sales data, cleaning it, performing necessary transformations,
and feature engineering.
Corresponds to the logic in 'data_preprocessing_original.md'.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import project-specific modules
import config
import utils

# Configure logger
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def reconcile_customer_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconciles inconsistencies between Customer Code and Customer Name.
    Uses a predefined mapping for known inconsistencies.
    """
    # Mapping from data_preprocessing_original.md (section 4.1.1)
    consistent_names_map = {
        'A013': 'AISHAH - TANJONG KATONG PRI SCHOOL / EUNOS PRI SCHOOL',
        'A030': 'AQUAMARINA HOTEL PTE LTD',
        'B009': 'BISTRO ONE THIRTYSIX (WHAMPOA) PTE LTD', # Note: Original notebook had one entry as BISTRO ONE THIRTYSIX PTE LTD
        'C024': 'PLATE & PALETTE - CARRARA CAFE', # Combined from CARRARA CAFE and PLATE & PALETTE
        'C039': 'CHRISTINE - SOTA / SPRINGDALE PRI SCHOOL',
        'E016': 'EIGHT PLUS TWO PTE LTD C/O JOE & DOUGH', # Kept the longer name
        'E017': 'E & W ENTERPRISE PTE LTD C/O JOE & DOUGH', # Kept the longer name
        'H001': 'HAIPEBRO PTE LTD C/O JOE & DOUGH', # Kept the longer name
        'J024': 'JEKYLL & HYDE PTE LTD', # Chose one, original also had 1011 PTE LTD
        'L004': 'LOH S+B PTE LTD C/O JOE & DOUGH', # Kept the longer name
        'L006': 'LOH BROTHERS COFFEE PTE LTD C/O JOE & DOUGH', # Kept the longer name
        'L011': 'LIM FOOD F&B PTE LTD', # Chose one, original also had L.B.FOOD
        'L019': 'LAO SI FROZEN GOOD', # Standardized
        'M040': 'ZAMEEL ENNYAH - MAS', # Combined
        'Q002': 'QUESADILLA - REPUBLIC POLY', # Combined
        'R015': 'RNY CAPITAL PTE LTD C/O JOE & DOUGH' # Kept the longer name
    }
    
    # Apply mapping: if Customer Code is in map, use mapped name, otherwise keep original
    df[config.COL_CUSTOMER_NAME] = df[config.COL_CUSTOMER_CODE].map(consistent_names_map).fillna(df[config.COL_CUSTOMER_NAME])
    logger.info("Reconciled Customer Name inconsistencies.")
    return df

def reconcile_inventory_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconciles inconsistencies between Inventory Code and Inventory Description.
    Uses a predefined mapping for known inconsistencies.
    """
    # Mapping from data_preprocessing_original.md (section 4.2.1)
    consistent_descriptions_map = {
        1155.0: '(99) S/L CHIX BREAST FILLET 1.15KG', # Standardized
        1602.0: 'BREADED POLLOCK FILLET (5PKT)' # Standardized
    }
    df[config.COL_INVENTORY_DESC] = df[config.COL_INVENTORY_CODE].map(consistent_descriptions_map).fillna(df[config.COL_INVENTORY_DESC])
    logger.info("Reconciled Inventory Description inconsistencies.")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in specified columns.
    """
    # Handle missing Customer Category Desc (section 4.4.1)
    # Specific case for 'FAIRY WONDERLAND FENG SHAN PRI SCH'
    df.loc[df[config.COL_CUSTOMER_NAME] == 'FAIRY WONDERLAND FENG SHAN PRI SCH', config.COL_CUSTOMER_CATEGORY_DESC] = 'SCHOOL'
    logger.info("Handled missing Customer Category Desc for specific customer.")

    # Handle missing Inventory Desc for specific Inventory Codes (section 4.4.2)
    # Codes 1146.0 and 1153.0 where Inventory Desc is NaN
    condition_specific_inv_desc_nan = df[config.COL_INVENTORY_CODE].isin([1146.0, 1153.0]) & df[config.COL_INVENTORY_DESC].isna()
    df.loc[condition_specific_inv_desc_nan, config.COL_INVENTORY_DESC] = 'CONTAINER / STOCKS'
    logger.info("Filled missing Inventory Desc for codes 1146.0, 1153.0.")
    
    # Handle rows where both Inventory Code and Inventory Desc might be NaN, or only one of them
    # Fill Inventory Desc first if it's NaN
    df.loc[df[config.COL_INVENTORY_DESC].isna(), config.COL_INVENTORY_DESC] = 'CONTAINER / STOCKS'
    
    # For rows where Inventory Code is NaN, generate new codes based on 'Price per qty'
    # First, calculate 'Price per qty' if it doesn't exist or has NaNs due to Qty=0
    # Ensure Qty is not zero to avoid division by zero; fillna for Total Base Amt and Qty if necessary
    df[config.COL_QTY] = df[config.COL_QTY].replace(0, np.nan) # Avoid division by zero
    df[config.COL_PRICE_PER_QTY] = (df[config.COL_TOTAL_BASE_AMT] / df[config.COL_QTY]).round(2)
    df[config.COL_PRICE_PER_QTY].fillna(0, inplace=True) # Fill NaN prices with 0 or another placeholder

    rows_with_nan_inv_code = df[config.COL_INVENTORY_CODE].isna()]
    if not rows_with_nan_inv_code.empty:
        unique_prices_for_nan_inv = rows_with_nan_inv_code[config.COL_PRICE_PER_QTY].unique()
        # Start new codes from 9000, ensure they don't clash with existing codes
        existing_max_code = df[config.COL_INVENTORY_CODE].max()
        start_code = 9000
        if pd.notna(existing_max_code) and existing_max_code >= start_code:
            start_code = int(existing_max_code // 1000 + 1) * 1000 # Ensure new codes are in a new range

        new_inventory_codes_map = {price: start_code + i for i, price in enumerate(unique_prices_for_nan_inv)}
        
        df.loc[df[config.COL_INVENTORY_CODE].isna(), config.COL_INVENTORY_CODE] = \
            df[config.COL_PRICE_PER_QTY].map(new_inventory_codes_map)
        logger.info(f"Generated and filled missing Inventory Codes. New codes start from {start_code}.")

    # Final check for any remaining NaNs in these critical columns
    if df[[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_INVENTORY_CODE, config.COL_INVENTORY_DESC]].isna().any().any():
        logger.warning("There are still NaNs in Customer Category Desc, Inventory Code, or Inventory Desc after handling.")
    else:
        logger.info("Missing values handled for critical descriptive columns.")
        
    return df

def aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction data by summing Qty and Total Base Amt for unique
    combinations of transaction identifiers and item details.
    """
    # Define grouping keys. Price per qty is included to keep distinct items if price differs.
    # Customer Name, Customer Category Desc, Inventory Desc are kept as they should be consistent per code after reconciliation.
    grouping_keys = [
        config.COL_TRANSACTION_DATE,
        config.COL_SALES_ORDER_NO,
        config.COL_CUSTOMER_CODE,
        config.COL_INVENTORY_CODE,
        config.COL_CUSTOMER_NAME, # Assuming reconciled
        config.COL_CUSTOMER_CATEGORY_DESC, # Assuming reconciled/filled
        config.COL_INVENTORY_DESC, # Assuming reconciled/filled
        config.COL_PRICE_PER_QTY # Calculated and rounded
    ]
    
    # Ensure all grouping keys exist
    for key in grouping_keys:
        if key not in df.columns:
            raise KeyError(f"Required grouping key '{key}' not found in DataFrame columns: {df.columns.tolist()}")

    aggregated_df = df.groupby(grouping_keys, observed=False, dropna=False).agg(
        Qty_Sum=(config.COL_QTY, 'sum'),
        Total_Base_Amt_Sum=(config.COL_TOTAL_BASE_AMT, 'sum')
    ).reset_index()

    # Rename summed columns back to original names
    aggregated_df.rename(columns={'Qty_Sum': config.COL_QTY, 
                                  'Total_Base_Amt_Sum': config.COL_TOTAL_BASE_AMT}, inplace=True)
    logger.info(f"Aggregated transactions. Shape before: {df.shape}, Shape after: {aggregated_df.shape}")
    return aggregated_df

def perform_type_casting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns to their appropriate data types.
    """
    try:
        df[config.COL_TRANSACTION_DATE] = pd.to_datetime(df[config.COL_TRANSACTION_DATE], format='mixed', dayfirst=True)
        logger.info(f"Converted '{config.COL_TRANSACTION_DATE}' to datetime.")
    except Exception as e:
        logger.error(f"Error converting '{config.COL_TRANSACTION_DATE}' to datetime: {e}. Trying without dayfirst=True.")
        try:
            df[config.COL_TRANSACTION_DATE] = pd.to_datetime(df[config.COL_TRANSACTION_DATE], format='mixed')
            logger.info(f"Successfully converted '{config.COL_TRANSACTION_DATE}' to datetime (without dayfirst=True).")
        except Exception as e2:
            logger.error(f"Second attempt to convert '{config.COL_TRANSACTION_DATE}' failed: {e2}")


    categorical_cols = [
        config.COL_CUSTOMER_CODE, config.COL_INVENTORY_CODE,
        config.COL_CUSTOMER_NAME, config.COL_CUSTOMER_CATEGORY_DESC,
        config.COL_INVENTORY_DESC
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            logger.info(f"Converted '{col}' to category dtype.")
        else:
            logger.warning(f"Column '{col}' not found for type casting.")
            
    # Ensure Qty is float if it contains non-integers, otherwise int
    if config.COL_QTY in df.columns:
        if (df[config.COL_QTY] % 1 == 0).all(): # Check if all are whole numbers
             df[config.COL_QTY] = df[config.COL_QTY].astype(int)
             logger.info(f"Converted '{config.COL_QTY}' to int.")
        else:
             df[config.COL_QTY] = df[config.COL_QTY].astype(float) # Keep as float if decimals exist
             logger.info(f"'{config.COL_QTY}' kept as float due to decimal values.")

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features like 'Customer Category Broad' and 'Item Category'.
    """
    # Create 'Customer Category Broad' (section 4.6.6)
    def categorize_customer_broad(desc):
        if pd.isna(desc): return 'Others'
        if desc in ['SUPERMARKET', 'MINI MART', 'RETAIL', 'WHOLESALER']:
            return 'Retail'
        elif desc in ['CAFE', 'RESTAURANT', 'FOOD MANUFACTURER', 'HAWKER', 'CATERER', 
                      'ABANG (FATTY BOM BOM)', 'CLOUD KITCHEN (TIFFLAB)', 'FAST FOOD', 
                      'CLUB/ENTERTAINMENT', 'HOTEL', 'AIRLINE']:
            return 'Food Services'
        elif desc == 'SCHOOL':
            return 'Institutional'
        elif desc == 'WET MARKET':
            return 'Market'
        else:
            return 'Others'
    
    if config.COL_CUSTOMER_CATEGORY_DESC in df.columns:
        df['Customer Category Broad'] = df[config.COL_CUSTOMER_CATEGORY_DESC].apply(categorize_customer_broad).astype('category')
        logger.info("Created 'Customer Category Broad' feature.")
    else:
        logger.warning(f"'{config.COL_CUSTOMER_CATEGORY_DESC}' not found, cannot create 'Customer Category Broad'.")

    # Create 'Item Category' (section 4.6.7)
    def categorize_item(desc):
        if pd.isna(desc): return "RTC" # Default if NaN
        desc_str = str(desc) # Ensure it's a string
        if desc_str.startswith("SP"):
            return "Raw"
        elif "BETAGRO" in desc_str: # Case sensitive
            return "RTE" # Ready-To-Eat
        else:
            return "RTC" # Ready-To-Cook
            
    if config.COL_INVENTORY_DESC in df.columns:
        df[config.COL_ITEM_CATEGORY] = df[config.COL_INVENTORY_DESC].apply(categorize_item).astype('category')
        logger.info("Created 'Item Category' feature.")
    else:
        logger.warning(f"'{config.COL_INVENTORY_DESC}' not found, cannot create 'Item Category'.")
        
    return df

def run_preprocessing():
    """
    Main function to run the entire data preprocessing pipeline.
    """
    logger.info("Starting data preprocessing...")

    # 1. Load Data
    raw_df = utils.load_csv_data(config.RAW_SALES_DATA_PATH)
    if raw_df is None:
        logger.error("Raw data loading failed. Exiting preprocessing.")
        return

    # Make a copy to avoid SettingWithCopyWarning
    df = raw_df.copy()

    # 2. Initial 'Price per qty' calculation (needed for imputing Inventory Code)
    # Handle potential division by zero if Qty can be 0
    df[config.COL_QTY] = df[config.COL_QTY].replace(0, np.nan)
    df[config.COL_PRICE_PER_QTY] = (df[config.COL_TOTAL_BASE_AMT] / df[config.COL_QTY]).round(2)
    # Fill NaNs in Price per qty that resulted from Qty=0 or Qty=NaN.
    # If Qty was NaN, Total Base Amt might also be NaN or 0.
    # If Qty was 0, Price per qty becomes inf; fill with 0 or a suitable placeholder.
    df[config.COL_PRICE_PER_QTY].replace([np.inf, -np.inf], 0, inplace=True) 
    df[config.COL_PRICE_PER_QTY].fillna(0, inplace=True) # For NaNs from Qty=NaN

    # 3. Reconcile Info & Handle Missing Values
    df = reconcile_customer_info(df)
    df = reconcile_inventory_info(df)
    df = handle_missing_values(df) # This also recalculates Price per qty if needed for new inv codes

    # 4. Aggregate Transactions
    # Price per qty should be consistent for aggregation, ensure it's up-to-date
    df[config.COL_QTY] = df[config.COL_QTY].replace(0, np.nan) # Recalculate before aggregation
    df[config.COL_PRICE_PER_QTY] = (df[config.COL_TOTAL_BASE_AMT] / df[config.COL_QTY]).round(2)
    df[config.COL_PRICE_PER_QTY].replace([np.inf, -np.inf], 0, inplace=True)
    df[config.COL_PRICE_PER_QTY].fillna(0, inplace=True)
    
    df = aggregate_transactions(df)

    # 5. Type Casting
    df = perform_type_casting(df)

    # 6. Feature Engineering
    df = engineer_features(df)
    
    # 7. Save Processed Data
    if utils.save_df_to_csv(df, config.AGGREGATED_DATA_PATH):
        logger.info(f"Successfully saved aggregated data to {config.AGGREGATED_DATA_PATH}")
    else:
        logger.error(f"Failed to save aggregated data to {config.AGGREGATED_DATA_PATH}")

    # Optional: Save a version without customer name
    if config.COL_CUSTOMER_NAME in df.columns:
        no_customer_name_df = df.drop(columns=[config.COL_CUSTOMER_NAME])
        if utils.save_df_to_csv(no_customer_name_df, config.NO_CUSTOMER_NAME_AGG_DATA_PATH):
            logger.info(f"Successfully saved data without customer name to {config.NO_CUSTOMER_NAME_AGG_DATA_PATH}")
        else:
            logger.error(f"Failed to save data without customer name to {config.NO_CUSTOMER_NAME_AGG_DATA_PATH}")

    logger.info("Data preprocessing finished.")
    return df


if __name__ == '__main__':
    # This allows the script to be run directly.
    # Ensure SalesData.csv is in data/raw/
    if not config.RAW_SALES_DATA_PATH.exists():
        print(f"ERROR: Raw sales data file not found at {config.RAW_SALES_DATA_PATH}")
        print("Please place 'SalesData.csv' in the 'data/raw/' directory before running.")
    else:
        processed_df = run_preprocessing()
        if processed_df is not None:
            print("\nPreprocessing complete. Sample of processed data:")
            print(processed_df.head())
            print(f"\nProcessed data saved to {config.PROCESSED_DATA_DIR}")