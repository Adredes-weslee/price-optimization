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
    For anonymized data, we skip customer name reconciliation since we're
    working with customer codes only. This function is kept as a stub
    to maintain the overall processing pipeline structure.
    """
    logger.info("Skipping customer name reconciliation for anonymized data.")
    
    # Remove Customer Name column if it exists to ensure we're not using it
    if config.COL_CUSTOMER_NAME in df.columns:
        logger.info(f"Removing '{config.COL_CUSTOMER_NAME}' column from anonymized dataset.")
        df = df.drop(columns=[config.COL_CUSTOMER_NAME])
    
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
    # No need to handle specific customer cases with anonymized data
    # We'll just ensure categories are filled properly
    if config.COL_CUSTOMER_CATEGORY_DESC in df.columns:
        # Fill any missing category with a default value
        missing_categories = df[config.COL_CUSTOMER_CATEGORY_DESC].isna().sum()
        if missing_categories > 0:
            df[config.COL_CUSTOMER_CATEGORY_DESC].fillna('UNKNOWN', inplace=True)
            logger.info(f"Filled {missing_categories} missing customer categories with 'UNKNOWN'")
    else:
        logger.warning(f"Column {config.COL_CUSTOMER_CATEGORY_DESC} not found in dataset")

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

    rows_with_nan_inv_code = df[df[config.COL_INVENTORY_CODE].isna()]
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
    # For anonymized data, we exclude Customer Name and only use codes and other attributes
    grouping_keys = [
        config.COL_TRANSACTION_DATE,
        config.COL_SALES_ORDER_NO,
        config.COL_CUSTOMER_CODE,
        config.COL_INVENTORY_CODE,
        # Only include columns that exist in the dataframe
        *(col for col in [
            config.COL_CUSTOMER_CATEGORY_DESC, # Assuming reconciled/filled
            config.COL_INVENTORY_DESC, # Assuming reconciled/filled
            config.COL_PRICE_PER_QTY # Calculated and rounded
        ] if col in df.columns)
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
    Casts columns to appropriate types.
    """
    # Ensure Transaction Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[config.COL_TRANSACTION_DATE]):
        df[config.COL_TRANSACTION_DATE] = pd.to_datetime(df[config.COL_TRANSACTION_DATE])
        logger.info("Cast Transaction Date to datetime format.")
    
    # Cast numeric columns to appropriate types
    numeric_columns = [
        config.COL_QTY,
        config.COL_TOTAL_BASE_AMT,
        config.COL_PRICE_PER_QTY,
        config.COL_INVENTORY_CODE
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # For Inventory Code, use float as it appears to be stored that way in the source data
            if col == config.COL_INVENTORY_CODE:
                df[col] = df[col].astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    logger.info("Cast numeric columns to appropriate types.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers additional features useful for analysis.
    """
    # 1. Extract Year/Month/Day from Transaction Date
    if pd.api.types.is_datetime64_any_dtype(df[config.COL_TRANSACTION_DATE]):
        df['Year'] = df[config.COL_TRANSACTION_DATE].dt.year
        df['Month'] = df[config.COL_TRANSACTION_DATE].dt.month
        df['Day'] = df[config.COL_TRANSACTION_DATE].dt.day
        logger.info("Extracted Year, Month, Day from Transaction Date.")
    
    # 2. Add Customer Category Broad if not already present
    if 'Customer Category Broad' not in df.columns:
        # Map from customer category descriptions to broader categories
        # This is a simplified example - adjust the mapping as needed
        category_mapping = {
            'SUPERMARKET': 'Retail',
            'RETAIL': 'Retail',
            'SCHOOL': 'Institutional',
            'F&B': 'Food Service',
            'RESTAURANT': 'Food Service'
            # Add more mappings as needed
        }
        
        df['Customer Category Broad'] = df[config.COL_CUSTOMER_CATEGORY_DESC].map(category_mapping)
        # For any unmapped categories, set a default
        df['Customer Category Broad'].fillna('Other', inplace=True)
        logger.info("Added Customer Category Broad column.")
    
    # 3. Add Item Category if not already present 
    if config.COL_ITEM_CATEGORY not in df.columns:
        # Create a basic mapping based on product names
        # This is a simplified example - adjust the logic as needed
        def map_item_category(description):
            desc_lower = str(description).lower()
            if any(keyword in desc_lower for keyword in ['raw', 'fresh', 'chix', 'chicken']):
                return 'Raw'
            elif any(keyword in desc_lower for keyword in ['rtc', 'ready', 'cooked', 'grilled', 'fried']):
                return 'RTC'  # Ready to Cook
            else:
                return 'Other'
        
        df[config.COL_ITEM_CATEGORY] = df[config.COL_INVENTORY_DESC].apply(map_item_category)
        logger.info("Added Item Category column based on Inventory Description.")
    
    return df

def run_preprocessing():
    """
    Main function to run the data preprocessing pipeline.
    """
    logger.info("Starting data preprocessing...")
    
    # 1. Load raw data
    raw_sales_df = utils.load_csv_data(config.RAW_SALES_DATA_PATH)
    if raw_sales_df is None:
        logger.error("Failed to load raw sales data. Preprocessing halted.")
        return None
    
    logger.info(f"Loaded raw sales data. Shape: {raw_sales_df.shape}")
    
    # 2. Reconcile data inconsistencies
    logger.info("Reconciling data inconsistencies...")
    reconciled_df = reconcile_customer_info(raw_sales_df)
    reconciled_df = reconcile_inventory_info(reconciled_df)
    
    # 3. Handle missing values
    logger.info("Handling missing values...")
    filled_df = handle_missing_values(reconciled_df)
    
    # 4. Aggregate transactions
    logger.info("Aggregating transactions...")
    aggregated_df = aggregate_transactions(filled_df)
    
    # 5. Perform type casting
    logger.info("Performing type casting...")
    typed_df = perform_type_casting(aggregated_df)
    
    # 6. Engineer features
    logger.info("Engineering features...")
    final_df = engineer_features(typed_df)
      # 7. Save processed data
    if utils.save_df_to_csv(final_df, config.AGGREGATED_DATA_PATH):
        logger.info(f"Successfully saved processed data to {config.AGGREGATED_DATA_PATH}")
    else:
        logger.error(f"Failed to save processed data to {config.AGGREGATED_DATA_PATH}")
    
    # 8. No need to create a version without customer names since we're already using anonymized data
    # But we'll copy to the NO_CUSTOMER_NAME path as well for compatibility with the rest of the pipeline
    if config.NO_CUSTOMER_NAME_AGG_DATA_PATH is not None:
        if utils.save_df_to_csv(final_df, config.NO_CUSTOMER_NAME_AGG_DATA_PATH):
            logger.info(f"Successfully saved data to {config.NO_CUSTOMER_NAME_AGG_DATA_PATH}")
    
    logger.info("Data preprocessing completed successfully.")
    return final_df

if __name__ == '__main__':
    # This allows the script to be run directly.
    # Ensure sales_data.csv is in data/raw/
    if not config.RAW_SALES_DATA_PATH.exists():
        print(f"ERROR: Raw sales data file not found at {config.RAW_SALES_DATA_PATH}")
        print("Please place 'sales_data.csv' in the 'data/raw/' directory before running.")
    else:
        processed_df = run_preprocessing()
        if processed_df is not None:
            print("\nPreprocessing complete. Sample of processed data:")
            print(processed_df.head())
            print(f"\nProcessed data saved to {config.PROCESSED_DATA_DIR}")