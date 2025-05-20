# price_elasticity_modeling.py
"""
Calculates own-price and cross-price elasticities for SKUs.
Corresponds to the logic in 'price_elasticity_original.md'.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm # For OLS regression
from itertools import combinations # For SKU pairs in cross-price elasticity

# Import project-specific modules
import config
import utils

# Configure logger
import logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def get_top_skus(df: pd.DataFrame, 
                 customer_categories: list, 
                 top_n: int) -> pd.DataFrame:
    """
    Identifies the top N SKUs for specified customer categories and item categories
    based on total sales amount.
    """
    logger.info(f"Identifying top {top_n} SKUs for customer categories: {customer_categories}...")
    
    # Filter for specified customer categories
    filtered_df = df[df[config.COL_CUSTOMER_CATEGORY_DESC].isin(customer_categories)].copy()
    
    # Group by Inventory Code, Desc, Customer Category, Item Category and sum Total Base Amt
    grouped_for_top_sku = filtered_df.groupby([
        config.COL_INVENTORY_CODE, 
        config.COL_INVENTORY_DESC, 
        config.COL_CUSTOMER_CATEGORY_DESC, 
        config.COL_ITEM_CATEGORY], observed=False # observed=False to align with pandas >= 1.5.0 behavior if categories are used
    )[config.COL_TOTAL_BASE_AMT].sum().reset_index()

    # Identify top N SKUs for each Item Category within each Customer Category Desc
    # The apply(lambda x: x.nlargest()) pattern is common for this.
    top_skus_df = grouped_for_top_sku.groupby(
        [config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY], observed=False
    ).apply(lambda x: x.nlargest(top_n, config.COL_TOTAL_BASE_AMT)).reset_index(drop=True)
    
    logger.info(f"Identified {len(top_skus_df)} top SKUs in total.")
    return top_skus_df

def prepare_data_for_elasticity(df: pd.DataFrame, top_skus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the main DataFrame for elasticity modeling by:
    1. Filtering for top SKUs.
    2. Aggregating daily sales data for these SKUs.
    3. Calculating Price Per Qty.
    4. Log-transforming price and quantity.
    5. Creating month dummy variables.
    """
    logger.info("Preparing data for elasticity modeling...")
    
    # Filter main df to include only the top SKUs
    # We need to filter based on Inventory Code AND Customer Category Desc, as top SKUs can differ
    # Create a unique identifier for top SKUs (Inventory Code + Customer Category Desc)
    top_skus_df['SKU_CustCat_ID'] = top_skus_df[config.COL_INVENTORY_CODE].astype(str) + "_" + top_skus_df[config.COL_CUSTOMER_CATEGORY_DESC].astype(str)
    df['SKU_CustCat_ID'] = df[config.COL_INVENTORY_CODE].astype(str) + "_" + df[config.COL_CUSTOMER_CATEGORY_DESC].astype(str)
    
    filtered_sales_df = df[df['SKU_CustCat_ID'].isin(top_skus_df['SKU_CustCat_ID'])].copy()
    filtered_sales_df.drop(columns=['SKU_CustCat_ID'], inplace=True)
    top_skus_df.drop(columns=['SKU_CustCat_ID'], inplace=True) # Clean up helper column

    logger.info(f"Filtered sales data to top SKUs. Shape: {filtered_sales_df.shape}")

    # Group by Transaction Date, Inventory Code, Desc, Customer Category, Item Category
    # This aggregates sales for each SKU on each day
    daily_sku_sales_df = filtered_sales_df.groupby([
        config.COL_TRANSACTION_DATE,
        config.COL_INVENTORY_CODE,
        config.COL_INVENTORY_DESC,
        config.COL_CUSTOMER_CATEGORY_DESC,
        config.COL_ITEM_CATEGORY
    ], observed=False).agg(
        Qty_Sum=(config.COL_QTY, 'sum'),
        Total_Base_Amt_Sum=(config.COL_TOTAL_BASE_AMT, 'sum')
    ).reset_index()

    daily_sku_sales_df.rename(columns={'Qty_Sum': config.COL_QTY, 
                                       'Total_Base_Amt_Sum': config.COL_TOTAL_BASE_AMT}, inplace=True)

    # Calculate Price per Qty
    daily_sku_sales_df[config.COL_QTY] = daily_sku_sales_df[config.COL_QTY].replace(0, np.nan) # Avoid division by zero
    daily_sku_sales_df[config.COL_PRICE_PER_QTY] = (daily_sku_sales_df[config.COL_TOTAL_BASE_AMT] / daily_sku_sales_df[config.COL_QTY])
    daily_sku_sales_df[config.COL_PRICE_PER_QTY].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Filter out rows where Price per Qty or Qty is zero, negative or NaN (critical for log transform)
    daily_sku_sales_df = daily_sku_sales_df[
        (daily_sku_sales_df[config.COL_PRICE_PER_QTY] > 0) & 
        (daily_sku_sales_df[config.COL_QTY] > 0)
    ].copy()
    daily_sku_sales_df.dropna(subset=[config.COL_PRICE_PER_QTY, config.COL_QTY], inplace=True)


    # Ensure Transaction Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(daily_sku_sales_df[config.COL_TRANSACTION_DATE]):
        daily_sku_sales_df[config.COL_TRANSACTION_DATE] = pd.to_datetime(daily_sku_sales_df[config.COL_TRANSACTION_DATE], errors='coerce')

    # Log transformations
    daily_sku_sales_df['log_price'] = np.log(daily_sku_sales_df[config.COL_PRICE_PER_QTY])
    daily_sku_sales_df['log_qty'] = np.log(daily_sku_sales_df[config.COL_QTY])

    # Create month dummy variables for seasonality
    daily_sku_sales_df['Month'] = daily_sku_sales_df[config.COL_TRANSACTION_DATE].dt.month.astype(str)
    # Month dummies will be created dynamically within the regression function
    
    logger.info(f"Data preparation complete. Shape: {daily_sku_sales_df.shape}")
    return daily_sku_sales_df

def calculate_own_price_elasticity(data_for_elasticity: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates own-price elasticity for each SKU using OLS regression.
    log_qty ~ log_price + month_dummies
    """
    logger.info("Calculating own-price elasticities...")
    results_list = []

    for category in data_for_elasticity[config.COL_CUSTOMER_CATEGORY_DESC].unique():
        category_data = data_for_elasticity[data_for_elasticity[config.COL_CUSTOMER_CATEGORY_DESC] == category]
        for sku_code in category_data[config.COL_INVENTORY_CODE].unique():
            sku_data = category_data[category_data[config.COL_INVENTORY_CODE] == sku_code].copy()
            
            if len(sku_data) > 10 : # Need enough data points for regression with month dummies
                # Create month dummies for this specific SKU's data to avoid all-zero columns
                month_dummies_sku = pd.get_dummies(sku_data['Month'], prefix='Month', drop_first=True).astype(float)
                
                X_sku = pd.concat([sku_data[['log_price']].reset_index(drop=True), 
                                   month_dummies_sku.reset_index(drop=True)], axis=1)
                X_sku = sm.add_constant(X_sku, has_constant='add') # Add constant term
                
                # Ensure no NaN/inf in X_sku or y_sku
                X_sku.replace([np.inf, -np.inf], np.nan, inplace=True)
                sku_data['log_qty'].replace([np.inf, -np.inf], np.nan, inplace=True)
                valid_indices = X_sku.dropna().index.intersection(sku_data['log_qty'].dropna().index)

                if len(valid_indices) > X_sku.shape[1]: # Check if enough valid data points remain
                    X_sku_clean = X_sku.loc[valid_indices]
                    y_sku_clean = sku_data['log_qty'].loc[valid_indices]
                
                    try:
                        model_sku = sm.OLS(y_sku_clean, X_sku_clean).fit()
                        price_elasticity = model_sku.params.get('log_price', np.nan) # Get param, default to NaN if not found
                        
                        results_list.append({
                            'SKU': sku_code,
                            config.COL_INVENTORY_DESC: sku_data[config.COL_INVENTORY_DESC].iloc[0],
                            config.COL_CUSTOMER_CATEGORY_DESC: category,
                            config.COL_ITEM_CATEGORY: sku_data[config.COL_ITEM_CATEGORY].iloc[0],
                            'Price_Elasticity': price_elasticity,
                            # 'Model_Summary_Own': model_sku.summary().as_text() # Optional
                        })
                    except Exception as e:
                        logger.warning(f"Could not fit OLS model for SKU {sku_code} in {category}: {e}")
                else:
                    logger.warning(f"Not enough valid data points after cleaning for SKU {sku_code} in {category} for OLS regression.")
            else:
                logger.warning(f"Not enough data points for SKU {sku_code} in {category} (found {len(sku_data)}). Skipping.")
                
    elasticities_df = pd.DataFrame(results_list)
    logger.info(f"Own-price elasticity calculation complete. Found {len(elasticities_df)} elasticities.")
    return elasticities_df

def calculate_cross_price_elasticity(data_for_elasticity: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cross-price elasticity between pairs of SKUs within the same
    Customer Category and Item Category.
    log_qty_A ~ log_price_B (+ month_dummies if desired)
    """
    logger.info("Calculating cross-price elasticities...")
    cross_elasticity_results = []

    for cust_cat in data_for_elasticity[config.COL_CUSTOMER_CATEGORY_DESC].unique():
        cust_cat_data = data_for_elasticity[data_for_elasticity[config.COL_CUSTOMER_CATEGORY_DESC] == cust_cat]
        for item_cat in cust_cat_data[config.COL_ITEM_CATEGORY].unique():
            item_cat_data = cust_cat_data[cust_cat_data[config.COL_ITEM_CATEGORY] == item_cat]
            skus_in_item_cat = item_cat_data[config.COL_INVENTORY_CODE].unique()

            if len(skus_in_item_cat) < 2:
                continue

            for sku_a, sku_b in combinations(skus_in_item_cat, 2):
                data_a = item_cat_data[item_cat_data[config.COL_INVENTORY_CODE] == sku_a].copy()
                data_b = item_cat_data[item_cat_data[config.COL_INVENTORY_CODE] == sku_b].copy()

                # Merge on Transaction Date to align observations
                # Suffixes are important to distinguish columns from SKU A and SKU B
                merged_pair_data = pd.merge(data_a, data_b, on=config.COL_TRANSACTION_DATE, suffixes=('_A', '_B'))

                if len(merged_pair_data) > 10: # Need enough common observations
                    # Dependent variable: log_qty of SKU A
                    y = merged_pair_data['log_qty_A']
                    
                    # Independent variable: log_price of SKU B
                    # Optionally include month dummies from merged data if desired
                    month_dummies_merged = pd.get_dummies(merged_pair_data['Month_A'], prefix='Month', drop_first=True).astype(float) # Use Month_A or Month_B
                    
                    X = pd.concat([merged_pair_data[['log_price_B']].reset_index(drop=True), 
                                   month_dummies_merged.reset_index(drop=True)], axis=1)
                    X = sm.add_constant(X, has_constant='add')
                    
                    X.replace([np.inf, -np.inf], np.nan, inplace=True)
                    y.replace([np.inf, -np.inf], np.nan, inplace=True)
                    valid_indices = X.dropna().index.intersection(y.dropna().index)

                    if len(valid_indices) > X.shape[1]:
                        X_clean = X.loc[valid_indices]
                        y_clean = y.loc[valid_indices]
                        try:
                            model_cross = sm.OLS(y_clean, X_clean).fit()
                            cross_elasticity = model_cross.params.get('log_price_B', np.nan)
                            
                            cross_elasticity_results.append({
                                config.COL_CUSTOMER_CATEGORY_DESC: cust_cat,
                                config.COL_ITEM_CATEGORY: item_cat,
                                'SKU_A': sku_a,
                                'SKU_A_Desc': data_a[config.COL_INVENTORY_DESC].iloc[0],
                                'SKU_B': sku_b,
                                'SKU_B_Desc': data_b[config.COL_INVENTORY_DESC].iloc[0],
                                'Cross_Price_Elasticity_A_on_B': cross_elasticity, # Qty_A vs Price_B
                                # 'Model_Summary_Cross': model_cross.summary().as_text() # Optional
                            })
                        except Exception as e:
                            logger.warning(f"Could not fit OLS model for cross-elasticity ({sku_a} vs {sku_b}): {e}")
                    else:
                        logger.warning(f"Not enough valid data points for cross-elasticity ({sku_a} vs {sku_b}).")
                else:
                    logger.debug(f"Not enough common observations for SKUs {sku_a} & {sku_b} (found {len(merged_pair_data)}).")
                    
    cross_elasticities_df = pd.DataFrame(cross_elasticity_results)
    logger.info(f"Cross-price elasticity calculation complete. Found {len(cross_elasticities_df)} relationships.")
    return cross_elasticities_df


def get_latest_transactions(df: pd.DataFrame, skus_of_interest: list) -> pd.DataFrame:
    """
    Gets the latest transaction price and quantity for a list of SKUs.
    """
    logger.info("Fetching latest transaction data for SKUs of interest...")
    # Filter for SKUs of interest from the original (or aggregated but not daily) sales data
    # This ensures we get the most recent 'Price per qty' as recorded, not daily average.
    
    # Using the main aggregated_df which has one row per order item after initial processing
    latest_df_filtered = df[df[config.COL_INVENTORY_CODE].isin(skus_of_interest)].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(latest_df_filtered[config.COL_TRANSACTION_DATE]):
        latest_df_filtered[config.COL_TRANSACTION_DATE] = pd.to_datetime(latest_df_filtered[config.COL_TRANSACTION_DATE], errors='coerce')

    latest_df_filtered.sort_values(by=config.COL_TRANSACTION_DATE, ascending=False, inplace=True)
    
    # Get the most recent entry for each SKU (and Customer Category, as price might vary)
    # The notebook groups by Inv Code, Trans Date, Inv Desc, Cust Cat, Price per Qty, then sums Qty.
    # Then drops duplicates on Inv Code. This seems to imply one price per SKU at its latest date.
    
    # Let's simplify: for each SKU and Customer Category, find the latest transaction and its price/qty.
    # This matches `combined_latest` logic from the notebook.
    
    latest_transactions = latest_df_filtered.loc[
        latest_df_filtered.groupby([config.COL_INVENTORY_CODE, config.COL_CUSTOMER_CATEGORY_DESC])[config.COL_TRANSACTION_DATE].idxmax()
    ]

    return latest_transactions[[
        config.COL_INVENTORY_CODE, 
        config.COL_INVENTORY_DESC, 
        config.COL_CUSTOMER_CATEGORY_DESC, 
        config.COL_PRICE_PER_QTY, # This is the price from that specific latest transaction
        config.COL_QTY, # Qty from that specific latest transaction
        config.COL_TRANSACTION_DATE
    ]].rename(columns={
        config.COL_INVENTORY_CODE: 'SKU',
        config.COL_PRICE_PER_QTY: 'Latest_Price_per_qty',
        config.COL_QTY: 'Latest_Qty',
        config.COL_TRANSACTION_DATE: 'Latest_Transaction_Date'
    })


def combine_and_categorize_elasticities(own_elasticities_df: pd.DataFrame, 
                                        cross_elasticities_df: pd.DataFrame,
                                        latest_transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges own-price, cross-price elasticities, and latest transaction data.
    Categorizes product pairs (complements/substitutes).
    """
    logger.info("Combining elasticity data and categorizing product pairs...")
    # Merge cross_elasticities with own_elasticities for SKU A
    merged_df = pd.merge(
        cross_elasticities_df,
        own_elasticities_df,
        how='left',
        left_on=[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 'SKU_A'],
        right_on=[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 'SKU'],
        suffixes=('_cross', '_ownA')
    )
    merged_df.rename(columns={'Price_Elasticity': 'Price_Elasticity_SKU_A'}, inplace=True)
    merged_df.drop(columns=['SKU', config.COL_INVENTORY_DESC + '_ownA'], inplace=True, errors='ignore') # Drop redundant SKU col

    # Merge again for SKU B's own-price elasticity
    merged_df = pd.merge(
        merged_df,
        own_elasticities_df,
        how='left',
        left_on=[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 'SKU_B'],
        right_on=[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 'SKU'],
        suffixes=('', '_ownB')
    )
    merged_df.rename(columns={'Price_Elasticity': 'Price_Elasticity_SKU_B'}, inplace=True)
    merged_df.drop(columns=['SKU', config.COL_INVENTORY_DESC + '_ownB'], inplace=True, errors='ignore')

    # Merge with latest transaction data for SKU A
    merged_df = pd.merge(
        merged_df,
        latest_transactions_df,
        how='left',
        left_on=['SKU_A', config.COL_CUSTOMER_CATEGORY_DESC],
        right_on=['SKU', config.COL_CUSTOMER_CATEGORY_DESC],
        suffixes=('', '_latestA')
    )
    merged_df.rename(columns={
        'Latest_Price_per_qty': 'SKU_A_Latest_Price',
        'Latest_Qty': 'SKU_A_Latest_Qty',
        'Latest_Transaction_Date': 'SKU_A_Latest_Date',
        config.COL_INVENTORY_DESC: 'SKU_A_Desc_Check' # Keep original SKU_A_Desc from cross_elasticities
    }, inplace=True)
    merged_df.drop(columns=['SKU', 'SKU_A_Desc_Check', config.COL_INVENTORY_DESC + '_latestA'], inplace=True, errors='ignore')


    # Merge with latest transaction data for SKU B
    merged_df = pd.merge(
        merged_df,
        latest_transactions_df,
        how='left',
        left_on=['SKU_B', config.COL_CUSTOMER_CATEGORY_DESC],
        right_on=['SKU', config.COL_CUSTOMER_CATEGORY_DESC],
        suffixes=('', '_latestB')
    )
    merged_df.rename(columns={
        'Latest_Price_per_qty': 'SKU_B_Latest_Price',
        'Latest_Qty': 'SKU_B_Latest_Qty',
        'Latest_Transaction_Date': 'SKU_B_Latest_Date',
        config.COL_INVENTORY_DESC: 'SKU_B_Desc_Check'  # Keep original SKU_B_Desc from cross_elasticities
    }, inplace=True)
    merged_df.drop(columns=['SKU', 'SKU_B_Desc_Check', config.COL_INVENTORY_DESC + '_latestB'], inplace=True, errors='ignore')
    
    # Categorize product pairs
    def categorize_pair(row):
        cross_elasticity = row['Cross_Price_Elasticity_A_on_B']
        if pd.isna(cross_elasticity):
            return 'independent - unknown' # Or some other placeholder
            
        if cross_elasticity < 0: pair_type = 'complements'
        elif cross_elasticity > 0: pair_type = 'substitutes'
        else: pair_type = 'independent'
        
        abs_elasticity = abs(cross_elasticity)
        if abs_elasticity > 1.5: closeness = 'close'
        elif 0.5 < abs_elasticity <= 1.5: closeness = 'moderate'
        elif 0 < abs_elasticity <= 0.5: closeness = 'weak'
        else: closeness = 'not related' # For zero elasticity
        return f"{pair_type} - {closeness}"

    merged_df['Product_Pair_Type'] = merged_df.apply(categorize_pair, axis=1)
    
    logger.info("Elasticity data combined and product pairs categorized.")
    return merged_df


def run_elasticity_modeling():
    """
    Main function to run the price elasticity modeling pipeline.
    """
    logger.info("Starting price elasticity modeling...")

    # 1. Load Processed Data (from data_preprocessing.py)
    aggregated_df = utils.load_csv_data(config.AGGREGATED_DATA_PATH)
    if aggregated_df is None:
        logger.error("Failed to load aggregated data. Exiting elasticity modeling.")
        return

    # Ensure Transaction Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(aggregated_df[config.COL_TRANSACTION_DATE]):
        aggregated_df[config.COL_TRANSACTION_DATE] = pd.to_datetime(aggregated_df[config.COL_TRANSACTION_DATE], errors='coerce')

    # 2. Identify Top SKUs
    top_skus_df = get_top_skus(aggregated_df, 
                               config.PRICE_ELASTICITY_CUSTOMER_CATEGORIES, 
                               config.PRICE_ELASTICITY_TOP_N_SKUS)
    if top_skus_df.empty:
        logger.error("No top SKUs identified. Exiting.")
        return

    # 3. Prepare Data for Elasticity Calculation
    data_for_elasticity_calc = prepare_data_for_elasticity(aggregated_df, top_skus_df)
    if data_for_elasticity_calc.empty:
        logger.error("Data preparation for elasticity resulted in an empty DataFrame. Exiting.")
        return

    # 4. Calculate Own-Price Elasticity
    own_elasticities_df = calculate_own_price_elasticity(data_for_elasticity_calc)

    # 5. Calculate Cross-Price Elasticity
    cross_elasticities_df = calculate_cross_price_elasticity(data_for_elasticity_calc)
    
    # 6. Get Latest Transaction Data for all SKUs involved in elasticity calculations
    all_involved_skus = pd.concat([
        own_elasticities_df['SKU'],
        cross_elasticities_df['SKU_A'],
        cross_elasticities_df['SKU_B']
    ]).unique()
    
    latest_transactions_df = get_latest_transactions(aggregated_df, list(all_involved_skus))

    # 7. Combine and Categorize
    final_elasticity_df = combine_and_categorize_elasticities(
        own_elasticities_df, 
        cross_elasticities_df,
        latest_transactions_df
    )

    # 8. Save Results
    if utils.save_df_to_csv(final_elasticity_df, config.PRICE_ELASTICITIES_OUTPUT_PATH):
        logger.info(f"Successfully saved final elasticity data to {config.PRICE_ELASTICITIES_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save final elasticity data.")

    logger.info("Price elasticity modeling finished.")
    return final_elasticity_df

if __name__ == '__main__':
    if not config.AGGREGATED_DATA_PATH.exists():
        print(f"ERROR: Aggregated data file not found at {config.AGGREGATED_DATA_PATH}")
        print("Please run data_preprocessing.py first.")
    else:
        elasticity_results_df = run_elasticity_modeling()
        if elasticity_results_df is not None:
            print("\nPrice elasticity modeling complete. Sample of results:")
            print(elasticity_results_df.head())
            print(f"\nElasticity data saved to {config.OPTIMIZATION_OUTPUT_DIR}")

