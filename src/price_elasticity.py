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
from . import config
from . import utils

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
    top_skus_df['SKU_CustCat_Key'] = top_skus_df[config.COL_INVENTORY_CODE].astype(str) + "_" + top_skus_df[config.COL_CUSTOMER_CATEGORY_DESC]
    df['SKU_CustCat_Key'] = df[config.COL_INVENTORY_CODE].astype(str) + "_" + df[config.COL_CUSTOMER_CATEGORY_DESC]
    
    # Filter the main df
    filtered_df = df[df['SKU_CustCat_Key'].isin(top_skus_df['SKU_CustCat_Key'])].copy()
    logger.info(f"Filtered to top SKUs. Shape after filtering: {filtered_df.shape}")
    
    # Drop the temporary key column
    filtered_df.drop(columns=['SKU_CustCat_Key'], inplace=True)
    
    # Aggregate daily SKU sales for each customer category
    # This ensures one datapoint per SKU per day, which is needed for time series regression
    daily_sku_sales_df = filtered_df.groupby([
        config.COL_TRANSACTION_DATE,
        config.COL_INVENTORY_CODE,
        config.COL_INVENTORY_DESC,
        config.COL_CUSTOMER_CATEGORY_DESC,
        config.COL_ITEM_CATEGORY,
        config.COL_PRICE_PER_QTY # Keep price granularity
    ], observed=False, dropna=False).agg({
        config.COL_QTY: 'sum',
        config.COL_TOTAL_BASE_AMT: 'sum'
    }).reset_index()
    
    logger.info(f"Aggregated daily SKU sales. Shape after aggregation: {daily_sku_sales_df.shape}")
    
    # Remove rows with zero or NaN quantity as they can't be log-transformed
    daily_sku_sales_df = daily_sku_sales_df[daily_sku_sales_df[config.COL_QTY] > 0].copy()
    
    # Remove rows with zero or NaN price as they can't be log-transformed
    daily_sku_sales_df = daily_sku_sales_df[daily_sku_sales_df[config.COL_PRICE_PER_QTY] > 0].copy()
    
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
                sku_data['log_qty'] = sku_data['log_qty'].replace([np.inf, -np.inf], np.nan)
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
                    month_dummies_merged = pd.get_dummies(merged_pair_data['Month_A'], prefix='Month', drop_first=True).astype(float)
                    
                    # Check for enough variation in log_price_B
                    if merged_pair_data['log_price_B'].nunique() > 1:
                        try:
                            X = sm.add_constant(pd.concat([merged_pair_data[['log_price_B']], month_dummies_merged], axis=1))
                            model_cross = sm.OLS(y, X).fit()
                            
                            # The cross-price elasticity is the coefficient of log_price_B
                            cross_elasticity_a_on_b = model_cross.params['log_price_B']

                            # P-value to determine statistical significance
                            p_value_a_on_b = model_cross.pvalues['log_price_B']
                            
                            # Calculate R-squared for model quality
                            r_squared_a_on_b = model_cross.rsquared
                            
                            # Only keep results with statistically significant elasticities (p < 0.05)
                            if p_value_a_on_b < 0.05:
                                cross_elasticity_results.append({
                                    'SKU_A': sku_a,
                                    'SKU_B': sku_b,
                                    'SKU_A_Desc': data_a[config.COL_INVENTORY_DESC].iloc[0],
                                    'SKU_B_Desc': data_b[config.COL_INVENTORY_DESC].iloc[0],
                                    config.COL_CUSTOMER_CATEGORY_DESC: cust_cat,
                                    config.COL_ITEM_CATEGORY: item_cat,
                                    'Cross_Price_Elasticity_A_on_B': cross_elasticity_a_on_b,
                                    'P_Value_A_on_B': p_value_a_on_b, 
                                    'R_Squared_A_on_B': r_squared_a_on_b,
                                })
                                
                                # Also calculate the reverse relationship: effect of SKU A's price on SKU B's quantity
                                y_reverse = merged_pair_data['log_qty_B']
                                X_reverse = sm.add_constant(pd.concat([merged_pair_data[['log_price_A']], month_dummies_merged], axis=1))
                                model_cross_reverse = sm.OLS(y_reverse, X_reverse).fit()
                                cross_elasticity_b_on_a = model_cross_reverse.params['log_price_A']
                                p_value_b_on_a = model_cross_reverse.pvalues['log_price_A']
                                r_squared_b_on_a = model_cross_reverse.rsquared
                                
                                if p_value_b_on_a < 0.05:
                                    cross_elasticity_results.append({
                                        'SKU_A': sku_b, # Note the swap: B is now A
                                        'SKU_B': sku_a, # Note the swap: A is now B
                                        'SKU_A_Desc': data_b[config.COL_INVENTORY_DESC].iloc[0],
                                        'SKU_B_Desc': data_a[config.COL_INVENTORY_DESC].iloc[0],
                                        config.COL_CUSTOMER_CATEGORY_DESC: cust_cat,
                                        config.COL_ITEM_CATEGORY: item_cat,
                                        'Cross_Price_Elasticity_A_on_B': cross_elasticity_b_on_a,
                                        'P_Value_A_on_B': p_value_b_on_a,
                                        'R_Squared_A_on_B': r_squared_b_on_a,
                                    })
                                    
                        except Exception as e:
                            logger.warning(f"Error calculating cross-price elasticity between SKU {sku_a} and SKU {sku_b}: {e}")
                    else:
                        logger.warning(f"Not enough price variation for SKU {sku_b} to calculate cross-price elasticity with SKU {sku_a}")
                else:
                    logger.debug(f"Not enough common observations for SKU pair {sku_a} and {sku_b} (found {len(merged_pair_data)})")

    cross_elasticities_df = pd.DataFrame(cross_elasticity_results)
    logger.info(f"Cross-price elasticity calculation complete. Found {len(cross_elasticities_df)} elasticities.")
    return cross_elasticities_df


def get_latest_transactions(df: pd.DataFrame, skus_of_interest: list) -> pd.DataFrame:
    """
    Gets the latest transactions for the SKUs of interest to determine 
    the most recent price and quantity for use in optimization.
    """
    logger.info(f"Getting latest transactions for {len(skus_of_interest)} SKUs...")
    
    if not skus_of_interest:
        logger.warning("No SKUs of interest provided")
        return pd.DataFrame()
    
    # Filter for SKUs of interest
    filtered_df = df[df[config.COL_INVENTORY_CODE].isin(skus_of_interest)].copy()
    
    if filtered_df.empty:
        logger.warning("No transactions found for the SKUs of interest")
        return pd.DataFrame()
    
    # Ensure Transaction Date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(filtered_df[config.COL_TRANSACTION_DATE]):
        filtered_df[config.COL_TRANSACTION_DATE] = pd.to_datetime(filtered_df[config.COL_TRANSACTION_DATE])
    
    # Create a new column to identify unique SKU + Customer Category combinations
    filtered_df['SKU_CustCat_Key'] = filtered_df[config.COL_INVENTORY_CODE].astype(str) + "_" + filtered_df[config.COL_CUSTOMER_CATEGORY_DESC]
    
    # Find the most recent date for each SKU-CustomerCategory combination
    latest_dates = filtered_df.groupby('SKU_CustCat_Key')[config.COL_TRANSACTION_DATE].max().reset_index()
    
    # Merge back to get the full records for the latest dates
    latest_transactions = []
    for _, row in latest_dates.iterrows():
        sku_cat_key = row['SKU_CustCat_Key']
        latest_date = row[config.COL_TRANSACTION_DATE]
        
        latest_for_sku_cat = filtered_df[
            (filtered_df['SKU_CustCat_Key'] == sku_cat_key) & 
            (filtered_df[config.COL_TRANSACTION_DATE] == latest_date)
        ]
        
        # If there are multiple transactions on the latest date, just take one
        if not latest_for_sku_cat.empty:
            latest_transactions.append(latest_for_sku_cat.iloc[0].to_dict())
    
    latest_transactions_df = pd.DataFrame(latest_transactions)
    
    if latest_transactions_df.empty:
        logger.warning("No latest transactions could be retrieved")
        return pd.DataFrame()
    
    # Drop the temporary key column
    latest_transactions_df.drop(columns=['SKU_CustCat_Key'], inplace=True, errors='ignore')
    
    logger.info(f"Retrieved {len(latest_transactions_df)} latest transactions")
    return latest_transactions_df


def combine_and_categorize_elasticities(own_elasticities_df: pd.DataFrame, 
                                        cross_elasticities_df: pd.DataFrame,
                                        latest_transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines own-price and cross-price elasticities, categorizes elasticity values,
    and adds latest price and quantity information for optimization.
    """
    logger.info("Combining and categorizing elasticities...")
    
    if own_elasticities_df.empty:
        logger.warning("Own-price elasticities DataFrame is empty")
        return pd.DataFrame()
        
    # 1. Start with own-price elasticities
    combined_df = own_elasticities_df.rename(columns={'SKU': 'SKU_A', 
                                                     'Price_Elasticity': 'Price_Elasticity_SKU_A'})
    
    # 2. If cross-price elasticities exist, set them up for merging
    if not cross_elasticities_df.empty:
        # For each SKU pair in cross-elasticities, we'll look up the own-price elasticity for SKU_B
        # We'll need the SKU_B and its own-price elasticity
        sku_b_own_price = own_elasticities_df.rename(columns={'SKU': 'SKU_B', 
                                                           'Price_Elasticity': 'Price_Elasticity_SKU_B'})
        
        # Select only the columns we need from sku_b_own_price
        sku_b_own_price = sku_b_own_price[['SKU_B', 'Price_Elasticity_SKU_B']]
        
        # Merge cross-elasticities with own-price elasticities for SKU_B
        cross_with_own_b = cross_elasticities_df.merge(sku_b_own_price, on='SKU_B', how='left')
        
        # Now combine own-price and cross-price elasticities
        combined_df = pd.concat([combined_df, cross_with_own_b], ignore_index=True, sort=False)
    
    # 3. Add latest price and quantity information
    if not latest_transactions_df.empty:
        # Prepare latest transactions data for merging
        latest_for_sku_a = latest_transactions_df.rename(columns={
            config.COL_INVENTORY_CODE: 'SKU_A',
            config.COL_PRICE_PER_QTY: 'SKU_A_Latest_Price',
            config.COL_QTY: 'SKU_A_Latest_Qty'
        })
        
        # Select only the columns we need
        latest_for_sku_a = latest_for_sku_a[['SKU_A', config.COL_CUSTOMER_CATEGORY_DESC, 
                                          'SKU_A_Latest_Price', 'SKU_A_Latest_Qty']]
        
        # Merge with combined elasticities
        combined_df = combined_df.merge(latest_for_sku_a, 
                                      on=['SKU_A', config.COL_CUSTOMER_CATEGORY_DESC],
                                      how='left')
        
        # If there are cross-elasticities, add latest price and quantity for SKU_B as well
        if 'SKU_B' in combined_df.columns:
            latest_for_sku_b = latest_transactions_df.rename(columns={
                config.COL_INVENTORY_CODE: 'SKU_B',
                config.COL_PRICE_PER_QTY: 'SKU_B_Latest_Price',
                config.COL_QTY: 'SKU_B_Latest_Qty'
            })
            
            latest_for_sku_b = latest_for_sku_b[['SKU_B', config.COL_CUSTOMER_CATEGORY_DESC, 
                                              'SKU_B_Latest_Price', 'SKU_B_Latest_Qty']]
            
            # Merge with combined elasticities
            combined_df = combined_df.merge(latest_for_sku_b, 
                                          on=['SKU_B', config.COL_CUSTOMER_CATEGORY_DESC],
                                          how='left')
    
    # 4. Categorize elasticities for easier interpretation
    # Categorize own-price elasticity
    def categorize_own_price_elasticity(elasticity):
        if pd.isnull(elasticity):
            return "Unknown"
        if elasticity > -0.5:
            return "Inelastic (< 0.5)"
        elif elasticity >= -0.5 and elasticity < -1:
            return "Unit Elastic (0.5-1)"
        else:
            return "Elastic (> 1)"
    
    # Categorize cross-price elasticity
    def categorize_cross_price_elasticity(elasticity):
        if pd.isnull(elasticity):
            return "Unknown"
        if elasticity > 0.5:
            return "Strong Substitutes (> 0.5)"
        elif elasticity > 0 and elasticity <= 0.5:
            return "Weak Substitutes (0-0.5)"
        elif elasticity >= -0.5 and elasticity < 0:
            return "Weak Complements (-0.5-0)"
        else:
            return "Strong Complements (< -0.5)"
    
    if 'Price_Elasticity_SKU_A' in combined_df.columns:
        combined_df['Own_Elasticity_Category'] = combined_df['Price_Elasticity_SKU_A'].apply(categorize_own_price_elasticity)
        
    if 'Cross_Price_Elasticity_A_on_B' in combined_df.columns:
        combined_df['Cross_Elasticity_Category'] = combined_df['Cross_Price_Elasticity_A_on_B'].apply(categorize_cross_price_elasticity)
    
    logger.info(f"Elasticities combined and categorized. Final DataFrame shape: {combined_df.shape}")
    return combined_df


def run_elasticity_modeling():
    """
    Main function to run the price elasticity modeling pipeline.
    """
    logger.info("Starting price elasticity modeling...")
    
    # 1. Load aggregated data
    aggregated_df = utils.load_csv_data(config.AGGREGATED_DATA_PATH)
    if aggregated_df is None:
        logger.error("Failed to load aggregated data. Elasticity modeling halted.")
        return None
    
    logger.info(f"Loaded aggregated data. Shape: {aggregated_df.shape}")
    
    # 2. Get top SKUs for specified customer categories
    top_skus_df = get_top_skus(aggregated_df, 
                             customer_categories=config.PRICE_ELASTICITY_CUSTOMER_CATEGORIES,
                             top_n=config.PRICE_ELASTICITY_TOP_N_SKUS)
    
    if top_skus_df.empty:
        logger.error("Failed to identify top SKUs. Elasticity modeling halted.")
        return None
    
    # 3. Prepare data for elasticity modeling
    logger.info("Preparing data for elasticity modeling...")
    data_for_elasticity = prepare_data_for_elasticity(aggregated_df, top_skus_df)
    
    if data_for_elasticity.empty:
        logger.error("Failed to prepare data for elasticity modeling. Process halted.")
        return None
    
    # 4. Calculate own-price elasticities
    logger.info("Calculating own-price elasticities...")
    own_elasticities_df = calculate_own_price_elasticity(data_for_elasticity)
    
    # 5. Calculate cross-price elasticities
    logger.info("Calculating cross-price elasticities...")
    cross_elasticities_df = calculate_cross_price_elasticity(data_for_elasticity)
    
    # 6. Get latest transactions for the SKUs included in elasticity calculations
    all_skus = set(top_skus_df[config.COL_INVENTORY_CODE].unique())
    logger.info("Getting latest transactions for elasticity calculations...")
    latest_transactions_df = get_latest_transactions(aggregated_df, list(all_skus))
    
    # 7. Combine and categorize elasticities
    logger.info("Combining and categorizing elasticities...")
    final_elasticity_df = combine_and_categorize_elasticities(
        own_elasticities_df, cross_elasticities_df, latest_transactions_df
    )
    
    # 8. Save the elasticity results
    if utils.save_df_to_csv(final_elasticity_df, config.PRICE_ELASTICITIES_OUTPUT_PATH):
        logger.info(f"Successfully saved elasticity results to {config.PRICE_ELASTICITIES_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save elasticity results to {config.PRICE_ELASTICITIES_OUTPUT_PATH}")
    
    logger.info("Price elasticity modeling completed successfully.")
    return final_elasticity_df

if __name__ == '__main__':
    # Configure logger
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)
    
    # Run the elasticity modeling pipeline
    elasticity_df = run_elasticity_modeling()
    
    # If run directly and elasticity calculation was successful, visualize the results
    if elasticity_df is not None and not elasticity_df.empty:
        try:
            # Set the style for visualizations
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set(style='whitegrid')
            
            # 1. Plot Own-Price Elasticity Distribution
            if 'Price_Elasticity_SKU_A' in elasticity_df.columns:
                plt.figure(figsize=(12, 6))
                sns.histplot(elasticity_df['Price_Elasticity_SKU_A'].dropna(), bins=20, kde=True)
                plt.title('Distribution of Own-Price Elasticities')
                plt.xlabel('Own-Price Elasticity')
                plt.ylabel('Frequency')
                plt.axvline(x=-1, color='r', linestyle='--', label='Unit Elastic')
                plt.legend()
                plt.tight_layout()
                plt.savefig('own_price_elasticity_distribution.png')
            
            # 2. Plot Cross-Price Elasticity Distribution
            if 'Cross_Price_Elasticity_A_on_B' in elasticity_df.columns:
                plt.figure(figsize=(12, 6))
                sns.histplot(elasticity_df['Cross_Price_Elasticity_A_on_B'].dropna(), bins=20, kde=True)
                plt.title('Distribution of Cross-Price Elasticities')
                plt.xlabel('Cross-Price Elasticity')
                plt.ylabel('Frequency')
                plt.axvline(x=0, color='r', linestyle='--', label='Substitutes/Complements Boundary')
                plt.legend()
                plt.tight_layout()
                plt.savefig('cross_price_elasticity_distribution.png')
            
            # 3. Show the plots if running interactively
            plt.show()
            
            logger.info("Visualizations completed and saved.")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            logger.info("Elasticity modeling completed successfully, but visualizations failed.")
    else:
        logger.error("Elasticity modeling failed. No visualizations generated.")

