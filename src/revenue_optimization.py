# revenue_optimization.py
"""
Performs revenue optimization using Gurobi based on calculated price elasticities.
Corresponds to the optimization logic in 'price_elasticity_original.md'.
"""
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Import project-specific modules
import config
import utils

# Configure logger
import logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def optimize_prices_for_category(elasticity_df_category: pd.DataFrame, category_name: str) -> pd.DataFrame | None:
    """
    Optimizes prices for a specific customer category to maximize total revenue.

    Args:
        elasticity_df_category (pd.DataFrame): DataFrame containing elasticity data,
            latest prices, and quantities for SKU pairs within a single customer category.
            Required columns: 'SKU_A', 'SKU_B', 'SKU_A_Desc', 'SKU_B_Desc',
                              'SKU_A_Latest_Price', 'SKU_A_Latest_Qty',
                              'SKU_B_Latest_Price', 'SKU_B_Latest_Qty',
                              'Price_Elasticity_SKU_A', 'Price_Elasticity_SKU_B',
                              'Cross_Price_Elasticity_A_on_B', 'Item_Category'.
        category_name (str): Name of the customer category being optimized.

    Returns:
        pd.DataFrame | None: DataFrame with optimization results (original vs. optimized
                              prices, quantities, revenues, and changes), or None if optimization fails.
    """
    logger.info(f"Starting price optimization for customer category: {category_name}...")
    
    # Check for required columns
    required_cols = [
        'SKU_A', 'SKU_B', 'SKU_A_Desc', 'SKU_B_Desc',
        'SKU_A_Latest_Price', 'SKU_A_Latest_Qty',
        'SKU_B_Latest_Price', 'SKU_B_Latest_Qty',
        'Price_Elasticity_SKU_A', 'Price_Elasticity_SKU_B',
        'Cross_Price_Elasticity_A_on_B', config.COL_ITEM_CATEGORY
    ]
    missing_cols = [col for col in required_cols if col not in elasticity_df_category.columns]
    if missing_cols:
        logger.error(f"Missing required columns for optimization in category {category_name}: {missing_cols}")
        return None
        
    # Handle NaN values in critical columns by dropping rows or imputing
    # For optimization, rows with NaN elasticities or prices/quantities are problematic
    critical_numeric_cols = [
        'SKU_A_Latest_Price', 'SKU_A_Latest_Qty', 'SKU_B_Latest_Price', 'SKU_B_Latest_Qty',
        'Price_Elasticity_SKU_A', 'Price_Elasticity_SKU_B', 'Cross_Price_Elasticity_A_on_B'
    ]
    df_clean = elasticity_df_category.dropna(subset=critical_numeric_cols).copy()
    
    if df_clean.empty:
        logger.warning(f"No valid data after cleaning NaNs for optimization in category {category_name}. Skipping.")
        return None

    try:
        # Create Gurobi model
        model = gp.Model(f'price_optimization_{category_name.replace(" ", "_")}')
        model.setParam('NonConvex', 2) # To handle quadratic objectives/constraints if they arise from bilinear terms

        # Define decision variables: percentage change in price for SKU A and SKU B
        # These are global for the category, as per the notebook's simpler model.
        # A more complex model might have per-SKU price changes.
        # The notebook implies two global price change variables for ALL pairs in the category.
        price_change_a = model.addVar(lb=config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND, 
                                      ub=config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND, 
                                      vtype=GRB.CONTINUOUS, name="price_change_sku_a_cat")
        price_change_b = model.addVar(lb=config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND, 
                                      ub=config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND, 
                                      vtype=GRB.CONTINUOUS, name="price_change_sku_b_cat")
        
        total_revenue_expr = gp.LinExpr() # Initialize as linear, will become quadratic

        # Store intermediate expressions for constraints if needed
        new_qty_a_exprs = {}
        new_qty_b_exprs = {}

        for index, row in df_clean.iterrows():
            p_a_orig = row['SKU_A_Latest_Price']
            q_a_orig = row['SKU_A_Latest_Qty']
            elas_a = row['Price_Elasticity_SKU_A']
            
            p_b_orig = row['SKU_B_Latest_Price']
            q_b_orig = row['SKU_B_Latest_Qty']
            elas_b = row['Price_Elasticity_SKU_B']
            
            cross_elas_ab = row['Cross_Price_Elasticity_A_on_B']
            # Assuming symmetric cross-elasticity for Qty_B w.r.t Price_A for simplicity, as in notebook
            # A more rigorous model would have cross_elas_ba.
            cross_elas_ba = cross_elas_ab 

            # New prices
            new_p_a = p_a_orig * (1 + price_change_a)
            new_p_b = p_b_orig * (1 + price_change_b)

            # New quantities (linear approximation of demand change)
            # Q_new = Q_old * (1 + E_own * Pct_Change_Own_Price + E_cross * Pct_Change_Other_Price)
            new_q_a = q_a_orig * (1 + elas_a * price_change_a + cross_elas_ab * price_change_b)
            new_q_b = q_b_orig * (1 + elas_b * price_change_b + cross_elas_ba * price_change_a)
            
            new_qty_a_exprs[index] = new_q_a
            new_qty_b_exprs[index] = new_q_b

            # Add non-negativity constraints for quantities
            model.addConstr(new_q_a >= 0, name=f"non_neg_qty_a_{index}")
            model.addConstr(new_q_b >= 0, name=f"non_neg_qty_b_{index}")
            
            # Revenue for this pair
            # Revenue = Price * Quantity. This will be quadratic.
            revenue_a_pair = new_p_a * new_q_a
            revenue_b_pair = new_p_b * new_q_b
            
            total_revenue_expr += revenue_a_pair + revenue_b_pair

        model.setObjective(total_revenue_expr, GRB.MAXIMIZE)
        model.optimize()

        revenue_changes_list = []
        if model.status == GRB.OPTIMAL:
            logger.info(f"Optimal solution found for category: {category_name}")
            opt_price_change_a = price_change_a.X
            opt_price_change_b = price_change_b.X

            for index, row in df_clean.iterrows():
                p_a_orig = row['SKU_A_Latest_Price']
                q_a_orig = row['SKU_A_Latest_Qty']
                elas_a = row['Price_Elasticity_SKU_A']
                
                p_b_orig = row['SKU_B_Latest_Price']
                q_b_orig = row['SKU_B_Latest_Qty']
                elas_b = row['Price_Elasticity_SKU_B']
                cross_elas_ab = row['Cross_Price_Elasticity_A_on_B']
                cross_elas_ba = cross_elas_ab # Symmetry assumption

                opt_p_a = p_a_orig * (1 + opt_price_change_a)
                opt_p_b = p_b_orig * (1 + opt_price_change_b)

                opt_q_a = q_a_orig * (1 + elas_a * opt_price_change_a + cross_elas_ab * opt_price_change_b)
                opt_q_b = q_b_orig * (1 + elas_b * opt_price_change_b + cross_elas_ba * opt_price_change_a)
                
                # Ensure non-negative quantities in output
                opt_q_a = max(0, opt_q_a)
                opt_q_b = max(0, opt_q_b)

                rev_a_orig = p_a_orig * q_a_orig
                rev_b_orig = p_b_orig * q_b_orig
                opt_rev_a = opt_p_a * opt_q_a
                opt_rev_b = opt_p_b * opt_q_b

                revenue_changes_list.append({
                    config.COL_CUSTOMER_CATEGORY_DESC: category_name,
                    config.COL_ITEM_CATEGORY: row[config.COL_ITEM_CATEGORY],
                    'SKU_A': row['SKU_A'], 'SKU_A_Desc': row['SKU_A_Desc'],
                    'Original_Price_A': p_a_orig, 'Original_Qty_A': q_a_orig,
                    'Optimized_Price_A': opt_p_a, 'Optimized_Qty_A': opt_q_a,
                    'SKU_B': row['SKU_B'], 'SKU_B_Desc': row['SKU_B_Desc'],
                    'Original_Price_B': p_b_orig, 'Original_Qty_B': q_b_orig,
                    'Optimized_Price_B': opt_p_b, 'Optimized_Qty_B': opt_q_b,
                    'Original_Revenue_A': rev_a_orig, 'Optimized_Revenue_A': opt_rev_a,
                    'Change_Revenue_A': opt_rev_a - rev_a_orig,
                    'Original_Revenue_B': rev_b_orig, 'Optimized_Revenue_B': opt_rev_b,
                    'Change_Revenue_B': opt_rev_b - rev_b_orig,
                    'Total_Change_Revenue_Pair': (opt_rev_a - rev_a_orig) + (opt_rev_b - rev_b_orig)
                })
            return pd.DataFrame(revenue_changes_list)
        else:
            logger.error(f"No optimal solution found for category: {category_name}. Status: {model.status}")
            return None
            
    except gp.GurobiError as e:
        logger.error(f"Gurobi error during optimization for {category_name}: {e.message} (code: {e.errno})")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during optimization for {category_name}: {e}")
        return None


def run_optimization():
    """
    Main function to run the revenue optimization pipeline.
    """
    logger.info("Starting revenue optimization...")

    # 1. Load Elasticity Data (output from price_elasticity_modeling.py)
    elasticity_df = utils.load_csv_data(config.PRICE_ELASTICITIES_OUTPUT_PATH)
    if elasticity_df is None:
        logger.error("Failed to load elasticity data. Exiting optimization.")
        return

    all_optimization_results = []
    
    # The notebook runs optimization separately for 'SUPERMARKET' and 'RETAIL'
    customer_categories_to_optimize = elasticity_df[config.COL_CUSTOMER_CATEGORY_DESC].unique()
    
    for category in customer_categories_to_optimize:
        logger.info(f"Processing optimization for customer category: {category}")
        category_specific_df = elasticity_df[elasticity_df[config.COL_CUSTOMER_CATEGORY_DESC] == category].copy()
        
        if category_specific_df.empty:
            logger.info(f"No elasticity data for category '{category}'. Skipping.")
            continue
            
        optimization_result_category = optimize_prices_for_category(category_specific_df, category)
        
        if optimization_result_category is not None:
            all_optimization_results.append(optimization_result_category)
        else:
            logger.warning(f"Optimization failed or returned no results for category: {category}")

    if not all_optimization_results:
        logger.error("No optimization results generated for any category.")
        return None

    final_revenue_changes_df = pd.concat(all_optimization_results, ignore_index=True)
    
    # Add back other columns from the original elasticity_df for full context if needed
    # (e.g., Product_Pair_Type, own elasticities, transaction dates)
    # This merge needs to be careful about column names.
    # The `final_revenue_changes_df` already has SKU_A, SKU_B, Customer Category, Item Category.
    # We need to select distinct columns from `elasticity_df` to merge.
    
    cols_to_merge_from_elasticity = [
        config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 
        'SKU_A', 'SKU_B', 
        'Price_Elasticity_SKU_A', 'Price_Elasticity_SKU_B', 
        'Cross_Price_Elasticity_A_on_B', 'Product_Pair_Type',
        'SKU_A_Latest_Date', 'SKU_B_Latest_Date' # Assuming these were in the output of elasticity script
    ]
    # Ensure these columns exist in elasticity_df
    cols_to_merge_from_elasticity = [col for col in cols_to_merge_from_elasticity if col in elasticity_df.columns]

    if cols_to_merge_from_elasticity:
         final_revenue_changes_df = pd.merge(
             final_revenue_changes_df,
             elasticity_df[cols_to_merge_from_elasticity].drop_duplicates(),
             on=[config.COL_CUSTOMER_CATEGORY_DESC, config.COL_ITEM_CATEGORY, 'SKU_A', 'SKU_B'],
             how='left',
             suffixes=('', '_orig_elasticity') # Avoids conflict if any same-named cols are generated
         )


    # 4. Save Optimization Results
    if utils.save_df_to_csv(final_revenue_changes_df, config.REVENUE_OPTIMIZATION_OUTPUT_PATH):
        logger.info(f"Successfully saved revenue optimization results to {config.REVENUE_OPTIMIZATION_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save revenue optimization results.")

    logger.info("Revenue optimization finished.")
    return final_revenue_changes_df


if __name__ == '__main__':
    if not config.PRICE_ELASTICITIES_OUTPUT_PATH.exists():
        print(f"ERROR: Price elasticity data file not found at {config.PRICE_ELASTICITIES_OUTPUT_PATH}")
        print("Please run price_elasticity_modeling.py first.")
    else:
        optimization_results_df = run_optimization()
        if optimization_results_df is not None:
            print("\nRevenue optimization complete. Sample of results:")
            print(optimization_results_df.head())
            
            # Summary of revenue changes by category
            if not optimization_results_df.empty:
                summary = optimization_results_df.groupby(config.COL_CUSTOMER_CATEGORY_DESC)['Total_Change_Revenue_Pair'].sum().reset_index()
                summary.columns = [config.COL_CUSTOMER_CATEGORY_DESC, 'Total_Optimized_Revenue_Change']
                print("\nTotal Optimized Revenue Change by Customer Category:")
                print(summary)
            
            print(f"\nOptimization results saved to {config.OPTIMIZATION_OUTPUT_DIR}")  