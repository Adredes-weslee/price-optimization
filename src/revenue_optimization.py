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
from . import config
from . import utils

# Configure logger
import logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def optimize_prices_for_category(elasticity_df_category: pd.DataFrame, category_name: str) -> pd.DataFrame | None:
    """
    Optimizes prices for a specific customer category to maximize total revenue.

    Args:
        elasticity_df_category (pd.DataFrame): DataFrame containing elasticity data,
            latest prices, and quantities for SKU pairs.
            Required columns: 'SKU_A', 'SKU_A_Desc', 'SKU_A_Latest_Price', 
                             'SKU_A_Latest_Qty', 'Price_Elasticity_SKU_A'
        category_name (str): Name of the category being optimized (can be a customer category or 'All Categories').

    Returns:
        pd.DataFrame | None: DataFrame with optimization results (original vs. optimized
                              prices, quantities, revenues, and changes), or None if optimization fails.
    """
    logger.info(f"Starting price optimization for category: {category_name}...")
    
    # Check for required columns for basic optimization (own-price elasticity)
    required_cols = [
        'SKU_A', 'SKU_A_Desc', 'SKU_A_Latest_Price', 'SKU_A_Latest_Qty', 'Price_Elasticity_SKU_A'
    ]
    missing_cols = [col for col in required_cols if col not in elasticity_df_category.columns]
    if missing_cols:
        logger.error(f"Missing required columns for optimization: {missing_cols}")
        logger.error(f"Available columns: {elasticity_df_category.columns.tolist()}")
        return None
    
    # Get SKUs with own-price elasticity (these are the ones we'll optimize)
    own_elasticity_data = elasticity_df_category.dropna(subset=['SKU_A', 'Price_Elasticity_SKU_A', 
                                                              'SKU_A_Latest_Price', 'SKU_A_Latest_Qty'])
    
    # Check if we have any own-price elasticity data
    if own_elasticity_data.empty:
        logger.error(f"No valid own-price elasticity data for {category_name}")
        return None
        
    # For cross-price effects, we'll use what we have (can be empty)
    # This makes the optimization more flexible if cross-price data is limited
    df_clean = elasticity_df_category.copy()
    
    # Create item category if missing
    if config.COL_ITEM_CATEGORY not in df_clean.columns:
        logger.warning(f"{config.COL_ITEM_CATEGORY} column not found, using default category")
        df_clean[config.COL_ITEM_CATEGORY] = 'Unknown'    
    try:
        # Get unique SKUs and their attributes from own elasticity data
        logger.info("Preparing optimization parameters...")
        
        # Get unique SKUs that have own-price elasticity
        unique_skus = set(own_elasticity_data['SKU_A'].unique())
        logger.info(f"Found {len(unique_skus)} SKUs with valid own-price elasticity data")
        
        # Create a dictionary for easy lookup of SKU attributes
        sku_attrs = {}
        for _, row in own_elasticity_data.drop_duplicates('SKU_A').iterrows():
            sku = row['SKU_A']
            sku_attrs[sku] = {
                'desc': row['SKU_A_Desc'],
                'current_price': row['SKU_A_Latest_Price'],
                'current_qty': row['SKU_A_Latest_Qty'],
                'own_elasticity': row['Price_Elasticity_SKU_A'],
                'item_category': row[config.COL_ITEM_CATEGORY] if config.COL_ITEM_CATEGORY in row else 'Unknown'
            }
        
        # Create a dictionary for cross-price elasticities (if available)
        cross_elasticities = {}
        
        # Only process cross elasticities for SKUs that have own-price elasticity
        if 'SKU_B' in df_clean.columns and 'Cross_Price_Elasticity_A_on_B' in df_clean.columns:
            cross_elasticity_rows = df_clean.dropna(subset=['SKU_A', 'SKU_B', 'Cross_Price_Elasticity_A_on_B'])
            
            for _, row in cross_elasticity_rows.iterrows():
                sku_a = row['SKU_A']
                sku_b = row['SKU_B']
                
                # Only include if both SKUs have own-price elasticity (both in our unique_skus list)
                if sku_a in unique_skus and sku_b in unique_skus:
                    cross_elasticity = row['Cross_Price_Elasticity_A_on_B']
                    cross_elasticities[(sku_a, sku_b)] = cross_elasticity
            
            logger.info(f"Found {len(cross_elasticities)} valid cross-price elasticity relationships")
        
        # Create a new Gurobi model
        logger.info("Creating Gurobi optimization model...")
        model = gp.Model(f"Price_Optimization_{category_name}")
        
        # Decision variables: price multipliers for each SKU
        price_multipliers = {}
        for sku in unique_skus:
            # Use config parameters instead of hardcoded values
            lower_bound = 1 + config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND
            upper_bound = 1 + config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND
            
            # Log the bounds being used
            logger.info(f"Setting price bounds for SKU {sku}: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            price_multipliers[sku] = model.addVar(
                lb=lower_bound,
                ub=upper_bound,
                name=f"price_mult_{sku}"
            )
        
        # Add objective function: Maximize total revenue
        logger.info("Setting up objective function and constraints...")
        obj = gp.LinExpr(0)
        
        for sku_a in unique_skus:
            # Base quantity for this SKU
            base_qty = sku_attrs[sku_a]['current_qty']
            base_price = sku_attrs[sku_a]['current_price']
            own_elasticity = sku_attrs[sku_a]['own_elasticity']
            
            # Quantity change due to own-price effect
            # If price increases by 1%, quantity decreases by own_elasticity%
            # Using the formula: new_qty = base_qty * (new_price/base_price)^elasticity
            
            # We'll linearize this for the optimization: 
            # log(new_qty) = log(base_qty) + elasticity * log(new_price/base_price)
            # For small changes, we can approximate: new_qty ≈ base_qty * (1 + elasticity * (price_change_percent))
            
            # Price change percent is (price_multiplier - 1)
            # So: new_qty ≈ base_qty * (1 + own_elasticity * (price_multiplier - 1))
            
            # Add own-price effect
            own_effect = base_qty * (1 + own_elasticity * (price_multipliers[sku_a] - 1))
            
            # Add cross-price effects from all other SKUs
            for sku_b in unique_skus:
                if sku_a != sku_b and (sku_a, sku_b) in cross_elasticities:
                    cross_elasticity = cross_elasticities[(sku_a, sku_b)]
                    
                    # Cross-price effect:
                    # If price of SKU B increases by 1%, quantity of SKU A changes by cross_elasticity%
                    cross_effect = base_qty * cross_elasticity * (price_multipliers[sku_b] - 1)
                    own_effect += cross_effect
            
            # Revenue for this SKU: new_price * new_qty
            revenue = base_price * price_multipliers[sku_a] * own_effect
            obj += revenue
        
        # Set the objective
        model.setObjective(obj, GRB.MAXIMIZE)
        
        # Add item category constraints (ensuring similar price changes within categories)
        item_categories = df_clean[config.COL_ITEM_CATEGORY].unique()
        for category in item_categories:
            skus_in_category = [sku for sku in unique_skus if sku_attrs[sku]['item_category'] == category]
            
            if len(skus_in_category) > 1:
                # For each pair of SKUs in the same category
                for i in range(len(skus_in_category)):
                    for j in range(i+1, len(skus_in_category)):
                        sku_i = skus_in_category[i]
                        sku_j = skus_in_category[j]
                        
                        # Add constraint: price changes shouldn't differ by more than 5%
                        model.addConstr(
                            price_multipliers[sku_i] - price_multipliers[sku_j] <= 0.05,
                            f"max_price_diff_{sku_i}_{sku_j}_upper"
                        )
                        model.addConstr(
                            price_multipliers[sku_i] - price_multipliers[sku_j] >= -0.05,
                            f"max_price_diff_{sku_i}_{sku_j}_lower"
                        )
        
        # Optimize the model
        logger.info("Solving optimization model...")
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            logger.info(f"Optimal solution found for {category_name}")
            
            # Extract results
            results = []
            total_original_revenue = 0
            total_optimized_revenue = 0
            
            for sku in unique_skus:
                # Get original values
                orig_price = sku_attrs[sku]['current_price']
                orig_qty = sku_attrs[sku]['current_qty']
                orig_revenue = orig_price * orig_qty
                total_original_revenue += orig_revenue
                
                # Get optimized price
                price_mult = price_multipliers[sku].x
                opt_price = orig_price * price_mult
                
                # Calculate optimized quantity using elasticity
                own_elasticity = sku_attrs[sku]['own_elasticity']
                pct_price_change = price_mult - 1  # e.g., 1.05 - 1 = 0.05 (5% increase)
                
                # Base quantity change due to own-price effect
                pct_qty_change = own_elasticity * pct_price_change
                opt_qty = orig_qty * (1 + pct_qty_change)
                
                # Add cross-price effects
                for other_sku in unique_skus:
                    if other_sku != sku and (sku, other_sku) in cross_elasticities:
                        cross_elasticity = cross_elasticities[(sku, other_sku)]
                        other_price_mult = price_multipliers[other_sku].x
                        other_pct_price_change = other_price_mult - 1
                        
                        # Additional quantity change due to cross-price effect
                        cross_pct_qty_change = cross_elasticity * other_pct_price_change
                        opt_qty += orig_qty * cross_pct_qty_change
                
                # Ensure quantity doesn't go negative (can happen with large elasticities)
                opt_qty = max(0, opt_qty)
                
                # Calculate optimized revenue
                opt_revenue = opt_price * opt_qty
                total_optimized_revenue += opt_revenue
                
                # Calculate changes
                price_change_pct = (opt_price / orig_price - 1) * 100
                qty_change_pct = (opt_qty / orig_qty - 1) * 100 if orig_qty > 0 else 0
                revenue_change_pct = (opt_revenue / orig_revenue - 1) * 100 if orig_revenue > 0 else 0
                
                results.append({
                    'SKU': sku,
                    'Description': sku_attrs[sku]['desc'],
                    'Item_Category': sku_attrs[sku]['item_category'],
                    'Original_Price': orig_price,
                    'Optimized_Price': opt_price,
                    'Price_Change_Pct': price_change_pct,
                    'Original_Qty': orig_qty,
                    'Optimized_Qty': opt_qty,
                    'Qty_Change_Pct': qty_change_pct,
                    'Original_Revenue': orig_revenue,
                    'Optimized_Revenue': opt_revenue,
                    'Revenue_Change_Pct': revenue_change_pct,
                    'Customer_Category': category_name
                })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Add summary row
            total_revenue_change_pct = ((total_optimized_revenue / total_original_revenue) - 1) * 100 if total_original_revenue > 0 else 0
            logger.info(f"Total revenue change for {category_name}: {total_revenue_change_pct:.2f}%")
            logger.info(f"Original revenue: {total_original_revenue:.2f}, Optimized revenue: {total_optimized_revenue:.2f}")
            
            return results_df
            
        else:
            logger.error(f"Optimization failed for {category_name}: Status {model.status}")
            return None
            
    except gp.GurobiError as e:
        logger.error(f"Gurobi error during optimization for {category_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during optimization for {category_name}: {e}")
        return None


def run_optimization():
    """
    Main function to run the revenue optimization pipeline.
    """
    logger.info("Starting revenue optimization...")
    
    # 1. Load elasticity data
    elasticity_df = utils.load_csv_data(config.PRICE_ELASTICITIES_OUTPUT_PATH)
    if elasticity_df is None or elasticity_df.empty:
        logger.error("Failed to load elasticity data. Optimization halted.")
        return None
    
    logger.info(f"Loaded elasticity data. Shape: {elasticity_df.shape}")
    
    # 2. Run optimization across all categories together
    results_list = []
    
    # Check if required columns exist
    required_cols = ['SKU_A', 'SKU_A_Desc', 'SKU_A_Latest_Price', 'SKU_A_Latest_Qty', 'Price_Elasticity_SKU_A']
    
    missing_cols = [col for col in required_cols if col not in elasticity_df.columns]
    
    if missing_cols:
        logger.error(f"Elasticity data is missing required columns: {missing_cols}")
        logger.error("Optimization cannot proceed without these columns.")
        return None
    
    # Run optimization on all data together
    logger.info("Optimizing prices across all SKUs...")
    
    # Get own price elasticity data (where SKU_A has no corresponding SKU_B)
    own_elasticity_data = elasticity_df[elasticity_df['SKU_B'].isnull()].copy()
    
    # Filter to remove rows with NaN in critical columns for own elasticity data
    own_elasticity_data = own_elasticity_data.dropna(subset=['SKU_A', 'Price_Elasticity_SKU_A', 'SKU_A_Latest_Price', 'SKU_A_Latest_Qty'])
    
    if own_elasticity_data.empty:
        logger.error("No valid own-price elasticity data available for optimization")
        return None
    
    # Use a simplified version of Item_Category if available, or create a default one
    if config.COL_ITEM_CATEGORY not in own_elasticity_data.columns:
        logger.warning(f"{config.COL_ITEM_CATEGORY} column not found, using default category for all SKUs")
        own_elasticity_data[config.COL_ITEM_CATEGORY] = 'Unknown'
    
    # Get cross-price elasticity data where we have valid data
    cross_elasticity_data = elasticity_df[
        ~elasticity_df['SKU_B'].isnull() & 
        ~elasticity_df['Cross_Price_Elasticity_A_on_B'].isnull()
    ].copy()
    
    # Add customer category column if missing
    if config.COL_CUSTOMER_CATEGORY_DESC not in own_elasticity_data.columns:
        own_elasticity_data[config.COL_CUSTOMER_CATEGORY_DESC] = 'All'
    
    # Combine the data for optimization
    optimization_data = own_elasticity_data.copy()
    optimization_data['SKU_B'] = None  # Placeholder for SKU_B
    optimization_data['Cross_Price_Elasticity_A_on_B'] = None  # Placeholder
    
    # Add cross-elasticity data to the SKU_A data
    for idx, row in cross_elasticity_data.iterrows():
        # Only include cross-elasticity data where SKU_A is in our main dataset
        if row['SKU_A'] in own_elasticity_data['SKU_A'].values:
            # Insert cross-elasticity data
            new_row = optimization_data[optimization_data['SKU_A'] == row['SKU_A']].iloc[0].copy()
            new_row['SKU_B'] = row['SKU_B']
            new_row['SKU_B_Desc'] = row['SKU_B_Desc']
            new_row['SKU_B_Latest_Price'] = row['SKU_B_Latest_Price']
            new_row['SKU_B_Latest_Qty'] = row['SKU_B_Latest_Qty']
            new_row['Cross_Price_Elasticity_A_on_B'] = row['Cross_Price_Elasticity_A_on_B']
            
            optimization_data = pd.concat([optimization_data, pd.DataFrame([new_row])], ignore_index=True)
    
    # Run the optimization
    optimization_results = optimize_prices_for_category(optimization_data, 'All Categories')
    
    if optimization_results is not None and not optimization_results.empty:
        results_list.append(optimization_results)
    else:
        logger.warning("No optimization results obtained")
    
    # Combine results
    if not results_list:
        logger.error("No optimization results obtained.")
        return None
        
    final_results = pd.concat(results_list, ignore_index=True)
    
    # Save the optimization results
    if utils.save_df_to_csv(final_results, config.REVENUE_OPTIMIZATION_OUTPUT_PATH):
        logger.info(f"Successfully saved optimization results to {config.REVENUE_OPTIMIZATION_OUTPUT_PATH}")
    else:
        logger.error(f"Failed to save optimization results to {config.REVENUE_OPTIMIZATION_OUTPUT_PATH}")
    
    # Calculate and log the overall revenue impact
    total_original = final_results['Original_Revenue'].sum()
    total_optimized = final_results['Optimized_Revenue'].sum()
    overall_change_pct = ((total_optimized / total_original) - 1) * 100 if total_original > 0 else 0
    
    logger.info(f"Overall optimization results:")
    logger.info(f"  - Total Original Revenue: {total_original:.2f}")
    logger.info(f"  - Total Optimized Revenue: {total_optimized:.2f}")
    logger.info(f"  - Total Revenue Change: {total_optimized - total_original:.2f} ({overall_change_pct:.2f}%)")
    logger.info(f"  - Number of SKUs optimized: {len(final_results)}")
    
    logger.info("Revenue optimization completed successfully.")
    return final_results

if __name__ == '__main__':
    # Configure logger
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
    logger = logging.getLogger(__name__)
    
    try:
        # Run the optimization pipeline
        optimization_results = run_optimization()
        
        # If run directly and optimization was successful, visualize the results
        if optimization_results is not None and not optimization_results.empty:
            try:
                # Set the style for visualizations
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.set(style='whitegrid')
                
                # 1. Plot Revenue Change by SKU
                plt.figure(figsize=(14, 8))
                sorted_results = optimization_results.sort_values('Revenue_Change_Pct', ascending=False)
                
                # If too many SKUs, only show the top and bottom 20
                if len(sorted_results) > 40:
                    top_results = sorted_results.head(20)
                    bottom_results = sorted_results.tail(20)
                    plot_results = pd.concat([top_results, bottom_results])
                else:
                    plot_results = sorted_results
                
                # Create the bar plot
                ax = sns.barplot(x='Description', y='Revenue_Change_Pct', hue='Customer_Category', data=plot_results)
                plt.title('Revenue Change by SKU (%) After Price Optimization')
                plt.xticks(rotation=90, fontsize=8)
                plt.xlabel('SKU Description')
                plt.ylabel('Revenue Change (%)')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig('revenue_change_by_sku.png')
                
                # 2. Plot Price Changes as a histogram
                plt.figure(figsize=(10, 6))
                sns.histplot(optimization_results['Price_Change_Pct'], bins=20, kde=True)
                plt.title('Distribution of Price Changes (%)')
                plt.xlabel('Price Change (%)')
                plt.ylabel('Frequency')
                plt.axvline(x=0, color='r', linestyle='--')
                plt.tight_layout()
                plt.savefig('price_changes_distribution.png')
                
                # 3. Plot Revenue Change by Customer Category
                plt.figure(figsize=(10, 6))
                category_summary = optimization_results.groupby('Customer_Category').agg({
                    'Original_Revenue': 'sum',
                    'Optimized_Revenue': 'sum'
                }).reset_index()
                
                category_summary['Revenue_Change_Pct'] = ((category_summary['Optimized_Revenue'] / 
                                                      category_summary['Original_Revenue']) - 1) * 100
                
                sns.barplot(x='Customer_Category', y='Revenue_Change_Pct', data=category_summary)
                plt.title('Revenue Change by Customer Category (%)')
                plt.xlabel('Customer Category')
                plt.ylabel('Revenue Change (%)')
                plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                plt.tight_layout()
                plt.savefig('revenue_change_by_category.png')
                
                # Show the plots
                plt.show()
                
                logger.info("Visualizations completed and saved.")
                
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                logger.info("Optimization completed successfully, but visualizations failed.")
    except Exception as e:
        logger.error(f"Error running optimization: {e}")
        logger.error("Optimization failed.")