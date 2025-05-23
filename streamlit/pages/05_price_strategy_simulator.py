import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from copy import deepcopy

def setup_paths():
    """Add necessary directories to Python path for imports"""
    import sys
    import os
    
    # Add project root (for src imports)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    
    # Add streamlit directory (for utils imports)
    streamlit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if streamlit_dir not in sys.path:
        sys.path.append(streamlit_dir)

setup_paths()

from src import config, revenue_optimization
from utils import visualizations as viz 

# Add this after imports, before any other Streamlit calls
st.set_page_config(
    page_title="Price Simulator",
    page_icon="🔍",
    layout="wide"
)

st.title("Price Strategy Simulator")

# Check if we have the required data
if st.session_state.elasticity_results is None:
    st.warning("Please complete the price elasticity analysis step first!")
    st.stop()

if st.session_state.optimization_results is None:
    st.warning("Please run the price optimization step first to establish a baseline!")
    st.stop()

# Display baseline results
st.subheader("Baseline Optimization Results")
baseline_results = st.session_state.optimization_results
total_original = baseline_results['Original_Revenue'].sum()
total_optimized = baseline_results['Optimized_Revenue'].sum()
baseline_change_pct = ((total_optimized / total_original) - 1) * 100 if total_original > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Original Revenue", f"${total_original:.2f}")
col2.metric("Optimized Revenue", f"${total_optimized:.2f}")
col3.metric("Revenue Change", f"{baseline_change_pct:.2f}%", delta=f"${total_optimized - total_original:.2f}")

# Scenario configuration
st.subheader("Configure Pricing Scenario")

scenario_type = st.selectbox(
    "Scenario Type",
    [
        "Price Bounds",
        "Product Selection",
        "Elasticity Adjustment",
        "Competitive Response"
    ]
)

if scenario_type == "Price Bounds":
    st.write("Adjust the allowed price change bounds for optimization.")
    
    col1, col2 = st.columns(2)
    with col1:
        lower_bound = st.slider(
            "Lower Price Bound (%)", 
            min_value=-50, 
            max_value=0, 
            value=int(config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND * 100)
        )
        lower_bound = lower_bound / 100
    with col2:
        upper_bound = st.slider(
            "Upper Price Bound (%)", 
            min_value=0, 
            max_value=200, 
            value=int(config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND * 100)
        )
        upper_bound = upper_bound / 100
    
    # Run scenario
    if st.button("Run Scenario Simulation"):
        with st.spinner("Running alternative optimization scenario..."):
            # Save original bounds
            original_lower = config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND
            original_upper = config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND
            
            # Override config parameters
            config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND = lower_bound
            config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND = upper_bound
            
            # Run optimization with new bounds
            scenario_results = revenue_optimization.run_optimization()
            
            # Restore original bounds
            config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND = original_lower
            config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND = original_upper
            
            # Display scenario results
            if scenario_results is not None and not scenario_results.empty:
                # Calculate summary statistics
                scenario_original = scenario_results['Original_Revenue'].sum()
                scenario_optimized = scenario_results['Optimized_Revenue'].sum()
                scenario_change_pct = ((scenario_optimized / scenario_original) - 1) * 100
                
                st.subheader("Scenario Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Scenario Revenue", f"${scenario_optimized:.2f}")
                col2.metric("Baseline Revenue", f"${total_optimized:.2f}")
                col3.metric(
                    "Difference", 
                    f"{scenario_change_pct:.2f}%", 
                    delta=f"${scenario_optimized - total_optimized:.2f}"
                )
                
                # Compare visually
                st.subheader("Comparison with Baseline")
                
                comparison_df = pd.merge(
                    baseline_results[['SKU', 'Description', 'Optimized_Price', 'Revenue_Change_Pct']],
                    scenario_results[['SKU', 'Optimized_Price', 'Revenue_Change_Pct']],
                    on=['SKU'],
                    suffixes=('_baseline', '_scenario')
                )
                
                st.dataframe(comparison_df.sort_values('Revenue_Change_Pct_scenario', ascending=False))
                
                # Plot comparison
                fig, ax = plt.subplots(figsize=(12, 8))
                
                ax.scatter(
                    comparison_df['Revenue_Change_Pct_baseline'],
                    comparison_df['Revenue_Change_Pct_scenario'],
                    alpha=0.7
                )
                
                # Add 45-degree line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])
                ]
                ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
                
                # Add labels
                ax.set_xlabel('Baseline Revenue Change (%)')
                ax.set_ylabel('Scenario Revenue Change (%)')
                ax.set_title('Comparison of Revenue Changes: Baseline vs Scenario')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Display price impact comparison
                scenario_results['Description'] = scenario_results['Description'].fillna('Unknown').astype(str)
                price_fig = viz.plot_price_change_impact(scenario_results)
                if price_fig:
                    st.pyplot(price_fig)
            else:
                st.error("Scenario optimization didn't produce valid results.")
                
elif scenario_type == "Product Selection":
    st.write("Optimize prices for only a subset of products.")
    
    # Get unique products from elasticity data
    elasticity_df = st.session_state.elasticity_results
    st.info(f"DEBUG: Columns in elasticity data: {elasticity_df.columns.tolist()}")

    # Create products dataframe with more flexible column mapping
    products = pd.DataFrame()

    # Check if we're working with already-processed data
    if 'SKU' in elasticity_df.columns:
        # We're working with already processed data (likely from optimization results)
        products['SKU'] = elasticity_df['SKU']
        if 'Description' in elasticity_df.columns:
            products['Description'] = elasticity_df['Description']
        else:
            products['Description'] = "Product " + elasticity_df['SKU'].astype(str)
        
        if config.COL_ITEM_CATEGORY in elasticity_df.columns:
            products[config.COL_ITEM_CATEGORY] = elasticity_df[config.COL_ITEM_CATEGORY]
        elif 'Item_Category' in elasticity_df.columns:
            products[config.COL_ITEM_CATEGORY] = elasticity_df['Item_Category']
        else:
            products[config.COL_ITEM_CATEGORY] = "Unknown"
            
    # Check if we have the raw elasticity data format
    elif 'SKU_A' in elasticity_df.columns:
        products['SKU'] = elasticity_df['SKU_A']
        
        if 'SKU_A_Desc' in elasticity_df.columns:
            products['Description'] = elasticity_df['SKU_A_Desc']
        elif 'Inventory Desc' in elasticity_df.columns:
            products['Description'] = elasticity_df['Inventory Desc']
        else:
            products['Description'] = "Product " + elasticity_df['SKU_A'].astype(str)
            
        if 'Item Category' in elasticity_df.columns:
            products[config.COL_ITEM_CATEGORY] = elasticity_df['Item Category']
        else:
            products[config.COL_ITEM_CATEGORY] = "Unknown"
    else:
        st.error("No product identifier columns found in data")
        products = pd.DataFrame({
            'SKU': ["Unknown"],
            'Description': ["Unknown"],
            config.COL_ITEM_CATEGORY: ["Unknown"]
        })

    # Remove duplicates
    products = products.drop_duplicates()

    # Make sure Description is a string to avoid len() errors later
    products['Description'] = products['Description'].fillna('Unknown').astype(str)
        
    # Group products by category
    categories = products[config.COL_ITEM_CATEGORY].unique()
    
    # Let user select product categories
    selected_categories = st.multiselect(
        "Product Categories to Include",
        options=categories,
        default=list(categories)
    )
    
    # Filter products by selected categories
    filtered_products = products[products[config.COL_ITEM_CATEGORY].isin(selected_categories)]
    
    # Let user select specific products (optional)
    custom_selection = st.checkbox("Select specific products")
    
    if custom_selection:
        selected_skus = st.multiselect(
            "Products to Include",
            options=filtered_products.apply(lambda x: f"{x['SKU']} - {x['Description']}", axis=1),
            default=filtered_products.apply(lambda x: f"{x['SKU']} - {x['Description']}", axis=1).tolist()
        )
        selected_skus = [sku.split(" - ")[0] for sku in selected_skus]
    else:
        selected_skus = filtered_products['SKU'].tolist()
    
    # Run scenario
    if st.button("Run Scenario Simulation"):
        st.info(f"Running optimization for {len(selected_skus)} selected products...")
        
        # TODO: Implement custom product selection in revenue_optimization or create a modified version
        st.error("This feature is not yet implemented.")
        
        # For demonstration purposes:
        st.write("Selected Products:")
        selected_products = filtered_products[filtered_products['SKU'].isin(selected_skus)]
        st.dataframe(selected_products)

# Other scenario types can be implemented similarly
elif scenario_type == "Elasticity Adjustment":
    st.write("Adjust elasticity values to simulate different market conditions.")
    st.info("This feature is coming soon.")
    
elif scenario_type == "Competitive Response":
    st.write("Simulate how competitors might respond to your price changes.")
    st.info("This feature is coming soon.")