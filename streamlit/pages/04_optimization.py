import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config, revenue_optimization, utils

st.title("Revenue Optimization")

# Check if elasticity results exist
if st.session_state.elasticity_results is None:
    st.warning("Please complete the price elasticity analysis step first!")
    st.stop()

# Optimization configuration
st.subheader("Optimization Configuration")

col1, col2 = st.columns(2)
with col1:
    lower_bound = st.slider("Lower Price Bound (%)", min_value=-50, max_value=0, value=-20)
    lower_bound = lower_bound / 100 + 1  # Convert -20% to 0.8
with col2:
    upper_bound = st.slider("Upper Price Bound (%)", min_value=0, max_value=50, value=20)
    upper_bound = upper_bound / 100 + 1  # Convert 20% to 1.2

# Run optimization
if st.button("Optimize Prices"):
    st.info("Running price optimization...")
    
    try:
        # Run optimization with custom bounds
        with st.spinner("Optimizing prices..."):
            # Override config parameters
            config.OPTIMIZATION_PRICE_CHANGE_LOWER_BOUND = lower_bound - 1  # Convert back to percentage change
            config.OPTIMIZATION_PRICE_CHANGE_UPPER_BOUND = upper_bound - 1  # Convert back to percentage change
            
            # Run optimization
            optimization_results = revenue_optimization.run_optimization()
            
            # Store results in session state
            st.session_state.optimization_results = optimization_results
            
            # Display results
            if optimization_results is not None and not optimization_results.empty:
                st.success("Price optimization complete!")
                
                # Calculate summary statistics
                total_original = optimization_results['Original_Revenue'].sum()
                total_optimized = optimization_results['Optimized_Revenue'].sum()
                overall_change_pct = ((total_optimized / total_original) - 1) * 100 if total_original > 0 else 0
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Revenue", f"${total_original:.2f}")
                col2.metric("Optimized Revenue", f"${total_optimized:.2f}")
                col3.metric("Revenue Change", f"{overall_change_pct:.2f}%", 
                           delta=f"${total_optimized - total_original:.2f}")
                
                # Show optimization results table
                st.subheader("Optimization Results by SKU")
                st.dataframe(optimization_results.sort_values('Revenue_Change_Pct', ascending=False))
                
                # Visualizations
                st.subheader("Revenue Change by SKU")
                
                # If too many SKUs, only show the top and bottom 10
                sorted_results = optimization_results.sort_values('Revenue_Change_Pct', ascending=False)
                if len(sorted_results) > 20:
                    top_results = sorted_results.head(10)
                    bottom_results = sorted_results.tail(10)
                    plot_results = pd.concat([top_results, bottom_results])
                else:
                    plot_results = sorted_results
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x='Description', y='Revenue_Change_Pct', data=plot_results, ax=ax)
                plt.xticks(rotation=90)
                plt.title("Revenue Change by SKU (%)")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Price Change Distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(optimization_results['Price_Change_Pct'], kde=True, ax=ax)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.title("Distribution of Price Changes (%)")
                plt.xlabel("Price Change (%)")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error("Optimization didn't produce valid results. Check the data and parameters.")
                
    except Exception as e:
        st.error(f"Error during optimization: {e}")