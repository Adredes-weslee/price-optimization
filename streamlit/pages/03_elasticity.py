import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config, price_elasticity, utils

st.title("Price Elasticity Analysis")

# Check if processed data exists
if st.session_state.processed_data is None:
    st.warning("Please complete the data preprocessing step first!")
    st.stop()

# Elasticity configuration
st.subheader("Price Elasticity Configuration")

# Customer category selection
available_categories = st.session_state.processed_data[config.COL_CUSTOMER_CATEGORY_DESC].unique().tolist()
selected_categories = st.multiselect("Select Customer Categories", available_categories, default=available_categories[:2] if len(available_categories) >= 2 else available_categories)

# Top SKUs configuration
col1, col2 = st.columns(2)
with col1:
    top_n_skus = st.slider("Number of Top SKUs to Analyze", min_value=5, max_value=50, value=10)
with col2:
    min_data_points = st.slider("Minimum Data Points for Elasticity", min_value=3, max_value=20, value=5)

# Run elasticity analysis
if st.button("Calculate Price Elasticities"):
    st.info("Running price elasticity analysis...")
    
    # Override config parameters
    config.PRICE_ELASTICITY_CUSTOMER_CATEGORIES = selected_categories
    config.PRICE_ELASTICITY_TOP_N_SKUS = top_n_skus
    
    try:
        # Prepare data for elasticity
        elasticity_data_dict = {}
        for category in selected_categories:
            prepared_data = price_elasticity.prepare_data_for_elasticity(
                st.session_state.processed_data, 
                category
            )
            elasticity_data_dict[category] = prepared_data
        
        # Calculate own-price elasticities
        own_price_elasticities = {}
        for category, data in elasticity_data_dict.items():
            category_elasticities = price_elasticity.calculate_own_price_elasticities(data, min_data_points)
            own_price_elasticities[category] = category_elasticities
        
        # Calculate cross-price elasticities
        cross_price_elasticities = {}
        for category, data in elasticity_data_dict.items():
            category_cross_elasticities = price_elasticity.calculate_cross_price_elasticities(
                data, 
                own_price_elasticities[category],
                min_data_points
            )
            cross_price_elasticities[category] = category_cross_elasticities
        
        # Combine elasticities
        combined_elasticities = price_elasticity.combine_and_categorize_elasticities(
            own_price_elasticities, 
            cross_price_elasticities
        )
        
        # Store results in session state
        st.session_state.elasticity_results = combined_elasticities
        
        # Display results
        st.success("Price elasticity analysis complete!")
        
        # Show elasticity results
        st.subheader("Own-Price Elasticity Results")
        
        own_elasticity_df = combined_elasticities[combined_elasticities['SKU_B'].isnull()]
        st.dataframe(own_elasticity_df)
        
        # Visualization
        st.subheader("Own-Price Elasticity Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(own_elasticity_df['Price_Elasticity_SKU_A'], kde=True, ax=ax)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title("Distribution of Own-Price Elasticities")
        plt.xlabel("Elasticity Value")
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error during elasticity analysis: {e}")