"""
About page for the Price Optimization Dashboard
"""
import streamlit as st
import sys
import os

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src import config

# Add this after imports, before any other Streamlit calls
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("About This Project")

st.markdown("""
## Price Optimization System

This project implements a comprehensive price optimization system for retail products based on
customer segmentation and price elasticity modeling. It uses advanced analytics to determine optimal
pricing strategies that maximize revenue while maintaining competitive market positioning.

### Why Price Optimization Matters

Price optimization is critical for retail businesses in today's competitive market. Finding the optimal 
price point that balances revenue, profitability, and customer satisfaction can significantly impact 
a company's bottom line. Key benefits include:

- **Revenue Enhancement**: Optimize prices based on customer willingness to pay
- **Inventory Management**: Reduce excess inventory through strategic pricing
- **Competitive Positioning**: Set prices strategically relative to market competitors
- **Customer Retention**: Maintain customer loyalty with fair, data-driven pricing

### Data Sources

This analysis uses the following data:

- **Transaction Data**: Historical sales records including products, quantities, and prices
- **Customer Data**: Anonymized customer information for segmentation
- **Product Data**: Product attributes and categorizations
- **Price Points**: Historical price changes and their impact on sales

### Analysis Methods

The project includes:

- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis and K-means clustering
- **Price Elasticity Modeling**: Own-price and cross-price elasticity calculations
- **Revenue Optimization**: Mathematical optimization to determine ideal price points
- **Scenario Simulation**: "What-if" analysis to model different pricing strategies

### Project Structure

This project is organized with the following components:

- **src/**: Core business logic modules
  - **data_preprocessing.py**: Cleans and prepares transaction data
  - **customer_segmentation.py**: Implements RFM and K-means clustering
  - **price_elasticity.py**: Calculates price elasticity values
  - **revenue_optimization.py**: Optimizes prices for maximum revenue
  - **utils.py**: Utility functions for data handling
  - **config.py**: Configuration parameters
  
- **data/**: Input and output data files
  - **raw/**: Original transaction data
  - **processed/**: Cleaned and aggregated data
  - **segmentation/**: Customer segment outputs
  - **optimization/**: Price elasticity and optimization results

- **streamlit/**: Interactive dashboard
  - **app.py**: Main entry point
  - **pages/**: Multi-page structure for different analytics views
  - **utils/**: Visualization and UI helper functions

### Methodology

The analysis follows these steps:

1. **Data Preprocessing**: Clean, validate, and prepare transaction data
2. **Customer Segmentation**: Group customers based on purchase behavior
3. **Price Elasticity Modeling**: Calculate how price changes affect demand
4. **Optimization**: Find optimal price points to maximize revenue
5. **Scenario Analysis**: Test different pricing strategies

### Business Applications

This price optimization system has numerous applications:

- **Promotional Planning**: Design effective promotions based on price sensitivity
- **Product Bundling**: Create profitable product bundles based on cross-elasticity
- **Strategic Pricing**: Set prices based on customer segments and their value
- **Competitive Response**: Model response strategies to market price changes
- **Revenue Management**: Optimize overall revenue across product categories

### References

- [Price Elasticity of Demand](https://en.wikipedia.org/wiki/Price_elasticity_of_demand)
- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))
- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
""")

st.subheader("Project Contributors")

st.markdown("""
This project was developed as a price optimization solution for retail businesses.

**Technologies Used:**
- Python for data processing and analysis
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning models
- Matplotlib and Seaborn for visualization
- Gurobi for mathematical optimization
- Streamlit for interactive dashboard
""")

# Add contact information
with st.expander("Contact Information"):
    st.write("""
    For questions or feedback about this project, please contact:
    - Email: weslee.qb@gamil.com
    - GitHub: https://github.com/Adredes-weslee/price-optimization
    """)
    
# Add data acknowledgments
with st.expander("Data Acknowledgments"):
    st.write("""
    This project uses anonymized retail transaction data.
    """)
    
# Display project version
st.sidebar.info("Project Version: 1.0.0")