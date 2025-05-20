# CS Tay: Customer Segmentation & Price Optimization - Creation Guide

This guide outlines the structure and workflow for converting the CS Tay project from notebooks (represented by Markdown files) into a set of modular Python scripts. The goal is to create a reproducible and maintainable data analysis pipeline.

## 1. Project Overview

The project aims to perform customer segmentation and price optimization for CS Tay, a frozen food distributor. It involves several key stages:
1.  **Data Preprocessing**: Cleaning and preparing the raw sales data.
2.  **Customer Segmentation**: Grouping customers based on their purchasing behavior (RFM) and using K-Means clustering.
3.  **Price Elasticity Modeling**: Calculating how changes in price affect demand for products (own-price and cross-price elasticity).
4.  **Revenue Optimization**: Using the elasticity models to find optimal pricing strategies to maximize revenue.

## 2. Proposed Project Structure

We will organize the project into the following Python scripts:


cs_tay_project/
│
├── data/                     # Directory to store input CSVs (e.g., SalesData.csv) and output CSVs
│   ├── raw/                  # Original input data
│   │   └── SalesData.csv     # (User needs to place the original sales data here)
│   ├── processed/            # Output from data_preprocessing.py
│   │   └── aggregated_df.csv
│   │   └── no_customer_name_agg_df.csv # (Optional, if Customer Name is to be excluded for some steps)
│   ├── segmentation/         # Output from customer_segmentation.py
│   │   └── customer_segmentation_df.csv
│   └── optimization/         # Output from price_elasticity_modeling.py & revenue_optimization.py
│       └── price_elasticity_final_df.csv
│       └── revenue_optimization_results.csv
│
├── scripts/                  # Python scripts for the pipeline
│   ├── config.py             # Configuration variables (file paths, S3 details if any)
│   ├── utils.py              # Utility functions (e.g., data loading from S3, common calculations)
│   ├── data_preprocessing.py # Script for data cleaning and preparation
│   ├── customer_segmentation.py # Script for RFM and K-Means segmentation
│   ├── price_elasticity_modeling.py # Script for elasticity calculations
│   ├── revenue_optimization.py # Script for price optimization using Gurobi
│   └── main.py               # Main script to run the entire pipeline
│
└── README.md                 # Project README
└── requirements.txt          # Python package dependencies


## 3. Script Breakdown and Logic

### 3.1. `config.py`

* **Purpose**: To store all configuration variables and constants in one place for easy management.
* **Key Contents**:
    * File paths for input data (e.g., `RAW_SALES_DATA_PATH`).
    * File paths for intermediate and final output CSVs (e.g., `AGGREGATED_DATA_PATH`, `CUSTOMER_SEGMENTATION_OUTPUT_PATH`).
    * S3 bucket and key details if data is to be loaded from/saved to S3 (though the example notebooks use local temporary paths after download).
    * Lists of columns, or other project-specific constants.

### 3.2. `utils.py`

* **Purpose**: To house common utility functions used across different scripts.
* **Potential Functions**:
    * `load_csv_data(file_path, encoding='ISO-8859-1')`: Loads data from a CSV file.
    * `save_df_to_csv(df, file_path)`: Saves a DataFrame to a CSV file.
    * `load_data_from_s3(bucket, key, local_path, encoding='ISO-8859-1')`: (If S3 integration is desired, based on notebook snippets) Downloads from S3 and loads into a DataFrame.
    * `save_data_to_s3(df, bucket, key)`: (If S3 integration is desired) Saves a DataFrame to S3.
    * `compute_kmeans_metrics(scaled_data, k, random_state=42)`: As seen in the customer segmentation notebook.

### 3.3. `data_preprocessing.py`

* **Purpose**: To load the raw sales data, clean it, perform necessary transformations, and save the processed data. This corresponds to the `data_preprocessing_original.md`.
* **Input**: Raw `SalesData.csv`.
* **Output**: `data/processed/aggregated_df.csv` and optionally `data/processed/no_customer_name_agg_df.csv`.
* **Key Steps**:
    1.  **Load Data**: Use `utils.load_csv_data()` or S3 loading function.
    2.  **Initial Inspection**: (Can be omitted in script, but was in notebook) Shape, head, tail.
    3.  **Handle Inconsistencies**:
        * Reconcile `Customer Code` with `Customer Name` (using a predefined mapping or logic from the notebook).
        * Reconcile `Inventory Code` with `Inventory Desc` (similarly).
    4.  **Handle Missing Values**:
        * Impute `Customer Category Desc` (e.g., for 'FAIRY WONDERLAND FENG SHAN PRI SCH' set to 'SCHOOL').
        * Impute `Inventory Desc` and `Inventory Code` (e.g., NaN values set to 'CONTAINER / STOCKS' and generate new codes like 9XXX based on `Price per qty`).
    5.  **Feature Engineering**:
        * Calculate `Price per qty` (`Total Base Amt` / `Qty`). Round to 2 decimal places.
    6.  **Aggregate Data**:
        * Group by `Transaction Date`, `Sales Order No.`, `Customer Code`, `Inventory Code`, `Customer Name`, `Customer Category Desc`, `Inventory Desc`, `Price per qty`.
        * Sum `Qty` and `Total Base Amt`. This handles cases where the same item might appear multiple times in the same order (e.g., due to different batches or data entry issues).
    7.  **Type Casting**:
        * Convert `Transaction Date` to datetime objects (handle mixed formats if necessary, `dayfirst=True`).
        * Convert categorical columns (`Customer Code`, `Inventory Code`, `Customer Name`, `Customer Category Desc`, `Inventory Desc`) to `category` dtype to save memory.
    8.  **Further Feature Engineering (Post-Aggregation)**:
        * Create `Customer Category Broad` based on `Customer Category Desc`.
        * Create `Item Category` based on `Inventory Desc` (e.g., "SP" prefix -> "Raw", "BETAGRO" -> "RTE", else "RTC").
    9.  **Save Processed Data**: Use `utils.save_df_to_csv()`.

### 3.4. `customer_segmentation.py`

* **Purpose**: To segment customers based on RFM analysis and K-Means clustering. This corresponds to `customer_segmentation_combined_original.md`.
* **Input**: `data/processed/aggregated_df.csv`.
* **Output**: `data/segmentation/customer_segmentation_df.csv`.
* **Key Steps**:
    1.  **Load Processed Data**: Load `aggregated_df.csv`.
    2.  **RFM Calculation**:
        * **Recency**: Calculate days since last transaction for each customer. Snapshot date is max transaction date + 1 day.
        * **Frequency**: Calculate the number of unique sales orders for each customer.
        * **Monetary**: Calculate the sum of `Total Base Amt` for each customer.
        * Also calculate `Total Quantity` per customer.
        * Merge these into an `rfm_df`.
    3.  **RFM Scoring**:
        * Cut Recency, Frequency, Monetary values into deciles (1-10 scores). Define appropriate labels (e.g., Rlabel: 10 for most recent, Flabel/Mlabel: 10 for highest).
        * Calculate a combined `Score` by summing R, F, M integer scores.
    4.  **RFM Segmentation (Rule-based)**:
        * Define segments based on the combined `Score` (e.g., Champions, Loyal Customers, At Risk, New Customers, Hibernating, Lost).
        * Apply an adjustment logic (e.g., reclassify 'At Risk'/'Hibernating' with high Recency but low F/M as 'New_Customers').
    5.  **K-Means Clustering Preparation**:
        * Select features for clustering from `rfm_df`: `Recency`, `Frequency`, `Monetary`, `Total Quantity`.
        * Scale these features using `RobustScaler`.
    6.  **K-Means Clustering (Iterative Process from Notebook)**:
        * **Run 1**:
            * Determine optimal K (e.g., 3) using Elbow method and Silhouette scores on the full scaled RFM data.
            * Assign cluster labels (`KMeans_Segment`).
            * Identify if one cluster is dominant and needs further sub-clustering (e.g., Cluster 1 in the notebook).
        * **Run 2 (on the dominant cluster from Run 1)**:
            * Take the data points belonging to the dominant cluster.
            * Re-scale these specific data points.
            * Determine optimal K for this subset (e.g., 3).
            * Assign new cluster labels to this subset (offset these labels, e.g., +4, to distinguish from Run 1's other clusters).
        * **Combine Clusters**: Concatenate the sub-clustered data with the other clusters from Run 1.
        * Remap all `KMeans_Segment` labels to be sequential (e.g., 1, 2, 3, 4, 5).
    7.  **Merge and Finalize**:
        * Merge the final K-Means segment labels back with the main RFM dataframe (`rfm_df`).
        * Select relevant columns (`Customer Code`, `Recency`, `Frequency`, `Monetary`, `Total Quantity`, RFM scores, `RFM_Segment`, `KMeans_Segment`).
        * Merge with `Customer Category Desc` and `Item Category` from the preprocessed data (ensure `cust_item_cat` is created by dropping duplicates from preprocessed data on `Customer Code`).
    8.  **Save Segmentation Data**: Output the final `customer_segmentation_df.csv`.

### 3.5. `price_elasticity_modeling.py`

* **Purpose**: To calculate own-price and cross-price elasticities. This corresponds to `price_elasticity_original.md`.
* **Input**: `data/processed/aggregated_df.csv`.
* **Output**: `data/optimization/price_elasticities_calculated.csv` (or similar, containing SKU, elasticities, product pair types, latest prices/qty).
* **Key Steps**:
    1.  **Load Processed Data**.
    2.  **Filter Data**:
        * Focus on specific `Customer Category Desc` (e.g., 'SUPERMARKET', 'RETAIL').
    3.  **Identify Top SKUs**:
        * Group by `Inventory Code`, `Inventory Desc`, `Customer Category Desc`, `Item Category` and sum `Total Base Amt`.
        * For each `Customer Category Desc` and `Item Category` combination, find the top N (e.g., 10) SKUs by `Total Base Amt`.
    4.  **Prepare Data for Elasticity Calculation**:
        * Filter the main dataframe to include only these top SKUs for the selected customer categories.
        * Group by `Transaction Date`, `Inventory Code`, `Inventory Desc`, `Customer Category Desc`, `Item Category`, summing `Qty` and `Total Base Amt`.
        * Recalculate `Price Per Qty`.
        * Ensure `Transaction Date` is datetime.
        * Log transform `Price Per Qty` (`log_price`) and `Qty` (`log_qty`). Handle zeros appropriately before log (e.g., filter out rows where price or qty is <= 0).
        * Create month dummy variables (`Month_2`, `Month_3`, ..., `Month_12`) for seasonality.
    5.  **Own-Price Elasticity Calculation**:
        * Loop through each `Customer Category Desc`.
        * Within each category, loop through each unique `Inventory Code` (SKU).
        * For each SKU:
            * Prepare design matrix `X` (constant, `log_price` of the SKU, month dummies).
            * Prepare target `y` (`log_qty` of the SKU).
            * Run OLS regression: `log_qty ~ const + log_price + Month_dummies`.
            * The coefficient of `log_price` is the own-price elasticity for that SKU.
            * Store SKU, Inventory Desc, Customer Category, Item Category, and Price Elasticity.
    6.  **Cross-Price Elasticity Calculation**:
        * Loop through each `Customer Category Desc`.
        * Loop through each `Item Category` within that customer category.
        * Get all unique SKUs within this Item Category.
        * For every pair of SKUs (SKU A, SKU B) within the same Item Category:
            * Merge data for SKU A and SKU B on `Transaction Date`.
            * If sufficient merged data points exist:
                * Run OLS regression: `log_qty_A ~ const + log_price_B` (controlling for month dummies if desired, though notebook example is simpler).
                * The coefficient of `log_price_B` is the cross-price elasticity of Qty A with respect to Price B.
                * Store Customer Category, Item Category, SKU A, SKU B, and Cross-Price Elasticity.
    7.  **Combine Elasticities and Latest Data**:
        * Merge own-price elasticities with cross-price elasticities.
        * Define a function `get_latest_transaction(df)` to find the most recent `Price per qty` and `Qty` for each SKU.
        * Merge this latest transaction data for both SKU A and SKU B into the combined elasticity dataframe. (This creates `merged_with_sku_ab` in the notebook).
    8.  **Categorize Product Pairs**:
        * Based on `Cross-Price Elasticity`:
            * `< 0`: Complements
            * `> 0`: Substitutes
            * `= 0`: Independent
        * Categorize strength based on magnitude (e.g., `abs(elasticity) > 1.5` is 'close').
        * Create a `Product Pair Type` column.
    9.  **Save Elasticity Data**: Output the dataframe (e.g., `price_elasticities_final_df.csv`).

### 3.6. `revenue_optimization.py`

* **Purpose**: To use Gurobi to find optimal price changes that maximize revenue, considering the calculated elasticities.
* **Input**: The output from `price_elasticity_modeling.py` (which includes elasticities and latest prices/quantities).
* **Output**: `data/optimization/revenue_optimization_results.csv`.
* **Key Steps**:
    1.  **Load Elasticity Data**: Load the combined dataframe from the previous step.
    2.  **Define Optimization Function**:
        * Create a function `optimize_prices_for_category(df_category_specific)` that takes a subset of the elasticity data (e.g., for 'SUPERMARKET' or 'RETAIL').
        * Inside this function:
            * Initialize Gurobi model (`gp.Model()`).
            * Define decision variables: `price_change_a`, `price_change_b` (percentage change, e.g., bounds -0.5 to 0.5).
            * Initialize `total_revenue` expression.
            * Loop through each row (SKU pair) in the input `df_category_specific`:
                * Extract current prices (`sku_a_price`, `sku_b_price`), quantities (`sku_a_qty`, `sku_b_qty`), own-price elasticities (`sku_a_elasticity`, `sku_b_elasticity`), and `cross_price_elasticity`.
                * Formulate `new_price_a = sku_a_price * (1 + price_change_a)`.
                * Formulate `new_price_b = sku_b_price * (1 + price_change_b)`.
                * Formulate `new_qty_a = sku_a_qty * (1 + sku_a_elasticity * price_change_a + cross_price_elasticity_ab * price_change_b)`. (Note: cross-price elasticity needs to be specific to the direction, e.g., effect of B's price on A's quantity). The notebook uses one `cross_price_elasticity` value for both directions in the quantity update formula; this should be clarified or used consistently. The formula in the notebook is:
                    * `new_qty_a = sku_a_qty * (1 + sku_a_elasticity * price_change_a + cross_price_elasticity * price_change_b)`
                    * `new_qty_b = sku_b_qty * (1 + sku_b_elasticity * price_change_b + cross_price_elasticity * price_change_a)`
                * Add constraints: `new_qty_a >= 0`, `new_qty_b >= 0`.
                * Add revenue contributions (`new_price_a * new_qty_a` + `new_price_b * new_qty_b`) to `total_revenue`.
            * Set objective: `model.setObjective(total_revenue, GRB.MAXIMIZE)`.
            * Optimize model: `model.optimize()`.
            * If optimal, extract results (`price_change_a.x`, `price_change_b.x`) and calculate original vs. optimized revenues and quantities for each pair. Store these in a list of dictionaries.
            * Return a DataFrame of these detailed results.
    3.  **Run Optimization**:
        * Split the loaded elasticity data by `Customer Category` (e.g., 'SUPERMARKET', 'RETAIL').
        * Call `optimize_prices_for_category()` for each category.
        * Concatenate the results from all categories.
    4.  **Save Optimization Results**.

### 3.7. `main.py`

* **Purpose**: To orchestrate the execution of the entire pipeline.
* **Key Steps**:
    1.  Import functions from other scripts.
    2.  Call `data_preprocessing.run_preprocessing()`.
    3.  Call `customer_segmentation.run_segmentation()`.
    4.  Call `price_elasticity_modeling.run_elasticity_modeling()`.
    5.  Call `revenue_optimization.run_optimization()`.
    6.  Include print statements or logging to track progress.

## 4. Data Files

* **Input**:
    * `data/raw/SalesData.csv`: The primary raw sales transaction data. (User needs to provide this).
* **Intermediate/Output**:
    * `data/processed/aggregated_df.csv`: Cleaned and aggregated data.
    * `data/segmentation/customer_segmentation_df.csv`: Data with RFM scores and cluster assignments.
    * `data/optimization/price_elasticities_final_df.csv`: Contains SKUs, their own-price and cross-price elasticities, latest prices/quantities, and product pair categorizations.
    * `data/optimization/revenue_optimization_results.csv`: Detailed results from the Gurobi optimization, showing original vs. optimized prices, quantities, and revenues.

## 5. Key Considerations for VSCode/Copilot

* **Step-by-Step Generation**: Use this guide to generate code for each function/section within each script.
* **Modularity**: Encourage Copilot to create small, well-defined functions.
* **Docstrings and Comments**: Ensure Copilot generates clear docstrings for all functions and modules, and comments for complex logic.
* **Error Handling**: Add `try-except` blocks for operations like file I/O and API calls (if any).
* **Configuration**: Emphasize using the `config.py` for file paths and other parameters.
* **Data Validation**: While not explicitly detailed in the notebooks, consider adding basic data validation steps (e.g., checking for expected columns after loading data).
* **Dependencies**: Make sure all necessary libraries (`pandas`, `numpy`, `sklearn`, `statsmodels`, `matplotlib`, `seaborn`, `squarify`, `yellowbrick`, `boto3`, `gurobipy`) are listed in `requirements.txt`. The S3 access (`boto3`) might be specific to the original environment; for local execution, direct CSV reads are simpler. The scripts will be written assuming local CSV files as primary, with S3 as an optional extension if you need it.

This guide should provide a solid foundation for recreating the project in a structured Python environment.
