# main.py
"""
Main script to run the entire CS Tay Customer Segmentation and Price Optimization pipeline.
"""
import logging

# Import project-specific modules
import config # To ensure paths are initialized and accessible
import data_preprocessing
import customer_segmentation
import price_elasticity_modeling
import revenue_optimization

# Configure logger
# Basic configuration, individual modules might have more specific logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__) # Use __name__ for the logger of this main script

def run_pipeline():
    """
    Executes the full data analysis pipeline.
    """
    logger.info("Starting CS Tay Data Analysis Pipeline...")

    # --- Step 1: Data Preprocessing ---
    logger.info("-" * 50)
    logger.info("STEP 1: Running Data Preprocessing...")
    try:
        # Ensure raw data file exists before running
        if not config.RAW_SALES_DATA_PATH.exists():
            logger.error(f"Raw sales data file NOT FOUND at: {config.RAW_SALES_DATA_PATH}")
            logger.error("Please ensure 'SalesData.csv' (or your configured raw data file) is in the 'data/raw/' directory.")
            logger.error("Pipeline halted.")
            return
        
        processed_df = data_preprocessing.run_preprocessing()
        if processed_df is None or processed_df.empty:
            logger.error("Data preprocessing failed or returned no data. Pipeline halted.")
            return
        logger.info("Data Preprocessing COMPLETED.")
    except Exception as e:
        logger.error(f"Error during Data Preprocessing: {e}", exc_info=True)
        logger.error("Pipeline halted due to error in preprocessing.")
        return

    # --- Step 2: Customer Segmentation ---
    logger.info("-" * 50)
    logger.info("STEP 2: Running Customer Segmentation...")
    try:
        segmentation_df = customer_segmentation.run_segmentation()
        if segmentation_df is None or segmentation_df.empty:
            logger.error("Customer segmentation failed or returned no data. Pipeline halted.")
            return
        logger.info("Customer Segmentation COMPLETED.")
    except Exception as e:
        logger.error(f"Error during Customer Segmentation: {e}", exc_info=True)
        logger.error("Pipeline halted due to error in segmentation.")
        return

    # --- Step 3: Price Elasticity Modeling ---
    logger.info("-" * 50)
    logger.info("STEP 3: Running Price Elasticity Modeling...")
    try:
        elasticity_df = price_elasticity_modeling.run_elasticity_modeling()
        if elasticity_df is None or elasticity_df.empty:
            logger.error("Price elasticity modeling failed or returned no data. Pipeline halted.")
            return
        logger.info("Price Elasticity Modeling COMPLETED.")
    except Exception as e:
        logger.error(f"Error during Price Elasticity Modeling: {e}", exc_info=True)
        logger.error("Pipeline halted due to error in elasticity modeling.")
        return

    # --- Step 4: Revenue Optimization ---
    logger.info("-" * 50)
    logger.info("STEP 4: Running Revenue Optimization...")
    try:
        optimization_results_df = revenue_optimization.run_optimization()
        if optimization_results_df is None or optimization_results_df.empty:
            logger.warning("Revenue optimization failed or returned no results. Check logs for details.")
            # Not halting pipeline here as previous steps might still be useful.
        else:
            logger.info("Revenue Optimization COMPLETED.")
    except ImportError as e:
        if 'gurobipy' in str(e).lower():
            logger.error("GurobiPy not found. Revenue optimization step cannot be run.")
            logger.error("Please install Gurobi and its Python interface if you wish to run optimization.")
        else:
            logger.error(f"ImportError during Revenue Optimization: {e}", exc_info=True)
        logger.warning("Revenue Optimization SKIPPED due to import error.")
    except Exception as e:
        logger.error(f"Error during Revenue Optimization: {e}", exc_info=True)
        logger.warning("Revenue Optimization SKIPPED due to an error.")


    logger.info("-" * 50)
    logger.info("CS Tay Data Analysis Pipeline execution finished.")
    logger.info(f"Outputs can be found in:")
    logger.info(f"  - Processed data: {config.PROCESSED_DATA_DIR}")
    logger.info(f"  - Segmentation data: {config.SEGMENTATION_OUTPUT_DIR}")
    logger.info(f"  - Optimization/Elasticity data: {config.OPTIMIZATION_OUTPUT_DIR}")

if __name__ == '__main__':
    # This structure assumes that config.py, utils.py and the step-specific
    # scripts (data_preprocessing.py etc.) are in a subdirectory named 'scripts'
    # relative to where main.py is.
    # If main.py is in the root, and scripts are in 'scripts/', then imports would be:
    # from scripts import config
    # from scripts import utils
    # etc.
    # The current structure (main.py in scripts/ alongside others) simplifies imports.
    
    # If you move main.py to the project root (cs_tay_project/), you'd adjust imports:
    # from scripts import config, utils, data_preprocessing, ...
    
    run_pipeline()
