# utils.py
"""
Utility functions for the CS Tay Customer Segmentation and Price Optimization project.
Includes functions for data loading, saving, and other common tasks.
"""
import pandas as pd
import logging
from pathlib import Path

# Import configuration
from . import config

# Configure logger
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def load_csv_data(file_path: Path, encoding: str = config.DEFAULT_ENCODING, **kwargs) -> pd.DataFrame | None:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the CSV file.
        encoding (str): The encoding to use for reading the file.
        **kwargs: Additional keyword arguments to pass to pd.read_csv().

    Returns:
        pd.DataFrame | None: Loaded DataFrame, or None if an error occurs.
    """
    try:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            print(f"Error: File not found at {file_path}. Please ensure the file exists.")
            return None
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print(f"Error: File not found at {file_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        print(f"An error occurred while loading {file_path}: {e}")
        return None

def save_df_to_csv(df: pd.DataFrame, file_path: Path, index: bool = False, **kwargs) -> bool:
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (Path): The path where the CSV file will be saved.
        index (bool): Whether to write the DataFrame index as a column.
        **kwargs: Additional keyword arguments to pass to df.to_csv().

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=index, **kwargs)
        logger.info(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        print(f"An error occurred while saving data to {file_path}: {e}")
        return False


if __name__ == '__main__':
    # Example usage (primarily for testing the functions)
    logger.info("Testing utility functions...")

    # Create a dummy DataFrame for testing
    test_data = {'col1': [1, 2], 'col2': ['a', 'b']}
    test_df = pd.DataFrame(test_data)
    
    # Test saving to CSV
    test_csv_path = config.PROCESSED_DATA_DIR / "test_utils_output.csv"
    if save_df_to_csv(test_df, test_csv_path):
        logger.info(f"Test CSV saved to {test_csv_path}")
        
        # Test loading from CSV
        loaded_df = load_csv_data(test_csv_path)
        if loaded_df is not None:
            logger.info("Test CSV loaded successfully.")
            print("Loaded DataFrame:\n", loaded_df)
            # Clean up the test file
            try:
                test_csv_path.unlink()
                logger.info(f"Cleaned up test file: {test_csv_path}")
            except OSError as e:
                logger.error(f"Error deleting test file {test_csv_path}: {e}")
        else:
            logger.error("Failed to load test CSV.")
    else:
        logger.error("Failed to save test CSV.")

    logger.info("Utility functions test complete.")