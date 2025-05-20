"""
Test script to debug streamlit imports
"""
import sys
import os

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from streamlit.utils import st_utils, visualizations
    print("Successfully imported from streamlit.utils")
except ImportError as e:
    print(f"Error importing from streamlit.utils: {e}")
    # Try an alternative import path
    try:
        sys.path.append(os.path.join(project_root, "streamlit"))
        from utils import st_utils, visualizations
        print("Successfully imported using alternative path")
    except ImportError as e:
        print(f"Error with alternative import path: {e}")
        print(f"Current sys.path: {sys.path}")
