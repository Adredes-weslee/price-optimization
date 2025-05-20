"""
Test script to check imports
"""
import sys
import os

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    from src import config
    print("Config imported successfully from src package")
    print(f"Config module path: {config.__file__}")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
