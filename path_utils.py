#!/usr/bin/env python3
"""
Path Resolution Utilities for Stock Prediction GUI

This module provides robust path resolution for finding scripts and files
regardless of where the GUI is run from.
"""

import os
import sys
import glob
from pathlib import Path

def find_script(script_name, search_paths=None):
    """
    Find a script file using multiple search strategies.
    
    Args:
        script_name (str): Name of the script to find (e.g., 'predict.py')
        search_paths (list): Additional paths to search (optional)
    
    Returns:
        str: Full path to the script if found, None otherwise
    """
    if search_paths is None:
        search_paths = []
    
    # Define search locations in order of preference
    possible_paths = []
    
    # 1. Same directory as the calling script (GUI)
    if '__file__' in globals():
        calling_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(calling_dir, script_name))
    
    # 2. Current working directory
    possible_paths.append(os.path.join(os.getcwd(), script_name))
    
    # 3. Simple subdirectory (for when GUI is run from parent directory)
    possible_paths.append(os.path.join('simple', script_name))
    
    # 4. Parent directory
    possible_paths.append(os.path.join(os.path.dirname(os.getcwd()), script_name))
    
    # 5. Additional search paths
    possible_paths.extend([os.path.join(path, script_name) for path in search_paths])
    
    # 6. Just the filename (if in PATH)
    possible_paths.append(script_name)
    
    # Search for the script
    for path in possible_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return os.path.abspath(path)
    
    return None

def find_model_directory(model_name=None, search_paths=None):
    """
    Find a model directory.
    
    Args:
        model_name (str): Specific model name (e.g., 'model_20250619_092747')
        search_paths (list): Additional paths to search (optional)
    
    Returns:
        str: Full path to the model directory if found, None otherwise
    """
    if search_paths is None:
        search_paths = []
    
    # Define search locations
    possible_paths = []
    
    # 1. Current directory
    possible_paths.append('.')
    
    # 2. Simple subdirectory
    possible_paths.append('simple')
    
    # 3. Models subdirectory
    possible_paths.append('models')
    
    # 4. Additional search paths
    possible_paths.extend(search_paths)
    
    # Search for model directories
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
            
        if model_name:
            # Look for specific model
            model_path = os.path.join(base_path, model_name)
            if os.path.exists(model_path) and os.path.isdir(model_path):
                return os.path.abspath(model_path)
        else:
            # Look for any model directory
            model_pattern = os.path.join(base_path, 'model_*')
            model_dirs = glob.glob(model_pattern)
            if model_dirs:
                # Return the most recent model
                latest_model = max(model_dirs, key=os.path.getctime)
                return os.path.abspath(latest_model)
    
    return None

def find_data_file(filename, search_paths=None):
    """
    Find a data file.
    
    Args:
        filename (str): Name of the data file (e.g., 'tsla_combined.csv')
        search_paths (list): Additional paths to search (optional)
    
    Returns:
        str: Full path to the data file if found, None otherwise
    """
    if search_paths is None:
        search_paths = []
    
    # Define search locations
    possible_paths = []
    
    # 1. Current directory
    possible_paths.append('.')
    
    # 2. Data subdirectories
    possible_paths.extend(['data', 'input', 'raw_data', 'datasets'])
    
    # 3. Simple subdirectory
    possible_paths.append('simple')
    
    # 4. Additional search paths
    possible_paths.extend(search_paths)
    
    # Search for the file
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
            
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return os.path.abspath(file_path)
    
    return None

def get_working_directory():
    """
    Get the appropriate working directory for running scripts.
    
    Returns:
        str: Path to the directory where scripts should be run from
    """
    # Try to find the simple directory
    simple_dir = find_script('stock_gui.py')
    if simple_dir:
        return os.path.dirname(simple_dir)
    
    # Fallback to current directory
    return os.getcwd()

def validate_script_execution(script_path, test_args=None):
    """
    Validate that a script can be executed.
    
    Args:
        script_path (str): Path to the script
        test_args (list): Arguments to test with (e.g., ['--help'])
    
    Returns:
        bool: True if script can be executed, False otherwise
    """
    if not script_path or not os.path.exists(script_path):
        return False
    
    try:
        if test_args is None:
            test_args = ['--help']
        
        result = subprocess.run([sys.executable, script_path] + test_args,
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False

def create_script_command(script_path, args=None, working_dir=None):
    """
    Create a command to run a script.
    
    Args:
        script_path (str): Path to the script
        args (list): Arguments to pass to the script
        working_dir (str): Working directory to run from
    
    Returns:
        tuple: (command_list, working_directory)
    """
    if args is None:
        args = []
    
    if working_dir is None:
        working_dir = get_working_directory()
    
    command = [sys.executable, script_path] + args
    
    return command, working_dir

# Example usage and testing
if __name__ == "__main__":
    print("Testing path resolution utilities...")
    
    # Test script finding
    scripts_to_test = ['predict.py', 'gradient_descent_3d.py', 'view_results.py']
    for script in scripts_to_test:
        path = find_script(script)
        if path:
            print(f"✅ Found {script}: {path}")
        else:
            print(f"❌ Could not find {script}")
    
    # Test model directory finding
    model_dir = find_model_directory()
    if model_dir:
        print(f"✅ Found model directory: {model_dir}")
    else:
        print("❌ Could not find model directory")
    
    # Test data file finding
    data_files = ['tsla_combined.csv', 'data.csv']
    for data_file in data_files:
        path = find_data_file(data_file)
        if path:
            print(f"✅ Found {data_file}: {path}")
        else:
            print(f"❌ Could not find {data_file}")
    
    print(f"Working directory: {get_working_directory()}") 