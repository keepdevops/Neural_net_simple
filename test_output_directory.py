#!/usr/bin/env python3
"""
Test script for output directory functionality

This script tests that the --output_dir and --output_file arguments work correctly
in the predict.py script.
"""

import os
import sys
import tempfile
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

def create_test_data():
    """Create test stock data for prediction"""
    # Create sample stock data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    data = {
        'date': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(110, 220, 100),
        'low': np.random.uniform(90, 180, 100),
        'close': np.random.uniform(100, 200, 100),
        'vol': np.random.uniform(1000000, 5000000, 100)
    }
    
    df = pd.DataFrame(data)
    return df

def test_output_directory():
    """Test the output directory functionality"""
    print("Testing output directory functionality...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = os.path.join(temp_dir, "test_data")
        test_output_dir = os.path.join(temp_dir, "test_output")
        test_model_dir = os.path.join(temp_dir, "test_model")
        
        os.makedirs(test_data_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        os.makedirs(test_model_dir, exist_ok=True)
        
        # Create test data
        test_data = create_test_data()
        test_data_file = os.path.join(test_data_dir, "test_stock_data.csv")
        test_data.to_csv(test_data_file, index=False)
        print(f"Created test data: {test_data_file}")
        
        # Create a simple model directory structure (mock)
        model_weights = {
            'W1': np.random.randn(4, 4),
            'b1': np.random.randn(1, 4),
            'W2': np.random.randn(4, 1),
            'b2': np.random.randn(1, 1),
            'X_min': np.array([90, 90, 90, 1000000]),
            'X_max': np.array([220, 220, 220, 5000000]),
            'Y_min': np.array([100]),
            'Y_max': np.array([200]),
            'has_target_norm': True,
            'input_size': 4,
            'hidden_size': 4
        }
        
        # Save model weights
        np.savez(os.path.join(test_model_dir, 'stock_model.npz'), **model_weights)
        
        # Save feature info
        feature_info = {
            'x_features': ['open', 'high', 'low', 'vol'],
            'y_feature': 'close',
            'input_size': 4
        }
        
        import json
        with open(os.path.join(test_model_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f)
        
        print(f"Created test model: {test_model_dir}")
        
        # Test 1: Basic output directory functionality
        print("\nTest 1: Basic output directory functionality")
        cmd = [
            sys.executable, "predict.py",
            test_data_file,
            "--model_dir", test_model_dir,
            "--x_features", "open,high,low,vol",
            "--y_feature", "close",
            "--output_dir", test_output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            print(f"Command output: {result.stdout}")
            if result.stderr:
                print(f"Command errors: {result.stderr}")
            
            # Check if files were created in output directory
            output_files = os.listdir(test_output_dir)
            print(f"Files in output directory: {output_files}")
            
            # Check for predictions file
            pred_files = [f for f in output_files if f.startswith('predictions_') and f.endswith('.csv')]
            if pred_files:
                print(f"✅ Predictions file created: {pred_files[0]}")
            else:
                print("❌ No predictions file found")
            
            # Check for plots directory
            plots_dir = os.path.join(test_output_dir, 'plots')
            if os.path.exists(plots_dir):
                plot_files = os.listdir(plots_dir)
                print(f"✅ Plots directory created with {len(plot_files)} files")
            else:
                print("❌ Plots directory not found")
                
        except Exception as e:
            print(f"❌ Test 1 failed: {e}")
        
        # Test 2: Custom output filename
        print("\nTest 2: Custom output filename")
        custom_output_file = "my_predictions.csv"
        cmd2 = [
            sys.executable, "predict.py",
            test_data_file,
            "--model_dir", test_model_dir,
            "--x_features", "open,high,low,vol",
            "--y_feature", "close",
            "--output_dir", test_output_dir,
            "--output_file", custom_output_file
        ]
        
        try:
            result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            print(f"Command output: {result2.stdout}")
            if result2.stderr:
                print(f"Command errors: {result2.stderr}")
            
            # Check if custom file was created
            custom_file_path = os.path.join(test_output_dir, custom_output_file)
            if os.path.exists(custom_file_path):
                print(f"✅ Custom output file created: {custom_file_path}")
            else:
                print("❌ Custom output file not found")
                
        except Exception as e:
            print(f"❌ Test 2 failed: {e}")
        
        # Test 3: Non-existent output directory (should be created)
        print("\nTest 3: Non-existent output directory")
        new_output_dir = os.path.join(temp_dir, "new_output_dir")
        cmd3 = [
            sys.executable, "predict.py",
            test_data_file,
            "--model_dir", test_model_dir,
            "--x_features", "open,high,low,vol",
            "--y_feature", "close",
            "--output_dir", new_output_dir
        ]
        
        try:
            result3 = subprocess.run(cmd3, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
            print(f"Command output: {result3.stdout}")
            if result3.stderr:
                print(f"Command errors: {result3.stderr}")
            
            # Check if directory was created
            if os.path.exists(new_output_dir):
                print(f"✅ New output directory created: {new_output_dir}")
                new_files = os.listdir(new_output_dir)
                print(f"Files in new directory: {new_files}")
            else:
                print("❌ New output directory not created")
                
        except Exception as e:
            print(f"❌ Test 3 failed: {e}")

if __name__ == "__main__":
    test_output_directory()
    print("\nOutput directory testing completed!") 