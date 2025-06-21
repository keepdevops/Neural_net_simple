#!/usr/bin/env python3
"""
Test script to trace where prediction files are being saved
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    print("Tracing prediction file locations...")
    print()
    
    # Change to the simple directory
    os.chdir('/Users/porupine/neural_net/simple')
    
    # Get the most recent model
    model_dirs = [d for d in os.listdir('.') if d.startswith('model_')]
    if not model_dirs:
        print("❌ No model directories found!")
        return
    
    latest_model = max(model_dirs, key=os.path.getctime)
    print(f"📁 Using model: {latest_model}")
    
    # Check what's in the model directory before prediction
    print(f"\n📂 Model directory contents BEFORE prediction:")
    model_files = os.listdir(latest_model)
    for file in model_files:
        print(f"  - {file}")
    
    # Run prediction command
    cmd = [
        sys.executable, 'predict.py',
        'enhanced_stock_data.csv',
        '--model_dir', latest_model,
        '--output_dir', latest_model,
        '--x_features', 'open,high,low,vol',
        '--y_feature', 'close'
    ]
    
    print(f"\nRunning prediction command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"📊 Prediction return code: {result.returncode}")
        print(f"📤 STDOUT:")
        print(result.stdout)
        print(f"📥 STDERR:")
        print(result.stderr)
        
        # Check what's in the model directory after prediction
        print(f"\n📂 Model directory contents AFTER prediction:")
        model_files_after = os.listdir(latest_model)
        for file in model_files_after:
            print(f"  - {file}")
        
        # Check for prediction files specifically
        print(f"\n🔍 Looking for prediction files:")
        prediction_files = [f for f in model_files_after if 'prediction' in f.lower() and f.endswith('.csv')]
        if prediction_files:
            print(f"✅ Found {len(prediction_files)} prediction CSV files:")
            for file in prediction_files:
                print(f"  - {file}")
        else:
            print("❌ No prediction CSV files found!")
        
        # Check plots directory
        plots_dir = os.path.join(latest_model, 'plots')
        if os.path.exists(plots_dir):
            print(f"\n📊 Plots directory contents:")
            plot_files = os.listdir(plots_dir)
            for file in plot_files:
                print(f"  - {file}")
        
        # Check simple directory for any prediction files
        print(f"\n🔍 Checking simple directory for prediction files:")
        simple_files = [f for f in os.listdir('.') if 'prediction' in f.lower() and f.endswith('.csv')]
        if simple_files:
            print(f"✅ Found {len(simple_files)} prediction CSV files in simple directory:")
            for file in simple_files:
                print(f"  - {file}")
        else:
            print("❌ No prediction CSV files found in simple directory!")
            
    except Exception as e:
        print(f"❌ Error running prediction: {e}")

if __name__ == "__main__":
    main() 