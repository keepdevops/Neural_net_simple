#!/usr/bin/env python3
"""
Test script to create prediction files
"""

import subprocess
import sys
import os

def main():
    print("🚀 Creating prediction files for GUI testing...")
    print()
    
    # Change to the simple directory
    os.chdir('/Users/porupine/neural_net/simple')
    
    # Run prediction command
    cmd = [
        sys.executable, 'predict.py',
        'enhanced_stock_data.csv',
        '--model_dir', 'model_20250620_153411',
        '--output_dir', 'model_20250620_153411',
        '--x_features', 'open,high,low,vol',
        '--y_feature', 'close'
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Prediction completed successfully!")
            print()
            print("Output:")
            print(result.stdout)
            
            # Check what files were created
            model_dir = 'model_20250620_153411'
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                prediction_files = [f for f in files if f.endswith('.csv') and any(pattern in f.lower() for pattern in ['prediction', 'pred', 'result', 'output'])]
                
                if prediction_files:
                    print(f"\nCreated prediction files:")
                    for file in prediction_files:
                        file_path = os.path.join(model_dir, file)
                        file_size = os.path.getsize(file_path)
                        print(f"  - {file} ({file_size} bytes)")
                else:
                    print("\n⚠️  No prediction files found in output")
                    
                # Also check plots directory
                plots_dir = os.path.join(model_dir, 'plots')
                if os.path.exists(plots_dir):
                    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png') and 'predicted' in f.lower()]
                    if plot_files:
                        print(f"\nCreated prediction plots:")
                        for file in plot_files:
                            file_path = os.path.join(plots_dir, file)
                            file_size = os.path.getsize(file_path)
                            print(f"  - {file} ({file_size} bytes)")
        else:
            print("❌ Prediction failed!")
            print("Error output:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error running prediction: {e}")

if __name__ == "__main__":
    main() 