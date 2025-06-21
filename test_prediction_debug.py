#!/usr/bin/env python3
"""
Debug script to test prediction functionality
"""

import subprocess
import sys
import os
import json

def main():
    print("🔍 Debugging prediction functionality...")
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
    
    # Load feature info
    feature_info_path = os.path.join(latest_model, 'feature_info.json')
    if os.path.exists(feature_info_path):
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
        print(f"📊 Model features: {feature_info}")
    else:
        print("❌ No feature_info.json found!")
        return
    
    # Test the current GUI prediction command (missing features)
    print(f"\n❌ Current GUI command (WRONG - missing features):")
    cmd_wrong = [
        sys.executable, 'predict.py',
        'enhanced_stock_data.csv',
        '--model_dir', latest_model,
        '--output_dir', latest_model
    ]
    print(" ".join(cmd_wrong))
    
    # Test the correct command with features
    print(f"\n✅ Correct command (with features):")
    cmd_correct = [
        sys.executable, 'predict.py',
        'enhanced_stock_data.csv',
        '--model_dir', latest_model,
        '--output_dir', latest_model,
        '--x_features', ','.join(feature_info['x_features']),
        '--y_feature', feature_info['y_feature']
    ]
    print(" ".join(cmd_correct))
    
    # Run the correct command
    print(f"\n🚀 Running correct prediction command...")
    try:
        result = subprocess.run(cmd_correct, capture_output=True, text=True)
        
        print(f"📊 Return code: {result.returncode}")
        print(f"📤 STDOUT:")
        print(result.stdout)
        print(f"📥 STDERR:")
        print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Prediction successful!")
            
            # Check for created files
            print(f"\n📁 Checking for created files:")
            model_files = os.listdir(latest_model)
            prediction_files = [f for f in model_files if 'prediction' in f.lower() and f.endswith('.csv')]
            
            if prediction_files:
                print(f"✅ Found {len(prediction_files)} prediction files:")
                for file in prediction_files:
                    print(f"  - {file}")
            else:
                print("❌ No prediction CSV files found!")
                
        else:
            print(f"\n❌ Prediction failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 