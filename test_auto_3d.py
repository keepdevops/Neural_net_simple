#!/usr/bin/env python3
"""
Test script to verify automatic 3D visualization generation
"""

import os
import glob
import json
from datetime import datetime

def test_auto_3d_visualization():
    """Test the automatic 3D visualization generation functionality."""
    print("Testing automatic 3D visualization generation...")
    
    # Find model directories
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        return False
    
    # Use the most recent model directory
    model_dirs.sort(key=os.path.getctime, reverse=True)
    model_dir = model_dirs[0]
    
    print(f"Testing with model: {model_dir}")
    
    # Check if model has training metadata
    metadata_file = os.path.join(model_dir, 'training_metadata.json')
    if not os.path.exists(metadata_file):
        print("No training metadata found - model may not be fully trained")
        return False
    
    # Load training metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Training completed at: {metadata.get('training_completed_at', 'Unknown')}")
    print(f"Total epochs: {metadata.get('total_epochs', 'Unknown')}")
    print(f"Final train loss: {metadata.get('final_train_loss', 'Unknown')}")
    
    # Check if plots directory exists
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        print("No plots directory found")
        return False
    
    # Check for loss curve
    loss_curve = os.path.join(plots_dir, 'loss_curve.png')
    if os.path.exists(loss_curve):
        print("✅ Loss curve found")
    else:
        print("❌ Loss curve not found")
    
    # Check for 3D visualization files
    gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
    if gd3d_files:
        print(f"✅ Found {len(gd3d_files)} 3D visualization files")
        print(f"   Sample files: {[os.path.basename(f) for f in gd3d_files[:3]]}")
        return True
    else:
        # Fallback check
        gd3d_files = glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png'))
        if gd3d_files:
            print(f"✅ Found {len(gd3d_files)} 3D visualization files (fallback pattern)")
            print(f"   Sample files: {[os.path.basename(f) for f in gd3d_files[:3]]}")
            return True
        else:
            print("❌ No 3D visualization files found")
            return False

def test_model_selection_3d():
    """Test 3D visualization loading when model is selected."""
    print("\nTesting model selection 3D visualization...")
    
    # Find model directories
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        return False
    
    # Use the most recent model directory
    model_dirs.sort(key=os.path.getctime, reverse=True)
    model_dir = model_dirs[0]
    
    # Check if model has 3D visualization
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        print("No plots directory found")
        return False
    
    # Check for 3D visualization files
    gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
    if not gd3d_files:
        gd3d_files = glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png'))
    
    if gd3d_files:
        print(f"✅ Model has 3D visualization: {len(gd3d_files)} files")
        
        # Select the best frame (middle or last)
        if len(gd3d_files) > 10:
            selected_file = gd3d_files[len(gd3d_files) // 2]
        else:
            selected_file = gd3d_files[-1]
        
        print(f"   Selected frame: {os.path.basename(selected_file)}")
        return True
    else:
        print("❌ Model does not have 3D visualization")
        return False

if __name__ == "__main__":
    print("=== Testing Automatic 3D Visualization Generation ===\n")
    
    # Test 1: Check if models have 3D visualizations
    success1 = test_auto_3d_visualization()
    
    # Test 2: Check model selection 3D loading
    success2 = test_model_selection_3d()
    
    print(f"\n=== Test Results ===")
    print(f"Auto 3D generation: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Model selection 3D: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! 3D visualization functionality is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.") 