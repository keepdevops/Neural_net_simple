#!/usr/bin/env python3
"""
Integration Test Script for PNG Plot Display System

This script tests the complete integration between:
- stock_gui.py (GUI with Saved Plots tab)
- predict.py (prediction with PNG saving)
- gradient_descent_3d.py (3D visualization with PNG saving)
- view_results.py (comprehensive plot viewing)

Usage:
    python test_integration.py
"""

import os
import sys
import subprocess
import glob
import time

def test_script_exists(script_name):
    """Test if a script exists and is executable."""
    possible_paths = [
        script_name,
        os.path.join(os.path.dirname(__file__), script_name),
        os.path.join(os.getcwd(), script_name)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found {script_name} at: {path}")
            return path
    
    print(f"❌ Could not find {script_name}")
    return None

def test_model_directory():
    """Test if there's a valid model directory."""
    model_dirs = glob.glob("model_*")
    if model_dirs:
        latest_model = max(model_dirs, key=os.path.getctime)
        print(f"✅ Found model directory: {latest_model}")
        
        # Check for required files
        required_files = ['feature_info.json', 'training_losses.csv']
        for file in required_files:
            file_path = os.path.join(latest_model, file)
            if os.path.exists(file_path):
                print(f"  ✅ Found {file}")
            else:
                print(f"  ❌ Missing {file}")
        
        return latest_model
    else:
        print("❌ No model directories found")
        return None

def test_plots_directory(model_dir):
    """Test if plots directory exists and contains PNG files."""
    plots_dir = os.path.join(model_dir, 'plots')
    if os.path.exists(plots_dir):
        print(f"✅ Found plots directory: {plots_dir}")
        
        png_files = glob.glob(os.path.join(plots_dir, '*.png'))
        if png_files:
            print(f"  ✅ Found {len(png_files)} PNG files:")
            for png_file in png_files:
                print(f"    - {os.path.basename(png_file)}")
        else:
            print("  ⚠️  No PNG files found in plots directory")
        
        return plots_dir
    else:
        print(f"⚠️  No plots directory found in {model_dir}")
        return None

def test_gradient_descent_script():
    """Test the gradient descent script."""
    script_path = test_script_exists('gradient_descent_3d.py')
    if not script_path:
        return False
    
    try:
        # Test help command
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and '--save_png' in result.stdout:
            print("✅ Gradient descent script works and supports --save_png")
            return True
        else:
            print("❌ Gradient descent script help test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing gradient descent script: {e}")
        return False

def test_predict_script():
    """Test the predict script."""
    script_path = test_script_exists('predict.py')
    if not script_path:
        return False
    
    try:
        # Test help command
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and '--x_features' in result.stdout:
            print("✅ Predict script works and supports feature selection")
            return True
        else:
            print("❌ Predict script help test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing predict script: {e}")
        return False

def test_view_results_script():
    """Test the view results script."""
    script_path = test_script_exists('view_results.py')
    if not script_path:
        return False
    
    try:
        # Test help command
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and '--plot-type' in result.stdout:
            print("✅ View results script works and supports plot types")
            return True
        else:
            print("❌ View results script help test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing view results script: {e}")
        return False

def test_gui_import():
    """Test if the GUI can be imported."""
    try:
        import stock_gui
        print("✅ GUI module imports successfully")
        
        # Test if required methods exist
        required_methods = ['load_saved_plots', 'trigger_gradient_descent_visualization']
        for method in required_methods:
            if hasattr(stock_gui.StockPredictionGUI, method):
                print(f"  ✅ Found method: {method}")
            else:
                print(f"  ❌ Missing method: {method}")
        
        return True
    except Exception as e:
        print(f"❌ Error importing GUI: {e}")
        return False

def test_pil_import():
    """Test if PIL/Pillow is available."""
    try:
        from PIL import Image, ImageTk
        print("✅ PIL/Pillow imports successfully")
        return True
    except ImportError as e:
        print(f"❌ PIL/Pillow not available: {e}")
        print("   Install with: pip install Pillow")
        return False

def main():
    """Run all integration tests."""
    print("🔍 Running PNG Plot Integration Tests")
    print("=" * 50)
    
    # Test basic requirements
    print("\n📋 Testing Basic Requirements:")
    pil_ok = test_pil_import()
    gui_ok = test_gui_import()
    
    # Test scripts
    print("\n📜 Testing Scripts:")
    gd_ok = test_gradient_descent_script()
    pred_ok = test_predict_script()
    view_ok = test_view_results_script()
    
    # Test model directory
    print("\n📁 Testing Model Directory:")
    model_dir = test_model_directory()
    plots_dir = None
    if model_dir:
        plots_dir = test_plots_directory(model_dir)
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    tests = [
        ("PIL/Pillow", pil_ok),
        ("GUI Import", gui_ok),
        ("Gradient Descent Script", gd_ok),
        ("Predict Script", pred_ok),
        ("View Results Script", view_ok),
        ("Model Directory", model_dir is not None),
        ("Plots Directory", plots_dir is not None)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The PNG plot integration is ready to use.")
        print("\n📖 Usage Instructions:")
        print("1. Run the GUI: python stock_gui.py")
        print("2. Select a model from the Model tab")
        print("3. Switch to the 'Saved Plots' tab to view PNG plots")
        print("4. Use 'Create 3D Visualization' to generate gradient descent plots")
        print("5. Use 'Make Prediction' to generate prediction plots")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before using the system.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 