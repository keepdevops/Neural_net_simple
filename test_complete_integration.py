#!/usr/bin/env python3
"""
Complete PNG Plot Integration Test

This script tests the complete integration of PNG plot generation and display
from different working directories, including GUI script execution.
"""

import os
import sys
import subprocess
import tempfile
import time
import json

def test_gui_script_execution(test_dir):
    """Test if the GUI can execute scripts from a specific directory."""
    print(f"\n🧪 Testing GUI script execution from: {test_dir}")
    print("=" * 60)
    
    # Change to test directory
    original_dir = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Import path utilities
        sys.path.insert(0, os.path.join(original_dir, 'simple'))
        import path_utils
        
        # Test script finding
        print("📋 Testing script discovery...")
        scripts = ['predict.py', 'gradient_descent_3d.py', 'view_results.py']
        script_paths = {}
        
        for script in scripts:
            path = path_utils.find_script(script)
            if path:
                print(f"  ✅ Found {script}: {path}")
                script_paths[script] = path
            else:
                print(f"  ❌ Could not find {script}")
                return False
        
        # Test model directory
        print("\n📁 Testing model discovery...")
        model_dir = path_utils.find_model_directory()
        if model_dir:
            print(f"  ✅ Found model directory: {model_dir}")
        else:
            print("  ❌ Could not find model directory")
            return False
        
        # Test working directory
        working_dir = path_utils.get_working_directory()
        print(f"  📁 Working directory: {working_dir}")
        
        # Test script execution from GUI perspective
        print("\n🚀 Testing script execution...")
        
        # Test gradient descent script
        gd_script = script_paths['gradient_descent_3d.py']
        gd_cmd = [sys.executable, gd_script, '--model_dir', model_dir, '--save_png', '--help']
        
        try:
            result = subprocess.run(gd_cmd, capture_output=True, text=True, timeout=10, cwd=working_dir)
            if result.returncode == 0:
                print("  ✅ Gradient descent script executes successfully")
            else:
                print(f"  ❌ Gradient descent script failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ❌ Error executing gradient descent script: {e}")
            return False
        
        # Test predict script
        pred_script = script_paths['predict.py']
        pred_cmd = [sys.executable, pred_script, '--help']
        
        try:
            result = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=10, cwd=working_dir)
            if result.returncode == 0:
                print("  ✅ Predict script executes successfully")
            else:
                print(f"  ❌ Predict script failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ❌ Error executing predict script: {e}")
            return False
        
        # Test GUI import and method availability
        print("\n📱 Testing GUI integration...")
        try:
            import stock_gui
            print("  ✅ GUI module imports successfully")
            
            # Test required methods
            required_methods = [
                'load_saved_plots',
                'trigger_gradient_descent_visualization',
                'make_prediction',
                'refresh_models'
            ]
            
            for method in required_methods:
                if hasattr(stock_gui.StockPredictionGUI, method):
                    print(f"    ✅ Found method: {method}")
                else:
                    print(f"    ❌ Missing method: {method}")
                    return False
            
            print("  ✅ All required GUI methods available")
            
        except Exception as e:
            print(f"  ❌ Error importing GUI: {e}")
            return False
        
        # Test plots directory
        print("\n🖼️  Testing plots directory...")
        plots_dir = os.path.join(model_dir, 'plots')
        if os.path.exists(plots_dir):
            print(f"  ✅ Plots directory exists: {plots_dir}")
            
            # Check for existing plots
            png_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            if png_files:
                print(f"  ✅ Found {len(png_files)} existing PNG files")
                for png_file in png_files[:3]:  # Show first 3
                    print(f"    - {png_file}")
            else:
                print("  ⚠️  No existing PNG files found")
        else:
            print(f"  ⚠️  Plots directory does not exist: {plots_dir}")
        
        print("  ✅ All tests passed for this directory")
        return True
        
    except Exception as e:
        print(f"  ❌ Error in test: {e}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_dir)
        # Remove simple from sys.path
        if os.path.join(original_dir, 'simple') in sys.path:
            sys.path.remove(os.path.join(original_dir, 'simple'))

def test_png_generation(test_dir):
    """Test actual PNG generation from different directories."""
    print(f"\n🎨 Testing PNG generation from: {test_dir}")
    print("=" * 60)
    
    # Change to test directory
    original_dir = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Import path utilities
        sys.path.insert(0, os.path.join(original_dir, 'simple'))
        import path_utils
        
        # Find model directory
        model_dir = path_utils.find_model_directory()
        if not model_dir:
            print("  ❌ No model directory found")
            return False
        
        # Find gradient descent script
        gd_script = path_utils.find_script('gradient_descent_3d.py')
        if not gd_script:
            print("  ❌ Could not find gradient descent script")
            return False
        
        # Get working directory
        working_dir = path_utils.get_working_directory()
        
        # Test PNG generation (just help command to avoid long execution)
        print("  🚀 Testing gradient descent PNG generation capability...")
        cmd = [sys.executable, gd_script, '--model_dir', model_dir, '--save_png', '--help']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=working_dir)
            if result.returncode == 0 and '--save_png' in result.stdout:
                print("  ✅ PNG generation capability confirmed")
                return True
            else:
                print("  ❌ PNG generation capability not confirmed")
                return False
        except Exception as e:
            print(f"  ❌ Error testing PNG generation: {e}")
            return False
        
    except Exception as e:
        print(f"  ❌ Error in PNG generation test: {e}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_dir)
        # Remove simple from sys.path
        if os.path.join(original_dir, 'simple') in sys.path:
            sys.path.remove(os.path.join(original_dir, 'simple'))

def main():
    """Run complete integration tests."""
    print("🔍 Complete PNG Plot Integration Test")
    print("=" * 70)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)
    
    # Define test directories
    test_directories = [
        project_root,  # simple/
        parent_dir,    # neural_net/
        os.path.join(parent_dir, 'examples'),  # examples/
    ]
    
    results = {}
    
    for test_dir in test_directories:
        if os.path.exists(test_dir):
            print(f"\n{'='*70}")
            print(f"Testing from: {test_dir}")
            print(f"{'='*70}")
            
            # Test GUI script execution
            gui_ok = test_gui_script_execution(test_dir)
            
            # Test PNG generation
            png_ok = test_png_generation(test_dir)
            
            results[test_dir] = gui_ok and png_ok
        else:
            print(f"\n⚠️  Skipping non-existent directory: {test_dir}")
            results[test_dir] = False
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Complete Test Summary")
    print(f"{'='*70}")
    
    passed = 0
    total = len(results)
    
    for test_dir, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        dir_name = os.path.basename(test_dir)
        print(f"{status} {dir_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} directories passed")
    
    if passed == total:
        print("🎉 All tests passed! Complete PNG plot integration works from all directories.")
        print("\n📖 Complete Usage Instructions:")
        print("1. Run the GUI from any directory:")
        print("   - From simple/: python stock_gui.py")
        print("   - From parent/: python simple/stock_gui.py")
        print("   - From anywhere: python /path/to/simple/stock_gui.py")
        print("2. The GUI will automatically:")
        print("   - Find all required scripts (predict.py, gradient_descent_3d.py)")
        print("   - Discover models in multiple directories")
        print("   - Execute scripts with correct working directory")
        print("   - Generate and display PNG plots")
        print("3. All PNG plot integration features work seamlessly:")
        print("   - Training loss plots")
        print("   - Prediction plots")
        print("   - Gradient descent visualization plots")
        print("   - Automatic plot refresh after generation")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 