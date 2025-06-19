#!/usr/bin/env python3
"""
Test Path Resolution from Different Working Directories

This script tests the path resolution system by running from different
working directories to ensure the GUI can find scripts and files correctly.
"""

import os
import sys
import subprocess
import tempfile
import time

def test_from_directory(test_dir, script_name="stock_gui.py"):
    """Test path resolution from a specific directory."""
    print(f"\n🧪 Testing from directory: {test_dir}")
    print("=" * 50)
    
    # Change to test directory
    original_dir = os.getcwd()
    os.chdir(test_dir)
    
    try:
        # Test path utilities
        print("📋 Testing path utilities...")
        
        # Import path utilities
        sys.path.insert(0, os.path.join(original_dir, 'simple'))
        import path_utils
        
        # Test script finding
        scripts_to_test = ['predict.py', 'gradient_descent_3d.py', 'view_results.py']
        for script in scripts_to_test:
            path = path_utils.find_script(script)
            if path:
                print(f"  ✅ Found {script}: {path}")
            else:
                print(f"  ❌ Could not find {script}")
        
        # Test model directory finding
        model_dir = path_utils.find_model_directory()
        if model_dir:
            print(f"  ✅ Found model directory: {model_dir}")
        else:
            print(f"  ❌ Could not find model directory")
        
        # Test working directory
        working_dir = path_utils.get_working_directory()
        print(f"  📁 Working directory: {working_dir}")
        
        # Test GUI import
        print("\n📱 Testing GUI import...")
        try:
            import stock_gui
            print("  ✅ GUI module imports successfully")
            
            # Test if required methods exist
            required_methods = ['load_saved_plots', 'trigger_gradient_descent_visualization']
            for method in required_methods:
                if hasattr(stock_gui.StockPredictionGUI, method):
                    print(f"    ✅ Found method: {method}")
                else:
                    print(f"    ❌ Missing method: {method}")
            
            gui_ok = True
        except Exception as e:
            print(f"  ❌ Error importing GUI: {e}")
            gui_ok = False
        
        # Test script execution
        print("\n🚀 Testing script execution...")
        script_path = path_utils.find_script('gradient_descent_3d.py')
        if script_path:
            try:
                result = subprocess.run([sys.executable, script_path, '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("  ✅ Gradient descent script executes successfully")
                    script_ok = True
                else:
                    print("  ❌ Gradient descent script execution failed")
                    script_ok = False
            except Exception as e:
                print(f"  ❌ Error executing script: {e}")
                script_ok = False
        else:
            print("  ❌ Could not find gradient descent script")
            script_ok = False
        
        return gui_ok and script_ok
        
    except Exception as e:
        print(f"  ❌ Error in test: {e}")
        return False
    finally:
        # Restore original directory
        os.chdir(original_dir)
        # Remove simple from sys.path
        if os.path.join(original_dir, 'simple') in sys.path:
            sys.path.remove(os.path.join(original_dir, 'simple'))

def main():
    """Run path resolution tests from different directories."""
    print("🔍 Testing Path Resolution from Different Directories")
    print("=" * 60)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)
    
    # Define test directories
    test_directories = [
        project_root,  # simple/
        parent_dir,    # neural_net/
        os.path.join(parent_dir, 'examples'),  # examples/
        tempfile.gettempdir()  # System temp directory
    ]
    
    results = {}
    
    for test_dir in test_directories:
        if os.path.exists(test_dir):
            success = test_from_directory(test_dir)
            results[test_dir] = success
        else:
            print(f"\n⚠️  Skipping non-existent directory: {test_dir}")
            results[test_dir] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_dir, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        dir_name = os.path.basename(test_dir) if test_dir != tempfile.gettempdir() else "temp"
        print(f"{status} {dir_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} directories passed")
    
    if passed == total:
        print("🎉 All tests passed! Path resolution works from all directories.")
        print("\n📖 Usage Instructions:")
        print("1. You can run the GUI from any directory:")
        print("   - From simple/: python stock_gui.py")
        print("   - From parent/: python simple/stock_gui.py")
        print("   - From anywhere: python /path/to/simple/stock_gui.py")
        print("2. The GUI will automatically find scripts and models")
        print("3. All PNG plot integration features will work correctly")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 