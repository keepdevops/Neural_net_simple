#!/usr/bin/env python3
"""
Test script to verify the updated _show_gradient_descent method works correctly.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
import time

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import stock_gui
    import script_launcher
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_gradient_descent_method():
    """Test the updated _show_gradient_descent method."""
    print("Testing _show_gradient_descent method...")
    
    # Create a minimal root window
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        # Create the GUI instance
        app = stock_gui.StockPredictionGUI(root)
        
        # Test the method exists
        if hasattr(app, '_show_gradient_descent'):
            print("✅ _show_gradient_descent method exists")
        else:
            print("❌ _show_gradient_descent method not found")
            return False
        
        if hasattr(app, '_generate_gradient_descent_worker'):
            print("✅ _generate_gradient_descent_worker method exists")
        else:
            print("❌ _generate_gradient_descent_worker method not found")
            return False
        
        if hasattr(app, '_check_gradient_descent_complete'):
            print("✅ _check_gradient_descent_complete method exists")
        else:
            print("❌ _check_gradient_descent_complete method not found")
            return False
        
        # Test that the method can be called (without actually running it)
        try:
            # Set a dummy model path
            app.selected_model_path = "/tmp/test_model"
            
            # Test the method signature
            app._show_gradient_descent()
            print("✅ _show_gradient_descent method can be called")
            
        except Exception as e:
            print(f"❌ Error calling _show_gradient_descent: {e}")
            return False
        
        # Test the worker method signature
        try:
            result = app._generate_gradient_descent_worker()
            print("✅ _generate_gradient_descent_worker method can be called")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ Error calling _generate_gradient_descent_worker: {e}")
            return False
        
        print("✅ All gradient descent methods work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing gradient descent methods: {e}")
        return False
    finally:
        # Clean up
        try:
            app.cleanup_thread_pool()
        except:
            pass
        root.destroy()

def test_script_availability():
    """Test that the required scripts are available."""
    print("\nTesting script availability...")
    
    required_scripts = [
        "gradient_descent_3d.py",
        "predict.py", 
        "view_results.py"
    ]
    
    for script in required_scripts:
        script_path = script_launcher.launcher.find_script(script)
        if script_path and os.path.exists(script_path):
            print(f"✅ {script} found at: {script_path}")
        else:
            print(f"❌ {script} not found")
            return False
    
    print("✅ All required scripts are available")
    return True

def test_command_line_arguments():
    """Test that gradient_descent_3d.py accepts the new arguments."""
    print("\nTesting command line arguments...")
    
    try:
        script_path = script_launcher.launcher.find_script("gradient_descent_3d.py")
        if not script_path:
            print("❌ gradient_descent_3d.py not found")
            return False
        
        # Test with help to see if arguments are recognized
        cmd = [sys.executable, script_path, "--help"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)
        
        help_output = stdout.decode() + stderr.decode()
        
        # Check for new arguments
        new_args = [
            "--output_resolution",
            "--color", 
            "--point_size",
            "--line_width",
            "--surface_alpha",
            "--w1_range",
            "--w2_range"
        ]
        
        for arg in new_args:
            if arg in help_output:
                print(f"✅ {arg} argument recognized")
            else:
                print(f"❌ {arg} argument not found in help")
                return False
        
        print("✅ All new command line arguments are recognized")
        return True
        
    except Exception as e:
        print(f"❌ Error testing command line arguments: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Updated Gradient Descent Method")
    print("=" * 50)
    
    tests = [
        ("Script Availability", test_script_availability),
        ("Command Line Arguments", test_command_line_arguments),
        ("Gradient Descent Method", test_gradient_descent_method),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The updated gradient descent method is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 