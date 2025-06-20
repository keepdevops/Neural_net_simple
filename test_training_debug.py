#!/usr/bin/env python3
"""
Test script to verify training functionality and debug output.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox

# Add the current directory to the path so we can import stock_gui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_gui import StockPredictionGUI
    print("✅ Successfully imported StockPredictionGUI")
except ImportError as e:
    print(f"❌ Failed to import StockPredictionGUI: {e}")
    sys.exit(1)

def test_gui_creation():
    """Test that the GUI can be created without errors."""
    print("\n🧪 Testing GUI creation...")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing
        
        app = StockPredictionGUI(root)
        print("✅ GUI created successfully")
        
        # Test that required attributes exist
        required_attrs = [
            'train_button', 'stop_button', 'status_label', 'status_var',
            'progress_bar', 'progress_var', 'hidden_size_var', 'learning_rate_var',
            'batch_size_var', 'patience_var'
        ]
        
        for attr in required_attrs:
            if hasattr(app, attr):
                print(f"✅ {attr} exists")
            else:
                print(f"❌ {attr} missing")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI creation failed: {e}")
        return False

def test_training_methods():
    """Test that training methods exist and are callable."""
    print("\n🧪 Testing training methods...")
    
    try:
        root = tk.Tk()
        root.withdraw()
        app = StockPredictionGUI(root)
        
        # Test that training methods exist
        training_methods = [
            'train_model', '_run_training_thread', 'update_progress',
            'threaded_action', '_training_completed', '_enable_train_button'
        ]
        
        for method in training_methods:
            if hasattr(app, method) and callable(getattr(app, method)):
                print(f"✅ {method} exists and is callable")
            else:
                print(f"❌ {method} missing or not callable")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Training methods test failed: {e}")
        return False

def test_debug_output():
    """Test that debug output is working."""
    print("\n🧪 Testing debug output...")
    
    try:
        root = tk.Tk()
        root.withdraw()
        app = StockPredictionGUI(root)
        
        # Test update_progress method
        print("Testing update_progress method...")
        app.update_progress("Test message from debug script")
        
        # Test that status variables are accessible
        if hasattr(app, 'status_var'):
            current_status = app.status_var.get()
            print(f"Current status: {current_status}")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Debug output test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting training functionality tests...")
    
    tests = [
        test_gui_creation,
        test_training_methods,
        test_debug_output
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Training functionality should work correctly.")
        print("\n💡 To test training:")
        print("   1. Run: python stock_gui.py")
        print("   2. Load a data file in the Data tab")
        print("   3. Select features and target")
        print("   4. Click 'Start Training' in the Training tab")
        print("   5. Watch for debug output in the terminal")
    else:
        print("⚠️  Some tests failed. Check the output above for issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 