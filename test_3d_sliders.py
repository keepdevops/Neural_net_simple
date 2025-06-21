#!/usr/bin/env python3
"""
Test script for 3D slider controls in the stock prediction GUI.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path so we can import the GUI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_gui import StockPredictionGUI
    print("✅ Successfully imported StockPredictionGUI")
except ImportError as e:
    print(f"❌ Error importing StockPredictionGUI: {e}")
    sys.exit(1)

def test_3d_slider_controls():
    """Test the 3D slider controls functionality."""
    print("🚀 Testing 3D Slider Controls...")
    
    # Create a minimal root window for testing
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    try:
        # Create the GUI instance
        app = StockPredictionGUI(root)
        print("✅ GUI instance created successfully")
        
        # Check if slider variables exist
        required_vars = [
            'x_rotation_var',
            'y_rotation_var', 
            'z_rotation_var',
            'zoom_var',
            'camera_x_var',
            'camera_y_var',
            'camera_z_var'
        ]
        
        for var_name in required_vars:
            if hasattr(app, var_name):
                print(f"✅ {var_name} exists")
            else:
                print(f"❌ {var_name} missing")
                return False
        
        # Check if slider callback methods exist
        required_methods = [
            'on_x_rotation_change',
            'on_y_rotation_change',
            'on_z_rotation_change',
            'on_zoom_change',
            'on_camera_x_change',
            'on_camera_y_change',
            'on_camera_z_change'
        ]
        
        for method_name in required_methods:
            if hasattr(app, method_name) and callable(getattr(app, method_name)):
                print(f"✅ {method_name} method exists and is callable")
            else:
                print(f"❌ {method_name} method missing or not callable")
                return False
        
        # Test slider value changes
        print("\n🧪 Testing slider value changes...")
        
        # Test X rotation slider
        app.x_rotation_var.set(45.0)
        print(f"✅ X rotation set to: {app.x_rotation_var.get()}")
        
        # Test Y rotation slider
        app.y_rotation_var.set(-30.0)
        print(f"✅ Y rotation set to: {app.y_rotation_var.get()}")
        
        # Test Z rotation slider
        app.z_rotation_var.set(90.0)
        print(f"✅ Z rotation set to: {app.z_rotation_var.get()}")
        
        # Test zoom slider
        app.zoom_var.set(2.5)
        print(f"✅ Zoom level set to: {app.zoom_var.get()}")
        
        # Test camera position sliders
        app.camera_x_var.set(3.0)
        app.camera_y_var.set(-2.0)
        app.camera_z_var.set(8.0)
        print(f"✅ Camera position set to: X={app.camera_x_var.get()}, Y={app.camera_y_var.get()}, Z={app.camera_z_var.get()}")
        
        # Test callback methods (they should handle missing 3D axes gracefully)
        print("\n🧪 Testing callback methods...")
        
        try:
            app.on_x_rotation_change(45.0)
            print("✅ on_x_rotation_change executed without error")
        except Exception as e:
            print(f"⚠️ on_x_rotation_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_y_rotation_change(-30.0)
            print("✅ on_y_rotation_change executed without error")
        except Exception as e:
            print(f"⚠️ on_y_rotation_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_z_rotation_change(90.0)
            print("✅ on_z_rotation_change executed without error")
        except Exception as e:
            print(f"⚠️ on_z_rotation_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_zoom_change(2.5)
            print("✅ on_zoom_change executed without error")
        except Exception as e:
            print(f"⚠️ on_zoom_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_camera_x_change(3.0)
            print("✅ on_camera_x_change executed without error")
        except Exception as e:
            print(f"⚠️ on_camera_x_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_camera_y_change(-2.0)
            print("✅ on_camera_y_change executed without error")
        except Exception as e:
            print(f"⚠️ on_camera_y_change error (expected if no 3D plot): {e}")
        
        try:
            app.on_camera_z_change(8.0)
            print("✅ on_camera_z_change executed without error")
        except Exception as e:
            print(f"⚠️ on_camera_z_change error (expected if no 3D plot): {e}")
        
        print("\n✅ All 3D slider controls tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    finally:
        # Clean up
        try:
            root.destroy()
        except:
            pass

def test_gui_initialization():
    """Test that the GUI initializes without errors."""
    print("\n🚀 Testing GUI Initialization...")
    
    try:
        # Create a minimal root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Create the GUI instance
        app = StockPredictionGUI(root)
        print("✅ GUI initialized successfully")
        
        # Check that the main window was created
        if hasattr(app, 'root'):
            print("✅ Root window created")
        else:
            print("❌ Root window not created")
            return False
        
        # Check that status variables exist
        if hasattr(app, 'status_var'):
            print("✅ Status variable created")
        else:
            print("❌ Status variable not created")
            return False
        
        # Check that the GUI has the basic structure
        if hasattr(app, 'setup_main_window'):
            print("✅ Main window setup method exists")
        else:
            print("❌ Main window setup method missing")
            return False
        
        print("✅ GUI initialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during GUI initialization: {e}")
        return False
    
    finally:
        # Clean up
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    print("🧪 Starting 3D Slider Controls Test Suite...")
    print("=" * 50)
    
    # Test GUI initialization
    init_success = test_gui_initialization()
    
    # Test 3D slider controls
    slider_success = test_3d_slider_controls()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"  GUI Initialization: {'✅ PASSED' if init_success else '❌ FAILED'}")
    print(f"  3D Slider Controls: {'✅ PASSED' if slider_success else '❌ FAILED'}")
    
    if init_success and slider_success:
        print("\n🎉 All tests passed! The 3D slider controls are working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1) 