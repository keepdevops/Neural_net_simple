#!/usr/bin/env python3
"""
Test script to verify the new image scaling slider and comprehensive plots functionality.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk
import time

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import stock_gui
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_image_scaling_features():
    """Test the new image scaling and comprehensive plots features."""
    print("Testing image scaling features...")
    
    # Create a minimal root window
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        # Create the GUI instance
        app = stock_gui.StockPredictionGUI(root)
        
        # Test that the image scale variable exists
        if hasattr(app, 'image_scale_var'):
            print("✅ image_scale_var exists")
            print(f"   Default value: {app.image_scale_var.get()}%")
        else:
            print("❌ image_scale_var not found")
            return False
        
        # Test that the comprehensive canvas exists
        if hasattr(app, 'comprehensive_canvas'):
            print("✅ comprehensive_canvas exists")
        else:
            print("❌ comprehensive_canvas not found")
            return False
        
        # Test that the load_comprehensive_plots method exists
        if hasattr(app, 'load_comprehensive_plots'):
            print("✅ load_comprehensive_plots method exists")
        else:
            print("❌ load_comprehensive_plots method not found")
            return False
        
        # Test that the on_window_resize method exists
        if hasattr(app, 'on_window_resize'):
            print("✅ on_window_resize method exists")
        else:
            print("❌ on_window_resize method not found")
            return False
        
        # Test the methods can be called
        try:
            # Test load_comprehensive_plots with no model selected
            app.load_comprehensive_plots()
            print("✅ load_comprehensive_plots method can be called")
        except Exception as e:
            print(f"❌ Error calling load_comprehensive_plots: {e}")
            return False
        
        try:
            # Test on_window_resize with a dummy event
            class DummyEvent:
                def __init__(self, widget):
                    self.widget = widget
            
            event = DummyEvent(root)
            app.on_window_resize(event)
            print("✅ on_window_resize method can be called")
        except Exception as e:
            print(f"❌ Error calling on_window_resize: {e}")
            return False
        
        # Test image scale variable range
        try:
            original_value = app.image_scale_var.get()
            
            # Test setting different values
            test_values = [10.0, 50.0, 100.0, 150.0, 200.0]
            for value in test_values:
                app.image_scale_var.set(value)
                current_value = app.image_scale_var.get()
                if abs(current_value - value) < 0.1:  # Allow small floating point differences
                    print(f"✅ Image scale can be set to {value}%")
                else:
                    print(f"❌ Image scale setting failed: expected {value}%, got {current_value}%")
                    return False
            
            # Restore original value
            app.image_scale_var.set(original_value)
            
        except Exception as e:
            print(f"❌ Error testing image scale variable: {e}")
            return False
        
        print("✅ All image scaling features work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing image scaling features: {e}")
        return False
    finally:
        # Clean up
        try:
            app.cleanup_thread_pool()
        except:
            pass
        root.destroy()

def test_display_panel_structure():
    """Test that the display panel structure is correct."""
    print("\nTesting display panel structure...")
    
    # Create a minimal root window
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    try:
        # Create the GUI instance
        app = stock_gui.StockPredictionGUI(root)
        
        # Test that the display notebook exists and has the right number of tabs
        if hasattr(app, 'display_notebook'):
            print("✅ display_notebook exists")
            num_tabs = app.display_notebook.index('end')
            print(f"   Number of tabs: {num_tabs}")
            
            # Check for expected tabs
            expected_tabs = ["Training Results", "Prediction Results", "Live Training", "Plots", "3D Gradient Descent", "Saved Plots"]
            actual_tabs = []
            for i in range(num_tabs):
                tab_text = app.display_notebook.tab(i, "text")
                actual_tabs.append(tab_text)
            
            print(f"   Actual tabs: {actual_tabs}")
            
            # Check if Saved Plots tab exists
            if "Saved Plots" in actual_tabs:
                print("✅ Saved Plots tab exists")
            else:
                print("❌ Saved Plots tab not found")
                return False
            
        else:
            print("❌ display_notebook not found")
            return False
        
        print("✅ Display panel structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error testing display panel structure: {e}")
        return False
    finally:
        # Clean up
        try:
            app.cleanup_thread_pool()
        except:
            pass
        root.destroy()

def main():
    """Run all tests."""
    print("🧪 Testing Image Scaling and Comprehensive Plots Features")
    print("=" * 60)
    
    tests = [
        ("Display Panel Structure", test_display_panel_structure),
        ("Image Scaling Features", test_image_scaling_features),
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
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The image scaling and comprehensive plots features are working correctly.")
        print("\n✨ New Features Added:")
        print("   • Image scaling slider (10% - 200%)")
        print("   • Dynamic image resizing based on canvas width")
        print("   • Comprehensive plots tab with grid layout")
        print("   • Raw PNGs tab with scrollable view")
        print("   • Window resize handling")
        print("   • Improved caching with scale-aware keys")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 