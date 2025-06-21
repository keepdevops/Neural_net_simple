#!/usr/bin/env python3
"""
Test script to verify cache clearing functionality in stock_gui.py
"""

import tkinter as tk
import sys
import os

# Add the current directory to the path so we can import stock_gui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cache_clearing():
    """Test the cache clearing functionality."""
    print("🧪 Testing cache clearing functionality...")
    
    try:
        # Import the GUI class
        from stock_gui import StockPredictionGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing
        
        # Create the GUI instance
        gui = StockPredictionGUI(root)
        
        print("✅ GUI instance created successfully")
        
        # Test if clear_cache_and_reinitialize method exists
        if hasattr(gui, 'clear_cache_and_reinitialize'):
            print("✅ clear_cache_and_reinitialize method exists")
        else:
            print("❌ clear_cache_and_reinitialize method not found")
            return False
        
        # Test if cache clear button exists
        if hasattr(gui, 'cache_clear_btn'):
            print("✅ Cache clear button exists")
        else:
            print("❌ Cache clear button not found")
            return False
        
        # Test if image caches exist
        if hasattr(gui, 'image_cache'):
            print("✅ image_cache exists")
        else:
            print("❌ image_cache not found")
            return False
        
        if hasattr(gui, 'plot_image_cache'):
            print("✅ plot_image_cache exists")
        else:
            print("❌ plot_image_cache not found")
            return False
        
        # Test cache clearing functionality
        print("🧹 Testing cache clearing...")
        initial_cache_size = len(gui.image_cache) if hasattr(gui, 'image_cache') else 0
        initial_plot_cache_size = len(gui.plot_image_cache) if hasattr(gui, 'plot_image_cache') else 0
        
        # Call the cache clearing method
        gui.clear_cache_and_reinitialize()
        
        # Check if caches were cleared
        final_cache_size = len(gui.image_cache) if hasattr(gui, 'image_cache') else 0
        final_plot_cache_size = len(gui.plot_image_cache) if hasattr(gui, 'plot_image_cache') else 0
        
        if final_cache_size == 0 and final_plot_cache_size == 0:
            print("✅ Caches cleared successfully")
        else:
            print(f"❌ Caches not cleared properly. Final sizes: {final_cache_size}, {final_plot_cache_size}")
            return False
        
        # Test if plots were reinitialized
        if hasattr(gui, 'results_ax') and gui.results_ax is not None:
            print("✅ Training results plot reinitialized")
        else:
            print("❌ Training results plot not reinitialized")
            return False
        
        if hasattr(gui, 'plots_ax') and gui.plots_ax is not None:
            print("✅ Plots tab reinitialized")
        else:
            print("❌ Plots tab not reinitialized")
            return False
        
        if hasattr(gui, 'gd3d_ax') and gui.gd3d_ax is not None:
            print("✅ 3D gradient descent plot reinitialized")
        else:
            print("❌ 3D gradient descent plot not reinitialized")
            return False
        
        if hasattr(gui, 'pred_ax') and gui.pred_ax is not None:
            print("✅ Prediction plot reinitialized")
        else:
            print("❌ Prediction plot not reinitialized")
            return False
        
        print("✅ All cache clearing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during cache clearing test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    success = test_cache_clearing()
    if success:
        print("\n🎉 Cache clearing functionality test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Cache clearing functionality test FAILED!")
        sys.exit(1)
