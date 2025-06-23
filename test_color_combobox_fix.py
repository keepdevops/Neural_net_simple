#!/usr/bin/env python3
"""
Test Color Combobox Fix
=======================

This script tests the fixed color combobox to ensure it displays values correctly.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_color_combobox_fix():
    """Test the fixed color combobox."""
    print("🧪 Testing Fixed Color Combobox...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Color Combobox Fix Test")
        root.geometry("1000x700")
        
        gui = StockPredictionGUI(root)
        
        # Test the color combobox
        print("\n📋 Color Combobox Test Results:")
        print("=" * 50)
        
        # Check if color_combo exists
        if hasattr(gui, 'color_combo'):
            print(f"✅ Color combobox found: {gui.color_combo}")
            
            # Check current value
            current_value = gui.color_var.get()
            print(f"✅ Color variable value: '{current_value}'")
            
            # Check combobox display value
            combo_value = gui.color_combo.get()
            print(f"✅ Combobox display value: '{combo_value}'")
            
            # Check if values are set
            combo_values = gui.color_combo['values']
            print(f"✅ Combobox has {len(combo_values)} values")
            print(f"✅ First 5 values: {combo_values[:5]}")
            
            # Test refresh method
            print("\n🔄 Testing refresh method...")
            success = gui.refresh_color_combobox()
            print(f"✅ Refresh method result: {success}")
            
            # Check after refresh
            new_combo_value = gui.color_combo.get()
            print(f"✅ Combobox value after refresh: '{new_combo_value}'")
            
            # Test changing the value
            print("\n🔄 Testing value change...")
            gui.color_var.set('plasma')
            gui.refresh_color_combobox()
            changed_value = gui.color_combo.get()
            print(f"✅ Combobox value after change to 'plasma': '{changed_value}'")
            
        else:
            print("❌ Color combobox not found!")
        
        print("\n✅ Color combobox test completed!")
        
        # Keep the GUI open for manual inspection
        print("\n🔍 GUI is open for manual inspection. Close the window when done.")
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error in color combobox test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_color_combobox_fix() 