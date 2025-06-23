#!/usr/bin/env python3
"""
Test Color Map Combobox Fix
===========================

This script tests the fixed color map combobox to ensure it displays values correctly.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_colormap_fix():
    """Test the fixed color map combobox."""
    print("🧪 Testing Fixed Color Map Combobox...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Color Map Fix Test")
        root.geometry("1000x700")
        
        gui = StockPredictionGUI(root)
        
        # Find Plot Controls tab
        plot_controls_tab_index = None
        for i in range(gui.control_notebook.index('end')):
            tab_name = gui.control_notebook.tab(i, 'text')
            if tab_name == 'Plot Controls':
                plot_controls_tab_index = i
                break
        
        if plot_controls_tab_index is not None:
            print(f"✅ Plot Controls tab found at index {plot_controls_tab_index}")
            
            # Switch to Plot Controls tab
            gui.control_notebook.select(plot_controls_tab_index)
            print("✅ Switched to Plot Controls tab")
            
            # Force update
            root.update()
            
            # Check if color_combo instance variable exists
            if hasattr(gui, 'color_combo'):
                print("✅ Color combo instance variable found")
                
                # Check the values
                values = gui.color_combo['values']
                current_value = gui.color_combo.get()
                
                print(f"✅ Combobox values count: {len(values)}")
                print(f"✅ Current selection: '{current_value}'")
                print(f"📋 First 5 values: {values[:5] if values else 'None'}")
                
                if values:
                    print("✅ Color map combobox has values - should be visible!")
                else:
                    print("❌ Color map combobox has no values")
            else:
                print("❌ Color combo instance variable not found")
            
            # Check if color map variable exists and has a value
            if hasattr(gui, 'color_var'):
                current_value = gui.color_var.get()
                print(f"✅ Color map variable found with value: '{current_value}'")
            else:
                print("❌ Color map variable not found")
            
            print("\n   Check if you can see the color map combobox in the Plot Controls tab")
            print("   It should show 'viridis' as the default and have a dropdown with 23 options")
            
        else:
            print("❌ Plot Controls tab not found")
        
        # Keep window open for inspection
        print("\n   Window will stay open for 10 seconds for inspection...")
        print("   Look for the 'Color Map:' label and dropdown in the Plot Controls tab")
        root.after(10000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_colormap_fix() 