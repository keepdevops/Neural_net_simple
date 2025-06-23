#!/usr/bin/env python3
"""
Test Color Value Display
=======================

This script tests if the current color value is properly displayed in the combobox.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_color_display():
    """Test if the current color value is displayed in the combobox."""
    print("🧪 Testing Color Value Display...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Color Display Test")
        root.geometry("1000x700")
        
        gui = StockPredictionGUI(root)
        
        # Check color variable before tab creation
        print(f"✅ Initial color_var value: '{gui.color_var.get()}'")
        
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
            
            # Check color variable after tab creation
            print(f"✅ After tab creation color_var value: '{gui.color_var.get()}'")
            
            # Check if color_combo instance variable exists
            if hasattr(gui, 'color_combo'):
                print("✅ Color combo instance variable found")
                
                # Check the values and current display
                values = gui.color_combo['values']
                current_value = gui.color_combo.get()
                var_value = gui.color_var.get()
                
                print(f"✅ Combobox values count: {len(values)}")
                print(f"✅ Combobox display value: '{current_value}'")
                print(f"✅ Variable value: '{var_value}'")
                
                if current_value == var_value:
                    print("✅ Color value is properly displayed in combobox!")
                else:
                    print("❌ Color value mismatch between combobox and variable")
                    print(f"   Combobox shows: '{current_value}'")
                    print(f"   Variable contains: '{var_value}'")
                
                # Test changing the value
                print("\n🧪 Testing value change...")
                gui.color_var.set('plasma')
                root.update()
                new_display = gui.color_combo.get()
                new_var = gui.color_var.get()
                print(f"✅ After setting to 'plasma':")
                print(f"   Combobox shows: '{new_display}'")
                print(f"   Variable contains: '{new_var}'")
                
                if new_display == 'plasma':
                    print("✅ Value change is working correctly!")
                else:
                    print("❌ Value change not reflected in combobox")
                
            else:
                print("❌ Color combo instance variable not found")
            
            print("\n   Check if you can see the current color value in the combobox")
            print("   It should show the current selection (viridis or plasma)")
            
        else:
            print("❌ Plot Controls tab not found")
        
        # Keep window open for inspection
        print("\n   Window will stay open for 10 seconds for inspection...")
        print("   Look for the current color value displayed in the combobox")
        root.after(10000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_color_display() 