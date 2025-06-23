#!/usr/bin/env python3
"""
Test Color Map Combobox Display
==============================

This script tests if the color map combobox is properly displayed in the Plot Controls tab.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_colormap_display():
    """Test if the color map combobox is displayed correctly."""
    print("🧪 Testing Color Map Combobox Display...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Color Map Combobox Test")
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
            
            # Check if color map variable exists and has a value
            if hasattr(gui, 'color_var'):
                current_value = gui.color_var.get()
                print(f"✅ Color map variable found with value: '{current_value}'")
            else:
                print("❌ Color map variable not found")
            
            # Check if the combobox widget exists in the GUI
            combobox_found = False
            for widget in root.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    continue
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Notebook):
                        for tab in child.winfo_children():
                            if isinstance(tab, ttk.Frame):
                                for grandchild in tab.winfo_children():
                                    if isinstance(grandchild, ttk.Combobox):
                                        combobox_found = True
                                        print(f"✅ Combobox found with values: {grandchild['values']}")
                                        print(f"   Current selection: {grandchild.get()}")
                                        break
                                if combobox_found:
                                    break
                        if combobox_found:
                            break
                if combobox_found:
                    break
            
            if not combobox_found:
                print("❌ Color map combobox not found in GUI")
            
            print("\n   Check if you can see the color map combobox in the Plot Controls tab")
            print("   It should be labeled 'Color Map:' with a dropdown list")
            
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

def test_colormap_values():
    """Test the predefined colormap values."""
    print("\n🧪 Testing Color Map Values...")
    
    expected_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'jet', 'hot', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter']
    
    print(f"✅ Expected colormaps count: {len(expected_colormaps)}")
    print(f"📋 Expected colormaps: {expected_colormaps}")
    
    # Test if matplotlib can use these colormaps
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        valid_colormaps = []
        invalid_colormaps = []
        
        for colormap in expected_colormaps:
            try:
                cm.get_cmap(colormap)
                valid_colormaps.append(colormap)
            except ValueError:
                invalid_colormaps.append(colormap)
        
        print(f"✅ Valid colormaps: {len(valid_colormaps)}")
        print(f"❌ Invalid colormaps: {len(invalid_colormaps)}")
        
        if invalid_colormaps:
            print(f"   Invalid colormaps: {invalid_colormaps}")
        
    except ImportError:
        print("⚠️  Matplotlib not available for colormap validation")

def main():
    """Run all tests."""
    print("🔍 Test Color Map Combobox Display")
    print("=" * 50)
    
    # Test 1: Color map display
    test_colormap_display()
    
    # Test 2: Color map values
    test_colormap_values()
    
    print("\n" + "=" * 50)
    print("🎯 Color map tests completed!")

if __name__ == "__main__":
    main() 