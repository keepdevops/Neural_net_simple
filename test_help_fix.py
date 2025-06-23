#!/usr/bin/env python3
"""
Test Help Display Fix
====================

This script tests the fixed help and manual display functionality.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_help_fix():
    """Test the fixed help display."""
    print("🧪 Testing Fixed Help Display...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Help Display Fix Test")
        root.geometry("1000x700")
        
        gui = StockPredictionGUI(root)
        
        # Find help tab
        help_tab_index = None
        for i in range(gui.control_notebook.index('end')):
            tab_name = gui.control_notebook.tab(i, 'text')
            if tab_name == '?':
                help_tab_index = i
                break
        
        if help_tab_index is not None:
            print(f"✅ Help tab found at index {help_tab_index}")
            
            # Switch to help tab
            gui.control_notebook.select(help_tab_index)
            print("✅ Switched to help tab")
            
            # Force update
            root.update()
            
            print("   Check if help content is now visible in the tab")
            print("   You should see help text in the text widget")
            
            # Test manual window
            print("\n   Testing manual window...")
            gui.show_manual_window()
            print("   Manual window should open with content")
            
        else:
            print("❌ Help tab not found")
        
        # Keep window open for inspection
        print("\n   Window will stay open for 15 seconds for inspection...")
        print("   Check both the help tab and manual window for content")
        root.after(15000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_help_fix() 