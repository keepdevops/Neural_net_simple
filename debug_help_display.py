#!/usr/bin/env python3
"""
Debug Script for Help Display Issue
==================================

This script helps debug why the help content is not displaying in the GUI.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_text_widget():
    """Test if text widgets work at all."""
    print("🧪 Testing Simple Text Widget...")
    
    root = tk.Tk()
    root.title("Simple Text Widget Test")
    root.geometry("400x300")
    
    # Create a simple text widget
    text_widget = tk.Text(root, wrap=tk.WORD, font=("Arial", 10))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Insert some test content
    test_content = "This is a test of the text widget.\n\nIf you can see this text, the text widget is working correctly."
    text_widget.insert(tk.END, test_content)
    
    print("✅ Simple text widget created with test content")
    print("   Check if you can see the test text in the window")
    
    root.mainloop()

def test_help_content_generation():
    """Test help content generation specifically."""
    print("\n🧪 Testing Help Content Generation...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        
        # Create GUI instance
        gui = StockPredictionGUI(root)
        
        # Test help content
        help_content = gui.get_help_content()
        print(f"✅ Help content length: {len(help_content)} characters")
        print(f"📄 First 100 chars: {help_content[:100]}...")
        
        # Test manual content
        manual_content = gui.get_full_manual_content()
        print(f"✅ Manual content length: {len(manual_content)} characters")
        print(f"📄 First 100 chars: {manual_content[:100]}...")
        
        # Clean up
        root.destroy()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_help_tab_display():
    """Test the help tab display specifically."""
    print("\n🧪 Testing Help Tab Display...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Help Tab Test")
        root.geometry("800x600")
        
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
            
            print("   Check if help content is visible in the tab")
            print("   If not, there might be a display issue")
        else:
            print("❌ Help tab not found")
        
        # Keep window open for inspection
        print("\n   Window will stay open for 10 seconds for inspection...")
        root.after(10000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_manual_window_display():
    """Test the manual window display specifically."""
    print("\n🧪 Testing Manual Window Display...")
    
    try:
        from stock_gui import StockPredictionGUI
        
        # Create GUI instance
        root = tk.Tk()
        root.title("Manual Window Test")
        root.geometry("400x300")
        
        gui = StockPredictionGUI(root)
        
        # Test manual window
        print("   Opening manual window...")
        gui.show_manual_window()
        
        print("   Check if manual window opens with content")
        print("   If not, there might be a display issue")
        
        # Keep window open for inspection
        print("\n   Window will stay open for 10 seconds for inspection...")
        root.after(10000, root.destroy)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debug tests."""
    print("🔍 Debug Help Display Issue")
    print("=" * 50)
    
    # Test 1: Simple text widget
    test_simple_text_widget()
    
    # Test 2: Help content generation
    test_help_content_generation()
    
    # Test 3: Help tab display
    test_help_tab_display()
    
    # Test 4: Manual window display
    test_manual_window_display()
    
    print("\n" + "=" * 50)
    print("🎯 Debug tests completed!")
    print("Check the output above for any issues.")

if __name__ == "__main__":
    main() 