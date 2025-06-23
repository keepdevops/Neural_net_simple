#!/usr/bin/env python3
"""
Simple Command-Line Test for Help Content
========================================

This script quickly tests the help content generation without opening GUI windows.
It's useful for debugging and verification.

Usage:
    python test_help_simple.py
"""

import sys
import os

# Add the current directory to the path so we can import the main GUI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_help_content():
    """Test help content generation."""
    print("🧪 Testing Help Content Generation...")
    print("=" * 50)
    
    try:
        # Import tkinter for the root window
        import tkinter as tk
        from stock_gui import StockPredictionGUI
        
        # Create a root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Create GUI instance
        gui = StockPredictionGUI(root)
        
        # Test 1: Help content method
        print("Test 1: Checking get_help_content method...")
        if hasattr(gui, 'get_help_content'):
            help_content = gui.get_help_content()
            if help_content and len(help_content.strip()) > 0:
                print(f"✅ Help content generated: {len(help_content)} characters")
                
                # Check for key sections
                sections = ["OVERVIEW", "INTERFACE OVERVIEW", "USAGE TIPS", "TROUBLESHOOTING"]
                found_sections = [s for s in sections if s in help_content]
                print(f"✅ Found {len(found_sections)}/{len(sections)} key sections")
                
                # Show first 200 characters
                print(f"📄 Preview: {help_content[:200]}...")
            else:
                print("❌ Help content is empty")
        else:
            print("❌ get_help_content method not found")
            
        # Test 2: Manual content method
        print("\nTest 2: Checking get_full_manual_content method...")
        if hasattr(gui, 'get_full_manual_content'):
            manual_content = gui.get_full_manual_content()
            if manual_content and len(manual_content.strip()) > 0:
                print(f"✅ Manual content generated: {len(manual_content)} characters")
                
                # Check for key sections
                sections = ["OVERVIEW", "INSTALLATION", "DETAILED TAB GUIDE", "USAGE WORKFLOW"]
                found_sections = [s for s in sections if s in manual_content]
                print(f"✅ Found {len(found_sections)}/{len(sections)} key sections")
                
                # Show first 200 characters
                print(f"📄 Preview: {manual_content[:200]}...")
            else:
                print("❌ Manual content is empty")
        else:
            print("❌ get_full_manual_content method not found")
            
        # Test 3: Check if help tab exists
        print("\nTest 3: Checking help tab creation...")
        if hasattr(gui, 'control_notebook'):
            tab_names = []
            for i in range(gui.control_notebook.index('end')):
                tab_name = gui.control_notebook.tab(i, 'text')
                tab_names.append(tab_name)
            
            print(f"📋 Available tabs: {', '.join(tab_names)}")
            
            if '?' in tab_names:
                print("✅ Help tab (?) found")
            else:
                print("❌ Help tab (?) not found")
        else:
            print("❌ Control notebook not found")
            
        # Test 4: Check manual window method
        print("\nTest 4: Checking show_manual_window method...")
        if hasattr(gui, 'show_manual_window'):
            print("✅ show_manual_window method found")
        else:
            print("❌ show_manual_window method not found")
            
        # Test 5: Check print manual method
        print("\nTest 5: Checking print_full_manual method...")
        if hasattr(gui, 'print_full_manual'):
            print("✅ print_full_manual method found")
        else:
            print("❌ print_full_manual method not found")
            
        # Test 6: Check open manual file method
        print("\nTest 6: Checking open_manual_file method...")
        if hasattr(gui, 'open_manual_file'):
            print("✅ open_manual_file method found")
        else:
            print("❌ open_manual_file method not found")
            
        # Clean up
        root.destroy()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_manual_file_exists():
    """Test if the manual file exists."""
    print("\n📁 Testing Manual File...")
    print("=" * 30)
    
    manual_file = os.path.join(os.path.dirname(__file__), "STOCK_GUI_USER_MANUAL.py")
    
    if os.path.exists(manual_file):
        print(f"✅ Manual file found: {manual_file}")
        
        # Check file size
        file_size = os.path.getsize(manual_file)
        print(f"📏 File size: {file_size} bytes")
        
        # Try to read the file
        try:
            with open(manual_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"📄 File content length: {len(content)} characters")
                
                # Check for docstring
                if '"""' in content:
                    print("✅ File contains docstring")
                else:
                    print("⚠️  File doesn't contain docstring")
                    
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ Manual file not found: {manual_file}")

if __name__ == "__main__":
    print("🧪 Simple Help Content Test")
    print("Testing help and manual functionality")
    print()
    
    test_help_content()
    test_manual_file_exists() 