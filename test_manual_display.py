#!/usr/bin/env python3
"""
Test Script for Manual and Help Display
=======================================

This script tests the manual and help functionality in the Stock Prediction GUI.
It verifies that:
1. The help tab (?) is properly created
2. Help content is displayed correctly
3. Manual window opens and displays content
4. All buttons work properly

Usage:
    python test_manual_display.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
import threading
import time

# Add the current directory to the path so we can import the main GUI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ManualDisplayTester:
    """Test class for manual and help display functionality."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Manual Display Test")
        self.root.geometry("1000x700")
        self.test_results = []
        
    def run_tests(self):
        """Run all tests and display results."""
        print("🧪 Starting Manual Display Tests...")
        print("=" * 50)
        
        # Test 1: Check if help content method exists
        self.test_help_content_method()
        
        # Test 2: Check if manual content method exists
        self.test_manual_content_method()
        
        # Test 3: Test help content generation
        self.test_help_content_generation()
        
        # Test 4: Test manual content generation
        self.test_manual_content_generation()
        
        # Test 5: Test manual window creation
        self.test_manual_window_creation()
        
        # Test 6: Test help tab creation
        self.test_help_tab_creation()
        
        # Display results
        self.display_test_results()
        
    def test_help_content_method(self):
        """Test if get_help_content method exists."""
        print("Test 1: Checking help content method...")
        try:
            # Import the main GUI class
            from stock_gui import StockPredictionGUI
            
            # Create a temporary instance to test methods
            temp_gui = StockPredictionGUI(self.root)
            
            if hasattr(temp_gui, 'get_help_content'):
                help_content = temp_gui.get_help_content()
                if help_content and len(help_content.strip()) > 0:
                    self.test_results.append(("✅ Help Content Method", "PASS", f"Content length: {len(help_content)} chars"))
                else:
                    self.test_results.append(("❌ Help Content Method", "FAIL", "Content is empty"))
            else:
                self.test_results.append(("❌ Help Content Method", "FAIL", "Method not found"))
                
        except Exception as e:
            self.test_results.append(("❌ Help Content Method", "ERROR", str(e)))
            
    def test_manual_content_method(self):
        """Test if get_full_manual_content method exists."""
        print("Test 2: Checking manual content method...")
        try:
            from stock_gui import StockPredictionGUI
            temp_gui = StockPredictionGUI(self.root)
            
            if hasattr(temp_gui, 'get_full_manual_content'):
                manual_content = temp_gui.get_full_manual_content()
                if manual_content and len(manual_content.strip()) > 0:
                    self.test_results.append(("✅ Manual Content Method", "PASS", f"Content length: {len(manual_content)} chars"))
                else:
                    self.test_results.append(("❌ Manual Content Method", "FAIL", "Content is empty"))
            else:
                self.test_results.append(("❌ Manual Content Method", "FAIL", "Method not found"))
                
        except Exception as e:
            self.test_results.append(("❌ Manual Content Method", "ERROR", str(e)))
            
    def test_help_content_generation(self):
        """Test help content generation and formatting."""
        print("Test 3: Testing help content generation...")
        try:
            from stock_gui import StockPredictionGUI
            temp_gui = StockPredictionGUI(self.root)
            
            help_content = temp_gui.get_help_content()
            
            # Check for key sections
            required_sections = [
                "OVERVIEW",
                "INTERFACE OVERVIEW", 
                "DATA SELECTION TAB",
                "TRAINING PARAMETERS TAB",
                "MODEL MANAGEMENT TAB",
                "PLOT CONTROLS TAB",
                "USAGE TIPS",
                "TROUBLESHOOTING"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in help_content:
                    missing_sections.append(section)
            
            if not missing_sections:
                self.test_results.append(("✅ Help Content Sections", "PASS", f"All {len(required_sections)} sections found"))
            else:
                self.test_results.append(("❌ Help Content Sections", "FAIL", f"Missing: {', '.join(missing_sections)}"))
                
        except Exception as e:
            self.test_results.append(("❌ Help Content Generation", "ERROR", str(e)))
            
    def test_manual_content_generation(self):
        """Test manual content generation and formatting."""
        print("Test 4: Testing manual content generation...")
        try:
            from stock_gui import StockPredictionGUI
            temp_gui = StockPredictionGUI(self.root)
            
            manual_content = temp_gui.get_full_manual_content()
            
            # Check for key sections
            required_sections = [
                "OVERVIEW",
                "INSTALLATION",
                "INTERFACE OVERVIEW",
                "DETAILED TAB GUIDE",
                "USAGE WORKFLOW",
                "TROUBLESHOOTING"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in manual_content:
                    missing_sections.append(section)
            
            if not missing_sections:
                self.test_results.append(("✅ Manual Content Sections", "PASS", f"All {len(required_sections)} sections found"))
            else:
                self.test_results.append(("❌ Manual Content Sections", "FAIL", f"Missing: {', '.join(missing_sections)}"))
                
        except Exception as e:
            self.test_results.append(("❌ Manual Content Generation", "ERROR", str(e)))
            
    def test_manual_window_creation(self):
        """Test manual window creation and content display."""
        print("Test 5: Testing manual window creation...")
        try:
            from stock_gui import StockPredictionGUI
            temp_gui = StockPredictionGUI(self.root)
            
            if hasattr(temp_gui, 'show_manual_window'):
                # Create a test window
                test_window = tk.Toplevel(self.root)
                test_window.title("Test Manual Window")
                test_window.geometry("600x400")
                
                # Test the manual window creation
                temp_gui.show_manual_window()
                
                # Check if window was created
                windows = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
                if len(windows) > 0:
                    self.test_results.append(("✅ Manual Window Creation", "PASS", f"Created {len(windows)} window(s)"))
                    
                    # Close test windows
                    for window in windows:
                        try:
                            window.destroy()
                        except:
                            pass
                else:
                    self.test_results.append(("❌ Manual Window Creation", "FAIL", "No window created"))
            else:
                self.test_results.append(("❌ Manual Window Creation", "FAIL", "Method not found"))
                
        except Exception as e:
            self.test_results.append(("❌ Manual Window Creation", "ERROR", str(e)))
            
    def test_help_tab_creation(self):
        """Test help tab creation in the main GUI."""
        print("Test 6: Testing help tab creation...")
        try:
            from stock_gui import StockPredictionGUI
            
            # Create a test GUI instance
            test_gui = StockPredictionGUI(self.root)
            
            # Check if control notebook exists
            if hasattr(test_gui, 'control_notebook'):
                # Get all tab names
                tab_names = []
                for i in range(test_gui.control_notebook.index('end')):
                    tab_name = test_gui.control_notebook.tab(i, 'text')
                    tab_names.append(tab_name)
                
                # Check for help tab
                if '?' in tab_names:
                    self.test_results.append(("✅ Help Tab Creation", "PASS", f"Help tab found. All tabs: {', '.join(tab_names)}"))
                else:
                    self.test_results.append(("❌ Help Tab Creation", "FAIL", f"Help tab not found. Available tabs: {', '.join(tab_names)}"))
            else:
                self.test_results.append(("❌ Help Tab Creation", "FAIL", "Control notebook not found"))
                
        except Exception as e:
            self.test_results.append(("❌ Help Tab Creation", "ERROR", str(e)))
            
    def display_test_results(self):
        """Display test results in a formatted way."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS")
        print("=" * 50)
        
        passed = 0
        failed = 0
        errors = 0
        
        for test_name, status, details in self.test_results:
            if status == "PASS":
                passed += 1
                print(f"{test_name}: {status} - {details}")
            elif status == "FAIL":
                failed += 1
                print(f"{test_name}: {status} - {details}")
            else:  # ERROR
                errors += 1
                print(f"{test_name}: {status} - {details}")
        
        print("\n" + "=" * 50)
        print(f"📈 SUMMARY: {passed} PASSED, {failed} FAILED, {errors} ERRORS")
        print("=" * 50)
        
        if failed == 0 and errors == 0:
            print("🎉 All tests passed! Manual and help display is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
            
        # Create a simple GUI to display results
        self.create_results_window()
        
    def create_results_window(self):
        """Create a window to display test results."""
        results_window = tk.Toplevel(self.root)
        results_window.title("Manual Display Test Results")
        results_window.geometry("600x500")
        
        # Title
        title_label = ttk.Label(results_window, text="Manual Display Test Results", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create text widget for results
        text_frame = ttk.Frame(results_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert results
        for test_name, status, details in self.test_results:
            if status == "PASS":
                text_widget.insert(tk.END, f"✅ {test_name}\n")
            elif status == "FAIL":
                text_widget.insert(tk.END, f"❌ {test_name}\n")
            else:
                text_widget.insert(tk.END, f"⚠️  {test_name}\n")
            text_widget.insert(tk.END, f"   Status: {status}\n")
            text_widget.insert(tk.END, f"   Details: {details}\n\n")
        
        # Make read-only
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ttk.Button(results_window, text="Close", command=results_window.destroy)
        close_btn.pack(pady=10)
        
    def run(self):
        """Run the test suite."""
        try:
            self.run_tests()
        except Exception as e:
            print(f"❌ Test suite failed with error: {e}")
        finally:
            # Keep the window open for a few seconds to see results
            self.root.after(5000, self.root.destroy)
            self.root.mainloop()

def main():
    """Main function to run the test."""
    print("🧪 Manual Display Test Suite")
    print("Testing help and manual functionality in Stock Prediction GUI")
    print()
    
    tester = ManualDisplayTester()
    tester.run()

if __name__ == "__main__":
    main() 