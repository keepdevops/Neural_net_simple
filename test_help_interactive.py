#!/usr/bin/env python3
"""
Interactive Test for Help and Manual Display
===========================================

This script provides an interactive way to test the help and manual functionality.
It creates a simple GUI with buttons to test different aspects of the help system.

Usage:
    python test_help_interactive.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add the current directory to the path so we can import the main GUI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class InteractiveHelpTester:
    """Interactive test class for help and manual functionality."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Interactive Help Test")
        self.root.geometry("800x600")
        
        # Create main GUI instance for testing
        try:
            from stock_gui import StockPredictionGUI
            self.main_gui = StockPredictionGUI(self.root)
            self.gui_loaded = True
        except Exception as e:
            print(f"Error loading main GUI: {e}")
            self.gui_loaded = False
        
        self.setup_test_interface()
        
    def setup_test_interface(self):
        """Setup the test interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Interactive Help and Manual Test", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Test buttons frame
        buttons_frame = ttk.LabelFrame(main_frame, text="Test Controls", padding="10")
        buttons_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Test buttons
        test_help_btn = ttk.Button(buttons_frame, text="Test Help Content", 
                                  command=self.test_help_content)
        test_help_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        test_manual_btn = ttk.Button(buttons_frame, text="Test Manual Content", 
                                    command=self.test_manual_content)
        test_manual_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        test_manual_window_btn = ttk.Button(buttons_frame, text="Open Manual Window", 
                                           command=self.test_manual_window)
        test_manual_window_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        test_help_tab_btn = ttk.Button(buttons_frame, text="Switch to Help Tab", 
                                      command=self.test_help_tab)
        test_help_tab_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        test_print_manual_btn = ttk.Button(buttons_frame, text="Print Full Manual", 
                                          command=self.test_print_manual)
        test_print_manual_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        test_open_manual_file_btn = ttk.Button(buttons_frame, text="Open Manual File", 
                                              command=self.test_open_manual_file)
        test_open_manual_file_btn.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Test Status", padding="10")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready to test help and manual functionality")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                wraplength=700)
        status_label.grid(row=0, column=0, sticky="w")
        
        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Test Results", padding="10")
        results_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10, 
                                   font=("Arial", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", 
                                         command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky="nsew")
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Clear results button
        clear_btn = ttk.Button(main_frame, text="Clear Results", 
                              command=self.clear_results)
        clear_btn.grid(row=4, column=0, pady=(10, 0))
        
    def log_result(self, message):
        """Log a result message."""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()
        
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete(1.0, tk.END)
        
    def test_help_content(self):
        """Test help content generation."""
        self.log_result("🧪 Testing Help Content...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'get_help_content'):
                help_content = self.main_gui.get_help_content()
                if help_content and len(help_content.strip()) > 0:
                    self.log_result(f"✅ Help content generated successfully ({len(help_content)} characters)")
                    
                    # Check for key sections
                    sections = ["OVERVIEW", "INTERFACE OVERVIEW", "USAGE TIPS", "TROUBLESHOOTING"]
                    found_sections = [s for s in sections if s in help_content]
                    self.log_result(f"✅ Found {len(found_sections)}/{len(sections)} key sections: {', '.join(found_sections)}")
                else:
                    self.log_result("❌ Help content is empty")
            else:
                self.log_result("❌ get_help_content method not found")
        except Exception as e:
            self.log_result(f"❌ Error testing help content: {e}")
            
    def test_manual_content(self):
        """Test manual content generation."""
        self.log_result("🧪 Testing Manual Content...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'get_full_manual_content'):
                manual_content = self.main_gui.get_full_manual_content()
                if manual_content and len(manual_content.strip()) > 0:
                    self.log_result(f"✅ Manual content generated successfully ({len(manual_content)} characters)")
                    
                    # Check for key sections
                    sections = ["OVERVIEW", "INSTALLATION", "DETAILED TAB GUIDE", "USAGE WORKFLOW"]
                    found_sections = [s for s in sections if s in manual_content]
                    self.log_result(f"✅ Found {len(found_sections)}/{len(sections)} key sections: {', '.join(found_sections)}")
                else:
                    self.log_result("❌ Manual content is empty")
            else:
                self.log_result("❌ get_full_manual_content method not found")
        except Exception as e:
            self.log_result(f"❌ Error testing manual content: {e}")
            
    def test_manual_window(self):
        """Test manual window creation."""
        self.log_result("🧪 Testing Manual Window...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'show_manual_window'):
                self.main_gui.show_manual_window()
                self.log_result("✅ Manual window opened successfully")
                self.log_result("   Check if the window appears with content")
            else:
                self.log_result("❌ show_manual_window method not found")
        except Exception as e:
            self.log_result(f"❌ Error opening manual window: {e}")
            
    def test_help_tab(self):
        """Test switching to help tab."""
        self.log_result("🧪 Testing Help Tab...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'control_notebook'):
                # Find help tab index
                help_tab_index = None
                for i in range(self.main_gui.control_notebook.index('end')):
                    tab_name = self.main_gui.control_notebook.tab(i, 'text')
                    if tab_name == '?':
                        help_tab_index = i
                        break
                
                if help_tab_index is not None:
                    self.main_gui.control_notebook.select(help_tab_index)
                    self.log_result("✅ Switched to help tab successfully")
                    self.log_result("   Check if help content is visible in the tab")
                else:
                    self.log_result("❌ Help tab (?) not found")
            else:
                self.log_result("❌ Control notebook not found")
        except Exception as e:
            self.log_result(f"❌ Error switching to help tab: {e}")
            
    def test_print_manual(self):
        """Test printing full manual."""
        self.log_result("🧪 Testing Print Full Manual...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'print_full_manual'):
                self.main_gui.print_full_manual()
                self.log_result("✅ Print full manual executed")
                self.log_result("   Check console output for printed manual")
            else:
                self.log_result("❌ print_full_manual method not found")
        except Exception as e:
            self.log_result(f"❌ Error printing manual: {e}")
            
    def test_open_manual_file(self):
        """Test opening manual file."""
        self.log_result("🧪 Testing Open Manual File...")
        
        if not self.gui_loaded:
            self.log_result("❌ Main GUI not loaded")
            return
            
        try:
            if hasattr(self.main_gui, 'open_manual_file'):
                self.main_gui.open_manual_file()
                self.log_result("✅ Open manual file executed")
                self.log_result("   Check if manual window opens")
            else:
                self.log_result("❌ open_manual_file method not found")
        except Exception as e:
            self.log_result(f"❌ Error opening manual file: {e}")
            
    def run(self):
        """Run the interactive test."""
        self.log_result("🚀 Interactive Help Test Started")
        self.log_result("Click the buttons above to test different functionality")
        self.log_result("Results will appear in this text area")
        self.log_result("=" * 50)
        
        self.root.mainloop()

def main():
    """Main function to run the interactive test."""
    print("🧪 Interactive Help Test")
    print("Testing help and manual functionality interactively")
    print()
    
    tester = InteractiveHelpTester()
    tester.run()

if __name__ == "__main__":
    main() 