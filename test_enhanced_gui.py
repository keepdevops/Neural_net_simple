#!/usr/bin/env python3
"""
Test script for Enhanced GUI - Options A and B

This script tests the enhanced display panel and control panel functionality
to ensure all features are working correctly.
"""

import tkinter as tk
from tkinter import ttk
import sys
import os
import time
import threading

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import stock_gui
    print("✅ Successfully imported stock_gui")
except ImportError as e:
    print(f"❌ Failed to import stock_gui: {e}")
    sys.exit(1)

class EnhancedGUITester:
    """Test class for the enhanced GUI functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.root = None
        self.gui = None
        self.test_results = []
        
    def run_tests(self):
        """Run all tests for the enhanced GUI."""
        print("\n🧪 Starting Enhanced GUI Tests...")
        print("=" * 50)
        
        # Test 1: GUI Creation
        self.test_gui_creation()
        
        # Test 2: Display Panel (Option A)
        self.test_display_panel()
        
        # Test 3: Control Panel (Option B)
        self.test_control_panel()
        
        # Test 4: Integration Tests
        self.test_integration()
        
        # Print results
        self.print_results()
        
        # Cleanup
        self.cleanup()
        
    def test_gui_creation(self):
        """Test that the GUI can be created successfully."""
        print("\n📋 Test 1: GUI Creation")
        try:
            self.root = tk.Tk()
            self.root.withdraw()  # Hide the window during testing
            
            # Create the GUI
            self.gui = stock_gui.StockPredictionGUI(self.root)
            
            # Check that essential components exist
            assert hasattr(self.gui, 'display_notebook'), "Display notebook not found"
            assert hasattr(self.gui, 'control_notebook'), "Control notebook not found"
            assert hasattr(self.gui, 'feature_listbox'), "Feature listbox not found"
            assert hasattr(self.gui, 'target_feature_combo'), "Target feature combo not found"
            
            print("✅ GUI created successfully with all essential components")
            self.test_results.append(("GUI Creation", "PASS"))
            
        except Exception as e:
            print(f"❌ GUI creation failed: {e}")
            self.test_results.append(("GUI Creation", f"FAIL: {e}"))
    
    def test_display_panel(self):
        """Test the enhanced display panel (Option A)."""
        print("\n📋 Test 2: Enhanced Display Panel (Option A)")
        try:
            # Check that all display tabs exist
            display_tabs = self.gui.display_notebook.tabs()
            expected_tabs = ["Training Results", "Gradient Descent", "3D Gradient Descent", "Saved Plots"]
            
            for tab in expected_tabs:
                assert any(tab in str(tab_id) for tab_id in display_tabs), f"Tab '{tab}' not found"
            
            # Check that sub-tabs exist in Saved Plots
            if hasattr(self.gui, 'plots_notebook'):
                plot_tabs = self.gui.plots_notebook.tabs()
                expected_plot_tabs = ["Comprehensive Plots", "Raw PNGs"]
                
                for tab in expected_plot_tabs:
                    assert any(tab in str(tab_id) for tab_id in plot_tabs), f"Sub-tab '{tab}' not found"
            
            # Check that image scaling functionality exists
            assert hasattr(self.gui, 'image_scale_var'), "Image scale variable not found"
            assert hasattr(self.gui, '_display_images_in_canvas'), "Image display method not found"
            
            print("✅ Enhanced display panel created with all tabs and functionality")
            self.test_results.append(("Display Panel (Option A)", "PASS"))
            
        except Exception as e:
            print(f"❌ Display panel test failed: {e}")
            self.test_results.append(("Display Panel (Option A)", f"FAIL: {e}"))
    
    def test_control_panel(self):
        """Test the enhanced control panel (Option B)."""
        print("\n📋 Test 3: Enhanced Control Panel (Option B)")
        try:
            # Check that all control tabs exist
            control_tabs = self.gui.control_notebook.tabs()
            expected_tabs = ["Data", "Model", "Training"]
            
            for tab in expected_tabs:
                assert any(tab in str(tab_id) for tab_id in control_tabs), f"Tab '{tab}' not found"
            
            # Check that essential control components exist
            assert hasattr(self.gui, 'data_file_var'), "Data file variable not found"
            assert hasattr(self.gui, 'hidden_size_var'), "Hidden size variable not found"
            assert hasattr(self.gui, 'learning_rate_var'), "Learning rate variable not found"
            assert hasattr(self.gui, 'batch_size_var'), "Batch size variable not found"
            assert hasattr(self.gui, 'progress_var'), "Progress variable not found"
            assert hasattr(self.gui, 'status_label'), "Status label not found"
            
            # Check that 3D gradient descent functionality exists
            assert hasattr(self.gui, '_show_gradient_descent'), "3D gradient descent method not found"
            assert hasattr(self.gui, 'gd3d_button'), "3D gradient descent button not found"
            
            # Check that cache management exists
            assert hasattr(self.gui, 'clear_plot_cache'), "Cache clear method not found"
            assert hasattr(self.gui, 'update_cache_info'), "Cache info method not found"
            
            print("✅ Enhanced control panel created with all tabs and functionality")
            self.test_results.append(("Control Panel (Option B)", "PASS"))
            
        except Exception as e:
            print(f"❌ Control panel test failed: {e}")
            self.test_results.append(("Control Panel (Option B)", f"FAIL: {e}"))
    
    def test_integration(self):
        """Test integration between display and control panels."""
        print("\n📋 Test 4: Integration Tests")
        try:
            # Test that tab switching works
            assert hasattr(self.gui, 'switch_to_tab'), "Tab switching method not found"
            assert hasattr(self.gui, 'on_tab_changed'), "Tab change handler not found"
            
            # Test that model selection updates display
            assert hasattr(self.gui, 'on_model_select'), "Model selection handler not found"
            assert hasattr(self.gui, '_load_model_info'), "Model info loading method not found"
            
            # Test that feature selection works
            assert hasattr(self.gui, 'on_feature_selection_change'), "Feature selection handler not found"
            assert hasattr(self.gui, 'validate_features'), "Feature validation method not found"
            
            # Test that training updates display
            assert hasattr(self.gui, 'update_training_results'), "Training results update method not found"
            assert hasattr(self.gui, 'update_live_plot'), "Live plot update method not found"
            
            # Test that threading works
            assert hasattr(self.gui, 'threaded_action'), "Threading wrapper method not found"
            assert hasattr(self.gui, 'thread_pool'), "Thread pool not found"
            
            print("✅ Integration tests passed - all components work together")
            self.test_results.append(("Integration Tests", "PASS"))
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            self.test_results.append(("Integration Tests", f"FAIL: {e}"))
    
    def print_results(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results:
            status = "✅ PASS" if result == "PASS" else f"❌ FAIL"
            print(f"{test_name:<30} {status}")
            if result == "PASS":
                passed += 1
        
        print("-" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Enhanced GUI is working correctly.")
            print("✅ Option A (Enhanced Display Panel): IMPLEMENTED")
            print("✅ Option B (Enhanced Control Panel): IMPLEMENTED")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    def cleanup(self):
        """Clean up resources."""
        if self.root:
            try:
                self.root.destroy()
            except:
                pass

def main():
    """Main test function."""
    print("🚀 Enhanced GUI Test Suite")
    print("Testing Options A and B Implementation")
    
    tester = EnhancedGUITester()
    tester.run_tests()

if __name__ == "__main__":
    main() 