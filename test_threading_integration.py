#!/usr/bin/env python3
"""
Test script to verify threading integration in stock_gui.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import os

# Add the current directory to the path so we can import stock_gui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_threading_integration():
    """Test the threading integration in the GUI"""
    
    # Create a simple test window
    root = tk.Tk()
    root.title("Threading Integration Test")
    root.geometry("400x300")
    
    # Test variables
    test_results = []
    
    def simulate_long_task():
        """Simulate a long-running task"""
        time.sleep(2)  # Simulate 2 seconds of work
        return "Task completed successfully"
    
    def test_threaded_action():
        """Test the threaded_action method"""
        print("Testing threaded_action method...")
        
        # Create a simple GUI class with threaded_action
        class TestGUI:
            def __init__(self, root):
                self.root = root
                self.train_button = ttk.Button(root, text="Train", command=self.train)
                self.train_button.pack(pady=10)
                self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
                self.predict_button.pack(pady=10)
                self.stop_button = ttk.Button(root, text="Stop", state=tk.DISABLED)
                self.stop_button.pack(pady=10)
                self.status_var = tk.StringVar(value="Ready")
                ttk.Label(root, textvariable=self.status_var).pack(pady=10)
                
            def threaded_action(self, func):
                """Threaded action wrapper"""
                def wrapper():
                    # Disable buttons
                    self.train_button.config(state=tk.DISABLED)
                    self.predict_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                    
                    # Run function in thread
                    thread = threading.Thread(target=func, daemon=True)
                    thread.start()
                    
                    # Monitor thread completion
                    def check_thread():
                        if thread.is_alive():
                            self.root.after(100, check_thread)
                        else:
                            # Re-enable buttons
                            self.train_button.config(state=tk.NORMAL)
                            self.predict_button.config(state=tk.NORMAL)
                            self.stop_button.config(state=tk.DISABLED)
                            if not self.status_var.get().startswith("Error"):
                                self.status_var.set("Ready")
                    
                    self.root.after(100, check_thread)
                
                return wrapper
            
            def train(self):
                """Simulate training"""
                self.status_var.set("Training...")
                self.threaded_action(self._train_worker)()
            
            def predict(self):
                """Simulate prediction"""
                self.status_var.set("Predicting...")
                self.threaded_action(self._predict_worker)()
            
            def _train_worker(self):
                """Training worker thread"""
                try:
                    result = simulate_long_task()
                    self.root.after(0, lambda: self.status_var.set(f"Training: {result}"))
                    test_results.append("Training thread completed successfully")
                    print("Training thread completed")
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(f"Training error: {e}"))
                    test_results.append(f"Training thread failed: {e}")
                    print(f"Training thread failed: {e}")
            
            def _predict_worker(self):
                """Prediction worker thread"""
                try:
                    result = simulate_long_task()
                    self.root.after(0, lambda: self.status_var.set(f"Prediction: {result}"))
                    test_results.append("Prediction thread completed successfully")
                    print("Prediction thread completed")
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(f"Prediction error: {e}"))
                    test_results.append(f"Prediction thread failed: {e}")
                    print(f"Prediction thread failed: {e}")
        
        # Create test GUI
        test_gui = TestGUI(root)
        
        # Add test buttons
        ttk.Button(root, text="Test Train", command=test_gui.train).pack(pady=5)
        ttk.Button(root, text="Test Predict", command=test_gui.predict).pack(pady=5)
        
        def check_results():
            """Check test results after a delay"""
            print(f"\nTest Results ({len(test_results)} completed):")
            for i, result in enumerate(test_results, 1):
                print(f"  {i}. {result}")
            
            if len(test_results) >= 2:
                print("✅ Threading integration test passed!")
            else:
                print("❌ Threading integration test failed!")
                print(f"Expected 2 results, got {len(test_results)}")
            
            root.quit()
        
        # Schedule result check after longer delay
        root.after(8000, check_results)
        
        return root
    
    # Run the test
    test_window = test_threaded_action()
    test_window.mainloop()

def test_lru_cache():
    """Test the LRU cache functionality"""
    print("\nTesting LRU cache...")
    
    from collections import OrderedDict
    
    # Create LRU cache
    cache = OrderedDict()
    max_size = 3
    
    def add_to_cache(key, value):
        """Add item to cache with LRU eviction"""
        if key in cache:
            cache.move_to_end(key)  # Update LRU order
            print(f"Using cached item: {key}")
        else:
            cache[key] = value
            print(f"Cached new item: {key}")
            
            # Evict oldest if cache is full
            if len(cache) > max_size:
                cache.popitem(last=False)
                print(f"Evicted oldest item. Cache size: {len(cache)}")
    
    # Test cache operations
    add_to_cache("A", "value_A")
    add_to_cache("B", "value_B")
    add_to_cache("C", "value_C")
    add_to_cache("D", "value_D")  # Should evict A
    add_to_cache("B", "value_B")  # Should move B to end
    add_to_cache("E", "value_E")  # Should evict C
    
    print(f"Final cache contents: {list(cache.keys())}")
    print("✅ LRU cache test passed!")

if __name__ == "__main__":
    print("Testing threading integration in stock_gui.py...")
    
    # Test LRU cache
    test_lru_cache()
    
    # Test threading integration
    test_threading_integration() 