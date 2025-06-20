#!/usr/bin/env python3
"""
Test script to verify multi-threading functionality for plot loading operations
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
import concurrent.futures
import os
import glob

class ThreadingTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-threading Test")
        self.root.geometry("600x400")
        
        # Initialize thread pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.futures = {}
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create test widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Multi-threading Test for Plot Loading", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Test buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky="nsew")
        
        # Test buttons
        self.test_button1 = ttk.Button(button_frame, text="Test Plot Loading (Threaded)", 
                                      command=self.test_plot_loading)
        self.test_button1.grid(row=0, column=0, padx=5, pady=5)
        
        self.test_button2 = ttk.Button(button_frame, text="Test Gradient Descent (Threaded)", 
                                      command=self.test_gradient_descent)
        self.test_button2.grid(row=0, column=1, padx=5, pady=5)
        
        self.test_button3 = ttk.Button(button_frame, text="Test Blocking Operation", 
                                      command=self.test_blocking_operation)
        self.test_button3.grid(row=0, column=2, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                     font=("Arial", 10))
        self.status_label.grid(row=2, column=0, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.grid(row=3, column=0, sticky="ew", pady=5)
        
        # Text area for output
        self.output_text = tk.Text(main_frame, height=10, width=70)
        self.output_text.grid(row=4, column=0, sticky="nsew", pady=5)
        
        # Scrollbar for text area
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.grid(row=4, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure text area grid weight
        main_frame.grid_rowconfigure(4, weight=1)
        
    def log_message(self, message):
        """Add message to output text area."""
        self.output_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        
    def test_plot_loading(self):
        """Test threaded plot loading operation."""
        self.log_message("Starting threaded plot loading test...")
        self.status_var.set("Loading plots in background...")
        self.progress_var.set(0)
        
        # Submit task to thread pool
        future = self.thread_pool.submit(self._plot_loading_worker)
        self.futures['plot_loading'] = future
        
        # Schedule callback to check completion
        self.root.after(100, self._check_plot_loading_complete)
        
    def _plot_loading_worker(self):
        """Worker function for plot loading test."""
        try:
            # Simulate plot loading work
            for i in range(10):
                time.sleep(0.2)  # Simulate work
                # Update progress (this won't work in background thread, but good for testing)
                progress = (i + 1) * 10
                print(f"Plot loading progress: {progress}%")
            
            # Simulate finding some plot files
            model_dirs = glob.glob("model_*")
            if model_dirs:
                plots_dir = os.path.join(model_dirs[0], 'plots')
                if os.path.exists(plots_dir):
                    plot_files = glob.glob(os.path.join(plots_dir, '*.png'))
                    return {
                        'success': True,
                        'message': f'Found {len(plot_files)} plot files',
                        'plot_count': len(plot_files)
                    }
            
            return {
                'success': True,
                'message': 'Plot loading completed (no plots found)',
                'plot_count': 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error in plot loading: {str(e)}'
            }
    
    def _check_plot_loading_complete(self):
        """Check if plot loading is complete."""
        if 'plot_loading' not in self.futures:
            return
        
        future = self.futures['plot_loading']
        if future.done():
            try:
                result = future.result()
                if result['success']:
                    self.status_var.set(result['message'])
                    self.progress_var.set(100)
                    self.log_message(f"✅ Plot loading completed: {result['message']}")
                    if 'plot_count' in result:
                        self.log_message(f"   Found {result['plot_count']} plot files")
                else:
                    self.status_var.set(result['message'])
                    self.log_message(f"❌ Plot loading failed: {result['message']}")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.log_message(f"❌ Exception in plot loading: {str(e)}")
            finally:
                del self.futures['plot_loading']
        else:
            # Update progress (simplified)
            self.progress_var.set(min(self.progress_var.get() + 10, 90))
            self.root.after(100, self._check_plot_loading_complete)
    
    def test_gradient_descent(self):
        """Test threaded gradient descent generation."""
        self.log_message("Starting threaded gradient descent test...")
        self.status_var.set("Generating gradient descent visualization...")
        self.progress_var.set(0)
        
        # Submit task to thread pool
        future = self.thread_pool.submit(self._gradient_descent_worker)
        self.futures['gradient_descent'] = future
        
        # Schedule callback to check completion
        self.root.after(100, self._check_gradient_descent_complete)
        
    def _gradient_descent_worker(self):
        """Worker function for gradient descent test."""
        try:
            # Simulate gradient descent generation work
            for i in range(15):
                time.sleep(0.1)  # Simulate work
                print(f"Gradient descent progress: {(i + 1) * 6.67:.1f}%")
            
            # Simulate finding model directories
            model_dirs = glob.glob("model_*")
            if model_dirs:
                return {
                    'success': True,
                    'message': f'Generated gradient descent visualization for {len(model_dirs)} models',
                    'model_count': len(model_dirs)
                }
            
            return {
                'success': True,
                'message': 'Gradient descent generation completed (no models found)',
                'model_count': 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error in gradient descent generation: {str(e)}'
            }
    
    def _check_gradient_descent_complete(self):
        """Check if gradient descent generation is complete."""
        if 'gradient_descent' not in self.futures:
            return
        
        future = self.futures['gradient_descent']
        if future.done():
            try:
                result = future.result()
                if result['success']:
                    self.status_var.set(result['message'])
                    self.progress_var.set(100)
                    self.log_message(f"✅ Gradient descent completed: {result['message']}")
                    if 'model_count' in result:
                        self.log_message(f"   Processed {result['model_count']} models")
                else:
                    self.status_var.set(result['message'])
                    self.log_message(f"❌ Gradient descent failed: {result['message']}")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.log_message(f"❌ Exception in gradient descent: {str(e)}")
            finally:
                del self.futures['gradient_descent']
        else:
            # Update progress
            self.progress_var.set(min(self.progress_var.get() + 6.67, 90))
            self.root.after(100, self._check_gradient_descent_complete)
    
    def test_blocking_operation(self):
        """Test blocking operation to show the difference."""
        self.log_message("Starting blocking operation test...")
        self.status_var.set("Running blocking operation (GUI will freeze)...")
        self.progress_var.set(0)
        
        # This will block the GUI
        for i in range(10):
            time.sleep(0.3)  # Simulate blocking work
            self.progress_var.set((i + 1) * 10)
            self.log_message(f"Blocking operation progress: {(i + 1) * 10}%")
            self.root.update()  # Force GUI update
        
        self.status_var.set("Blocking operation completed")
        self.log_message("✅ Blocking operation completed (notice GUI was frozen)")
    
    def cleanup(self):
        """Clean up thread pool."""
        try:
            # Cancel ongoing operations
            for operation_name, future in list(self.futures.items()):
                if not future.done():
                    future.cancel()
                    print(f"Cancelled operation: {operation_name}")
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=False)
            print("Thread pool cleaned up")
        except Exception as e:
            print(f"Error cleaning up thread pool: {e}")

def main():
    root = tk.Tk()
    app = ThreadingTestGUI(root)
    
    # Set up cleanup on window close
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 