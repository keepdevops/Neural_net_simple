#!/usr/bin/env python3
"""
Test script for Matplotlib live plot functionality.
This script simulates the live training plot updates to verify they work correctly.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

class TestLivePlot:
    def __init__(self, root):
        self.root = root
        self.root.title("Test Matplotlib Live Plot")
        self.root.geometry("400x300")
        
        # Initialize plot variables
        self.live_plot_fig = None
        self.live_plot_ax = None
        self.live_plot_epochs = []
        self.live_plot_losses = []
        self.live_plot_window_open = False
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Matplotlib Live Plot Test", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons
        self.open_btn = ttk.Button(main_frame, text="Open Live Plot", command=self.open_live_plot)
        self.open_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.start_btn = ttk.Button(main_frame, text="Start Simulated Training", command=self.start_simulation)
        self.start_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.close_btn = ttk.Button(main_frame, text="Close Plot", command=self.close_live_plot)
        self.close_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Info
        info_text = """
This test simulates live training updates using Matplotlib.
        
1. Click 'Open Live Plot' to create a matplotlib window
2. Click 'Start Simulated Training' to begin updating the plot
3. The plot will update in real-time with simulated loss values
4. Click 'Close Plot' to close the matplotlib window
        
This demonstrates that Matplotlib works correctly with threading.
        """
        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=4, column=0, columnspan=2, pady=10)
        
    def open_live_plot(self):
        """Open a live training plot window using Matplotlib."""
        try:
            # Create a new matplotlib window
            self.live_plot_fig, self.live_plot_ax = plt.subplots(figsize=(10, 6))
            self.live_plot_window_open = True
            self.live_plot_epochs = []
            self.live_plot_losses = []
            
            # Set up the plot
            self.live_plot_ax.set_title('Test Live Training Loss')
            self.live_plot_ax.set_xlabel('Epoch')
            self.live_plot_ax.set_ylabel('Loss')
            self.live_plot_ax.grid(True, alpha=0.3)
            
            # Show the plot in non-blocking mode
            plt.show(block=False)
            
            self.status_var.set("Live plot opened successfully")
            print("Live plot opened successfully")
            
        except Exception as e:
            print(f"Error opening live plot: {e}")
            self.live_plot_window_open = False
            self.status_var.set(f"Error: {str(e)}")

    def update_live_plot(self, epoch, loss):
        """Update the live matplotlib plot with new data."""
        if not self.live_plot_window_open:
            return
        
        try:
            # Add new data point
            self.live_plot_epochs.append(epoch)
            self.live_plot_losses.append(loss)
            
            # Clear and redraw the plot
            self.live_plot_ax.clear()
            self.live_plot_ax.plot(self.live_plot_epochs, self.live_plot_losses, 'b-', linewidth=2, marker='o', markersize=4)
            
            # Update labels and title
            self.live_plot_ax.set_title('Test Live Training Loss')
            self.live_plot_ax.set_xlabel('Epoch')
            self.live_plot_ax.set_ylabel('Loss')
            self.live_plot_ax.grid(True, alpha=0.3)
            
            # Auto-scale the axes
            if len(self.live_plot_epochs) > 1:
                self.live_plot_ax.set_xlim(0, max(self.live_plot_epochs) + 1)
                if len(self.live_plot_losses) > 1:
                    min_loss = min(self.live_plot_losses)
                    max_loss = max(self.live_plot_losses)
                    margin = (max_loss - min_loss) * 0.1
                    self.live_plot_ax.set_ylim(min_loss - margin, max_loss + margin)
            
            # Update the display
            self.live_plot_fig.canvas.draw()
            self.live_plot_fig.canvas.flush_events()
            
            print(f"Live plot updated: Epoch {epoch}, Loss {loss:.6f}")
            
        except Exception as e:
            print(f"Error updating live plot: {e}")

    def close_live_plot(self):
        """Close the live training plot."""
        self.live_plot_window_open = False
        if hasattr(self, 'live_plot_fig') and self.live_plot_fig is not None:
            plt.close(self.live_plot_fig)
        self.live_plot_fig = None
        self.live_plot_ax = None
        self.live_plot_epochs = []
        self.live_plot_losses = []
        self.status_var.set("Live plot closed")
        print("Live plot closed")

    def start_simulation(self):
        """Start a simulated training process in a separate thread."""
        if not self.live_plot_window_open:
            self.status_var.set("Please open live plot first")
            return
            
        self.status_var.set("Starting simulation...")
        
        # Start simulation in a separate thread
        simulation_thread = threading.Thread(target=self._run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()
        
    def _run_simulation(self):
        """Run the simulation in a separate thread."""
        try:
            # Simulate training for 20 epochs
            for epoch in range(1, 21):
                # Generate a realistic loss value (decreasing with some noise)
                base_loss = 0.5 * np.exp(-epoch / 10)  # Exponential decay
                noise = np.random.normal(0, 0.02)  # Small random noise
                loss = max(0.01, base_loss + noise)  # Ensure positive
                
                # Update the plot on the main thread
                self.root.after(0, lambda e=epoch, l=loss: self.update_live_plot(e, l))
                self.root.after(0, lambda: self.status_var.set(f"Simulating epoch {epoch}/20"))
                
                # Wait a bit to simulate training time
                time.sleep(0.5)
            
            # Simulation complete
            self.root.after(0, lambda: self.status_var.set("Simulation completed!"))
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            self.root.after(0, lambda: self.status_var.set(f"Simulation error: {str(e)}"))

def main():
    root = tk.Tk()
    app = TestLivePlot(root)
    root.mainloop()

if __name__ == "__main__":
    main() 