#!/usr/bin/env python3
"""
Test script to verify 3D visualization loading functionality
"""

import os
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class Test3DVisualization:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Visualization Test")
        self.root.geometry("800x600")
        
        # Create 3D plot
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Test button
        test_button = ttk.Button(self.root, text="Test 3D Visualization Loading", 
                                command=self.test_3d_loading)
        test_button.pack(pady=10)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready to test")
        self.status_label.pack(pady=5)
        
    def find_3d_visualization_files(self):
        """Find 3D gradient descent visualization files."""
        model_dirs = glob.glob("model_*")
        if not model_dirs:
            return []
        
        # Use the first model directory
        model_dir = model_dirs[0]
        plots_dir = os.path.join(model_dir, 'plots')
        
        if not os.path.exists(plots_dir):
            return []
        
        # Find gradient descent 3D visualization files
        gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
        
        if not gd3d_files:
            # Fallback to any gradient descent related image
            gd3d_files = sorted(glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png')))
        
        return gd3d_files
    
    def load_3d_visualization_image(self, image_path):
        """Load a 3D visualization image into the 3D plot."""
        try:
            # Clear the 3D plot
            self.ax.clear()
            
            # Load the image
            img = Image.open(image_path)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Calculate display dimensions
            max_width = 800
            max_height = 600
            
            # Calculate scale to fit
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            scale = min(scale_x, scale_y, 1.0)
            
            # Resize if necessary
            if scale < 1.0:
                new_size = (int(img_width * scale), int(img_height * scale))
                img = img.resize(new_size, Image.Resampling.BILINEAR)
            
            # Convert to numpy array for matplotlib
            img_array = np.array(img)
            
            # Create a 2D plot in the 3D tab (since we're showing a static image)
            self.ax.imshow(img_array, aspect='auto', extent=[-2, 2, -2, 2])
            
            # Set title with filename
            filename = os.path.basename(image_path)
            self.ax.set_title(f"3D Gradient Descent Visualization\n{filename}")
            
            # Remove axis labels since this is an image
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Draw the plot
            self.canvas.draw()
            
            return True
            
        except Exception as e:
            print(f"Error loading 3D visualization image: {e}")
            # Fallback to error message
            self.ax.clear()
            self.ax.text(0.5, 0.5, 0.5, f'Error loading 3D visualization image: {str(e)}', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.ax.set_title("3D Gradient Descent Visualization")
            self.canvas.draw()
            return False
    
    def test_3d_loading(self):
        """Test the 3D visualization loading functionality."""
        try:
            self.status_label.config(text="Searching for 3D visualization files...")
            self.root.update()
            
            # Find 3D visualization files
            gd3d_files = self.find_3d_visualization_files()
            
            if not gd3d_files:
                self.status_label.config(text="No 3D visualization files found")
                # Show placeholder
                self.ax.clear()
                self.ax.text(0.5, 0.5, 0.5, 'No 3D gradient descent visualization found.\nRun gradient_descent_3d.py to generate one.', 
                            ha='center', va='center', transform=self.ax.transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.ax.set_title("3D Gradient Descent Visualization")
                self.canvas.draw()
                return
            
            self.status_label.config(text=f"Found {len(gd3d_files)} 3D visualization files")
            
            # Use the last (most recent) frame or a middle frame for better visualization
            if len(gd3d_files) > 10:
                # Use a middle frame for better visualization
                selected_file = gd3d_files[len(gd3d_files) // 2]
            else:
                # Use the last frame
                selected_file = gd3d_files[-1]
            
            self.status_label.config(text=f"Loading: {os.path.basename(selected_file)}")
            self.root.update()
            
            # Load and display the image
            success = self.load_3d_visualization_image(selected_file)
            
            if success:
                self.status_label.config(text=f"Successfully loaded: {os.path.basename(selected_file)}")
            else:
                self.status_label.config(text="Failed to load 3D visualization")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Error in test_3d_loading: {e}")
    
    def run(self):
        """Run the test application."""
        self.root.mainloop()

def main():
    test = Test3DVisualization()
    test.run()

if __name__ == "__main__":
    main() 