#!/usr/bin/env python3
"""
Test script to verify 3D gradient descent image resizing
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np
import os
import glob

def test_3d_image_resize():
    """Test the 3D image resizing functionality"""
    
    # Create test window
    root = tk.Tk()
    root.title("3D Image Resize Test")
    root.geometry("800x600")
    
    # Create a frame for the 3D plot
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Create matplotlib figure and canvas
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def load_3d_image(image_path):
        """Load and resize a 3D visualization image"""
        try:
            # Clear the plot
            ax.clear()
            
            # Load the image
            img = Image.open(image_path)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Get canvas dimensions (approximate)
            canvas_width = canvas.get_tk_widget().winfo_width()
            canvas_height = canvas.get_tk_widget().winfo_height()
            
            # If canvas dimensions are not available yet, use defaults
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 400
            
            # Calculate the maximum available space (leave some margin)
            max_width = int(canvas_width * 0.9)
            max_height = int(canvas_height * 0.9)
            
            # Calculate optimal scale to fit within the canvas
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
            
            # Resize the image to fit within the canvas
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for matplotlib
            img_array = np.array(img)
            
            # Calculate extent to center the image properly
            # Use aspect ratio to determine the extent
            aspect_ratio = new_size[0] / new_size[1]
            
            # Create a square extent that maintains aspect ratio
            if aspect_ratio > 1:  # Wider than tall
                x_extent = [-1.0, 1.0]
                y_extent = [-1.0/aspect_ratio, 1.0/aspect_ratio]
            else:  # Taller than wide
                x_extent = [-aspect_ratio, aspect_ratio]
                y_extent = [-1.0, 1.0]
            
            # Display the image
            ax.imshow(img_array, aspect='auto', extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
            
            # Set title with filename
            filename = os.path.basename(image_path)
            ax.set_title(f"3D Gradient Descent Visualization\n{filename}", fontsize=12, pad=10)
            
            # Remove axis labels since this is an image
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add a border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            
            # Set background color
            ax.set_facecolor('#f0f0f0')
            
            # Draw the plot
            canvas.draw()
            
            print(f"Loaded 3D visualization image: {filename}")
            print(f"  Original size: {img_width}x{img_height}")
            print(f"  New size: {new_size[0]}x{new_size[1]} ({scale:.1%} of original)")
            print(f"  Canvas size: {canvas_width}x{canvas_height}")
            print(f"  Extent: x={x_extent}, y={y_extent}")
            
        except Exception as e:
            print(f"Error loading 3D visualization image: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f'Error loading 3D visualization image:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title("3D Gradient Descent Visualization")
            canvas.draw()
    
    # Find 3D gradient descent images
    model_dirs = glob.glob("model_*")
    if model_dirs:
        latest_model = max(model_dirs, key=os.path.getctime)
        plots_dir = os.path.join(latest_model, "plots")
        
        if os.path.exists(plots_dir):
            # Find 3D gradient descent images
            gd_images = glob.glob(os.path.join(plots_dir, "gradient_descent_3d_frame_*.png"))
            
            if gd_images:
                # Load the first image
                load_3d_image(gd_images[0])
                
                # Add buttons to cycle through images
                button_frame = ttk.Frame(root)
                button_frame.pack(fill="x", padx=10, pady=5)
                
                current_index = [0]  # Use list to make it mutable
                
                def next_image():
                    current_index[0] = (current_index[0] + 1) % len(gd_images)
                    load_3d_image(gd_images[current_index[0]])
                
                def prev_image():
                    current_index[0] = (current_index[0] - 1) % len(gd_images)
                    load_3d_image(gd_images[current_index[0]])
                
                ttk.Button(button_frame, text="Previous", command=prev_image).pack(side="left", padx=5)
                ttk.Button(button_frame, text="Next", command=next_image).pack(side="left", padx=5)
                
                # Add info label
                info_label = ttk.Label(button_frame, text=f"Image 1 of {len(gd_images)}")
                info_label.pack(side="right", padx=5)
                
                def update_info():
                    info_label.config(text=f"Image {current_index[0] + 1} of {len(gd_images)}")
                
                # Update info when buttons are clicked
                def next_with_info():
                    next_image()
                    update_info()
                
                def prev_with_info():
                    prev_image()
                    update_info()
                
                # Replace button commands
                for widget in button_frame.winfo_children():
                    if isinstance(widget, ttk.Button):
                        if widget.cget("text") == "Next":
                            widget.config(command=next_with_info)
                        elif widget.cget("text") == "Previous":
                            widget.config(command=prev_with_info)
                
                print(f"Found {len(gd_images)} 3D gradient descent images")
                print(f"Testing with model: {latest_model}")
            else:
                print("No 3D gradient descent images found")
                ax.text(0.5, 0.5, "No 3D gradient descent images found", 
                       ha='center', va='center', transform=ax.transAxes)
                canvas.draw()
        else:
            print(f"No plots directory found in {latest_model}")
            ax.text(0.5, 0.5, f"No plots directory found in {latest_model}", 
                   ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
    else:
        print("No model directories found")
        ax.text(0.5, 0.5, "No model directories found", 
               ha='center', va='center', transform=ax.transAxes)
        canvas.draw()
    
    # Add resize test button
    def test_resize():
        """Test resizing by changing window size"""
        current_size = root.geometry()
        if "800x600" in current_size:
            root.geometry("1000x800")
        else:
            root.geometry("800x600")
    
    ttk.Button(root, text="Test Resize", command=test_resize).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    test_3d_image_resize() 