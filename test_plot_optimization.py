#!/usr/bin/env python3
"""
Test script to verify plot loading optimizations
"""

import os
import time
import glob
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

class PlotOptimizationTest:
    def __init__(self):
        self.plot_image_cache = {}
        self.max_cache_size = 50
        
    def test_original_loading(self, plot_files, max_width=600, max_height=400):
        """Test original loading method (slow)"""
        start_time = time.time()
        images = []
        
        for plot_file in plot_files[:5]:  # Test with 5 images
            try:
                # Load image
                img = Image.open(plot_file)
                # Resize image to fit
                img_width, img_height = img.size
                scale = min(max_width / img_width, 1.0)
                new_size = (int(img_width * scale), int(img_height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)  # Slow method
                photo = ImageTk.PhotoImage(img)
                images.append(photo)
                
            except Exception as e:
                print(f"Error loading {plot_file}: {e}")
                continue
        
        end_time = time.time()
        print(f"Original method: Loaded {len(images)} images in {end_time - start_time:.3f} seconds")
        return images
    
    def test_optimized_loading(self, plot_files, max_width=600, max_height=400):
        """Test optimized loading method (fast)"""
        start_time = time.time()
        images = []
        
        for plot_file in plot_files[:5]:  # Test with 5 images
            try:
                # Check cache first
                cache_key = f"{plot_file}_{max_width}_{max_height}"
                if cache_key in self.plot_image_cache:
                    photo = self.plot_image_cache[cache_key]
                    print(f"Using cached image: {os.path.basename(plot_file)}")
                else:
                    # Load and resize image with faster method
                    img = Image.open(plot_file)
                    img_width, img_height = img.size
                    
                    # Calculate optimal scale to fit within bounds
                    scale_x = max_width / img_width
                    scale_y = max_height / img_height
                    scale = min(scale_x, scale_y, 1.0)
                    
                    # Only resize if necessary
                    if scale < 1.0:
                        new_size = (int(img_width * scale), int(img_height * scale))
                        # Use faster resampling method
                        img = img.resize(new_size, Image.Resampling.BILINEAR)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    
                    # Cache the image
                    if len(self.plot_image_cache) >= self.max_cache_size:
                        oldest_key = next(iter(self.plot_image_cache))
                        del self.plot_image_cache[oldest_key]
                    
                    self.plot_image_cache[cache_key] = photo
                    print(f"Cached new image: {os.path.basename(plot_file)}")
                
                images.append(photo)
                
            except Exception as e:
                print(f"Error loading {plot_file}: {e}")
                continue
        
        end_time = time.time()
        print(f"Optimized method: Loaded {len(images)} images in {end_time - start_time:.3f} seconds")
        return images

def main():
    # Create a hidden root window for ImageTk
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Find a model directory with plots
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        root.destroy()
        return
    
    # Use the first model directory
    model_dir = model_dirs[0]
    plots_dir = os.path.join(model_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"No plots directory found in {model_dir}")
        root.destroy()
        return
    
    plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
    if not plot_files:
        print(f"No PNG plots found in {plots_dir}")
        root.destroy()
        return
    
    print(f"Found {len(plot_files)} plot files in {plots_dir}")
    print(f"Testing with first 5 plots: {[os.path.basename(f) for f in plot_files[:5]]}")
    
    # Create test instance
    test = PlotOptimizationTest()
    
    # Test original method
    print("\n=== Testing Original Loading Method ===")
    original_images = test.test_original_loading(plot_files)
    
    # Test optimized method
    print("\n=== Testing Optimized Loading Method ===")
    optimized_images = test.test_optimized_loading(plot_files)
    
    # Test cache hit
    print("\n=== Testing Cache Hit (Second Load) ===")
    cached_images = test.test_optimized_loading(plot_files)
    
    print(f"\nCache size: {len(test.plot_image_cache)}")
    print("Optimization test completed!")
    
    # Clean up
    root.destroy()

if __name__ == "__main__":
    main() 