#!/usr/bin/env python3
"""
Simple test for 3D visualization loading
"""

import os
import glob
from PIL import Image
import numpy as np

def test_3d_visualization_loading():
    """Test the 3D visualization loading functionality."""
    print("Testing 3D visualization loading...")
    
    # Find model directories
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        return False
    
    # Use the first model directory
    model_dir = model_dirs[0]
    plots_dir = os.path.join(model_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"No plots directory found in {model_dir}")
        return False
    
    # Find gradient descent 3D visualization files
    gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
    
    if not gd3d_files:
        # Fallback to any gradient descent related image
        gd3d_files = sorted(glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png')))
    
    if not gd3d_files:
        print("No 3D visualization files found")
        return False
    
    print(f"Found {len(gd3d_files)} 3D visualization files")
    
    # Use the last (most recent) frame or a middle frame for better visualization
    if len(gd3d_files) > 10:
        # Use a middle frame for better visualization
        selected_file = gd3d_files[len(gd3d_files) // 2]
    else:
        # Use the last frame
        selected_file = gd3d_files[-1]
    
    print(f"Selected file: {os.path.basename(selected_file)}")
    
    try:
        # Load the image
        img = Image.open(selected_file)
        
        # Get image dimensions
        img_width, img_height = img.size
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Calculate display dimensions
        max_width = 800
        max_height = 600
        
        # Calculate scale to fit
        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y, 1.0)
        
        print(f"Scale factor: {scale}")
        
        # Resize if necessary
        if scale < 1.0:
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            print(f"Resized to: {new_size}")
        
        # Convert to numpy array for matplotlib
        img_array = np.array(img)
        print(f"Array shape: {img_array.shape}")
        
        print("✅ 3D visualization loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading 3D visualization image: {e}")
        return False

if __name__ == "__main__":
    success = test_3d_visualization_loading()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!") 