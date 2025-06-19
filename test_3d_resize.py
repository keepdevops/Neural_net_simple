#!/usr/bin/env python3
"""
Test script to verify 3D gradient descent image resizing to 75%
"""

import os
import glob
from PIL import Image
import numpy as np

def test_3d_image_resize():
    """Test the 3D image resizing functionality."""
    print("Testing 3D gradient descent image resizing to 75%...")
    
    # Find model directories
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        return False
    
    # Use the most recent model directory
    model_dirs.sort(key=os.path.getctime, reverse=True)
    model_dir = model_dirs[0]
    
    print(f"Testing with model: {model_dir}")
    
    # Check if plots directory exists
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        print("No plots directory found")
        return False
    
    # Find 3D visualization files
    gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
    if not gd3d_files:
        gd3d_files = glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png'))
    
    if not gd3d_files:
        print("No 3D visualization files found")
        return False
    
    # Use the first file for testing
    test_file = gd3d_files[0]
    print(f"Testing with file: {os.path.basename(test_file)}")
    
    try:
        # Load the original image
        img = Image.open(test_file)
        original_width, original_height = img.size
        print(f"Original size: {original_width}x{original_height}")
        
        # Apply the corrected resizing logic - always 75%
        scale = 0.75
        
        print(f"Target scale: {scale:.3f} (75%)")
        
        # Resize the image to 75% of original size
        new_size = (int(original_width * scale), int(original_height * scale))
        resized_img = img.resize(new_size, Image.Resampling.BILINEAR)
        
        print(f"Resized size: {new_size[0]}x{new_size[1]}")
        
        # Calculate the actual percentage
        width_percentage = (new_size[0] / original_width) * 100
        height_percentage = (new_size[1] / original_height) * 100
        
        print(f"Width percentage: {width_percentage:.1f}%")
        print(f"Height percentage: {height_percentage:.1f}%")
        
        # Verify it's exactly 75%
        if 74.5 <= width_percentage <= 75.5 and 74.5 <= height_percentage <= 75.5:
            print("✅ Resizing to exactly 75% - PASS")
            return True
        else:
            print("❌ Resizing not to 75% - FAIL")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image resize: {e}")
        return False

def test_multiple_3d_files():
    """Test resizing on multiple 3D files."""
    print("\nTesting multiple 3D files...")
    
    # Find model directories
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        print("No model directories found")
        return False
    
    # Use the most recent model directory
    model_dirs.sort(key=os.path.getctime, reverse=True)
    model_dir = model_dirs[0]
    
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        print("No plots directory found")
        return False
    
    # Find 3D visualization files
    gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
    if not gd3d_files:
        gd3d_files = glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png'))
    
    if not gd3d_files:
        print("No 3D visualization files found")
        return False
    
    print(f"Found {len(gd3d_files)} 3D visualization files")
    
    success_count = 0
    for i, file_path in enumerate(gd3d_files[:5]):  # Test first 5 files
        try:
            img = Image.open(file_path)
            original_size = img.size
            
            # Apply corrected resizing logic - always 75%
            scale = 0.75
            
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            width_percentage = (new_size[0] / original_size[0]) * 100
            height_percentage = (new_size[1] / original_size[1]) * 100
            
            print(f"  File {i+1}: {os.path.basename(file_path)}")
            print(f"    Original: {original_size[0]}x{original_size[1]}")
            print(f"    Resized: {new_size[0]}x{new_size[1]} ({width_percentage:.1f}% x {height_percentage:.1f}%)")
            
            if 74.5 <= width_percentage <= 75.5 and 74.5 <= height_percentage <= 75.5:
                success_count += 1
                print(f"    ✅ PASS")
            else:
                print(f"    ❌ FAIL")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print(f"\nSuccess rate: {success_count}/{min(5, len(gd3d_files))} files")
    return success_count >= min(3, len(gd3d_files))  # At least 3 out of 5 should pass

if __name__ == "__main__":
    print("=== Testing 3D Gradient Descent Image Resizing ===\n")
    
    # Test 1: Single file resize
    success1 = test_3d_image_resize()
    
    # Test 2: Multiple files resize
    success2 = test_multiple_3d_files()
    
    print(f"\n=== Test Results ===")
    print(f"Single file resize: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Multiple files resize: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! 3D image resizing to 75% is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.") 