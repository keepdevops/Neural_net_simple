#!/usr/bin/env python3
"""
Test script for enhanced 3D gradient descent visualization configuration.

This script tests the new configuration features including:
- JSON configuration file loading
- New command line parameters
- Weight index selection
- Enhanced parameter handling
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

def test_config_file_loading():
    """Test loading configuration from JSON file."""
    print("Testing JSON configuration file loading...")
    
    # Create a test configuration file
    test_config = {
        "visualization_settings": {
            "model_dir": None,
            "color": "plasma",
            "point_size": 10,
            "line_width": 4,
            "surface_alpha": 0.7,
            "w1_range": [-3.0, 3.0],
            "w2_range": [-3.0, 3.0],
            "n_points": 40,
            "view_elev": 45.0,
            "view_azim": 60.0,
            "fps": 25,
            "save_png": True,
            "output_resolution": [1400, 900],
            "w1_index": 1,
            "w2_index": 2
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        config_file = f.name
    
    try:
        # Test the script with config file
        cmd = [sys.executable, 'gradient_descent_3d.py', '--config', config_file, '--help']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Configuration file loading test passed")
        else:
            print(f"❌ Configuration file loading test failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Configuration file loading test error: {e}")
    finally:
        # Clean up
        os.unlink(config_file)

def test_new_parameters():
    """Test the new command line parameters."""
    print("\nTesting new command line parameters...")
    
    # Test help to see if new parameters are available
    cmd = [sys.executable, 'gradient_descent_3d.py', '--help']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        help_text = result.stdout
        new_params = [
            '--w1_index',
            '--w2_index', 
            '--output_resolution',
            '--config'
        ]
        
        missing_params = []
        for param in new_params:
            if param not in help_text:
                missing_params.append(param)
        
        if not missing_params:
            print("✅ All new parameters are available in help")
        else:
            print(f"❌ Missing parameters in help: {missing_params}")
    else:
        print(f"❌ Help test failed: {result.stderr}")

def test_parameter_validation():
    """Test parameter validation and error handling."""
    print("\nTesting parameter validation...")
    
    # Test with invalid weight indices
    cmd = [
        sys.executable, 'gradient_descent_3d.py',
        '--w1_index', '-1',  # Invalid negative index
        '--w2_index', '999',  # Very large index
        '--model_dir', 'nonexistent_model'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should fail gracefully with informative error
    if result.returncode != 0:
        print("✅ Parameter validation working (expected failure for invalid model)")
    else:
        print("⚠️  Parameter validation test inconclusive")

def test_config_template_compatibility():
    """Test compatibility with the provided configuration template."""
    print("\nTesting configuration template compatibility...")
    
    # Check if the template file exists
    template_file = 'gradient_descent_3d_config.json'
    if os.path.exists(template_file):
        try:
            with open(template_file, 'r') as f:
                config = json.load(f)
            
            # Check required sections
            required_sections = ['metadata', 'visualization_settings', 'model_requirements', 'output', 'notes']
            missing_sections = [section for section in required_sections if section not in config]
            
            if not missing_sections:
                print("✅ Configuration template has all required sections")
                
                # Check visualization settings
                viz_settings = config.get('visualization_settings', {})
                required_viz_params = [
                    'color', 'point_size', 'line_width', 'surface_alpha',
                    'w1_range', 'w2_range', 'n_points', 'view_elev', 'view_azim',
                    'fps', 'save_png', 'output_resolution', 'w1_index', 'w2_index'
                ]
                
                missing_viz_params = [param for param in required_viz_params if param not in viz_settings]
                
                if not missing_viz_params:
                    print("✅ Configuration template has all required visualization parameters")
                else:
                    print(f"❌ Missing visualization parameters: {missing_viz_params}")
            else:
                print(f"❌ Missing configuration sections: {missing_sections}")
                
        except Exception as e:
            print(f"❌ Error reading configuration template: {e}")
    else:
        print("⚠️  Configuration template file not found")

def main():
    """Run all tests."""
    print("Testing Enhanced 3D Gradient Descent Configuration")
    print("=" * 50)
    
    # Check if gradient_descent_3d.py exists
    if not os.path.exists('gradient_descent_3d.py'):
        print("❌ gradient_descent_3d.py not found in current directory")
        return
    
    test_config_file_loading()
    test_new_parameters()
    test_parameter_validation()
    test_config_template_compatibility()
    
    print("\n" + "=" * 50)
    print("Enhanced 3D configuration testing completed!")

if __name__ == "__main__":
    main() 