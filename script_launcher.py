#!/usr/bin/env python3
"""
Script Launcher Utility for Stock Prediction GUI

This module provides a robust way to launch scripts from the GUI
regardless of the working directory, with proper error handling.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptLauncher:
    """Utility class for launching scripts with robust path resolution."""
    
    def __init__(self):
        self.script_cache = {}
        self.working_dir_cache = None
    
    def find_script(self, script_name):
        """
        Find a script using multiple search strategies.
        
        Args:
            script_name (str): Name of the script to find
            
        Returns:
            str: Full path to the script if found, None otherwise
        """
        # Check cache first
        if script_name in self.script_cache:
            return self.script_cache[script_name]
        
        # Define search locations in order of preference
        possible_paths = []
        
        # 1. Same directory as this launcher
        launcher_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(launcher_dir, script_name))
        
        # 2. Current working directory
        possible_paths.append(os.path.join(os.getcwd(), script_name))
        
        # 3. Simple subdirectory (for when GUI is run from parent directory)
        possible_paths.append(os.path.join('simple', script_name))
        
        # 4. Parent directory
        possible_paths.append(os.path.join(os.path.dirname(os.getcwd()), script_name))
        
        # 5. Just the filename (if in PATH)
        possible_paths.append(script_name)
        
        # Search for the script
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                abs_path = os.path.abspath(path)
                self.script_cache[script_name] = abs_path
                logger.info(f"Found {script_name} at: {abs_path}")
                return abs_path
        
        logger.error(f"Could not find {script_name} in any of the search paths")
        return None
    
    def get_working_directory(self):
        """
        Get the appropriate working directory for running scripts.
        
        Returns:
            str: Path to the directory where scripts should be run from
        """
        if self.working_dir_cache:
            return self.working_dir_cache
        
        # Try to find the simple directory
        simple_dir = self.find_script('stock_gui.py')
        if simple_dir:
            working_dir = os.path.dirname(simple_dir)
        else:
            # Fallback to current directory
            working_dir = os.getcwd()
        
        self.working_dir_cache = working_dir
        logger.info(f"Working directory: {working_dir}")
        return working_dir
    
    def launch_script(self, script_name, args=None, timeout=30):
        """
        Launch a script with the given arguments.
        
        Args:
            script_name (str): Name of the script to launch
            args (list): Arguments to pass to the script
            timeout (int): Timeout in seconds
            
        Returns:
            tuple: (success, stdout, stderr)
        """
        if args is None:
            args = []
        
        # Find the script
        script_path = self.find_script(script_name)
        if not script_path:
            return False, "", f"Could not find {script_name}"
        
        # Get working directory
        working_dir = self.get_working_directory()
        
        # Build command
        cmd = [sys.executable, script_path] + args
        
        logger.info(f"Launching command: {' '.join(cmd)}")
        logger.info(f"Working directory: {working_dir}")
        
        try:
            # Run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            if success:
                logger.info(f"Script {script_name} executed successfully")
            else:
                logger.error(f"Script {script_name} failed with return code {result.returncode}")
                logger.error(f"stderr: {stderr}")
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            error_msg = f"Script {script_name} timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Error launching {script_name}: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg
    
    def test_script_availability(self):
        """
        Test if all required scripts are available.
        
        Returns:
            dict: Dictionary with script names as keys and availability as values
        """
        required_scripts = ['predict.py', 'gradient_descent_3d.py', 'view_results.py']
        results = {}
        
        for script in required_scripts:
            path = self.find_script(script)
            results[script] = path is not None
            if path:
                logger.info(f"✅ {script} available at: {path}")
            else:
                logger.error(f"❌ {script} not found")
        
        return results
    
    def get_script_help(self, script_name):
        """
        Get help information for a script.
        
        Args:
            script_name (str): Name of the script
            
        Returns:
            str: Help text or error message
        """
        success, stdout, stderr = self.launch_script(script_name, ['--help'])
        if success:
            return stdout
        else:
            return f"Error getting help for {script_name}: {stderr}"

# Global instance
launcher = ScriptLauncher()

def launch_gradient_descent(model_dir, save_png=True, **kwargs):
    """
    Launch gradient descent visualization.
    
    Args:
        model_dir (str): Model directory path
        save_png (bool): Whether to save PNG snapshots
        **kwargs: Additional arguments for gradient descent script
        
    Returns:
        tuple: (success, stdout, stderr)
    """
    args = ['--model_dir', model_dir]
    
    if save_png:
        args.append('--save_png')
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            args.extend([f'--{key}', str(value)])
    
    # Use longer timeout for gradient descent visualization (120 seconds)
    return launcher.launch_script('gradient_descent_3d.py', args, timeout=120)

def launch_prediction(data_file, model_dir, x_features, y_feature):
    """
    Launch prediction script.
    
    Args:
        data_file (str): Input data file path
        model_dir (str): Model directory path
        x_features (list): Input features
        y_feature (str): Target feature
        
    Returns:
        tuple: (success, stdout, stderr)
    """
    args = [
        data_file,
        '--model_dir', model_dir,
        '--x_features', ','.join(x_features),
        '--y_feature', y_feature
    ]
    
    return launcher.launch_script('predict.py', args)

def launch_view_results(model_dir, plot_type='all', save_plot=False):
    """
    Launch view results script.
    
    Args:
        model_dir (str): Model directory path
        plot_type (str): Type of plot to generate
        save_plot (bool): Whether to save plots
        
    Returns:
        tuple: (success, stdout, stderr)
    """
    args = [model_dir, '--plot-type', plot_type]
    
    if save_plot:
        args.append('--save-plot')
    
    return launcher.launch_script('view_results.py', args)

# Test function
def test_launcher():
    """Test the script launcher functionality."""
    print("Testing Script Launcher...")
    print("=" * 50)
    
    # Test script availability
    print("\n1. Testing script availability:")
    availability = launcher.test_script_availability()
    for script, available in availability.items():
        status = "✅ Available" if available else "❌ Not found"
        print(f"   {script}: {status}")
    
    # Test working directory
    print(f"\n2. Working directory: {launcher.get_working_directory()}")
    
    # Test help for gradient descent
    print("\n3. Testing gradient descent help:")
    help_text = launcher.get_script_help('gradient_descent_3d.py')
    if '--save_png' in help_text:
        print("   ✅ Gradient descent script supports --save_png")
    else:
        print("   ❌ Gradient descent script does not support --save_png")
    
    # Test help for predict
    print("\n4. Testing predict help:")
    help_text = launcher.get_script_help('predict.py')
    if '--x_features' in help_text:
        print("   ✅ Predict script supports feature selection")
    else:
        print("   ❌ Predict script does not support feature selection")
    
    print("\nScript launcher test completed!")

if __name__ == "__main__":
    test_launcher() 