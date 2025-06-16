"""
Model Results Viewer

This script displays the visualization plots generated during model training.
It allows you to view the training results, including loss curves, predictions,
and error distributions.

Usage:
    python view_results.py [model_dir]

Arguments:
    model_dir    Path to the model directory (default: most recent model directory)

Example:
    python view_results.py
    python view_results.py model_20240315_123456
"""

import os
import glob
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def find_latest_model_dir():
    """Find the most recent model directory."""
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        raise FileNotFoundError("No model directories found. Please train a model first.")
    return max(model_dirs, key=os.path.getctime)

def load_plot_data(model_dir):
    """Load all plot data from the model directory."""
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        raise FileNotFoundError(f"Plots directory not found in {model_dir}")
    
    # Load metadata if available
    metadata = {}
    metadata_file = os.path.join(model_dir, 'model_metadata.txt')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return plots_dir, metadata

def display_plots(plots_dir, metadata):
    """Display all plots in a grid layout."""
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Display metadata if available
    if metadata:
        metadata_text = "Model Information:\n"
        for key, value in metadata.items():
            metadata_text += f"{key}: {value}\n"
        fig.text(0.02, 0.98, metadata_text, fontsize=8, va='top')
    
    # Load and display each plot
    plot_files = {
        'loss_curves.png': (0, 0, 'Training and Validation Loss'),
        'training_predictions.png': (0, 1, 'Training Set Predictions'),
        'test_predictions.png': (1, 0, 'Test Set Predictions'),
        'error_distribution.png': (1, 1, 'Error Distribution')
    }
    
    for filename, (row, col, title) in plot_files.items():
        filepath = os.path.join(plots_dir, filename)
        if os.path.exists(filepath):
            ax = fig.add_subplot(gs[row, col])
            img = plt.imread(filepath)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
    
    # Add actual vs predicted scatter plot
    scatter_file = os.path.join(plots_dir, 'actual_vs_predicted.png')
    if os.path.exists(scatter_file):
        fig2 = plt.figure(figsize=(8, 8))
        img = plt.imread(scatter_file)
        plt.imshow(img)
        plt.title('Actual vs Predicted Prices')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to display model results."""
    # Get model directory from command line or use most recent
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        if not os.path.exists(model_dir):
            print(f"Error: Model directory '{model_dir}' not found.")
            sys.exit(1)
    else:
        try:
            model_dir = find_latest_model_dir()
            print(f"Using most recent model directory: {model_dir}")
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    try:
        # Load plot data
        plots_dir, metadata = load_plot_data(model_dir)
        
        # Display plots
        print(f"\nDisplaying plots from {model_dir}...")
        display_plots(plots_dir, metadata)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 