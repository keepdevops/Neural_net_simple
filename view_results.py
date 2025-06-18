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
import json
import numpy as np
import pandas as pd

# Add project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_latest_model_dir():
    """Find the most recent model directory."""
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if os.path.exists(models_dir):
            model_dirs = glob.glob(os.path.join(models_dir, "model_*"))
        if not model_dirs:
            raise FileNotFoundError("No model directories found. Please train a model first.")
    return max(model_dirs, key=os.path.getctime)

def load_model_info(model_dir):
    """Load normalization, feature info, and training loss from the model directory."""
    info = {}
    # Feature info
    feature_info_file = os.path.join(model_dir, 'feature_info.json')
    if os.path.exists(feature_info_file):
        with open(feature_info_file, 'r') as f:
            feature_info = json.load(f)
        info['x_features'] = feature_info.get('x_features', [])
        info['y_feature'] = feature_info.get('y_feature', '')
    else:
        info['x_features'] = []
        info['y_feature'] = ''
    # Normalization
    scaler_mean_file = os.path.join(model_dir, 'scaler_mean.csv')
    scaler_std_file = os.path.join(model_dir, 'scaler_std.csv')
    if os.path.exists(scaler_mean_file) and os.path.exists(scaler_std_file):
        info['X_mean'] = np.loadtxt(scaler_mean_file, delimiter=',')
        info['X_std'] = np.loadtxt(scaler_std_file, delimiter=',')
    else:
        info['X_mean'] = None
        info['X_std'] = None
    # Target normalization
    target_min_file = os.path.join(model_dir, 'target_min.csv')
    target_max_file = os.path.join(model_dir, 'target_max.csv')
    if os.path.exists(target_min_file) and os.path.exists(target_max_file):
        info['Y_min'] = float(np.loadtxt(target_min_file, delimiter=','))
        info['Y_max'] = float(np.loadtxt(target_max_file, delimiter=','))
    else:
        info['Y_min'] = None
        info['Y_max'] = None
    # Training loss
    training_loss_file = os.path.join(model_dir, 'training_losses.csv')
    if os.path.exists(training_loss_file):
        info['losses'] = np.loadtxt(training_loss_file, delimiter=',')
        info['epochs'] = np.arange(1, len(info['losses']) + 1)
    else:
        info['losses'] = None
        info['epochs'] = None
    # Model metadata (optional)
    metadata = {}
    metadata_file = os.path.join(model_dir, 'model_metadata.txt')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    info['metadata'] = metadata
    return info

def display_plots(model_dir, info):
    print(f"\nDisplaying results for model: {model_dir}")
    print(f"Features: {info['x_features']}")
    print(f"Target: {info['y_feature']}")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot 1: Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if info['epochs'] is not None and info['losses'] is not None:
        ax1.plot(info['epochs'], info['losses'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
    else:
        ax1.set_title('Training Loss (not available)')

    # Plot 2: Feature Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if info['x_features']:
        ax2.bar(range(len(info['x_features'])), [1]*len(info['x_features']), tick_label=info['x_features'])
        ax2.set_title('Input Features')
    else:
        ax2.set_title('Input Features (not available)')

    # Plot 3: Normalization Parameters
    ax3 = fig.add_subplot(gs[1, 0])
    if info['X_mean'] is not None and info['X_std'] is not None and info['x_features']:
        means = np.array(info['X_mean'])
        stds = np.array(info['X_std'])
        ax3.bar(range(len(means)), means, yerr=stds, capsize=5)
        ax3.set_title('Feature Normalization')
        ax3.set_xticks(range(len(means)))
        ax3.set_xticklabels(info['x_features'], rotation=45)
    else:
        ax3.set_title('Feature Normalization (not available)')

    # Plot 4: Model Metadata
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    metadata_text = "\n".join([f"{k}: {v}" for k, v in info['metadata'].items()])
    ax4.text(0.5, 0.5, metadata_text or 'No metadata', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Show all PNG plots in the plots/ directory
    plots_dir = os.path.join(model_dir, 'plots')
    if os.path.exists(plots_dir):
        plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
        for plot_file in plot_files:
            img = plt.imread(plot_file)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(plot_file))
            plt.show()
    else:
        print('No plots directory found.')

def main():
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = find_latest_model_dir()
    try:
        info = load_model_info(model_dir)
        display_plots(model_dir, info)
    except Exception as e:
        print(f"Error displaying results: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()