"""
Model Results Viewer

This script displays comprehensive visualization plots generated during model training.
It allows you to view training results, including loss curves, predictions, feature distributions,
and error metrics.

Usage:
    python view_results.py [model_dir] [--plot-type TYPE] [--save-plots]

Arguments:
    model_dir    Path to the model directory (default: most recent model directory)
    --plot-type  Select specific plot type: loss, features, normalization, predictions, all
    --save-plots Save generated plots as PNG files

Example:
    python view_results.py
    python view_results.py model_20240315_123456
    python view_results.py --plot-type loss
    python view_results.py --save-plots
"""

import os
import glob
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import numpy as np
import pandas as pd
from datetime import datetime

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
    """Load comprehensive model information from the model directory."""
    info = {}
    
    # Feature info
    feature_info_file = os.path.join(model_dir, 'feature_info.json')
    if os.path.exists(feature_info_file):
        with open(feature_info_file, 'r') as f:
            feature_info = json.load(f)
        info['x_features'] = feature_info.get('x_features', [])
        info['y_feature'] = feature_info.get('y_feature', '')
        info['input_size'] = feature_info.get('input_size', 0)
    else:
        info['x_features'] = []
        info['y_feature'] = ''
        info['input_size'] = 0
    
    # Normalization parameters (clarified as min and range)
    scaler_mean_file = os.path.join(model_dir, 'scaler_mean.csv')
    scaler_std_file = os.path.join(model_dir, 'scaler_std.csv')
    if os.path.exists(scaler_mean_file) and os.path.exists(scaler_std_file):
        info['X_min'] = np.loadtxt(scaler_mean_file, delimiter=',')
        info['X_range'] = np.loadtxt(scaler_std_file, delimiter=',')  # Renamed from X_std to X_range
    else:
        info['X_min'] = None
        info['X_range'] = None
    
    # Target normalization
    target_min_file = os.path.join(model_dir, 'target_min.csv')
    target_max_file = os.path.join(model_dir, 'target_max.csv')
    if os.path.exists(target_min_file) and os.path.exists(target_max_file):
        info['Y_min'] = float(np.loadtxt(target_min_file, delimiter=','))
        info['Y_max'] = float(np.loadtxt(target_max_file, delimiter=','))
    else:
        info['Y_min'] = None
        info['Y_max'] = None
    
    # Training and validation losses
    training_loss_file = os.path.join(model_dir, 'training_losses.csv')
    if os.path.exists(training_loss_file):
        losses = np.loadtxt(training_loss_file, delimiter=',')
        if losses.ndim > 1:
            info['train_losses'] = losses[:, 0]
            info['val_losses'] = losses[:, 1]
        else:
            info['train_losses'] = losses
            info['val_losses'] = None
        info['epochs'] = np.arange(1, len(info['train_losses']) + 1)
    else:
        info['train_losses'] = None
        info['val_losses'] = None
        info['epochs'] = None
    
    # Model metadata with robust parsing
    metadata = {}
    metadata_file = os.path.join(model_dir, 'model_metadata.txt')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                try:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
                except ValueError:
                    continue
    info['metadata'] = metadata
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(model_dir, 'predictions_*.csv'))
    if pred_files:
        info['prediction_file'] = max(pred_files, key=os.path.getctime)
    else:
        info['prediction_file'] = None
    
    return info

def plot_training_loss(ax, info):
    """Plot training and validation loss curves."""
    if info['epochs'] is not None and info['train_losses'] is not None:
        ax.plot(info['epochs'], info['train_losses'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        if info['val_losses'] is not None:
            ax.plot(info['epochs'], info['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final loss values as text
        final_train_loss = info['train_losses'][-1] if info['train_losses'] is not None else 0
        final_val_loss = info['val_losses'][-1] if info['val_losses'] is not None else 0
        ax.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.6f}\nFinal Val Loss: {final_val_loss:.6f}', 
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Training Loss (not available)', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Training Loss')

def plot_feature_distributions(ax, info):
    """Plot actual feature distributions using histograms."""
    # COMMENTED OUT: Original robust implementation
    # if info['x_features'] and 'data_file' in info['metadata']:
    #     data_file = info['metadata']['data_file']
    #     if os.path.exists(data_file):
    #         try:
    #             df = pd.read_csv(data_file)
    #             for feature in info['x_features']:
    #                 if feature in df.columns:
    #                     ax.hist(df[feature], bins=20, alpha=0.5, label=feature, density=True)
    #             ax.set_title('Feature Distributions')
    #             ax.set_xlabel('Value')
    #             ax.set_ylabel('Density')
    #             ax.legend()
    #             ax.grid(True, alpha=0.3)
    #         except Exception as e:
    #             ax.text(0.5, 0.5, f'Error loading data: {str(e)}', ha='center', va='center', 
    #                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    #             ax.set_title('Feature Distributions (data loading error)')
    #     else:
    #         ax.text(0.5, 0.5, 'Data file not found', ha='center', va='center', 
    #                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    #         ax.set_title('Feature Distributions (data file not found)')
    # else:
    #     # Fallback to simple feature count
    #     if info['x_features']:
    #         ax.bar(range(len(info['x_features'])), [1]*len(info['x_features']), tick_label=info['x_features'])
    #         ax.set_title('Input Features (Count)')
    #         ax.set_ylabel('Count')
    #     else:
    #         ax.text(0.5, 0.5, 'Feature information not available', ha='center', va='center', 
    #                transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    #         ax.set_title('Input Features (not available)')
    
    # NEW: Simpler implementation
    if info['x_features'] and 'data_file' in info['metadata']:
        data_file = info['metadata']['data_file']
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            for feature in info['x_features']:
                if feature in df.columns:
                    ax.hist(df[feature], bins=20, alpha=0.5, label=feature, density=True)
            ax.set_title('Feature Distributions')
            ax.legend()
        else:
            ax.set_title('Feature Distributions (data file not found)')
    else:
        ax.set_title('Feature Distributions (not available)')

def plot_normalization_parameters(ax, info):
    """Plot feature normalization parameters (min and range)."""
    if info['X_min'] is not None and info['X_range'] is not None and info['x_features']:
        mins = np.array(info['X_min'])
        ranges = np.array(info['X_range'])
        ax.bar(range(len(mins)), mins, yerr=ranges, capsize=5, alpha=0.7)
        ax.set_title('Feature Normalization (Min and Range)')
        ax.set_xlabel('Features')
        ax.set_ylabel('Value (Min with Range as Error Bars)')
        ax.set_xticks(range(len(mins)))
        ax.set_xticklabels(info['x_features'], rotation=45)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Normalization parameters not available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Feature Normalization (not available)')

def plot_predictions(ax, info):
    """Plot actual vs predicted prices."""
    if info['prediction_file'] and os.path.exists(info['prediction_file']):
        try:
            df = pd.read_csv(info['prediction_file'])
            if 'close' in df.columns and 'predicted_close' in df.columns:
                # Check if we have date/time columns for x-axis
                if 'date' in df.columns or 'timestamp' in df.columns:
                    date_col = 'date' if 'date' in df.columns else 'timestamp'
                    df[date_col] = pd.to_datetime(df[date_col])
                    ax.plot(df[date_col], df['close'], 'b-', label='Actual', alpha=0.7, linewidth=2)
                    ax.plot(df[date_col], df['predicted_close'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax.get_xticklabels(), rotation=45)
                else:
                    ax.plot(df['close'], 'b-', label='Actual', alpha=0.7, linewidth=2)
                    ax.plot(df['predicted_close'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
                    ax.set_xlabel('Sample')
                
                ax.set_title('Actual vs Predicted Prices')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calculate and display error metrics
                mse = np.mean((df['close'] - df['predicted_close']) ** 2)
                mae = np.mean(np.abs(df['close'] - df['predicted_close']))
                rmse = np.sqrt(mse)
                ax.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}',
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Prediction columns not found', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                ax.set_title('Predictions (columns not found)')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading predictions: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Predictions (loading error)')
    else:
        ax.text(0.5, 0.5, 'No prediction file found', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Predictions (not available)')

def plot_metadata(ax, info):
    """Plot model metadata."""
    ax.axis('off')
    if info['metadata']:
        metadata_text = "\n".join([f"{k}: {v}" for k, v in info['metadata'].items()])
        ax.text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No metadata available', ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax.set_title('Model Metadata')

def display_plots(model_dir, info, plot_type='all', save_plots=False):
    """Display comprehensive model results."""
    print(f"\nDisplaying results for model: {model_dir}")
    print(f"Features: {info['x_features']}")
    print(f"Target: {info['y_feature']}")
    print(f"Input size: {info['input_size']}")
    
    if plot_type == 'all':
        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
        
        # Plot 1: Training Loss
        ax1 = fig.add_subplot(gs[0, 0])
        plot_training_loss(ax1, info)
        
        # Plot 2: Feature Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        plot_feature_distributions(ax2, info)
        
        # Plot 3: Normalization Parameters
        ax3 = fig.add_subplot(gs[1, 0])
        plot_normalization_parameters(ax3, info)
        
        # Plot 4: Predictions (from prediction files in model directory)
        ax4 = fig.add_subplot(gs[1, 1])
        pred_files = glob.glob(os.path.join(model_dir, 'predictions_*.csv'))
        if pred_files:
            latest_pred = max(pred_files, key=os.path.getctime)
            df = pd.read_csv(latest_pred)
            if 'close' in df.columns and 'predicted_close' in df.columns:
                ax4.plot(df['close'], label='Actual', alpha=0.7)
                ax4.plot(df['predicted_close'], label='Predicted', alpha=0.7)
                ax4.set_title('Actual vs Predicted Prices')
                ax4.set_xlabel('Sample')
                ax4.set_ylabel('Price')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Calculate and display error metrics
                mse = np.mean((df['close'] - df['predicted_close']) ** 2)
                mae = np.mean(np.abs(df['close'] - df['predicted_close']))
                ax4.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}',
                         transform=ax4.transAxes, va='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                plot_predictions(ax4, info)  # Fallback to original predictions plot
        else:
            plot_predictions(ax4, info)  # Fallback to original predictions plot
        
        # Plot 5: Metadata
        ax5 = fig.add_subplot(gs[2, :])
        plot_metadata(ax5, info)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(model_dir, 'results_summary.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved summary plot to: {plot_file}")
        
        plt.show()
        
    else:
        # Create individual plot based on plot_type
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_type == 'loss':
            plot_training_loss(ax, info)
        elif plot_type == 'features':
            plot_feature_distributions(ax, info)
        elif plot_type == 'normalization':
            plot_normalization_parameters(ax, info)
        elif plot_type == 'predictions':
            plot_predictions(ax, info)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(model_dir, f'{plot_type}_plot.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved {plot_type} plot to: {plot_file}")
        
        plt.show()
    
    # Show PNG plots from plots directory with captions
    plots_dir = os.path.join(model_dir, 'plots')
    if os.path.exists(plots_dir):
        plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
        for plot_file in plot_files:
            try:
                img = plt.imread(plot_file)
                plt.figure(figsize=(10, 6))
                plt.imshow(img)
                plt.axis('off')
                title = os.path.basename(plot_file).replace('_', ' ').title()
                plt.title(title)
                
                if save_plots:
                    save_file = os.path.join(model_dir, f'displayed_{os.path.basename(plot_file)}')
                    plt.savefig(save_file, dpi=300, bbox_inches='tight')
                    print(f"Saved displayed plot to: {save_file}")
                
                plt.show()
            except Exception as e:
                print(f"Error displaying plot {plot_file}: {str(e)}")
    else:
        print('No plots directory found.')

def main():
    parser = argparse.ArgumentParser(description='View comprehensive model results.')
    parser.add_argument('model_dir', nargs='?', default=None, help='Path to model directory')
    parser.add_argument('--plot-type', choices=['loss', 'features', 'normalization', 'predictions', 'all'], 
                       default='all', help='Select specific plot type')
    parser.add_argument('--save-plots', action='store_true', help='Save generated plots as PNG files')
    
    args = parser.parse_args()
    
    try:
        model_dir = args.model_dir or find_latest_model_dir()
        info = load_model_info(model_dir)
        display_plots(model_dir, info, args.plot_type, args.save_plots)
    except Exception as e:
        print(f"Error displaying results: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()