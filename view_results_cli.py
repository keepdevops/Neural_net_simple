#!/usr/bin/env python3
"""
Command-line interface for viewing model results.

This script provides a command-line interface to view various plots and results
from trained models, similar to the GUI but accessible from the terminal.

Usage:
    python view_results_cli.py [model_dir] [--plot-type TYPE]
    python view_results_cli.py --help
"""

import argparse
import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import path_utils
import script_launcher

def find_latest_model_dir():
    """Find the most recent model directory in the current directory."""
    try:
        model_dirs = []
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            if os.path.isdir(item_path) and item.startswith('model_'):
                model_dirs.append(item)
        
        if not model_dirs:
            print("No model directories found in current directory")
            return None
        
        # Sort by creation time (newest first)
        model_dirs.sort(key=lambda x: os.path.getctime(os.path.join('.', x)), reverse=True)
        latest_model = model_dirs[0]
        print(f"Using latest model: {latest_model}")
        return latest_model
        
    except Exception as e:
        print(f"Error finding latest model directory: {e}")
        return None

def load_model_info(model_dir):
    """Load model information from the model directory."""
    try:
        info = {}
        
        # Load feature info
        feature_info_file = os.path.join(model_dir, 'feature_info.json')
        if os.path.exists(feature_info_file):
            with open(feature_info_file, 'r') as f:
                feature_info = json.load(f)
            info['x_features'] = feature_info.get('x_features', [])
            info['y_feature'] = feature_info.get('y_feature', '')
            print(f"Model features: {len(info['x_features'])} input features, target: {info['y_feature']}")
        
        # Load training losses
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
            print(f"Training data: {len(info['train_losses'])} epochs")
        
        # Check for prediction files
        pred_files = glob.glob(os.path.join(model_dir, 'predictions_*.csv'))
        if pred_files:
            info['prediction_files'] = pred_files
            print(f"Found {len(pred_files)} prediction files")
        
        # Check for plot files
        plots_dir = os.path.join(model_dir, 'plots')
        if os.path.exists(plots_dir):
            plot_files = glob.glob(os.path.join(plots_dir, '*.png'))
            info['plot_files'] = plot_files
            print(f"Found {len(plot_files)} plot files")
        
        return info
        
    except Exception as e:
        print(f"Error loading model info: {e}")
        return {}

def display_loss_plot(model_dir, info):
    """Display training loss plot."""
    try:
        if 'train_losses' not in info:
            print("No training loss data available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(info['epochs'], info['train_losses'], label='Training Loss', linewidth=2)
        
        if info['val_losses'] is not None:
            plt.plot(info['epochs'], info['val_losses'], label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_dir, 'plots', 'loss_plot_cli.png')
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {plot_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error displaying loss plot: {e}")

def display_feature_plot(model_dir, info):
    """Display feature information plot."""
    try:
        if 'x_features' not in info:
            print("No feature information available")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create feature count bar chart
        feature_counts = {}
        for feature in info['x_features']:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        features = list(feature_counts.keys())
        counts = list(feature_counts.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(features)), counts)
        plt.xlabel('Features')
        plt.ylabel('Count')
        plt.title('Feature Usage')
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        
        # Create pie chart of feature distribution
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=features, autopct='%1.1f%%')
        plt.title('Feature Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_dir, 'plots', 'feature_plot_cli.png')
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Feature plot saved to: {plot_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error displaying feature plot: {e}")

def display_normalization_plot(model_dir, info):
    """Display normalization information plot."""
    try:
        # Look for normalization files
        norm_files = glob.glob(os.path.join(model_dir, '*_normalization.csv'))
        
        if not norm_files:
            print("No normalization data available")
            return
        
        plt.figure(figsize=(12, 8))
        
        for i, norm_file in enumerate(norm_files[:4]):  # Limit to 4 files
            try:
                df = pd.read_csv(norm_file)
                
                plt.subplot(2, 2, i + 1)
                
                if 'X_min' in df.columns and 'X_range' in df.columns:
                    plt.errorbar(range(len(df)), df['X_min'], yerr=df['X_range'], 
                               fmt='o', capsize=5, capthick=2)
                    plt.title(f'Feature Normalization: {os.path.basename(norm_file)}')
                    plt.ylabel('Value (Min with Range as Error Bars)')
                else:
                    plt.plot(df.iloc[:, 0], label='Data')
                    plt.title(f'Data: {os.path.basename(norm_file)}')
                
                plt.xlabel('Feature Index')
                plt.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error processing {norm_file}: {e}")
                continue
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_dir, 'plots', 'normalization_plot_cli.png')
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Normalization plot saved to: {plot_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error displaying normalization plot: {e}")

def display_predictions_plot(model_dir, info):
    """Display predictions plot."""
    try:
        if 'prediction_files' not in info or not info['prediction_files']:
            print("No prediction files available")
            return
        
        # Use the latest prediction file
        pred_file = info['prediction_files'][-1]
        df = pd.read_csv(pred_file)
        
        plt.figure(figsize=(12, 8))
        
        # Handle date/time axis
        if 'date' in df.columns or 'timestamp' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            try:
                dates = pd.to_datetime(df[date_col])
                x_axis = dates
                plt.gca().xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
                plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
            except:
                x_axis = range(len(df))
                plt.xlabel('Sample')
        else:
            x_axis = range(len(df))
            plt.xlabel('Sample')
        
        # Plot predictions
        if 'predicted' in df.columns:
            plt.plot(x_axis, df['predicted'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
        
        if 'actual' in df.columns:
            plt.plot(x_axis, df['actual'], 'b-', label='Actual', alpha=0.7, linewidth=2)
        
        # Special handling for close vs predicted_close
        if 'close' in df.columns and 'predicted_close' in df.columns:
            plt.plot(x_axis, df['close'], label='Actual Close', alpha=0.7, linewidth=2)
            plt.plot(x_axis, df['predicted_close'], label='Predicted Close', alpha=0.7, linewidth=2)
            
            # Calculate correlation
            correlation = np.corrcoef(df['close'], df['predicted_close'])[0, 1]
            plt.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                    transform=plt.gca().transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.title(f'Prediction Results: {os.path.basename(pred_file)}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(model_dir, 'plots', 'predictions_plot_cli.png')
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to: {plot_file}")
        plt.show()
        
    except Exception as e:
        print(f"Error displaying predictions plot: {e}")

def create_results_summary(model_dir, info):
    """Create a comprehensive results summary plot combining multiple plots."""
    try:
        # Create a large figure for the summary
        fig = plt.figure(figsize=(16, 12))
        
        # Track subplot position
        subplot_pos = 1
        
        # 1. Training Loss Plot (top left)
        if 'train_losses' in info:
            ax1 = plt.subplot(2, 3, subplot_pos)
            ax1.plot(info['epochs'], info['train_losses'], label='Training Loss', linewidth=2)
            if info['val_losses'] is not None:
                ax1.plot(info['epochs'], info['val_losses'], label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            subplot_pos += 1
        
        # 2. Feature Information (top middle)
        if 'x_features' in info:
            ax2 = plt.subplot(2, 3, subplot_pos)
            feature_counts = {}
            for feature in info['x_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            features = list(feature_counts.keys())
            counts = list(feature_counts.values())
            
            ax2.bar(range(len(features)), counts)
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Count')
            ax2.set_title('Feature Usage')
            ax2.set_xticks(range(len(features)))
            ax2.set_xticklabels(features, rotation=45, ha='right')
            subplot_pos += 1
        
        # 3. Model Info Summary (top right)
        ax3 = plt.subplot(2, 3, subplot_pos)
        ax3.axis('off')
        
        info_text = f"Model Summary\n"
        info_text += f"=============\n"
        info_text += f"Model: {os.path.basename(model_dir)}\n"
        if 'x_features' in info:
            info_text += f"Input Features: {len(info['x_features'])}\n"
        if 'y_feature' in info:
            info_text += f"Target: {info['y_feature']}\n"
        if 'train_losses' in info:
            info_text += f"Training Epochs: {len(info['train_losses'])}\n"
            final_loss = info['train_losses'][-1] if len(info['train_losses']) > 0 else 0
            info_text += f"Final Loss: {final_loss:.6f}\n"
        if 'prediction_files' in info:
            info_text += f"Prediction Files: {len(info['prediction_files'])}\n"
        if 'plot_files' in info:
            info_text += f"Plot Files: {len(info['plot_files'])}\n"
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        subplot_pos += 1
        
        # 4. Predictions Plot (bottom left)
        if 'prediction_files' in info and info['prediction_files']:
            ax4 = plt.subplot(2, 3, subplot_pos)
            pred_file = info['prediction_files'][-1]
            df = pd.read_csv(pred_file)
            
            # Handle date/time axis
            if 'date' in df.columns or 'timestamp' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'timestamp'
                try:
                    dates = pd.to_datetime(df[date_col])
                    x_axis = dates
                    ax4.xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
                    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
                except:
                    x_axis = range(len(df))
                    ax4.set_xlabel('Sample')
            else:
                x_axis = range(len(df))
                ax4.set_xlabel('Sample')
            
            # Plot predictions
            if 'predicted' in df.columns:
                ax4.plot(x_axis, df['predicted'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
            if 'actual' in df.columns:
                ax4.plot(x_axis, df['actual'], 'b-', label='Actual', alpha=0.7, linewidth=2)
            
            # Special handling for close vs predicted_close
            if 'close' in df.columns and 'predicted_close' in df.columns:
                ax4.plot(x_axis, df['close'], label='Actual Close', alpha=0.7, linewidth=2)
                ax4.plot(x_axis, df['predicted_close'], label='Predicted Close', alpha=0.7, linewidth=2)
                
                # Calculate correlation
                correlation = np.corrcoef(df['close'], df['predicted_close'])[0, 1]
                ax4.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                        transform=ax4.transAxes, va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            ax4.set_title('Predictions')
            ax4.set_ylabel('Price')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            subplot_pos += 1
        
        # 5. Normalization Plot (bottom middle)
        norm_files = glob.glob(os.path.join(model_dir, '*_normalization.csv'))
        if norm_files:
            ax5 = plt.subplot(2, 3, subplot_pos)
            norm_file = norm_files[0]  # Use first normalization file
            df = pd.read_csv(norm_file)
            
            if 'X_min' in df.columns and 'X_range' in df.columns:
                ax5.errorbar(range(len(df)), df['X_min'], yerr=df['X_range'], 
                           fmt='o', capsize=3, capthick=1, markersize=4)
                ax5.set_title('Feature Normalization')
                ax5.set_ylabel('Value (Min ± Range)')
            else:
                ax5.plot(df.iloc[:, 0], label='Data')
                ax5.set_title('Data Distribution')
            
            ax5.set_xlabel('Feature Index')
            ax5.grid(True, alpha=0.3)
            subplot_pos += 1
        
        # 6. Performance Metrics (bottom right)
        ax6 = plt.subplot(2, 3, subplot_pos)
        ax6.axis('off')
        
        metrics_text = f"Performance Metrics\n"
        metrics_text += f"==================\n"
        
        if 'prediction_files' in info and info['prediction_files']:
            pred_file = info['prediction_files'][-1]
            df = pd.read_csv(pred_file)
            
            if 'actual' in df.columns and 'predicted' in df.columns:
                actual = df['actual'].values
                predicted = df['predicted'].values
                
                # Calculate metrics
                mse = np.mean((actual - predicted) ** 2)
                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                correlation = np.corrcoef(actual, predicted)[0, 1]
                
                metrics_text += f"MSE: {mse:.6f}\n"
                metrics_text += f"MAE: {mae:.6f}\n"
                metrics_text += f"RMSE: {rmse:.6f}\n"
                metrics_text += f"MAPE: {mape:.2f}%\n"
                metrics_text += f"Correlation: {correlation:.4f}\n"
            elif 'close' in df.columns and 'predicted_close' in df.columns:
                actual = df['close'].values
                predicted = df['predicted_close'].values
                
                mse = np.mean((actual - predicted) ** 2)
                mae = np.mean(np.abs(actual - predicted))
                correlation = np.corrcoef(actual, predicted)[0, 1]
                
                metrics_text += f"MSE: {mse:.6f}\n"
                metrics_text += f"MAE: {mae:.6f}\n"
                metrics_text += f"Correlation: {correlation:.4f}\n"
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add overall title
        fig.suptitle(f'Model Results Summary: {os.path.basename(model_dir)}', 
                    fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the comprehensive summary
        summary_file = os.path.join(model_dir, 'results_summary.png')
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        print(f"Results summary saved to: {summary_file}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error creating results summary: {e}")

def display_plots(model_dir, info, plot_type=None):
    """Display plots based on the specified type."""
    try:
        print(f"\n=== Model Directory: {model_dir} ===")
        
        if plot_type == 'loss' or plot_type is None:
            print("\n--- Training Loss Plot ---")
            display_loss_plot(model_dir, info)
        
        if plot_type == 'features' or plot_type is None:
            print("\n--- Feature Information Plot ---")
            display_feature_plot(model_dir, info)
        
        if plot_type == 'normalization' or plot_type is None:
            print("\n--- Normalization Plot ---")
            display_normalization_plot(model_dir, info)
        
        if plot_type == 'predictions' or plot_type is None:
            print("\n--- Predictions Plot ---")
            display_predictions_plot(model_dir, info)
        
        if plot_type == 'summary' or plot_type is None:
            print("\n--- Results Summary ---")
            create_results_summary(model_dir, info)
        
        if plot_type is None:
            print("\n=== All plots displayed ===")
        else:
            print(f"\n=== {plot_type.capitalize()} plot displayed ===")
            
    except Exception as e:
        print(f"Error displaying plots: {e}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='View model results.')
    parser.add_argument('model_dir', nargs='?', default=None, 
                       help='Path to model directory (default: latest model)')
    parser.add_argument('--plot-type', choices=['loss', 'features', 'normalization', 'predictions', 'summary'], 
                       help='Select specific plot type (default: all plots)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available model directories')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        model_dirs = []
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            if os.path.isdir(item_path) and item.startswith('model_'):
                model_dirs.append(item)
        
        if model_dirs:
            print("Available model directories:")
            for model_dir in sorted(model_dirs, key=lambda x: os.path.getctime(os.path.join('.', x)), reverse=True):
                ctime = datetime.fromtimestamp(os.path.getctime(os.path.join('.', model_dir)))
                print(f"  {model_dir} (created: {ctime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("No model directories found")
        return
    
    # Find model directory
    model_dir = args.model_dir or find_latest_model_dir()
    if not model_dir:
        print("No model directory specified or found")
        return
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    # Load model info and display plots
    info = load_model_info(model_dir)
    display_plots(model_dir, info, plot_type=args.plot_type)

if __name__ == "__main__":
    main() 