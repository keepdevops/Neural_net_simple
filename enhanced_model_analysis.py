"""
Enhanced Model Analysis and Visualization

This module provides comprehensive model analysis capabilities with:
- Robust model information loading
- Advanced plotting functions
- Performance optimizations
- Enhanced error handling
- Memory management
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
from PIL import Image, ImageTk
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TEXT_COLOR = "#000000"
FRAME_COLOR = "#F0F0F0"
BACKGROUND_COLOR = "#FFFFFF"
ACCENT_COLOR = "#007ACC"
ERROR_COLOR = "#DC3545"
SUCCESS_COLOR = "#28A745"
WARNING_COLOR = "#FFC107"

class EnhancedModelAnalyzer:
    """Enhanced model analyzer with comprehensive visualization capabilities."""
    
    def __init__(self):
        """Initialize the model analyzer."""
        self.current_model_dir = None
        self.model_info_cache = {}  # Cache for model information
        self.plot_images = []  # Keep references to prevent garbage collection
        
        logger.info("Enhanced model analyzer initialized")
    
    def load_model_info(self, model_dir: str) -> Dict[str, Any]:
        """
        Load comprehensive model information from the model directory.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Dictionary containing comprehensive model information
        """
        try:
            # Check cache first
            if model_dir in self.model_info_cache:
                logger.info(f"Using cached model info for {model_dir}")
                return self.model_info_cache[model_dir]
            
            info = {}
            
            # Feature info with enhanced error handling
            feature_info_file = os.path.join(model_dir, 'feature_info.json')
            if os.path.exists(feature_info_file):
                try:
                    with open(feature_info_file, 'r') as f:
                        feature_info = json.load(f)
                    info['x_features'] = feature_info.get('x_features', [])
                    info['y_feature'] = feature_info.get('y_feature', '')
                    info['input_size'] = feature_info.get('input_size', 0)
                    info['feature_info_loaded'] = True
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error loading feature info: {e}")
                    info['x_features'] = []
                    info['y_feature'] = ''
                    info['input_size'] = 0
                    info['feature_info_loaded'] = False
            else:
                info['x_features'] = []
                info['y_feature'] = ''
                info['input_size'] = 0
                info['feature_info_loaded'] = False
            
            # Normalization parameters with robust loading
            info.update(self._load_normalization_params(model_dir))
            
            # Training and validation losses
            info.update(self._load_training_losses(model_dir))
            
            # Model metadata with enhanced parsing
            info['metadata'] = self._load_model_metadata(model_dir)
            
            # Find prediction files
            info['prediction_file'] = self._find_prediction_file(model_dir)
            
            # Additional model statistics
            info.update(self._calculate_model_statistics(info))
            
            # Cache the results
            self.model_info_cache[model_dir] = info
            
            logger.info(f"Successfully loaded model info for {model_dir}")
            return info
            
        except Exception as e:
            error_msg = f"Error loading model info from {model_dir}: {str(e)}"
            logger.error(error_msg)
            return self._create_error_info(error_msg)
    
    def _load_normalization_params(self, model_dir: str) -> Dict[str, Any]:
        """Load normalization parameters with error handling."""
        params = {}
        
        # Feature normalization
        scaler_mean_file = os.path.join(model_dir, 'scaler_mean.csv')
        scaler_std_file = os.path.join(model_dir, 'scaler_std.csv')
        
        if os.path.exists(scaler_mean_file) and os.path.exists(scaler_std_file):
            try:
                params['X_min'] = np.loadtxt(scaler_mean_file, delimiter=',')
                params['X_range'] = np.loadtxt(scaler_std_file, delimiter=',')
                params['normalization_loaded'] = True
            except Exception as e:
                logger.warning(f"Error loading normalization params: {e}")
                params['X_min'] = None
                params['X_range'] = None
                params['normalization_loaded'] = False
        else:
            params['X_min'] = None
            params['X_range'] = None
            params['normalization_loaded'] = False
        
        # Target normalization
        target_min_file = os.path.join(model_dir, 'target_min.csv')
        target_max_file = os.path.join(model_dir, 'target_max.csv')
        
        if os.path.exists(target_min_file) and os.path.exists(target_max_file):
            try:
                params['Y_min'] = float(np.loadtxt(target_min_file, delimiter=','))
                params['Y_max'] = float(np.loadtxt(target_max_file, delimiter=','))
                params['target_normalization_loaded'] = True
            except Exception as e:
                logger.warning(f"Error loading target normalization: {e}")
                params['Y_min'] = None
                params['Y_max'] = None
                params['target_normalization_loaded'] = False
        else:
            params['Y_min'] = None
            params['Y_max'] = None
            params['target_normalization_loaded'] = False
        
        return params
    
    def _load_training_losses(self, model_dir: str) -> Dict[str, Any]:
        """Load training and validation losses with error handling."""
        losses = {}
        training_loss_file = os.path.join(model_dir, 'training_losses.csv')
        
        if os.path.exists(training_loss_file):
            try:
                loss_data = np.loadtxt(training_loss_file, delimiter=',')
                if loss_data.ndim > 1:
                    losses['train_losses'] = loss_data[:, 0]
                    losses['val_losses'] = loss_data[:, 1]
                else:
                    losses['train_losses'] = loss_data
                    losses['val_losses'] = None
                losses['epochs'] = np.arange(1, len(losses['train_losses']) + 1)
                losses['losses_loaded'] = True
            except Exception as e:
                logger.warning(f"Error loading training losses: {e}")
                losses['train_losses'] = None
                losses['val_losses'] = None
                losses['epochs'] = None
                losses['losses_loaded'] = False
        else:
            losses['train_losses'] = None
            losses['val_losses'] = None
            losses['epochs'] = None
            losses['losses_loaded'] = False
        
        return losses
    
    def _load_model_metadata(self, model_dir: str) -> Dict[str, str]:
        """Load model metadata with enhanced parsing."""
        metadata = {}
        metadata_file = os.path.join(model_dir, 'model_metadata.txt')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and ':' in line:
                            try:
                                key, value = line.split(':', 1)
                                metadata[key.strip()] = value.strip()
                            except ValueError:
                                logger.debug(f"Skipping malformed metadata line {line_num}: {line}")
                                continue
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        return metadata
    
    def _find_prediction_file(self, model_dir: str) -> Optional[str]:
        """Find the most recent prediction file."""
        try:
            pred_files = glob.glob(os.path.join(model_dir, 'predictions_*.csv'))
            if pred_files:
                return max(pred_files, key=os.path.getctime)
        except Exception as e:
            logger.warning(f"Error finding prediction file: {e}")
        return None
    
    def _calculate_model_statistics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional model statistics."""
        stats = {}
        
        # Training statistics
        if info.get('train_losses') is not None:
            train_losses = info['train_losses']
            stats['final_train_loss'] = float(train_losses[-1])
            stats['min_train_loss'] = float(np.min(train_losses))
            stats['max_train_loss'] = float(np.max(train_losses))
            stats['train_loss_std'] = float(np.std(train_losses))
            stats['total_epochs'] = len(train_losses)
        
        if info.get('val_losses') is not None:
            val_losses = info['val_losses']
            stats['final_val_loss'] = float(val_losses[-1])
            stats['min_val_loss'] = float(np.min(val_losses))
            stats['max_val_loss'] = float(np.max(val_losses))
            stats['val_loss_std'] = float(np.std(val_losses))
        
        # Feature statistics
        if info.get('x_features'):
            stats['num_features'] = len(info['x_features'])
        
        return stats
    
    def _create_error_info(self, error_msg: str) -> Dict[str, Any]:
        """Create error information structure."""
        return {
            'error': error_msg,
            'x_features': [],
            'y_feature': '',
            'input_size': 0,
            'X_min': None,
            'X_range': None,
            'Y_min': None,
            'Y_max': None,
            'train_losses': None,
            'val_losses': None,
            'epochs': None,
            'metadata': {},
            'prediction_file': None,
            'feature_info_loaded': False,
            'normalization_loaded': False,
            'target_normalization_loaded': False,
            'losses_loaded': False
        }
    
    def plot_training_loss(self, ax, info: Dict[str, Any]) -> None:
        """
        Plot training and validation loss curves with enhanced styling.
        
        Args:
            ax: Matplotlib axis
            info: Model information dictionary
        """
        try:
            if info.get('epochs') is not None and info.get('train_losses') is not None:
                epochs = info['epochs']
                train_losses = info['train_losses']
                
                # Plot training loss
                ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
                
                # Plot validation loss if available
                if info.get('val_losses') is not None:
                    ax.plot(epochs, info['val_losses'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
                
                # Enhanced styling
                ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('MSE Loss', fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # Add final loss annotations
                final_train_loss = info.get('final_train_loss', train_losses[-1])
                final_val_loss = info.get('final_val_loss', info.get('val_losses', [None])[-1])
                
                annotation_text = f'Final Train Loss: {final_train_loss:.6f}'
                if final_val_loss is not None:
                    annotation_text += f'\nFinal Val Loss: {final_val_loss:.6f}'
                
                ax.text(0.02, 0.98, annotation_text, 
                       transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=10)
                
                # Add trend analysis
                if len(train_losses) > 10:
                    trend = "Decreasing" if train_losses[-1] < train_losses[0] else "Increasing"
                    ax.text(0.02, 0.02, f'Trend: {trend}', 
                           transform=ax.transAxes, va='bottom',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                           fontsize=9)
                
            else:
                ax.text(0.5, 0.5, 'Training Loss (not available)', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title('Training Loss')
                
        except Exception as e:
            logger.error(f"Error plotting training loss: {e}")
            ax.text(0.5, 0.5, f'Error plotting training loss: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Training Loss (Error)')
    
    def plot_feature_distributions(self, ax, info: Dict[str, Any]) -> None:
        """
        Plot actual feature distributions using histograms with enhanced error handling.
        
        Args:
            ax: Matplotlib axis
            info: Model information dictionary
        """
        try:
            if info.get('x_features') and info.get('metadata', {}).get('data_file'):
                data_file = info['metadata']['data_file']
                if os.path.exists(data_file):
                    try:
                        df = pd.read_csv(data_file)
                        available_features = [f for f in info['x_features'] if f in df.columns]
                        
                        if available_features:
                            for feature in available_features:
                                ax.hist(df[feature], bins=20, alpha=0.5, label=feature, density=True)
                            
                            ax.set_title('Feature Distributions', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Value', fontsize=12)
                            ax.set_ylabel('Density', fontsize=12)
                            ax.legend(fontsize=10)
                            ax.grid(True, alpha=0.3)
                            
                            # Add statistics
                            stats_text = f"Features: {len(available_features)}/{len(info['x_features'])}"
                            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
                                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                                   fontsize=9)
                        else:
                            ax.text(0.5, 0.5, 'No matching features found in data', 
                                   ha='center', va='center', transform=ax.transAxes, 
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                            ax.set_title('Feature Distributions (no matching features)')
                            
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error loading data: {str(e)}', 
                               ha='center', va='center', transform=ax.transAxes, 
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                        ax.set_title('Feature Distributions (data loading error)')
                else:
                    ax.text(0.5, 0.5, 'Data file not found', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                    ax.set_title('Feature Distributions (data file not found)')
            else:
                if info.get('x_features'):
                    # Show feature count as bar chart
                    ax.bar(range(len(info['x_features'])), [1]*len(info['x_features']), 
                           tick_label=info['x_features'], alpha=0.7)
                    ax.set_title('Input Features (Count)', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Count', fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Feature information not available', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title('Input Features (not available)')
                    
        except Exception as e:
            logger.error(f"Error plotting feature distributions: {e}")
            ax.text(0.5, 0.5, f'Error plotting features: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Feature Distributions (Error)')
    
    def plot_normalization_parameters(self, ax, info: Dict[str, Any]) -> None:
        """
        Plot feature normalization parameters with enhanced visualization.
        
        Args:
            ax: Matplotlib axis
            info: Model information dictionary
        """
        try:
            if (info.get('X_min') is not None and info.get('X_range') is not None 
                and info.get('x_features')):
                
                mins = np.array(info['X_min'])
                ranges = np.array(info['X_range'])
                
                # Create bar plot with error bars
                x_pos = np.arange(len(mins))
                ax.bar(x_pos, mins, yerr=ranges, capsize=5, alpha=0.7, 
                       color='skyblue', edgecolor='navy', linewidth=1)
                
                ax.set_title('Feature Normalization (Min and Range)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Features', fontsize=12)
                ax.set_ylabel('Value (Min with Range as Error Bars)', fontsize=12)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(info['x_features'], rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f"Features: {len(mins)}\nMean Range: {np.mean(ranges):.4f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=9)
                
            else:
                ax.text(0.5, 0.5, 'Normalization parameters not available', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title('Feature Normalization (not available)')
                
        except Exception as e:
            logger.error(f"Error plotting normalization parameters: {e}")
            ax.text(0.5, 0.5, f'Error plotting normalization: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Feature Normalization (Error)')
    
    def plot_predictions(self, ax, info: Dict[str, Any]) -> None:
        """
        Plot actual vs predicted prices with enhanced metrics.
        
        Args:
            ax: Matplotlib axis
            info: Model information dictionary
        """
        try:
            if info.get('prediction_file') and os.path.exists(info['prediction_file']):
                df = pd.read_csv(info['prediction_file'])
                
                if 'actual' in df.columns and 'predicted' in df.columns:
                    # Handle date/time axis
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        ax.plot(df['date'], df['actual'], 'b-', label='Actual', 
                               alpha=0.7, linewidth=2)
                        ax.plot(df['date'], df['predicted'], 'r-', label='Predicted', 
                               alpha=0.7, linewidth=2)
                        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                        plt.setp(ax.get_xticklabels(), rotation=45)
                    else:
                        ax.plot(df['actual'], 'b-', label='Actual', alpha=0.7, linewidth=2)
                        ax.plot(df['predicted'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
                        ax.set_xlabel('Sample', fontsize=12)
                    
                    ax.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Price', fontsize=12)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate and display metrics
                    mse = np.mean((df['actual'] - df['predicted']) ** 2)
                    mae = np.mean(np.abs(df['actual'] - df['predicted']))
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((df['actual'] - df['predicted']) / df['actual'])) * 100
                    
                    metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\nMAPE: {mape:.2f}%'
                    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                           fontsize=9)
                    
                else:
                    ax.text(0.5, 0.5, 'Prediction columns not found', 
                           ha='center', va='center', transform=ax.transAxes, 
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                    ax.set_title('Predictions (columns not found)')
                    
            else:
                ax.text(0.5, 0.5, 'No prediction file found', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title('Predictions (not available)')
                
        except Exception as e:
            logger.error(f"Error plotting predictions: {e}")
            ax.text(0.5, 0.5, f'Error loading predictions: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Predictions (loading error)')
    
    def plot_metadata(self, ax, info: Dict[str, Any]) -> None:
        """
        Plot model metadata with enhanced formatting.
        
        Args:
            ax: Matplotlib axis
            info: Model information dictionary
        """
        try:
            ax.axis('off')
            
            if info.get('metadata'):
                # Format metadata for display
                metadata_items = []
                for key, value in info['metadata'].items():
                    # Truncate long values
                    if len(str(value)) > 50:
                        value = str(value)[:47] + "..."
                    metadata_items.append(f"{key}: {value}")
                
                # Add calculated statistics
                if info.get('final_train_loss') is not None:
                    metadata_items.append(f"Final Train Loss: {info['final_train_loss']:.6f}")
                if info.get('final_val_loss') is not None:
                    metadata_items.append(f"Final Val Loss: {info['final_val_loss']:.6f}")
                if info.get('total_epochs') is not None:
                    metadata_items.append(f"Total Epochs: {info['total_epochs']}")
                if info.get('num_features') is not None:
                    metadata_items.append(f"Number of Features: {info['num_features']}")
                
                metadata_text = "\n".join(metadata_items)
                ax.text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No metadata available', ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            ax.set_title('Model Metadata', fontsize=14, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Error plotting metadata: {e}")
            ax.text(0.5, 0.5, f'Error displaying metadata: {str(e)}', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_title('Model Metadata (Error)')
    
    def clear_cache(self) -> None:
        """Clear the model information cache."""
        self.model_info_cache.clear()
        logger.info("Model information cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            'cached_models': list(self.model_info_cache.keys()),
            'cache_size': len(self.model_info_cache),
            'memory_usage': sum(len(str(v)) for v in self.model_info_cache.values())
        }

def main():
    """Test the enhanced model analyzer."""
    analyzer = EnhancedModelAnalyzer()
    
    # Test with a sample model directory
    test_model_dir = "model_20250619_095746"  # Replace with actual model directory
    
    if os.path.exists(test_model_dir):
        print(f"Testing model analyzer with: {test_model_dir}")
        
        # Load model info
        info = analyzer.load_model_info(test_model_dir)
        
        # Display summary
        print("\nModel Information Summary:")
        print(f"Features: {info.get('x_features', [])}")
        print(f"Target: {info.get('y_feature', 'N/A')}")
        print(f"Input Size: {info.get('input_size', 0)}")
        print(f"Total Epochs: {info.get('total_epochs', 0)}")
        print(f"Final Train Loss: {info.get('final_train_loss', 'N/A')}")
        print(f"Final Val Loss: {info.get('final_val_loss', 'N/A')}")
        
        # Show cache info
        cache_info = analyzer.get_cache_info()
        print(f"\nCache Info: {cache_info}")
        
    else:
        print(f"Test model directory not found: {test_model_dir}")
        print("Please update the test_model_dir variable with an existing model directory.")

if __name__ == "__main__":
    main() 