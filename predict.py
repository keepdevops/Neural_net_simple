"""
Stock Price Prediction Inference Script

This script loads a trained neural network model and makes predictions on new stock data.
It uses the weights and biases saved by stock_net.py to perform inference.

Usage:
    python predict.py <input_csv_file> [--model_dir MODEL_DIR] [--x_features FEATURES] [--y_feature TARGET]

Arguments:
    input_csv_file    Path to the input CSV file containing stock data
    --model_dir       Path to the model directory (default: models)
    --x_features      Comma-separated list of input features
    --y_feature       Target feature name

Example:
    python predict.py /Users/porupine/redline/data/gamestop_us.csv --x_features open,high,low,vol --y_feature close
    python predict.py /Users/porupine/redline/data/gamestop_us.csv --model_dir model_20240315_123456 --x_features open,high,low,vol --y_feature close
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Numerically stable sigmoid activation function.
    Same implementation as in stock_net.py to ensure consistent predictions.
    
    Args:
        x (numpy.ndarray): Input array of any shape
        
    Returns:
        numpy.ndarray: Sigmoid activation of the input, same shape as x
    """
    mask = x >= 0
    pos = np.zeros_like(x)
    neg = np.zeros_like(x)
    
    pos[mask] = 1 / (1 + np.exp(-x[mask]))
    neg[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    
    return pos + neg

class StockPredictor:
    """
    Neural network predictor for stock prices.
    
    Architecture:
    - Input layer: Configurable number of features
    - Hidden layer: Configurable number of neurons
    - Output layer: 1 neuron (predicted price)
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the predictor by loading saved weights and biases.
        
        Args:
            model_dir (str): Path to the model directory
        """
        if model_dir is None:
            # Find the most recent model directory
            model_dirs = glob.glob("model_*")
            if not model_dirs:
                raise FileNotFoundError("No model directories found. Please train a model first.")
            model_dir = max(model_dirs, key=os.path.getctime)
            print(f"Using most recent model directory: {model_dir}")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        # Load all parameters from NPZ file
        weights_file = os.path.join(model_dir, 'stock_model.npz')
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No model weights found in {model_dir}")
            
        with np.load(weights_file, allow_pickle=True) as data:
            # Load weights and biases
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            
            # Load normalization parameters
            self.X_min = data['X_min']
            self.X_max = data['X_max']
            
            # Load target normalization parameters if they exist
            self.Y_min = data['Y_min'] if data['Y_min'] is not None else None
            self.Y_max = data['Y_max'] if data['Y_max'] is not None else None
            self.has_target_norm = bool(data['has_target_norm'])
            
            # Store architecture parameters
            self.input_size = int(data['input_size'])
            self.hidden_size = int(data['hidden_size'])
        
        # Load feature info if available
        feature_info_path = os.path.join(model_dir, 'feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                self.expected_x_features = feature_info['x_features']
                self.expected_y_feature = feature_info['y_feature']
        else:
            self.expected_x_features = ['open', 'high', 'low', 'close', 'vol']
            self.expected_y_feature = 'close'

    def forward(self, X):
        """
        Make predictions using the loaded model.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted values of shape (n_samples, 1)
        """
        # Forward pass through the network
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        predictions = sigmoid(z2)
        
        return predictions

    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted values of shape (n_samples, 1)
        """
        # Normalize input data
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Get predictions
        predictions = self.forward(X_norm)
        
        # Denormalize predictions if target normalization parameters are available
        if self.has_target_norm:
            predictions = predictions * (self.Y_max - self.Y_min) + self.Y_min
            
        return predictions.flatten()

    @staticmethod
    def load_model(model_dir):
        """
        Load a trained model from a directory.
        
        Args:
            model_dir (str): Path to the model directory
        
        Returns:
            StockPredictor: Loaded model
        """
        return StockPredictor(model_dir)

def main():
    """
    Main function to run predictions on input data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using a trained neural network model.')
    parser.add_argument('input_file', type=str, help='Input CSV file containing features')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the model')
    parser.add_argument('--x_features', help='Comma-separated list of input features')
    parser.add_argument('--y_feature', help='Target feature')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
        
    # Load model
    try:
        model = StockPredictor.load_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load and prepare data
    try:
        df = pd.read_csv(args.input_file)
        
        # Determine features to use
        if args.x_features and args.y_feature:
            # Use command line arguments
            x_features = args.x_features.split(',')
            y_feature = args.y_feature
            
            # Validate that y_feature is not in x_features
            if y_feature in x_features:
                raise ValueError("Target feature cannot be used as an input feature")
                
            # Validate that all required features exist in the data
            required_features = x_features + [y_feature]
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features in CSV: {missing_features}")
                
        else:
            # Use features from model's feature_info.json
            x_features = model.expected_x_features
            y_feature = model.expected_y_feature
            
            # Validate that all expected features exist in the data
            missing_features = [f for f in x_features + [y_feature] if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features in CSV: {missing_features}")
        
        # Extract features
        X = df[x_features].values
        Y = df[y_feature].values.reshape(-1, 1) if y_feature in df.columns else None
        
        # Handle dates/timestamps
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            dates = pd.RangeIndex(len(df))
            
        # Make predictions
        predictions = model.predict(X)
        
        # Create results DataFrame
        results_data = {
            'date': dates,
            'predicted': predictions.flatten()
        }
        
        if Y is not None:
            results_data['actual'] = Y.flatten()
            results_data['error'] = (Y.flatten() - predictions).flatten()
        
        results = pd.DataFrame(results_data)
        
        # Save to CSV in model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(args.model_dir, f'predictions_{timestamp}.csv')
        results.to_csv(predictions_file, index=False)
        print(f"Predictions saved to: {predictions_file}")
        
        # Save prediction plot
        plots_dir = os.path.join(args.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if Y is not None:
            plt.plot(dates, Y, 'b-', label='Actual', alpha=0.7, linewidth=2)
        plt.plot(dates, predictions, 'r-', label='Predicted', alpha=0.7, linewidth=2)
        
        # Format x-axis for dates
        if isinstance(dates, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.xlabel('Sample')
        
        plt.title(f'Actual vs Predicted {y_feature.capitalize()}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display error metrics if we have actual values
        if Y is not None:
            mse = np.mean((Y.flatten() - predictions) ** 2)
            mae = np.mean(np.abs(Y.flatten() - predictions))
            rmse = np.sqrt(mse)
            plt.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}',
                     transform=plt.gca().transAxes, va='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plot_file = os.path.join(plots_dir, f'actual_vs_predicted_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Prediction plot saved to: {plot_file}")
        
        print(f"\nPrediction completed successfully!")
        print(f"Model: {args.model_dir}")
        print(f"Input features: {x_features}")
        print(f"Target feature: {y_feature}")
        print(f"Predictions: {predictions_file}")
        print(f"Plot: {plot_file}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()