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
            
        with np.load(weights_file) as data:
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
    parser.add_argument('--x_features', type=str, help='Comma-separated list of input features')
    parser.add_argument('--y_feature', type=str, help='Target feature name')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.x_features or not args.y_feature:
        print("Error: Both --x_features and --y_feature must be specified")
        parser.print_help()
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
        
        # Validate features
        x_features = args.x_features.split(',')
        if not all(feature in df.columns for feature in x_features):
            print(f"Error: Some features not found in data: {x_features}")
            return
            
        if args.y_feature not in df.columns:
            print(f"Error: Target feature not found in data: {args.y_feature}")
            return
            
        X = df[x_features].values
        Y = df[args.y_feature].values.reshape(-1, 1)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Save results
        results = pd.DataFrame({
            'Actual': Y.flatten(),
            'Predicted': predictions.flatten(),
            'Error': (Y - predictions).flatten()
        })
        
        # Save to CSV in model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.model_dir, f'predictions_{timestamp}.csv')
        results.to_csv(results_file, index=False)
        print(f"Saved predictions to: {results_file}")
        
        # Print summary statistics
        print("\nPrediction Statistics:")
        print(f"Mean Absolute Error: {np.mean(np.abs(results['Error'])):.4f}")
        print(f"Mean Squared Error: {np.mean(results['Error'] ** 2):.4f}")
        print(f"R-squared: {1 - np.sum((Y - predictions) ** 2) / np.sum((Y - Y.mean()) ** 2):.4f}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()