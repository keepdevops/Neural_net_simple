"""
Stock Price Prediction Inference Script

This script loads a trained neural network model and makes predictions on new stock data.
It uses the weights and biases saved by stock_net.py to perform inference.

Usage:
    python predict.py <input_csv_file> [--model_dir MODEL_DIR]

Arguments:
    input_csv_file    Path to the input CSV file containing stock data
    --model_dir       Path to the model directory (default: most recent model directory)

Example:
    python predict.py /Users/porupine/redline/data/gamestop_us.csv
    python predict.py /Users/porupine/redline/data/gamestop_us.csv --model_dir model_20240315_123456
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
    This is a simplified version of StockNet that only includes the forward pass
    and is used for making predictions with pre-trained weights.
    
    Architecture:
    - Input layer: 5 features (open, high, low, close, volume)
    - Hidden layer: 64 neurons
    - Output layer: 1 neuron (predicted price)
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the predictor by loading saved weights and biases.
        
        Args:
            model_dir (str): Path to the model directory. If None, uses the most recent model.
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
        
        # Find the model files
        model_files = glob.glob(os.path.join(model_dir, "stock_model_*_W1.csv"))
        if not model_files:
            raise FileNotFoundError(f"No model weights found in {model_dir}")
        
        # Get the model prefix from the first weight file
        model_prefix = os.path.splitext(model_files[0])[0].replace("_W1", "")
        
        # Load weights and biases
        self.W1 = np.loadtxt(f'{model_prefix}_W1.csv', delimiter=',')
        self.b1 = np.loadtxt(f'{model_prefix}_b1.csv', delimiter=',')
        self.W2 = np.loadtxt(f'{model_prefix}_W2.csv', delimiter=',')
        self.b2 = np.loadtxt(f'{model_prefix}_b2.csv', delimiter=',')
        
        # Load normalization parameters
        self.X_min = np.loadtxt(os.path.join(model_dir, 'scaler_mean.csv'), delimiter=',')
        self.X_max = np.loadtxt(os.path.join(model_dir, 'scaler_std.csv'), delimiter=',') + self.X_min
        
        # Load target normalization parameters
        try:
            self.Y_min = np.loadtxt(os.path.join(model_dir, 'target_min.csv'), delimiter=',')
            self.Y_max = np.loadtxt(os.path.join(model_dir, 'target_max.csv'), delimiter=',')
            self.has_target_norm = True
        except:
            self.has_target_norm = False
            print("Warning: Target normalization parameters not found. Predictions will be in normalized form.")
        
        # Load and print model metadata if available
        metadata_file = os.path.join(model_dir, 'model_metadata.txt')
        if os.path.exists(metadata_file):
            print("\nModel Information:")
            with open(metadata_file, 'r') as f:
                print(f.read())

    def predict(self, X):
        """
        Make predictions using the loaded model.
        
        Args:
            X (numpy.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted values, denormalized if target normalization parameters are available
        """
        # Normalize input features
        X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Forward pass through the network
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Denormalize predictions if target normalization parameters are available
        if self.has_target_norm:
            predictions = z2 * (self.Y_max - self.Y_min) + self.Y_min
        else:
            predictions = z2
            
        return predictions.flatten()

def main():
    """
    Main function to run predictions on input data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions using a trained stock price model.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--model_dir', help='Path to the model directory')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        print("\nPlease provide the full path to the input file.")
        print("Example: python predict.py /Users/porupine/redline/data/gamestop_us.csv")
        sys.exit(1)
    
    try:
        # Load input data
        print(f"Loading data from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'vol']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Prepare features
        print("Preparing features...")
        X = df[required_columns].values
        
        # Initialize predictor and make predictions
        print("Making predictions...")
        predictor = StockPredictor(model_dir=args.model_dir)
        predictions = predictor.predict(X)
        
        # Add predictions to dataframe
        df['predicted_close'] = predictions
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'predictions_{timestamp}.csv'
        
        # Save results
        print(f"Saving predictions to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print("\nPrediction Summary:")
        print(f"Number of predictions made: {len(predictions)}")
        print(f"Predictions saved to: {output_file}")
        
        # Print some example predictions
        print("\nExample predictions (first 5 rows):")
        print(df[['close', 'predicted_close']].head().to_string())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 