"""
Training script for Stock Price Prediction Neural Network

Usage:
    python train.py --data_file <path_to_csv> --model_dir <model_directory> --x_features <features> --y_feature <target>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stock_net import StockNet

def train_model(data_file, model_dir, x_features, y_feature, hidden_size=4, learning_rate=0.001, batch_size=32):
    """
    Train a neural network model for stock price prediction.
    
    Args:
        data_file (str): Path to the input CSV data file
        model_dir (str): Directory to save the trained model
        x_features (list): List of input feature names
        y_feature (str): Name of the target feature
        hidden_size (int): Size of the hidden layer
        learning_rate (float): Learning rate for weight updates
        batch_size (int): Size of mini-batches for training
    """
    print(f"\nTraining parameters:")
    print(f"Data File: {data_file}")
    print(f"Model Dir: {model_dir}")
    print(f"X Features: {x_features}")
    print(f"Y Feature: {y_feature}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    
    try:
        # Validate input paths
        if not os.path.exists(data_file):
            raise ValueError(f"Data file not found: {data_file}")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load and preprocess data
        print("\nLoading data...")
        df = pd.read_csv(data_file)
        
        # Validate features
        if not all(col in df.columns for col in x_features):
            raise ValueError(f"Some X features not found in data: {x_features}")
        if y_feature not in df.columns:
            raise ValueError(f"Y feature not found in data: {y_feature}")
        
        X = df[x_features].values
        y = df[y_feature].values.reshape(-1, 1)
        
        # Normalize data
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        y_mean = y.mean()
        y_std = y.std()
        
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
        
        # Initialize neural network
        input_size = len(x_features)
        net = StockNet(input_size, hidden_size, 1)
        
        # Train the model
        print(f"\nTraining model with parameters:")
        print(f"Hidden Size: {hidden_size}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Batch Size: {batch_size}")
        print(f"Input Features: {x_features}")
        print(f"Target Feature: {y_feature}")
        print(f"Data File: {data_file}")
        print(f"Model Directory: {model_dir}")
        
        # Train the model
        net.train(X, y, 
                 learning_rate=learning_rate,
                 batch_size=batch_size,
                 epochs=1000)
        
        # Save model
        model_path = os.path.join(model_dir, "stock_model.npz")
        net.save(model_path)
        
        # Save normalization parameters
        norm_params = {
            'X_mean': X_mean.tolist(),
            'X_std': X_std.tolist(),
            'y_mean': float(y_mean),
            'y_std': float(y_std),
            'x_features': x_features,
            'y_feature': y_feature
        }
        
        with open(os.path.join(model_dir, "normalization.json"), 'w') as f:
            json.dump(norm_params, f, indent=4)
        
        print(f"\nModel training completed successfully!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a stock price prediction model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input CSV data file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--x_features", type=str, required=True, help="Comma-separated list of input features")
    parser.add_argument("--y_feature", type=str, required=True, help="Target feature name")
    parser.add_argument("--hidden_size", type=int, default=4, help="Size of hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    print(f"\nReceived arguments:")
    print(f"Data File: {args.data_file}")
    print(f"Model Dir: {args.model_dir}")
    print(f"X Features: {args.x_features}")
    print(f"Y Feature: {args.y_feature}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    
    # Convert x_features from comma-separated string to list
    x_features = args.x_features.split(',')
    
    # Validate features
    if not x_features:
        raise ValueError("No input features specified")
    
    # Run training
    train_model(args.data_file, args.model_dir, x_features, args.y_feature,
               args.hidden_size, args.learning_rate, args.batch_size)
