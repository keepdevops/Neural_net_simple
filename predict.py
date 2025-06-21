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

def compute_rsi(prices, period=14):
    """
    Compute Relative Strength Index (RSI) for a price series.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        prices (pandas.Series): Price series
        period (int): Period for RSI calculation (default: 14)
        
    Returns:
        pandas.Series: RSI values
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using exponential moving average
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying original
    df_enhanced = df.copy()
    
    # Moving averages
    df_enhanced['ma_5'] = df_enhanced['close'].rolling(window=5).mean()
    df_enhanced['ma_10'] = df_enhanced['close'].rolling(window=10).mean()
    df_enhanced['ma_20'] = df_enhanced['close'].rolling(window=20).mean()
    
    # RSI
    df_enhanced['rsi'] = compute_rsi(df_enhanced['close'], 14)
    
    # Price changes
    df_enhanced['price_change'] = df_enhanced['close'].pct_change()
    df_enhanced['price_change_5'] = df_enhanced['close'].pct_change(periods=5)
    
    # Volatility (rolling standard deviation)
    df_enhanced['volatility_10'] = df_enhanced['close'].rolling(window=10).std()
    
    # Bollinger Bands
    df_enhanced['bb_middle'] = df_enhanced['close'].rolling(window=20).mean()
    bb_std = df_enhanced['close'].rolling(window=20).std()
    df_enhanced['bb_upper'] = df_enhanced['bb_middle'] + (bb_std * 2)
    df_enhanced['bb_lower'] = df_enhanced['bb_middle'] - (bb_std * 2)
    
    # MACD
    exp1 = df_enhanced['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_enhanced['close'].ewm(span=26, adjust=False).mean()
    df_enhanced['macd'] = exp1 - exp2
    df_enhanced['macd_signal'] = df_enhanced['macd'].ewm(span=9, adjust=False).mean()
    
    # Volume indicators
    df_enhanced['volume_ma'] = df_enhanced['vol'].rolling(window=10).mean()
    df_enhanced['volume_ratio'] = df_enhanced['vol'] / df_enhanced['volume_ma']
    
    return df_enhanced

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
            
        # Initialize normalization parameters to None by default
        self.X_min = None
        self.X_max = None
        self.X_mean = None
        self.X_std = None
        self.Y_min = None
        self.Y_max = None
        self.use_standardization = False
        self.has_target_norm = False
        
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
            
            # Validate weight shapes for consistency
            if self.W1.shape[1] != self.b1.shape[1] or self.W2.shape[0] != self.W1.shape[1]:
                raise ValueError("Inconsistent weight shapes in model")
            
            # Print hidden layer size
            print(f"Loaded model with hidden layer size: {self.W1.shape[1]}")
            
            # Try to load normalization parameters from NPZ first
            npz_has_norm = False
            try:
                # Check if normalization parameters exist in NPZ and are valid
                if 'X_min' in data and 'X_max' in data:
                    x_min = data['X_min']
                    x_max = data['X_max']
                    
                    # Check if the values are valid (not None and not NaN)
                    if x_min is not None and x_max is not None:
                        # Convert to numpy arrays if they aren't already
                        if not isinstance(x_min, np.ndarray):
                            x_min = np.array(x_min)
                        if not isinstance(x_max, np.ndarray):
                            x_max = np.array(x_max)
                        
                        # Check for NaN values
                        if not np.any(np.isnan(x_min)) and not np.any(np.isnan(x_max)):
                            self.X_min = x_min
                            self.X_max = x_max
                            self.Y_min = data['Y_min'] if 'Y_min' in data else None
                            self.Y_max = data['Y_max'] if 'Y_max' in data else None
                            self.has_target_norm = bool(data['has_target_norm']) if 'has_target_norm' in data else False
                            self.use_standardization = False
                            npz_has_norm = True
                            print("Loaded normalization parameters from NPZ file")
                        else:
                            print("Invalid normalization parameters in NPZ file (NaN values), loading from CSV files...")
                    else:
                        print("Invalid normalization parameters in NPZ file (None values), loading from CSV files...")
                else:
                    print("No normalization parameters in NPZ file, loading from CSV files...")
            except Exception as e:
                print(f"Error loading from NPZ: {e}, loading from CSV files...")
            
            # If NPZ doesn't have valid normalization parameters, load from CSV files
            if not npz_has_norm:
                self._load_normalization_from_csv(model_dir)
            
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

    def _load_normalization_from_csv(self, model_dir):
        """Load normalization parameters from separate CSV files."""
        try:
            # Load standardization parameters (mean and std)
            scaler_mean_file = os.path.join(model_dir, 'scaler_mean.csv')
            scaler_std_file = os.path.join(model_dir, 'scaler_std.csv')
            
            if os.path.exists(scaler_mean_file) and os.path.exists(scaler_std_file):
                self.X_mean = np.loadtxt(scaler_mean_file)
                self.X_std = np.loadtxt(scaler_std_file)
                self.use_standardization = True
                print(f"Loaded standardization parameters: mean={self.X_mean}, std={self.X_std}")
            else:
                # Fallback to min-max normalization if standardization files don't exist
                self.use_standardization = False
                print("Standardization files not found, using min-max normalization")
            
            # Load target normalization parameters
            target_min_file = os.path.join(model_dir, 'target_min.csv')
            target_max_file = os.path.join(model_dir, 'target_max.csv')
            
            if os.path.exists(target_min_file) and os.path.exists(target_max_file):
                self.Y_min = np.loadtxt(target_min_file)
                self.Y_max = np.loadtxt(target_max_file)
                self.has_target_norm = True
                print(f"Loaded target normalization: min={self.Y_min}, max={self.Y_max}")
            else:
                self.Y_min = None
                self.Y_max = None
                self.has_target_norm = False
                print("Target normalization files not found")
                
        except Exception as e:
            print(f"Error loading normalization parameters: {e}")
            # Set default values to prevent crashes
            self.X_mean = None
            self.X_std = None
            self.use_standardization = False
            self.Y_min = None
            self.Y_max = None
            self.has_target_norm = False

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
        if self.use_standardization and self.X_mean is not None and self.X_std is not None:
            # Use standardization (z-score normalization)
            X_norm = (X - self.X_mean) / (self.X_std + 1e-8)
            print(f"Using standardization: mean={self.X_mean}, std={self.X_std}")
        elif self.X_min is not None and self.X_max is not None:
            # Use min-max normalization
            X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
            print(f"Using min-max normalization: min={self.X_min}, max={self.X_max}")
        else:
            # No normalization available, use raw data
            print("Warning: No normalization parameters found, using raw data")
            X_norm = X
        
        # Get predictions
        predictions = self.forward(X_norm)
        
        # Denormalize predictions if target normalization parameters are available
        if self.has_target_norm and self.Y_min is not None and self.Y_max is not None:
            predictions = predictions * (self.Y_max - self.Y_min) + self.Y_min
            print(f"Denormalized predictions using: min={self.Y_min}, max={self.Y_max}")
            
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
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save predictions and plots')
    parser.add_argument('--output_file', type=str, help='Output filename for predictions (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
        
    # Validate and create output directory
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error: Cannot create output directory '{args.output_dir}': {e}")
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
        
        # Add technical indicators
        print("Adding technical indicators...")
        df = add_technical_indicators(df)
        
        # Determine features to use
        if args.x_features and args.y_feature:
            required_columns = args.x_features.split(',')
            # Validate that y_feature is not in x_features
            if args.y_feature in required_columns:
                raise ValueError("Target feature cannot be used as an input feature")
            X = df[required_columns].values
        else:
            required_columns = ['open', 'high', 'low', 'close', 'vol']
            X = df[required_columns].values
        
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
        
        if args.y_feature in df.columns:
            results_data['actual'] = df[args.y_feature].values.flatten()
            results_data['error'] = (df[args.y_feature].values.flatten() - predictions).flatten()
        
        results = pd.DataFrame(results_data)
        
        # Save to CSV in output directory
        if args.output_file:
            predictions_file = os.path.join(args.output_dir, args.output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictions_file = os.path.join(args.output_dir, f'predictions_{timestamp}.csv')
        
        results.to_csv(predictions_file, index=False)
        print(f"Predictions saved to: {predictions_file}")
        
        # Save prediction plot in output directory
        plots_dir = os.path.join(args.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if args.y_feature in df.columns:
            plt.plot(dates, df[args.y_feature].values, 'b-', label='Actual', alpha=0.7, linewidth=2)
        plt.plot(dates, predictions, 'r-', label='Predicted', alpha=0.7, linewidth=2)
        
        # Format x-axis for dates
        if isinstance(dates, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.xlabel('Sample')
        
        plt.title(f'Actual vs Predicted {args.y_feature.capitalize()}')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display error metrics if we have actual values
        if args.y_feature in df.columns:
            mse = np.mean((df[args.y_feature].values.flatten() - predictions) ** 2)
            mae = np.mean(np.abs(df[args.y_feature].values.flatten() - predictions))
            rmse = np.sqrt(mse)
            plt.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}',
                     transform=plt.gca().transAxes, va='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Generate timestamp for plot filename
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(plots_dir, f'actual_vs_predicted_{plot_timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Prediction plot saved to: {plot_file}")
        
        # Also save plot in model directory for easy access
        model_plots_dir = os.path.join(args.model_dir, 'plots')
        os.makedirs(model_plots_dir, exist_ok=True)
        
        # Create a simpler version for the model directory
        plt.figure(figsize=(10, 6))
        if args.y_feature in df.columns:
            plt.plot(df[args.y_feature], label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title('Actual vs Predicted Prices')
        plt.xlabel('Sample')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        model_plot_file = os.path.join(model_plots_dir, f'actual_vs_predicted_{plot_timestamp}.png')
        plt.savefig(model_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model plot saved to: {model_plot_file}")
        
        print(f"\nPrediction completed successfully!")
        print(f"Model: {args.model_dir}")
        print(f"Input features: {args.x_features}")
        print(f"Target feature: {args.y_feature}")
        print(f"Predictions: {predictions_file}")
        print(f"Plot: {plot_file}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()