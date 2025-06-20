"""
Stock Price Prediction Neural Network

This module implements a neural network for predicting stock prices using historical price and volume data.
The network uses a two-layer architecture with sigmoid activation functions and implements various
optimization techniques including momentum, mini-batch training, and early stopping.

Key Features:
- Two-layer neural network (input -> hidden -> output)
- Sigmoid activation with numerical stability
- Momentum-based optimization
- Mini-batch training
- Early stopping to prevent overfitting
- Data normalization
- Model weight saving/loading with timestamps
- Performance visualization and analysis

Usage:
    python stock_net.py

The script will:
1. Load stock data from CSV files
2. Preprocess and normalize the data
3. Train the neural network
4. Evaluate and visualize the model's performance
5. Save the trained weights
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import json

def sigmoid(x):
    """
    Numerically stable sigmoid activation function.
    
    This implementation handles positive and negative inputs separately to prevent overflow.
    For positive inputs: 1 / (1 + exp(-x))
    For negative inputs: exp(x) / (1 + exp(x))
    
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

def sigmoid_derivative(x):
    """
    Numerically stable derivative of sigmoid function.
    
    The derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)).
    This implementation includes clipping to prevent numerical instability.
    
    Args:
        x (numpy.ndarray): Input array (should be output of sigmoid function)
        
    Returns:
        numpy.ndarray: Derivative of sigmoid, same shape as x
    """
    return np.clip(x * (1 - x), 1e-8, 1.0)

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU(x) = max(0, x)
    
    Args:
        x (numpy.ndarray): Input array of any shape
        
    Returns:
        numpy.ndarray: ReLU activation of the input, same shape as x
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of ReLU activation function.
    
    The derivative of ReLU(x) is 1 if x > 0, 0 otherwise.
    
    Args:
        x (numpy.ndarray): Input array (should be pre-activation values)
        
    Returns:
        numpy.ndarray: Derivative of ReLU, same shape as x
    """
    return np.where(x > 0, 1, 0)

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

class StockNet:
    """
    Neural network for stock price prediction.
    
    Architecture:
    - Input layer: 5 features (open, high, low, close, volume)
    - Hidden layer: 16 neurons
    - Output layer: 1 neuron (predicted price)
    
    Features:
    - Xavier/Glorot weight initialization
    - Adam optimizer with adaptive learning rates
    - Mini-batch training
    - Early stopping
    - Weight saving/loading
    - Support for both sigmoid and ReLU activation functions
    """
    
    def __init__(self, input_size, hidden_size=4, output_size=1):
        """
        Initialize the neural network with specified architecture.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons
        """
        # Initialize weights using Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Adam optimizer parameters
        self.beta1 = 0.9  # First moment decay rate
        self.beta2 = 0.999  # Second moment decay rate
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # Initialize Adam moment variables (first moment - mean)
        self.m_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        
        # Initialize Adam moment variables (second moment - variance)
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        # Initialize time step for bias correction
        self.t = 0
        
        # Initialize normalization parameters
        self.X_min = None
        self.X_max = None
        self.Y_min = None
        self.Y_max = None
        self.has_target_norm = False

    def normalize(self, X, Y=None):
        """
        Normalize input features and target values.
        
        Args:
            X (numpy.ndarray): Input features
            Y (numpy.ndarray, optional): Target values
        
        Returns:
            tuple: Normalized X and Y (if provided)
        """
        if self.X_min is None or self.X_max is None:
            # Initialize normalization parameters
            self.X_min = X.min(axis=0)
            self.X_max = X.max(axis=0)
            # Add small epsilon to avoid division by zero
            self.X_max = np.where(self.X_max == self.X_min, 
                                 self.X_max + 1e-8, 
                                 self.X_max)
            
        # Normalize X
        X_norm = (X - self.X_min) / (self.X_max - self.X_min)
        
        if Y is not None:
            if not self.has_target_norm:
                self.Y_min = Y.min()
                self.Y_max = Y.max()
                self.has_target_norm = True
            
            # Normalize Y
            Y_norm = (Y - self.Y_min) / (self.Y_max - self.Y_min)
            return X_norm, Y_norm
            
        return X_norm

    def denormalize(self, Y_norm):
        """
        Denormalize target values.
        
        Args:
            Y_norm (numpy.ndarray): Normalized target values
            
        Returns:
            numpy.ndarray: Denormalized target values
        """
        if self.has_target_norm:
            return Y_norm * (self.Y_max - self.Y_min) + self.Y_min
        return Y_norm

    def save_weights(self, model_dir, prefix="stock_model"):
        """
        Save model weights and parameters to NPZ file.
        
        Args:
            model_dir (str): Directory to save the model
            prefix (str): Prefix for the saved files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save weights and normalization parameters
        np.savez(os.path.join(model_dir, f'{prefix}.npz'),
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 X_min=self.X_min,
                 X_max=self.X_max,
                 Y_min=self.Y_min,
                 Y_max=self.Y_max,
                 has_target_norm=self.has_target_norm,
                 input_size=self.W1.shape[0],
                 hidden_size=self.W1.shape[1])

    @classmethod
    def load_weights(cls, model_dir, prefix="stock_model"):
        """
        Load model weights and parameters from NPZ file.
        
        Args:
            model_dir (str): Directory containing the model
            prefix (str): Prefix of the saved files
            
        Returns:
            StockNet: Initialized model with loaded weights
        """
        weights_file = os.path.join(model_dir, f'{prefix}.npz')
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No model weights found in {model_dir}")
            
        with np.load(weights_file) as data:
            model = cls(input_size=int(data['input_size']), hidden_size=int(data['hidden_size']))
            model.W1 = data['W1']
            model.b1 = data['b1']
            model.W2 = data['W2']
            model.b2 = data['b2']
            
            # Validate weight shapes for consistency
            if model.W1.shape[1] != model.b1.shape[1] or model.W2.shape[0] != model.W1.shape[1]:
                raise ValueError("Inconsistent weight shapes in model")
            
            model.X_min = data['X_min']
            model.X_max = data['X_max']
            model.Y_min = data['Y_min']
            model.Y_max = data['Y_max']
            model.has_target_norm = bool(data['has_target_norm'])
        
        return model

    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Network output of shape (n_samples, 1)
        """
        # Store intermediate values for backward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        
        return self.output

    def backward(self, X, y, output, learning_rate=0.001):
        """
        Backward pass to update weights and biases using Adam optimizer.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            y (numpy.ndarray): Target values of shape (n_samples, 1)
            output (numpy.ndarray): Network output from forward pass
            learning_rate (float): Learning rate for weight updates
        """
        m = X.shape[0]  # Number of samples
        
        # Compute gradients for each layer
        self.error = y - output
        self.delta2 = np.clip(self.error, -1, 1)  # Output layer error
        self.delta1 = np.clip(np.dot(self.delta2, self.W2.T) * sigmoid_derivative(self.a1), -1, 1)  # Hidden layer error

        # Compute gradients
        dW2 = np.dot(self.a1.T, self.delta2) / m
        db2 = np.sum(self.delta2, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, self.delta1) / m
        db1 = np.sum(self.delta1, axis=0, keepdims=True) / m

        # Increment time step
        self.t += 1

        # Update first moment (mean) estimates
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2

        # Update second moment (variance) estimates
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (db1 ** 2)
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (db2 ** 2)

        # Bias correction
        m_W1_corrected = self.m_W1 / (1 - self.beta1 ** self.t)
        m_b1_corrected = self.m_b1 / (1 - self.beta1 ** self.t)
        m_W2_corrected = self.m_W2 / (1 - self.beta1 ** self.t)
        m_b2_corrected = self.m_b2 / (1 - self.beta1 ** self.t)

        v_W1_corrected = self.v_W1 / (1 - self.beta2 ** self.t)
        v_b1_corrected = self.v_b1 / (1 - self.beta2 ** self.t)
        v_W2_corrected = self.v_W2 / (1 - self.beta2 ** self.t)
        v_b2_corrected = self.v_b2 / (1 - self.beta2 ** self.t)

        # Update weights and biases with Adam
        self.W1 += learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)
        self.b1 += learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)
        self.W2 += learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)
        self.b2 += learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)

    def train(self, X, y, X_val=None, y_val=None, epochs=1000, learning_rate=0.001, batch_size=32, save_history=True, history_interval=10):
        """
        Train the neural network using mini-batch gradient descent with early stopping.
        
        Args:
            X (numpy.ndarray): Training data
            y (numpy.ndarray): Target values
            X_val (numpy.ndarray): Validation data (optional)
            y_val (numpy.ndarray): Validation target values (optional)
            epochs (int): Maximum number of training epochs
            learning_rate (float): Learning rate for weight updates
            batch_size (int): Size of mini-batches for training
            save_history (bool): Whether to save weight history for visualization
            history_interval (int): How often to save weight history (every N epochs)
            
        Returns:
            tuple: (train_losses, val_losses) containing loss history
        """
        n_samples = X.shape[0]
        best_mse = float('inf')
        patience = 20  # Number of epochs to wait for improvement
        patience_counter = 0
        
        # Initialize loss tracking
        train_losses = []
        val_losses = []
        
        # Create weights history directory if saving history
        if save_history:
            weights_history_dir = os.path.join(os.getcwd(), "weights_history")
            os.makedirs(weights_history_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            total_mse = 0
            n_batches = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get current batch
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)
                
                # Calculate batch MSE
                batch_mse = np.mean((output - y_batch) ** 2)
                total_mse += batch_mse
                n_batches += 1
            
            # Calculate average MSE for the epoch
            avg_mse = total_mse / n_batches
            train_losses.append(avg_mse)
            
            # Calculate validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_mse = np.mean((val_output - y_val) ** 2)
                val_losses.append(val_mse)
                current_mse = val_mse  # Use validation loss for early stopping
            else:
                val_losses.append(avg_mse)  # Use training loss as validation loss
                current_mse = avg_mse
            
            # Save weight history at regular intervals
            if save_history and (epoch % history_interval == 0 or epoch == epochs - 1):
                history_file = os.path.join(weights_history_dir, f"weights_history_{epoch:04d}.npz")
                np.savez(history_file, W1=self.W1, W2=self.W2)
            
            # Early stopping check
            if current_mse < best_mse:
                best_mse = current_mse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            # Print progress
            if epoch % 10 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch}, Train MSE: {avg_mse:.6f}, Val MSE: {val_mse:.6f}")
                else:
                    print(f"Epoch {epoch}, MSE: {avg_mse:.6f}")
        
        return train_losses, val_losses

def load_data_from_directory(directory_path):
    """
    Load and combine CSV files from a directory.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        
    Returns:
        pandas.DataFrame: Combined data from all CSV files
        
    Raises:
        FileNotFoundError: If no CSV files are found
        ValueError: If required columns are missing
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")
    
    # Read and concatenate all CSV files
    dfs = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'vol']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV files must contain columns: {required_columns}")
    
    return df

def normalize_features(X):
    """
    Normalize features using min-max scaling to [0,1] range.
    
    Args:
        X (numpy.ndarray): Input features
        
    Returns:
        tuple: (X_norm, X_min, X_max) containing normalized features and scaling parameters
    """
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    # Add small epsilon to prevent division by zero
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    np.savetxt('scaler_mean.csv', X_min, delimiter=',')
    np.savetxt('scaler_std.csv', X_max - X_min, delimiter=',')
    return X_norm, X_min, X_max

def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    """
    Manual train-test split implementation.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Target values
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) containing split data
    """
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_size = int(X.shape[0] * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """
    Calculate various regression metrics.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, R², and MAPE
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Calculate basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R²
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / (ss_total + 1e-10))
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Price Prediction Neural Network")
    parser.add_argument("--hidden_size", type=int, default=4,
                       help="Number of neurons in hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--x_features", type=str, default="open,high,low,close,vol,ma_10,rsi,price_change,volatility_10",
                       help="Comma-separated list of input features")
    parser.add_argument("--y_feature", type=str, default="close",
                       help="Target feature to predict")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to the input CSV file")
    
    args = parser.parse_args()
    
    # Validate and normalize data file path
    if not args.data_file:
        raise ValueError("Data file path is required")
    
    # Convert to absolute path if not already
    if not os.path.isabs(args.data_file):
        args.data_file = os.path.abspath(args.data_file)
    
    # Validate data file exists
    if not os.path.exists(args.data_file):
        raise ValueError(f"Data file not found: {args.data_file}")
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"model_{timestamp}"
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    # Create plots directory
    plots_dir = os.path.join(model_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(args.data_file)
    
    # Add technical indicators
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Validate features
    x_features = args.x_features.split(',')
    y_feature = args.y_feature
    
    required_features = x_features + [y_feature]
    if not all(col in df.columns for col in required_features):
        raise ValueError(f"CSV file must contain columns: {required_features}")
    
    # Prepare features and target
    print("Preparing features and target...")
    X = df[x_features].values
    Y = df[y_feature].values.reshape(-1, 1)
    dates = pd.to_datetime(df.index) if 'timestamp' in df.columns else pd.date_range(start='2020-01-01', periods=len(df))
    
    # Normalize data
    print("Normalizing data...")
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    
    Y_min = np.min(Y)
    Y_max = np.max(Y)
    Y = (Y - Y_min) / (Y_max - Y_min + 1e-8)
    
    # Save normalization parameters for inference
    print("Saving normalization parameters...")
    np.savetxt(os.path.join(model_dir, 'scaler_mean.csv'), X_min, delimiter=',')
    np.savetxt(os.path.join(model_dir, 'scaler_std.csv'), X_max - X_min, delimiter=',')
    np.savetxt(os.path.join(model_dir, 'target_min.csv'), np.array([Y_min]).reshape(1, -1), delimiter=',')
    np.savetxt(os.path.join(model_dir, 'target_max.csv'), np.array([Y_max]).reshape(1, -1), delimiter=',')
    
    # Save feature selection information
    feature_info = {
        'x_features': x_features,
        'y_feature': y_feature,
        'input_size': len(x_features)
    }
    with open(os.path.join(model_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f)
    
    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split_manual(X, Y, test_size=0.2, random_state=42)
    dates_train = dates[:len(X_train)]
    dates_test = dates[len(X_train):]
    
    # Initialize and train model
    print("\nInitializing and training model...")
    model = StockNet(input_size=len(x_features), hidden_size=args.hidden_size)
    
    # Save training data for visualization
    training_data = pd.DataFrame({
        **{feat: X_train[:, i] for i, feat in enumerate(x_features)},
        y_feature: Y_train.flatten()
    })
    training_data.to_csv(os.path.join(model_dir, 'training_data.csv'), index=False)
    
    # Train the model with weight history saving
    print("\nTraining model...")
    train_losses, val_losses = model.train(X_train, Y_train, X_val=X_test, y_val=Y_test, epochs=1000, learning_rate=args.learning_rate, 
                batch_size=args.batch_size, save_history=True, history_interval=10)
    
    # Move weights history to model directory
    if os.path.exists("weights_history"):
        import shutil
        weights_history_dir = os.path.join(model_dir, "weights_history")
        shutil.move("weights_history", weights_history_dir)
        print(f"Weight history saved to: {weights_history_dir}")
    
    # Save training losses
    np.savetxt(os.path.join(model_dir, 'training_losses.csv'), 
               np.column_stack((train_losses, val_losses)), delimiter=',')
    
    # Save model weights
    model.save_weights(model_dir, prefix="stock_model")
    
    # Save simple loss curve plot
    plots_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    print("\nTraining complete!")
    print(f"Final validation MSE: {val_losses[-1]:.6f}")
    print(f"Model directory: {model_dir}")
    print("\nUse predict.py to make predictions on new data.")
    print("Example: python predict.py /Users/porupine/redline/data/gamestop_us.csv --model_dir", model_dir)

    # Evaluate model on both training and test sets
    print("\nEvaluating model...")
    
    # Training set evaluation
    train_predictions = model.forward(X_train)
    train_predictions = train_predictions * (Y_max - Y_min) + Y_min
    Y_train_denorm = Y_train * (Y_max - Y_min) + Y_min
    train_metrics = calculate_metrics(Y_train_denorm, train_predictions)
    
    # Test set evaluation
    test_predictions = model.forward(X_test)
    test_predictions = test_predictions * (Y_max - Y_min) + Y_min
    Y_test_denorm = Y_test * (Y_max - Y_min) + Y_min
    test_metrics = calculate_metrics(Y_test_denorm, test_predictions)
    
    print("\nTraining Set Results:")
    print(f"Mean Squared Error: {train_metrics['mse']:.6f}")
    print(f"Root Mean Squared Error: {train_metrics['rmse']:.6f}")
    print(f"Mean Absolute Error: {train_metrics['mae']:.6f}")
    print(f"R² Score: {train_metrics['r2']:.6f}")
    print(f"Mean Absolute Percentage Error: {train_metrics['mape']:.2f}%")
    
    print("\nTest Set Results:")
    print(f"Mean Squared Error: {test_metrics['mse']:.6f}")
    print(f"Root Mean Squared Error: {test_metrics['rmse']:.6f}")
    print(f"Mean Absolute Error: {test_metrics['mae']:.6f}")
    print(f"R² Score: {test_metrics['r2']:.6f}")
    print(f"Mean Absolute Percentage Error: {test_metrics['mape']:.2f}%")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot actual vs predicted for training set
    plt.figure(figsize=(12, 6))
    plt.plot(dates_train, Y_train_denorm, label='Actual', alpha=0.7)
    plt.plot(dates_train, train_predictions, label='Predicted', alpha=0.7)
    plt.title('Training Set: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_predictions.png'))
    plt.close()
    
    # Plot actual vs predicted for test set
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, Y_test_denorm, label='Actual', alpha=0.7)
    plt.plot(dates_test, test_predictions, label='Predicted', alpha=0.7)
    plt.title('Test Set: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'test_predictions.png'))
    plt.close()
    
    # Plot prediction errors
    plt.figure(figsize=(12, 6))
    train_errors = train_predictions - Y_train_denorm.flatten()
    test_errors = test_predictions - Y_test_denorm.flatten()
    plt.hist(train_errors, bins=50, alpha=0.5, label='Training Errors')
    plt.hist(test_errors, bins=50, alpha=0.5, label='Test Errors')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
    plt.close()
    
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 10))
    plt.scatter(Y_train_denorm, train_predictions, alpha=0.5, label='Training')
    plt.scatter(Y_test_denorm, test_predictions, alpha=0.5, label='Test')
    plt.plot([Y_min, Y_max], [Y_min, Y_max], 'r--', label='Perfect Prediction')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    print(f"\nModel training complete. Model files and plots saved in directory: {model_dir}")
    print("Use predict.py to make predictions on new data.")
    print("Example: python predict.py /Users/porupine/redline/data/gamestop_us.csv --model_dir", model_dir)
