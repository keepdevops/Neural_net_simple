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

class StockNet:
    """
    Neural network for stock price prediction.
    
    Architecture:
    - Input layer: 5 features (open, high, low, close, volume)
    - Hidden layer: 16 neurons
    - Output layer: 1 neuron (predicted price)
    
    Features:
    - Xavier/Glorot weight initialization
    - Momentum-based optimization
    - Mini-batch training
    - Early stopping
    - Weight saving/loading
    """
    
    def __init__(self, input_size, hidden_size=16, output_size=1):
        """
        Initialize the neural network with specified architecture.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons
        """
        # Initialize weights using Xavier/Glorot initialization
        # This helps prevent vanishing/exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Initialize momentum parameters
        self.momentum = 0.9  # Momentum coefficient
        self.v_W1 = np.zeros_like(self.W1)  # Velocity for W1
        self.v_b1 = np.zeros_like(self.b1)  # Velocity for b1
        self.v_W2 = np.zeros_like(self.W2)  # Velocity for W2
        self.v_b2 = np.zeros_like(self.b2)  # Velocity for b2

    def forward(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Network output of shape (n_samples, 1)
        """
        # Forward propagation through each layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Hidden layer linear transformation
        self.a1 = sigmoid(self.z1)              # Hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Output layer linear transformation
        return self.z2                          # Return output (no activation for regression)

    def backward(self, X, y, output, learning_rate=0.001):
        """
        Perform backward propagation and update weights using momentum.
        
        Args:
            X (numpy.ndarray): Input data
            y (numpy.ndarray): Target values
            output (numpy.ndarray): Network output from forward pass
            learning_rate (float): Learning rate for weight updates
        """
        m = X.shape[0]  # batch size
        
        # Compute gradients for each layer
        self.error = y - output
        self.delta2 = np.clip(self.error, -1, 1)  # Output layer error
        self.delta1 = np.clip(np.dot(self.delta2, self.W2.T) * sigmoid_derivative(self.a1), -1, 1)  # Hidden layer error

        # Compute gradients with momentum
        dW2 = np.dot(self.a1.T, self.delta2) / m
        db2 = np.sum(self.delta2, axis=0, keepdims=True) / m
        dW1 = np.dot(X.T, self.delta1) / m
        db1 = np.sum(self.delta1, axis=0, keepdims=True) / m

        # Update momentum
        self.v_W2 = self.momentum * self.v_W2 + learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + learning_rate * db1

        # Update weights with momentum
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train(self, X, y, epochs=1000, learning_rate=0.001, batch_size=32):
        """
        Train the neural network using mini-batch gradient descent with early stopping.
        
        Args:
            X (numpy.ndarray): Training data
            y (numpy.ndarray): Target values
            epochs (int): Maximum number of training epochs
            learning_rate (float): Learning rate for weight updates
            batch_size (int): Size of mini-batches for training
        """
        n_samples = X.shape[0]
        best_mse = float('inf')
        patience = 20  # Number of epochs to wait for improvement
        patience_counter = 0
        
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
            
            # Early stopping check
            if avg_mse < best_mse:
                best_mse = avg_mse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE: {avg_mse:.6f}")

    def save_weights(self, prefix='model'):
        """
        Save network weights and biases to CSV files.
        
        Args:
            prefix (str): Prefix for the saved files
        """
        np.savetxt(f'{prefix}_W1.csv', self.W1, delimiter=',')
        np.savetxt(f'{prefix}_b1.csv', self.b1, delimiter=',')
        np.savetxt(f'{prefix}_W2.csv', self.W2, delimiter=',')
        np.savetxt(f'{prefix}_b2.csv', self.b2, delimiter=',')

    @staticmethod
    def load_weights(prefix='model'):
        """
        Load network weights and biases from CSV files.
        
        Args:
            prefix (str): Prefix of the saved files
            
        Returns:
            tuple: (W1, b1, W2, b2) containing the loaded weights and biases
        """
        W1 = np.loadtxt(f'{prefix}_W1.csv', delimiter=',')
        b1 = np.loadtxt(f'{prefix}_b1.csv', delimiter=',')
        W2 = np.loadtxt(f'{prefix}_W2.csv', delimiter=',')
        b2 = np.loadtxt(f'{prefix}_b2.csv', delimiter=',')
        return W1, b1, W2, b2

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
    directory_path = "/Users/porupine/redline/data/gamestop_us.csv"
    print("Loading data...")
    df = pd.read_csv(directory_path)
    
    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close', 'vol']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    # Prepare features and target
    print("Preparing features and target...")
    X = df[['open', 'high', 'low', 'close', 'vol']].values
    Y = df['close'].values.reshape(-1, 1)
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

    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split_manual(X, Y, test_size=0.2, random_state=42)
    dates_train = dates[:len(X_train)]
    dates_test = dates[len(X_train):]

    # Initialize and train model
    print("\nInitializing and training model...")
    model = StockNet(input_size=5, hidden_size=16)
    
    # Track training metrics
    train_losses = []
    val_losses = []
    
    def train_with_validation(model, X_train, Y_train, X_val, Y_val, epochs=1000, learning_rate=0.0001, batch_size=32):
        n_samples = X_train.shape[0]
        best_val_mse = float('inf')
        patience = 20
        patience_counter = 0
        
        # Create directory for saving weights history
        weights_dir = os.path.join(model_dir, 'weights_history')
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        
        # Save initial weights
        weights_history = []
        weights_history.append(np.concatenate([model.W1.flatten(), model.W2.flatten()]))
        np.savetxt(os.path.join(weights_dir, f'weights_history_0000.csv'), 
                   weights_history[-1], delimiter=',')
        
        for epoch in range(epochs):
            # Training
            indices = np.random.permutation(n_samples)
            total_mse = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = Y_train[batch_indices]
                
                output = model.forward(X_batch)
                model.backward(X_batch, y_batch, output, learning_rate)
                
                batch_mse = np.mean((output - y_batch) ** 2)
                total_mse += batch_mse
                n_batches += 1
            
            avg_train_mse = total_mse / n_batches
            train_losses.append(avg_train_mse)
            
            # Save weights every 10 epochs
            if epoch % 10 == 0:
                weights_history.append(np.concatenate([model.W1.flatten(), model.W2.flatten()]))
                np.savetxt(os.path.join(weights_dir, f'weights_history_{epoch:04d}.csv'), 
                          weights_history[-1], delimiter=',')
            
            # Validation
            val_output = model.forward(X_val)
            val_mse = np.mean((val_output - Y_val) ** 2)
            val_losses.append(val_mse)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train MSE: {avg_train_mse:.6f}, Val MSE: {val_mse:.6f}")
        
        # Save final weights
        weights_history.append(np.concatenate([model.W1.flatten(), model.W2.flatten()]))
        np.savetxt(os.path.join(weights_dir, f'weights_history_{epoch:04d}.csv'), 
                  weights_history[-1], delimiter=',')
        
        # Save training losses
        np.savetxt(os.path.join(model_dir, 'training_losses.csv'), 
                  np.array(train_losses), delimiter=',')
    
    # Train with validation
    train_with_validation(model, X_train, Y_train, X_test, Y_test)

    # Save trained weights with timestamp
    print("\nSaving model weights...")
    model_prefix = os.path.join(model_dir, f'stock_model_{timestamp}')
    model.save_weights(prefix=model_prefix)

    # Save model metadata
    metadata = {
        'timestamp': timestamp,
        'input_size': 5,
        'hidden_size': 16,
        'output_size': 1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10000,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Save metadata to a text file
    with open(os.path.join(model_dir, 'model_metadata.txt'), 'w') as f:
        f.write("Model Training Metadata\n")
        f.write("=====================\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

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
    print("Example: python predict.py gamestop_us.csv --model_dir", model_dir)
