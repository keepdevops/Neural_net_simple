"""
Stock Price Prediction GUI

This module provides a graphical user interface for the stock price prediction system.
It integrates training, prediction, and visualization capabilities into a single interface.

Features:
- Train new models
- Make predictions
- View training results and plots
- Compare different model versions
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from datetime import datetime
import glob
import json
import numpy as np
import tempfile
import threading
import time
import re
from collections import OrderedDict
import concurrent.futures
import importlib.util
from PIL import Image, ImageTk
import path_utils
import script_launcher
from matplotlib.animation import FuncAnimation

# Optional plotly import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Interactive plots will be disabled.")

# Custom NavigationToolbar that uses grid instead of pack
class GridMatplotlibToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        # Create a frame to hold the toolbar
        self.toolbar_frame = tk.Frame(window)
        super().__init__(canvas, self.toolbar_frame)
        # Use grid instead of pack for the toolbar frame
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        # Do not create custom buttons; NavigationToolbar2Tk handles this

# Import stock_net module
try:
    import stock_net
except ImportError:
    # If stock_net is not available, we'll implement the needed functions inline
    pass

# Neural Network Implementation
def sigmoid(x):
    """Numerically stable sigmoid activation function."""
    mask = x >= 0
    pos = np.zeros_like(x)
    neg = np.zeros_like(x)
    
    pos[mask] = 1 / (1 + np.exp(-x[mask]))
    neg[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    
    return pos + neg

def sigmoid_derivative(x):
    """Numerically stable derivative of sigmoid function."""
    return np.clip(x * (1 - x), 1e-8, 1.0)

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
    """Neural network for stock price prediction."""
    
    def __init__(self, input_size, hidden_size=4, output_size=1):
        """Initialize the neural network with specified architecture."""
        # Initialize weights using Xavier/Glorot initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Initialize momentum parameters
        self.momentum = 0.9
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        
        # Initialize normalization parameters
        self.X_min = None
        self.X_max = None
        self.Y_min = None
        self.Y_max = None
        self.has_target_norm = False

    def forward(self, X):
        """Forward pass through the network."""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        
        return self.output

    def backward(self, X, y, output, learning_rate=0.001):
        """Backward pass to compute gradients and update weights."""
        # Compute gradients
        delta2 = (output - y) * sigmoid_derivative(output)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)
        
        # Compute weight gradients
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Update weights with momentum
        self.v_W2 = self.momentum * self.v_W2 + learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + learning_rate * db1
        
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1

    def save_weights(self, model_dir, prefix="stock_model"):
        """Save model weights and parameters to NPZ file."""
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

# Color Scheme
BACKGROUND_COLOR = "#333333"  # Dark steel grey
TEXT_COLOR = "#CCCCCC"       # Light grey for text
FRAME_COLOR = "#444444"      # Slightly lighter steel grey
BUTTON_COLOR = "#555555"     # Button background
HOVER_COLOR = "#666666"      # Hover effect

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction System")
        self.root.geometry("1600x900")  # Increased window size
        
        # Configure root window style
        self.root.configure(bg=BACKGROUND_COLOR)
        
        # Set minimum window size to ensure panels are visible
        self.root.minsize(1200, 600)
        
        # Initialize style
        self.style = ttk.Style()
        self.style.theme_use("default")  # Use default theme for better control
        
        # Configure style for all widgets
        self.style.configure("TLabel", 
                           background=BACKGROUND_COLOR,
                           foreground=TEXT_COLOR)
        self.style.configure("Bold.TLabel", 
                           background=BACKGROUND_COLOR,
                           foreground=TEXT_COLOR,
                           font=("TkDefaultFont", 9, "bold"))
        self.style.configure("TFrame",
                           background=FRAME_COLOR)
        
        # Initialize image cache for plots (optimization) - Updated to OrderedDict for LRU
        self.plot_image_cache = OrderedDict()  # LRU cache for images
        self.max_cache_size = 10  # Maximum number of cached images
        
        # Initialize thread pool for plot loading operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.plot_futures = {}  # Track ongoing plot loading operations
        
        # Initialize variables
        self.data_file = None
        self.data_file_var = tk.StringVar()  # Variable for data file entry
        self.current_model_dir = "."
        self.selected_model_path = None  # Track the currently selected model path
        self.x_features = []
        self.y_feature = ""
        self.features_locked = False
        self.locked_features = []
        self.is_training = False
        self.training_thread = None
        
        # Initialize output directory
        self.output_dir = "."  # Default to current directory
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        
        # Initialize model directory and create it if it doesn't exist
        os.makedirs(self.current_model_dir, exist_ok=True)
        
        # Initialize feature variables
        self.feature_list = []  # Available features from data file
        
        # Initialize training parameters
        self.hidden_size_var = tk.StringVar(value="4")  # Default hidden layer size
        self.learning_rate_var = tk.StringVar(value="0.001")  # Default learning rate
        self.batch_size_var = tk.StringVar(value="32")  # Default batch size
        
        # Gradient descent visualization parameters
        self.color_var = tk.StringVar(value="red")
        self.point_size_var = tk.StringVar(value="100")
        self.line_width_var = tk.StringVar(value="3")
        self.surface_alpha_var = tk.StringVar(value="0.3")
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")  # Status bar variable
        self.feature_status_var = tk.StringVar(value="")  # Feature status variable
        
        # Selected prediction file
        self.selected_prediction_file = None
        
        # Test script availability on startup
        self.test_script_availability()
        
        # Create main layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=7)  # Controls panel (70%)
        self.root.grid_columnconfigure(1, weight=3)  # Display panel (30%)
        
        # Create control panel
        self.create_control_panel()
        
        # Create display panel
        self.create_display_panel()
        
        # Initialize model list
        self.refresh_models()

    def test_script_availability(self):
        """Test if all required scripts are available and log the results."""
        try:
            availability = script_launcher.launcher.test_script_availability()
            
            # Log results
            print("Script availability check:")
            for script, available in availability.items():
                status = "✅ Available" if available else "❌ Not found"
                print(f"  {script}: {status}")
            
            # Check if all scripts are available
            all_available = all(availability.values())
            if all_available:
                print("✅ All required scripts are available")
                self.status_var.set("Ready - All scripts available")
            else:
                missing_scripts = [script for script, available in availability.items() if not available]
                print(f"⚠️  Missing scripts: {missing_scripts}")
                self.status_var.set(f"Warning - Missing scripts: {', '.join(missing_scripts)}")
                
        except Exception as e:
            print(f"Error testing script availability: {e}")
            self.status_var.set("Error testing script availability")

    def create_control_panel(self):
        """Create the left control panel with data selection, training parameters, and model management."""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for different control sections
        self.control_notebook = ttk.Notebook(control_frame)
        self.control_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Data Selection Tab
        data_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(data_frame, text="Data Selection")
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Data file selection
        data_label = ttk.Label(data_frame, text="Data File:", style="Bold.TLabel")
        data_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        data_entry_frame = ttk.Frame(data_frame)
        data_entry_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        data_entry_frame.grid_columnconfigure(0, weight=1)
        
        self.data_entry = ttk.Entry(data_entry_frame, textvariable=self.data_file_var, state="readonly")
        self.data_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        browse_btn = ttk.Button(data_entry_frame, text="Browse", command=self.browse_data_file)
        browse_btn.grid(row=0, column=1)
        
        # Feature selection
        feature_label = ttk.Label(data_frame, text="Features:", style="Bold.TLabel")
        feature_label.grid(row=2, column=0, sticky="w", pady=(10, 5))
        
        # Feature listbox with scrollbar
        feature_frame = ttk.Frame(data_frame)
        feature_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        feature_frame.grid_columnconfigure(0, weight=1)
        feature_frame.grid_rowconfigure(0, weight=1)
        
        self.feature_listbox = tk.Listbox(feature_frame, selectmode=tk.MULTIPLE, height=8)
        self.feature_listbox.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        feature_scrollbar = ttk.Scrollbar(feature_frame, orient="vertical", command=self.feature_listbox.yview)
        feature_scrollbar.grid(row=0, column=1, sticky="ns")
        self.feature_listbox.configure(yscrollcommand=feature_scrollbar.set)
        
        # Feature buttons
        feature_btn_frame = ttk.Frame(data_frame)
        feature_btn_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        feature_btn_frame.grid_columnconfigure(0, weight=1)
        feature_btn_frame.grid_columnconfigure(1, weight=1)
        
        lock_btn = ttk.Button(feature_btn_frame, text="Lock Selected", command=self.lock_features)
        lock_btn.grid(row=0, column=0, padx=(0, 5))
        
        unlock_btn = ttk.Button(feature_btn_frame, text="Unlock All", command=self.unlock_features)
        unlock_btn.grid(row=0, column=1, padx=(5, 0))
        
        # Target feature selection
        target_label = ttk.Label(data_frame, text="Target Feature:", style="Bold.TLabel")
        target_label.grid(row=5, column=0, sticky="w", pady=(10, 5))
        
        self.target_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.target_combo.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        
        # Output directory
        output_label = ttk.Label(data_frame, text="Output Directory:", style="Bold.TLabel")
        output_label.grid(row=7, column=0, sticky="w", pady=(10, 5))
        
        output_entry_frame = ttk.Frame(data_frame)
        output_entry_frame.grid(row=8, column=0, sticky="ew", pady=(0, 10))
        output_entry_frame.grid_columnconfigure(0, weight=1)
        
        self.output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir_var)
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        output_browse_btn = ttk.Button(output_entry_frame, text="Browse", command=self.browse_output_dir)
        output_browse_btn.grid(row=0, column=1)
        
        # Training Parameters Tab
        training_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(training_frame, text="Training Parameters")
        training_frame.grid_columnconfigure(0, weight=1)
        
        # Hidden layer size
        hidden_label = ttk.Label(training_frame, text="Hidden Layer Size:", style="Bold.TLabel")
        hidden_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        hidden_entry = ttk.Entry(training_frame, textvariable=self.hidden_size_var, width=10)
        hidden_entry.grid(row=1, column=0, sticky="w", pady=(0, 10))
        
        # Learning rate
        lr_label = ttk.Label(training_frame, text="Learning Rate:", style="Bold.TLabel")
        lr_label.grid(row=2, column=0, sticky="w", pady=(0, 5))
        
        lr_entry = ttk.Entry(training_frame, textvariable=self.learning_rate_var, width=10)
        lr_entry.grid(row=3, column=0, sticky="w", pady=(0, 10))
        
        # Batch size
        batch_label = ttk.Label(training_frame, text="Batch Size:", style="Bold.TLabel")
        batch_label.grid(row=4, column=0, sticky="w", pady=(0, 5))
        
        batch_entry = ttk.Entry(training_frame, textvariable=self.batch_size_var, width=10)
        batch_entry.grid(row=5, column=0, sticky="w", pady=(0, 10))
        
        # Training button
        self.train_button = ttk.Button(training_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        
        # Model Management Tab
        model_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(model_frame, text="Model Management")
        model_frame.grid_columnconfigure(0, weight=1)
        
        # Model selection
        model_label = ttk.Label(model_frame, text="Select Model:", style="Bold.TLabel")
        model_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.model_combo = ttk.Combobox(model_frame, state="readonly")
        self.model_combo.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)
        
        # Model buttons
        model_btn_frame = ttk.Frame(model_frame)
        model_btn_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        model_btn_frame.grid_columnconfigure(0, weight=1)
        model_btn_frame.grid_columnconfigure(1, weight=1)
        
        refresh_btn = ttk.Button(model_btn_frame, text="Refresh Models", command=self.refresh_models)
        refresh_btn.grid(row=0, column=0, padx=(0, 5))
        
        predict_btn = ttk.Button(model_btn_frame, text="Make Prediction", command=self.make_prediction)
        predict_btn.grid(row=0, column=1, padx=(5, 0))
        
        # 3D Visualization parameters
        gd3d_label = ttk.Label(model_frame, text="3D Visualization Parameters:", style="Bold.TLabel")
        gd3d_label.grid(row=3, column=0, sticky="w", pady=(20, 5))
        
        # Color map
        color_label = ttk.Label(model_frame, text="Color Map:")
        color_label.grid(row=4, column=0, sticky="w", pady=(0, 5))
        
        color_entry = ttk.Entry(model_frame, textvariable=self.color_var, width=10)
        color_entry.grid(row=5, column=0, sticky="w", pady=(0, 10))
        
        # Point size
        point_label = ttk.Label(model_frame, text="Point Size:")
        point_label.grid(row=6, column=0, sticky="w", pady=(0, 5))
        
        point_entry = ttk.Entry(model_frame, textvariable=self.point_size_var, width=10)
        point_entry.grid(row=7, column=0, sticky="w", pady=(0, 10))
        
        # Line width
        line_label = ttk.Label(model_frame, text="Line Width:")
        line_label.grid(row=8, column=0, sticky="w", pady=(0, 5))
        
        line_entry = ttk.Entry(model_frame, textvariable=self.line_width_var, width=10)
        line_entry.grid(row=9, column=0, sticky="w", pady=(0, 10))
        
        # Surface alpha
        alpha_label = ttk.Label(model_frame, text="Surface Alpha:")
        alpha_label.grid(row=10, column=0, sticky="w", pady=(0, 5))
        
        alpha_entry = ttk.Entry(model_frame, textvariable=self.surface_alpha_var, width=10)
        alpha_entry.grid(row=11, column=0, sticky="w", pady=(0, 10))
        
        # 3D Visualization button
        gd3d_btn = ttk.Button(model_frame, text="Create 3D Visualization", command=self.create_3d_visualization)
        gd3d_btn.grid(row=12, column=0, sticky="ew", pady=(10, 0))
        
        # Status bar
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=0, column=0, sticky="ew")
        
        print("Control panel created with 3 tabs")

    def browse_data_file(self):
        """Browse for data file."""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_file = filename
            self.data_file_var.set(filename)
            self.load_data_features()
            self.status_var.set(f"Loaded data file: {os.path.basename(filename)}")

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_var.set(directory)
            self.status_var.set(f"Output directory: {directory}")

    def load_data_features(self):
        """Load features from the selected data file."""
        try:
            if not self.data_file or not os.path.exists(self.data_file):
                return
            
            # Load data to get column names
            df = pd.read_csv(self.data_file)
            self.feature_list = list(df.columns)
            
            # Update feature listbox
            self.feature_listbox.delete(0, tk.END)
            for feature in self.feature_list:
                self.feature_listbox.insert(tk.END, feature)
            
            # Update target combo
            self.target_combo['values'] = self.feature_list
            if 'close' in self.feature_list:
                self.target_combo.set('close')
            elif len(self.feature_list) > 0:
                self.target_combo.set(self.feature_list[0])
            
            self.status_var.set(f"Loaded {len(self.feature_list)} features from data file")
            
        except Exception as e:
            print(f"Error loading data features: {e}")
            self.status_var.set(f"Error loading features: {str(e)}")

    def lock_features(self):
        """Lock the currently selected features."""
        try:
            selected_indices = self.feature_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("No Selection", "Please select features to lock.")
                return
            
            self.locked_features = [self.feature_list[i] for i in selected_indices]
            self.features_locked = True
            self.feature_status_var.set(f"Locked {len(self.locked_features)} features")
            self.status_var.set(f"Locked features: {', '.join(self.locked_features)}")
            
        except Exception as e:
            print(f"Error locking features: {e}")
            self.status_var.set(f"Error locking features: {str(e)}")

    def unlock_features(self):
        """Unlock all features."""
        self.locked_features = []
        self.features_locked = False
        self.feature_status_var.set("Features unlocked")
        self.status_var.set("All features unlocked")

    def start_training(self):
        """Start the training process in a separate thread."""
        if self.is_training:
            messagebox.showwarning("Training in Progress", "Training is already in progress.")
            return
        
        if not self.data_file:
            messagebox.showerror("No Data File", "Please select a data file first.")
            return
        
        if not self.target_combo.get():
            messagebox.showerror("No Target", "Please select a target feature.")
            return
        
        try:
            self.is_training = True
            self.train_button.config(state=tk.DISABLED)
            self.status_var.set("Starting training...")
            
            # Start training in a separate thread
            self.training_thread = threading.Thread(target=self._train_model)
            self.training_thread.daemon = True
            self.training_thread.start()
            
        except Exception as e:
            print(f"Error starting training: {e}")
            self.is_training = False
            self.train_button.config(state=tk.NORMAL)
            self.status_var.set(f"Error starting training: {str(e)}")

    def _train_model(self):
        """Train the model (runs in separate thread)."""
        try:
            # Get training parameters
            hidden_size = int(self.hidden_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Get features
            if self.features_locked:
                x_features = self.locked_features
            else:
                selected_indices = self.feature_listbox.curselection()
                if not selected_indices:
                    x_features = self.feature_list[:5]  # Default to first 5 features
                else:
                    x_features = [self.feature_list[i] for i in selected_indices]
            
            y_feature = self.target_combo.get()
            
            # Run training using subprocess (stock_net.train_model doesn't exist)
            cmd = [
                sys.executable, 'stock_net.py',
                '--data_file', self.data_file,
                '--x_features', ','.join(x_features),
                '--y_feature', y_feature,
                '--hidden_size', str(hidden_size),
                '--learning_rate', str(learning_rate),
                '--batch_size', str(batch_size)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Training failed: {result.stderr}")
            
            # Training completed successfully
            self.root.after(0, self._training_completed_success)
            
        except Exception as e:
            print(f"Error in training: {e}")
            # Fix lambda scoping issue by capturing the exception variable
            error_msg = str(e)
            self.root.after(0, lambda: self._training_completed_error(error_msg))

    def _training_completed_success(self):
        """Handle successful training completion."""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set("Training completed successfully!")
        self.refresh_models()
        messagebox.showinfo("Success", "Model training completed successfully!")

    def _training_completed_error(self, error_msg):
        """Handle training completion with error."""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set(f"Training failed: {error_msg}")
        messagebox.showerror("Training Error", f"Training failed: {error_msg}")

    def make_prediction(self):
        """Make a prediction using the selected model."""
        if not self.selected_model_path:
            messagebox.showerror("No Model", "Please select a model first.")
            return
        
        if not self.data_file:
            messagebox.showerror("No Data", "Please select a data file for prediction.")
            return
        
        try:
            self.status_var.set("Making prediction...")
            
            # Run prediction using predict.py
            cmd = [
                sys.executable, 'predict.py',
                self.data_file,  # Input file as positional argument
                '--model_dir', self.selected_model_path,
                '--output_dir', self.output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Prediction failed: {result.stderr}")
            
            self.status_var.set("Prediction completed successfully!")
            messagebox.showinfo("Success", "Prediction completed successfully!")
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            self.status_var.set(f"Prediction failed: {str(e)}")
            messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")

    def create_3d_visualization(self):
        """Create 3D gradient descent visualization."""
        if not self.selected_model_path:
            messagebox.showerror("No Model", "Please select a model first.")
            return
        
        try:
            self.status_var.set("Creating 3D visualization...")
            
            # Get visualization parameters
            color = self.color_var.get()
            point_size = int(self.point_size_var.get())
            line_width = int(self.line_width_var.get())
            surface_alpha = float(self.surface_alpha_var.get())
            
            # Run 3D visualization using gradient_descent_3d.py
            cmd = [
                sys.executable, 'gradient_descent_3d.py',
                '--model_dir', self.selected_model_path,
                '--color', color,
                '--point_size', str(point_size),
                '--line_width', str(line_width),
                '--surface_alpha', str(surface_alpha),
                '--save_png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"3D visualization failed: {result.stderr}")
            
            self.status_var.set("3D visualization created successfully!")
            self.refresh_models()  # Refresh to show new plots
            messagebox.showinfo("Success", "3D visualization created successfully!")
            
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
            self.status_var.set(f"3D visualization failed: {str(e)}")
            messagebox.showerror("3D Visualization Error", f"3D visualization failed: {str(e)}")

    def _load_model_info(self):
        """Load information about the selected model."""
        try:
            if not self.selected_model_path:
                return
            
            # Load model configuration if available
            config_file = os.path.join(self.selected_model_path, 'model_config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"Model config: {config}")
            
            # Load training losses if available
            loss_file = os.path.join(self.selected_model_path, 'training_losses.csv')
            if os.path.exists(loss_file):
                losses = np.loadtxt(loss_file, delimiter=',')
                print(f"Training losses loaded: {len(losses)} epochs")
            
        except Exception as e:
            print(f"Error loading model info: {e}")

    def display_training_plots(self):
        """Display training plots for the selected model."""
        try:
            if not self.selected_model_path:
                return
            
            # Load training losses
            loss_file = os.path.join(self.selected_model_path, 'training_losses.csv')
            if os.path.exists(loss_file):
                losses = np.loadtxt(loss_file, delimiter=',')
                if losses.ndim > 1:
                    train_losses = losses[:, 0]
                    val_losses = losses[:, 1] if losses.shape[1] > 1 else None
                else:
                    train_losses = losses
                    val_losses = None
                
                # Update plots
                self.update_training_results(train_losses)
                self.update_plots_tab(train_losses, val_losses)
                
            self.load_saved_plots()
            
        except Exception as e:
            print(f"Error displaying training plots: {e}")

    def load_saved_plots(self):
        """Load and display PNG plots from the selected model's plots directory."""
        try:
            # Clear existing images
            for widget in self.saved_plots_inner_frame.winfo_children():
                widget.destroy()
            
            if not self.selected_model_path:
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="Select a model to view saved plots", 
                                                        foreground=TEXT_COLOR, background=FRAME_COLOR)
                self.saved_plots_placeholder.grid(row=0, column=0, pady=20)
                return
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if not os.path.exists(plots_dir):
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="No plots found in model directory", 
                                                        foreground=TEXT_COLOR, background=FRAME_COLOR)
                self.saved_plots_placeholder.grid(row=0, column=0, pady=20)
                self.status_var.set("No plots directory found")
                return
            
            plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
            if not plot_files:
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="No PNG plots found in model directory", 
                                                        foreground=TEXT_COLOR, background=FRAME_COLOR)
                self.saved_plots_placeholder.grid(row=0, column=0, pady=20)
                self.status_var.set("No PNG plots found")
                return
            
            # Limit to 10 plots for better performance
            plot_files = plot_files[:10]  # Limit to 10 plots
            total_plots = len(glob.glob(os.path.join(plots_dir, '*.png')))
            show_limit_message = len(plot_files) < total_plots
            
            # Load and display each PNG
            self.saved_plots_images = []  # Keep references to avoid garbage collection
            max_width = 400  # Smaller images for faster loading
            for i, plot_file in enumerate(plot_files):
                # Load image
                img = Image.open(plot_file)
                # Resize image to fit
                img_width, img_height = img.size
                scale = min(max_width / img_width, 1.0)  # Scale down if too wide
                new_size = (int(img_width * scale), int(img_height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.saved_plots_images.append(photo)
                
                # Create label with image and filename using grid
                frame = ttk.Frame(self.saved_plots_inner_frame)
                frame.grid(row=i*2, column=0, pady=5, padx=10, sticky="ew")
                frame.grid_columnconfigure(0, weight=1)
                
                ttk.Label(frame, text=os.path.basename(plot_file), 
                          foreground=TEXT_COLOR, background=FRAME_COLOR).grid(row=0, column=0, sticky="w")
                ttk.Label(frame, image=photo).grid(row=1, column=0, sticky="w")
            
            # Show limit message if there are more plots available
            if show_limit_message:
                limit_label = ttk.Label(self.saved_plots_inner_frame, 
                                       text=f"Showing first 10 plots (more available - {total_plots} total)", 
                                       foreground=TEXT_COLOR, background=FRAME_COLOR)
                limit_label.grid(row=len(plot_files)*2, column=0, pady=10)
            
            self.status_var.set(f"Loaded {len(plot_files)} plot(s) from {plots_dir}")
            print(f"Loaded {len(plot_files)} plot(s): {plot_files}")
            
            # Update scroll region
            self.saved_plots_canvas.configure(scrollregion=self.saved_plots_canvas.bbox("all"))
            
        except Exception as e:
            self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                    text=f"Error loading plots: {str(e)}", 
                                                    foreground="red", background=FRAME_COLOR)
            self.saved_plots_placeholder.grid(row=0, column=0, pady=20)
            self.status_var.set(f"Error loading plots: {str(e)}")
            print(f"Error loading plots: {e}")

    def create_3d_gradient_descent_visualization(self, model_dir_or_losses):
        """Create 3D visualization of gradient descent training path."""
        try:
            # Check if model_dir_or_losses is a directory path or training losses
            if isinstance(model_dir_or_losses, str) and os.path.isdir(model_dir_or_losses):
                model_dir = model_dir_or_losses
                # Load training losses from file
                loss_file = os.path.join(model_dir, 'training_losses.csv')
                if os.path.exists(loss_file):
                    train_losses = np.loadtxt(loss_file, delimiter=',')
                else:
                    print(f"Training losses file not found: {loss_file}")
                    return
            else:
                # Assume it's training losses array
                train_losses = model_dir_or_losses
                model_dir = self.selected_model_path
            
            if not model_dir:
                print("No model directory available for 3D visualization")
                return
            
            # Load weight history files
            weights_files = sorted(glob.glob(os.path.join(model_dir, 'weights_history_*.csv')))
            if weights_files:
                weights = [np.loadtxt(f, delimiter=',') for f in weights_files]
                w1 = [w[0] for w in weights]  # Example: first weight
                w2 = [w[1] for w in weights]  # Example: second weight
                z = train_losses[:len(w1)]    # Align with losses
                
                # Clear the 3D plot
                self.gd3d_ax.clear()
                
                # Plot the training path
                self.gd3d_ax.plot(w1, w2, z, 'r-', linewidth=3, label='Training Path')
                
                # Add scatter points for key positions
                self.gd3d_ax.scatter(w1[0], w2[0], z[0], c='green', s=100, label='Start', marker='o')
                self.gd3d_ax.scatter(w1[-1], w2[-1], z[-1], c='red', s=100, label='End', marker='s')
                
                # Set labels and title
                self.gd3d_ax.set_xlabel('Weight 1')
                self.gd3d_ax.set_ylabel('Weight 2')
                self.gd3d_ax.set_zlabel('Training Loss')
                self.gd3d_ax.set_title('3D Gradient Descent Training Path')
                
                # Add legend
                self.gd3d_ax.legend()
                
                # Refresh the canvas
                self.gd3d_canvas.draw()
                
                print(f"Created 3D gradient descent visualization with {len(weights)} weight points")
            else:
                print(f"No weight history files found in {model_dir}")
                
        except Exception as e:
            print(f"Error creating 3D gradient descent visualization: {e}")
            import traceback
            traceback.print_exc()

    def create_display_panel(self):
        """Create the right display panel for plots and results (now using grid)"""
        display_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        display_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        
        # Create notebook for different displays
        self.display_notebook = ttk.Notebook(display_frame)
        self.display_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Training Results Tab
        train_results_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(train_results_frame, text="Training Results")
        train_results_frame.grid_columnconfigure(0, weight=1)
        train_results_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for training results
        self.results_fig = plt.Figure(figsize=(8, 6))
        self.results_ax = self.results_fig.add_subplot(111)
        
        # Create canvas and embed in the tab using grid
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, train_results_frame)
        self.results_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.results_canvas, train_results_frame)
        
        # Initialize with a placeholder plot
        self.results_ax.text(0.5, 0.5, 'Training results will appear here', 
                            ha='center', va='center', transform=self.results_ax.transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.results_ax.set_title("Training Results")
        self.results_canvas.draw()
        
        print("Training Results tab created successfully")

        # Prediction Results Tab
        pred_results_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(pred_results_frame, text="Prediction Results")
        pred_results_frame.grid_columnconfigure(0, weight=1)
        pred_results_frame.grid_rowconfigure(0, weight=1)
        
        # Create figure for prediction plot
        self.pred_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=pred_results_frame)
        self.pred_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Gradient Descent Tab
        gd_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(gd_frame, text="Live Training")
        gd_frame.grid_columnconfigure(0, weight=1)
        gd_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for gradient descent visualization
        self.gd_fig = plt.Figure(figsize=(8, 6))
        self.gd_ax = self.gd_fig.add_subplot(111)
        
        # Create canvas and embed in the tab using grid
        self.gd_canvas = FigureCanvasTkAgg(self.gd_fig, gd_frame)
        self.gd_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.gd_canvas, gd_frame)
        
        # Initialize with a placeholder plot
        self.gd_ax.text(0.5, 0.5, 'Training visualization will appear here', 
                       ha='center', va='center', transform=self.gd_ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.gd_ax.set_title("Live Training Progress")
        self.gd_canvas.draw()
        
        print("Gradient Descent tab created successfully")

        # Plots Tab
        plots_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(plots_frame, text="Plots")
        plots_frame.grid_columnconfigure(0, weight=1)
        plots_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for plots
        self.plots_fig = plt.Figure(figsize=(8, 6))
        self.plots_ax = self.plots_fig.add_subplot(111)
        
        # Create canvas and embed in the tab using grid
        self.plots_canvas = FigureCanvasTkAgg(self.plots_fig, plots_frame)
        self.plots_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.plots_canvas, plots_frame)
        
        # Initialize with a placeholder plot
        self.plots_ax.text(0.5, 0.5, 'Model plots will appear here', 
                          ha='center', va='center', transform=self.plots_ax.transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.plots_ax.set_title("Model Plots")
        self.plots_canvas.draw()
        
        # 3D Gradient Descent Tab
        gd3d_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(gd3d_frame, text="3D Gradient Descent")
        gd3d_frame.grid_columnconfigure(0, weight=1)
        gd3d_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for 3D gradient descent visualization
        self.gd3d_fig = plt.Figure(figsize=(8, 6))
        self.gd3d_ax = self.gd3d_fig.add_subplot(111, projection='3d')
        
        # Create canvas and embed in the tab using grid
        self.gd3d_canvas = FigureCanvasTkAgg(self.gd3d_fig, gd3d_frame)
        self.gd3d_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.gd3d_canvas, gd3d_frame)
        
        # Initialize with a placeholder plot
        self.gd3d_ax.text(0.5, 0.5, 0.5, '3D Gradient Descent visualization will appear here after training', 
                          ha='center', va='center', transform=self.gd3d_ax.transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.gd3d_ax.set_title("3D Gradient Descent Visualization")
        self.gd3d_canvas.draw()
        
        print("3D Gradient Descent tab created successfully")
        
        # Saved Plots Tab (New)
        saved_plots_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(saved_plots_frame, text="Saved Plots")
        saved_plots_frame.grid_columnconfigure(0, weight=1)
        saved_plots_frame.grid_rowconfigure(0, weight=1)
        
        # Create scrollable canvas for images using grid
        self.saved_plots_canvas = tk.Canvas(saved_plots_frame, bg=FRAME_COLOR)
        self.saved_plots_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar using grid
        scrollbar = ttk.Scrollbar(saved_plots_frame, orient="vertical", command=self.saved_plots_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.saved_plots_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas to hold images
        self.saved_plots_inner_frame = ttk.Frame(self.saved_plots_canvas)
        self.saved_plots_canvas.create_window((0, 0), window=self.saved_plots_inner_frame, anchor="nw")
        
        # Placeholder label using grid
        self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                text="Select a model to view saved plots", 
                                                foreground=TEXT_COLOR, background=FRAME_COLOR)
        self.saved_plots_placeholder.grid(row=0, column=0, pady=20)
        
        # Bind canvas resizing
        self.saved_plots_inner_frame.bind("<Configure>", lambda e: self.saved_plots_canvas.configure(
            scrollregion=self.saved_plots_canvas.bbox("all")))
        
        print("Saved Plots tab created successfully")
        
        print(f"Display panel created with {self.display_notebook.index('end')} tabs")

    def on_model_select(self, event):
        """Handle model selection from combobox"""
        try:
            model_name = self.model_combo.get()
            if model_name:
                # Find the actual model directory using path utilities
                model_dir = path_utils.find_model_directory(model_name)
                if model_dir:
                    self.selected_model_path = model_dir
                else:
                    # Fallback to current directory
                    self.selected_model_path = os.path.join(self.current_model_dir, model_name)
                
                # Load model info and plots
                self._load_model_info()
                self.load_saved_plots()
                
                # Display training plots
                self.display_training_plots()
                
                # Check for 3D visualization
                if self.has_3d_visualization():
                    self.update_3d_gradient_descent_tab()
                
                # Try to create 3D gradient descent visualization if weight history files exist
                try:
                    weights_files = glob.glob(os.path.join(self.selected_model_path, 'weights_history_*.csv'))
                    loss_file = os.path.join(self.selected_model_path, 'training_losses.csv')
                    if weights_files and os.path.exists(loss_file):
                        self.create_3d_gradient_descent_visualization(self.selected_model_path)
                        print(f"Created 3D gradient descent visualization for {model_name}")
                except Exception as e:
                    print(f"Could not create 3D visualization for {model_name}: {e}")
                
                self.status_var.set(f"Selected model: {model_name}")
                print(f"Selected model: {model_name} at {self.selected_model_path}")
            else:
                self.selected_model_path = None
                self.status_var.set("No model selected")
        except Exception as e:
            print(f"Error in model selection: {e}")
        
    def refresh_models(self):
        """Refresh the list of available models and update plots."""
        try:
            # Clear existing models from combobox
            self.model_combo['values'] = ()
            self.selected_model_path = None
            
            if not os.path.exists(self.current_model_dir):
                os.makedirs(self.current_model_dir, exist_ok=True)
                self.status_var.set(f"Created models directory: {self.current_model_dir}")
                self.load_saved_plots()
                return
            
            model_dirs = sorted([
                d for d in os.listdir(self.current_model_dir)
                if os.path.isdir(os.path.join(self.current_model_dir, d)) and d.startswith('model_')
            ], reverse=True)
            
            # Update combobox with model directories
            self.model_combo['values'] = model_dirs
            
            if model_dirs:
                self.model_combo.set(model_dirs[0])
                self.on_model_select(None)  # Trigger plot loading
            else:
                self.status_var.set("No models available")
                self.load_saved_plots()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh model list: {str(e)}")
            self.status_var.set(f"Error refreshing models: {str(e)}")
    
    def _training_completed(self, model_dir, train_losses, val_losses):
        """Handle training completion and save plots."""
        try:
            self.status_var.set("Training completed successfully!")
            self.train_button.config(state=tk.NORMAL)
            
            # Save loss plot
            plots_dir = os.path.join(model_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            fig = plt.Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            epochs = list(range(1, len(train_losses) + 1))
            ax.plot(epochs, train_losses, 'b-', label='Training Loss')
            if len(val_losses) == len(train_losses):
                ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
            ax.set_title(f"Training Progress ({self.get_ticker_from_filename()})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend()
            ax.grid(True)
            fig.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Refresh models and plots
            self.refresh_models()
            self.update_training_results(train_losses)
            self.update_plots_tab(train_losses, val_losses)
            self.create_3d_gradient_descent_visualization(train_losses)
            self.switch_to_tab(4)  # 3D Gradient Descent tab
            
            messagebox.showinfo("Success", f"Model training completed successfully!\nModel saved in: {model_dir}\n3D gradient descent visualization created!")
            self.selected_model_path = model_dir
            
        except Exception as e:
            print(f"Error in training completion: {e}")
            messagebox.showerror("Error", f"Error updating results: {str(e)}")
    
    def get_ticker_from_filename(self):
        """Extract ticker symbol from data filename."""
        if self.data_file:
            filename = os.path.basename(self.data_file)
            # Extract ticker from filename (e.g., 'tsla_combined.csv' -> 'TSLA')
            ticker = filename.split('_')[0].upper()
            return ticker
        return "STOCK"
    
    def update_training_results(self, train_losses):
        """Update the training results tab with loss plot."""
        try:
            self.results_ax.clear()
            epochs = list(range(1, len(train_losses) + 1))
            self.results_ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
            self.results_ax.set_title("Training Loss Over Time")
            self.results_ax.set_xlabel("Epoch")
            self.results_ax.set_ylabel("Loss")
            self.results_ax.legend()
            self.results_ax.grid(True)
            self.results_canvas.draw()
        except Exception as e:
            print(f"Error updating training results: {e}")
    
    def update_plots_tab(self, train_losses, val_losses):
        """Update the plots tab with training results and saved PNG plots."""
        try:
            self.plots_ax.clear()
            ticker = self.get_ticker_from_filename()
            epochs = list(range(1, len(train_losses) + 1))
            
            # Plot training and validation losses
            self.plots_ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if val_losses is not None and len(val_losses) == len(train_losses):
                self.plots_ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            self.plots_ax.set_title(f"Training Progress Overview ({ticker})")
            self.plots_ax.set_xlabel("Epoch")
            self.plots_ax.set_ylabel("MSE Loss")
            self.plots_ax.legend()
            self.plots_ax.grid(True, alpha=0.3)
            
            self.plots_canvas.draw()
            
            # Check for PNG plots and display them in the Saved Plots tab
            plot_files = []
            if self.selected_model_path:
                plots_dir = os.path.join(self.selected_model_path, 'plots')
                if os.path.exists(plots_dir):
                    plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
            
            # Update the Saved Plots tab with the new plots
            if plot_files:
                # Trigger refresh of saved plots tab
                self.load_saved_plots()
                print(f"Plots tab updated with {len(train_losses)} epochs and {len(plot_files)} PNG plots available")
            else:
                print(f"Plots tab updated with {len(train_losses)} epochs (no PNG plots found)")
                
        except Exception as e:
            print(f"Error updating plots tab: {e}")
    
    def switch_to_tab(self, tab_index):
        """Switch to a specific tab in the display notebook."""
        try:
            if 0 <= tab_index < self.display_notebook.index('end'):
                self.display_notebook.select(tab_index)
        except Exception as e:
            print(f"Error switching to tab {tab_index}: {e}")
    
    def has_3d_visualization(self):
        """Check if the selected model has 3D visualization files."""
        if not self.selected_model_path:
            return False
        plots_dir = os.path.join(self.selected_model_path, 'plots')
        if not os.path.exists(plots_dir):
            return False
        # Check for 3D gradient descent frame files
        gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
        return len(gd3d_files) > 0
    
    def update_3d_gradient_descent_tab(self):
        """Update the 3D gradient descent tab with visualization."""
        try:
            if not self.has_3d_visualization():
                return
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
            
            if gd3d_files:
                # Load the last frame (most complete visualization)
                latest_frame = gd3d_files[-1]
                self.load_3d_visualization_image(latest_frame)
        except Exception as e:
            print(f"Error updating 3D gradient descent tab: {e}")
    
    def load_3d_visualization_image(self, image_path):
        """Load and display 3D visualization image in the 3D tab."""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Resize image to fit the canvas (75% of original size)
            img_width, img_height = img.size
            scale = 0.75
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Clear the 3D axis and display the image
            self.gd3d_ax.clear()
            self.gd3d_ax.imshow(np.array(img))
            self.gd3d_ax.set_title("3D Gradient Descent Visualization")
            self.gd3d_ax.axis('off')
            
            # Refresh the canvas
            self.gd3d_canvas.draw()
            
            print(f"Loaded 3D visualization image: {os.path.basename(image_path)} (resized to {new_size[0]}x{new_size[1]} - {scale*100}% of original)")
            
        except Exception as e:
            print(f"Error loading 3D visualization image: {e}")

    def display_plotly_loss(self, train_losses=None, val_losses=None):
        """Display interactive loss plot using plotly if available."""
        if not PLOTLY_AVAILABLE:
            messagebox.showwarning("Plotly Not Available", 
                                 "Plotly is not installed. Please install it with 'pip install plotly' to use interactive plots.")
            return
        
        try:
            # Use provided losses or get from current model
            if train_losses is None and self.selected_model_path:
                loss_file = os.path.join(self.selected_model_path, 'training_losses.csv')
                if os.path.exists(loss_file):
                    losses = np.loadtxt(loss_file, delimiter=',')
                    if losses.ndim > 1:
                        train_losses = losses[:, 0]  # Training losses
                        val_losses = losses[:, 1] if losses.shape[1] > 1 else None
                    else:
                        train_losses = losses
                        val_losses = None
            
            if train_losses is None:
                messagebox.showwarning("No Data", "No training loss data available to plot.")
                return
            
            # Create plotly figure
            fig = go.Figure()
            
            epochs = list(range(1, len(train_losses) + 1))
            
            # Add training loss trace
            fig.add_trace(go.Scatter(
                x=epochs,
                y=train_losses,
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add validation loss trace if available
            if val_losses is not None and len(val_losses) == len(train_losses):
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=val_losses,
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
            
            # Update layout
            ticker = self.get_ticker_from_filename()
            fig.update_layout(
                title=f'Interactive Training Loss ({ticker})',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                hovermode='x unified',
                showlegend=True,
                template='plotly_white'
            )
            
            # Show the plot
            fig.show()
            
        except Exception as e:
            print(f"Error creating plotly plot: {e}")
            messagebox.showerror("Error", f"Failed to create interactive plot: {str(e)}")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()
        