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
from tkinter import ttk, filedialog, messagebox, scrolledtext
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
import matplotlib.animation as animation
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import io

# Import the data converter
try:
    from data_converter import convert_data_file, detect_data_format
    DATA_CONVERTER_AVAILABLE = True
except ImportError:
    DATA_CONVERTER_AVAILABLE = False
    print("⚠️  Data converter not available - manual data conversion may be required")


# Custom NavigationToolbar that uses grid instead of pack
class GridMatplotlibToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        # Create a frame to hold the toolbar
        self.toolbar_frame = tk.Frame(window)
        super().__init__(canvas, self.toolbar_frame)
        
        # Do not automatically place the toolbar - let the calling code control placement
        # The calling code should call: toolbar.grid(row=X, column=0, sticky="ew")
        
        # Configure the parent window to handle the toolbar frame properly
        window.grid_columnconfigure(0, weight=1)  # Column should expand
        
        # Ensure the toolbar frame itself uses grid properly
        self.toolbar_frame.grid_columnconfigure(0, weight=1)
        
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
        self.root.title("Stock Prediction Neural Network GUI")
        self.root.geometry("1400x900")
        
        # Set up the main window
        self.setup_main_window()
        
        # Initialize variables
        self.data_file = None
        self.output_dir = None
        self.selected_model_path = None
        self.training_process = None
        self.visualization_process = None
        
        # Training variables
        self.epochs_var = tk.StringVar(value="100")
        self.learning_rate_var = tk.StringVar(value="0.001")
        self.batch_size_var = tk.StringVar(value="32")
        self.validation_split_var = tk.StringVar(value="0.2")
        
        # Feature selection variables
        self.feature_vars = {}
        self.locked_features = set()
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")  # Status bar variable
        self.feature_status_var = tk.StringVar(value="")  # Feature status variable
        self.prediction_status_var = tk.StringVar(value="No prediction data")  # Prediction status variable
        
        # Selected prediction file
        self.selected_prediction_file = None
        
        # Live plot variables with rate limiting
        self.live_plot_fig = None
        self.live_plot_epochs = []
        self.live_plot_losses = []
        self.live_plot_window_open = False
        self.last_plot_update = 0  # Timestamp of last plot update
        self.plot_update_interval = 0.1  # Minimum seconds between updates
        
        # 3D visualization variables
        self.gd3d_fig = None
        self.gd3d_ax = None
        self.gd3d_canvas = None
        
        # 3D animation variables
        self.animation_running = False
        self.current_frame = 0
        self.total_frames = 0
        self.anim_speed_var = tk.DoubleVar(value=1.0)
        
        # Model directory
        self.current_model_dir = "."
        
        # Image cache for saved plots
        self.image_cache = {}
        self.max_cache_size = 10
        
        # Thread pool for background tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Test script availability
        self.test_script_availability()
        
        # Create GUI components
        self.create_control_panel()
        self.create_display_panel()
        
        # Load initial data
        self.refresh_models(load_plots=False)
        
        # Cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle cleanup when the GUI window is closed."""
        try:
            # Stop any ongoing training
            if self.is_training and self.training_thread and self.training_thread.is_alive():
                print("Stopping training thread...")
                self.is_training = False
                self.training_thread.join(timeout=2)
            
            # Close live training plot if open
            if hasattr(self, 'live_plot_window_open') and self.live_plot_window_open:
                self.close_live_plot()
            
            # Clean up thread pool
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            
            # Cancel any pending plot loading operations
            if hasattr(self, 'plot_futures'):
                for future in self.plot_futures.values():
                    if not future.done():
                        future.cancel()
            
            print("Thread pool cleaned up")
            
            # Destroy the root window
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Force destroy even if cleanup fails
            try:
                self.root.destroy()
            except:
                pass

    def setup_main_window(self):
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
        self.color_var = tk.StringVar(value="viridis")
        self.point_size_var = tk.StringVar(value="8")
        self.line_width_var = tk.StringVar(value="3")
        self.surface_alpha_var = tk.StringVar(value="0.6")
        
        # Additional 3D visualization parameters
        self.w1_range_min_var = tk.StringVar(value="-2.0")
        self.w1_range_max_var = tk.StringVar(value="2.0")
        self.w2_range_min_var = tk.StringVar(value="-2.0")
        self.w2_range_max_var = tk.StringVar(value="2.0")
        self.n_points_var = tk.StringVar(value="30")
        self.view_elev_var = tk.StringVar(value="30.0")
        self.view_azim_var = tk.StringVar(value="45.0")
        self.fps_var = tk.StringVar(value="30")
        self.w1_index_var = tk.StringVar(value="0")
        self.w2_index_var = tk.StringVar(value="0")
        self.output_width_var = tk.StringVar(value="1200")
        self.output_height_var = tk.StringVar(value="800")
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")  # Status bar variable
        self.feature_status_var = tk.StringVar(value="")  # Feature status variable
        self.prediction_status_var = tk.StringVar(value="No prediction data")  # Prediction status variable
        
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
        self.refresh_models(load_plots=False)

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
        
        # Live plot button
        live_plot_btn = ttk.Button(training_frame, text="Open Live Training Plot (Matplotlib)", command=self.open_live_training_plot)
        live_plot_btn.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        
        # Cache clear button
        self.cache_clear_btn = ttk.Button(training_frame, text="Clear Cache", command=self.clear_cache_only)
        self.cache_clear_btn.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        
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
        
        # Prediction Files Section
        pred_files_label = ttk.Label(model_frame, text="Prediction Files:", style="Bold.TLabel")
        pred_files_label.grid(row=3, column=0, sticky="w", pady=(20, 5))
        
        # Create frame for prediction files listbox and scrollbar
        pred_files_frame = ttk.Frame(model_frame)
        pred_files_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        pred_files_frame.grid_columnconfigure(0, weight=1)
        pred_files_frame.grid_rowconfigure(0, weight=1)
        
        # Prediction files listbox
        self.prediction_files_listbox = tk.Listbox(pred_files_frame, height=6, selectmode=tk.SINGLE)
        self.prediction_files_listbox.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.prediction_files_listbox.bind('<<ListboxSelect>>', self.on_prediction_file_select)
        
        # Scrollbar for prediction files
        pred_files_scrollbar = ttk.Scrollbar(pred_files_frame, orient="vertical", command=self.prediction_files_listbox.yview)
        pred_files_scrollbar.grid(row=0, column=1, sticky="ns")
        self.prediction_files_listbox.configure(yscrollcommand=pred_files_scrollbar.set)
        
        # Prediction files buttons
        pred_files_btn_frame = ttk.Frame(model_frame)
        pred_files_btn_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        pred_files_btn_frame.grid_columnconfigure(0, weight=1)
        pred_files_btn_frame.grid_columnconfigure(1, weight=1)
        
        refresh_pred_btn = ttk.Button(pred_files_btn_frame, text="Refresh Prediction Files", command=self.refresh_prediction_files)
        refresh_pred_btn.grid(row=0, column=0, padx=(0, 5))
        
        view_pred_btn = ttk.Button(pred_files_btn_frame, text="View Prediction Results", command=self.view_prediction_results)
        view_pred_btn.grid(row=0, column=1, padx=(5, 0))
        
        # 3D Visualization button
        gd3d_btn = ttk.Button(model_frame, text="Create 3D Visualization", command=self.create_3d_visualization)
        gd3d_btn.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        
        # Plot Controls Tab
        plot_controls_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(plot_controls_frame, text="Plot Controls")
        plot_controls_frame.grid_columnconfigure(0, weight=1)
        plot_controls_frame.grid_rowconfigure(0, weight=1)
        
        # Help Tab
        help_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(help_frame, text="?")
        help_frame.grid_columnconfigure(0, weight=1)
        help_frame.grid_rowconfigure(0, weight=1)
        
        # Create help content
        help_content_frame = ttk.Frame(help_frame)
        help_content_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        help_content_frame.grid_columnconfigure(0, weight=1)
        help_content_frame.grid_rowconfigure(1, weight=1)
        
        # Help title
        help_title = ttk.Label(help_content_frame, text="Stock Prediction GUI - User Manual", 
                              style="Bold.TLabel", font=("Arial", 14, "bold"))
        help_title.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Create scrollable text widget for help content
        help_text_frame = ttk.Frame(help_content_frame)
        help_text_frame.grid(row=1, column=0, sticky="nsew")
        help_text_frame.grid_columnconfigure(0, weight=1)
        help_text_frame.grid_rowconfigure(0, weight=1)
        
        # Create text widget with better configuration
        help_text = tk.Text(help_text_frame, wrap=tk.WORD, width=50, height=20, 
                           font=("Arial", 10), bg="white", relief=tk.SUNKEN,
                           padx=10, pady=10, state=tk.NORMAL)
        help_scrollbar = ttk.Scrollbar(help_text_frame, orient="vertical", command=help_text.yview)
        help_text.configure(yscrollcommand=help_scrollbar.set)
        
        help_text.grid(row=0, column=0, sticky="nsew")
        help_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Insert help content immediately
        help_content = self.get_help_content()
        if help_content.strip():
            help_text.insert(tk.END, help_content)
            print(f"✅ Help tab: Inserted {len(help_content)} characters of help content")
        else:
            # Fallback content if help content is empty
            fallback_content = """
STOCK PREDICTION GUI - QUICK HELP
================================

OVERVIEW
--------
This GUI provides neural network-based stock price prediction with advanced visualization.

QUICK START
-----------
1. Select a data file in the Data Selection tab
2. Choose features and target variable
3. Configure training parameters
4. Click "Start Training"
5. Monitor progress in Training Results tab
6. Use Model Management for predictions

TABS GUIDE
----------
• Data Selection: Load and configure data
• Training Parameters: Set model parameters
• Model Management: Load models and make predictions
• Plot Controls: Configure 3D visualizations
• Help (?): This help section

For detailed information, use the buttons below.
"""
            help_text.insert(tk.END, fallback_content)
            print("✅ Help tab: Inserted fallback help content")
        
        # Make read-only after inserting content
        help_text.config(state=tk.DISABLED)
        
        # Force update to ensure content is displayed
        help_text.update()
        
        # Add buttons at the bottom
        help_buttons_frame = ttk.Frame(help_content_frame)
        help_buttons_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        help_buttons_frame.grid_columnconfigure(0, weight=1)
        help_buttons_frame.grid_columnconfigure(1, weight=1)
        
        print_manual_btn = ttk.Button(help_buttons_frame, text="Print Full Manual", 
                                     command=self.print_full_manual)
        print_manual_btn.grid(row=0, column=0, padx=(0, 5))
        
        open_manual_file_btn = ttk.Button(help_buttons_frame, text="Open Manual File", 
                                         command=self.open_manual_file)
        open_manual_file_btn.grid(row=0, column=1, padx=(5, 0))
        
        # Grid the canvas and scrollbar
        plot_canvas = tk.Canvas(plot_controls_frame, bg='white')
        plot_scrollbar = ttk.Scrollbar(plot_controls_frame, orient="vertical", command=plot_canvas.yview)
        plot_scrollable_frame = ttk.Frame(plot_canvas)
        
        plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
        )
        
        plot_canvas.create_window((0, 0), window=plot_scrollable_frame, anchor="nw")
        plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
        
        # Grid the canvas and scrollbar
        plot_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        plot_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure canvas scrolling with mouse wheel
        def _on_mousewheel(event):
            plot_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        plot_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 3D Visualization parameters
        gd3d_label = ttk.Label(plot_scrollable_frame, text="3D Visualization Parameters:", style="Bold.TLabel")
        gd3d_label.grid(row=0, column=0, sticky="w", pady=(10, 15))
        
        # Create a LabelFrame for 3D parameters with better organization
        gd3d_inner_frame = ttk.LabelFrame(plot_scrollable_frame, text="Basic 3D Settings", padding="10")
        gd3d_inner_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        gd3d_inner_frame.grid_columnconfigure(0, weight=1)
        gd3d_inner_frame.grid_columnconfigure(1, weight=1)
        
        # Color map - Row 0
        color_label = ttk.Label(gd3d_inner_frame, text="Color Map:")
        color_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Use combobox with valid colormap options
        valid_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'jet', 'hot', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter']
        self.color_combo = ttk.Combobox(gd3d_inner_frame, textvariable=self.color_var, values=valid_colormaps, width=20, state="readonly")
        self.color_combo.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Ensure the current value is displayed properly
        try:
            current_color = self.color_var.get()
            print(f"🔍 Debug: color_var.get() = '{current_color}'")
            
            if current_color and current_color in valid_colormaps:
                self.color_combo.set(current_color)
                print(f"✅ Set combobox to existing value: '{current_color}'")
            else:
                # Set default value
                default_color = 'viridis'
                self.color_combo.set(default_color)
                self.color_var.set(default_color)
                print(f"✅ Set combobox to default value: '{default_color}'")
            
            # Force update the combobox display
            self.color_combo.update()
            
        except Exception as e:
            print(f"❌ Error setting color combobox: {e}")
            # Fallback: set to viridis
            self.color_combo.set('viridis')
            self.color_var.set('viridis')
        
        print(f"✅ Color map combobox created with {len(valid_colormaps)} values")
        print(f"✅ Current color value: '{self.color_var.get()}'")
        print(f"✅ Combobox display value: '{self.color_combo.get()}'")
        print(f"✅ Combobox values: {valid_colormaps[:5]}...")  # Show first 5 values
        
        # Point size and Line width - Row 1
        point_label = ttk.Label(gd3d_inner_frame, text="Point Size:")
        point_label.grid(row=1, column=0, sticky="w", pady=(10, 5))
        
        point_entry = ttk.Entry(gd3d_inner_frame, textvariable=self.point_size_var, width=12)
        point_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 5))
        
        # Line width - Row 2
        line_label = ttk.Label(gd3d_inner_frame, text="Line Width:")
        line_label.grid(row=2, column=0, sticky="w", pady=(10, 5))
        
        line_entry = ttk.Entry(gd3d_inner_frame, textvariable=self.line_width_var, width=12)
        line_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(10, 5))
        
        # Surface alpha - Row 3
        alpha_label = ttk.Label(gd3d_inner_frame, text="Surface Alpha:")
        alpha_label.grid(row=3, column=0, sticky="w", pady=(10, 5))
        
        alpha_entry = ttk.Entry(gd3d_inner_frame, textvariable=self.surface_alpha_var, width=12)
        alpha_entry.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(10, 5))
        
        # Grid points - Row 4
        n_points_label = ttk.Label(gd3d_inner_frame, text="Grid Points:")
        n_points_label.grid(row=4, column=0, sticky="w", pady=(10, 5))
        
        n_points_entry = ttk.Entry(gd3d_inner_frame, textvariable=self.n_points_var, width=12)
        n_points_entry.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(10, 5))
        
        # FPS - Row 5
        fps_label = ttk.Label(gd3d_inner_frame, text="FPS:")
        fps_label.grid(row=5, column=0, sticky="w", pady=(10, 5))
        
        fps_entry = ttk.Entry(gd3d_inner_frame, textvariable=self.fps_var, width=12)
        fps_entry.grid(row=5, column=1, sticky="w", padx=(10, 0), pady=(10, 5))
        
        # Weight Ranges Section
        weight_ranges_frame = ttk.LabelFrame(plot_scrollable_frame, text="Weight Ranges", padding="10")
        weight_ranges_frame.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        weight_ranges_frame.grid_columnconfigure(0, weight=1)
        weight_ranges_frame.grid_columnconfigure(1, weight=1)
        weight_ranges_frame.grid_columnconfigure(2, weight=1)
        weight_ranges_frame.grid_columnconfigure(3, weight=1)
        
        # W1 Range
        w1_range_label = ttk.Label(weight_ranges_frame, text="W1 Range [min, max]:")
        w1_range_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        w1_min_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w1_range_min_var, width=10)
        w1_min_entry.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(0, 10))
        
        w1_max_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w1_range_max_var, width=10)
        w1_max_entry.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(0, 10))
        
        # W2 Range
        w2_range_label = ttk.Label(weight_ranges_frame, text="W2 Range [min, max]:")
        w2_range_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        w2_min_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w2_range_min_var, width=10)
        w2_min_entry.grid(row=3, column=0, sticky="w", padx=(0, 5), pady=(0, 5))
        
        w2_max_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w2_range_max_var, width=10)
        w2_max_entry.grid(row=3, column=1, sticky="w", padx=(5, 0), pady=(0, 5))
        
        # Weight Indices
        w1_index_label = ttk.Label(weight_ranges_frame, text="W1 Index:")
        w1_index_label.grid(row=0, column=2, sticky="w", pady=(0, 5))
        
        w1_index_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w1_index_var, width=10)
        w1_index_entry.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(0, 10))
        
        w2_index_label = ttk.Label(weight_ranges_frame, text="W2 Index:")
        w2_index_label.grid(row=2, column=2, sticky="w", pady=(0, 5))
        
        w2_index_entry = ttk.Entry(weight_ranges_frame, textvariable=self.w2_index_var, width=10)
        w2_index_entry.grid(row=3, column=2, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # View Settings Section
        view_settings_frame = ttk.LabelFrame(plot_scrollable_frame, text="View Settings", padding="10")
        view_settings_frame.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        view_settings_frame.grid_columnconfigure(0, weight=1)
        view_settings_frame.grid_columnconfigure(1, weight=1)
        
        # View angles
        view_elev_label = ttk.Label(view_settings_frame, text="View Elevation:")
        view_elev_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        view_elev_entry = ttk.Entry(view_settings_frame, textvariable=self.view_elev_var, width=12)
        view_elev_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 10))
        
        view_azim_label = ttk.Label(view_settings_frame, text="View Azimuth:")
        view_azim_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        view_azim_entry = ttk.Entry(view_settings_frame, textvariable=self.view_azim_var, width=12)
        view_azim_entry.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(0, 10))
        
        # Output resolution
        output_res_label = ttk.Label(view_settings_frame, text="Output Resolution [width, height]:")
        output_res_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        output_res_frame = ttk.Frame(view_settings_frame)
        output_res_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 5))
        output_res_frame.grid_columnconfigure(0, weight=1)
        output_res_frame.grid_columnconfigure(1, weight=1)
        
        output_width_entry = ttk.Entry(output_res_frame, textvariable=self.output_width_var, width=10)
        output_width_entry.grid(row=0, column=0, padx=(0, 5))
        
        output_height_entry = ttk.Entry(output_res_frame, textvariable=self.output_height_var, width=10)
        output_height_entry.grid(row=0, column=1, padx=(5, 0))
        
        # 3D Visualization Parameters Section
        gd3d_params_label = ttk.Label(plot_scrollable_frame, text="3D Visualization Parameters:", style="Bold.TLabel")
        gd3d_params_label.grid(row=4, column=0, sticky="w", pady=(20, 15))
        
        # Create frame for 3D parameters
        gd3d_params_frame = ttk.LabelFrame(plot_scrollable_frame, text="Animation & Control Parameters", padding="10")
        gd3d_params_frame.grid(row=5, column=0, sticky="ew", pady=(0, 20))
        gd3d_params_frame.grid_columnconfigure(0, weight=1)
        gd3d_params_frame.grid_columnconfigure(1, weight=1)
        
        # Animation Speed Control
        anim_speed_label = ttk.Label(gd3d_params_frame, text="Animation Speed:")
        anim_speed_label.grid(row=0, column=0, sticky="w", pady=(5, 2))
        
        self.anim_speed_var = tk.DoubleVar(value=1.0)
        anim_speed_scale = ttk.Scale(gd3d_params_frame, from_=0.1, to=5.0, variable=self.anim_speed_var, 
                                    orient="horizontal", length=200)
        anim_speed_scale.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        anim_speed_value_label = ttk.Label(gd3d_params_frame, textvariable=self.anim_speed_var)
        anim_speed_value_label.grid(row=0, column=1, sticky="e", padx=(0, 10))
        
        # Frame Position Control
        frame_pos_label = ttk.Label(gd3d_params_frame, text="Frame Position:")
        frame_pos_label.grid(row=1, column=0, sticky="w", pady=(15, 2))
        
        self.frame_pos_var = tk.DoubleVar(value=0.0)
        frame_pos_scale = ttk.Scale(gd3d_params_frame, from_=0, to=100, variable=self.frame_pos_var, 
                                   orient="horizontal", length=200, command=self.on_frame_pos_change)
        frame_pos_scale.grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        frame_pos_value_label = ttk.Label(gd3d_params_frame, textvariable=self.frame_pos_var)
        frame_pos_value_label.grid(row=1, column=1, sticky="e", padx=(0, 10))
        
        # View Preset Control (replaced listbox with combobox)
        view_preset_label = ttk.Label(gd3d_params_frame, text="View Preset:")
        view_preset_label.grid(row=2, column=0, sticky="w", pady=(15, 2))
        
        self.view_preset_var = tk.StringVar(value="Default")
        view_presets = ["Default", "Top View", "Side View", "Isometric", "Front View", "Back View"]
        view_preset_combo = ttk.Combobox(gd3d_params_frame, textvariable=self.view_preset_var, 
                                        values=view_presets, width=20, state="readonly")
        view_preset_combo.grid(row=2, column=1, sticky="w", padx=(10, 0))
        view_preset_combo.bind('<<ComboboxSelected>>', self.on_view_preset_change)
        
        # X Rotation Control
        x_rot_label = ttk.Label(gd3d_params_frame, text="X Rotation:")
        x_rot_label.grid(row=3, column=0, sticky="w", pady=(15, 2))
        
        self.x_rotation_var = tk.DoubleVar(value=0.0)
        x_rot_scale = ttk.Scale(gd3d_params_frame, from_=-180, to=180, variable=self.x_rotation_var, 
                               orient="horizontal", length=200, command=self.on_x_rotation_change)
        x_rot_scale.grid(row=3, column=1, sticky="w", padx=(10, 0))
        
        x_rot_value_label = ttk.Label(gd3d_params_frame, textvariable=self.x_rotation_var)
        x_rot_value_label.grid(row=3, column=1, sticky="e", padx=(0, 10))
        
        # Y Rotation Control
        y_rot_label = ttk.Label(gd3d_params_frame, text="Y Rotation:")
        y_rot_label.grid(row=4, column=0, sticky="w", pady=(15, 2))
        
        self.y_rotation_var = tk.DoubleVar(value=0.0)
        y_rot_scale = ttk.Scale(gd3d_params_frame, from_=-180, to=180, variable=self.y_rotation_var, 
                               orient="horizontal", length=200, command=self.on_y_rotation_change)
        y_rot_scale.grid(row=4, column=1, sticky="w", padx=(10, 0))
        
        y_rot_value_label = ttk.Label(gd3d_params_frame, textvariable=self.y_rotation_var)
        y_rot_value_label.grid(row=4, column=1, sticky="e", padx=(0, 10))
        
        # Z Rotation Control
        z_rot_label = ttk.Label(gd3d_params_frame, text="Z Rotation:")
        z_rot_label.grid(row=5, column=0, sticky="w", pady=(15, 2))
        
        self.z_rotation_var = tk.DoubleVar(value=0.0)
        z_rot_scale = ttk.Scale(gd3d_params_frame, from_=-180, to=180, variable=self.z_rotation_var, 
                               orient="horizontal", length=200, command=self.on_z_rotation_change)
        z_rot_scale.grid(row=5, column=1, sticky="w", padx=(10, 0))
        
        z_rot_value_label = ttk.Label(gd3d_params_frame, textvariable=self.z_rotation_var)
        z_rot_value_label.grid(row=5, column=1, sticky="e", padx=(0, 10))
        
        # Zoom Control
        zoom_label = ttk.Label(gd3d_params_frame, text="Zoom Level:")
        zoom_label.grid(row=6, column=0, sticky="w", pady=(15, 2))
        
        self.zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(gd3d_params_frame, from_=0.1, to=5.0, variable=self.zoom_var, 
                              orient="horizontal", length=200, command=self.on_zoom_change)
        zoom_scale.grid(row=6, column=1, sticky="w", padx=(10, 0))
        
        zoom_value_label = ttk.Label(gd3d_params_frame, textvariable=self.zoom_var)
        zoom_value_label.grid(row=6, column=1, sticky="e", padx=(0, 10))
        
        # Camera Position Controls
        camera_label = ttk.Label(gd3d_params_frame, text="Camera Position:", style="Bold.TLabel")
        camera_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(25, 10))
        
        # Camera X position
        cam_x_label = ttk.Label(gd3d_params_frame, text="Camera X:")
        cam_x_label.grid(row=8, column=0, sticky="w", pady=(5, 2))
        
        self.camera_x_var = tk.DoubleVar(value=0.0)
        cam_x_scale = ttk.Scale(gd3d_params_frame, from_=-10, to=10, variable=self.camera_x_var, 
                               orient="horizontal", length=200, command=self.on_camera_x_change)
        cam_x_scale.grid(row=8, column=1, sticky="w", padx=(10, 0))
        
        cam_x_value_label = ttk.Label(gd3d_params_frame, textvariable=self.camera_x_var)
        cam_x_value_label.grid(row=8, column=1, sticky="e", padx=(0, 10))
        
        # Camera Y position
        cam_y_label = ttk.Label(gd3d_params_frame, text="Camera Y:")
        cam_y_label.grid(row=9, column=0, sticky="w", pady=(15, 2))
        
        self.camera_y_var = tk.DoubleVar(value=0.0)
        cam_y_scale = ttk.Scale(gd3d_params_frame, from_=-10, to=10, variable=self.camera_y_var, 
                               orient="horizontal", length=150, command=self.on_camera_y_change)
        cam_y_scale.grid(row=9, column=1, sticky="w", padx=(10, 0))
        
        cam_y_value_label = ttk.Label(gd3d_params_frame, textvariable=self.camera_y_var)
        cam_y_value_label.grid(row=9, column=1, sticky="e", padx=(0, 10))
        
        # Camera Z position
        cam_z_label = ttk.Label(gd3d_params_frame, text="Camera Z:")
        cam_z_label.grid(row=10, column=0, sticky="w", pady=(5, 2))
        
        self.camera_z_var = tk.DoubleVar(value=5.0)
        cam_z_scale = ttk.Scale(gd3d_params_frame, from_=1, to=20, variable=self.camera_z_var, 
                               orient="horizontal", length=150, command=self.on_camera_z_change)
        cam_z_scale.grid(row=10, column=1, sticky="w", padx=(10, 0))
        
        cam_z_value_label = ttk.Label(gd3d_params_frame, textvariable=self.camera_z_var)
        cam_z_value_label.grid(row=10, column=1, sticky="e", padx=(0, 10))
        
        # Animation Controls Section
        # (REMOVED DUPLICATE: All controls and buttons in this section are now only inside plot_scrollable_frame)
        # The following block is intentionally left blank to remove the duplicate controls.

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
            # Try to find a suitable data file automatically
            auto_file = self.find_suitable_data_file()
            if auto_file:
                # Validate and convert the auto-selected file
                validated_file = self.validate_and_convert_data_file(auto_file)
                if validated_file:
                    self.data_file = validated_file
                    self.data_file_var.set(os.path.basename(validated_file))
                    self.load_data_features()  # Load features for the auto-selected file
                    messagebox.showinfo("Data File Auto-Selected", 
                                      f"Automatically selected and validated data file: {os.path.basename(validated_file)}\n\n"
                                      f"Training will proceed with this file.")
                else:
                    messagebox.showerror("Data File Error", 
                                       "No suitable data file found. Please select a data file manually.")
                    return
            else:
                messagebox.showerror("No Data File", 
                                   "Please select a data file before starting training.\n\n"
                                   "The file should have OHLCV columns (open, high, low, close, vol) "
                                   "or generic feature columns (feature_1, feature_2, etc.).")
                return
        
        # Validate the selected data file
        validated_file = self.validate_and_convert_data_file(self.data_file)
        if not validated_file:
            messagebox.showerror("Data File Error", 
                               "The selected data file is not valid for training.\n\n"
                               "Please select a file with proper OHLCV format or "
                               "generic feature columns that can be converted.")
            return
        
        # Update the data file to the validated/converted version
        if validated_file != self.data_file:
            self.data_file = validated_file
            self.data_file_var.set(os.path.basename(validated_file))
            self.load_data_features()
        
        # Check if output directory is set
        if not self.output_dir:
            messagebox.showerror("No Output Directory", "Please select an output directory.")
            return
        
        # Clear all plots and reset plot state before starting new training
        self.clear_all_plots()
        
        # Start training in a separate thread
        self.is_training = True
        self.training_thread = threading.Thread(target=self._train_model)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        # Update UI
        self.train_button.config(state='disabled')
        self.status_var.set("Training in progress...")

    def find_suitable_data_file(self):
        """Find a suitable data file for training."""
        # Check for sample data files first
        sample_files = [
            'sample_stock_data.csv',
            'sample_stock_data_extended.csv',
            'tsla_combined.csv'
        ]
        
        for file in sample_files:
            if os.path.exists(file):
                return file
        
        # Check for any CSV files in current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            return csv_files[0]
        
        return None

    def validate_and_convert_data_file(self, file_path):
        """
        Validate a data file and convert it to OHLCV format if needed.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            str: Path to the validated/converted file, or None if failed
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            # Read the file to check its format
            df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for analysis
            
            if DATA_CONVERTER_AVAILABLE:
                # Use the data converter to detect format
                format_info = detect_data_format(df)
                
                if format_info['status'] == 'ready':
                    # File is already in OHLCV format
                    return file_path
                
                elif format_info['status'] == 'convertible':
                    # File can be converted
                    response = messagebox.askyesno(
                        "Data Conversion Required",
                        f"Data file '{os.path.basename(file_path)}' needs to be converted to OHLCV format.\n\n"
                        f"Detected format: {format_info['format']}\n"
                        f"Message: {format_info['message']}\n\n"
                        "Would you like to convert it automatically?"
                    )
                    
                    if response:
                        # Convert the file
                        converted_file = convert_data_file(file_path)
                        if converted_file:
                            messagebox.showinfo(
                                "Conversion Successful",
                                f"Data file converted successfully!\n\n"
                                f"Original: {os.path.basename(file_path)}\n"
                                f"Converted: {os.path.basename(converted_file)}\n\n"
                                f"The converted file will be used for training."
                            )
                            return converted_file
                        else:
                            messagebox.showerror(
                                "Conversion Failed",
                                "Failed to convert the data file. Please check the file format."
                            )
                            return None
                    else:
                        # User chose not to convert
                        return None
                
                else:
                    # File is incompatible
                    messagebox.showerror(
                        "Incompatible Data Format",
                        f"Data file '{os.path.basename(file_path)}' has an incompatible format.\n\n"
                        f"Error: {format_info['message']}\n\n"
                        "Please use a file with OHLCV columns (open, high, low, close, vol) or "
                        "generic feature columns (feature_1, feature_2, etc.)."
                    )
                    return None
            
            else:
                # Data converter not available - do basic validation
                columns = list(df.columns)
                ohlcv_columns = ['open', 'high', 'low', 'close', 'vol']
                
                if all(col in columns for col in ohlcv_columns):
                    return file_path
                else:
                    messagebox.showerror(
                        "Invalid Data Format",
                        f"Data file '{os.path.basename(file_path)}' does not have the required OHLCV columns.\n\n"
                        f"Required columns: {ohlcv_columns}\n"
                        f"Found columns: {columns}\n\n"
                        "Please use a file with proper OHLCV format or install the data converter."
                    )
                    return None
                    
        except Exception as e:
            messagebox.showerror(
                "File Read Error",
                f"Error reading data file '{os.path.basename(file_path)}':\n\n{str(e)}"
            )
            return None

    def append_training_log(self, line):
        """Safely append a line to the training log text widget."""
        try:
            self.training_log_text.config(state="normal")
            self.training_log_text.insert("end", line)
            self.training_log_text.see("end")
            self.training_log_text.config(state="disabled")
        except Exception as e:
            print(f"Error appending to training log: {e}")

    def clear_training_log(self):
        """Clear the training log text widget."""
        try:
            self.training_log_text.config(state="normal")
            self.training_log_text.delete(1.0, tk.END)
            self.training_log_text.config(state="disabled")
        except Exception as e:
            print(f"Error clearing training log: {e}")

    def clear_all_plots(self):
        """Clear all plots and reset plot state before starting new training."""
        try:
            print("🧹 Clearing all plots and resetting plot state...")
            
            # 1. Close any existing live plot
            if hasattr(self, 'live_plot_window_open') and self.live_plot_window_open:
                self.close_live_plot()
            
            # 2. Clear training results plot
            if hasattr(self, 'results_ax') and self.results_ax is not None:
                self.results_ax.clear()
                self.results_ax.set_title("Training Results")
                self.results_ax.set_xlabel("Epoch")
                self.results_ax.set_ylabel("Loss")
                self.results_ax.grid(True)
                try:
                    self.results_canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating results canvas: {e}")
            
            # 3. Clear plots tab
            if hasattr(self, 'plots_ax') and self.plots_ax is not None:
                self.plots_ax.clear()
                self.plots_ax.set_title("Training Plots")
                self.plots_ax.set_xlabel("Epoch")
                self.plots_ax.set_ylabel("Loss")
                self.plots_ax.grid(True)
                try:
                    self.plots_canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating plots canvas: {e}")
            
            # 4. Clear 3D gradient descent plot
            if hasattr(self, 'gd3d_ax') and self.gd3d_ax is not None:
                self.gd3d_ax.clear()
                self.gd3d_ax.set_title("3D Gradient Descent")
                self.gd3d_ax.set_xlabel("Weight 1")
                self.gd3d_ax.set_ylabel("Weight 2")
                self.gd3d_ax.set_zlabel("Loss")
                try:
                    self.gd3d_canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating 3D canvas: {e}")
            
            # 5. Clear prediction plots
            if hasattr(self, 'pred_ax') and self.pred_ax is not None:
                self.pred_ax.clear()
                self.pred_ax.set_title("Prediction Results")
                self.pred_ax.set_xlabel(self.get_ticker_from_filename())
                self.pred_ax.set_ylabel("Value")
                self.pred_ax.grid(True)
                try:
                    self.pred_canvas.draw_idle()
                except Exception as e:
                    print(f"Error updating prediction canvas: {e}")
            
            # 6. Clear saved plots
            if hasattr(self, 'saved_plots_inner_frame'):
                for widget in self.saved_plots_inner_frame.winfo_children():
                    widget.destroy()
                if hasattr(self, 'saved_plots_images'):
                    self.saved_plots_images.clear()
            
            # 7. Clear training log
            self.clear_training_log()
            
            # 8. Reset selected model
            self.selected_model_path = None
            
            # 9. Update status
            self.status_var.set("All plots cleared and reset")
            print("✅ All plots cleared and reset successfully")
            
        except Exception as e:
            print(f"Error clearing plots: {e}")
            self.status_var.set(f"Error clearing plots: {str(e)}")

    def clear_cache_only(self):
        """Clear all caches without reinitializing plots."""
        try:
            print("🧹 Clearing all caches...")
            
            # 1. Clear image caches
            if hasattr(self, 'image_cache'):
                self.image_cache.clear()
                print("✅ Image cache cleared")
            
            if hasattr(self, 'plot_image_cache'):
                self.plot_image_cache.clear()
                print("✅ Plot image cache cleared")
            
            # 2. Clear saved plots images
            if hasattr(self, 'saved_plots_images'):
                self.saved_plots_images.clear()
                print("✅ Saved plots images cleared")
            
            # 3. Clear any pending plot futures
            if hasattr(self, 'plot_futures'):
                for future in self.plot_futures.values():
                    if not future.done():
                        future.cancel()
                self.plot_futures.clear()
                print("✅ Plot futures cleared")
            
            # 4. Update status
            self.status_var.set("Cache cleared successfully")
            print("✅ Cache cleared successfully")
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            self.status_var.set(f"Error clearing cache: {str(e)}")

    def clear_cache_and_reinitialize(self):
        """Clear all caches and reinitialize all plots."""
        try:
            print("🧹 Clearing all caches and reinitializing plots...")
            
            # 1. Clear image caches
            if hasattr(self, 'image_cache'):
                self.image_cache.clear()
                print("✅ Image cache cleared")
            
            if hasattr(self, 'plot_image_cache'):
                self.plot_image_cache.clear()
                print("✅ Plot image cache cleared")
            
            # 2. Clear saved plots images
            if hasattr(self, 'saved_plots_images'):
                self.saved_plots_images.clear()
                print("✅ Saved plots images cleared")
            
            # 3. Clear any pending plot futures
            if hasattr(self, 'plot_futures'):
                for future in self.plot_futures.values():
                    if not future.done():
                        future.cancel()
                self.plot_futures.clear()
                print("✅ Plot futures cleared")
            
            # 4. Clear all plots
            self.clear_all_plots()
            
            # 5. Reinitialize all plot canvases
            self._reinitialize_plot_canvases()
            
            # 6. Refresh models to reload data
            self.refresh_models()
            
            # 7. Update status
            self.status_var.set("Cache cleared and plots reinitialized")
            print("✅ Cache cleared and plots reinitialized successfully")
            
        except Exception as e:
            print(f"Error clearing cache and reinitializing: {e}")
            self.status_var.set(f"Error clearing cache: {str(e)}")

    def _reinitialize_plot_canvases(self):
        """Reinitialize all plot canvases with fresh matplotlib figures."""
        try:
            print("🔄 Reinitializing plot canvases...")
            
            # 1. Reinitialize training results canvas
            if hasattr(self, 'train_results_frame'):
                # Clear existing canvas
                for widget in self.train_results_frame.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for child in widget.winfo_children():
                            if isinstance(child, FigureCanvasTkAgg):
                                child.get_tk_widget().destroy()
                
                # Create new figure and canvas
                fig, ax = plt.subplots(figsize=(8, 6))
                self.results_ax = ax
                self.results_canvas = FigureCanvasTkAgg(fig, self.train_results_frame)
                self.results_canvas.draw()
                self.results_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                
                # Add toolbar
                toolbar = GridMatplotlibToolbar(self.results_canvas, self.train_results_frame)
                toolbar.grid(row=2, column=0, sticky="ew")
                
                # Set initial plot
                ax.set_title("Training Results")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True)
                ax.text(0.5, 0.5, 'Training in progress...', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            # 2. Reinitialize plots tab canvas
            if hasattr(self, 'plots_frame'):
                # Clear existing canvas
                for widget in self.plots_frame.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for child in widget.winfo_children():
                            if isinstance(child, FigureCanvasTkAgg):
                                child.get_tk_widget().destroy()
                
                # Create new figure and canvas
                fig, ax = plt.subplots(figsize=(8, 6))
                self.plots_ax = ax
                self.plots_canvas = FigureCanvasTkAgg(fig, self.plots_frame)
                self.plots_canvas.draw()
                self.plots_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                
                # Add toolbar
                toolbar = GridMatplotlibToolbar(self.plots_canvas, self.plots_frame)
                toolbar.grid(row=2, column=0, sticky="ew")
                
                # Set initial plot
                ax.set_title("Training Plots")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True)
                ax.text(0.5, 0.5, 'Training in progress...', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
            
            # 3. Reinitialize 3D gradient descent canvas
            if hasattr(self, 'gd3d_frame'):
                # Clear existing canvas
                for widget in self.gd3d_frame.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for child in widget.winfo_children():
                            if isinstance(child, FigureCanvasTkAgg):
                                child.get_tk_widget().destroy()
                
                # Create new figure and canvas
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                self.gd3d_ax = ax
                self.gd3d_canvas = FigureCanvasTkAgg(fig, self.gd3d_frame)
                self.gd3d_canvas.draw()
                self.gd3d_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                
                # Add toolbar
                toolbar = GridMatplotlibToolbar(self.gd3d_canvas, self.gd3d_frame)
                toolbar.grid(row=2, column=0, sticky="ew")
                
                # Set initial plot
                ax.set_title("3D Gradient Descent")
                ax.set_xlabel("Weight 1")
                ax.set_ylabel("Weight 2")
                ax.set_zlabel("Loss")
                ax.text2D(0.5, 0.5, 'Training in progress...', ha='center', va='center', 
                         transform=ax.transAxes, fontsize=12)
            
            # 4. Reinitialize prediction canvas
            if hasattr(self, 'pred_results_frame'):
                # Clear existing canvas
                for widget in self.pred_results_frame.winfo_children():
                    if hasattr(widget, 'winfo_children'):
                        for child in widget.winfo_children():
                            if isinstance(child, FigureCanvasTkAgg):
                                child.get_tk_widget().destroy()
                
                # Create new figure and canvas
                fig, ax = plt.subplots(figsize=(8, 6))
                self.pred_ax = ax
                self.pred_canvas = FigureCanvasTkAgg(fig, self.pred_results_frame)
                self.pred_canvas.draw()
                self.pred_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                
                # Add toolbar
                toolbar = GridMatplotlibToolbar(self.pred_canvas, self.pred_results_frame)
                toolbar.grid(row=2, column=0, sticky="ew")
                
                # Set initial plot
                ax.set_title("Prediction Results")
                ax.set_xlabel(self.get_ticker_from_filename())
                ax.set_ylabel("Value")
                ax.grid(True)
                ax.text(0.5, 0.5, 'Select a model and data file to view predictions', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            print("✅ All plot canvases reinitialized successfully")
            
        except Exception as e:
            print(f"Error reinitializing plot canvases: {e}")
            self.status_var.set(f"Error reinitializing plots: {str(e)}")

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
            
            # Validate data file exists and has required columns
            if not os.path.exists(self.data_file):
                raise Exception(f"Data file not found: {self.data_file}")
            
            # Build command
            cmd = [
                sys.executable, 'stock_net.py',
                '--data_file', self.data_file,
                '--x_features', ','.join(x_features),
                '--y_feature', y_feature,
                '--hidden_size', str(hidden_size),
                '--learning_rate', str(learning_rate),
                '--batch_size', str(batch_size)
            ]
            
            # Start training process with live output
            self.root.after(0, lambda: self.append_training_log(f"Starting training with command: {' '.join(cmd)}\n"))
            self.root.after(0, lambda: self.append_training_log(f"Data file: {self.data_file}\n"))
            self.root.after(0, lambda: self.append_training_log(f"Features: {x_features}\n"))
            self.root.after(0, lambda: self.append_training_log(f"Target: {y_feature}\n"))
            self.root.after(0, lambda: self.append_training_log("-" * 50 + "\n"))
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # Update GUI with the line
                    self.root.after(0, lambda l=line: self.append_training_log(l))
                    
                    # Parse line for loss information and update live plot
                    self.root.after(0, lambda l=line: self.parse_training_output_for_loss(l))
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                error_msg = f"Training failed with return code {return_code}"
                self.root.after(0, lambda: self.append_training_log(f"\n{error_msg}\n"))
                
                # Provide more specific error information
                if return_code == 1:
                    error_msg += "\n\nCommon causes:\n"
                    error_msg += "1. Data file missing required columns (open, high, low, close, vol)\n"
                    error_msg += "2. Data file is empty or corrupted\n"
                    error_msg += "3. Invalid feature names specified\n"
                    error_msg += "4. Insufficient data for training\n\n"
                    error_msg += "Please check your data file and try again."
                
                raise Exception(error_msg)
            
            # Training completed successfully
            self.root.after(0, lambda: self.append_training_log("\nTraining completed successfully!\n"))
            self.root.after(0, self._training_completed_success)
            
        except Exception as e:
            print(f"Error in training: {e}")
            # Fix lambda scoping issue by capturing the exception variable
            error_msg = str(e)
            self.root.after(0, lambda: self.append_training_log(f"\nError: {error_msg}\n"))
            self.root.after(0, lambda: self._training_completed_error(error_msg))

    def _training_completed_success(self):
        """Handle successful training completion."""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set("Training completed successfully!")
        self.refresh_models()
        # Display model info and training plots after training
        self._load_model_info()
        self.display_training_plots()
        # Close live plot if it's open
        if self.live_plot_window_open:
            self.close_live_plot()
        messagebox.showinfo("Success", "Model training completed successfully!")

    def _training_completed_error(self, error_msg):
        """Handle training completion with error."""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set(f"Training failed: {error_msg}")
        # Close live plot if it's open
        if self.live_plot_window_open:
            self.close_live_plot()
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
            
            # Load feature info from the model
            feature_info_path = os.path.join(self.selected_model_path, 'feature_info.json')
            if not os.path.exists(feature_info_path):
                messagebox.showerror("Model Error", "No feature_info.json found in model directory.")
                return
            
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            
            # Run prediction using predict.py with correct feature parameters
            cmd = [
                sys.executable, 'predict.py',
                self.data_file,  # Input file as positional argument
                '--model_dir', self.selected_model_path,
                '--output_dir', self.selected_model_path,  # Save in model directory
                '--x_features', ','.join(feature_info['x_features']),  # Add feature parameters
                '--y_feature', feature_info['y_feature']  # Add target feature
            ]
            
            print(f"Running prediction command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print(f"Prediction return code: {result.returncode}")
            print(f"Prediction stdout: {result.stdout}")
            print(f"Prediction stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error occurred"
                raise Exception(f"Prediction failed: {error_msg}")
            
            self.status_var.set("Prediction completed successfully!")
            messagebox.showinfo("Success", "Prediction completed successfully!")
            
            # Automatically update the Prediction Results tab
            self.update_prediction_results()
            
            # Refresh prediction files list
            self.refresh_prediction_files()
            
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
            
            # Start 3D visualization in a separate thread
            self.visualization_thread = threading.Thread(target=self._create_3d_visualization_thread)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            
        except Exception as e:
            print(f"Error starting 3D visualization: {e}")
            self.status_var.set(f"Error starting 3D visualization: {str(e)}")
            messagebox.showerror("3D Visualization Error", f"Error starting 3D visualization: {str(e)}")

    def _create_3d_visualization_thread(self):
        """Create 3D gradient descent visualization (runs in separate thread)."""
        try:
            # Get visualization parameters
            color = self.color_var.get()
            point_size = int(self.point_size_var.get())
            line_width = int(self.line_width_var.get())
            surface_alpha = float(self.surface_alpha_var.get())
            w1_range_min = float(self.w1_range_min_var.get())
            w1_range_max = float(self.w1_range_max_var.get())
            w2_range_min = float(self.w2_range_min_var.get())
            w2_range_max = float(self.w2_range_max_var.get())
            n_points = int(self.n_points_var.get())
            view_elev = float(self.view_elev_var.get())
            view_azim = float(self.view_azim_var.get())
            fps = int(self.fps_var.get())
            w1_index = int(self.w1_index_var.get())
            w2_index = int(self.w2_index_var.get())
            output_width = int(self.output_width_var.get())
            output_height = int(self.output_height_var.get())
            
            # Run 3D visualization using gradient_descent_3d.py
            cmd = [
                sys.executable, 'gradient_descent_3d.py',
                '--model_dir', self.selected_model_path,
                '--color', color,
                '--point_size', str(point_size),
                '--line_width', str(line_width),
                '--surface_alpha', str(surface_alpha),
                '--w1_range', str(w1_range_min), str(w1_range_max),
                '--w2_range', str(w2_range_min), str(w2_range_max),
                '--n_points', str(n_points),
                '--view_elev', str(view_elev),
                '--view_azim', str(view_azim),
                '--fps', str(fps),
                '--w1_index', str(w1_index),
                '--w2_index', str(w2_index),
                '--output_resolution', str(output_width), str(output_height),
                '--save_png'
            ]
            
            print(f"Running 3D visualization command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print(f"3D visualization return code: {result.returncode}")
            print(f"3D visualization stdout: {result.stdout}")
            print(f"3D visualization stderr: {result.stderr}")
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error occurred"
                raise Exception(f"3D visualization failed: {error_msg}")
            
            # Check if any PNG files were created
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if os.path.exists(plots_dir):
                png_files = glob.glob(os.path.join(plots_dir, '*.png'))
                print(f"Found {len(png_files)} PNG files in plots directory")
            
            # 3D visualization completed successfully
            self.root.after(0, self._3d_visualization_completed_success)
            
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
            # Fix lambda scoping issue by capturing the exception variable
            error_msg = str(e) if str(e) else "Unknown error occurred"
            self.root.after(0, lambda: self._3d_visualization_completed_error(error_msg))

    def _3d_visualization_completed_success(self):
        """Handle successful 3D visualization completion."""
        self.status_var.set("3D visualization created successfully!")
        self.refresh_models()  # Refresh to show new plots
        messagebox.showinfo("Success", "3D visualization created successfully!")

    def _3d_visualization_completed_error(self, error_msg):
        """Handle 3D visualization completion with error."""
        self.status_var.set(f"3D visualization failed: {error_msg}")
        messagebox.showerror("3D Visualization Error", f"3D visualization failed: {error_msg}")

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
            
            # Load normalization parameters
            self.X_min = np.loadtxt(os.path.join(self.selected_model_path, 'scaler_mean.csv'), delimiter=',')
            self.X_max = np.loadtxt(os.path.join(self.selected_model_path, 'scaler_std.csv'), delimiter=',') + self.X_min
            
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
                
                # Safe canvas update
                try:
                    self.gd3d_canvas.draw_idle()
                except Exception as canvas_error:
                    print(f"Canvas update error in 3D gradient descent creation: {canvas_error}")
                
                print(f"Created 3D gradient descent visualization with {len(weights)} weight points")
            else:
                print(f"No weight history files found in {model_dir}")
                
        except Exception as e:
            print(f"Error creating 3D gradient descent visualization: {e}")
            import traceback
            traceback.print_exc()

    def initialize_prediction_canvas(self):
        """Initialize the prediction canvas and axes if they don't exist."""
        try:
            # Find the prediction results frame in the display panel
            for widget in self.root.winfo_children():
                if hasattr(widget, 'winfo_children'):
                    for child in widget.winfo_children():
                        if hasattr(child, 'winfo_children'):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Notebook):
                                    # This is the display notebook
                                    for tab_id in range(grandchild.index('end')):
                                        tab_text = grandchild.tab(tab_id, 'text')
                                        if 'prediction' in tab_text.lower():
                                            # Found prediction results tab
                                            for tab_child in grandchild.winfo_children():
                                                if hasattr(tab_child, 'winfo_children'):
                                                    for frame in tab_child.winfo_children():
                                                        if isinstance(frame, ttk.Frame):
                                                            # Create figure and canvas
                                                            fig, ax = plt.subplots(figsize=(10, 6))
                                                            self.pred_ax = ax
                                                            self.pred_canvas = FigureCanvasTkAgg(fig, frame)
                                                            self.pred_canvas.draw()
                                                            self.pred_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                                                            
                                                            # Add toolbar
                                                            toolbar = GridMatplotlibToolbar(self.pred_canvas, frame)
                                                            toolbar.grid(row=2, column=0, sticky="ew")
                                                            
                                                            # Show placeholder
                                                            ax.text(0.5, 0.5, 'Select a model and data file to view predictions', 
                                                                   ha='center', va='center', transform=ax.transAxes,
                                                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                                                            ax.set_title("Prediction Results")
                                                            ax.axis('off')
                                                            
                                                            # Safe canvas update
                                                            try:
                                                                self.pred_canvas.draw_idle()
                                                            except Exception as canvas_error:
                                                                print(f"Canvas update error: {canvas_error}")
                                                            
                                                            return True
                                            break
                                    break
                        break
                break
            return False
        except Exception as e:
            print(f"Error initializing prediction canvas: {e}")
            return False

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
        train_results_frame.grid_rowconfigure(1, weight=1)
        
        # Create matplotlib figure for training results
        self.results_fig = plt.Figure(figsize=(8, 4))
        self.results_ax = self.results_fig.add_subplot(111)
        
        # Create canvas and embed in the tab using grid
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, train_results_frame)
        self.results_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.results_canvas, train_results_frame)
        
        # Add live training log text widget
        log_frame = ttk.LabelFrame(train_results_frame, text="Live Training Log", padding="5")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar for live training messages
        self.training_log_text = tk.Text(log_frame, height=8, bg="#222222", fg="#CCCCCC", 
                                        font=("Consolas", 9), state="disabled")
        self.training_log_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Add scrollbar for the text widget
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.training_log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.training_log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Initialize with a placeholder plot
        self.results_ax.text(0.5, 0.5, 'Training results will appear here', 
                            ha='center', va='center', transform=self.results_ax.transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.results_ax.set_title("Training Results")
        
        # Safe canvas update
        try:
            self.results_canvas.draw_idle()
        except Exception as canvas_error:
            print(f"Canvas update error in training results initialization: {canvas_error}")
        
        print("Training Results tab created successfully")

        # Prediction Results Tab
        pred_results_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(pred_results_frame, text="Prediction Results")
        pred_results_frame.grid_columnconfigure(0, weight=1)
        pred_results_frame.grid_rowconfigure(0, weight=0)  # Controls
        pred_results_frame.grid_rowconfigure(1, weight=1)  # Plot canvas
        pred_results_frame.grid_rowconfigure(2, weight=0)  # Toolbar
        
        # Add controls frame for prediction results
        pred_controls_frame = ttk.Frame(pred_results_frame)
        pred_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        pred_controls_frame.grid_columnconfigure(0, weight=1)
        
        # Add refresh button for prediction results
        self.refresh_pred_btn = ttk.Button(pred_controls_frame, text="🔄 Refresh Prediction Results", 
                                          command=self.update_prediction_results)
        self.refresh_pred_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Add status label for prediction results
        self.pred_status_label = ttk.Label(pred_controls_frame, text="Select a model to view prediction results", 
                                          foreground=TEXT_COLOR, background=FRAME_COLOR)
        self.pred_status_label.grid(row=1, column=0, padx=5, pady=5)
        
        # Create figure for prediction plot
        self.pred_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=pred_results_frame)
        self.pred_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Add toolbar for prediction results
        pred_toolbar = GridMatplotlibToolbar(self.pred_canvas, pred_results_frame)
        pred_toolbar.grid(row=2, column=0, sticky="ew")
        
        # Initialize with a placeholder plot
        self.pred_ax.text(0.5, 0.5, 'Select a model and data file to view predictions', 
                         ha='center', va='center', transform=self.pred_ax.transAxes,
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.pred_ax.set_title("Prediction Results")
        self.pred_ax.axis('off')
        
        # Safe canvas update
        try:
            self.pred_canvas.draw_idle()
        except Exception as canvas_error:
            print(f"Canvas update error in prediction results initialization: {canvas_error}")
        
        print("Prediction Results tab created successfully")

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
        
        # Safe canvas update
        try:
            self.gd_canvas.draw_idle()
        except Exception as canvas_error:
            print(f"Canvas update error in gradient descent initialization: {canvas_error}")
        
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
        
        # Safe canvas update
        try:
            self.plots_canvas.draw_idle()
        except Exception as canvas_error:
            print(f"Canvas update error in plots initialization: {canvas_error}")
        
        # 3D Gradient Descent Tab
        gd3d_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(gd3d_frame, text="3D Gradient Descent")
        gd3d_frame.grid_columnconfigure(0, weight=1)
        gd3d_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for 3D gradient descent visualization
        self.gd3d_fig = plt.Figure(figsize=(8, 6))
        self.gd3d_ax = self.gd3d_fig.add_subplot(111)
        
        # Create canvas and embed in the tab using grid
        self.gd3d_canvas = FigureCanvasTkAgg(self.gd3d_fig, gd3d_frame)
        self.gd3d_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar using grid
        toolbar = GridMatplotlibToolbar(self.gd3d_canvas, gd3d_frame)
        
        # Initialize with a placeholder plot
        self.gd3d_ax.text(0.5, 0.5, '3D Gradient Descent visualization will appear here after training', 
                          ha='center', va='center', transform=self.gd3d_ax.transAxes,
                          fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.gd3d_ax.set_title("3D Gradient Descent Visualization")
        
        # Safe canvas update
        try:
            self.gd3d_canvas.draw_idle()
        except Exception as canvas_error:
            print(f"Canvas update error in 3D gradient descent initialization: {canvas_error}")
        
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
        
        # Live Training Plot Tab (New)
        live_plot_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(live_plot_frame, text="Live Training Plot")
        live_plot_frame.grid_columnconfigure(0, weight=1)
        live_plot_frame.grid_rowconfigure(0, weight=1)
        
        # Create frame for live plot controls
        live_controls_frame = ttk.Frame(live_plot_frame)
        live_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        live_controls_frame.grid_columnconfigure(0, weight=1)
        
        # Add button to open live plot
        self.open_live_plot_btn = ttk.Button(live_controls_frame, text="Open Live Training Plot (Matplotlib)", 
                                            command=self.open_live_training_plot)
        self.open_live_plot_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Add status label for live plot
        self.live_plot_status = ttk.Label(live_controls_frame, text="Click 'Open Live Training Plot (Matplotlib)' to start monitoring", 
                                         foreground=TEXT_COLOR, background=FRAME_COLOR)
        self.live_plot_status.grid(row=1, column=0, padx=5, pady=5)
        
        # Initialize live plot variables
        self.live_plot_fig = None
        self.live_plot_epochs = []
        self.live_plot_losses = []
        self.live_plot_window_open = False
        
        print("Live Training Plot tab created successfully")
        
        print(f"Display panel created with {self.display_notebook.index('end')} tabs")
        
        # Ensure prediction canvas is properly initialized
        if not hasattr(self, 'pred_canvas') or not self.pred_canvas:
            print("Initializing prediction canvas...")
            self.initialize_prediction_canvas()

    def on_model_select(self, event):
        """Handle model selection."""
        try:
            selected_model = self.model_combo.get()
            if selected_model:
                # Find the model directory
                model_dir = None
                for item in os.listdir(self.current_model_dir):
                    if item.startswith("model_") and os.path.isdir(os.path.join(self.current_model_dir, item)):
                        if item == selected_model:
                            model_dir = os.path.join(self.current_model_dir, item)
                            break
                
                if model_dir and os.path.exists(model_dir):
                    self.selected_model_path = model_dir
                    print(f"Selected model: {selected_model} at {model_dir}")
                    
                    # Update prediction results automatically
                    self.update_prediction_results()
                    
                    # Update training results
                    self.display_training_plots()
                    
                    # Update 3D gradient descent tab
                    self.update_3d_gradient_descent_tab()
                    
                    # Update saved plots tab
                    self.load_saved_plots()
                    
                    # Refresh prediction files for this model
                    self.refresh_prediction_files()
                    
                    self.status_var.set(f"Selected model: {selected_model}")
                else:
                    self.status_var.set(f"Model directory not found: {selected_model}")
        except Exception as e:
            print(f"Error in model selection: {e}")
            self.status_var.set(f"Error selecting model: {str(e)}")

    def on_prediction_file_select(self, event):
        """Handle prediction file selection."""
        try:
            selection = self.prediction_files_listbox.curselection()
            if selection:
                selected_file = self.prediction_files_listbox.get(selection[0])
                self.selected_prediction_file = selected_file
                print(f"Selected prediction file: {selected_file}")
                self.status_var.set(f"Selected prediction file: {os.path.basename(selected_file)}")
        except Exception as e:
            print(f"Error in prediction file selection: {e}")
            self.status_var.set(f"Error selecting prediction file: {str(e)}")

    def refresh_prediction_files(self):
        """Refresh the list of prediction files for the selected model."""
        try:
            self.prediction_files_listbox.delete(0, tk.END)
            
            if not self.selected_model_path:
                self.status_var.set("No model selected")
                return
            
            # Look for prediction files in the model directory
            prediction_files = []
            model_dir = self.selected_model_path
            
            # Check for common prediction file patterns
            for file in os.listdir(model_dir):
                if file.endswith('.csv') and any(pattern in file.lower() for pattern in ['prediction', 'pred', 'result', 'output']):
                    prediction_files.append(file)
            
            # Also check for prediction files in subdirectories
            for subdir in ['predictions', 'results', 'output']:
                subdir_path = os.path.join(model_dir, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.csv'):
                            prediction_files.append(os.path.join(subdir, file))
            
            # Sort files by modification time (newest first)
            prediction_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            
            # Add files to listbox
            for file in prediction_files:
                self.prediction_files_listbox.insert(tk.END, file)
            
            if prediction_files:
                self.status_var.set(f"Found {len(prediction_files)} prediction file(s)")
            else:
                # Add helpful message to listbox when no files found
                self.prediction_files_listbox.insert(tk.END, "No prediction files found")
                self.prediction_files_listbox.insert(tk.END, "Use 'Make Prediction' button to create predictions")
                self.prediction_files_listbox.insert(tk.END, "or run predict.py manually")
                self.status_var.set("No prediction files found - use 'Make Prediction' to create them")
                
        except Exception as e:
            print(f"Error refreshing prediction files: {e}")
            self.status_var.set(f"Error refreshing prediction files: {str(e)}")

    def view_prediction_results(self):
        """View the selected prediction file results."""
        try:
            selection = self.prediction_files_listbox.curselection()
            if not selection:
                self.status_var.set("No prediction file selected")
                return
            
            selected_file = self.prediction_files_listbox.get(selection[0])
            
            if not self.selected_model_path:
                self.status_var.set("No model selected")
                return
            
            # Construct full path to prediction file
            if os.path.dirname(selected_file):
                # File is in a subdirectory
                pred_file_path = os.path.join(self.selected_model_path, selected_file)
            else:
                # File is directly in model directory
                pred_file_path = os.path.join(self.selected_model_path, selected_file)
            
            if not os.path.exists(pred_file_path):
                self.status_var.set(f"Prediction file not found: {selected_file}")
                return
            
            # Load and display prediction results
            self.load_prediction_file(pred_file_path)
            
            # Switch to Prediction Results tab
            self.switch_to_tab(1)  # Prediction Results tab
            
            self.status_var.set(f"Loaded prediction results: {os.path.basename(selected_file)}")
            
        except Exception as e:
            print(f"Error viewing prediction results: {e}")
            self.status_var.set(f"Error viewing prediction results: {str(e)}")

    def load_prediction_file(self, file_path):
        """Load prediction file and update the Prediction Results tab."""
        try:
            import pandas as pd
            
            # Load the prediction file
            df = pd.read_csv(file_path)
            
            # Update prediction results display
            self.update_prediction_results_with_data(df, file_path)
            
        except Exception as e:
            print(f"Error loading prediction file: {e}")
            self.status_var.set(f"Error loading prediction file: {str(e)}")

    def update_prediction_results_with_data(self, df, file_path):
        """Update the Prediction Results tab with loaded prediction data."""
        try:
            # Clear the existing plot instead of destroying the canvas
            if hasattr(self, 'pred_ax') and self.pred_ax:
                self.pred_ax.clear()
            else:
                # If no existing axes, create a new figure
                fig, ax = plt.subplots(figsize=(10, 6))
                self.pred_ax = ax
                # Create canvas if it doesn't exist
                if not hasattr(self, 'pred_canvas') or not self.pred_canvas:
                    # Find the prediction results frame
                    for widget in self.root.winfo_children():
                        if hasattr(widget, 'winfo_children'):
                            for child in widget.winfo_children():
                                if hasattr(child, 'winfo_children'):
                                    for grandchild in child.winfo_children():
                                        if isinstance(grandchild, ttk.Notebook):
                                            # This is the display notebook
                                            for tab_id in range(grandchild.index('end')):
                                                tab = grandchild.select()
                                                if 'prediction' in grandchild.tab(tab_id, 'text').lower():
                                                    # Found prediction results tab
                                                    for tab_child in grandchild.winfo_children():
                                                        if hasattr(tab_child, 'winfo_children'):
                                                            for frame in tab_child.winfo_children():
                                                                if isinstance(frame, ttk.Frame):
                                                                    self.pred_canvas = FigureCanvasTkAgg(fig, frame)
                                                                    self.pred_canvas.draw()
                                                                    self.pred_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
                                                                    break
                                                    break
                                            break
                                    break
                        break
                return
            
            # Get ticker name for x-axis label
            ticker = self.get_ticker_from_filename()
            
            # Plot prediction results based on data structure
            if 'actual' in df.columns and 'predicted' in df.columns:
                # Create single combined plot with both actual and predicted
                x_axis = range(len(df))
                
                # Plot both lines on the same axis
                self.pred_ax.plot(x_axis, df['actual'], label='Actual', color='blue', alpha=0.7, linewidth=2)
                self.pred_ax.plot(x_axis, df['predicted'], label='Predicted', color='red', alpha=0.7, linewidth=2)
                
                # Calculate and display correlation
                correlation = np.corrcoef(df['actual'], df['predicted'])[0, 1]
                self.pred_ax.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                                transform=self.pred_ax.transAxes, va='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                # Calculate error metrics
                mae = np.mean(np.abs(df['actual'] - df['predicted']))
                mse = np.mean((df['actual'] - df['predicted']) ** 2)
                rmse = np.sqrt(mse)
                
                title = f"Prediction Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}"
                
            elif 'prediction' in df.columns:
                # Plot predictions only
                x_axis = range(len(df))
                self.pred_ax.plot(x_axis, df['prediction'], label='Prediction', color='green', linewidth=2)
                title = 'Predictions'
                
            else:
                # Plot first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    x_axis = range(len(df))
                    self.pred_ax.plot(x_axis, df[numeric_cols[0]], label=numeric_cols[0], color='purple', linewidth=2)
                    title = f'Prediction Results: {numeric_cols[0]}'
                else:
                    # No numeric columns, show data info
                    self.pred_ax.text(0.5, 0.5, f'No numeric data to plot\nFile: {os.path.basename(file_path)}\nShape: {df.shape}', 
                                   transform=self.pred_ax.transAxes, ha='center', va='center', fontsize=12)
                    self.pred_ax.set_title('Prediction File Info')
                    self.pred_ax.axis('off')
                    
                    # Safe canvas update
                    try:
                        if hasattr(self, 'pred_canvas') and self.pred_canvas:
                            self.pred_canvas.draw_idle()
                    except Exception as canvas_error:
                        print(f"Canvas update error: {canvas_error}")
                    return
            
            # Set plot properties with ticker name instead of "Sample"
            self.pred_ax.set_title(title)
            self.pred_ax.set_xlabel(ticker)
            self.pred_ax.set_ylabel("Value")
            self.pred_ax.legend()
            self.pred_ax.grid(True, alpha=0.3)
            
            # Safe canvas update with error handling
            try:
                # Use draw_idle() instead of draw() to avoid recursion
                if hasattr(self, 'pred_canvas') and self.pred_canvas:
                    self.pred_canvas.draw_idle()
                    # Force a small delay to allow the canvas to update
                    self.root.after(100, lambda: None)
            except Exception as canvas_error:
                print(f"Canvas update error: {canvas_error}")
                # Try alternative update method
                try:
                    if hasattr(self, 'pred_canvas') and self.pred_canvas:
                        self.pred_canvas.flush_events()
                except:
                    pass
            
            # Update status
            self.prediction_status_var.set(f"Loaded: {os.path.basename(file_path)} | Shape: {df.shape}")
            
            # Also update the status label
            if hasattr(self, 'pred_status_label'):
                self.pred_status_label.config(text=f"Loaded: {os.path.basename(file_path)} | Shape: {df.shape}")
            
        except Exception as e:
            print(f"Error updating prediction results with data: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message in the plot
            try:
                if hasattr(self, 'pred_ax') and self.pred_ax:
                    self.pred_ax.clear()
                    self.pred_ax.text(0.5, 0.5, f'Error loading prediction data:\n{str(e)}', 
                                    ha='center', va='center', transform=self.pred_ax.transAxes,
                                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    self.pred_ax.set_title("Prediction Results - Error")
                    self.pred_ax.axis('off')
                    
                    # Safe canvas update
                    try:
                        if hasattr(self, 'pred_canvas') and self.pred_canvas:
                            self.pred_canvas.draw_idle()
                    except Exception as canvas_error:
                        print(f"Canvas update error: {canvas_error}")
            except Exception as plot_error:
                print(f"Error creating error plot: {plot_error}")
            
            self.prediction_status_var.set(f"Error displaying prediction data: {str(e)}")
            
            # Also update the status label
            if hasattr(self, 'pred_status_label'):
                self.pred_status_label.config(text=f"Error displaying prediction data: {str(e)}")

    def update_prediction_results(self):
        """Update the Prediction Results tab with prediction data and statistics."""
        try:
            if not self.selected_model_path or not self.data_file:
                # Show placeholder if no model or data selected
                if hasattr(self, 'pred_ax') and self.pred_ax:
                    self.pred_ax.clear()
                    self.pred_ax.text(0.5, 0.5, 'Select a model and data file to view predictions', 
                                    ha='center', va='center', transform=self.pred_ax.transAxes,
                                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    self.pred_ax.set_title("Prediction Results")
                    self.pred_ax.axis('off')
                    
                    # Safe canvas update
                    try:
                        if hasattr(self, 'pred_canvas') and self.pred_canvas:
                            self.pred_canvas.draw_idle()
                    except Exception as canvas_error:
                        print(f"Canvas update error: {canvas_error}")
                
                # Update status label
                if not self.selected_model_path:
                    self.prediction_status_var.set("No model selected")
                elif not self.data_file:
                    self.prediction_status_var.set("No data file selected")
                return
            
            # Look for prediction files in the model directory
            pred_files = glob.glob(os.path.join(self.selected_model_path, 'predictions_*.csv'))
            
            # Also look for other possible prediction file patterns
            all_csv_files = glob.glob(os.path.join(self.selected_model_path, '*.csv'))
            
            # Check if model directory exists and list its contents
            if os.path.exists(self.selected_model_path):
                model_contents = os.listdir(self.selected_model_path)
            
            if not pred_files:
                # Show placeholder if no prediction files found
                if hasattr(self, 'pred_ax') and self.pred_ax:
                    self.pred_ax.clear()
                    self.pred_ax.text(0.5, 0.5, 'No prediction files found.\nRun prediction to generate results.', 
                                    ha='center', va='center', transform=self.pred_ax.transAxes,
                                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                    self.pred_ax.set_title("Prediction Results")
                    self.pred_ax.axis('off')
                    
                    # Safe canvas update
                    try:
                        if hasattr(self, 'pred_canvas') and self.pred_canvas:
                            self.pred_canvas.draw_idle()
                    except Exception as canvas_error:
                        print(f"Canvas update error: {canvas_error}")
                
                # Update status label
                self.prediction_status_var.set("No prediction files found - run prediction first")
                return
            
            # Use the latest prediction file
            latest_pred_file = max(pred_files, key=os.path.getctime)
            
            # Load prediction data
            import pandas as pd
            df = pd.read_csv(latest_pred_file)
            
            # Clear the plot
            if hasattr(self, 'pred_ax') and self.pred_ax:
                self.pred_ax.clear()
            else:
                print("Warning: pred_ax not found, cannot update prediction results")
                return
            
            # Determine what columns are available for plotting
            has_close = 'close' in df.columns
            has_predicted_close = 'predicted_close' in df.columns
            has_actual = 'actual' in df.columns
            has_predicted = 'predicted' in df.columns
            
            if has_close and has_predicted_close:
                # Plot actual vs predicted close prices
                x_axis = range(len(df))
                self.pred_ax.plot(x_axis, df['close'], 'b-', label='Actual Close', alpha=0.7, linewidth=2)
                self.pred_ax.plot(x_axis, df['predicted_close'], 'r-', label='Predicted Close', alpha=0.7, linewidth=2)
                
                # Calculate and display correlation
                correlation = np.corrcoef(df['close'], df['predicted_close'])[0, 1]
                self.pred_ax.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                                transform=self.pred_ax.transAxes, va='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                # Calculate error metrics
                mae = np.mean(np.abs(df['close'] - df['predicted_close']))
                mse = np.mean((df['close'] - df['predicted_close']) ** 2)
                rmse = np.sqrt(mse)
                
                # Add metrics to title
                title = f"Prediction Results - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}"
                
                # Update status label
                self.prediction_status_var.set(f"Showing {len(df)} predictions - MAE: ${mae:.2f}, Correlation: {correlation:.4f}")
                
            elif has_actual and has_predicted:
                # Plot actual vs predicted values
                x_axis = range(len(df))
                self.pred_ax.plot(x_axis, df['actual'], 'b-', label='Actual', alpha=0.7, linewidth=2)
                self.pred_ax.plot(x_axis, df['predicted'], 'r-', label='Predicted', alpha=0.7, linewidth=2)
                
                # Calculate and display correlation
                correlation = np.corrcoef(df['actual'], df['predicted'])[0, 1]
                self.pred_ax.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
                                transform=self.pred_ax.transAxes, va='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                # Calculate error metrics
                mae = np.mean(np.abs(df['actual'] - df['predicted']))
                mse = np.mean((df['actual'] - df['predicted']) ** 2)
                rmse = np.sqrt(mse)
                
                # Add metrics to title
                title = f"Prediction Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}"
                
                # Update status label
                self.prediction_status_var.set(f"Showing {len(df)} predictions - MAE: {mae:.4f}, Correlation: {correlation:.4f}")
                
            else:
                # Show available columns for debugging
                self.pred_ax.text(0.5, 0.5, f'Available columns: {list(df.columns)}\nNo standard prediction columns found', 
                                ha='center', va='center', transform=self.pred_ax.transAxes,
                                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                title = "Prediction Results - Column Format Issue"
                
                # Update status label
                self.prediction_status_var.set(f"Column format issue - found: {list(df.columns)}")
            
            # Set plot properties
            self.pred_ax.set_title(title)
            self.pred_ax.set_xlabel(self.get_ticker_from_filename())
            self.pred_ax.set_ylabel("Value")
            self.pred_ax.legend()
            self.pred_ax.grid(True, alpha=0.3)
            
            # Safe canvas update with error handling
            try:
                # Use draw_idle() instead of draw() to avoid recursion
                if hasattr(self, 'pred_canvas') and self.pred_canvas:
                    self.pred_canvas.draw_idle()
                    # Force a small delay to allow the canvas to update
                    self.root.after(100, lambda: None)
            except Exception as canvas_error:
                print(f"Canvas update error: {canvas_error}")
                # Try alternative update method
                try:
                    self.pred_canvas.flush_events()
                except:
                    pass
                    
        except Exception as e:
            print(f"Error updating prediction results: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message in the plot
            try:
                if hasattr(self, 'pred_ax') and self.pred_ax:
                    self.pred_ax.clear()
                    self.pred_ax.text(0.5, 0.5, f'Error updating prediction results:\n{str(e)}', 
                                    ha='center', va='center', transform=self.pred_ax.transAxes,
                                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                    self.pred_ax.set_title("Prediction Results - Error")
                    self.pred_ax.axis('off')
                    
                    # Safe canvas update
                    try:
                        if hasattr(self, 'pred_canvas') and self.pred_canvas:
                            self.pred_canvas.draw_idle()
                    except Exception as canvas_error:
                        print(f"Canvas update error: {canvas_error}")
            except Exception as plot_error:
                print(f"Error creating error plot: {plot_error}")
            
            self.prediction_status_var.set(f"Error updating prediction results: {str(e)}")

    def refresh_models(self, load_plots=True):
        """Refresh the list of available models and update plots."""
        try:
            # Clear existing models from combobox
            self.model_combo['values'] = ()
            self.selected_model_path = None
            
            if not os.path.exists(self.current_model_dir):
                os.makedirs(self.current_model_dir, exist_ok=True)
                self.status_var.set(f"Created models directory: {self.current_model_dir}")
                if load_plots:
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
                if load_plots:
                    self.on_model_select(None)  # Trigger plot loading
            else:
                self.status_var.set("No models available")
                if load_plots:
                    self.load_saved_plots()
            
            # --- ADD THIS LINE ---
            self.refresh_prediction_files()
                
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
            
            # Set the newly trained model as selected
            self.selected_model_path = model_dir
            
            # Automatically make a prediction with the newly trained model
            if self.data_file:
                print("Training completed - making automatic prediction...")
                self.status_var.set("Training completed - making automatic prediction...")
                
                # Use after() to schedule the automatic prediction to run after GUI updates
                self.root.after(1000, self.auto_make_prediction)
            else:
                print("Training completed - no data file selected for automatic prediction")
                self.status_var.set("Training completed - select data file for prediction")
            
            messagebox.showinfo("Success", f"Model training completed successfully!\nModel saved in: {model_dir}\n3D gradient descent visualization created!")
            
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
            
            # Safe canvas update
            try:
                self.results_canvas.draw_idle()
            except Exception as canvas_error:
                print(f"Canvas update error in training results: {canvas_error}")
                
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
            
            # Safe canvas update
            try:
                self.plots_canvas.draw_idle()
            except Exception as canvas_error:
                print(f"Canvas update error in plots tab: {canvas_error}")
            
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
                print("No 3D visualization files found")
                # Show a placeholder message
                self.gd3d_ax.clear()
                self.gd3d_ax.text(0.5, 0.5, 'No 3D gradient descent visualization found.\nRun 3D visualization to generate one.', 
                                ha='center', va='center', transform=self.gd3d_ax.transAxes,
                                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.gd3d_ax.set_title("3D Gradient Descent Visualization")
                self.gd3d_ax.axis('off')
                
                # Update frame label to show no frames
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text="Frame: 0/0")
                
                # Safe canvas update
                try:
                    self.gd3d_canvas.draw_idle()
                except Exception as canvas_error:
                    print(f"Canvas update error in 3D tab: {canvas_error}")
                return
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
            
            print(f"Found {len(gd3d_files)} 3D visualization files: {[os.path.basename(f) for f in gd3d_files]}")
            
            if gd3d_files:
                # Update animation variables
                self.total_frames = len(gd3d_files)
                self.current_frame = 0
                
                # Update frame label
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text=f"Frame: 1/{self.total_frames}")
                
                # Load the last frame (most complete visualization)
                latest_frame = gd3d_files[-1]
                print(f"Loading latest 3D visualization: {os.path.basename(latest_frame)}")
                self.load_3d_visualization_image(latest_frame)
                
                # Update status
                self.status_var.set(f"3D visualization loaded: {self.total_frames} frames available")
            else:
                print("No 3D visualization files found in plots directory")
                # Show a placeholder message
                self.gd3d_ax.clear()
                self.gd3d_ax.text(0.5, 0.5, 'No 3D gradient descent visualization found.\nRun 3D visualization to generate one.', 
                                ha='center', va='center', transform=self.gd3d_ax.transAxes,
                                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.gd3d_ax.set_title("3D Gradient Descent Visualization")
                self.gd3d_ax.axis('off')
                
                # Update frame label to show no frames
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text="Frame: 0/0")
                
                # Safe canvas update
                try:
                    self.gd3d_canvas.draw_idle()
                except Exception as canvas_error:
                    print(f"Canvas update error in 3D tab: {canvas_error}")
                
        except Exception as e:
            print(f"Error updating 3D gradient descent tab: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Convert to numpy array for matplotlib
            img_array = np.array(img)
            
            # Clear the 3D axis and display the image
            self.gd3d_ax.clear()
            
            # Use imshow to display the image properly
            # The image array should be in the format (height, width, channels)
            if len(img_array.shape) == 3:  # RGB image
                self.gd3d_ax.imshow(img_array)
            else:  # Grayscale image
                self.gd3d_ax.imshow(img_array, cmap='gray')
            
            self.gd3d_ax.set_title(f"3D Gradient Descent Visualization\n{os.path.basename(image_path)}")
            self.gd3d_ax.axis('off')
            
            # Safe canvas update
            try:
                self.gd3d_canvas.draw_idle()
            except Exception as canvas_error:
                print(f"Canvas update error in 3D visualization: {canvas_error}")
            
            print(f"Loaded 3D visualization image: {os.path.basename(image_path)} (resized to {new_size[0]}x{new_size[1]} - {scale*100}% of original)")
            
        except Exception as e:
            print(f"Error loading 3D visualization image: {e}")
            import traceback
            traceback.print_exc()

            messagebox.showerror("Error", f"Failed to create interactive plot: {str(e)}")

    def open_live_training_plot(self):
        """Open a live training plot window using Matplotlib."""
        try:
            # Close any existing plot first
            if hasattr(self, 'live_plot_fig') and self.live_plot_fig is not None:
                try:
                    plt.close(self.live_plot_fig)
                except:
                    pass
            
            # Create a new matplotlib window with error handling
            try:
                self.live_plot_fig, self.live_plot_ax = plt.subplots(figsize=(10, 6))
            except Exception as fig_error:
                print(f"Error creating figure: {fig_error}")
                raise
            
            self.live_plot_window_open = True
            self.live_plot_epochs = []
            self.live_plot_losses = []
            
            # Set up the plot with error handling
            try:
                ticker = self.get_ticker_from_filename()
                self.live_plot_ax.set_title(f'Live Training Loss ({ticker})')
                self.live_plot_ax.set_xlabel('Epoch')
                self.live_plot_ax.set_ylabel('Loss')
                self.live_plot_ax.grid(True, alpha=0.3)
                
                # Set initial limits
                self.live_plot_ax.set_xlim(0, 10)
                self.live_plot_ax.set_ylim(0, 1)
                
            except Exception as setup_error:
                print(f"Error setting up plot: {setup_error}")
                # Continue anyway, the plot will still work
            
            # Show the plot in non-blocking mode with error handling
            try:
                plt.show(block=False)
                # Give the plot a moment to initialize
                plt.pause(0.1)
            except Exception as show_error:
                print(f"Error showing plot: {show_error}")
                # Try alternative method
                try:
                    self.live_plot_fig.show()
                except:
                    pass
            
            self.live_plot_status.config(text="Live training plot opened. Training progress will be shown here.")
            print("Live training plot opened successfully")
            
        except Exception as e:
            print(f"Error opening live training plot: {e}")
            import traceback
            traceback.print_exc()
            self.live_plot_window_open = False
            self.live_plot_fig = None
            self.live_plot_ax = None
            messagebox.showerror("Error", f"Failed to open live training plot: {str(e)}")

    def update_live_plot(self, epoch, loss):
        """Update the live matplotlib plot with new data."""
        if not self.live_plot_window_open:
            return
        
        # Rate limiting: only update if enough time has passed
        import time
        current_time = time.time()
        if current_time - self.last_plot_update < self.plot_update_interval:
            # Store the data but don't update the plot yet
            if epoch not in self.live_plot_epochs:
                self.live_plot_epochs.append(epoch)
                self.live_plot_losses.append(loss)
            return
        
        self.last_plot_update = current_time
        
        try:
            # Add new data point (if not already added)
            if epoch not in self.live_plot_epochs:
                self.live_plot_epochs.append(epoch)
                self.live_plot_losses.append(loss)
            
            # Check if the plot window is still valid
            if not hasattr(self, 'live_plot_fig') or self.live_plot_fig is None:
                print("Live plot figure is no longer valid")
                self.live_plot_window_open = False
                return
            
            # Check if the figure is still valid
            if not plt.fignum_exists(self.live_plot_fig.number):
                print("Live plot window was closed")
                self.live_plot_window_open = False
                return
            
            # Clear and redraw the plot with error handling
            try:
                self.live_plot_ax.clear()
                self.live_plot_ax.plot(self.live_plot_epochs, self.live_plot_losses, 'b-', linewidth=2, marker='o', markersize=4)
                
                # Update labels and title
                ticker = self.get_ticker_from_filename()
                self.live_plot_ax.set_title(f'Live Training Loss ({ticker})')
                self.live_plot_ax.set_xlabel('Epoch')
                self.live_plot_ax.set_ylabel('Loss')
                self.live_plot_ax.grid(True, alpha=0.3)
                
                # Auto-scale the axes with bounds checking
                if len(self.live_plot_epochs) > 1:
                    self.live_plot_ax.set_xlim(0, max(self.live_plot_epochs) + 1)
                    if len(self.live_plot_losses) > 1:
                        min_loss = min(self.live_plot_losses)
                        max_loss = max(self.live_plot_losses)
                        if max_loss > min_loss:  # Avoid division by zero
                            margin = (max_loss - min_loss) * 0.1
                            self.live_plot_ax.set_ylim(min_loss - margin, max_loss + margin)
                        else:
                            # If all losses are the same, set a small range
                            self.live_plot_ax.set_ylim(min_loss - 0.1, min_loss + 0.1)
                
            except Exception as plot_error:
                print(f"Error updating plot data: {plot_error}")
                return
            
            # Safe canvas update with multiple fallback methods
            update_success = False
            
            # Method 1: Try draw_idle()
            try:
                self.live_plot_fig.canvas.draw_idle()
                update_success = True
            except Exception as e1:
                print(f"draw_idle() failed: {e1}")
                
                # Method 2: Try flush_events()
                try:
                    self.live_plot_fig.canvas.flush_events()
                    update_success = True
                except Exception as e2:
                    print(f"flush_events() failed: {e2}")
                    
                    # Method 3: Try basic draw() with error handling
                    try:
                        self.live_plot_fig.canvas.draw()
                        update_success = True
                    except Exception as e3:
                        print(f"draw() failed: {e3}")
                        
                        # Method 4: Try to recreate the plot
                        try:
                            print("Attempting to recreate live plot...")
                            self.close_live_plot()
                            self.open_live_training_plot()
                            # Re-add all data points
                            for e, l in zip(self.live_plot_epochs, self.live_plot_losses):
                                self.update_live_plot(e, l)
                            update_success = True
                        except Exception as e4:
                            print(f"Plot recreation failed: {e4}")
                            self.live_plot_window_open = False
            
            if update_success:
                print(f"Live plot updated: Epoch {epoch}, Loss {loss:.6f}")
            else:
                print(f"Failed to update live plot for epoch {epoch}")
            
        except Exception as e:
            print(f"Error updating live plot: {e}")
            import traceback
            traceback.print_exc()
            # If we get a serious error, close the plot to prevent further issues
            try:
                self.close_live_plot()
            except:
                pass

    def close_live_plot(self):
        """Close the live training plot."""
        try:
            self.live_plot_window_open = False
            
            if hasattr(self, 'live_plot_fig') and self.live_plot_fig is not None:
                try:
                    # Check if the figure still exists
                    if plt.fignum_exists(self.live_plot_fig.number):
                        plt.close(self.live_plot_fig)
                except Exception as close_error:
                    print(f"Error closing plot: {close_error}")
                    # Try alternative closing method
                    try:
                        self.live_plot_fig.canvas.get_tk_widget().destroy()
                    except:
                        pass
            
            # Clean up variables
            self.live_plot_fig = None
            self.live_plot_ax = None
            self.live_plot_epochs = []
            self.live_plot_losses = []
            
            # Update status
            try:
                self.live_plot_status.config(text="Live training plot closed.")
            except:
                pass
                
            print("Live training plot closed")
            
        except Exception as e:
            print(f"Error closing live plot: {e}")
            # Force cleanup even if there's an error
            self.live_plot_window_open = False
            self.live_plot_fig = None
            self.live_plot_ax = None
            self.live_plot_epochs = []
            self.live_plot_losses = []

    def parse_training_output_for_loss(self, line):
        """Parse training output line to extract epoch and loss information."""
        try:
            # Look for new LOSS format: "LOSS:epoch,loss_value"
            loss_pattern = r"LOSS:(\d+),([\d.]+)"
            match = re.search(loss_pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                if self.live_plot_window_open:
                    self.root.after(0, lambda e=epoch, l=loss: self.update_live_plot(e, l))
                return epoch, loss

            # Look for new WEIGHTS format: "WEIGHTS:epoch,w1_avg,w2_avg"
            weights_pattern = r"WEIGHTS:(\d+),([\d.-]+),([\d.-]+)"
            match = re.search(weights_pattern, line)
            if match:
                epoch = int(match.group(1))
                w1_avg = float(match.group(2))
                w2_avg = float(match.group(3))
                if not hasattr(self, "live_gd_weights"):
                    self.live_gd_weights = []
                self.live_gd_weights.append([w1_avg, w2_avg])
                return epoch, None

            # Legacy patterns
            epoch_pattern = r'Epoch\s+(\d+)/(\d+),\s+Loss:\s+([\d.]+)'
            match = re.search(epoch_pattern, line)
            if match:
                epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                loss = float(match.group(3))
                if self.live_plot_window_open:
                    self.root.after(0, lambda e=epoch, l=loss: self.update_live_plot(e, l))
                return epoch, loss

            alt_pattern = r'epoch\s+(\d+)\s+loss:\s+([\d.]+)'
            match = re.search(alt_pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                if self.live_plot_window_open:
                    self.root.after(0, lambda e=epoch, l=loss: self.update_live_plot(e, l))
                return epoch, loss

            return None, None
        except Exception as e:
            print(f"Error parsing training output: {e}")
            return None, None

    def auto_make_prediction(self):
        """Automatically make a prediction and update the Prediction Results tab."""
        try:
            if not self.selected_model_path:
                print("No model selected for automatic prediction")
                return
            
            if not self.data_file:
                print("No data file selected for automatic prediction")
                return
            
            print("Making automatic prediction...")
            self.status_var.set("Making automatic prediction...")
            
            # Run prediction using predict.py, saving results in the model directory
            cmd = [
                sys.executable, 'predict.py',
                self.data_file,  # Input file as positional argument
                '--model_dir', self.selected_model_path,
                '--output_dir', self.selected_model_path  # Save in model directory
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Automatic prediction failed: {result.stderr}")
                self.status_var.set("Automatic prediction failed")
                return
            
            print("Automatic prediction completed successfully!")
            self.status_var.set("Automatic prediction completed")
            
            # Update the Prediction Results tab
            self.update_prediction_results()
            
        except Exception as e:
            print(f"Error in automatic prediction: {e}")
            self.status_var.set(f"Automatic prediction failed: {str(e)}")

    def select_data_file(self):
        """Automatically select a suitable data file if available."""
        try:
            # Look for common data files in the current directory
            data_files = []
            for file in os.listdir('.'):
                if file.endswith('.csv') and any(pattern in file.lower() for pattern in ['stock', 'price', 'data', 'tsla', 'aapl', 'msft']):
                    data_files.append(file)
            
            if data_files:
                # Use the first found data file
                selected_file = data_files[0]
                self.data_file = selected_file
                self.data_file_var.set(selected_file)
                self.load_data_features()
                self.status_var.set(f"Auto-selected data file: {selected_file}")
                return True
            else:
                self.status_var.set("No suitable data file found - please browse manually")
                return False
                
        except Exception as e:
            print(f"Error auto-selecting data file: {e}")
            self.status_var.set(f"Error selecting data file: {str(e)}")
            return False

    # 3D Animation and Control Methods
    def play_3d_animation(self):
        """Start playing the 3D animation."""
        try:
            if not hasattr(self, 'gd3d_ax') or not self.gd3d_ax:
                self.status_var.set("No 3D visualization available")
                return
            
            if not hasattr(self, 'animation_running'):
                self.animation_running = False
            
            if not self.animation_running:
                self.animation_running = True
                self.current_frame = 0
                self.animate_3d_frames()
                self.status_var.set("3D animation started")
            else:
                self.status_var.set("Animation already running")
                
        except Exception as e:
            print(f"Error starting 3D animation: {e}")
            self.status_var.set(f"Error starting animation: {str(e)}")

    def pause_3d_animation(self):
        """Pause the 3D animation."""
        try:
            if hasattr(self, 'animation_running'):
                self.animation_running = False
                self.status_var.set("3D animation paused")
            else:
                self.status_var.set("No animation running")
                
        except Exception as e:
            print(f"Error pausing 3D animation: {e}")
            self.status_var.set(f"Error pausing animation: {str(e)}")

    def stop_3d_animation(self):
        """Stop the 3D animation and reset to first frame."""
        try:
            if hasattr(self, 'animation_running'):
                self.animation_running = False
                self.current_frame = 0
                self.load_3d_frame(0)
                self.status_var.set("3D animation stopped")
            else:
                self.status_var.set("No animation running")
                
        except Exception as e:
            print(f"Error stopping 3D animation: {e}")
            self.status_var.set(f"Error stopping animation: {str(e)}")

    def animate_3d_frames(self):
        """Animate through 3D frames."""
        try:
            if not hasattr(self, 'animation_running') or not self.animation_running:
                return
            
            if not hasattr(self, 'current_frame'):
                self.current_frame = 0
            
            if not hasattr(self, 'total_frames'):
                self.total_frames = self.get_total_3d_frames()
            
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            
            # Load current frame
            self.load_3d_frame(self.current_frame)
            
            # Update frame label
            if hasattr(self, 'frame_label'):
                self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{self.total_frames}")
            
            # Schedule next frame
            delay = int(1000 / (self.anim_speed_var.get() * 10))  # Convert speed to milliseconds
            self.root.after(delay, self.animate_3d_frames)
            
            self.current_frame += 1
            
        except Exception as e:
            print(f"Error in 3D animation: {e}")
            self.animation_running = False

    def get_total_3d_frames(self):
        """Get the total number of 3D frames available."""
        try:
            if not self.selected_model_path:
                return 0
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if not os.path.exists(plots_dir):
                return 0
            
            gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
            return len(gd3d_files)
            
        except Exception as e:
            print(f"Error getting total 3D frames: {e}")
            return 0

    def load_3d_frame(self, frame_index):
        """Load a specific 3D frame."""
        try:
            if not self.selected_model_path:
                return
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if not os.path.exists(plots_dir):
                return
            
            gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
            
            if 0 <= frame_index < len(gd3d_files):
                frame_path = gd3d_files[frame_index]
                self.load_3d_visualization_image(frame_path)
                
        except Exception as e:
            print(f"Error loading 3D frame {frame_index}: {e}")

    def prev_3d_frame(self):
        """Go to previous 3D frame."""
        try:
            if not hasattr(self, 'current_frame'):
                self.current_frame = 0
            
            total_frames = self.get_total_3d_frames()
            if total_frames > 0:
                self.current_frame = (self.current_frame - 1) % total_frames
                self.load_3d_frame(self.current_frame)
                
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{total_frames}")
                
                self.status_var.set(f"Frame {self.current_frame + 1}/{total_frames}")
                
        except Exception as e:
            print(f"Error going to previous frame: {e}")
            self.status_var.set(f"Error: {str(e)}")

    def next_3d_frame(self):
        """Go to next 3D frame."""
        try:
            if not hasattr(self, 'current_frame'):
                self.current_frame = 0
            
            total_frames = self.get_total_3d_frames()
            if total_frames > 0:
                self.current_frame = (self.current_frame + 1) % total_frames
                self.load_3d_frame(self.current_frame)
                
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{total_frames}")
                
                self.status_var.set(f"Frame {self.current_frame + 1}/{total_frames}")
                
        except Exception as e:
            print(f"Error going to next frame: {e}")
            self.status_var.set(f"Error: {str(e)}")

    def reset_3d_view(self):
        """Reset the 3D view to default."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                self.current_3d_ax.view_init(elev=30, azim=45)
                
                # Update rotation sliders
                self.x_rotation_var.set(30.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(45.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D view reset")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error resetting 3D view: {e}")
            self.status_var.set(f"Error resetting view: {str(e)}")

    def set_top_view(self):
        """Set 3D view to top view."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                self.current_3d_ax.view_init(elev=90, azim=0)
                
                # Update rotation sliders
                self.x_rotation_var.set(0.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(0.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D view: Top")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error setting top view: {e}")
            self.status_var.set(f"Error setting view: {str(e)}")

    def set_side_view(self):
        """Set 3D view to side view."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                self.current_3d_ax.view_init(elev=0, azim=0)
                
                # Update rotation sliders
                self.x_rotation_var.set(0.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(0.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D view: Side")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error setting side view: {e}")
            self.status_var.set(f"Error setting view: {str(e)}")

    def set_isometric_view(self):
        """Set 3D view to isometric."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                self.current_3d_ax.view_init(elev=35.264, azim=45)
                
                # Update rotation sliders
                self.x_rotation_var.set(0.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(45.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D view: Isometric")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error setting isometric view: {e}")
            self.status_var.set(f"Error setting view: {str(e)}")

    def rotate_x(self):
        """Rotate 3D view around X axis."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                current_elev, current_azim = self.current_3d_ax.elev, self.current_3d_ax.azim
                self.current_3d_ax.view_init(elev=current_elev + 15, azim=current_azim)
                
                # Update rotation sliders
                self.x_rotation_var.set(current_elev + 15)
                self.y_rotation_var.set(current_azim)
                self.z_rotation_var.set(current_azim)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("Rotated around X axis")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error rotating X: {e}")
            self.status_var.set(f"Error rotating: {str(e)}")

    def rotate_y(self):
        """Rotate 3D view around Y axis."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                current_elev, current_azim = self.current_3d_ax.elev, self.current_3d_ax.azim
                self.current_3d_ax.view_init(elev=current_elev, azim=current_azim + 15)
                
                # Update rotation sliders
                self.x_rotation_var.set(current_elev)
                self.y_rotation_var.set(current_azim + 15)
                self.z_rotation_var.set(current_azim + 15)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("Rotated around Y axis")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error rotating Y: {e}")
            self.status_var.set(f"Error rotating: {str(e)}")

    def rotate_z(self):
        """Rotate 3D view around Z axis."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax and hasattr(self.current_3d_ax, 'view_init'):
                current_elev, current_azim = self.current_3d_ax.elev, self.current_3d_ax.azim
                self.current_3d_ax.view_init(elev=current_elev, azim=current_azim - 15)
                
                # Update rotation sliders
                self.x_rotation_var.set(current_elev)
                self.y_rotation_var.set(current_azim - 15)
                self.z_rotation_var.set(current_azim - 15)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("Rotated around Z axis")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error rotating Z: {e}")
            self.status_var.set(f"Error rotating: {str(e)}")

    def zoom_in(self):
        """Zoom in on 3D plot."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Get current limits
                xlim = self.current_3d_ax.get_xlim()
                ylim = self.current_3d_ax.get_ylim()
                zlim = self.current_3d_ax.get_zlim()
                
                # Zoom in by reducing range
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0]
                z_range = zlim[1] - zlim[0]
                
                self.current_3d_ax.set_xlim(xlim[0] + x_range * 0.1, xlim[1] - x_range * 0.1)
                self.current_3d_ax.set_ylim(ylim[0] + y_range * 0.1, ylim[1] - y_range * 0.1)
                self.current_3d_ax.set_zlim(zlim[0] + z_range * 0.1, zlim[1] - z_range * 0.1)
                
                # Update zoom slider
                current_zoom = self.zoom_var.get()
                self.zoom_var.set(min(5.0, current_zoom * 1.2))
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("Zoomed in")
            else:
                self.status_var.set("No 3D visualization available")
                
        except Exception as e:
            print(f"Error zooming in: {e}")
            self.status_var.set(f"Error zooming: {str(e)}")

    def zoom_out(self):
        """Zoom out the 3D visualization."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Get current view limits
                xlim = self.current_3d_ax.get_xlim()
                ylim = self.current_3d_ax.get_ylim()
                zlim = self.current_3d_ax.get_zlim()
                
                # Calculate new limits (zoom out by 20%)
                x_range = xlim[1] - xlim[0]
                y_range = ylim[1] - ylim[0]
                z_range = zlim[1] - zlim[0]
                
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                z_center = (zlim[0] + zlim[1]) / 2
                
                new_x_range = x_range * 1.2
                new_y_range = y_range * 1.2
                new_z_range = z_range * 1.2
                
                self.current_3d_ax.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
                self.current_3d_ax.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
                self.current_3d_ax.set_zlim(z_center - new_z_range/2, z_center + new_z_range/2)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D visualization zoomed out")
                
        except Exception as e:
            print(f"Error zooming out 3D visualization: {e}")
            self.status_var.set(f"Error zooming out: {str(e)}")

    def on_x_rotation_change(self, value):
        """Handle X rotation slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Convert slider value to radians
                angle = float(value) * np.pi / 180
                
                # Apply rotation to the 3D plot
                self.current_3d_ax.view_init(elev=self.current_3d_ax.elev, azim=angle)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set(f"X rotation: {value}°")
                
        except Exception as e:
            print(f"Error applying X rotation: {e}")

    def on_y_rotation_change(self, value):
        """Handle Y rotation slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Convert slider value to radians
                angle = float(value) * np.pi / 180
                
                # Apply rotation to the 3D plot
                self.current_3d_ax.view_init(elev=angle, azim=self.current_3d_ax.azim)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set(f"Y rotation: {value}°")
                
        except Exception as e:
            print(f"Error applying Y rotation: {e}")

    def on_z_rotation_change(self, value):
        """Handle Z rotation slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Convert slider value to radians
                angle = float(value) * np.pi / 180
                
                # Apply rotation to the 3D plot (Z rotation affects both elev and azim)
                current_elev = self.current_3d_ax.elev
                current_azim = self.current_3d_ax.azim
                
                # Apply Z rotation by adjusting both elevation and azimuth
                new_elev = current_elev * np.cos(angle) - current_azim * np.sin(angle)
                new_azim = current_elev * np.sin(angle) + current_azim * np.cos(angle)
                
                self.current_3d_ax.view_init(elev=new_elev, azim=new_azim)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set(f"Z rotation: {value}°")
                
        except Exception as e:
            print(f"Error applying Z rotation: {e}")

    def on_zoom_change(self, value):
        """Handle zoom slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                zoom_level = float(value)
                
                # Get current view limits
                xlim = self.current_3d_ax.get_xlim()
                ylim = self.current_3d_ax.get_ylim()
                zlim = self.current_3d_ax.get_zlim()
                
                # Calculate centers
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                z_center = (zlim[0] + zlim[1]) / 2
                
                # Calculate base ranges (assuming original zoom level of 1.0)
                base_x_range = (xlim[1] - xlim[0]) / zoom_level
                base_y_range = (ylim[1] - ylim[0]) / zoom_level
                base_z_range = (zlim[1] - zlim[0]) / zoom_level
                
                # Apply new zoom level
                new_x_range = base_x_range * zoom_level
                new_y_range = base_y_range * zoom_level
                new_z_range = base_z_range * zoom_level
                
                self.current_3d_ax.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
                self.current_3d_ax.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
                self.current_3d_ax.set_zlim(z_center - new_z_range/2, z_center + new_z_range/2)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set(f"Zoom level: {zoom_level:.2f}")
                
        except Exception as e:
            print(f"Error applying zoom: {e}")

    def on_camera_x_change(self, value):
        """Handle camera X position slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                camera_x = float(value)
                
                # Get current camera position
                current_pos = self.current_3d_ax.get_proj()
                
                # Update camera position (this is a simplified approach)
                # In a real implementation, you might need to use different methods
                # depending on the matplotlib version and backend
                
                self.status_var.set(f"Camera X: {camera_x}")
                
        except Exception as e:
            print(f"Error applying camera X position: {e}")

    def on_camera_y_change(self, value):
        """Handle camera Y position slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                camera_y = float(value)
                
                # Get current camera position
                current_pos = self.current_3d_ax.get_proj()
                
                # Update camera position (this is a simplified approach)
                # In a real implementation, you might need to use different methods
                # depending on the matplotlib version and backend
                
                self.status_var.set(f"Camera Y: {camera_y}")
                
        except Exception as e:
            print(f"Error applying camera Y position: {e}")

    def on_camera_z_change(self, value):
        """Handle camera Z position slider change."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                camera_z = float(value)
                
                # Get current camera position
                current_pos = self.current_3d_ax.get_proj()
                
                # Update camera position (this is a simplified approach)
                # In a real implementation, you might need to use different methods
                # depending on the matplotlib version and backend
                
                self.status_var.set(f"Camera Z: {camera_z}")
                
        except Exception as e:
            print(f"Error applying camera Z position: {e}")

    def on_frame_pos_change(self, value):
        """Handle frame position slider change."""
        try:
            frame_pos = float(value)
            
            # Calculate frame index based on percentage
            total_frames = self.get_total_3d_frames()
            if total_frames > 0:
                frame_index = int((frame_pos / 100.0) * (total_frames - 1))
                frame_index = max(0, min(frame_index, total_frames - 1))
                
                # Load the frame
                self.load_3d_frame(frame_index)
                
                # Update frame label if it exists
                if hasattr(self, 'frame_label'):
                    self.frame_label.config(text=f"Frame: {frame_index}/{total_frames-1}")
                
                self.status_var.set(f"Frame position: {frame_pos:.1f}% (Frame {frame_index})")
            else:
                self.status_var.set("No frames available")
                
        except Exception as e:
            print(f"Error changing frame position: {e}")

    def on_view_preset_change(self, event):
        """Handle view preset listbox selection change."""
        try:
            # Get the selected item
            selection = event.widget.curselection()
            if selection:
                preset_name = event.widget.get(selection[0])
                
                # Apply the selected view preset
                if preset_name == "Default":
                    self.reset_3d_view()
                elif preset_name == "Top View":
                    self.set_top_view()
                elif preset_name == "Side View":
                    self.set_side_view()
                elif preset_name == "Isometric":
                    self.set_isometric_view()
                elif preset_name == "Front View":
                    self.set_front_view()
                elif preset_name == "Back View":
                    self.set_back_view()
                
                self.status_var.set(f"View preset: {preset_name}")
                
        except Exception as e:
            print(f"Error applying view preset: {e}")

    def set_front_view(self):
        """Set front view of the 3D visualization."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Set front view (elev=0, azim=0)
                self.current_3d_ax.view_init(elev=0, azim=0)
                
                # Update rotation sliders
                self.x_rotation_var.set(0.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(0.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D visualization set to front view")
                
        except Exception as e:
            print(f"Error setting front view: {e}")
            self.status_var.set(f"Error setting front view: {str(e)}")

    def set_back_view(self):
        """Set back view of the 3D visualization."""
        try:
            if hasattr(self, 'current_3d_ax') and self.current_3d_ax:
                # Set back view (elev=0, azim=180)
                self.current_3d_ax.view_init(elev=0, azim=180)
                
                # Update rotation sliders
                self.x_rotation_var.set(0.0)
                self.y_rotation_var.set(0.0)
                self.z_rotation_var.set(180.0)
                
                # Update the canvas
                if hasattr(self, 'current_3d_canvas') and self.current_3d_canvas:
                    self.current_3d_canvas.draw()
                
                self.status_var.set("3D visualization set to back view")
                
        except Exception as e:
            print(f"Error setting back view: {e}")
            self.status_var.set(f"Error setting back view: {str(e)}")

    def on_anim_speed_change(self, value):
        """Handle animation speed slider change."""
        try:
            self.anim_speed_var.set(float(value))
            self.status_var.set(f"Animation speed: {value}x")
        except ValueError:
            print(f"Invalid animation speed format: {value}")

    def on_frame_rate_change(self, value):
        """Handle frame rate slider change."""
        try:
            self.frame_rate_var.set(int(value))
            self.status_var.set(f"Frame rate: {value} FPS")
        except ValueError:
            print(f"Invalid frame rate format: {value}")

    def on_loop_mode_change(self, event):
        """Handle loop mode listbox selection change."""
        try:
            # Get the selected item
            selection = event.widget.curselection()
            if selection:
                self.loop_mode_var.set(event.widget.get(selection[0]))
                self.status_var.set(f"Loop mode: {event.widget.get(selection[0])}")
        except Exception as e:
            print(f"Error handling loop mode change: {e}")

    def on_playback_dir_change(self, event):
        """Handle playback direction listbox selection change."""
        try:
            # Get the selected item
            selection = event.widget.curselection()
            if selection:
                self.playback_dir_var.set(event.widget.get(selection[0]))
                self.status_var.set(f"Playback direction: {event.widget.get(selection[0])}")
        except Exception as e:
            print(f"Error handling playback direction change: {e}")

    def on_autoplay_change(self):
        """Handle autoplay checkbox change."""
        try:
            self.autoplay_var.set(not self.autoplay_var.get())
            self.status_var.set(f"Auto-play: {'Enabled' if self.autoplay_var.get() else 'Disabled'}")
        except Exception as e:
            print(f"Error handling autoplay change: {e}")

    def on_anim_progress_change(self, value):
        """Handle animation progress slider change."""
        try:
            progress = float(value)
            self.anim_progress_var.set(progress)
            # Calculate frame index based on percentage
            total_frames = self.get_total_3d_frames()
            if total_frames > 0:
                frame_index = int((progress / 100.0) * (total_frames - 1))
                frame_index = max(0, min(frame_index, total_frames - 1))
                # Load the frame
                self.load_3d_frame(frame_index)
        except Exception as e:
            print(f"Error updating animation progress: {e}")

    def reset_3d_animation(self):
        """Reset 3D animation to beginning."""
        try:
            # Reset animation progress
            self.anim_progress_var.set(0.0)
            
            # Stop any ongoing animation
            if hasattr(self, 'animation_running') and self.animation_running:
                self.stop_3d_animation()
            
            # Load first frame
            self.load_3d_frame(0)
            
            # Reset animation speed to default
            self.anim_speed_var.set(1.0)
            
            # Reset frame position slider
            if hasattr(self, 'frame_pos_var'):
                self.frame_pos_var.set(0.0)
            
            self.status_var.set("3D animation reset to beginning")
            
        except Exception as e:
            print(f"Error resetting 3D animation: {e}")
            self.status_var.set(f"Error resetting animation: {str(e)}")

    def get_help_content(self):
        """Get the help content for the GUI."""
        help_text = """
STOCK PREDICTION GUI - USER MANUAL
=================================

OVERVIEW
--------
The Stock Prediction GUI is a comprehensive neural network application for stock price prediction with advanced visualization capabilities.

Key Features:
• Neural network training with real-time visualization
• 3D gradient descent visualization
• Live training plots with Plotly integration
• Comprehensive model management
• Advanced plot controls and animation
• Prediction capabilities with multiple data sources

INTERFACE OVERVIEW
------------------
The GUI consists of two main panels:

1. CONTROL PANEL (Left Side)
   • Data Selection Tab
   • Training Parameters Tab
   • Model Management Tab
   • Plot Controls Tab
   • Help Tab (?)

2. DISPLAY PANEL (Right Side)
   • Training Results Tab
   • Prediction Results Tab
   • Gradient Descent Tab
   • 3D Gradient Descent Tab
   • Saved Plots Tab
   • Live Training Plot Tab

USAGE TIPS
----------
1. Start by selecting a data file in the Data Selection tab
2. Configure training parameters in the Training Parameters tab
3. Click "Start Training" to begin model training
4. Monitor training progress in the Training Results tab
5. Use Model Management to make predictions
6. Explore 3D visualizations in the Plot Controls tab

TROUBLESHOOTING
---------------
• If training fails, check data format and feature selection
• Clear cache if experiencing display issues
• Ensure sufficient disk space for model saving
• Check console output for detailed error messages

For more detailed information, use the "Print Full Manual" button.
"""
        return help_text

    def print_full_manual(self):
        """Print the full user manual to console."""
        try:
            from STOCK_GUI_USER_MANUAL import StockGUIUserManual
            manual = StockGUIUserManual()
            manual.print_full_manual()
        except ImportError:
            print("Manual file not found. Printing basic help...")
            print(self.get_help_content())

    def open_manual_file(self):
        """Open the manual file in a dedicated window within the GUI."""
        import os
        
        manual_file = os.path.join(os.path.dirname(__file__), "STOCK_GUI_USER_MANUAL.py")
        
        if os.path.exists(manual_file):
            self.show_manual_window(manual_file)
        else:
            print(f"Manual file not found at: {manual_file}")
            # Show basic help in a window instead
            self.show_manual_window(None)

    def show_manual_window(self, manual_file_path=None):
        """Show the manual in a dedicated window."""
        # Create a new top-level window
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Stock Prediction GUI - Complete User Manual")
        manual_window.geometry("800x600")
        manual_window.configure(bg='white')
        
        # Make the window modal (user must close it before using main window)
        manual_window.transient(self.root)
        manual_window.grab_set()
        
        # Configure grid weights
        manual_window.grid_columnconfigure(0, weight=1)
        manual_window.grid_rowconfigure(1, weight=1)
        
        # Title frame
        title_frame = ttk.Frame(manual_window)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="Stock Prediction GUI - Complete User Manual", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w")
        
        # Close button
        close_btn = ttk.Button(title_frame, text="Close", command=manual_window.destroy)
        close_btn.grid(row=0, column=1, padx=(10, 0))
        
        # Text widget frame
        text_frame = ttk.Frame(manual_window)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar - improved configuration
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Arial", 11), 
                             bg="white", relief=tk.SUNKEN, padx=10, pady=10,
                             state=tk.NORMAL)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Get content
        if manual_file_path:
            try:
                # Try to load the actual manual file
                with open(manual_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract the docstring content from the manual file
                import re
                docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if docstring_match:
                    manual_content = docstring_match.group(1).strip()
                else:
                    manual_content = content
                    
            except Exception as e:
                print(f"Error reading manual file: {e}")
                manual_content = self.get_full_manual_content()
        else:
            manual_content = self.get_full_manual_content()
        
        # Insert content - make sure it's not empty
        if manual_content.strip():
            text_widget.insert(tk.END, manual_content)
            print(f"✅ Manual window: Inserted {len(manual_content)} characters of manual content")
        else:
            # Fallback content
            fallback_content = """
STOCK PREDICTION GUI - USER MANUAL
=================================

OVERVIEW
--------
The Stock Prediction GUI is a comprehensive neural network application for stock price prediction with advanced visualization capabilities.

Key Features:
• Neural network training with real-time visualization
• 3D gradient descent visualization
• Live training plots with Plotly integration
• Comprehensive model management
• Advanced plot controls and animation
• Prediction capabilities with multiple data sources

INTERFACE OVERVIEW
------------------
The GUI consists of two main panels:

1. CONTROL PANEL (Left Side)
   • Data Selection Tab
   • Training Parameters Tab
   • Model Management Tab
   • Plot Controls Tab
   • Help Tab (?)

2. DISPLAY PANEL (Right Side)
   • Training Results Tab
   • Prediction Results Tab
   • Gradient Descent Tab
   • 3D Gradient Descent Tab
   • Saved Plots Tab
   • Live Training Plot Tab

USAGE TIPS
----------
1. Start by selecting a data file in the Data Selection tab
2. Configure training parameters in the Training Parameters tab
3. Click "Start Training" to begin model training
4. Monitor training progress in the Training Results tab
5. Use Model Management to make predictions
6. Explore 3D visualizations in the Plot Controls tab

TROUBLESHOOTING
---------------
• If training fails, check data format and feature selection
• Clear cache if experiencing display issues
• Ensure sufficient disk space for model saving
• Check console output for detailed error messages

For more detailed information, use the "Print Full Manual" button.
"""
            text_widget.insert(tk.END, fallback_content)
            print("✅ Manual window: Inserted fallback manual content")
        
        # Make read-only after inserting content
        text_widget.config(state=tk.DISABLED)
        
        # Force update to ensure content is displayed
        text_widget.update()
        
        # Focus on the window
        manual_window.focus_set()
        
        # Center the window on screen
        manual_window.update_idletasks()
        x = (manual_window.winfo_screenwidth() // 2) - (800 // 2)
        y = (manual_window.winfo_screenheight() // 2) - (600 // 2)
        manual_window.geometry(f"800x600+{x}+{y}")
        
        # Force update to ensure content is displayed
        manual_window.update()

    def get_full_manual_content(self):
        """Get the full manual content as a formatted string."""
        return """
STOCK PREDICTION GUI - COMPREHENSIVE USER MANUAL
===============================================

OVERVIEW
--------
The Stock Prediction GUI is a comprehensive neural network application for stock price prediction with advanced visualization capabilities.

Key Features:
• Neural network training with real-time visualization
• 3D gradient descent visualization
• Live training plots with Plotly integration
• Comprehensive model management
• Advanced plot controls and animation
• Prediction capabilities with multiple data sources

INSTALLATION
------------
1. Ensure Python 3.8+ is installed
2. Install required dependencies:
   pip install tkinter matplotlib numpy pandas scikit-learn plotly
3. Navigate to the simple directory:
   cd /path/to/neural_net/simple
4. Run the GUI:
   python stock_gui.py

INTERFACE OVERVIEW
------------------
The GUI consists of two main panels:

1. CONTROL PANEL (Left Side)
   • Data Selection Tab
   • Training Parameters Tab
   • Model Management Tab
   • Plot Controls Tab
   • Help Tab (?)

2. DISPLAY PANEL (Right Side)
   • Training Results Tab
   • Prediction Results Tab
   • Gradient Descent Tab
   • 3D Gradient Descent Tab
   • Saved Plots Tab
   • Live Training Plot Tab

DETAILED TAB GUIDE
==================

DATA SELECTION TAB
-----------------
Location: Control Panel → Data Selection

Features:
• Data File Selection: Browse and select CSV files with stock data
• Feature Selection: Multi-select listbox for choosing input features
• Target Feature: Dropdown to select the target variable (default: close)
• Output Directory: Choose where to save results

Data File Requirements:
• CSV format with columns like: open, high, low, close, volume
• Or generic feature columns: feature_1, feature_2, etc.
• Minimum 100 rows recommended for training

Feature Selection:
• Lock Selected: Permanently select specific features
• Unlock All: Clear feature selection
• Target Feature: The variable to predict (usually 'close' price)

TRAINING PARAMETERS TAB
----------------------
Location: Control Panel → Training Parameters

Model Configuration:
• Hidden Layer Size: 4-128 neurons (default: 32)
• Learning Rate: 0.001-0.1 (default: 0.01)
• Batch Size: 1-512 (default: 32)

Training Process:
• Start Training: Begin model training process
• Live Training Plot: Open real-time training visualization
• Clear Cache: Clear cached data and plots

Training Tips:
• Start with default parameters for first training
• Adjust hidden layer size based on data complexity
• Lower learning rate for more stable training
• Larger batch sizes for faster training

MODEL MANAGEMENT TAB
-------------------
Location: Control Panel → Model Management

Model Operations:
• Select Model: Choose from trained models in dropdown
• Refresh Models: Update the model list
• Make Prediction: Generate predictions with selected model

Prediction Files:
• View saved prediction results
• Refresh prediction files list
• View detailed prediction results

3D Visualization:
• Create 3D Visualization: Generate 3D gradient descent plots
• Requires a trained model with saved weights

PLOT CONTROLS TAB
-----------------
Location: Control Panel → Plot Controls

3D Visualization Parameters:
• View Preset: Default, Top View, Side View, Isometric, Front View, Back View
• Rotation X: -180° to 180° (default: 0°)
• Rotation Y: -180° to 180° (default: 0°)
• Rotation Z: -180° to 180° (default: 0°)
• Zoom Level: 0.1x to 5.0x (default: 1.0x)
• Camera Position: Adjust 3D camera view (X, Y, Z coordinates)

Animation Controls:
• Animation Speed: 0.1x to 5.0x (default: 1.0x)
• Frame Rate: 1 to 60 FPS (default: 30 FPS)
• Loop Mode: Loop, Once, Ping-Pong (default: Loop)
• Playback Direction: Forward, Reverse (default: Forward)
• Auto-play: Enable/disable automatic animation (default: Disabled)
• Animation Progress: 0% to 100% (default: 0%)

Animation Buttons:
• Play: Start/resume animation
• Pause: Pause animation
• Stop: Stop and reset animation
• Reset: Reset to beginning

Visualization Parameters:
• Color Map: Choose from various matplotlib colormaps
• Point Size: Adjust size of 3D points
• Line Width: Adjust width of 3D lines
• Surface Alpha: Transparency of 3D surfaces
• Grid Points: Number of points for 3D grid
• Output Resolution: Width and height for saved images

DISPLAY PANEL TABS
==================

TRAINING RESULTS TAB
-------------------
• Real-time training progress display
• Loss curve visualization
• Training metrics and statistics
• Matplotlib toolbar for plot interaction

PREDICTION RESULTS TAB
---------------------
• Display prediction results
• Actual vs Predicted plots
• Prediction accuracy metrics
• Export functionality

GRADIENT DESCENT TAB
-------------------
• 2D gradient descent visualization
• Training path visualization
• Loss surface plots
• Interactive plot controls

3D GRADIENT DESCENT TAB
----------------------
• 3D visualization of gradient descent
• Interactive 3D controls
• Animation playback
• Multiple view presets

SAVED PLOTS TAB
--------------
• Browse and view saved plots
• Plot management
• Export functionality

LIVE TRAINING PLOT TAB
---------------------
• Real-time training visualization
• Live loss curve updates
• Training progress monitoring

USAGE WORKFLOW
==============

Step 1: Data Preparation
• Select a data file in the Data Selection tab
• Choose appropriate features for training
• Set the target feature (usually 'close' price)
• Select an output directory

Step 2: Model Configuration
• Configure training parameters in Training Parameters tab
• Start with default values for first training
• Adjust parameters based on results

Step 3: Training
• Click "Start Training" to begin
• Monitor progress in Training Results tab
• Use Live Training Plot for real-time monitoring

Step 4: Model Management
• Select trained model in Model Management tab
• Make predictions on new data
• View prediction results and accuracy

Step 5: Visualization
• Create 3D visualizations in Plot Controls tab
• Explore different view angles and animations
• Save plots for later use

TROUBLESHOOTING
===============

Common Issues:

1. Training Fails
   • Check data format and feature selection
   • Ensure sufficient data (minimum 100 rows)
   • Verify target feature is numeric
   • Check console output for error messages

2. Display Issues
   • Clear cache using "Clear Cache" button
   • Restart the application if needed
   • Check available system memory

3. 3D Visualization Problems
   • Ensure model has been trained successfully
   • Check that 3D visualization files exist
   • Try different view presets

4. Performance Issues
   • Reduce batch size for large datasets
   • Use smaller hidden layer sizes
   • Clear cache regularly

5. File Not Found Errors
   • Verify file paths are correct
   • Check file permissions
   • Ensure files are in the expected format

KEYBOARD SHORTCUTS
==================
• Ctrl+O: Open data file
• Ctrl+S: Save model
• Ctrl+P: Make prediction
• Ctrl+Q: Quit application

ADVANCED FEATURES
=================

Custom Optimizers:
• Support for custom optimization algorithms
• Integration with custom_optimizers module
• Dynamic optimizer loading

Technical Indicators:
• Automatic calculation of technical indicators
• RSI, Moving Averages, Bollinger Bands
• Custom indicator support

Data Conversion:
• Automatic data format detection
• OHLCV format conversion
• Support for various data sources

Model Persistence:
• Automatic model saving
• Training history preservation
• Scalable model storage

LIMITATIONS
===========
• Requires Python 3.8 or higher
• Limited to tabular data (CSV format)
• Single target variable prediction
• No real-time data streaming
• Memory usage scales with dataset size

FUTURE ENHANCEMENTS
===================
• Support for multiple target variables
• Real-time data integration
• Advanced model architectures
• Cloud deployment options
• Enhanced visualization options

CONTACT AND SUPPORT
==================
For issues and questions:
• Check the console output for error messages
• Review this manual for troubleshooting
• Ensure all dependencies are properly installed
• Verify data format and file paths

Version: 1.0
Last Updated: 2025-01-27
"""

    def format_manual_text(self, text_widget):
        """Apply formatting tags to the manual text."""
        content = text_widget.get("1.0", tk.END)
        
        # Find and tag titles (lines with =)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().endswith('=') and len(line.strip()) > 3:
                start = f"{i+1}.0"
                end = f"{i+1}.end"
                text_widget.tag_add("title", start, end)
        
        # Find and tag section headers (lines with -)
        for i, line in enumerate(lines):
            if line.strip().endswith('-') and len(line.strip()) > 3:
                start = f"{i+1}.0"
                end = f"{i+1}.end"
                text_widget.tag_add("section", start, end)
        
        # Find and tag subsections (lines with •)
        for i, line in enumerate(lines):
            if line.strip().startswith('•'):
                start = f"{i+1}.0"
                end = f"{i+1}.end"
                text_widget.tag_add("subsection", start, end)

    def refresh_color_combobox(self):
        """Refresh the color combobox to ensure it displays the current value."""
        try:
            if hasattr(self, 'color_combo') and self.color_combo:
                current_color = self.color_var.get()
                valid_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'jet', 'hot', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter']
                
                if current_color and current_color in valid_colormaps:
                    self.color_combo.set(current_color)
                    print(f"🔄 Refreshed color combobox to: '{current_color}'")
                else:
                    self.color_combo.set('viridis')
                    self.color_var.set('viridis')
                    print(f"🔄 Reset color combobox to default: 'viridis'")
                
                # Force update
                self.color_combo.update()
                return True
        except Exception as e:
            print(f"❌ Error refreshing color combobox: {e}")
            return False

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

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check for test mode flag
    test_mode = "--test_mode" in sys.argv
    
    root = tk.Tk()
    app = StockPredictionGUI(root)
    
    if test_mode:
        print("🧪 Running in TEST MODE - Live updates disabled")
        app.status_var.set("TEST MODE - Live updates disabled")
    
    root.mainloop()
        