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
import concurrent.futures
import importlib.util
import time
from PIL import Image, ImageTk
import path_utils
import script_launcher

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
        self.root.geometry("1200x800")
        
        # Configure root window style
        self.root.configure(bg=BACKGROUND_COLOR)
        
        # Initialize style
        self.style = ttk.Style()
        self.style.theme_use("default")  # Use default theme for better control
        
        # Configure style for all widgets
        self.style.configure("TLabel", 
                           background=BACKGROUND_COLOR,
                           foreground=TEXT_COLOR)
        self.style.configure("TFrame",
                           background=FRAME_COLOR)
        
        # Initialize image cache for plots (optimization)
        self.plot_image_cache = {}  # Cache for resized plot images
        self.max_cache_size = 50  # Maximum number of cached images
        
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
        
        # Initialize model directory and create it if it doesn't exist
        os.makedirs(self.current_model_dir, exist_ok=True)
        
        # Initialize feature variables
        self.feature_list = []  # Available features from data file
        
        # Initialize training parameters
        self.hidden_size_var = tk.StringVar(value="4")  # Default hidden layer size
        self.learning_rate_var = tk.StringVar(value="0.001")  # Default learning rate
        self.batch_size_var = tk.StringVar(value="32")  # Default batch size
        self.patience_var = tk.StringVar(value="20")
        
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
        
        # Create canvas and embed in the tab
        self.results_canvas = FigureCanvasTkAgg(self.results_fig, train_results_frame)
        self.results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.results_canvas, train_results_frame)
        toolbar.update()
        
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
        self.pred_fig = plt.Figure(figsize=(5, 4), dpi=100)  # Smaller size for results panel
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=pred_results_frame)
        self.pred_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Gradient Descent Tab (now used for live training)
        gd_frame = ttk.Frame(self.display_notebook)
        self.display_notebook.add(gd_frame, text="Live Training")
        gd_frame.grid_columnconfigure(0, weight=1)
        gd_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for gradient descent visualization (2D to avoid matplotlib issues)
        self.gd_fig = plt.Figure(figsize=(8, 6))
        self.gd_ax = self.gd_fig.add_subplot(111)  # Remove 3D projection
        
        # Create canvas and embed in the tab
        self.gd_canvas = FigureCanvasTkAgg(self.gd_fig, gd_frame)
        self.gd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.gd_canvas, gd_frame)
        toolbar.update()
        
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
        
        # Create canvas and embed in the tab
        self.plots_canvas = FigureCanvasTkAgg(self.plots_fig, plots_frame)
        self.plots_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.plots_canvas, plots_frame)
        toolbar.update()
        
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
        
        # Create canvas and embed in the tab
        self.gd3d_canvas = FigureCanvasTkAgg(self.gd3d_fig, gd3d_frame)
        self.gd3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.gd3d_canvas, gd3d_frame)
        toolbar.update()
        
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
        
        # Create scrollable canvas for images
        self.saved_plots_canvas = tk.Canvas(saved_plots_frame, bg=FRAME_COLOR)
        self.saved_plots_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(saved_plots_frame, orient="vertical", command=self.saved_plots_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.saved_plots_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas to hold images
        self.saved_plots_inner_frame = ttk.Frame(self.saved_plots_canvas)
        self.saved_plots_canvas.create_window((0, 0), window=self.saved_plots_inner_frame, anchor="nw")
        
        # Placeholder label
        self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                text="Select a model to view saved plots", 
                                                foreground=TEXT_COLOR, background=FRAME_COLOR)
        self.saved_plots_placeholder.pack(pady=20)
        
        # Bind canvas resizing
        self.saved_plots_inner_frame.bind("<Configure>", lambda e: self.saved_plots_canvas.configure(
            scrollregion=self.saved_plots_canvas.bbox("all")))
        
        print("Saved Plots tab created successfully")
        
        # Bind tab selection events
        self.display_notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        print(f"Display panel created with {self.display_notebook.index('end')} tabs")

    def on_tab_changed(self, event):
        """Handle tab selection changes"""
        try:
            current_tab = self.display_notebook.select()
            tab_index = self.display_notebook.index(current_tab)
            tab_text = self.display_notebook.tab(tab_index, "text")
            
            # Auto-refresh 3D Gradient Descent tab when selected
            if tab_text == "3D Gradient Descent":
                self.update_3d_gradient_descent_tab()
                if self.has_3d_visualization():
                    self.status_var.set("3D Gradient Descent visualization loaded")
                else:
                    self.status_var.set("No 3D visualization found - Use 'Generate 3D Gradient Descent' button")
            
            # Auto-refresh Saved Plots tab when selected
            elif tab_text == "Saved Plots":
                self.load_saved_plots()
                
        except Exception as e:
            print(f"Error in tab change handler: {e}")

    def browse_data_file(self):
        """Open file dialog to select data file"""
        filename = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.data_file) if self.data_file else os.path.dirname(os.path.abspath(__file__)),
            title="Select Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            # Store full path internally
            self.data_file = os.path.abspath(filename)
            
            # Update UI display
            self.data_file_var.set(os.path.basename(filename))
            self.status_var.set(f"Selected data file: {os.path.basename(filename)}")
            
            # Clear existing preprocessed data
            if hasattr(self, 'preprocessed_df'):
                delattr(self, 'preprocessed_df')
            
            # Clear existing features
            self.x_features_listbox.delete(0, tk.END)
            self.y_features_combo.set("")
            
            # Load features from CSV file
            self.load_features()

    def load_features(self):
        """Load features from the selected data file"""
        try:
            if not self.data_file:
                messagebox.showerror("Error", "Please select a data file first")
                return
            
            # Load the CSV file using the full path stored in self.data_file
            df = pd.read_csv(self.data_file)
            
            if df.empty:
                messagebox.showerror("Error", "The selected file is empty")
                self.status_var.set("Error: Empty file")
                return
            
            # Define expected column types
            expected_numeric_columns = ['open', 'high', 'low', 'close', 'vol', 'openint']
            expected_text_columns = ['ticker', 'timestamp', 'format']
            
            # Find which columns actually exist in the dataframe
            existing_numeric_columns = [col for col in expected_numeric_columns if col in df.columns]
            existing_text_columns = [col for col in expected_text_columns if col in df.columns]
            
            # Log what columns were found
            print(f"Found numeric columns: {existing_numeric_columns}")
            print(f"Found text columns: {existing_text_columns}")
            print(f"All columns in file: {list(df.columns)}")
            
            # Convert only existing numeric columns to float
            for col in existing_numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Only drop rows with NaN in existing numeric columns (if any exist)
            if existing_numeric_columns:
                initial_rows = len(df)
                df = df.dropna(subset=existing_numeric_columns)
                final_rows = len(df)
                print(f"Dropped {initial_rows - final_rows} rows with NaN values in numeric columns")
            else:
                print("No numeric columns found, skipping NaN removal")
            
            if df.empty:
                messagebox.showerror("Error", "No valid numeric data found in the file")
                self.status_var.set("Error: No valid numeric data")
                return
            
            # Store the preprocessed dataframe for later use
            self.preprocessed_df = df.copy()
            
            # Get cleaned feature list
            self.feature_list = df.columns.tolist()
            
            # Filter out any remaining empty strings
            self.feature_list = [f for f in self.feature_list if f and f.strip()]
            
            if not self.feature_list:
                messagebox.showerror("Error", "No valid features found in the file")
                self.status_var.set("Error: No valid features")
                return
            
            # Clear existing features from UI
            self.x_features_listbox.delete(0, tk.END)
            self.y_features_combo['values'] = []
            self.y_features_combo.set("")
            
            # Update X features listbox
            for feature in self.feature_list:
                self.x_features_listbox.insert(tk.END, feature)
            
            # Update Y features combobox
            self.y_features_combo['values'] = self.feature_list
            
            # Set default Y feature if available
            if 'close' in self.feature_list:
                self.y_features_combo.set('close')
            elif len(self.feature_list) > 0:
                self.y_features_combo.set(self.feature_list[0])
            
            self.status_var.set(f"Features loaded successfully: {len(self.feature_list)} features")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load features: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            print("Error occurred")
            print(f"Error: {e}")

    def validate_features(self):
        """Validate the selected features"""
        try:
            # Get X features - handle both locked and unlocked states
            if self.features_locked and hasattr(self, 'locked_features') and self.locked_features:
                # Use locked features
                x_indices = self.locked_features
                x_features = [self.x_features_listbox.get(i) for i in x_indices if i < self.x_features_listbox.size()]
            else:
                # Get selected features from UI
                x_indices = self.x_features_listbox.curselection()
                x_features = [self.x_features_listbox.get(i) for i in x_indices]
            
            # Get Y feature
            y_feature = self.y_features_combo.get()
            
            # Clear previous error message
            self.feature_status_var.set("")
            
            # Don't show error if no features are selected yet
            if not x_features and not y_feature:
                return True
                
            # Validate features only if both X and Y are selected
            if x_features and y_feature:
                # Check if Y feature is also selected as an X feature
                if y_feature in x_features:
                    self.feature_status_var.set("Error: Target feature cannot be used as an input feature")
                    return False
                    
                # If we get here, features are valid - store them
                self.x_features = x_features
                self.y_feature = y_feature
                self.feature_status_var.set(f"Features are valid! X: {len(x_features)}, Y: {y_feature}")
                return True
                
            return True
            
        except Exception as e:
            self.feature_status_var.set(f"Validation error: {str(e)}")
            return False

    def create_control_panel(self):
        """Create the left control panel (now using grid for all widgets)"""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(control_frame)
        notebook.grid(row=0, column=0, sticky="nsew")

        # Data Selection Tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data")
        for i in range(8):
            data_frame.grid_rowconfigure(i, weight=0)
        data_frame.grid_columnconfigure(0, weight=1)

        # Data File Selection
        ttk.Label(data_frame, text="Data File:").grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Entry(data_frame, textvariable=self.data_file_var).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(data_frame, text="Browse", command=self.browse_data_file).grid(row=2, column=0, sticky="ew", pady=2)

        # Feature Selection
        ttk.Label(data_frame, text="Input Features (X):", style="Bold.TLabel").grid(row=3, column=0, sticky="ew", pady=2)
        x_features_frame = ttk.Frame(data_frame)
        x_features_frame.grid(row=4, column=0, sticky="ew", pady=2)
        x_features_frame.grid_columnconfigure(0, weight=1)
        x_features_frame.grid_columnconfigure(1, weight=0)
        x_features_frame.grid_columnconfigure(2, weight=0)
        x_features_frame.grid_columnconfigure(3, weight=0)
        self.x_features_count_label = ttk.Label(x_features_frame, text="0 features selected")
        self.x_features_count_label.grid(row=0, column=0, sticky="w", padx=5)
        ttk.Button(x_features_frame, text="Select All", command=self.select_all_features).grid(row=0, column=3, sticky="e", padx=5)
        ttk.Button(x_features_frame, text="Clear", command=self.clear_feature_selection).grid(row=0, column=2, sticky="e", padx=5)
        self.lock_button = ttk.Button(x_features_frame, text="Lock", command=self.toggle_lock_features)
        self.lock_button.grid(row=0, column=1, sticky="e", padx=5)
        self.x_features_listbox = tk.Listbox(data_frame, selectmode=tk.MULTIPLE, height=10)
        self.x_features_listbox.grid(row=5, column=0, sticky="ew", pady=2)
        self.x_features_listbox.bind('<<ListboxSelect>>', self.on_feature_selection_change)
        self.x_features_listbox.bind('<ButtonRelease-1>', self.on_feature_selection_change)
        self.x_features_listbox.bind("<Enter>", lambda e: self.show_tooltip("Select multiple features by holding Ctrl/Cmd and clicking"))
        self.x_features_listbox.bind("<Leave>", lambda e: self.hide_tooltip())
        ttk.Label(data_frame, text="Target Feature (Y):", style="Bold.TLabel").grid(row=6, column=0, sticky="ew", pady=2)
        self.y_features_combo = ttk.Combobox(data_frame)
        self.y_features_combo.grid(row=7, column=0, sticky="ew", pady=2)
        ttk.Label(data_frame, textvariable=self.feature_status_var, foreground="red").grid(row=8, column=0, sticky="ew", pady=2)

        # Model Selection Tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model")
        for i in range(10):
            model_frame.grid_rowconfigure(i, weight=0)
        model_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(model_frame, text="Model Directory:").grid(row=0, column=0, sticky="ew", pady=2)
        self.model_dir_var = tk.StringVar(value=self.current_model_dir)
        ttk.Entry(model_frame, textvariable=self.model_dir_var).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(model_frame, text="Browse", command=self.browse_model_dir).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Label(model_frame, text="Available Models:").grid(row=3, column=0, sticky="ew", pady=2)
        model_list_frame = ttk.Frame(model_frame)
        model_list_frame.grid(row=4, column=0, sticky="ew", pady=2)
        model_list_frame.grid_columnconfigure(0, weight=1)
        model_list_frame.grid_columnconfigure(1, weight=0)
        self.model_listbox = tk.Listbox(model_list_frame, height=10)
        self.model_listbox.grid(row=0, column=0, sticky="ew")
        self.model_listbox.bind("<<ListboxSelect>>", self.on_model_select)
        scrollbar = ttk.Scrollbar(model_list_frame, orient="vertical", command=self.model_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.model_listbox.configure(yscrollcommand=scrollbar.set)
        delete_button = ttk.Button(model_list_frame, text="Delete Selected", command=self.delete_model)
        delete_button.grid(row=0, column=2, sticky="e", padx=5)
        ttk.Label(model_frame, text="Saved Predictions:").grid(row=5, column=0, sticky="ew", pady=(10,2))
        self.predictions_var = tk.StringVar()
        self.predictions_combo = ttk.Combobox(model_frame, textvariable=self.predictions_var, state="readonly")
        self.predictions_combo.grid(row=6, column=0, sticky="ew", pady=2)
        self.predictions_combo.bind("<<ComboboxSelected>>", self.on_prediction_select)
        ttk.Button(model_frame, text="Open Prediction File", command=self.open_selected_prediction).grid(row=7, column=0, sticky="ew", pady=2)
        self.refresh_predictions()

        # Training Parameters Tab
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="Training")
        for i in range(10):
            train_frame.grid_rowconfigure(i, weight=0)
        train_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(train_frame, text="Hidden Layer Size:").grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Entry(train_frame, textvariable=self.hidden_size_var).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Label(train_frame, text="Learning Rate:").grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Entry(train_frame, textvariable=self.learning_rate_var).grid(row=3, column=0, sticky="ew", pady=2)
        ttk.Label(train_frame, text="Batch Size:").grid(row=4, column=0, sticky="ew", pady=2)
        ttk.Entry(train_frame, textvariable=self.batch_size_var).grid(row=5, column=0, sticky="ew", pady=2)
        ttk.Label(train_frame, text="Patience Epoch:").grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Entry(train_frame, textvariable=self.patience_var).grid(row=7, column=0, sticky="ew", pady=2)
        ttk.Label(train_frame, text="Training Progress:").grid(row=8, column=0, sticky="ew", pady=2)
        
        # Add progress bar
        progress_frame = ttk.Frame(train_frame)
        progress_frame.grid(row=9, column=0, sticky="ew", pady=2)
        progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=2)
        
        # Progress text area
        text_frame = ttk.Frame(train_frame)
        text_frame.grid(row=10, column=0, sticky="nsew", pady=2)
        text_frame.grid_columnconfigure(0, weight=1)
        text_frame.grid_rowconfigure(0, weight=1)
        self.progress_text = tk.Text(text_frame, height=10, wrap=tk.WORD)
        self.progress_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.progress_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.progress_text.configure(yscrollcommand=scrollbar.set)
        self.progress_text.configure(state="disabled")
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=1, column=0, sticky="ew", pady=10)
        self.train_button = ttk.Button(action_frame, text="Train Model", command=self.train_model)
        self.train_button.grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Make Prediction", command=self.make_prediction).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="View Results", command=self._view_training_results).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Generate 3D Gradient Descent", command=self.trigger_gradient_descent_visualization).grid(row=3, column=0, sticky="ew", pady=2)
        self.stop_button = ttk.Button(action_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=4, column=0, sticky="ew", pady=2)
        
        # Cache management button
        cache_frame = ttk.Frame(action_frame)
        cache_frame.grid(row=5, column=0, sticky="ew", pady=2)
        cache_frame.grid_columnconfigure(0, weight=1)
        cache_frame.grid_columnconfigure(1, weight=0)
        ttk.Button(cache_frame, text="Clear Plot Cache", command=self.clear_plot_cache).grid(row=0, column=0, sticky="ew", pady=2)
        self.cache_info_label = ttk.Label(cache_frame, text="Cache: 0 images", foreground="blue")
        self.cache_info_label.grid(row=0, column=1, sticky="e", padx=5)
        
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="ew")

    def on_feature_selection_change(self, event=None):
        """Update feature count and validation when selection changes"""
        if not self.features_locked:
            self.update_feature_count()
            self.validate_features()
        else:
            # Restore locked selections
            self.x_features_listbox.selection_clear(0, tk.END)
            for index in self.locked_features:
                self.x_features_listbox.selection_set(index)

    def update_feature_count(self):
        """Update the feature count label based on current selection."""
        selected_count = len(self.x_features_listbox.curselection())
        self.x_features_count_label.config(text=f"{selected_count} feature(s) selected")
        self.status_var.set(f"{selected_count} feature(s) selected")

    def select_all_features(self):
        """Select all features in the listbox."""
        self.x_features_listbox.selection_set(0, tk.END)
        self.update_feature_count()
        self.validate_features()

    def clear_feature_selection(self):
        """Clear all selected features."""
        self.x_features_listbox.selection_clear(0, tk.END)
        self.update_feature_count()
        self.validate_features()

    def make_prediction(self):
        """Make predictions using the selected model and save plots"""
        if not self.selected_model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        if not self.data_file:
            messagebox.showerror("Error", "Please select a data file first")
            return
        
        try:
            # Get selected features
            if self.features_locked and hasattr(self, 'locked_features') and self.locked_features:
                x_indices = self.locked_features
                x_features = [self.x_features_listbox.get(i) for i in x_indices]
            else:
                x_indices = self.x_features_listbox.curselection()
                if not x_indices:
                    messagebox.showerror("Error", "Please select input features")
                    return
                x_features = [self.x_features_listbox.get(i) for i in x_indices]
            
            y_feature = self.y_features_combo.get()
            if not y_feature:
                messagebox.showerror("Error", "Please select a target feature")
                return
            
            self.status_var.set("Making predictions...")
            self.root.update()
            
            # Use script launcher to run prediction
            success, stdout, stderr = script_launcher.launch_prediction(
                data_file=self.data_file,
                model_dir=self.selected_model_path,
                x_features=x_features,
                y_feature=y_feature
            )
            
            if success:
                # Refresh the saved plots and predictions
                self.load_saved_plots()
                self.refresh_predictions()
                
                # Switch to Saved Plots tab to show the new prediction plot
                self.switch_to_tab(1)
                
                self.status_var.set("Prediction completed successfully")
                messagebox.showinfo("Success", 
                                  "Prediction completed successfully!\n"
                                  "Check the 'Saved Plots' tab to view the results.")
            else:
                error_msg = stderr if stderr else stdout
                messagebox.showerror("Error", f"Prediction failed:\n{error_msg}")
                self.status_var.set("Prediction failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")
            self.status_var.set("Error in prediction")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _view_training_results(self):
        """View training results and plots"""
        if not self.selected_model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
            
        try:
            # Load and display loss curves
            loss_file = os.path.join(self.selected_model_path, "training_losses.csv")
            if os.path.exists(loss_file):
                losses = np.loadtxt(loss_file, delimiter=',')
                
                # Display in the training results tab
                self.results_ax.clear()
                self.results_ax.plot(losses, label='Training Loss')
                self.results_ax.set_title("Training Loss Over Time")
                self.results_ax.set_xlabel("Epoch")
                self.results_ax.set_ylabel("MSE")
                self.results_ax.legend()
                self.results_ax.grid(True)
                self.results_canvas.draw()
                
                # Switch to training results tab (index 0)
                self.switch_to_tab(0)
                
                self.status_var.set("Training results displayed")
            else:
                messagebox.showwarning("Warning", "No training results found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view results: {str(e)}")

    def _load_model_info(self):
        """Load and display information about the selected model"""
        if not self.selected_model_path:
            return
            
        # Load feature info
        feature_info_path = os.path.join(self.selected_model_path, 'feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                self.x_features = feature_info['x_features']
                self.y_feature = feature_info['y_feature']
                
                # Update feature selection in GUI
                if self.x_features_listbox:
                    self.x_features_listbox.delete(0, tk.END)
                    for feature in self.x_features:
                        self.x_features_listbox.insert(tk.END, feature)
                if self.y_features_combo:
                    self.y_features_combo.set(self.y_feature)

    def on_model_select(self, event):
        """Handle model selection from listbox"""
        selection = self.model_listbox.curselection()
        if selection:
            model_name = self.model_listbox.get(selection[0])
            
            # Find the actual model directory using path utilities
            model_dir = path_utils.find_model_directory(model_name)
            if model_dir:
                self.selected_model_path = model_dir
            else:
                # Fallback to current directory
                self.selected_model_path = os.path.join(self.current_model_dir, model_name)
            
            # Load feature info from selected model
            feature_info_path = os.path.join(self.selected_model_path, 'feature_info.json')
            if os.path.exists(feature_info_path):
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.x_features = feature_info['x_features']
                    self.y_feature = feature_info['y_feature']
                    
                    # Update feature selection in GUI
                    self.x_features_listbox.delete(0, tk.END)
                    for feature in self.x_features:
                        self.x_features_listbox.insert(tk.END, feature)
                    self.y_features_combo.set(self.y_feature)
                    
                    self.status_var.set(f"Loaded features: X={self.x_features}, Y={self.y_feature}")
            else:
                self.status_var.set("No feature info found for selected model")
            
            # Load saved plots
            self.load_saved_plots()
            
            # Update 3D Gradient Descent tab with visualization if available
            self.update_3d_gradient_descent_tab()
            
            # Check if 3D visualization exists and show notification
            if self.has_3d_visualization():
                self.status_var.set(f"Model loaded with 3D visualization available - Check 3D Gradient Descent tab")
            else:
                self.status_var.set(f"Model loaded - Use 'Create 3D Visualization' button to generate 3D plots")
        else:
            self.selected_model_path = None
            self.load_saved_plots()  # Clear plots
            # Clear 3D tab
            self.gd3d_ax.clear()
            self.gd3d_ax.text(0.5, 0.5, 0.5, 'Select a model to view 3D gradient descent visualization', 
                             ha='center', va='center', transform=self.gd3d_ax.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            self.gd3d_ax.set_title("3D Gradient Descent Visualization")
            self.gd3d_canvas.draw()

    def has_3d_visualization(self):
        """Check if the selected model has 3D gradient descent visualization files."""
        if not self.selected_model_path:
            return False
        
        plots_dir = os.path.join(self.selected_model_path, 'plots')
        if not os.path.exists(plots_dir):
            return False
        
        # Check for gradient descent 3D visualization files
        gd3d_files = glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png'))
        if gd3d_files:
            return True
        
        # Fallback check for any gradient descent related image
        gd3d_files = glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png'))
        return len(gd3d_files) > 0

    def refresh_models(self):
        """Refresh the list of available models"""
        try:
            # Clear existing list
            self.model_listbox.delete(0, tk.END)
            
            # Clear selected model path when refreshing
            self.selected_model_path = None
            
            # Use path utilities to find model directories
            model_dirs = []
            
            # Search in multiple locations
            search_paths = ['.', 'simple', 'models', 'production_models']
            for base_path in search_paths:
                if os.path.exists(base_path):
                    pattern = os.path.join(base_path, 'model_*')
                    found_models = glob.glob(pattern)
                    model_dirs.extend(found_models)
            
            # Remove duplicates and sort by creation time (newest first)
            model_dirs = list(set(model_dirs))
            model_dirs.sort(key=os.path.getctime, reverse=True)
            
            # Add models to listbox
            for model_dir in model_dirs:
                model_name = os.path.basename(model_dir)
                self.model_listbox.insert(tk.END, model_name)
                
            # If there are any models, select the first one
            if model_dirs:
                self.model_listbox.selection_set(0)
                self.on_model_select(None)  # Trigger plot loading
                self.status_var.set(f"Loaded {len(model_dirs)} model(s)")
                print(f"Found {len(model_dirs)} models: {[os.path.basename(d) for d in model_dirs]}")
            else:
                self.status_var.set("No models available")
                self.load_saved_plots()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh model list: {str(e)}")
            self.status_var.set(f"Error refreshing models: {str(e)}")
            
    def delete_model(self):
        """Delete the selected model(s)"""
        # Get selected models
        selections = self.model_listbox.curselection()
        if not selections:
            messagebox.showwarning("Warning", "Please select one or more models to delete")
            return
        # Get model directory names
        model_names = [self.model_listbox.get(i) for i in selections]
        # Confirm deletion
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the following model(s)?\n\n" + "\n".join(model_names) + "\n\nThis action cannot be undone."):
            return
        try:
            import shutil
            for model_name in model_names:
                # Find the actual model directory using path utilities
                model_dir = path_utils.find_model_directory(model_name)
                if model_dir:
                    model_path = model_dir
                else:
                    # Fallback to current directory
                    model_path = os.path.join(self.current_model_dir, model_name)
                
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    print(f"Deleted model: {model_path}")
                else:
                    print(f"Model not found: {model_path}")
            
            # Refresh model list
            self.refresh_models()
            # Update status
            self.status_var.set(f"Deleted {len(model_names)} model(s) successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete model(s): {str(e)}")
            self.status_var.set(f"Error deleting model(s): {str(e)}")

    def toggle_lock_features(self):
        """Toggle the lock state of the features"""
        self.features_locked = not self.features_locked
        
        if self.features_locked:
            # Get current selected features and lock them
            self.locked_features = self.x_features_listbox.curselection()
            self.lock_button.config(text="Unlock")
            self.status_var.set("Features locked")
            print(f"Features locked: {self.locked_features}")
            print(f"Locked feature names: {[self.x_features_listbox.get(i) for i in self.locked_features]}")
        else:
            self.lock_button.config(text="Lock")
            self.status_var.set("Features unlocked")
            print("Features unlocked")
        
        # Update the listbox state
        self.update_feature_lock_state()

    def update_feature_lock_state(self):
        """Update the listbox to show locked features"""
        if self.features_locked:
            # Restore locked selections
            self.x_features_listbox.selection_clear(0, tk.END)
            for index in self.locked_features:
                self.x_features_listbox.selection_set(index)
            print(f"Restored locked selections: {self.locked_features}")
        
        # Update the listbox state
        self.x_features_listbox.config(state=tk.DISABLED if self.features_locked else tk.NORMAL)
        print(f"Listbox state: {'DISABLED' if self.features_locked else 'NORMAL'}")

    def show_tooltip(self, text):
        """Show a tooltip with the given text."""
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        x, y, _, _ = self.root.bbox("insert")
        x += self.root.winfo_rootx() + 25
        y += self.root.winfo_rooty() + 25
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tooltip, text=text, background="lightyellow", padding=5)
        label.pack()

    def hide_tooltip(self):
        """Hide the tooltip."""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
            del self.tooltip

    def refresh_predictions(self):
        """Refresh the list of saved predictions CSV files."""
        pred_files = sorted(glob.glob("predictions_*.csv"), key=os.path.getctime, reverse=True)
        self.predictions_combo['values'] = pred_files
        if pred_files:
            self.predictions_combo.current(0)
            self.selected_prediction_file = pred_files[0]
        else:
            self.predictions_combo.set("")
            self.selected_prediction_file = None

    def on_prediction_select(self, event=None):
        """Handle selection of a prediction file from the combo box."""
        self.selected_prediction_file = self.predictions_var.get()

    def open_selected_prediction(self):
        """Open the selected prediction file in the default CSV viewer"""
        if self.selected_prediction_file and os.path.exists(self.selected_prediction_file):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", self.selected_prediction_file])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", self.selected_prediction_file], shell=True)
                else:  # Linux
                    subprocess.run(["xdg-open", self.selected_prediction_file])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Please select a valid prediction file")

    def display_predictions(self, pred_file):
        """Display prediction results in the prediction results tab"""
        try:
            # Load predictions
            df = pd.read_csv(pred_file)
            ticker = self.get_ticker_from_filename()
            
            # Get current model name for display
            current_model = os.path.basename(self.selected_model_path) if self.selected_model_path else "Unknown Model"
            
            # Display in the prediction results tab
            self.pred_ax.clear()
            
            if 'actual' in df.columns and 'predicted' in df.columns:
                # Check if we have date/time columns for x-axis
                if 'date' in df.columns or 'timestamp' in df.columns:
                    date_col = 'date' if 'date' in df.columns else 'timestamp'
                    # Convert to datetime if not already
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    self.pred_ax.plot(df[date_col], df['actual'], label='Actual', alpha=0.7, linewidth=2)
                    self.pred_ax.plot(df[date_col], df['predicted'], label='Predicted', alpha=0.7, linewidth=2)
                    self.pred_ax.set_title(f"Actual vs Predicted Values ({ticker})\nModel: {current_model}")
                    self.pred_ax.set_xlabel("Date")
                    self.pred_ax.set_ylabel("Price")
                    self.pred_ax.legend()
                    
                    # Format x-axis dates
                    self.pred_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    plt.setp(self.pred_ax.get_xticklabels(), rotation=45)
                else:
                    # No date column, use index
                    self.pred_ax.plot(df['actual'], label='Actual', alpha=0.7, linewidth=2)
                    self.pred_ax.plot(df['predicted'], label='Predicted', alpha=0.7, linewidth=2)
                    self.pred_ax.set_title(f"Actual vs Predicted Values ({ticker})\nModel: {current_model}")
                    self.pred_ax.set_xlabel("Data Point")
                    self.pred_ax.set_ylabel("Price")
                    self.pred_ax.legend()
                
                # Add correlation info
                correlation = np.corrcoef(df['actual'], df['predicted'])[0, 1]
                self.pred_ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                                transform=self.pred_ax.transAxes, 
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                # Plot predictions only
                if 'date' in df.columns or 'timestamp' in df.columns:
                    date_col = 'date' if 'date' in df.columns else 'timestamp'
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    self.pred_ax.plot(df[date_col], df.iloc[:, -1], label='Predictions', linewidth=2)
                    self.pred_ax.set_title(f"Predictions ({ticker})\nModel: {current_model}")
                    self.pred_ax.set_xlabel("Date")
                    self.pred_ax.set_ylabel("Price")
                    self.pred_ax.legend()
                    
                    # Format x-axis dates
                    self.pred_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                    plt.setp(self.pred_ax.get_xticklabels(), rotation=45)
                else:
                    self.pred_ax.plot(df.iloc[:, -1], label='Predictions', linewidth=2)
                    self.pred_ax.set_title(f"Predictions ({ticker})\nModel: {current_model}")
                    self.pred_ax.set_xlabel("Data Point")
                    self.pred_ax.set_ylabel("Price")
                    self.pred_ax.legend()
            
            self.pred_ax.grid(True, alpha=0.3)
            self.pred_canvas.draw()
            
            # Switch to prediction results tab (index 1)
            self.switch_to_tab(1)
            
            self.status_var.set(f"Displaying results from {pred_file} using model: {current_model}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")

    def switch_to_tab(self, tab_index):
        """Switch to a specific tab in the display notebook"""
        if hasattr(self, 'display_notebook') and tab_index < self.display_notebook.index('end'):
            self.display_notebook.select(tab_index)

    def update_training_results(self, losses):
        """Update the training results plot"""
        try:
            self.results_ax.clear()
            
            if not losses or len(losses) == 0:
                print("No losses data to plot")
                return
                
            epochs = list(range(1, len(losses) + 1))
            
            # Ensure epochs and losses have the same length
            if len(epochs) != len(losses):
                min_length = min(len(epochs), len(losses))
                epochs = epochs[:min_length]
                losses = losses[:min_length]
                print(f"Warning: Array length mismatch in training results. Using min length: {min_length}")
            
            self.results_ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
            self.results_ax.set_title("Training Loss Over Time")
            self.results_ax.set_xlabel("Epoch")
            self.results_ax.set_ylabel("MSE")
            self.results_ax.legend()
            self.results_ax.grid(True, alpha=0.3)
            self.results_canvas.draw()
            self.results_canvas.flush_events()
            print(f"Training results updated with {len(losses)} epochs")
        except Exception as e:
            print(f"Error updating training results: {e}")
            print("Error occurred")
            print(f"Error: {e}")

    def get_ticker_from_filename(self):
        """Extract ticker symbol from the data file name"""
        try:
            if not self.data_file:
                return ""
            filename = os.path.basename(self.data_file)
            if "_us_data.csv" in filename:
                ticker = filename.split("_us_data.csv")[0].upper()
                return ticker
            elif "_data.csv" in filename:
                ticker = filename.split("_data.csv")[0].upper()
                return ticker
            else:
                import re
                match = re.match(r"^([A-Z]+)", filename.upper())
                if match:
                    return match.group(1)
                return ""
        except Exception as e:
            print(f"Error extracting ticker: {e}")
            return ""

    def update_live_plot(self, epochs, losses):
        """Update the live plot with training progress (2D only to avoid matplotlib 3D issues)"""
        try:
            # Check if GUI is still active
            if not hasattr(self, 'root') or not self.root or not self.root.winfo_exists():
                return
                
            ticker = self.get_ticker_from_filename()
            # Ensure epochs and losses have the same length
            if len(epochs) != len(losses):
                min_length = min(len(epochs), len(losses))
                epochs_plot = epochs[:min_length]
                losses_plot = losses[:min_length]
                print(f"Warning: Array length mismatch. Using min length: {min_length}")
            else:
                epochs_plot = epochs
                losses_plot = losses
            
            if len(epochs_plot) == 0:
                return
            
            # Clear the plot
            self.gd_ax.clear()
            
            # Simple 2D plot to avoid matplotlib 3D issues
            self.gd_ax.plot(epochs_plot, losses_plot, 'r-', linewidth=2, label='Training Loss')
            self.gd_ax.set_title(f"Live Training Progress - {ticker} (Epoch {epochs_plot[-1]})")
            self.gd_ax.set_xlabel("Epoch")
            self.gd_ax.set_ylabel("MSE Loss")
            self.gd_ax.legend()
            self.gd_ax.grid(True, alpha=0.3)
            
            # Add current loss value as text
            if losses_plot:
                current_loss = losses_plot[-1]
                self.gd_ax.text(0.02, 0.98, f'Current Loss: {current_loss:.6f}', 
                              transform=self.gd_ax.transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Use a try-catch for the canvas draw to handle any remaining matplotlib issues
            try:
                self.gd_canvas.draw()
                self.gd_canvas.flush_events()
            except Exception as draw_error:
                print(f"Canvas draw error (non-critical): {draw_error}")
            
            print(f"Live plot updated with {len(epochs_plot)} epochs")
            
        except Exception as e:
            print(f"Error updating live plot: {e}")
            # Don't print additional error messages to avoid recursion

    def update_progress(self, message):
        """Update the progress text widget with a message"""
        try:
            # Check if the GUI is still active and progress_text exists
            if (hasattr(self, 'progress_text') and 
                self.progress_text and 
                hasattr(self, 'root') and 
                self.root and 
                self.root.winfo_exists()):
                
                # Enable the text widget for writing
                self.progress_text.configure(state="normal")
                
                # Insert the message with timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                
                self.progress_text.insert(tk.END, formatted_message)
                self.progress_text.see(tk.END)
                
                # Disable the text widget to prevent editing
                self.progress_text.configure(state="disabled")
                
                # Force update
                self.root.update()
                
            print(f"Progress: {message}")
        except Exception as e:
            # Use a simple print to avoid recursion
            print(f"Error updating progress: {str(e)}")

    def browse_model_dir(self):
        """Open directory dialog to select model directory"""
        new_dir = filedialog.askdirectory(
            title="Select Model Directory",
            initialdir=self.current_model_dir
        )
        if new_dir:
            self.current_model_dir = new_dir
            self.model_dir_var.set(new_dir)
            self.refresh_models()

    def train_model(self):
        """Train a new model with the specified parameters"""
        try:
            # Validate inputs
            hidden_size = int(self.hidden_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())
            patience = int(self.patience_var.get())
            
            # Get selected features - handle both locked and unlocked states
            if self.features_locked and hasattr(self, 'locked_features') and self.locked_features:
                # Use locked features
                x_indices = self.locked_features
                self.x_features = [self.x_features_listbox.get(i) for i in x_indices]
                print(f"Using locked features: {self.x_features}")
            else:
                # Get selected features from UI
                x_indices = self.x_features_listbox.curselection()
                if not x_indices:
                    messagebox.showerror("Error", "Please select at least one input feature")
                    return
                self.x_features = [self.x_features_listbox.get(i) for i in x_indices]
                print(f"Using selected features: {self.x_features}")
            
            self.y_feature = self.y_features_combo.get()
            if not self.y_feature or self.y_feature.strip() == "":
                messagebox.showerror("Error", "Please select a target feature")
                return
            
            # Validate features
            if not self.validate_features():
                return
            
            # Check if we have a data file
            if not self.data_file:
                messagebox.showerror("Error", "Please select a data file first")
                return
                
            # Check if we have a model directory
            if not self.current_model_dir:
                messagebox.showerror("Error", "Please select a model directory")
                return
            
            # Update status
            self.status_var.set("Training model...")
            self.root.update()
            
            # Print debug information
            print(f"\nTraining parameters:")
            print(f"Data File: {self.data_file}")
            print(f"Model Dir: {self.current_model_dir}")
            print(f"X Features: {self.x_features}")
            print(f"Y Feature: {self.y_feature}")
            print(f"Hidden Size: {hidden_size}")
            print(f"Learning Rate: {learning_rate}")
            print(f"Batch Size: {batch_size}")
            print(f"Patience Epoch: {patience}")
            print(f"Features Locked: {self.features_locked}")
            
            # Switch to gradient descent tab for live visualization
            self.switch_to_tab(2)  # Gradient Descent tab
            
            # Clear progress text
            if hasattr(self, 'progress_text'):
                self.progress_text.configure(state="normal")
                self.progress_text.delete(1.0, tk.END)
                self.progress_text.configure(state="disabled")
            
            # Set training flag
            self.is_training = True
            
            # Start training in background thread
            import threading
            training_thread = threading.Thread(
                target=self._run_training_thread,
                args=(hidden_size, learning_rate, batch_size, patience),
                daemon=True
            )
            training_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def _run_training_thread(self, hidden_size, learning_rate, batch_size, patience):
        """Run training in a background thread with live updates"""
        try:
            # Import required modules for training
            import numpy as np
            import pandas as pd
            from datetime import datetime
            
            # Use the preprocessed dataframe that was already loaded and validated
            if not hasattr(self, 'preprocessed_df') or self.preprocessed_df is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "No preprocessed data available. Please load a data file first."))
                self.root.after(0, self._enable_train_button)
                return
            
            df = self.preprocessed_df.copy()
            
            print("Using preprocessed data...")
            self.root.after(0, lambda: self.update_progress("Using preprocessed data..."))
            
            # Verify that all required features exist
            missing_features = []
            for feature in self.x_features + [self.y_feature]:
                if feature not in df.columns:
                    missing_features.append(feature)
            
            if missing_features:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Missing features: {missing_features}"))
                self.root.after(0, self._enable_train_button)
                return
            
            # Prepare features and target
            X = df[self.x_features].values.astype(np.float64)
            Y = df[self.y_feature].values.reshape(-1, 1).astype(np.float64)
            
            # Check for any remaining NaN or infinite values
            if np.any(np.isnan(X)) or np.any(np.isnan(Y)) or np.any(np.isinf(X)) or np.any(np.isinf(Y)):
                self.root.after(0, lambda: messagebox.showerror("Error", "Data contains NaN or infinite values after preprocessing"))
                self.root.after(0, self._enable_train_button)
                return
            
            # Normalize data
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            X_norm = (X - X_min) / (X_max - X_min + 1e-8)
            
            Y_min = np.min(Y)
            Y_max = np.max(Y)
            Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
            
            # Split data
            split_idx = int(0.8 * len(X_norm))
            X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
            Y_train, Y_test = Y_norm[:split_idx], Y_norm[split_idx:]
            
            # Initialize neural network
            input_size = len(self.x_features)
            model = StockNet(input_size, hidden_size)
            
            # Training parameters
            epochs = 1000
            n_samples = X_train.shape[0]
            best_val_mse = float('inf')
            patience_counter = 0
            
            # Configure progress bar
            self.root.after(0, lambda: self.progress_bar.configure(maximum=epochs))
            self.root.after(0, lambda: self.progress_bar.configure(value=0))
            
            # Track training progress
            train_losses = []
            val_losses = []
            epochs_list = []
            
            # Create model directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"model_{timestamp}"
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"Training started in directory: {model_dir}")
            self.root.after(0, lambda: self.update_progress(f"Training started in directory: {model_dir}"))
            
            # Enable stop button and disable train button
            self.root.after(0, lambda: self.stop_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.train_button.config(state=tk.DISABLED))
            
            # Training loop
            for epoch in range(epochs):
                if not self.is_training:  # Check if training was stopped
                    break
                    
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
                
                # Validation
                val_output = model.forward(X_test)
                val_mse = np.mean((val_output - Y_test) ** 2)
                val_losses.append(val_mse)
                
                epochs_list.append(epoch + 1)
                
                # Update live plot less frequently to prevent recursion
                # Ensure arrays have the same length for plotting
                plot_epochs = epochs_list.copy()
                plot_losses = train_losses.copy()
                
                # Ensure both arrays have the same length
                min_len = min(len(plot_epochs), len(plot_losses))
                plot_epochs = plot_epochs[:min_len]
                plot_losses = plot_losses[:min_len]
                
                if min_len > 0 and epoch % 20 == 0:  # Update every 20 epochs instead of 5
                    try:
                        self.root.after(0, lambda e=plot_epochs, l=plot_losses: self.update_live_plot(e, l))
                    except Exception as plot_error:
                        print(f"Live plot update error: {plot_error}")
                
                # Update progress less frequently to avoid recursion issues
                if epoch % 20 == 0 or epoch < 10:  # Update every 20 epochs or first 10 epochs
                    try:
                        self.root.after(0, lambda: self.update_progress(f"Epoch {epoch+1}, Train MSE: {avg_train_mse:.6f}, Val MSE: {val_mse:.6f}"))
                    except Exception as progress_error:
                        print(f"Progress update error: {progress_error}")
                
                # Early stopping
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    self.root.after(0, lambda: self.update_progress(f"Early stopping at epoch {epoch+1}"))
                    break
                
                # Update progress bar
                self.root.after(0, lambda e=epoch: self.progress_bar.configure(value=e+1))
            
            # Save model and results
            model.save_weights(model_dir, prefix="stock_model")
            
            # Also save as NPZ file for compatibility with predict.py
            np.savez_compressed(
                os.path.join(model_dir, 'stock_model.npz'),
                W1=model.W1,
                b1=model.b1,
                W2=model.W2,
                b2=model.b2,
                X_min=X_min,
                X_max=X_max,
                Y_min=Y_min,
                Y_max=Y_max,
                has_target_norm=True,
                input_size=input_size,
                hidden_size=hidden_size
            )
            
            # Save training losses
            np.savetxt(os.path.join(model_dir, 'training_losses.csv'), 
                      np.column_stack((train_losses, val_losses)), delimiter=',')
            
            # Save feature info
            feature_info = {
                'x_features': self.x_features,
                'y_feature': self.y_feature,
                'input_size': len(self.x_features)
            }
            with open(os.path.join(model_dir, 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f)
            
            # Save model metadata
            metadata = {
                'data_file': self.data_file,
                'hidden_size': hidden_size,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'patience': patience,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_epochs': len(train_losses),
                'final_train_loss': train_losses[-1] if train_losses else 0,
                'final_val_loss': val_losses[-1] if val_losses else 0
            }
            with open(os.path.join(model_dir, 'model_metadata.txt'), 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            # Save normalization parameters
            np.savetxt(os.path.join(model_dir, 'scaler_mean.csv'), X_min, delimiter=',')
            np.savetxt(os.path.join(model_dir, 'scaler_std.csv'), X_max - X_min, delimiter=',')
            np.savetxt(os.path.join(model_dir, 'target_min.csv'), np.array([Y_min]).reshape(1, -1), delimiter=',')
            np.savetxt(os.path.join(model_dir, 'target_max.csv'), np.array([Y_max]).reshape(1, -1), delimiter=',')
            
            # Disable stop button
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            
            # Update GUI on completion
            self.root.after(0, lambda: self._training_completed(model_dir, train_losses, val_losses))
            
        except Exception as e:
            print(f"Training error: {e}")
            print("Error occurred")
            print(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_bar.configure(value=0))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, self._enable_train_button)

    def _training_completed(self, model_dir, train_losses, val_losses):
        """Handle training completion and save plots."""
        try:
            self.status_var.set("Training completed successfully!")
            self.progress_bar.configure(value=0)
            self.stop_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.NORMAL)
            
            # Calculate training statistics
            final_train_loss = train_losses[-1] if train_losses else 0.0
            final_val_loss = val_losses[-1] if val_losses else 0.0
            total_epochs = len(train_losses)
            
            # Save loss plot with enhanced styling
            plots_dir = os.path.join(model_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            fig = plt.Figure(figsize=(10, 7))
            ax = fig.add_subplot(111)
            epochs = list(range(1, len(train_losses) + 1))
            
            # Plot training loss
            ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            
            # Plot validation loss if available
            if val_losses and len(val_losses) == len(train_losses):
                ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            # Enhanced styling
            ax.set_title(f"Training Progress - {self.get_ticker_from_filename()}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("MSE Loss", fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add final loss annotations
            ax.annotate(f'Final Train Loss: {final_train_loss:.6f}', 
                       xy=(total_epochs, final_train_loss), 
                       xytext=(total_epochs*0.7, final_train_loss*1.5),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                       fontsize=10, color='blue')
            
            if val_losses and len(val_losses) == len(train_losses):
                ax.annotate(f'Final Val Loss: {final_val_loss:.6f}', 
                           xy=(total_epochs, final_val_loss), 
                           xytext=(total_epochs*0.7, final_val_loss*0.5),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=10, color='red')
            
            # Save with high quality
            fig.savefig(os.path.join(plots_dir, 'loss_curve.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Save training metadata
            metadata = {
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'total_epochs': total_epochs,
                'training_completed_at': datetime.now().isoformat(),
                'ticker': self.get_ticker_from_filename(),
                'model_directory': model_dir,
                'data_file': self.data_file,
                'x_features': self.x_features,
                'y_feature': self.y_feature,
                'hidden_size': self.hidden_size_var.get(),
                'learning_rate': self.learning_rate_var.get(),
                'batch_size': self.batch_size_var.get(),
                'patience': self.patience_var.get()
            }
            
            with open(os.path.join(model_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Refresh models list
            self.refresh_models()
            
            # Update training results tab
            self.update_training_results(train_losses)
            
            # Update plots tab
            self.update_plots_tab(train_losses, val_losses)
            
            # Set the newly created model as selected
            self.selected_model_path = model_dir
            self.load_saved_plots()
            
            # Automatically generate 3D gradient descent visualization
            self.status_var.set("Generating 3D gradient descent visualization...")
            self.root.update()
            
            try:
                # Generate 3D visualization automatically
                success, stdout, stderr = script_launcher.launch_gradient_descent(
                    model_dir=model_dir,
                    save_png=True,
                    color='viridis',
                    point_size=8,
                    line_width=2,
                    surface_alpha=0.6
                )
                
                if success:
                    # Update the 3D Gradient Descent tab with the visualization
                    self.update_3d_gradient_descent_tab()
                    # Refresh saved plots to include the new 3D visualization
                    self.load_saved_plots()
                    
                    # Switch to 3D Gradient Descent tab to show the visualization
                    self.switch_to_tab(4)  # 3D Gradient Descent tab index
                    
                    # Enhanced success message with 3D visualization info
                    success_msg = f"Model training completed successfully!\n\n"
                    success_msg += f"📊 Training Statistics:\n"
                    success_msg += f"   • Total Epochs: {total_epochs}\n"
                    success_msg += f"   • Final Training Loss: {final_train_loss:.6f}\n"
                    if val_losses and len(val_losses) == len(train_losses):
                        success_msg += f"   • Final Validation Loss: {final_val_loss:.6f}\n"
                    success_msg += f"   • Model saved in: {model_dir}\n"
                    success_msg += f"   • Loss plot saved to: {plots_dir}/loss_curve.png\n"
                    success_msg += f"   • 3D visualization generated and displayed!\n\n"
                    success_msg += f"🎯 The 3D Gradient Descent tab now shows your training visualization!"
                    
                    messagebox.showinfo("Training Success", success_msg)
                else:
                    # Fallback success message without 3D visualization
                    success_msg = f"Model training completed successfully!\n\n"
                    success_msg += f"📊 Training Statistics:\n"
                    success_msg += f"   • Total Epochs: {total_epochs}\n"
                    success_msg += f"   • Final Training Loss: {final_train_loss:.6f}\n"
                    if val_losses and len(val_losses) == len(train_losses):
                        success_msg += f"   • Final Validation Loss: {final_val_loss:.6f}\n"
                    success_msg += f"   • Model saved in: {model_dir}\n"
                    success_msg += f"   • Loss plot saved to: {plots_dir}/loss_curve.png\n\n"
                    success_msg += f"💡 Tip: Use the 'Create 3D Visualization' button to generate gradient descent plots!"
                    
                    messagebox.showinfo("Training Success", success_msg)
                    
            except Exception as viz_error:
                print(f"Error generating 3D visualization: {viz_error}")
                # Continue with normal success message even if 3D visualization fails
                success_msg = f"Model training completed successfully!\n\n"
                success_msg += f"📊 Training Statistics:\n"
                success_msg += f"   • Total Epochs: {total_epochs}\n"
                success_msg += f"   • Final Training Loss: {final_train_loss:.6f}\n"
                if val_losses and len(val_losses) == len(train_losses):
                    success_msg += f"   • Final Validation Loss: {final_val_loss:.6f}\n"
                success_msg += f"   • Model saved in: {model_dir}\n"
                success_msg += f"   • Loss plot saved to: {plots_dir}/loss_curve.png\n\n"
                success_msg += f"💡 Tip: Use the 'Create 3D Visualization' button to generate gradient descent plots!"
                
                messagebox.showinfo("Training Success", success_msg)
            
            # Log completion
            print(f"✅ Training completed - Epochs: {total_epochs}, Final Loss: {final_train_loss:.6f}")
            
        except Exception as e:
            error_msg = f"Error in training completion: {e}"
            print(error_msg)
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Error updating results: {str(e)}")
            
            # Ensure UI is reset even on error
            self.stop_button.config(state=tk.DISABLED)
            self.train_button.config(state=tk.NORMAL)
            self.progress_bar.configure(value=0)

    def _enable_train_button(self):
        """Re-enable the train button"""
        self.train_button.config(state=tk.NORMAL)
        self.status_var.set("Ready")

    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Training stopped")
        # Reset progress bar
        self.progress_bar.configure(value=0)

    def clear_plot_cache(self):
        """Clear the plot image cache to free memory."""
        self.plot_image_cache.clear()
        print("Plot image cache cleared")
        self.status_var.set("Plot cache cleared")
        self.update_cache_info()

    def update_cache_info(self):
        """Update the cache info label."""
        cache_size = len(self.plot_image_cache)
        self.cache_info_label.config(text=f"Cache: {cache_size} images")
        if cache_size > 0:
            self.cache_info_label.config(foreground="green")
        else:
            self.cache_info_label.config(foreground="blue")

    def load_saved_plots(self):
        """Load and display PNG plots from the selected model's plots directory with optimizations."""
        # Cancel any ongoing plot loading operations
        self.cancel_plot_loading()
        
        # Submit the plot loading task to thread pool
        future = self.thread_pool.submit(self._load_saved_plots_worker)
        self.plot_futures['load_saved_plots'] = future
        
        # Schedule a callback to update the GUI when done
        self.root.after(100, self._check_plot_loading_complete, 'load_saved_plots')

    def _load_saved_plots_worker(self):
        """Worker function to load plots in background thread."""
        try:
            # Clear existing images
            for widget in self.saved_plots_inner_frame.winfo_children():
                widget.destroy()
            
            if not self.selected_model_path:
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="Select a model to view saved plots", 
                                                        foreground='black', background='white')
                self.saved_plots_placeholder.pack(pady=20)
                return {'success': True, 'message': 'No model selected'}
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if not os.path.exists(plots_dir):
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="No plots directory found in model directory", 
                                                        foreground='black', background='white')
                self.saved_plots_placeholder.pack(pady=20)
                return {'success': False, 'message': 'No plots directory found'}
            
            plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
            if not plot_files:
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="No PNG plots found in model directory", 
                                                        foreground='black', background='white')
                self.saved_plots_placeholder.pack(pady=20)
                return {'success': False, 'message': 'No PNG plots found'}
            
            # Show loading progress
            loading_label = ttk.Label(self.saved_plots_inner_frame, 
                                     text="Loading plots...", 
                                     foreground='blue', background='white')
            loading_label.pack(pady=20)
            
            # Load and display each PNG with optimizations
            self.saved_plots_images = []  # Keep references to avoid garbage collection
            max_width = 600  # Max width for images to fit in GUI
            max_height = 400  # Max height to prevent oversized images
            displayed_count = 0
            
            # Limit to first 10 plots for performance
            plot_files_to_load = plot_files[:10]
            
            for i, plot_file in enumerate(plot_files_to_load):
                try:
                    # Update loading progress
                    loading_label.config(text=f"Loading plot {i+1}/{len(plot_files_to_load)}...")
                    
                    # Check cache first
                    cache_key = f"{plot_file}_{max_width}_{max_height}"
                    if cache_key in self.plot_image_cache:
                        photo = self.plot_image_cache[cache_key]
                        print(f"Using cached image: {os.path.basename(plot_file)}")
                    else:
                        # Load and resize image with faster method
                        img = Image.open(plot_file)
                        
                        # Get original dimensions
                        img_width, img_height = img.size
                        
                        # Calculate optimal scale to fit within bounds
                        scale_x = max_width / img_width
                        scale_y = max_height / img_height
                        scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
                        
                        # Only resize if necessary
                        if scale < 1.0:
                            new_size = (int(img_width * scale), int(img_height * scale))
                            # Use faster resampling method for better performance
                            img = img.resize(new_size, Image.Resampling.BILINEAR)
                        
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(img)
                        
                        # Cache the image (with size limit)
                        if len(self.plot_image_cache) >= self.max_cache_size:
                            # Remove oldest entry (simple FIFO)
                            oldest_key = next(iter(self.plot_image_cache))
                            del self.plot_image_cache[oldest_key]
                        
                        self.plot_image_cache[cache_key] = photo
                        print(f"Cached new image: {os.path.basename(plot_file)}")
                    
                    # Create frame for this plot
                    frame = ttk.Frame(self.saved_plots_inner_frame)
                    frame.pack(pady=5, fill="x", padx=10)
                    
                    # Create filename label with better formatting
                    filename = os.path.basename(plot_file)
                    filename_label = ttk.Label(frame, text=filename, 
                                              foreground='black', background='white',
                                              font=('Arial', 10, 'bold'))
                    filename_label.pack(anchor="w", pady=(5,2))
                    
                    # Create image label
                    img_label = ttk.Label(frame, image=photo)
                    img_label.pack(anchor="w")
                    
                    # Store reference to prevent garbage collection
                    self.saved_plots_images.append(photo)
                    displayed_count += 1
                    
                except Exception as img_error:
                    print(f"Error loading image {plot_file}: {img_error}")
                    continue
            
            # Remove loading label
            loading_label.destroy()
            
            if displayed_count == 0:
                self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                        text="Error loading all plot images", 
                                                        foreground='red', background='white')
                self.saved_plots_placeholder.pack(pady=20)
                return {'success': False, 'message': 'Error loading plot images'}
            
            # Show summary with cache info
            summary_text = f"Showing {displayed_count} of {len(plot_files)} plots"
            if len(plot_files) > 10:
                summary_text += " (showing first 10)"
            summary_text += f" | Cache: {len(self.plot_image_cache)} images"
            
            summary_label = ttk.Label(self.saved_plots_inner_frame, 
                                     text=summary_text, 
                                     foreground='blue', background='white')
            summary_label.pack(pady=5)
            
            # Update scroll region
            self.saved_plots_canvas.configure(scrollregion=self.saved_plots_canvas.bbox("all"))
            
            return {
                'success': True, 
                'message': f'Loaded {displayed_count} plot(s) from {plots_dir}',
                'displayed_count': displayed_count,
                'total_count': len(plot_files)
            }
            
        except Exception as e:
            self.saved_plots_placeholder = ttk.Label(self.saved_plots_inner_frame, 
                                                    text=f"Error loading plots: {str(e)}", 
                                                    foreground='red', background='white')
            self.saved_plots_placeholder.pack(pady=20)
            return {'success': False, 'message': f'Error loading plots: {str(e)}'}

    def _check_plot_loading_complete(self, operation_name):
        """Check if plot loading operation is complete and update GUI."""
        if operation_name not in self.plot_futures:
            return
        
        future = self.plot_futures[operation_name]
        if future.done():
            try:
                result = future.result()
                if result['success']:
                    self.status_var.set(result['message'])
                    if 'displayed_count' in result:
                        print(f"Loaded {result['displayed_count']} plot(s) from {result['total_count']} available")
                        print(f"Image cache size: {len(self.plot_image_cache)}")
                        self.update_cache_info()
                else:
                    self.status_var.set(result['message'])
                    print(f"Plot loading failed: {result['message']}")
            except Exception as e:
                self.status_var.set(f"Error in plot loading: {str(e)}")
                print(f"Error in plot loading: {e}")
            finally:
                # Remove the completed future
                del self.plot_futures[operation_name]
        else:
            # Schedule another check in 100ms
            self.root.after(100, self._check_plot_loading_complete, operation_name)

    def cancel_plot_loading(self):
        """Cancel ongoing plot loading operations."""
        for operation_name, future in list(self.plot_futures.items()):
            if not future.done():
                future.cancel()
                print(f"Cancelled plot loading operation: {operation_name}")
        self.plot_futures.clear()

    def cleanup_thread_pool(self):
        """Clean up thread pool on application exit."""
        try:
            self.cancel_plot_loading()
            self.thread_pool.shutdown(wait=False)
            print("Thread pool cleaned up")
        except Exception as e:
            print(f"Error cleaning up thread pool: {e}")

    def update_plots_tab(self, train_losses, val_losses):
        """Update the plots tab with comprehensive training results"""
        try:
            self.plots_ax.clear()
            ticker = self.get_ticker_from_filename()
            epochs = list(range(1, len(train_losses) + 1))
            
            # Plot training and validation losses
            self.plots_ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            if len(val_losses) == len(train_losses):
                self.plots_ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            # Add moving average for smoother visualization
            if len(train_losses) > 10:
                window = min(10, len(train_losses) // 4)
                train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
                ma_epochs = epochs[window-1:]
                self.plots_ax.plot(ma_epochs, train_ma, 'g--', linewidth=1, label=f'Training MA ({window})', alpha=0.6)
            
            self.plots_ax.set_title(f"Training Progress Overview ({ticker})")
            self.plots_ax.set_xlabel("Epoch")
            self.plots_ax.set_ylabel("MSE Loss")
            self.plots_ax.legend()
            self.plots_ax.grid(True, alpha=0.3)
            
            # Add final loss values as text
            final_train_loss = train_losses[-1] if train_losses else 0
            final_val_loss = val_losses[-1] if val_losses else 0
            self.plots_ax.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.6f}\nFinal Val Loss: {final_val_loss:.6f}', 
                             transform=self.plots_ax.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            self.plots_canvas.draw()
            self.plots_canvas.flush_events()
            print(f"Plots tab updated with {len(train_losses)} epochs")
            
        except Exception as e:
            print(f"Error updating plots tab: {e}")
            print(f"Error: {e}")

    def create_3d_gradient_descent_visualization(self, train_losses):
        """Create 3D gradient descent visualization with animation"""
        try:
            ticker = self.get_ticker_from_filename()
            
            # Clear the 3D plot
            self.gd3d_ax.clear()
            
            if len(train_losses) < 3:
                self.gd3d_ax.text(0.5, 0.5, 0.5, 'Need at least 3 epochs for 3D visualization', 
                                 ha='center', va='center', transform=self.gd3d_ax.transAxes,
                                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.gd3d_ax.set_title("3D Gradient Descent Visualization")
                self.gd3d_canvas.draw()
                return
            
            # Create 3D surface representing the loss landscape
            x = np.linspace(-2, 2, 50)
            y = np.linspace(-2, 2, 50)
            X, Y = np.meshgrid(x, y)
            
            # Create a bowl-shaped loss surface with some complexity
            Z = X**2 + Y**2 + 0.1 * np.sin(3*X) * np.cos(3*Y)
            
            # Plot the 3D surface
            self.gd3d_ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', edgecolor='none')
            
            # Create spiral trajectory representing weight updates during training
            t = np.linspace(0, 4*np.pi, len(train_losses))
            spiral_x = 1.5 * np.exp(-0.3*t) * np.cos(t)
            spiral_y = 1.5 * np.exp(-0.3*t) * np.sin(t)
            spiral_z = spiral_x**2 + spiral_y**2 + 0.1 * np.sin(3*spiral_x) * np.cos(3*spiral_y)
            
            # Plot the spiral trajectory
            self.gd3d_ax.plot(spiral_x, spiral_y, spiral_z, 'r-', linewidth=3, label='Training Path')
            
            # Mark starting and ending positions
            self.gd3d_ax.scatter([spiral_x[0]], [spiral_y[0]], [spiral_z[0]], 
                               color='green', s=100, marker='o', label='Start')
            self.gd3d_ax.scatter([spiral_x[-1]], [spiral_y[-1]], [spiral_z[-1]], 
                               color='red', s=100, marker='o', label='End')
            
            # Add intermediate points for animation effect
            for i in range(0, len(spiral_x), max(1, len(spiral_x)//20)):
                self.gd3d_ax.scatter([spiral_x[i]], [spiral_y[i]], [spiral_z[i]], 
                                   color='orange', s=50, marker='o', alpha=0.6)
            
            # Set labels and title
            self.gd3d_ax.set_xlabel('Weight 1')
            self.gd3d_ax.set_ylabel('Weight 2')
            self.gd3d_ax.set_zlabel('Loss')
            self.gd3d_ax.set_title(f'3D Gradient Descent Visualization ({ticker})\nEpochs: {len(train_losses)}')
            
            # Add legend
            self.gd3d_ax.legend()
            
            # Set view angle for better visualization
            self.gd3d_ax.view_init(elev=20, azim=45)
            
            # Draw the plot
            self.gd3d_canvas.draw()
            self.gd3d_canvas.flush_events()
            
            print(f"3D gradient descent visualization created with {len(train_losses)} epochs")
            
        except Exception as e:
            print(f"Error creating 3D gradient descent visualization: {e}")
            # Fallback to simple text if 3D plotting fails
            self.gd3d_ax.clear()
            self.gd3d_ax.text(0.5, 0.5, 0.5, f'3D visualization failed: {str(e)}', 
                             ha='center', va='center', transform=self.gd3d_ax.transAxes,
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.gd3d_ax.set_title("3D Gradient Descent Visualization")
            self.gd3d_canvas.draw()

    def trigger_gradient_descent_visualization(self):
        """Trigger gradient descent visualization and save PNG plots."""
        if not self.selected_model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        try:
            self.status_var.set("Creating gradient descent visualization...")
            self.root.update()
            
            # Configuration options for the visualization
            config = {
                'color': 'viridis',
                'point_size': 8,
                'line_width': 2,
                'surface_alpha': 0.6
            }
            
            # Use script launcher to run gradient descent with enhanced options
            success, stdout, stderr = script_launcher.launch_gradient_descent(
                model_dir=self.selected_model_path,
                save_png=True,
                **config
            )
            
            if success:
                # Refresh the saved plots tab
                self.load_saved_plots()
                # Update the 3D Gradient Descent tab with the latest visualization
                self.update_3d_gradient_descent_tab()
                # Switch to 3D Gradient Descent tab to show the visualization
                self.switch_to_tab(4)  # 3D Gradient Descent tab index
                self.status_var.set("Gradient descent visualization completed successfully")
                messagebox.showinfo("Success", 
                                  "3D gradient descent visualization completed!\n"
                                  "Check the '3D Gradient Descent' tab to view the visualization.")
            else:
                error_msg = stderr if stderr else stdout
                messagebox.showerror("Error", f"Failed to create visualization:\n{error_msg}")
                self.status_var.set("Gradient descent visualization failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error creating gradient descent visualization: {str(e)}")
            self.status_var.set("Error in gradient descent visualization")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def update_3d_gradient_descent_tab(self):
        """Update the 3D Gradient Descent tab with the latest visualization image."""
        try:
            if not self.selected_model_path:
                return
            
            plots_dir = os.path.join(self.selected_model_path, 'plots')
            if not os.path.exists(plots_dir):
                return
            
            # Find the latest gradient descent 3D visualization image
            gd3d_files = sorted(glob.glob(os.path.join(plots_dir, 'gradient_descent_3d_frame_*.png')))
            
            if not gd3d_files:
                # Fallback to any gradient descent related image
                gd3d_files = sorted(glob.glob(os.path.join(plots_dir, '*gradient_descent*3d*.png')))
            
            if not gd3d_files:
                # Show placeholder if no 3D visualization found
                self.gd3d_ax.clear()
                self.gd3d_ax.text(0.5, 0.5, 0.5, 'No 3D gradient descent visualization found.\nClick "Create 3D Visualization" to generate one.', 
                                 ha='center', va='center', transform=self.gd3d_ax.transAxes,
                                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.gd3d_ax.set_title("3D Gradient Descent Visualization")
                self.gd3d_canvas.draw()
                return
            
            # Use the last (most recent) frame or a middle frame for better visualization
            if len(gd3d_files) > 10:
                # Use a middle frame for better visualization
                selected_file = gd3d_files[len(gd3d_files) // 2]
            else:
                # Use the last frame
                selected_file = gd3d_files[-1]
            
            # Load and display the image in the 3D tab
            self.load_3d_visualization_image(selected_file)
            
        except Exception as e:
            print(f"Error updating 3D gradient descent tab: {e}")
            # Fallback to placeholder
            self.gd3d_ax.clear()
            self.gd3d_ax.text(0.5, 0.5, 0.5, f'Error loading 3D visualization: {str(e)}', 
                             ha='center', va='center', transform=self.gd3d_ax.transAxes,
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.gd3d_ax.set_title("3D Gradient Descent Visualization")
            self.gd3d_canvas.draw()

    def load_3d_visualization_image(self, image_path):
        """Load a 3D visualization image into the 3D Gradient Descent tab."""
        try:
            # Clear the 3D plot
            self.gd3d_ax.clear()
            
            # Load the image
            img = Image.open(image_path)
            
            # Get image dimensions
            img_width, img_height = img.size
            
            # Always resize to exactly 75% of original size
            scale = 0.75
            
            # Resize the image to 75% of original size
            new_size = (int(img_width * scale), int(img_height * scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            
            # Convert to numpy array for matplotlib
            img_array = np.array(img)
            
            # Create a 2D plot in the 3D tab (since we're showing a static image)
            # We'll use a 2D projection to display the 3D visualization image
            # Adjust extent to center the image better
            self.gd3d_ax.imshow(img_array, aspect='auto', extent=[-1.5, 1.5, -1.5, 1.5])
            
            # Set title with filename
            filename = os.path.basename(image_path)
            self.gd3d_ax.set_title(f"3D Gradient Descent Visualization\n{filename}")
            
            # Remove axis labels since this is an image
            self.gd3d_ax.set_xticks([])
            self.gd3d_ax.set_yticks([])
            
            # Add a border
            self.gd3d_ax.spines['top'].set_visible(True)
            self.gd3d_ax.spines['right'].set_visible(True)
            self.gd3d_ax.spines['bottom'].set_visible(True)
            self.gd3d_ax.spines['left'].set_visible(True)
            
            # Draw the plot
            self.gd3d_canvas.draw()
            self.gd3d_canvas.flush_events()
            
            print(f"Loaded 3D visualization image: {filename} (resized to {new_size[0]}x{new_size[1]} - 75% of original)")
            
        except Exception as e:
            print(f"Error loading 3D visualization image: {e}")
            # Fallback to error message
            self.gd3d_ax.clear()
            self.gd3d_ax.text(0.5, 0.5, 0.5, f'Error loading 3D visualization image: {str(e)}', 
                             ha='center', va='center', transform=self.gd3d_ax.transAxes,
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            self.gd3d_ax.set_title("3D Gradient Descent Visualization")
            self.gd3d_canvas.draw()

    def _show_gradient_descent(self):
        """Run gradient_descent_3d.py to generate PNG snapshots and refresh Saved Plots tab."""
        if not self.selected_model_path:
            messagebox.showerror("Error", "Please select a model first")
            return
        
        try:
            self.status_var.set("Generating 3D gradient descent visualization...")
            self.root.update()
            
            # Use script launcher for better path resolution
            script_path = script_launcher.find_script("gradient_descent_3d.py")
            if not script_path:
                messagebox.showerror("Error", "gradient_descent_3d.py not found")
                self.status_var.set("Error: gradient_descent_3d.py not found")
                return
            
            cmd = [
                sys.executable, script_path,
                "--model_dir", self.selected_model_path,
                "--save_png"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     cwd=os.path.dirname(script_path))
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                # Refresh the saved plots tab
                self.load_saved_plots()
                # Switch to Saved Plots tab (index 1)
                self.switch_to_tab(1)
                self.status_var.set("3D gradient descent visualization generated successfully")
                messagebox.showinfo("Success", 
                                  "3D gradient descent visualization generated and saved as PNGs.\n"
                                  "Check the 'Saved Plots' tab to view the results.")
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                messagebox.showerror("Error", f"Failed to generate visualization:\n{error_msg}")
                self.status_var.set(f"Error: {error_msg}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run visualization: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

    def preload_plot_images(self, plot_files, max_width=600, max_height=400):
        """Preload plot images in the background for better performance."""
        try:
            preloaded_count = 0
            for plot_file in plot_files[:10]:  # Limit to 10 for performance
                cache_key = f"{plot_file}_{max_width}_{max_height}"
                if cache_key not in self.plot_image_cache:
                    try:
                        # Load and resize image
                        img = Image.open(plot_file)
                        img_width, img_height = img.size
                        
                        # Calculate optimal scale
                        scale_x = max_width / img_width
                        scale_y = max_height / img_height
                        scale = min(scale_x, scale_y, 1.0)
                        
                        # Only resize if necessary
                        if scale < 1.0:
                            new_size = (int(img_width * scale), int(img_height * scale))
                            img = img.resize(new_size, Image.Resampling.BILINEAR)
                        
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(img)
                        
                        # Cache the image
                        if len(self.plot_image_cache) >= self.max_cache_size:
                            oldest_key = next(iter(self.plot_image_cache))
                            del self.plot_image_cache[oldest_key]
                        
                        self.plot_image_cache[cache_key] = photo
                        preloaded_count += 1
                        
                    except Exception as e:
                        print(f"Error preloading {plot_file}: {e}")
                        continue
            
            print(f"Preloaded {preloaded_count} images")
            return preloaded_count
            
        except Exception as e:
            print(f"Error in preload_plot_images: {e}")
            return 0

    def create_thumbnail(self, image_path, max_width=150, max_height=100):
        """Create a small thumbnail for faster loading."""
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Calculate scale for thumbnail
            scale_x = max_width / img_width
            scale_y = max_height / img_height
            scale = min(scale_x, scale_y, 1.0)
            
            if scale < 1.0:
                new_size = (int(img_width * scale), int(img_height * scale))
                img = img.resize(new_size, Image.Resampling.BILINEAR)
            
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
            return None

def main():
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
