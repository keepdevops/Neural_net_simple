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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import glob
import json
import numpy as np
import tempfile
import threading
import importlib.util
import time

# Import stock_net module
try:
    import stock_net
except ImportError:
    # If stock_net is not available, we'll implement the needed functions inline
    pass

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
        
        # Initialize variables
        self.data_file = ""  # Path to data file
        self.data_file_var = tk.StringVar()  # Variable for data file entry
        
        # Thread management
        self.training_thread = None
        self.is_training = False
        
        # Initialize model directory and create it if it doesn't exist
        self.current_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.current_model_dir, exist_ok=True)
        
        # Initialize feature variables
        self.x_features = []  # Selected X features
        self.y_feature = ""   # Selected Y feature
        self.feature_list = []  # Available features from data file
        
        # Initialize training parameters
        self.hidden_size_var = tk.StringVar(value="4")  # Default hidden layer size
        self.learning_rate_var = tk.StringVar(value="0.001")  # Default learning rate
        self.batch_size_var = tk.StringVar(value="32")  # Default batch size
        
        # Status variables
        self.status_var = tk.StringVar(value="Ready")  # Status bar variable
        self.feature_status_var = tk.StringVar(value="")  # Feature status variable
        
        # Feature lock state
        self.locked_features = []
        self.features_locked = False
        self.lock_button = None
        
        # Selected prediction file
        self.selected_prediction_file = None
        
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
        
        # Create figure for training plot
        self.train_fig = plt.Figure(figsize=(5, 4), dpi=100)  # Smaller size for results panel
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, master=train_results_frame)
        self.train_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
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
        
        # Create figure for gradient descent
        self.gd_fig = plt.Figure(figsize=(5, 4), dpi=100)  # Smaller size for results panel
        self.gd_ax = self.gd_fig.add_subplot(111, projection='3d')
        self.gd_canvas = FigureCanvasTkAgg(self.gd_fig, master=gd_frame)
        self.gd_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        print(f"Display panel created with {self.display_notebook.index('end')} tabs")

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
            
            # Clear existing features
            self.x_features_listbox.delete(0, tk.END)
            self.y_features_combo.set("")
            
            # Load features from CSV file
            try:
                df = pd.read_csv(self.data_file)
                self.feature_list = df.columns.tolist()
                
                # Update X features listbox
                for feature in self.feature_list:
                    self.x_features_listbox.insert(tk.END, feature)
                
                # Update Y features combobox
                self.y_features_combo['values'] = self.feature_list
                
                # Set default Y feature if available
                if 'close' in self.feature_list:
                    self.y_features_combo.set('close')
                
                self.status_var.set("Features loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load features: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                self.data_file = ""  # Clear invalid data file
                self.data_file_var.set("")

    def validate_features(self):
        """Validate the selected features"""
        # Get X features
        x_indices = self.x_features_listbox.curselection()
        x_features = [self.x_features_listbox.get(i) for i in x_indices]
        
        # Get Y feature
        y_feature = self.y_features_combo.get()
        
        # Clear previous error message
        self.feature_status_var.set("")
        
        # Don't show error if no features are selected yet
        if not x_indices and not y_feature:
            return True
            
        # Validate features only if both X and Y are selected
        if x_indices and y_feature:
            # Check if Y feature is also selected as an X feature
            if y_feature in x_features:
                self.feature_status_var.set("Error: Target feature cannot be used as an input feature")
                return False
                
            # If we get here, features are valid - store them
            self.x_features = x_features
            self.y_feature = y_feature
            self.feature_status_var.set("Features are valid!")
            return True
            
        return True

    def _train_model(self):
        """Train a new model with the specified parameters"""
        try:
            # Validate inputs
            hidden_size = int(self.hidden_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Get selected features from UI
            x_indices = self.x_features_listbox.curselection()
            if not x_indices:
                messagebox.showerror("Error", "Please select at least one input feature")
                return
            
            # Get the feature names from indices
            x_features = []
            for i in x_indices:
                feature = self.x_features_listbox.get(i)
                if feature in self.feature_list:
                    x_features.append(feature)
            
            if not x_features:
                messagebox.showerror("Error", "No valid input features selected")
                return
            
            y_feature = self.y_features_combo.get()
            if not y_feature or y_feature not in self.feature_list:
                messagebox.showerror("Error", "Please select a valid target feature")
                return
            
            # Validate features
            if y_feature in x_features:
                messagebox.showerror("Error", "Target feature cannot be used as an input feature")
                return
            
            # Check if we have a data file
            if not self.data_file or not os.path.exists(self.data_file):
                messagebox.showerror("Error", f"Please select a valid data file: {self.data_file}")
                return
            
            # Update status
            self.status_var.set("Training model...")
            self.root.update()
            
            # Print debug information
            print(f"\n=== Training Parameters ===")
            print(f"Data File: {self.data_file}")
            print(f"Data File Exists: {os.path.exists(self.data_file)}")
            print(f"X Features: {x_features}")
            print(f"Y Feature: {y_feature}")
            print(f"Hidden Size: {hidden_size}")
            print(f"Learning Rate: {learning_rate}")
            print(f"Batch Size: {batch_size}")
            
            # Run training script with proper arguments
            cmd = [
                sys.executable, "stock_net.py",
                "--hidden_size", str(hidden_size),
                "--learning_rate", str(learning_rate),
                "--batch_size", str(batch_size),
                "--x_features", ",".join(x_features),
                "--y_feature", y_feature,
                "--data_file", self.data_file  # Pass the data file path directly
            ]
            
            # Print command with quotes around arguments
            print(f"\n=== Command ===")
            print(" ".join([f'"{arg}"' if ' ' in arg else arg for arg in cmd]))
            
            # Run the command with PAGER=cat to ensure proper output
            env = os.environ.copy()
            env["PAGER"] = "cat"
            
            # Create a temporary file to capture output
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                print(f"\n=== Running Command ===")
                print(f"Working Directory: {os.path.dirname(os.path.abspath(__file__))}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=temp_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    shell=False
                )
                stdout, _ = process.communicate()
                
                # Read the output from the temporary file
                temp_file.seek(0)
                output = temp_file.read()
                
                # Delete the temporary file
                os.unlink(temp_file.name)
            
            print(f"\n=== Training Output ===")
            print(output)
            
            if process.returncode == 0:
                # Get the latest model directory
                model_dirs = glob.glob("model_*")
                if model_dirs:
                    latest_model_dir = max(model_dirs)
                    self.current_model_dir = latest_model_dir
                    messagebox.showinfo("Success", f"Model training completed successfully!\nModel saved in: {latest_model_dir}")
                    self.refresh_models()
            else:
                messagebox.showerror("Error", f"Training failed:\n{output}")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.status_var.set("Ready")

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
        ttk.Label(train_frame, text="Training Progress:").grid(row=6, column=0, sticky="ew", pady=2)
        progress_frame = ttk.Frame(train_frame)
        progress_frame.grid(row=7, column=0, sticky="nsew", pady=2)
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(0, weight=1)
        self.progress_text = tk.Text(progress_frame, height=10, wrap=tk.WORD)
        self.progress_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(progress_frame, orient="vertical", command=self.progress_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.progress_text.configure(yscrollcommand=scrollbar.set)
        self.progress_text.configure(state="disabled")
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=1, column=0, sticky="ew", pady=10)
        ttk.Button(action_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Make Prediction", command=self.make_prediction).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="View Results", command=self.view_training_results).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Show Visualization", command=self.show_gradient_descent).grid(row=3, column=0, sticky="ew", pady=2)
        status_frame = ttk.Frame(control_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="ew")

    def load_features(self):
        """Load features from the selected data file."""
        try:
            # Clear existing features
            self.x_features_listbox.delete(0, tk.END)
            self.y_features_combo['values'] = []
            self.y_features_combo.set("")
            
            # Load features from CSV file
            df = pd.read_csv(self.data_file)
            self.feature_list = df.columns.tolist()
            
            # Update X features listbox
            for feature in self.feature_list:
                self.x_features_listbox.insert(tk.END, feature)
            
            # Update Y features combobox
            self.y_features_combo['values'] = self.feature_list
            
            # Set default Y feature if available
            if 'close' in self.feature_list:
                self.y_features_combo.set('close')
            
            self.status_var.set("Features loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load features: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            
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

    def _create_display_panel(self):
        """Create the right display panel for plots and results"""
        self.display_frame = ttk.LabelFrame(self.root, text="Display", padding="10")
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.plots_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.plots_tab, text="Plots")
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.metrics_tab, text="Metrics")
        
        # Configure grid weights for display frame
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)
        
    def _train_model(self):
        """Train a new model with the specified parameters"""
        try:
            # Validate inputs
            hidden_size = int(self.hidden_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Get selected features from UI
            x_indices = self.x_features_listbox.curselection()
            if not x_indices:
                messagebox.showerror("Error", "Please select at least one input feature")
                return
                
            self.x_features = [self.x_features_listbox.get(i) for i in x_indices]
            
            self.y_feature = self.y_features_combo.get()
            if not self.y_feature:
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
            
            # Run training script
            cmd = [
                sys.executable, "stock_net.py",
                "--data_file", str(self.data_file),
                "--model_dir", str(self.current_model_dir),
                "--x_features", ",".join(self.x_features),
                "--y_feature", self.y_feature,
                "--hidden_size", str(hidden_size),
                "--learning_rate", str(learning_rate),
                "--batch_size", str(batch_size)
            ]
            
            print(f"\nTraining command: {cmd}")
            
            # Run the command with PAGER=cat to ensure proper output
            env = os.environ.copy()
            env["PAGER"] = "cat"
            
            # Create a temporary file to capture output
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=temp_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )
                stdout, _ = process.communicate()
                
                # Read the output from the temporary file
                temp_file.seek(0)
                output = temp_file.read()
                
                # Delete the temporary file
                os.unlink(temp_file.name)
            
            print(f"\nTraining output:")
            print(output)
            
            if process.returncode == 0:
                messagebox.showinfo("Success", "Model training completed successfully!")
                self.refresh_models()
            else:
                messagebox.showerror("Error", f"Training failed:\n{output}")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.status_var.set("Ready")
            
    def _make_prediction(self):
        """Make predictions using the selected model"""
        if not self.current_model_dir:
            messagebox.showwarning("Warning", "Please select a model first")
            return
            
        try:
            # Get selected features
            if not self.validate_features():
                return
            
            self.status_var.set("Making predictions...")
            self.root.update()
            
            # Run prediction script
            cmd = [
                sys.executable, "predict.py",
                self.data_file,
                "--model_dir", self.current_model_dir,
                "--x_features", ",".join(self.x_features),
                "--y_feature", self.y_feature
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                # Find the predictions file
                pred_files = glob.glob("predictions_*.csv")
                if pred_files:
                    latest_pred = max(pred_files, key=os.path.getctime)
                    self.display_predictions(latest_pred)
                    messagebox.showinfo("Success", "Predictions completed successfully!")
                else:
                    messagebox.showerror("Error", "Predictions file not found")
            else:
                messagebox.showerror("Error", f"Prediction failed:\n{stderr.decode()}")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.status_var.set("Ready")
            
    def _view_training_results(self):
        """View training results and plots"""
        if not self.current_model_dir:
            messagebox.showerror("Error", "Please select a model directory first")
            return
            
        try:
            # Load and display loss curves
            loss_file = os.path.join(self.current_model_dir, "training_losses.csv")
            if os.path.exists(loss_file):
                losses = np.loadtxt(loss_file, delimiter=',')
                
                # Display in the training results tab
                self.train_ax.clear()
                self.train_ax.plot(losses, label='Training Loss')
                self.train_ax.set_title("Training Loss Over Time")
                self.train_ax.set_xlabel("Epoch")
                self.train_ax.set_ylabel("MSE")
                self.train_ax.legend()
                self.train_ax.grid(True)
                self.train_canvas.draw()
                
                # Switch to training results tab (index 0)
                self.switch_to_tab(0)
                
                self.status_var.set("Training results displayed")
            else:
                messagebox.showwarning("Warning", "No training results found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view results: {str(e)}")

    def _show_gradient_descent(self):
        """Show the 3D gradient descent visualization"""
        if not self.current_model_dir:
            messagebox.showerror("Error", "Please select a model directory first")
            return
            
        # Get visualization settings
        color = self.color_var.get()
        point_size = self.point_size_var.get()
        line_width = self.line_width_var.get()
        surface_alpha = self.surface_alpha_var.get()
        
        # Construct the command
        cmd = [
            "python3", "gradient_descent_3d.py",
            "--model_dir", self.current_model_dir,
            "--color", color,
            "--point_size", str(point_size),
            "--line_width", str(line_width),
            "--surface_alpha", str(surface_alpha)
        ]
        
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start visualization: {str(e)}")
            
    def _load_model_info(self):
        """Load and display information about the selected model"""
        if not self.current_model_dir:
            return
            
        metadata_file = os.path.join(self.current_model_dir, "model_metadata.txt")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = f.read()
                
            # Update metrics tab
            for widget in self.metrics_tab.winfo_children():
                widget.destroy()
                
            text = tk.Text(self.metrics_tab, wrap=tk.WORD, padx=10, pady=10)
            text.insert(tk.END, metadata)
            text.config(state=tk.DISABLED)
            text.grid(row=0, column=0, sticky="nsew")
            
            # Load and display plots
            self.load_plots()
            
    def _load_plots(self):
        """Load and display plots from the selected model"""
        if not self.current_model_dir:
            return
            
        plots_dir = os.path.join(self.current_model_dir, "plots")
        if not os.path.exists(plots_dir):
            return
            
        # Clear existing plots
        for widget in self.plots_tab.winfo_children():
            widget.destroy()
            
        # Create a canvas with scrollbars
        canvas_frame = ttk.Frame(self.plots_tab)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for canvas frame
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvas
        canvas = tk.Canvas(canvas_frame)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        # Create scrollbars
        y_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        
        x_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        canvas.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Create a frame inside the canvas to hold the plots
        plots_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=plots_frame, anchor="nw")
        
        # Get plot files
        plot_files = glob.glob(os.path.join(plots_dir, "*.png"))
        if not plot_files:
            return
            
        # Configure grid for plots
        n_cols = 2
        n_rows = (len(plot_files) + 1) // 2
        
        # Set minimum size for plots frame
        min_width = 800  # Minimum width in pixels
        min_height = max(400, n_rows * 300)  # Minimum height based on number of plots
        
        # Store references to prevent garbage collection
        self._canvas = canvas
        self._plots_frame = plots_frame
        self._canvas_window = canvas_window
        
        # Add plots to the frame
        for i, plot_file in enumerate(plot_files):
            row = i // n_cols
            col = i % n_cols
            
            try:
                # Create figure and load plot
                fig = plt.figure(figsize=(6, 4))
                img = plt.imread(plot_file)
                plt.imshow(img)
                plt.axis('off')
                
                # Add to GUI
                plot_canvas = FigureCanvasTkAgg(fig, master=plots_frame)
                plot_canvas.draw()
                plot_widget = plot_canvas.get_tk_widget()
                plot_widget.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Store reference to prevent garbage collection
                setattr(self, f'_plot_canvas_{i}', plot_canvas)
                setattr(self, f'_fig_{i}', fig)
                
                # Configure grid weights for plots frame
                plots_frame.grid_columnconfigure(col, weight=1)
                plots_frame.grid_rowconfigure(row, weight=1)
                
            except Exception as e:
                print(f"Error loading plot {plot_file}: {str(e)}")
                continue
        
        # Update scroll region after all plots are added
        def configure_scroll_region(event=None):
            try:
                # Update the scroll region to encompass the inner frame
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
                    # Set minimum size
                    width = max(min_width, plots_frame.winfo_reqwidth())
                    height = max(min_height, plots_frame.winfo_reqheight())
                    canvas.itemconfig(canvas_window, width=width)
                    plots_frame.configure(width=width, height=height)
            except Exception as e:
                print(f"Error in configure_scroll_region: {str(e)}")
        
        # Configure the canvas to expand with the window
        def on_canvas_configure(event=None):
            try:
                if event and event.width > 1:  # Prevent zero width
                    canvas.itemconfig(canvas_window, width=event.width)
            except Exception as e:
                print(f"Error in on_canvas_configure: {str(e)}")
        
        # Bind events with error handling
        try:
            plots_frame.bind("<Configure>", lambda e: configure_scroll_region())
            canvas.bind("<Configure>", on_canvas_configure)
            
            # Initial configuration
            self.root.update_idletasks()
            configure_scroll_region()
            
        except Exception as e:
            print(f"Error binding events: {str(e)}")
        
        # Make the canvas expand with the window
        self.plots_tab.grid_columnconfigure(0, weight=1)
        self.plots_tab.grid_rowconfigure(0, weight=1)
        
    def _display_predictions(self, pred_file):
        """Display prediction results in the results tab with detailed statistics"""
        try:
            # Clear existing results
            for widget in self.results_tab.winfo_children():
                widget.destroy()
            
            # Create main frame with padding
            main_frame = ttk.Frame(self.results_tab, padding="10")
            main_frame.grid(row=0, column=0, sticky="nsew")
            self.results_tab.grid_columnconfigure(0, weight=1)
            self.results_tab.grid_rowconfigure(0, weight=1)
            
            # Load predictions
            df = pd.read_csv(pred_file)
            
            # Calculate statistics
            stats_frame = ttk.LabelFrame(main_frame, text="Prediction Statistics", padding="5")
            stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            
            # Calculate error metrics
            df['error'] = df['predicted_close'] - df['close']
            df['abs_error'] = abs(df['error'])
            df['pct_error'] = (df['abs_error'] / df['close']) * 100
            
            stats = {
                'Total Predictions': len(df),
                'Mean Absolute Error': f"${df['abs_error'].mean():.2f}",
                'Mean Percentage Error': f"{df['pct_error'].mean():.2f}%",
                'Max Error': f"${df['abs_error'].max():.2f}",
                'Min Error': f"${df['abs_error'].min():.2f}",
                'Average Actual Price': f"${df['close'].mean():.2f}",
                'Average Predicted Price': f"${df['predicted_close'].mean():.2f}"
            }
            
            # Display statistics in a grid
            for i, (label, value) in enumerate(stats.items()):
                row = i // 2
                col = (i % 2) * 2
                ttk.Label(stats_frame, text=label + ":", font=('TkDefaultFont', 9, 'bold')).grid(
                    row=row, column=col, sticky="w", padx=5, pady=2)
                ttk.Label(stats_frame, text=value).grid(
                    row=row, column=col+1, sticky="w", padx=5, pady=2)
            
            # Add top 5 predictions section
            top_predictions_frame = ttk.LabelFrame(main_frame, text="Latest 5 Predictions", padding="5")
            top_predictions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
            
            # Get the last 5 predictions
            last_5 = df.tail(5).copy()
            last_5 = last_5.reset_index(drop=True)
            
            # Create headers
            headers = ['Index', 'Actual Price', 'Predicted Price', 'Error', '% Error']
            for col, header in enumerate(headers):
                ttk.Label(top_predictions_frame, text=header, 
                         font=('TkDefaultFont', 9, 'bold')).grid(
                    row=0, column=col, padx=5, pady=2, sticky="w")
            
            # Display the last 5 predictions
            for i, (_, row) in enumerate(last_5.iterrows()):
                # Format the values
                actual = f"${row['close']:.2f}"
                predicted = f"${row['predicted_close']:.2f}"
                error = f"${row['error']:.2f}"
                pct_error = f"{row['pct_error']:.2f}%"
                
                # Add row with alternating colors
                bg_color = '#f0f0f0' if i % 2 == 0 else '#ffffff'
                frame = ttk.Frame(top_predictions_frame, style='Custom.TFrame')
                frame.grid(row=i+1, column=0, columnspan=5, sticky="ew", padx=2, pady=1)
                frame.configure(style='Custom.TFrame')
                
                # Create a style for the frame
                style = ttk.Style()
                style.configure('Custom.TFrame', background=bg_color)
                
                # Add the values
                ttk.Label(frame, text=str(i+1), 
                         background=bg_color).grid(row=0, column=0, padx=5, pady=2, sticky="w")
                ttk.Label(frame, text=actual, 
                         background=bg_color).grid(row=0, column=1, padx=5, pady=2, sticky="w")
                ttk.Label(frame, text=predicted, 
                         background=bg_color).grid(row=0, column=2, padx=5, pady=2, sticky="w")
                ttk.Label(frame, text=error, 
                         background=bg_color).grid(row=0, column=3, padx=5, pady=2, sticky="w")
                ttk.Label(frame, text=pct_error, 
                         background=bg_color).grid(row=0, column=4, padx=5, pady=2, sticky="w")
            
            # Create frame for the full table
            table_frame = ttk.LabelFrame(main_frame, text="All Predictions", padding="5")
            table_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(2, weight=1)
            
            # Create treeview with scrollbars
            tree = ttk.Treeview(table_frame)
            tree.grid(row=0, column=0, sticky="nsew")
            
            # Add scrollbars
            y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            y_scrollbar.grid(row=0, column=1, sticky="ns")
            
            x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
            x_scrollbar.grid(row=1, column=0, sticky="ew")
            
            tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
            
            # Configure columns
            columns = list(df.columns)
            tree["columns"] = columns
            tree["show"] = "headings"
            
            # Set column headings and widths
            column_widths = {
                'close': 100,
                'predicted_close': 120,
                'error': 100,
                'abs_error': 100,
                'pct_error': 100,
                'open': 100,
                'high': 100,
                'low': 100,
                'vol': 100
            }
            
            for column in columns:
                # Format column heading
                heading = column.replace('_', ' ').title()
                tree.heading(column, text=heading)
                
                # Set column width
                width = column_widths.get(column, 100)
                tree.column(column, width=width, anchor="e" if column in ['error', 'abs_error', 'pct_error'] else "w")
            
            # Add data with formatting
            for i, row in df.iterrows():
                values = []
                for col in columns:
                    if col in ['error', 'abs_error', 'pct_error']:
                        # Format error columns
                        if col == 'pct_error':
                            values.append(f"{row[col]:.2f}%")
                        else:
                            values.append(f"${row[col]:.2f}")
                    elif col in ['close', 'predicted_close', 'open', 'high', 'low']:
                        # Format price columns
                        values.append(f"${row[col]:.2f}")
                    elif col == 'vol':
                        # Format volume with commas
                        values.append(f"{int(row[col]):,}")
                    else:
                        values.append(str(row[col]))
                
                # Add row with alternating colors
                tag = 'even' if i % 2 == 0 else 'odd'
                tree.insert("", "end", values=values, tags=(tag,))
            
            # Configure tags for alternating colors
            tree.tag_configure('even', background='#f0f0f0')
            tree.tag_configure('odd', background='#ffffff')
            
            # Configure grid weights for table frame
            table_frame.grid_columnconfigure(0, weight=1)
            table_frame.grid_rowconfigure(0, weight=1)
            
            # Add a note about the predictions
            note_frame = ttk.Frame(main_frame)
            note_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
            ttk.Label(note_frame, 
                     text="Note: Errors are calculated as (Predicted - Actual). Positive errors indicate overprediction.",
                     font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=0, sticky="w")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying results: {str(e)}")
            print(f"Detailed error: {str(e)}")
            
    def _view_results(self):
        """Open the view_results.py script to display plots"""
        if not self.current_model_dir:
            messagebox.showwarning("Warning", "Please select a model first")
            return
            
        try:
            cmd = [sys.executable, "view_results.py", self.current_model_dir]
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing results: {str(e)}")

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
        """Train a new model with selected parameters (now with live plotting)."""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress. Please wait for it to complete.")
            return
            
        if not self.data_file:
            messagebox.showwarning("Warning", "Please select a data file first")
            return
        if not self.validate_features():
            return
            
        # Set training state
        self.is_training = True
        
        # Disable Train button during training
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, ttk.Button) and btn['text'] == 'Train Model':
                        btn.config(state=tk.DISABLED)
        # Clear previous progress and plot
        self.progress_text.configure(state="normal")
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.configure(state="disabled")
        self.gd_ax.clear()
        
        # Create initial gradient descent landscape
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2 + 0.1 * np.sin(3*X) * np.cos(3*Y)
        
        # Plot the loss landscape
        self.gd_ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', 
                               linewidth=0, antialiased=True)
        self.gd_ax.contour(X, Y, Z, levels=10, offset=0, alpha=0.5)
        
        # Show starting position
        self.gd_ax.scatter([2], [2], [4], c='red', s=100, marker='o', 
                          label='Starting Position')
        self.gd_ax.scatter([0], [0], [0], c='green', s=100, marker='*',
                          label='Global Minimum')
        
        self.gd_ax.set_title("3D Gradient Descent - Ready to Train")
        self.gd_ax.set_xlabel("Weight 1")
        self.gd_ax.set_ylabel("Weight 2")
        self.gd_ax.set_zlabel("Loss")
        self.gd_ax.legend()
        self.gd_ax.view_init(elev=20, azim=45)
        self.gd_canvas.draw()
        
        # Start training in a background thread
        self.training_thread = threading.Thread(target=self.run_training_thread, daemon=True)
        self.training_thread.start()
        
        # Switch to Gradient Descent tab to show live training
        self.switch_to_tab(2)

    def run_training_thread(self):
        # Validate features before starting training
        if not hasattr(self, 'x_features') or not self.x_features:
            self.root.after(0, lambda: messagebox.showerror("Error", "No input features selected. Please select features in the Data tab."))
            self.root.after(0, self.enable_train_button)
            return
            
        if not hasattr(self, 'y_feature') or not self.y_feature:
            self.root.after(0, lambda: messagebox.showerror("Error", "No target feature selected. Please select a target feature in the Data tab."))
            self.root.after(0, self.enable_train_button)
            return
            
        # Prepare data
        df = pd.read_csv(self.data_file)
        x_features = self.x_features
        y_feature = self.y_feature
        
        # Additional validation
        if y_feature not in df.columns:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Target feature '{y_feature}' not found in data file."))
            self.root.after(0, self.enable_train_button)
            return
            
        for feature in x_features:
            if feature not in df.columns:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Input feature '{feature}' not found in data file."))
                self.root.after(0, self.enable_train_button)
                return
        
        X = df[x_features].values
        Y = df[y_feature].values.reshape(-1, 1)
        # Normalize
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        Y_min = np.min(Y)
        Y_max = np.max(Y)
        Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
        # Split
        # Simple train-test split implementation
        n_samples = X_norm.shape[0]
        n_test = int(0.2 * n_samples)
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X_norm[train_indices]
        X_test = X_norm[test_indices]
        Y_train = Y_norm[train_indices]
        Y_test = Y_norm[test_indices]
        
        # Inline StockNet implementation
        class StockNet:
            def __init__(self, input_size, hidden_size):
                self.input_size = input_size
                self.hidden_size = hidden_size
                # Initialize weights with Xavier/Glorot initialization
                self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
                self.b2 = np.zeros((1, 1))
            
            def forward(self, X):
                # Forward pass
                self.z1 = np.dot(X, self.W1) + self.b1
                self.a1 = np.tanh(self.z1)  # tanh activation
                self.z2 = np.dot(self.a1, self.W2) + self.b2
                return self.z2
            
            def backward(self, X, y, output, learning_rate):
                # Backward pass
                m = X.shape[0]
                
                # Output layer gradients
                dz2 = output - y
                dW2 = np.dot(self.a1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m
                
                # Hidden layer gradients
                dz1 = np.dot(dz2, self.W2.T) * (1 - np.tanh(self.z1)**2)
                dW1 = np.dot(X.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m
                
                # Update weights
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
            
            def save_weights(self, model_dir, prefix="stock_model"):
                # Save weights to files
                np.save(os.path.join(model_dir, f"{prefix}_W1.npy"), self.W1)
                np.save(os.path.join(model_dir, f"{prefix}_b1.npy"), self.b1)
                np.save(os.path.join(model_dir, f"{prefix}_W2.npy"), self.W2)
                np.save(os.path.join(model_dir, f"{prefix}_b2.npy"), self.b2)
        
        # Model params
        hidden_size = int(self.hidden_size_var.get())
        learning_rate = float(self.learning_rate_var.get())
        batch_size = int(self.batch_size_var.get())
        epochs = 1000
        # Create model
        model = StockNet(input_size=len(x_features), hidden_size=hidden_size)
        train_losses = []
        best_mse = float('inf')
        patience = 20
        patience_counter = 0
        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            total_mse = 0
            n_batches = 0
            for start_idx in range(0, X_train.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, X_train.shape[0])
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
            # Early stopping
            if avg_train_mse < best_mse:
                best_mse = avg_train_mse
                patience_counter = 0
            else:
                patience_counter += 1
            # Update progress and plot every epoch for better visualization
            if epoch % 1 == 0 or epoch == epochs - 1:
                print(f"Updating plot at epoch {epoch}, loss: {avg_train_mse:.6f}")
                self.root.after(0, self.update_live_plot, list(range(1, len(train_losses)+1)), train_losses)
                self.root.after(0, self.update_progress, f"Epoch {epoch}, MSE: {avg_train_mse:.6f}")
            if patience_counter >= patience:
                self.root.after(0, self.update_progress, f"Early stopping at epoch {epoch}")
                break
        # Save model and results (as in stock_net.py)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.current_model_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        np.savetxt(os.path.join(model_dir, 'training_losses.csv'), train_losses, delimiter=',')
        # Save weights
        model.save_weights(model_dir, prefix="stock_model")
        # Save feature info
        feature_info = {'x_features': x_features, 'y_feature': y_feature, 'input_size': len(x_features)}
        with open(os.path.join(model_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f)
        # Save normalization
        np.savetxt(os.path.join(model_dir, 'scaler_mean.csv'), X_min, delimiter=',')
        np.savetxt(os.path.join(model_dir, 'scaler_std.csv'), X_max - X_min, delimiter=',')
        np.savetxt(os.path.join(model_dir, 'target_min.csv'), np.array([Y_min]).reshape(1, -1), delimiter=',')
        np.savetxt(os.path.join(model_dir, 'target_max.csv'), np.array([Y_max]).reshape(1, -1), delimiter=',')
        # Re-enable Train button
        self.root.after(0, self.enable_train_button)
        self.root.after(0, self.status_var.set, f"Model trained: {os.path.basename(model_dir)}")
        self.root.after(0, self.refresh_models)
        self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed successfully!"))

    def update_live_plot(self, epochs, losses):
        """Update the live plot with training progress"""
        print(f"update_live_plot called with {len(epochs)} epochs, {len(losses)} losses")
        try:
            # Ensure epochs and losses have the same length
            min_length = min(len(epochs), len(losses))
            epochs_plot = epochs[:min_length]
            losses_plot = losses[:min_length]
            
            print(f"Plotting {len(epochs_plot)} points, latest loss: {losses_plot[-1] if losses_plot else 'N/A'}")
            
            self.gd_ax.clear()
            
            # Simple 2D fallback plot to ensure something is always shown
            if len(epochs_plot) > 0:
                self.gd_ax.plot(epochs_plot, losses_plot, 'r-', linewidth=2, label='Training Loss')
                self.gd_ax.set_title(f"Training Loss - Epoch {epochs_plot[-1]}")
                self.gd_ax.set_xlabel("Epoch")
                self.gd_ax.set_ylabel("Loss")
                self.gd_ax.legend()
                self.gd_ax.grid(True)
                self.gd_canvas.draw()
                print("2D plot updated successfully")
                return
            
            # Create a 3D gradient descent visualization
            # Show the loss landscape and current position
            
            # Create a meshgrid for the loss landscape
            x = np.linspace(-2, 2, 50)
            y = np.linspace(-2, 2, 50)
            X, Y = np.meshgrid(x, y)
            
            # Create a simple loss landscape (bowl-shaped)
            # This represents the loss function in weight space
            Z = X**2 + Y**2 + 0.1 * np.sin(3*X) * np.cos(3*Y)
            
            # Plot the loss landscape
            self.gd_ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis', 
                                   linewidth=0, antialiased=True)
            
            # Add contour lines on the surface
            self.gd_ax.contour(X, Y, Z, levels=10, offset=0, alpha=0.5)
            
            # Show the training trajectory
            if len(epochs_plot) > 1:
                # Create a trajectory that moves toward the minimum
                # Simulate weight updates moving toward the center (minimum)
                progress = np.array(epochs_plot) / max(epochs_plot)
                
                # Create a spiral trajectory toward the minimum
                radius = 2 * (1 - progress)
                angle = 10 * progress * 2 * np.pi
                
                traj_x = radius * np.cos(angle)
                traj_y = radius * np.sin(angle)
                traj_z = radius**2 + 0.1 * np.sin(3*traj_x) * np.cos(3*traj_y)
                
                # Plot the training trajectory
                self.gd_ax.plot(traj_x, traj_y, traj_z, 'r-', linewidth=3, 
                               label='Training Path')
                
                # Highlight current position
                current_x, current_y, current_z = traj_x[-1], traj_y[-1], traj_z[-1]
                self.gd_ax.scatter([current_x], [current_y], [current_z], 
                                  c='red', s=100, marker='o', 
                                  label=f'Current (Loss: {losses_plot[-1]:.4f})')
                
                # Show the minimum point
                self.gd_ax.scatter([0], [0], [0], c='green', s=100, marker='*',
                                  label='Global Minimum')
            
            self.gd_ax.set_title(f"3D Gradient Descent - Epoch {epochs_plot[-1] if epochs_plot else 0}")
            self.gd_ax.set_xlabel("Weight 1")
            self.gd_ax.set_ylabel("Weight 2") 
            self.gd_ax.set_zlabel("Loss")
            self.gd_ax.legend()
            
            # Set consistent view angle
            self.gd_ax.view_init(elev=20, azim=45)
            
            self.gd_canvas.draw()
            
        except Exception as e:
            print(f"Error updating live plot: {e}")
            # Fallback to simple 2D plot if 3D fails
            try:
                self.gd_ax.clear()
                self.gd_ax.plot(epochs_plot, losses_plot, label='Training Loss')
                self.gd_ax.set_title("Live Training Loss")
                self.gd_ax.set_xlabel("Epoch")
                self.gd_ax.set_ylabel("Loss")
                self.gd_ax.legend()
                self.gd_canvas.draw()
            except:
                pass

    def update_progress(self, message):
        """Update the progress text area with a new message"""
        self.progress_text.configure(state="normal")
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)  # Auto-scroll to bottom
        self.progress_text.configure(state="disabled")
        self.root.update_idletasks()  # Force GUI update

    def enable_train_button(self):
        # Reset training state
        self.is_training = False
        self.training_thread = None
        
        # Re-enable train button
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, ttk.Button) and btn['text'] == 'Train Model':
                        btn.config(state=tk.NORMAL)

    def make_prediction(self):
        """Make predictions using the selected model"""
        if not self.current_model_dir:
            messagebox.showerror("Error", "Please select a model directory first")
            return
            
        if not self.data_file:
            messagebox.showerror("Error", "Please select a data file first")
            return
            
        if not self.validate_features():
            return
            
        try:
            self.status_var.set("Making predictions...")
            self.root.update()
            
            cmd = [
                sys.executable, "predict.py",
                self.data_file,
                "--model_dir", self.current_model_dir,
                "--x_features", ",".join(self.x_features),
                "--y_feature", self.y_feature
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                messagebox.showerror("Error", f"Prediction failed:\n{stderr.decode()}")
            else:
                self.view_results()
            
            self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Error: Prediction failed")

    def view_training_results(self):
        """View training results and plots"""
        if not self.current_model_dir:
            messagebox.showerror("Error", "Please select a model directory first")
            return
            
        try:
            # Load and display loss curves
            loss_file = os.path.join(self.current_model_dir, "training_losses.csv")
            if os.path.exists(loss_file):
                losses = np.loadtxt(loss_file, delimiter=',')
                
                # Display in the training results tab
                self.train_ax.clear()
                self.train_ax.plot(losses, label='Training Loss')
                self.train_ax.set_title("Training Loss Over Time")
                self.train_ax.set_xlabel("Epoch")
                self.train_ax.set_ylabel("MSE")
                self.train_ax.legend()
                self.train_ax.grid(True)
                self.train_canvas.draw()
                
                # Switch to training results tab (index 0)
                self.switch_to_tab(0)
                
                self.status_var.set("Training results displayed")
            else:
                messagebox.showwarning("Warning", "No training results found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view results: {str(e)}")

    def show_gradient_descent(self):
        """Show 3D gradient descent visualization"""
        if not self.current_model_dir:
            messagebox.showerror("Error", "Please select a model directory first")
            return
            
        try:
            cmd = [
                sys.executable, "gradient_descent_3d.py",
                "--model_dir", self.current_model_dir
            ]
            
            process = subprocess.Popen(cmd)
            process.wait()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show visualization: {str(e)}")

    def load_model_info(self):
        """Load model information from selected model"""
        if not self.current_model_dir:
            return
            
        # Load feature info
        feature_info_path = os.path.join(self.current_model_dir, 'feature_info.json')
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
            self.current_model_dir = os.path.join(self.current_model_dir, model_name)
            
            # Load feature info from selected model
            feature_info_path = os.path.join(self.current_model_dir, 'feature_info.json')
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

    def refresh_models(self):
        """Refresh the list of available models"""
        try:
            # Clear existing list
            self.model_listbox.delete(0, tk.END)
            
            # Check if model directory exists
            if not os.path.exists(self.current_model_dir):
                os.makedirs(self.current_model_dir, exist_ok=True)
                self.status_var.set(f"Created models directory: {self.current_model_dir}")
                return
                
            # Get list of model directories
            model_dirs = sorted([
                d for d in os.listdir(self.current_model_dir)
                if os.path.isdir(os.path.join(self.current_model_dir, d))
            ], reverse=True)  # Sort by name (most recent first)
            
            # Add models to listbox
            for model_dir in model_dirs:
                self.model_listbox.insert(tk.END, model_dir)
                
            # If there are any models, select the first one
            if model_dirs:
                self.model_listbox.selection_set(0)
            else:
                self.status_var.set("No models available")
                
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
                model_path = os.path.join(self.current_model_dir, model_name)
                shutil.rmtree(model_path)
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
        else:
            self.lock_button.config(text="Lock")
            self.status_var.set("Features unlocked")
        
        # Update the listbox state
        self.update_feature_lock_state()

    def update_feature_lock_state(self):
        """Update the listbox to show locked features"""
        if self.features_locked:
            # Restore locked selections
            self.x_features_listbox.selection_clear(0, tk.END)
            for index in self.locked_features:
                self.x_features_listbox.selection_set(index)
        
        # Update the listbox state
        self.x_features_listbox.config(state=tk.DISABLED if self.features_locked else tk.NORMAL)

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

    def view_results(self):
        """View prediction results"""
        # Find the most recent prediction file
        prediction_files = glob.glob("predictions_*.csv")
        if not prediction_files:
            messagebox.showwarning("Warning", "No prediction files found")
            return
            
        # Get the most recent file
        latest_prediction = max(prediction_files, key=os.path.getctime)
        
        try:
            # Load and display predictions
            df = pd.read_csv(latest_prediction)
            
            # Display in the prediction results tab
            self.pred_ax.clear()
            
            if 'actual' in df.columns and 'predicted' in df.columns:
                # Plot actual vs predicted
                self.pred_ax.plot(df['actual'], label='Actual', alpha=0.7)
                self.pred_ax.plot(df['predicted'], label='Predicted', alpha=0.7)
                self.pred_ax.set_title("Actual vs Predicted Values")
                self.pred_ax.set_xlabel("Sample")
                self.pred_ax.set_ylabel("Value")
                self.pred_ax.legend()
            else:
                # Plot predictions only
                self.pred_ax.plot(df.iloc[:, -1], label='Predictions')
                self.pred_ax.set_title("Predictions")
                self.pred_ax.set_xlabel("Sample")
                self.pred_ax.set_ylabel("Value")
                self.pred_ax.legend()
            
            self.pred_ax.grid(True)
            self.pred_canvas.draw()
            
            # Switch to prediction results tab (index 1)
            self.switch_to_tab(1)
            
            self.status_var.set(f"Displaying results from {latest_prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")

    def switch_to_tab(self, tab_index):
        """Switch to a specific tab in the display notebook"""
        if hasattr(self, 'display_notebook') and tab_index < self.display_notebook.index('end'):
            self.display_notebook.select(tab_index)

def main():
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
