"""
Enhanced Data Visualization and Analysis Tool

This module provides comprehensive data visualization capabilities with:
- Interactive plotting with matplotlib
- Tkinter GUI integration
- Data analysis and processing
- File handling and export capabilities
"""

# Standard library imports
import os
import sys
import glob
import json
import subprocess
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Third-party imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.gridspec import GridSpec
    from matplotlib.figure import Figure
    import seaborn as sns
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install numpy pandas matplotlib seaborn pillow")
    sys.exit(1)

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, scrolledtext
except ImportError as e:
    print(f"Error importing tkinter: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure matplotlib for better display
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Constants
TEXT_COLOR = "#000000"
FRAME_COLOR = "#F0F0F0"
BACKGROUND_COLOR = "#FFFFFF"
ACCENT_COLOR = "#007ACC"
ERROR_COLOR = "#DC3545"
SUCCESS_COLOR = "#28A745"
WARNING_COLOR = "#FFC107"

# Plot themes
PLOT_THEMES = {
    'default': 'default',
    'seaborn': 'seaborn-v0_8',
    'ggplot': 'ggplot',
    'bmh': 'bmh',
    'classic': 'classic'
}

# Supported file formats
SUPPORTED_FORMATS = {
    'data': ['.csv', '.xlsx', '.json', '.parquet'],
    'images': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
    'plots': ['.png', '.pdf', '.svg', '.eps']
}

class DataVisualizationApp:
    """Main application class for data visualization."""
    
    def __init__(self, root):
        """Initialize the application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Enhanced Data Visualization Tool")
        self.root.geometry("1400x900")
        
        # Configure window
        self.setup_window()
        
        # Initialize variables
        self.current_data = None
        self.current_plot = None
        self.plot_history = []
        
        # Create GUI
        self.create_gui()
        
        logger.info("Data visualization application initialized")
    
    def setup_window(self):
        """Configure the main window."""
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Set window properties
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.minsize(800, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('TFrame', background=FRAME_COLOR)
        self.style.configure('TLabel', background=FRAME_COLOR, foreground=TEXT_COLOR)
        self.style.configure('TButton', background=ACCENT_COLOR)
    
    def create_gui(self):
        """Create the main GUI components."""
        # Control panel (left)
        self.create_control_panel()
        
        # Display panel (right)
        self.create_display_panel()
        
        logger.info("GUI components created")
    
    def create_control_panel(self):
        """Create the left control panel."""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights for control frame
        control_frame.grid_columnconfigure(0, weight=1)
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.grid(row=0, column=0, sticky="ew", pady=5)
        file_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Save Plot", command=self.save_plot).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(file_frame, text="Export Data", command=self.export_data).grid(row=2, column=0, sticky="ew", pady=2)
        
        # Plot controls
        plot_frame = ttk.LabelFrame(control_frame, text="Plot Controls", padding="5")
        plot_frame.grid(row=1, column=0, sticky="ew", pady=5)
        plot_frame.grid_columnconfigure(0, weight=1)
        
        # Plot type selection
        ttk.Label(plot_frame, text="Plot Type:").grid(row=0, column=0, sticky="w", pady=2)
        self.plot_type_var = tk.StringVar(value="Line Plot")
        plot_types = ["Line Plot", "Scatter Plot", "Bar Plot", "Histogram", "Box Plot", "Heatmap"]
        plot_combo = ttk.Combobox(plot_frame, textvariable=self.plot_type_var, values=plot_types)
        plot_combo.grid(row=1, column=0, sticky="ew", pady=2)
        plot_combo.bind('<<ComboboxSelected>>', self.on_plot_type_change)
        
        # Theme selection
        ttk.Label(plot_frame, text="Theme:").grid(row=2, column=0, sticky="w", pady=2)
        self.theme_var = tk.StringVar(value="default")
        theme_combo = ttk.Combobox(plot_frame, textvariable=self.theme_var, values=list(PLOT_THEMES.keys()))
        theme_combo.grid(row=3, column=0, sticky="ew", pady=2)
        theme_combo.bind('<<ComboboxSelected>>', self.on_theme_change)
        
        # Action buttons
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding="5")
        action_frame.grid(row=2, column=0, sticky="ew", pady=5)
        action_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(action_frame, text="Create Plot", command=self.create_plot).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Clear Plot", command=self.clear_plot).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(action_frame, text="Refresh", command=self.refresh_display).grid(row=2, column=0, sticky="ew", pady=2)
    
    def create_display_panel(self):
        """Create the right display panel."""
        display_frame = ttk.LabelFrame(self.root, text="Display", padding="10")
        display_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, display_frame)
        self.toolbar.grid(row=1, column=0, sticky="ew")
        self.toolbar.update()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(display_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.grid(row=2, column=0, sticky="ew", pady=(5, 0))
    
    def load_data(self):
        """Load data from file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    self.current_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.current_data = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    self.current_data = pd.read_json(file_path)
                else:
                    messagebox.showerror("Error", "Unsupported file format")
                    return
                
                self.status_var.set(f"Loaded {len(self.current_data)} rows from {os.path.basename(file_path)}")
                logger.info(f"Data loaded: {len(self.current_data)} rows from {file_path}")
                
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def create_plot(self):
        """Create a plot based on current data and settings."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
        
        try:
            # Clear previous plot
            self.ax.clear()
            
            plot_type = self.plot_type_var.get()
            
            if plot_type == "Line Plot":
                self.current_data.plot(ax=self.ax, kind='line')
            elif plot_type == "Scatter Plot":
                if len(self.current_data.columns) >= 2:
                    self.ax.scatter(self.current_data.iloc[:, 0], self.current_data.iloc[:, 1])
                    self.ax.set_xlabel(self.current_data.columns[0])
                    self.ax.set_ylabel(self.current_data.columns[1])
            elif plot_type == "Bar Plot":
                self.current_data.plot(ax=self.ax, kind='bar')
            elif plot_type == "Histogram":
                self.current_data.hist(ax=self.ax, bins=30)
            elif plot_type == "Box Plot":
                self.current_data.plot(ax=self.ax, kind='box')
            elif plot_type == "Heatmap":
                sns.heatmap(self.current_data.corr(), ax=self.ax, annot=True, cmap='coolwarm')
            
            self.ax.set_title(f"{plot_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.ax.grid(True, alpha=0.3)
            
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_var.set(f"Created {plot_type}")
            logger.info(f"Plot created: {plot_type}")
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def save_plot(self):
        """Save the current plot to file."""
        if self.current_plot is None:
            messagebox.showwarning("Warning", "No plot to save")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Plot",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved to {os.path.basename(file_path)}")
                logger.info(f"Plot saved: {file_path}")
                
        except Exception as e:
            error_msg = f"Error saving plot: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def export_data(self):
        """Export current data to file."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Data",
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    self.current_data.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    self.current_data.to_excel(file_path, index=False)
                elif file_path.endswith('.json'):
                    self.current_data.to_json(file_path, orient='records')
                
                self.status_var.set(f"Data exported to {os.path.basename(file_path)}")
                logger.info(f"Data exported: {file_path}")
                
        except Exception as e:
            error_msg = f"Error exporting data: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def clear_plot(self):
        """Clear the current plot."""
        self.ax.clear()
        self.ax.set_title("Plot Cleared")
        self.canvas.draw()
        self.status_var.set("Plot cleared")
        logger.info("Plot cleared")
    
    def refresh_display(self):
        """Refresh the display."""
        self.canvas.draw()
        self.status_var.set("Display refreshed")
        logger.info("Display refreshed")
    
    def on_plot_type_change(self, event=None):
        """Handle plot type change."""
        plot_type = self.plot_type_var.get()
        self.status_var.set(f"Plot type changed to: {plot_type}")
        logger.info(f"Plot type changed: {plot_type}")
    
    def on_theme_change(self, event=None):
        """Handle theme change."""
        theme = self.theme_var.get()
        plt.style.use(PLOT_THEMES.get(theme, 'default'))
        self.status_var.set(f"Theme changed to: {theme}")
        logger.info(f"Theme changed: {theme}")

def main():
    """Main function to run the application."""
    try:
        root = tk.Tk()
        app = DataVisualizationApp(root)
        logger.info("Starting application")
        root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Application error: {e}")

if __name__ == "__main__":
    main() 