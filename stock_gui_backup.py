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

class StockPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction System")
        self.root.geometry("1300x800")
        
        # Initialize variables first
        self.current_model_dir = None
        self.data_file = "/Users/porupine/redline/data/gamestop_us.csv"
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self.create_control_panel()
        self.create_display_panel()
        
    def create_control_panel(self):
        """Create the left control panel with buttons and inputs"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Data Selection
        ttk.Label(control_frame, text="Data File:").grid(row=0, column=0, sticky="w", pady=5)
        self.data_path_var = tk.StringVar(value=self.data_file)
        ttk.Entry(control_frame, textvariable=self.data_path_var, width=30).grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_data_file).grid(row=1, column=1, padx=5)
        
        # Training Controls
        training_frame = ttk.LabelFrame(control_frame, text="Training", padding="5")
        training_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(training_frame, text="Hidden Layer Size:").grid(row=0, column=0, sticky="w")
        self.hidden_size_var = tk.StringVar(value="4")
        ttk.Entry(training_frame, textvariable=self.hidden_size_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(training_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        self.learning_rate_var = tk.StringVar(value="0.001")
        ttk.Entry(training_frame, textvariable=self.learning_rate_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(training_frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(training_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Button(training_frame, text="Train Model", command=self.train_model).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Model Selection
        model_frame = ttk.LabelFrame(control_frame, text="Model Selection", padding="5")
        model_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.model_listbox = tk.Listbox(model_frame, height=5)
        self.model_listbox.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        self.model_listbox.bind('<<ListboxSelect>>', self.on_model_select)
        
        ttk.Button(model_frame, text="Refresh Models", command=self.refresh_models).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Prediction Controls
        prediction_frame = ttk.LabelFrame(control_frame, text="Prediction", padding="5")
        prediction_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Button(prediction_frame, text="Make Predictions", command=self.make_predictions).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Button(prediction_frame, text="View Results", command=self.view_results).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, relief="sunken").grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Initial model refresh
        self.refresh_models()
        
    def create_display_panel(self):
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
        
    def browse_data_file(self):
        """Open file dialog to select data file"""
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_path_var.set(filename)
            self.data_file = filename
            
    def refresh_models(self):
        """Refresh the list of available models"""
        self.model_listbox.delete(0, tk.END)
        model_dirs = glob.glob("model_*")
        model_dirs.sort(reverse=True)  # Most recent first
        for model_dir in model_dirs:
            self.model_listbox.insert(tk.END, model_dir)
            
    def on_model_select(self, event):
        """Handle model selection from listbox"""
        selection = self.model_listbox.curselection()
        if selection:
            self.current_model_dir = self.model_listbox.get(selection[0])
            self.load_model_info()
            
    def load_model_info(self):
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
            
    def load_plots(self):
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
        
        # Configure canvas
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
        
    def train_model(self):
        """Train a new model with the specified parameters"""
        try:
            # Validate inputs
            hidden_size = int(self.hidden_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Update status
            self.status_var.set("Training model...")
            self.root.update()
            
            # Run training script
            cmd = [
                sys.executable, "stock_net.py",
                "--hidden_size", str(hidden_size),
                "--learning_rate", str(learning_rate),
                "--batch_size", str(batch_size)
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                messagebox.showinfo("Success", "Model training completed successfully!")
                self.refresh_models()
            else:
                messagebox.showerror("Error", f"Training failed:\n{stderr.decode()}")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input parameters: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.status_var.set("Ready")
            
    def make_predictions(self):
        """Make predictions using the selected model"""
        if not self.current_model_dir:
            messagebox.showwarning("Warning", "Please select a model first")
            return
            
        try:
            self.status_var.set("Making predictions...")
            self.root.update()
            
            # Run prediction script
            cmd = [
                sys.executable, "predict.py",
                self.data_file,
                "--model_dir", self.current_model_dir
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
            
    def display_predictions(self, pred_file):
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
            
    def view_results(self):
        """Open the view_results.py script to display plots"""
        if not self.current_model_dir:
            messagebox.showwarning("Warning", "Please select a model first")
            return
            
        try:
            cmd = [sys.executable, "view_results.py", self.current_model_dir]
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing results: {str(e)}")

def main():
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
