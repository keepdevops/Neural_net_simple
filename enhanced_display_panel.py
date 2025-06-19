"""
Enhanced Display Panel for Data Visualization Application

This module provides a comprehensive display panel with multiple tabs for:
- Training results visualization
- Saved plots management
- Comprehensive analysis plots
- Raw PNG image viewing
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import os
import glob
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedDisplayPanel:
    """Enhanced display panel with multiple visualization tabs."""
    
    def __init__(self, root):
        """Initialize the display panel.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.current_model_dir = None
        self.plot_images = []  # Keep references to prevent garbage collection
        
        # Create the main notebook
        self.create_main_notebook()
        
        logger.info("Enhanced display panel initialized")
    
    def create_main_notebook(self):
        """Create the main notebook with all tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create individual tabs
        self.create_training_results_tab()
        self.create_saved_plots_tab()
        self.create_comprehensive_plots_tab()
        self.create_raw_pngs_tab()
        
        # Store tab switching function
        self.switch_to_tab = lambda index: self.notebook.select(index)
        
        logger.info("Main notebook created with all tabs")
    
    def create_training_results_tab(self):
        """Create the training results tab with matplotlib canvas."""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training Results")
        
        # Configure grid weights
        training_frame.grid_columnconfigure(0, weight=1)
        training_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure and canvas
        self.training_fig = Figure(figsize=(10, 7), dpi=100)
        self.training_ax = self.training_fig.add_subplot(111)
        
        # Create canvas with toolbar
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, master=training_frame)
        self.training_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add navigation toolbar
        self.training_toolbar = NavigationToolbar2Tk(self.training_canvas, training_frame)
        self.training_toolbar.grid(row=1, column=0, sticky="ew")
        self.training_toolbar.update()
        
        # Initialize with placeholder
        self.training_ax.text(0.5, 0.5, 'Training results will appear here', 
                             ha='center', va='center', transform=self.training_ax.transAxes,
                             fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        self.training_ax.set_title("Training Results")
        self.training_canvas.draw()
        
        logger.info("Training results tab created")
    
    def create_saved_plots_tab(self):
        """Create the saved plots tab with sub-tabs."""
        saved_plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(saved_plots_frame, text="Saved Plots")
        
        # Configure grid weights
        saved_plots_frame.grid_columnconfigure(0, weight=1)
        saved_plots_frame.grid_rowconfigure(0, weight=1)
        
        # Create sub-notebook for different plot types
        self.plots_notebook = ttk.Notebook(saved_plots_frame)
        self.plots_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create sub-tabs
        self.create_comprehensive_plots_subtab()
        self.create_raw_pngs_subtab()
        
        logger.info("Saved plots tab created with sub-tabs")
    
    def create_comprehensive_plots_subtab(self):
        """Create the comprehensive plots sub-tab."""
        comprehensive_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(comprehensive_frame, text="Comprehensive Plots")
        
        # Configure grid weights
        comprehensive_frame.grid_columnconfigure(0, weight=1)
        comprehensive_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for comprehensive plots
        self.comprehensive_fig = Figure(figsize=(16, 12), dpi=100)
        self.comprehensive_ax = self.comprehensive_fig.add_subplot(111)
        
        # Create canvas with toolbar
        self.comprehensive_canvas = FigureCanvasTkAgg(self.comprehensive_fig, master=comprehensive_frame)
        self.comprehensive_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add navigation toolbar
        self.comprehensive_toolbar = NavigationToolbar2Tk(self.comprehensive_canvas, comprehensive_frame)
        self.comprehensive_toolbar.grid(row=1, column=0, sticky="ew")
        self.comprehensive_toolbar.update()
        
        # Initialize with placeholder
        self.comprehensive_ax.text(0.5, 0.5, 'Comprehensive analysis plots will appear here', 
                                  ha='center', va='center', transform=self.comprehensive_ax.transAxes,
                                  fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.comprehensive_ax.set_title("Comprehensive Analysis")
        self.comprehensive_canvas.draw()
        
        logger.info("Comprehensive plots sub-tab created")
    
    def create_raw_pngs_subtab(self):
        """Create the raw PNGs sub-tab with scrollable canvas."""
        raw_pngs_frame = ttk.Frame(self.plots_notebook)
        self.plots_notebook.add(raw_pngs_frame, text="Raw PNGs")
        
        # Configure grid weights
        raw_pngs_frame.grid_columnconfigure(0, weight=1)
        raw_pngs_frame.grid_rowconfigure(0, weight=1)
        
        # Create scrollable canvas for images
        self.saved_plots_canvas = tk.Canvas(raw_pngs_frame, bg="#F0F0F0")
        self.saved_plots_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(raw_pngs_frame, orient="vertical", command=self.saved_plots_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.saved_plots_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas to hold images
        self.saved_plots_inner_frame = ttk.Frame(self.saved_plots_canvas)
        self.saved_plots_canvas.create_window((0, 0), window=self.saved_plots_inner_frame, anchor="nw")
        
        # Add placeholder label
        self.saved_plots_placeholder = ttk.Label(
            self.saved_plots_inner_frame, 
            text="Select a model to view saved plots", 
            foreground="#000000", 
            background="#F0F0F0",
            font=("Arial", 12)
        )
        self.saved_plots_placeholder.pack(pady=50)
        
        # Bind canvas resizing
        self.saved_plots_inner_frame.bind(
            "<Configure>", 
            lambda e: self.saved_plots_canvas.configure(scrollregion=self.saved_plots_canvas.bbox("all"))
        )
        
        # Bind mouse wheel for scrolling
        self.saved_plots_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        logger.info("Raw PNGs sub-tab created")
    
    def create_comprehensive_plots_tab(self):
        """Create a separate comprehensive plots tab (alternative to sub-tab)."""
        comprehensive_frame = ttk.Frame(self.notebook)
        self.notebook.add(comprehensive_frame, text="Comprehensive Analysis")
        
        # Configure grid weights
        comprehensive_frame.grid_columnconfigure(0, weight=1)
        comprehensive_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for comprehensive analysis
        self.analysis_fig = Figure(figsize=(14, 10), dpi=100)
        self.analysis_ax = self.analysis_fig.add_subplot(111)
        
        # Create canvas with toolbar
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, master=comprehensive_frame)
        self.analysis_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Add navigation toolbar
        self.analysis_toolbar = NavigationToolbar2Tk(self.analysis_canvas, comprehensive_frame)
        self.analysis_toolbar.grid(row=1, column=0, sticky="ew")
        self.analysis_toolbar.update()
        
        # Initialize with placeholder
        self.analysis_ax.text(0.5, 0.5, 'Comprehensive analysis will appear here', 
                             ha='center', va='center', transform=self.analysis_ax.transAxes,
                             fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.analysis_ax.set_title("Comprehensive Analysis")
        self.analysis_canvas.draw()
        
        logger.info("Comprehensive analysis tab created")
    
    def create_raw_pngs_tab(self):
        """Create a separate raw PNGs tab (alternative to sub-tab)."""
        raw_pngs_frame = ttk.Frame(self.notebook)
        self.notebook.add(raw_pngs_frame, text="Raw Images")
        
        # Configure grid weights
        raw_pngs_frame.grid_columnconfigure(0, weight=1)
        raw_pngs_frame.grid_rowconfigure(0, weight=1)
        
        # Create scrollable canvas for images
        self.raw_images_canvas = tk.Canvas(raw_pngs_frame, bg="#F0F0F0")
        self.raw_images_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(raw_pngs_frame, orient="vertical", command=self.raw_images_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.raw_images_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create frame inside canvas to hold images
        self.raw_images_inner_frame = ttk.Frame(self.raw_images_canvas)
        self.raw_images_canvas.create_window((0, 0), window=self.raw_images_inner_frame, anchor="nw")
        
        # Add placeholder label
        self.raw_images_placeholder = ttk.Label(
            self.raw_images_inner_frame, 
            text="Raw images will appear here", 
            foreground="#000000", 
            background="#F0F0F0",
            font=("Arial", 12)
        )
        self.raw_images_placeholder.pack(pady=50)
        
        # Bind canvas resizing
        self.raw_images_inner_frame.bind(
            "<Configure>", 
            lambda e: self.raw_images_canvas.configure(scrollregion=self.raw_images_canvas.bbox("all"))
        )
        
        # Bind mouse wheel for scrolling
        self.raw_images_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        logger.info("Raw images tab created")
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for canvases."""
        try:
            canvas = event.widget
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception as e:
            logger.warning(f"Mouse wheel scrolling failed: {e}")
    
    def load_saved_plots(self, model_dir: str):
        """Load saved plots from a model directory.
        
        Args:
            model_dir: Path to the model directory
        """
        try:
            self.current_model_dir = model_dir
            plots_dir = os.path.join(model_dir, 'plots')
            
            if not os.path.exists(plots_dir):
                logger.warning(f"Plots directory not found: {plots_dir}")
                self.show_no_plots_message()
                return
            
            # Find PNG files
            png_files = glob.glob(os.path.join(plots_dir, '*.png'))
            png_files.sort()  # Sort for consistent ordering
            
            if not png_files:
                logger.warning(f"No PNG files found in: {plots_dir}")
                self.show_no_plots_message()
                return
            
            # Clear existing images
            self.clear_saved_plots()
            
            # Load and display images
            self.display_png_images(png_files)
            
            logger.info(f"Loaded {len(png_files)} plots from {model_dir}")
            
        except Exception as e:
            error_msg = f"Error loading saved plots: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def clear_saved_plots(self):
        """Clear all saved plot images."""
        # Clear the inner frame
        for widget in self.saved_plots_inner_frame.winfo_children():
            widget.destroy()
        
        # Clear the raw images frame
        for widget in self.raw_images_inner_frame.winfo_children():
            widget.destroy()
        
        # Clear image references
        self.plot_images.clear()
    
    def display_png_images(self, png_files: List[str]):
        """Display PNG images in the scrollable canvas.
        
        Args:
            png_files: List of PNG file paths
        """
        try:
            # Remove placeholder
            if hasattr(self, 'saved_plots_placeholder'):
                self.saved_plots_placeholder.destroy()
            
            if hasattr(self, 'raw_images_placeholder'):
                self.raw_images_placeholder.destroy()
            
            # Display images in both canvases
            self._display_images_in_canvas(png_files, self.saved_plots_inner_frame)
            self._display_images_in_canvas(png_files, self.raw_images_inner_frame)
            
        except Exception as e:
            error_msg = f"Error displaying PNG images: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def _display_images_in_canvas(self, png_files: List[str], canvas_frame: ttk.Frame):
        """Display images in a specific canvas frame.
        
        Args:
            png_files: List of PNG file paths
            canvas_frame: Frame to display images in
        """
        max_width = 800  # Maximum width for images
        
        for i, png_file in enumerate(png_files):
            try:
                # Load and resize image
                image = Image.open(png_file)
                
                # Calculate new size maintaining aspect ratio
                width, height = image.size
                if width > max_width:
                    ratio = max_width / width
                    new_width = max_width
                    new_height = int(height * ratio)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Create label with image
                label = ttk.Label(canvas_frame, image=photo)
                label.image = photo  # Keep a reference
                label.pack(pady=10, padx=10)
                
                # Add filename label
                filename = os.path.basename(png_file)
                filename_label = ttk.Label(canvas_frame, text=filename, font=("Arial", 10))
                filename_label.pack(pady=(0, 10))
                
                # Store references to prevent garbage collection
                self.plot_images.append((photo, label, filename_label))
                
            except Exception as e:
                logger.warning(f"Failed to load image {png_file}: {e}")
                # Add error label
                error_label = ttk.Label(canvas_frame, text=f"Error loading: {os.path.basename(png_file)}", 
                                       foreground="red")
                error_label.pack(pady=5)
    
    def show_no_plots_message(self):
        """Show message when no plots are available."""
        message = "No saved plots found for this model"
        
        # Update both canvases
        for frame in [self.saved_plots_inner_frame, self.raw_images_inner_frame]:
            placeholder = ttk.Label(frame, text=message, foreground="#666666", 
                                   background="#F0F0F0", font=("Arial", 12))
            placeholder.pack(pady=50)
    
    def update_training_results(self, train_losses: List[float], val_losses: Optional[List[float]] = None):
        """Update the training results plot.
        
        Args:
            train_losses: List of training loss values
            val_losses: Optional list of validation loss values
        """
        try:
            # Clear previous plot
            self.training_ax.clear()
            
            # Plot training loss
            epochs = list(range(1, len(train_losses) + 1))
            self.training_ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
            
            # Plot validation loss if available
            if val_losses and len(val_losses) == len(train_losses):
                self.training_ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
            
            # Enhance plot styling
            self.training_ax.set_title("Training Progress", fontsize=14, fontweight='bold')
            self.training_ax.set_xlabel("Epoch", fontsize=12)
            self.training_ax.set_ylabel("Loss", fontsize=12)
            self.training_ax.legend(fontsize=11)
            self.training_ax.grid(True, alpha=0.3)
            
            # Adjust layout and redraw
            self.training_fig.tight_layout()
            self.training_canvas.draw()
            
            logger.info("Training results plot updated")
            
        except Exception as e:
            error_msg = f"Error updating training results: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def update_comprehensive_plots(self, data: Dict[str, Any]):
        """Update comprehensive analysis plots.
        
        Args:
            data: Dictionary containing analysis data
        """
        try:
            # Clear previous plot
            self.comprehensive_ax.clear()
            
            # Create comprehensive analysis based on data
            # This is a placeholder - implement based on your specific needs
            self.comprehensive_ax.text(0.5, 0.5, 'Comprehensive analysis plots\nwill be implemented here', 
                                      ha='center', va='center', transform=self.comprehensive_ax.transAxes,
                                      fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            self.comprehensive_ax.set_title("Comprehensive Analysis")
            
            # Adjust layout and redraw
            self.comprehensive_fig.tight_layout()
            self.comprehensive_canvas.draw()
            
            logger.info("Comprehensive plots updated")
            
        except Exception as e:
            error_msg = f"Error updating comprehensive plots: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

def main():
    """Test the enhanced display panel."""
    root = tk.Tk()
    root.title("Enhanced Display Panel Test")
    root.geometry("1200x800")
    
    # Create display panel
    display_panel = EnhancedDisplayPanel(root)
    
    # Test with sample data
    def test_training_results():
        import numpy as np
        train_losses = np.random.rand(100).cumsum()
        val_losses = train_losses + np.random.rand(100) * 0.5
        display_panel.update_training_results(train_losses.tolist(), val_losses.tolist())
    
    # Add test button
    test_button = ttk.Button(root, text="Test Training Results", command=test_training_results)
    test_button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main() 