"""
3D Gradient Descent Visualization

This module creates a 3D visualization of the gradient descent process
for the neural network training. It shows how the loss function changes
as the weights are updated during training.

Features:
- 3D surface plot of the loss function
- Animated gradient descent path
- Interactive 3D rotation and zoom
- Real-time weight and loss tracking

Usage:
    python gradient_descent_3d.py [model_dir]

Arguments:
    model_dir    Path to the model directory (default: most recent model directory)

Example:
    python gradient_descent_3d.py
    python gradient_descent_3d.py model_20240315_123456
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import glob
import json
from datetime import datetime
import sys

# Add project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_latest_model_dir():
    """Find the most recent model directory."""
    # Look for model directories in the current directory
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        # Also check in a 'models' subdirectory
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if os.path.exists(models_dir):
            model_dirs = glob.glob(os.path.join(models_dir, "model_*"))
        
        if not model_dirs:
            raise FileNotFoundError("No model directories found. Please train a model first.")
    
    return max(model_dirs, key=os.path.getctime)

def load_training_data(model_dir):
    """Load training history and normalization parameters."""
    # Ensure absolute path
    model_dir = os.path.abspath(model_dir)
    
    # Load normalization parameters
    norm_params_file = os.path.join(model_dir, "normalization.json")
    if not os.path.exists(norm_params_file):
        raise FileNotFoundError(f"Normalization parameters not found in {model_dir}")
    
    with open(norm_params_file, 'r') as f:
        norm_params = json.load(f)
    
    # Load training history
    history_file = os.path.join(model_dir, "training_history.json")
    if not os.path.exists(history_file):
        raise FileNotFoundError(f"Training history not found in {model_dir}")
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return norm_params, history

def compute_loss_surface(X, y, w1_range, w2_range, n_points=50):
    """Compute the loss surface for visualization."""
    w1 = np.linspace(w1_range[0], w1_range[1], n_points)
    w2 = np.linspace(w2_range[0], w2_range[1], n_points)
    W1, W2 = np.meshgrid(w1, w2)
    
    # Compute loss for each weight combination
    Z = np.zeros_like(W1)
    for i in range(n_points):
        for j in range(n_points):
            w = np.array([[W1[i, j]], [W2[i, j]]])
            y_pred = X @ w
            Z[i, j] = np.mean((y - y_pred) ** 2)
    
    return W1, W2, Z

class GradientDescentVisualizer:
    def __init__(self, model_dir=None, w1_range=(-2, 2), w2_range=(-2, 2), n_points=50,
                 view_elev=30, view_azim=45, fps=30):
        """
        Initialize the gradient descent visualizer.
        
        Args:
            model_dir (str): Directory containing the model files
            w1_range (tuple): Range for weight 1
            w2_range (tuple): Range for weight 2
            n_points (int): Number of points in surface grid
            view_elev (float): Initial elevation angle
            view_azim (float): Initial azimuth angle
            fps (int): Frames per second for animation
        """
        self.model_dir = model_dir or find_latest_model_dir()
        self.w1_range = w1_range
        self.w2_range = w2_range
        self.n_points = n_points
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.fps = fps
        
        # Load training data
        self.norm_params, self.history = load_training_data(self.model_dir)
        
        # Load the data used for training
        data_file = os.path.join(self.model_dir, "training_data.csv")
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            X = df[self.norm_params['x_features']].values
            y = df[self.norm_params['y_feature']].values.reshape(-1, 1)
            self.X = X
            self.y = y
        else:
            raise FileNotFoundError(f"Training data not found in {self.model_dir}")
        
        # Compute loss surface
        self.W1, self.W2, self.Z = compute_loss_surface(X, y, w1_range, w2_range, n_points)
        
        # Initialize figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set initial view
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Plot loss surface
        self.surface = self.ax.plot_surface(self.W1, self.W2, self.Z, 
                                          cmap='viridis', alpha=0.6)
        
        # Plot gradient descent path
        self.path = self.ax.plot([], [], [], 'r-', lw=2)[0]
        self.current_point = self.ax.plot([], [], [], 'ro', markersize=8)[0]
        
        # Add labels
        self.ax.set_xlabel('Weight 1')
        self.ax.set_ylabel('Weight 2')
        self.ax.set_zlabel('Loss')
        
        # Add title with model info
        title_text = f"Gradient Descent Visualization\nModel: {os.path.basename(self.model_dir)}"
        self.title = self.fig.suptitle(title_text, fontsize=14)
        
        # Add text annotation for current weights and loss
        self.text = self.fig.text(0.02, 0.02, '', fontsize=10)
        
        # Initialize animation
        self.animation = None
        
    def update(self, frame):
        """Update the animation frame."""
        if frame == 0:
            # Clear path and start from beginning
            self.path.set_data([], [])
            self.path.set_3d_properties([])
            self.current_point.set_data([], [])
            self.current_point.set_3d_properties([])
            
            # Start from initial weights
            w1 = self.history['weights'][0][0]
            w2 = self.history['weights'][0][1]
            loss = self.history['losses'][0]
            
            # Update text
            self.text.set_text(f'Epoch: 0\nW1: {w1:.4f}\nW2: {w2:.4f}\nLoss: {loss:.4f}')
            
            return self.path, self.current_point, self.text
        
        # Get current weights and loss
        w1 = self.history['weights'][frame][0]
        w2 = self.history['weights'][frame][1]
        loss = self.history['losses'][frame]
        
        # Update path
        x_data = [w[0] for w in self.history['weights'][:frame+1]]
        y_data = [w[1] for w in self.history['weights'][:frame+1]]
        z_data = self.history['losses'][:frame+1]
        
        self.path.set_data(x_data, y_data)
        self.path.set_3d_properties(z_data)
        
        # Update current point
        self.current_point.set_data([w1], [w2])
        self.current_point.set_3d_properties([loss])
        
        # Update text
        self.text.set_text(f'Epoch: {frame}\nW1: {w1:.4f}\nW2: {w2:.4f}\nLoss: {loss:.4f}')
        
        return self.path, self.current_point, self.text
    
    def animate(self):
        """Create and display the animation."""
        self.animation = animation.FuncAnimation(
            self.fig, self.update,
            frames=len(self.history['losses']),
            interval=1000/self.fps,
            blit=True
        )
        
        plt.show()

def main():
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = find_latest_model_dir()
    
    try:
        visualizer = GradientDescentVisualizer(model_dir=model_dir)
        visualizer.animate()
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
