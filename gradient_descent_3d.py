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
    python gradient_descent_3d.py [options]

Options:
    --model_dir MODEL_DIR    Directory containing model files (default: most recent)
    --save_animation         Save animation as GIF (default: False)
    --color COLOR            Path color (default: blue)
    --point_size SIZE        Size of points (default: 50)
    --line_width WIDTH       Width of path line (default: 2)
    --surface_alpha ALPHA    Surface transparency (default: 0.6)
    --n_points POINTS        Number of points in surface grid (default: 50)
    --w1_range MIN MAX       Range for weight 1 (default: -2 2)
    --w2_range MIN MAX       Range for weight 2 (default: -2 2)
    --view_elev ELEV         Initial elevation angle (default: 30)
    --view_azim AZIM         Initial azimuth angle (default: 45)
    --fps FPS                Frames per second for animation (default: 30)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import glob
from datetime import datetime
import argparse

class GradientDescentVisualizer:
    def __init__(self, model_dir=None, color='blue', point_size=50, line_width=2,
                 surface_alpha=0.6, n_points=50, w1_range=(-2, 2), w2_range=(-2, 2),
                 view_elev=30, view_azim=45, fps=30):
        """
        Initialize the gradient descent visualizer.
        
        Args:
            model_dir (str): Directory containing the model files
            color (str): Color for the gradient descent path
            point_size (int): Size of points in the path
            line_width (int): Width of the path line
            surface_alpha (float): Transparency of the surface
            n_points (int): Number of points in surface grid
            w1_range (tuple): Range for weight 1
            w2_range (tuple): Range for weight 2
            view_elev (float): Initial elevation angle
            view_azim (float): Initial azimuth angle
            fps (int): Frames per second for animation
        """
        self.model_dir = model_dir or self._find_latest_model_dir()
        self.weights_history = []
        self.losses = []
        self.color = color
        self.point_size = point_size
        self.line_width = line_width
        self.surface_alpha = surface_alpha
        self.n_points = n_points
        self.w1_range = w1_range
        self.w2_range = w2_range
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.fps = fps
        self.load_training_history()
        
    def _find_latest_model_dir(self):
        """Find the most recent model directory."""
        model_dirs = glob.glob("model_*")
        if not model_dirs:
            raise FileNotFoundError("No model directories found")
        return max(model_dirs, key=os.path.getctime)
    
    def load_training_history(self):
        """Load training history from the model directory."""
        # Load weights history if available
        weights_files = glob.glob(os.path.join(self.model_dir, "weights_history_*.csv"))
        if weights_files:
            for file in sorted(weights_files):
                weights = np.loadtxt(file, delimiter=',')
                self.weights_history.append(weights)
        
        # Load losses if available
        loss_file = os.path.join(self.model_dir, "training_losses.csv")
        if os.path.exists(loss_file):
            self.losses = np.loadtxt(loss_file, delimiter=',')
    
    def create_loss_surface(self, w1_range=(-2, 2), w2_range=(-2, 2), n_points=50):
        """
        Create a 3D surface of the loss function.
        
        Args:
            w1_range (tuple): Range for first weight dimension
            w2_range (tuple): Range for second weight dimension
            n_points (int): Number of points in each dimension
        """
        # Create mesh grid
        w1 = np.linspace(w1_range[0], w1_range[1], n_points)
        w2 = np.linspace(w2_range[0], w2_range[1], n_points)
        W1, W2 = np.meshgrid(w1, w2)
        
        # Calculate loss for each point
        Z = np.zeros_like(W1)
        for i in range(n_points):
            for j in range(n_points):
                # Simplified loss function for visualization
                Z[i, j] = W1[i, j]**2 + W2[i, j]**2
        
        return W1, W2, Z
    
    def plot_gradient_descent(self, save_animation=False):
        """
        Create and display the 3D gradient descent visualization.
        
        Args:
            save_animation (bool): Whether to save the animation as a GIF
        """
        # Create figure and 3D axes
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create loss surface
        W1, W2, Z = self.create_loss_surface(
            w1_range=self.w1_range,
            w2_range=self.w2_range,
            n_points=self.n_points
        )
        
        # Plot the surface
        surface = ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=self.surface_alpha)
        
        # If we have weights history, plot the path
        if self.weights_history:
            weights = np.array(self.weights_history)
            if weights.shape[1] >= 2:  # Ensure we have at least 2 weights to plot
                # Plot the gradient descent path
                path = ax.plot(weights[:, 0], weights[:, 1], 
                             [w[0]**2 + w[1]**2 for w in weights],
                             f'{self.color[0]}-', linewidth=self.line_width,
                             label='Gradient Descent Path')
                
                # Add points for each step
                scatter = ax.scatter(weights[:, 0], weights[:, 1],
                                   [w[0]**2 + w[1]**2 for w in weights],
                                   c=self.color, s=self.point_size)
        
        # Set labels and title
        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_zlabel('Loss')
        ax.set_title('3D Gradient Descent Visualization')
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        
        # Add legend
        if self.weights_history:
            ax.legend()
        
        # Set initial view
        ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        if save_animation:
            # Create animation
            def update(frame):
                ax.view_init(elev=self.view_elev, azim=frame)
                return fig,
            
            anim = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, 180),
                                         interval=1000//self.fps, blit=True)
            
            # Save animation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anim.save(os.path.join(self.model_dir, f'gradient_descent_3d_{timestamp}.gif'),
                     writer='pillow', fps=self.fps)
        
        plt.show()
    
    def plot_loss_curves(self):
        """Plot the training and validation loss curves."""
        if not self.losses.size:
            print("No loss data available")
            return
        
        plt.figure(figsize=(10, 6))
        epochs = range(len(self.losses))
        plt.plot(epochs, self.losses, 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.model_dir, f'loss_curves_{timestamp}.png'))
        plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Gradient Descent Visualization')
    parser.add_argument('--model_dir', type=str, help='Directory containing model files')
    parser.add_argument('--save_animation', action='store_true', help='Save animation as GIF')
    parser.add_argument('--color', type=str, default='blue', help='Path color')
    parser.add_argument('--point_size', type=int, default=50, help='Size of points')
    parser.add_argument('--line_width', type=int, default=2, help='Width of path line')
    parser.add_argument('--surface_alpha', type=float, default=0.6, help='Surface transparency')
    parser.add_argument('--n_points', type=int, default=50, help='Number of points in surface grid')
    parser.add_argument('--w1_range', type=float, nargs=2, default=[-2, 2], help='Range for weight 1')
    parser.add_argument('--w2_range', type=float, nargs=2, default=[-2, 2], help='Range for weight 2')
    parser.add_argument('--view_elev', type=float, default=30, help='Initial elevation angle')
    parser.add_argument('--view_azim', type=float, default=45, help='Initial azimuth angle')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for animation')
    return parser.parse_args()

def main():
    """Main function to run the visualization."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Create visualizer with specified options
        visualizer = GradientDescentVisualizer(
            model_dir=args.model_dir,
            color=args.color,
            point_size=args.point_size,
            line_width=args.line_width,
            surface_alpha=args.surface_alpha,
            n_points=args.n_points,
            w1_range=tuple(args.w1_range),
            w2_range=tuple(args.w2_range),
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            fps=args.fps
        )
        
        # Plot gradient descent in 3D
        print("Creating 3D gradient descent visualization...")
        visualizer.plot_gradient_descent(save_animation=args.save_animation)
        
        # Plot loss curves
        print("Creating loss curves...")
        visualizer.plot_loss_curves()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nUsage:")
        print("python gradient_descent_3d.py [options]")
        print("\nOptions:")
        print("  --model_dir MODEL_DIR    Directory containing model files (default: most recent)")
        print("  --save_animation         Save animation as GIF (default: False)")
        print("  --color COLOR            Path color (default: blue)")
        print("  --point_size SIZE        Size of points (default: 50)")
        print("  --line_width WIDTH       Width of path line (default: 2)")
        print("  --surface_alpha ALPHA    Surface transparency (default: 0.6)")
        print("  --n_points POINTS        Number of points in surface grid (default: 50)")
        print("  --w1_range MIN MAX       Range for weight 1 (default: -2 2)")
        print("  --w2_range MIN MAX       Range for weight 2 (default: -2 2)")
        print("  --view_elev ELEV         Initial elevation angle (default: 30)")
        print("  --view_azim AZIM         Initial azimuth angle (default: 45)")
        print("  --fps FPS                Frames per second for animation (default: 30)")

if __name__ == "__main__":
    main() 
