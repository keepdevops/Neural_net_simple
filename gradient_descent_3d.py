"""
3D Gradient Descent Visualization

This module creates a 3D visualization of the gradient descent process for neural network training,
emphasizing the loss path as weights are updated, and saves PNG snapshots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import glob
import json
from datetime import datetime
import argparse
import sys

def find_latest_model_dir():
    """Find the most recent model directory."""
    model_dirs = glob.glob("model_*")
    if not model_dirs:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if os.path.exists(models_dir):
            model_dirs = glob.glob(os.path.join(models_dir, "model_*"))
        if not model_dirs:
            raise FileNotFoundError("No model directories found. Please train a model first.")
    return max(model_dirs, key=os.path.getctime)

def load_training_data(model_dir):
    """Load training history and normalization parameters."""
    model_dir = os.path.abspath(model_dir)
    
    scaler_mean_file = os.path.join(model_dir, "scaler_mean.csv")
    scaler_std_file = os.path.join(model_dir, "scaler_std.csv")
    target_min_file = os.path.join(model_dir, "target_min.csv")
    target_max_file = os.path.join(model_dir, "target_max.csv")
    feature_info_file = os.path.join(model_dir, "feature_info.json")
    weights_dir = os.path.join(model_dir, "weights_history")
    
    if not all(os.path.exists(f) for f in [scaler_mean_file, scaler_std_file, feature_info_file]):
        raise FileNotFoundError(f"Normalization parameters not found in {model_dir}")
    
    with open(feature_info_file, 'r') as f:
        feature_info = json.load(f)
    
    X_mean = np.loadtxt(scaler_mean_file, delimiter=',')
    X_range = np.loadtxt(scaler_std_file, delimiter=',')
    Y_min = float(np.loadtxt(target_min_file, delimiter=',')) if os.path.exists(target_min_file) else None
    Y_max = float(np.loadtxt(target_max_file, delimiter=',')) if os.path.exists(target_max_file) else None
    
    norm_params = {
        'X_mean': X_mean.tolist() if X_mean.ndim > 0 else [X_mean.item()],
        'X_range': X_range.tolist() if X_range.ndim > 0 else [X_range.item()],
        'Y_min': Y_min,
        'Y_max': Y_max,
        'x_features': feature_info.get('x_features', []),
        'y_feature': feature_info.get('y_feature', '')
    }
    
    training_losses_file = os.path.join(model_dir, "training_losses.csv")
    if not os.path.exists(training_losses_file):
        raise FileNotFoundError(f"Training history not found in {model_dir}")
    
    losses = np.loadtxt(training_losses_file, delimiter=',')
    if losses.ndim > 1:
        losses = losses[:, 0]  # Use training losses
    else:
        losses = losses
    
    # Load weight history
    weights_files = sorted(glob.glob(os.path.join(weights_dir, "weights_history_*.npz")))
    weights = []
    if weights_files:
        for wf in weights_files:
            with np.load(wf) as data:
                weights.append({
                    'W1': data['W1'],
                    'W2': data['W2']
                })
    else:
        print("No weight history found, using placeholder weights")
        weights = [{'W1': np.zeros((len(norm_params['x_features']), 4)), 
                    'W2': np.zeros((4, 1))} for _ in range(len(losses))]
    
    history = {
        'losses': losses.tolist(),
        'weights': weights,
        'epochs': list(range(len(losses)))
    }
    
    return norm_params, history

def compute_loss_surface(X, y, w1_range, w2_range, n_points=50):
    """Compute the loss surface for visualization."""
    w1 = np.linspace(w1_range[0], w1_range[1], n_points)
    w2 = np.linspace(w2_range[0], w2_range[1], n_points)
    W1, W2 = np.meshgrid(w1, w2)
    Z = np.zeros_like(W1)
    
    X_viz = X[:, :2] if X.shape[1] > 2 else X
    print(f"Using first 2 features out of {X.shape[1]} for visualization")
    
    for i in range(n_points):
        for j in range(n_points):
            w = np.array([[W1[i, j]], [W2[i, j]]])
            if w.shape[0] != X_viz.shape[1]:
                w = np.vstack([w, np.zeros((X_viz.shape[1] - w.shape[0], 1))])
            y_pred = X_viz @ w
            Z[i, j] = np.mean((y - y_pred) ** 2)
    
    return W1, W2, Z

class GradientDescentVisualizer:
    def __init__(self, model_dir=None, w1_range=(-2, 2), w2_range=(-2, 2), n_points=50,
                 view_elev=30, view_azim=45, fps=30, color='viridis', point_size=8, 
                 line_width=3, surface_alpha=0.6, output_resolution=(1200, 800)):
        self.model_dir = model_dir or find_latest_model_dir()
        self.w1_range = w1_range
        self.w2_range = w2_range
        self.n_points = n_points
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.fps = fps
        self.color = color
        self.point_size = point_size
        self.line_width = line_width  # Increased for prominent loss path
        self.surface_alpha = surface_alpha
        self.output_resolution = output_resolution
        
        self.norm_params, self.history = load_training_data(self.model_dir)
        
        data_file = os.path.join(self.model_dir, "training_data.csv")
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            X = df[self.norm_params['x_features']].values
            y = df[self.norm_params['y_feature']].values.reshape(-1, 1)
            self.X = X
            self.y = y
            self.has_training_data = True
        else:
            print(f"Training data not found in {self.model_dir}, creating synthetic visualization...")
            self.X = np.random.randn(100, len(self.norm_params['x_features']))
            self.y = np.random.randn(100, 1)
            self.has_training_data = False
        
        self.W1, self.W2, self.Z = compute_loss_surface(self.X, self.y, w1_range, w2_range, n_points)
        
        # Set figure size based on output_resolution (pixels to inches at 100 DPI)
        dpi = 100
        fig_width = self.output_resolution[0] / dpi
        fig_height = self.output_resolution[1] / dpi
        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
        
        # Plot loss surface
        self.surface = self.ax.plot_surface(self.W1, self.W2, self.Z, 
                                          cmap=self.color, alpha=self.surface_alpha)
        
        # Initialize loss path (thicker line, distinct color)
        self.progress_line = self.ax.plot([], [], [], 'r-', lw=self.line_width, 
                                        label='Loss Path', alpha=0.8)[0]
        self.current_point = self.ax.plot([], [], [], 'ro', markersize=self.point_size, 
                                        label='Current Position')[0]
        
        # Initialize start and end points
        self.start_point = self.ax.plot([], [], [], 'go', markersize=self.point_size + 2, 
                                      label='Start')[0]
        self.end_point = self.ax.plot([], [], [], 'bo', markersize=self.point_size + 2, 
                                    label='End')[0]
        
        self.ax.set_xlabel('Weight 1')
        self.ax.set_ylabel('Weight 2')
        self.ax.set_zlabel('Loss')
        self.ax.legend(loc='upper right')
        
        title_text = f"Gradient Descent Loss Path\nModel: {os.path.basename(self.model_dir)}"
        if not self.has_training_data:
            title_text += " (Synthetic)"
        self.fig.suptitle(title_text, fontsize=14)
        
        self.text = self.fig.text(0.02, 0.02, '', fontsize=10)
        self.animation = None

    def update(self, frame):
        """Update the animation frame with loss path."""
        if frame >= len(self.history['losses']):
            return
        
        if frame == 0:
            self.progress_line.set_data([], [])
            self.progress_line.set_3d_properties([])
            self.current_point.set_data([], [])
            self.current_point.set_3d_properties([])
            self.start_point.set_data([], [])
            self.start_point.set_3d_properties([])
            self.end_point.set_data([], [])
            self.end_point.set_3d_properties([])
        
        weights = self.history['weights'][min(frame, len(self.history['weights']) - 1)]
        w1 = weights['W1'][0, 0] if weights['W1'].size > 0 else 0.0
        w2 = weights['W2'][0, 0] if weights['W2'].size > 0 else 0.0
        loss = self.history['losses'][frame]
        
        # Collect path data
        x = [w['W1'][0, 0] for w in self.history['weights'][:frame + 1] 
             if w['W1'].size > 0]
        y = [w['W2'][0, 0] for w in self.history['weights'][:frame + 1] 
             if w['W2'].size > 0]
        z = self.history['losses'][:frame + 1]
        
        # Update loss path
        self.progress_line.set_data(x, y)
        self.progress_line.set_3d_properties(z)
        
        # Update current position
        self.current_point.set_data([w1], [w2])
        self.current_point.set_3d_properties([loss])
        
        # Update start point (first epoch)
        if frame == 0 and x:
            self.start_point.set_data([x[0]], [y[0]])
            self.start_point.set_3d_properties([z[0]])
            # Add annotation
            self.ax.text(x[0], y[0], z[0], 'Start', color='green', fontsize=10)
        
        # Update end point (last epoch)
        if frame == len(self.history['losses']) - 1 and x:
            self.end_point.set_data([x[-1]], [y[-1]])
            self.end_point.set_3d_properties([z[-1]])
            # Add annotation
            self.ax.text(x[-1], y[-1], z[-1], 'End', color='blue', fontsize=10)
        
        self.text.set_text(f'Epoch: {frame + 1}\nLoss: {loss:.6f}')
        
        return (self.progress_line, self.current_point, self.start_point, self.end_point, self.text)

    def save_plots(self, frames=[0, None, -1], plots_dir=None):
        """Save PNG snapshots of the visualization at specified frames."""
        if plots_dir is None:
            plots_dir = os.path.join(self.model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        if None in frames:
            frames[frames.index(None)] = len(self.history['losses']) // 2
        
        for frame in frames:
            if frame < 0:
                frame = len(self.history['losses']) + frame
            if frame >= len(self.history['losses']) or frame < 0:
                continue
            
            self.update(frame)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'gradient_descent_3d_frame_{frame}_{timestamp}.png'
            filepath = os.path.join(plots_dir, filename)
            self.fig.savefig(filepath, dpi=100, bbox_inches='tight')
            print(f"Saved plot: {filepath}")

    def animate(self, save_png=False):
        """Create and display the animation, optionally saving PNGs."""
        if save_png:
            self.save_plots(frames=[0, None, -1])
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update,
            frames=len(self.history['losses']),
            interval=1000/self.fps,
            blit=False,
            repeat=True
        )
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='3D Gradient Descent Visualization')
    parser.add_argument('--model_dir', type=str, help='Directory containing the model files')
    parser.add_argument('--color', type=str, default='viridis', help='Color map for the surface')
    parser.add_argument('--point_size', type=int, default=8, help='Size of the current point marker')
    parser.add_argument('--line_width', type=int, default=3, help='Width of the gradient descent path')
    parser.add_argument('--surface_alpha', type=float, default=0.6, help='Alpha transparency of the surface')
    parser.add_argument('--w1_range', type=float, nargs=2, default=[-2, 2], help='Range for weight 1')
    parser.add_argument('--w2_range', type=float, nargs=2, default=[-2, 2], help='Range for weight 2')
    parser.add_argument('--n_points', type=int, default=50, help='Number of points in surface grid')
    parser.add_argument('--view_elev', type=float, default=30, help='Initial elevation angle')
    parser.add_argument('--view_azim', type=float, default=45, help='Initial azimuth angle')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for animation')
    parser.add_argument('--save_png', action='store_true', help='Save PNG snapshots of visualization')
    parser.add_argument('--output_resolution', type=int, nargs=2, default=[1200, 800], 
                       help='Output resolution for PNGs (width height)')
    
    args = parser.parse_args()
    
    try:
        visualizer = GradientDescentVisualizer(
            model_dir=args.model_dir,
            w1_range=tuple(args.w1_range),
            w2_range=tuple(args.w2_range),
            n_points=args.n_points,
            view_elev=args.view_elev,
            view_azim=args.view_azim,
            fps=args.fps,
            color=args.color,
            point_size=args.point_size,
            line_width=args.line_width,
            surface_alpha=args.surface_alpha,
            output_resolution=tuple(args.output_resolution)
        )
        visualizer.animate(save_png=args.save_png)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
