#!/usr/bin/env python3
"""
STOCK PREDICTION GUI - QUICK REFERENCE CARD
==========================================

Quick reference for essential commands, settings, and shortcuts.
For detailed information, see STOCK_GUI_USER_MANUAL.py

Author: AI Assistant
Version: 1.0
Date: 2025-01-27
"""

def print_quick_reference():
    """Print the quick reference card."""
    print("=" * 80)
    print("STOCK PREDICTION GUI - QUICK REFERENCE CARD")
    print("=" * 80)
    print()
    
    print("🚀 QUICK START")
    print("-" * 40)
    print("1. python stock_gui.py                    # Launch GUI")
    print("2. Select data file → Configure model → Start Training")
    print("3. Monitor progress → View results → Make predictions")
    print()
    
    print("📁 ESSENTIAL COMMANDS")
    print("-" * 40)
    print("python stock_gui.py                       # Launch GUI")
    print("python STOCK_GUI_USER_MANUAL.py          # Full manual")
    print("python STOCK_GUI_USER_MANUAL.py overview # Quick overview")
    print("python STOCK_GUI_USER_MANUAL.py troubleshooting # Help")
    print()
    
    print("⚙️ DEFAULT SETTINGS")
    print("-" * 40)
    print("Hidden Layer Size: 32 neurons")
    print("Learning Rate: 0.01")
    print("Epochs: 100")
    print("Batch Size: 32")
    print("Train/Test Split: 80/20")
    print("Optimizer: Adam")
    print("Loss Function: Mean Squared Error")
    print()
    
    print("📊 DATA FORMAT")
    print("-" * 40)
    print("Required CSV columns: date, open, high, low, close, volume")
    print("Example:")
    print("date,open,high,low,close,volume")
    print("2025-01-01,100.50,102.30,99.80,101.20,1500000")
    print()
    
    print("🎮 KEYBOARD SHORTCUTS")
    print("-" * 40)
    print("Ctrl+Q: Quit application")
    print("Ctrl+S: Save model")
    print("Ctrl+O: Open data file")
    print("Ctrl+R: Refresh plots")
    print("Space: Play/pause animation")
    print("R: Reset 3D view")
    print("Z/X: Zoom in/out")
    print("Arrow Keys: Rotate 3D view")
    print()
    
    print("🎛️ PLOT CONTROLS (3D)")
    print("-" * 40)
    print("View Presets: Default, Top, Side, Isometric")
    print("Rotation: -180° to 180° (X, Y, Z)")
    print("Zoom: 0.1x to 5.0x")
    print("Camera: X/Y (-10 to 10), Z (1 to 20)")
    print("Animation Speed: 0.1x to 5.0x")
    print("Frame Rate: 1 to 60 FPS")
    print("Loop Modes: Loop, Once, Ping-Pong")
    print()
    
    print("📈 PERFORMANCE METRICS")
    print("-" * 40)
    print("Training: Loss (MSE), Validation Loss, Training Time")
    print("Prediction: R² Score, MAE, RMSE, MAPE, Directional Accuracy")
    print()
    
    print("🔧 TROUBLESHOOTING")
    print("-" * 40)
    print("GUI won't start: Check Python 3.8+, dependencies")
    print("Training fails: Verify data format, check for missing values")
    print("Plots freeze: Restart GUI, clear cache")
    print("Memory issues: Reduce batch size, use smaller datasets")
    print()
    
    print("📋 USAGE WORKFLOW")
    print("-" * 40)
    print("1. TRAINING:")
    print("   • Select data file")
    print("   • Choose features (open, high, low, volume)")
    print("   • Set target (close)")
    print("   • Configure model parameters")
    print("   • Click 'Start Training'")
    print("   • Monitor in Training Results tab")
    print()
    print("2. PREDICTION:")
    print("   • Select trained model")
    print("   • Choose input data file")
    print("   • Click 'Run Prediction'")
    print("   • View results in Prediction Results tab")
    print()
    print("3. VISUALIZATION:")
    print("   • Go to 3D Gradient Descent tab")
    print("   • Use Plot Controls to adjust view")
    print("   • Start animation with Play button")
    print("   • Use mouse to interact with 3D plot")
    print()
    
    print("📁 FILE STRUCTURE")
    print("-" * 40)
    print("model_YYYYMMDD_HHMMSS/")
    print("├── model.h5                    # Trained network")
    print("├── training_history.json       # Training metrics")
    print("├── scaler_params.npz          # Data preprocessing")
    print("├── plots/                     # Visualizations")
    print("│   ├── loss_curve.png")
    print("│   ├── actual_vs_predicted.png")
    print("│   └── gradient_descent_3d_frame_*.png")
    print("└── predictions_*.csv          # Results")
    print()
    
    print("🎯 BEST PRACTICES")
    print("-" * 40)
    print("• Use at least 100 data points")
    print("• Clean data (no missing values)")
    print("• Start with default settings")
    print("• Monitor validation loss for overfitting")
    print("• Save models regularly")
    print("• Clear cache periodically")
    print()
    
    print("📞 HELP & SUPPORT")
    print("-" * 40)
    print("Full Manual: python STOCK_GUI_USER_MANUAL.py")
    print("Troubleshooting: python STOCK_GUI_USER_MANUAL.py troubleshooting")
    print("Examples: python STOCK_GUI_USER_MANUAL.py examples")
    print()
    
    print("=" * 80)
    print("For detailed information, run: python STOCK_GUI_USER_MANUAL.py")
    print("=" * 80)

if __name__ == "__main__":
    print_quick_reference() 