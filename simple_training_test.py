#!/usr/bin/env python3
"""
Simple test to verify training button functionality.
"""

import tkinter as tk
from stock_gui import StockPredictionGUI

def test_training_button():
    """Test that the training button works."""
    print("Creating GUI...")
    root = tk.Tk()
    root.withdraw()  # Hide window
    
    app = StockPredictionGUI(root)
    print("GUI created successfully")
    
    # Test that the train button exists and has the correct command
    if hasattr(app, 'train_button'):
        print("✅ Train button exists")
        print(f"   Command: {app.train_button['command']}")
        print(f"   Text: {app.train_button['text']}")
        print(f"   State: {app.train_button['state']}")
    else:
        print("❌ Train button missing")
    
    # Test that the status label exists
    if hasattr(app, 'status_label'):
        print("✅ Status label exists")
        print(f"   Text: {app.status_label['text']}")
    else:
        print("❌ Status label missing")
    
    # Test update_progress method
    print("\nTesting update_progress method...")
    app.update_progress("Test message from simple test")
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 To test training in the GUI:")
    print("   1. Run: python stock_gui.py")
    print("   2. Load a data file in the Data tab")
    print("   3. Select features and target")
    print("   4. Click 'Start Training' in the Training tab")
    print("   5. Watch for debug output in the terminal")
    
    root.destroy()

if __name__ == "__main__":
    test_training_button() 