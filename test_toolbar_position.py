#!/usr/bin/env python3
"""
Test script to verify toolbar positioning in prediction results tab
"""

import tkinter as tk
import sys
import os

# Add the current directory to the path so we can import stock_gui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_toolbar_positioning():
    """Test the toolbar positioning in prediction results tab."""
    print("🧪 Testing toolbar positioning in prediction results tab...")
    
    try:
        # Import the GUI class
        from stock_gui import StockPredictionGUI
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing
        
        # Create the GUI instance
        gui = StockPredictionGUI(root)
        
        print("✅ GUI instance created successfully")
        
        # Check if the prediction results tab exists
        if hasattr(gui, 'display_notebook'):
            print("✅ Display notebook found")
            
            # Find the prediction results tab
            for i in range(gui.display_notebook.index('end')):
                tab_text = gui.display_notebook.tab(i, 'text')
                if tab_text == "Prediction Results":
                    print("✅ Prediction Results tab found")
                    
                    # Check if the toolbar is properly positioned
                    if hasattr(gui, 'pred_canvas') and gui.pred_canvas:
                        print("✅ Prediction canvas found")
                        
                        # Check the grid layout of the prediction results frame
                        pred_frame = gui.display_notebook.select()
                        if pred_frame:
                            print("✅ Prediction frame found")
                            
                            # Check grid configuration
                            grid_info = gui.pred_canvas.get_tk_widget().grid_info()
                            print(f"Canvas grid info: {grid_info}")
                            
                            # Check if toolbar exists and is positioned correctly
                            for child in pred_frame.winfo_children():
                                if hasattr(child, 'winfo_children'):
                                    for grandchild in child.winfo_children():
                                        if hasattr(grandchild, 'winfo_children'):
                                            for great_grandchild in grandchild.winfo_children():
                                                if hasattr(great_grandchild, 'winfo_class'):
                                                    if great_grandchild.winfo_class() == 'Frame':
                                                        grid_info = great_grandchild.grid_info()
                                                        print(f"Toolbar frame grid info: {grid_info}")
                            
                            print("✅ Toolbar positioning check completed")
                            return True
                        else:
                            print("❌ Prediction frame not found")
                            return False
                    else:
                        print("❌ Prediction canvas not found")
                        return False
            else:
                print("❌ Prediction Results tab not found")
                return False
        else:
            print("❌ Display notebook not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing toolbar positioning: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    success = test_toolbar_positioning()
    if success:
        print("✅ Toolbar positioning test completed successfully")
    else:
        print("❌ Toolbar positioning test failed") 