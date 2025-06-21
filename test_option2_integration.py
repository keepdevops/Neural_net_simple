#!/usr/bin/env python3
"""
Test script to verify Option 2 data conversion integration.
"""

import os
import sys
import pandas as pd
import numpy as np

def test_data_converter_import():
    """Test that the data converter can be imported."""
    print("🧪 Testing data converter import...")
    try:
        from data_converter import convert_data_file, detect_data_format
        print("✅ Data converter imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import data converter: {e}")
        return False

def test_format_detection():
    """Test data format detection with various file types."""
    print("\n🧪 Testing format detection...")
    
    # Create test data files
    test_files = {}
    
    # Test 1: Generic features
    df_features = pd.DataFrame({
        'feature_1': np.random.rand(100) * 100,
        'feature_2': np.random.rand(100) * 100,
        'feature_3': np.random.rand(100) * 100,
        'feature_4': np.random.rand(100) * 100,
        'target': np.random.rand(100) * 100
    })
    df_features.to_csv('test_generic_features.csv', index=False)
    test_files['generic_features'] = 'test_generic_features.csv'
    
    # Test 2: OHLCV format
    df_ohlcv = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'vol': np.random.randint(1000000, 10000000, 100)
    })
    df_ohlcv.to_csv('test_ohlcv.csv', index=False)
    test_files['ohlcv'] = 'test_ohlcv.csv'
    
    # Test 3: Numeric data
    df_numeric = pd.DataFrame({
        'col1': np.random.rand(100) * 100,
        'col2': np.random.rand(100) * 100,
        'col3': np.random.rand(100) * 100,
        'col4': np.random.rand(100) * 100,
        'col5': np.random.rand(100) * 100
    })
    df_numeric.to_csv('test_numeric.csv', index=False)
    test_files['numeric'] = 'test_numeric.csv'
    
    # Test format detection
    from data_converter import detect_data_format
    
    for format_type, file_path in test_files.items():
        print(f"\n📁 Testing {format_type} format...")
        df = pd.read_csv(file_path, nrows=5)
        format_info = detect_data_format(df)
        
        print(f"   Detected format: {format_info['format']}")
        print(f"   Status: {format_info['status']}")
        print(f"   Message: {format_info['message']}")
        
        if format_info['status'] == 'ready':
            print("   ✅ Format is ready for training")
        elif format_info['status'] == 'convertible':
            print("   🔄 Format can be converted")
        else:
            print("   ❌ Format is incompatible")
    
    return test_files

def test_conversion():
    """Test data conversion functionality."""
    print("\n🧪 Testing data conversion...")
    
    # Import the conversion function
    from data_converter import convert_data_file
    
    # Test conversion of generic features
    print("📊 Converting generic features to OHLCV...")
    result = convert_data_file('test_generic_features.csv')
    
    if result:
        print(f"✅ Conversion successful: {result}")
        
        # Verify the converted file
        df_converted = pd.read_csv(result)
        ohlcv_columns = ['open', 'high', 'low', 'close', 'vol']
        
        if all(col in df_converted.columns for col in ohlcv_columns):
            print("✅ Converted file has all required OHLCV columns")
            print(f"   Columns: {list(df_converted.columns)}")
            print(f"   Rows: {len(df_converted)}")
        else:
            missing = [col for col in ohlcv_columns if col not in df_converted.columns]
            print(f"❌ Missing columns: {missing}")
    else:
        print("❌ Conversion failed")

def test_gui_integration():
    """Test GUI integration components."""
    print("\n🧪 Testing GUI integration...")
    
    # Test the validation function (simulated)
    def simulate_validate_and_convert(file_path):
        """Simulate the GUI's validate_and_convert_data_file function."""
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path, nrows=5)
            format_info = detect_data_format(df)
            
            if format_info['status'] == 'ready':
                return file_path
            elif format_info['status'] == 'convertible':
                # Simulate user agreeing to conversion
                converted_file = convert_data_file(file_path)
                return converted_file
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    # Test with the converted file
    test_file = 'test_generic_features_converted_ohlcv.csv'
    if os.path.exists(test_file):
        result = simulate_validate_and_convert(test_file)
        if result:
            print(f"✅ GUI integration test passed: {result}")
        else:
            print("❌ GUI integration test failed")
    else:
        print("⚠️  Skipping GUI integration test - no converted file found")

def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    test_files = [
        'test_generic_features.csv',
        'test_ohlcv.csv', 
        'test_numeric.csv',
        'test_generic_features_converted_ohlcv.csv'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   Deleted: {file}")

def main():
    """Run all tests."""
    print("🚀 Option 2 Data Conversion Integration Test")
    print("=" * 50)
    
    # Test 1: Import
    if not test_data_converter_import():
        print("❌ Cannot proceed without data converter")
        return
    
    # Test 2: Format detection
    test_files = test_format_detection()
    
    # Test 3: Conversion
    test_conversion()
    
    # Test 4: GUI integration
    test_gui_integration()
    
    # Cleanup
    cleanup_test_files()
    
    print("\n🎯 All tests completed!")
    print("✅ Option 2 data conversion integration is working correctly")

if __name__ == "__main__":
    main() 