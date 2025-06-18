#!/usr/bin/env python3
"""
Test script to verify data loading and numeric conversion
"""

import pandas as pd
import numpy as np

def test_data_loading():
    """Test the data loading process"""
    
    # Load the data file
    data_file = "/Users/porupine/redline/data/amc_us_data.csv"
    print(f"Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original data types: {df.dtypes.to_dict()}")
    
    # Clean column names - remove empty strings and whitespace
    df.columns = [col.strip() if isinstance(col, str) else str(col) for col in df.columns]
    df.columns = [col for col in df.columns if col]  # Remove empty strings
    
    print(f"Cleaned columns: {df.columns.tolist()}")
    
    # Define expected stock data columns that should be numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'vol', 'openint']
    text_columns = ['ticker', 'timestamp', 'format']
    
    # Convert only numeric columns to numeric, handling any non-numeric values
    for col in df.columns:
        if col in numeric_columns:
            # For numeric stock data columns, convert to float
            print(f"Converting numeric column '{col}' to float...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif col in text_columns:
            # For text columns, keep as object/string
            print(f"Keeping text column '{col}' as string...")
            # No conversion needed
        else:
            # For other columns, try to convert to numeric
            print(f"Converting column '{col}' to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"After conversion data types: {df.dtypes.to_dict()}")
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    print(f"Data shape after dropping NaN: {df.shape}")
    
    if df.empty:
        print("ERROR: No valid numeric data found in the file")
        return False
    
    # Test feature selection
    x_features = ['open', 'high', 'low', 'vol']
    y_feature = 'close'
    
    print(f"Testing feature selection:")
    print(f"X features: {x_features}")
    print(f"Y feature: {y_feature}")
    
    # Check if all features exist
    missing_features = []
    for feature in x_features + [y_feature]:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"ERROR: Missing features: {missing_features}")
        return False
    
    # Prepare features and target
    X = df[x_features].values.astype(np.float64)
    Y = df[y_feature].values.reshape(-1, 1).astype(np.float64)
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Check for any remaining NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)) or np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        print("ERROR: Data contains NaN or infinite values after preprocessing")
        return False
    
    print("SUCCESS: Data loading and preprocessing completed successfully!")
    print(f"Sample X data (first 3 rows):")
    print(X[:3])
    print(f"Sample Y data (first 3 rows):")
    print(Y[:3])
    
    return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!") 