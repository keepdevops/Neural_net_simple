#!/usr/bin/env python3
"""
Test script for technical indicators

This script tests the technical indicators functions to ensure they work correctly.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the technical indicators functions
from stock_net import compute_rsi, add_technical_indicators

def create_sample_data():
    """Create sample stock data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create realistic stock price data
    base_price = 100
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Create OHLCV data
    data = {
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'vol': np.random.randint(1000000, 10000000, 100)
    }
    
    return pd.DataFrame(data)

def test_rsi():
    """Test RSI calculation"""
    print("Testing RSI calculation...")
    
    # Create sample data
    df = create_sample_data()
    
    # Calculate RSI
    rsi = compute_rsi(df['close'], 14)
    
    # Check that RSI values are within expected range (0-100)
    assert rsi.min() >= 0, f"RSI minimum should be >= 0, got {rsi.min()}"
    assert rsi.max() <= 100, f"RSI maximum should be <= 100, got {rsi.max()}"
    
    # Check that RSI is not all NaN
    assert not rsi.isna().all(), "RSI should not be all NaN"
    
    # Check that first value is NaN (no previous data for price change)
    assert pd.isna(rsi.iloc[0]), "First RSI value should be NaN (no price change)"
    
    # Check that we have some valid RSI values
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0, "Should have some valid RSI values"
    
    print(f"✅ RSI test passed! RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
    print(f"   Valid RSI values: {len(valid_rsi)} out of {len(rsi)}")
    return rsi

def test_technical_indicators():
    """Test all technical indicators"""
    print("Testing technical indicators...")
    
    # Create sample data
    df = create_sample_data()
    
    # Add technical indicators
    df_enhanced = add_technical_indicators(df)
    
    # Check that new columns were added
    expected_indicators = [
        'ma_5', 'ma_10', 'ma_20', 'rsi', 'price_change', 'price_change_5',
        'volatility_10', 'bb_middle', 'bb_upper', 'bb_lower',
        'macd', 'macd_signal', 'volume_ma', 'volume_ratio'
    ]
    
    for indicator in expected_indicators:
        assert indicator in df_enhanced.columns, f"Missing indicator: {indicator}"
    
    # Check that moving averages are reasonable
    assert df_enhanced['ma_10'].iloc[-1] > 0, "MA10 should be positive"
    assert df_enhanced['ma_20'].iloc[-1] > 0, "MA20 should be positive"
    
    # Check that RSI is in valid range
    rsi_valid = df_enhanced['rsi'].dropna()
    assert rsi_valid.min() >= 0, f"RSI minimum should be >= 0, got {rsi_valid.min()}"
    assert rsi_valid.max() <= 100, f"RSI maximum should be <= 100, got {rsi_valid.max()}"
    
    # Check that price changes are reasonable
    assert abs(df_enhanced['price_change'].max()) < 1, "Price change should be reasonable"
    
    # Check that volatility is positive
    volatility_valid = df_enhanced['volatility_10'].dropna()
    assert volatility_valid.min() >= 0, "Volatility should be non-negative"
    
    print(f"✅ Technical indicators test passed! Added {len(expected_indicators)} indicators")
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Enhanced columns: {len(df_enhanced.columns)}")
    
    return df_enhanced

def test_with_real_data():
    """Test with real data if available"""
    print("Testing with real data...")
    
    # Look for CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found for real data test")
        return
    
    # Use the first CSV file found
    csv_file = csv_files[0]
    print(f"Using real data file: {csv_file}")
    
    try:
        # Load real data
        df = pd.read_csv(csv_file)
        
        # Check if it has required columns
        required_cols = ['open', 'high', 'low', 'close', 'vol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Real data missing columns: {missing_cols}")
            print("Skipping real data test")
            return
        
        # Add technical indicators
        df_enhanced = add_technical_indicators(df)
        
        print(f"✅ Real data test passed!")
        print(f"   Original data shape: {df.shape}")
        print(f"   Enhanced data shape: {df_enhanced.shape}")
        print(f"   Sample RSI values: {df_enhanced['rsi'].dropna().head().tolist()}")
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Technical Indicators")
    print("=" * 50)
    
    try:
        # Test RSI calculation
        rsi = test_rsi()
        
        # Test all technical indicators
        df_enhanced = test_technical_indicators()
        
        # Test with real data if available
        test_with_real_data()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Technical indicators are working correctly.")
        
        # Show sample of enhanced data
        print("\nSample of enhanced data:")
        print(df_enhanced[['close', 'ma_10', 'rsi', 'price_change', 'volatility_10']].tail())
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 