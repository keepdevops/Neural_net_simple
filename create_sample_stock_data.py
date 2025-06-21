#!/usr/bin/env python3
"""
Create a sample stock data file with proper OHLCV column names.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_stock_data():
    """Create a sample stock data file with OHLCV columns."""
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock data
    np.random.seed(42)
    base_price = 100.0
    
    data = []
    for i, date in enumerate(dates):
        # Skip weekends (Saturday=5, Sunday=6)
        if date.weekday() >= 5:
            continue
            
        # Simulate stock price movement with some trend and volatility
        trend = 0.0001 * i  # Slight upward trend
        volatility = 0.02  # 2% daily volatility
        
        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        base_price *= (1 + price_change)
        
        # Ensure price doesn't go negative
        base_price = max(base_price, 1.0)
        
        # Generate OHLCV data
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = np.random.randint(1000000, 10000000)  # 1M to 10M volume
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'vol': volume
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'sample_stock_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created sample stock data file: {output_file}")
    print(f"📊 Shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"📈 Sample data:")
    print(df.head())
    print(f"\n📋 Columns: {list(df.columns)}")
    
    return output_file

def create_enhanced_stock_data():
    """Create an enhanced stock data file with additional features."""
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock data
    np.random.seed(42)
    base_price = 100.0
    
    data = []
    for i, date in enumerate(dates):
        # Skip weekends (Saturday=5, Sunday=6)
        if date.weekday() >= 5:
            continue
            
        # Simulate stock price movement with some trend and volatility
        trend = 0.0001 * i  # Slight upward trend
        volatility = 0.02  # 2% daily volatility
        
        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        base_price *= (1 + price_change)
        
        # Ensure price doesn't go negative
        base_price = max(base_price, 1.0)
        
        # Generate OHLCV data
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, base_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, base_price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = np.random.randint(1000000, 10000000)  # 1M to 10M volume
        
        # Calculate additional features
        price_change_pct = (close_price - open_price) / open_price * 100
        high_low_range = high_price - low_price
        volume_ma = volume  # Simplified for now
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'vol': volume,
            'price_change_pct': round(price_change_pct, 2),
            'high_low_range': round(high_low_range, 2),
            'volume_ma': volume_ma
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'enhanced_stock_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created enhanced stock data file: {output_file}")
    print(f"📊 Shape: {df.shape}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"📈 Sample data:")
    print(df.head())
    print(f"\n📋 Columns: {list(df.columns)}")
    
    return output_file

if __name__ == "__main__":
    print("🚀 Creating Sample Stock Data Files")
    print("=" * 50)
    
    # Create basic sample data
    basic_file = create_sample_stock_data()
    
    print("\n" + "=" * 50)
    
    # Create enhanced sample data
    enhanced_file = create_enhanced_stock_data()
    
    print(f"\n🎯 Created two sample files:")
    print(f"  1. {basic_file} - Basic OHLCV data")
    print(f"  2. {enhanced_file} - Enhanced data with additional features")
    print(f"\n💡 You can now use these files for training in the GUI!") 