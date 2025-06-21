#!/usr/bin/env python3
"""
Data Converter Utility

This script converts various data file formats into the proper OHLCV format
required by the stock prediction neural network.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

def detect_data_format(df):
    """
    Detect the format of the input data and suggest conversion strategy.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        dict: Format information and conversion strategy
    """
    columns = list(df.columns)
    
    # Check for OHLCV format
    ohlcv_columns = ['open', 'high', 'low', 'close', 'vol']
    if all(col in columns for col in ohlcv_columns):
        return {
            'format': 'OHLCV',
            'status': 'ready',
            'message': 'Data already in OHLCV format'
        }
    
    # Check for generic feature format (feature_1, feature_2, etc.)
    feature_columns = [col for col in columns if col.startswith('feature_')]
    if len(feature_columns) >= 4:
        return {
            'format': 'generic_features',
            'status': 'convertible',
            'message': f'Found {len(feature_columns)} generic feature columns',
            'strategy': 'map_features_to_ohlcv'
        }
    
    # Check for numeric columns that could be converted
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) >= 4:
        return {
            'format': 'numeric_data',
            'status': 'convertible',
            'message': f'Found {len(numeric_columns)} numeric columns',
            'strategy': 'generate_ohlcv_from_numeric'
        }
    
    # Check for date/time columns
    date_columns = []
    for col in columns:
        try:
            pd.to_datetime(df[col].iloc[0])
            date_columns.append(col)
        except:
            pass
    
    if date_columns:
        return {
            'format': 'time_series',
            'status': 'convertible',
            'message': f'Found time series data with date columns: {date_columns}',
            'strategy': 'generate_ohlcv_from_timeseries'
        }
    
    return {
        'format': 'unknown',
        'status': 'incompatible',
        'message': 'Unable to determine data format'
    }

def convert_generic_features_to_ohlcv(df):
    """
    Convert generic feature columns to OHLCV format.
    
    Args:
        df (pandas.DataFrame): Input dataframe with feature_1, feature_2, etc.
        
    Returns:
        pandas.DataFrame: Dataframe with OHLCV columns
    """
    feature_columns = [col for col in df.columns if col.startswith('feature_')]
    
    if len(feature_columns) < 4:
        raise ValueError(f"Need at least 4 feature columns, found {len(feature_columns)}")
    
    # Map features to OHLCV
    # feature_1 -> open, feature_2 -> high, feature_3 -> low, feature_4 -> close
    # If there's a feature_5, use it as volume, otherwise generate volume
    
    converted_df = pd.DataFrame()
    
    # Map the first 4 features to OHLC
    if len(feature_columns) >= 4:
        converted_df['open'] = df[feature_columns[0]]
        converted_df['high'] = df[feature_columns[1]]
        converted_df['low'] = df[feature_columns[2]]
        converted_df['close'] = df[feature_columns[3]]
    else:
        # If less than 4 features, duplicate the available ones
        available_features = feature_columns[:len(feature_columns)]
        while len(available_features) < 4:
            available_features.append(available_features[-1])
        
        converted_df['open'] = df[available_features[0]]
        converted_df['high'] = df[available_features[1]]
        converted_df['low'] = df[available_features[2]]
        converted_df['close'] = df[available_features[3]]
    
    # Generate volume data
    if len(feature_columns) >= 5:
        # Use feature_5 as volume if available
        converted_df['vol'] = df[feature_columns[4]]
    else:
        # Generate realistic volume data based on price range
        price_range = converted_df['high'] - converted_df['low']
        base_volume = 1000000  # 1M base volume
        volume_multiplier = np.random.uniform(0.5, 2.0, len(df))
        converted_df['vol'] = (base_volume * volume_multiplier * (1 + price_range / 100)).astype(int)
    
    # Add date column if not present
    if 'date' not in df.columns:
        start_date = datetime.now() - timedelta(days=len(df))
        dates = pd.date_range(start=start_date, periods=len(df), freq='D')
        converted_df['date'] = dates
    
    return converted_df

def convert_numeric_to_ohlcv(df):
    """
    Convert numeric data to OHLCV format by treating it as price data.
    
    Args:
        df (pandas.DataFrame): Input dataframe with numeric columns
        
    Returns:
        pandas.DataFrame: Dataframe with OHLCV columns
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 4:
        raise ValueError(f"Need at least 4 numeric columns, found {len(numeric_columns)}")
    
    converted_df = pd.DataFrame()
    
    # Use first 4 numeric columns as OHLC
    converted_df['open'] = df[numeric_columns[0]]
    converted_df['high'] = df[numeric_columns[1]]
    converted_df['low'] = df[numeric_columns[2]]
    converted_df['close'] = df[numeric_columns[3]]
    
    # Generate volume data
    if len(numeric_columns) >= 5:
        converted_df['vol'] = df[numeric_columns[4]]
    else:
        price_range = converted_df['high'] - converted_df['low']
        base_volume = 1000000
        volume_multiplier = np.random.uniform(0.5, 2.0, len(df))
        converted_df['vol'] = (base_volume * volume_multiplier * (1 + price_range / 100)).astype(int)
    
    # Add date column
    start_date = datetime.now() - timedelta(days=len(df))
    dates = pd.date_range(start=start_date, periods=len(df), freq='D')
    converted_df['date'] = dates
    
    return converted_df

def convert_timeseries_to_ohlcv(df, date_column=None):
    """
    Convert time series data to OHLCV format.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        date_column (str): Name of the date column
        
    Returns:
        pandas.DataFrame: Dataframe with OHLCV columns
    """
    # Find date column if not specified
    if date_column is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].iloc[0])
                date_column = col
                break
            except:
                continue
    
    if date_column is None:
        raise ValueError("No date column found")
    
    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Get numeric columns for price data
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 4:
        raise ValueError(f"Need at least 4 numeric columns for OHLCV, found {len(numeric_columns)}")
    
    converted_df = pd.DataFrame()
    converted_df['date'] = df[date_column]
    
    # Map numeric columns to OHLCV
    converted_df['open'] = df[numeric_columns[0]]
    converted_df['high'] = df[numeric_columns[1]]
    converted_df['low'] = df[numeric_columns[2]]
    converted_df['close'] = df[numeric_columns[3]]
    
    # Generate volume if not available
    if len(numeric_columns) >= 5:
        converted_df['vol'] = df[numeric_columns[4]]
    else:
        price_range = converted_df['high'] - converted_df['low']
        base_volume = 1000000
        volume_multiplier = np.random.uniform(0.5, 2.0, len(df))
        converted_df['vol'] = (base_volume * volume_multiplier * (1 + price_range / 100)).astype(int)
    
    return converted_df

def convert_data_file(input_file, output_file=None, strategy='auto'):
    """
    Convert a data file to OHLCV format.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        strategy (str): Conversion strategy ('auto', 'generic_features', 'numeric', 'timeseries')
        
    Returns:
        str: Path to the converted file
    """
    print(f"📊 Converting {input_file} to OHLCV format...")
    
    # Read input file
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Detect format
    format_info = detect_data_format(df)
    print(f"🔍 Detected format: {format_info['format']}")
    print(f"   Status: {format_info['status']}")
    print(f"   Message: {format_info['message']}")
    
    if format_info['status'] == 'ready':
        print("✅ Data is already in OHLCV format")
        if output_file is None:
            output_file = input_file.replace('.csv', '_ohlcv.csv')
        df.to_csv(output_file, index=False)
        return output_file
    
    elif format_info['status'] == 'incompatible':
        print("❌ Data format is incompatible")
        return None
    
    # Convert based on strategy
    try:
        if strategy == 'auto':
            strategy = format_info.get('strategy', 'generic_features')
        
        if strategy == 'map_features_to_ohlcv':
            converted_df = convert_generic_features_to_ohlcv(df)
        elif strategy == 'generate_ohlcv_from_numeric':
            converted_df = convert_numeric_to_ohlcv(df)
        elif strategy == 'generate_ohlcv_from_timeseries':
            converted_df = convert_timeseries_to_ohlcv(df)
        else:
            print(f"❌ Unknown strategy: {strategy}")
            return None
        
        # Generate output filename
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_converted_ohlcv.csv"
        
        # Save converted data
        converted_df.to_csv(output_file, index=False)
        
        print(f"✅ Conversion completed successfully!")
        print(f"   Output file: {output_file}")
        print(f"   Rows: {len(converted_df)}")
        print(f"   Columns: {list(converted_df.columns)}")
        
        # Validate OHLCV format
        ohlcv_columns = ['open', 'high', 'low', 'close', 'vol']
        if all(col in converted_df.columns for col in ohlcv_columns):
            print("✅ Output file has all required OHLCV columns")
        else:
            missing = [col for col in ohlcv_columns if col not in converted_df.columns]
            print(f"⚠️  Missing columns: {missing}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return None

def batch_convert_directory(input_dir, output_dir=None):
    """
    Convert all CSV files in a directory to OHLCV format.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path (optional)
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'converted')
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    print(f"🔄 Converting {len(csv_files)} CSV files in {input_dir}")
    
    successful_conversions = 0
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file.replace('.csv', '_ohlcv.csv'))
        
        print(f"\n📁 Processing: {csv_file}")
        result = convert_data_file(input_path, output_path)
        
        if result:
            successful_conversions += 1
    
    print(f"\n🎯 Batch conversion completed!")
    print(f"   Successful: {successful_conversions}/{len(csv_files)}")
    print(f"   Output directory: {output_dir}")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Convert data files to OHLCV format")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-s", "--strategy", choices=['auto', 'generic_features', 'numeric', 'timeseries'],
                       default='auto', help="Conversion strategy")
    parser.add_argument("--batch", action="store_true", help="Process all CSV files in directory")
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.input):
        batch_convert_directory(args.input, args.output)
    else:
        convert_data_file(args.input, args.output, args.strategy)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Interactive mode
        print("🔄 Data Converter - Interactive Mode")
        print("=" * 50)
        
        # Find CSV files in current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ No CSV files found in current directory")
            sys.exit(1)
        
        print(f"📁 Found {len(csv_files)} CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"   {i}. {file}")
        
        try:
            choice = int(input(f"\nSelect file to convert (1-{len(csv_files)}): ")) - 1
            if 0 <= choice < len(csv_files):
                input_file = csv_files[choice]
                convert_data_file(input_file)
            else:
                print("❌ Invalid choice")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
    else:
        main() 