"""
BitcoinBot: Data Fetching and Live Prediction Module

This module fetches stock price data from Yahoo Finance API and provides
real-time price predictions using a simple Linear Regression model.

Usage:
    python main.py  # Run continuous data fetching and prediction
"""

import os
import time
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Settings
SYMBOL = 'SPY'  # S&P 500 ETF - generally trends upward
DATA_DIR = 'project3/data'
HISTORY_MINUTES = 1000

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_historical_data(symbol, days=365):
    """
    Fetch historical stock price data from Yahoo Finance API.
    
    Args:
        symbol (str): Stock symbol (e.g., 'SPY')
        days (int): Number of days of historical data to fetch
    
    Returns:
        pd.DataFrame: DataFrame with timestamp, open, high, low, close prices
    """
    # Get historical daily data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1d")
    
    # Reset index to get timestamp as column
    df = df.reset_index()
    
    # The index column is 'Date', rename it to 'timestamp'
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'timestamp'})
    
    # Select only the OHLC columns we need and rename them
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close']
    
    return df.reset_index(drop=True)

def fetch_stock_data(symbol, minutes=None):
    """
    Fetch stock price data from Yahoo Finance API.
    
    Args:
        symbol (str): Stock symbol (e.g., 'SPY')
        minutes (int, optional): Number of minutes to fetch
    
    Returns:
        pd.DataFrame: DataFrame with timestamp, open, high, low, close prices
    """
    # Get 1 day of 1-minute data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="1d", interval="1m")
    
    # Reset index to get timestamp as column
    df = df.reset_index()
    
    # The index column is 'Datetime', rename it to 'timestamp'
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'timestamp'})
    
    # Select only the OHLC columns we need
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close']].copy()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close']
    
    if minutes:
        df = df.tail(minutes)
    
    return df.reset_index(drop=True)

def save_data(df, symbol):
    """Save DataFrame to CSV file in the data directory."""
    file_name = symbol.lower() + '_usd.csv'
    output_file_path = os.path.join(DATA_DIR, file_name)
    df.to_csv(output_file_path, index=False)
    print(f"Saved data to {output_file_path}")

def prepare_ml_data(df):
    """Prepare data for machine learning model training."""
    ml_df = df[['close', 'high', 'low', 'open']].copy()
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

def train_model(df):
    """Train a Linear Regression model on the given DataFrame."""
    ml_df = prepare_ml_data(df)
    X_train = ml_df[['close', 'high', 'low', 'open']]
    y_train = ml_df['next_output']
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_price(model, df):
    """Predict the next price using the trained model."""
    ml_df = prepare_ml_data(df)
    X_pred = pd.DataFrame([ml_df.iloc[-1][['close', 'high', 'low', 'open']].values], columns=['close', 'high', 'low', 'open'])
    return model.predict(X_pred)[0]

def main():
    """Main function to run the trading bot."""
    print("StockBot: Educational Algorithmic Trading Bot")
    print("=" * 50)
    
    # Fetch historical data for training
    print("Fetching historical data for model training...")
    historical_data = fetch_historical_data('SPY', days=365)
    print(f"Fetched {len(historical_data)} days of historical data")
    
    # Save historical data
    historical_data.to_csv(os.path.join(DATA_DIR, 'spy_historical.csv'), index=False)
    print(f"Saved historical data to {os.path.join(DATA_DIR, 'spy_historical.csv')}")
    
    # Train model on historical data
    print("Training model on historical data...")
    model = train_model(historical_data)
    print("Model training completed!")
    
    print("\nStarting live prediction loop...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Fetch current data
            current_data = fetch_stock_data('SPY', minutes=60)
            
            if current_data is not None and len(current_data) > 0:
                # Save current data
                current_data.to_csv(os.path.join(DATA_DIR, 'spy_usd.csv'), index=False)
                
                # Make prediction
                current_price = current_data['close'].iloc[-1]
                prediction = predict_price(model, current_data)
                
                if prediction is not None:
                    change_pct = ((prediction - current_price) / current_price) * 100
                    print(f"Current price: ${current_price:.2f}, Predicted next: ${prediction:.2f} ({change_pct:+.2f}%)")
                else:
                    print(f"Current price: ${current_price:.2f}, Prediction failed")
            else:
                print("Failed to fetch current data")
            
            time.sleep(60)  # Wait 1 minute before next prediction
            
    except KeyboardInterrupt:
        print("\nStopping StockBot...")
        print("Thank you for using StockBot!")

if __name__ == '__main__':
    main() 