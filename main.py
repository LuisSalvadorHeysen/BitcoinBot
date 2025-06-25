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
HISTORY_MINUTES = 100

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

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

def main():
    """
    Main function that continuously fetches data, trains model, and makes predictions.
    Runs in an infinite loop with 60-second intervals.
    """
    print("StockBot: Starting data fetching and prediction...")
    print(f"Fetching {SYMBOL} price data every 60 seconds")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            # Fetch and save data
            stock_data = fetch_stock_data(SYMBOL, HISTORY_MINUTES)
            save_data(stock_data, SYMBOL)

            # Prepare ML data
            ml_df = prepare_ml_data(stock_data)
            if len(ml_df) < 10:
                print("Not enough data to train model.")
                time.sleep(60)
                continue
            train_data = ml_df.iloc[:-1]
            test_data = ml_df.iloc[-1:]
            X_train = train_data[['close', 'high', 'low', 'open']]
            y_train = train_data['next_output']

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict next price (for demonstration)
            latest_row = test_data.iloc[0]
            X_pred = latest_row[['close', 'high', 'low', 'open']].values.reshape(1, -1)
            predicted_next = model.predict(X_pred)[0]
            current_price = latest_row['close']
            pct_change = ((predicted_next - current_price) / current_price) * 100
            
            print(f"Current price: ${current_price:,.2f}, Predicted next: ${predicted_next:,.2f} ({pct_change:+.2f}%)")

            time.sleep(60)
        except KeyboardInterrupt:
            print("\nStockBot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main() 