import os
import time
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Settings
SYMBOL = 'bitcoin'  # CoinGecko uses coin names
VS_CURRENCY = 'usd'
DATA_DIR = 'project3/data'
INTERVAL_MINUTES = 1
HISTORY_MINUTES = 100

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_coingecko_data(symbol, vs_currency, minutes=100):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": 1  # 1 day (CoinGecko returns up to 288 5-min points)
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; BitcoinBot/1.0; +https://github.com/yourusername/bitcoinbot)'
    }
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    # data['prices'] is a list of [timestamp, price]
    # data['market_caps'], data['total_volumes'] also available
    prices = data['prices'][-minutes:]
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # For ML, we need open, high, low, close. We'll approximate:
    df['open'] = df['close'].shift(1)
    df['high'] = df[['open', 'close']].max(axis=1)
    df['low'] = df[['open', 'close']].min(axis=1)
    df = df.dropna().reset_index(drop=True)
    return df

def save_data(df, symbol):
    file_name = symbol + '_' + VS_CURRENCY + '.csv'
    output_file_path = os.path.join(DATA_DIR, file_name)
    df.to_csv(output_file_path, index=False)
    print(f"Saved data to {output_file_path}")

def prepare_ml_data(df):
    ml_df = df[['close', 'high', 'low', 'open']].copy()
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

def main():
    while True:
        try:
            # Fetch and save data
            crypto_data = fetch_coingecko_data(SYMBOL, VS_CURRENCY, HISTORY_MINUTES)
            save_data(crypto_data, SYMBOL)

            # Prepare ML data
            ml_df = prepare_ml_data(crypto_data)
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
            print(f"Current price: {latest_row['close']:.2f}, Predicted next: {predicted_next:.2f}")

            time.sleep(60)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main() 