import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time

# --- Config ---
SYMBOL = 'bitcoin'  # CoinGecko ID for BTC
VS_CURRENCY = 'usd'
DAYS = 30  # How many days of historical data (for demo, 30 days)
FEE_RATE = 0.001  # 0.1% per trade
SLIPPAGE_RATE = 0.0005  # 0.05% slippage
INITIAL_CASH = 10000
WINDOW = 60  # Rolling window for ML
TEST_DAYS = 7  # Out-of-sample test period (last 7 days)

# --- Data Fetching ---
def fetch_coingecko_ohlc(symbol, vs_currency, days):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/ohlc?vs_currency={vs_currency}&days={days}'
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"CoinGecko API error: {resp.status_code}")
    data = resp.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# --- Feature Engineering ---
def add_indicators(df):
    df = df.copy()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_upper'] = df['sma_20'] + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['sma_20'] - 2 * df['close'].rolling(20).std()
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['next_output'] = df['close'].shift(-1)
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Walk-Forward Out-of-Sample Backtest ---
def walkforward_backtest(df, test_days=7):
    # Split into train and test
    last_day = df['timestamp'].dt.date.iloc[-1]
    test_start = last_day - pd.Timedelta(days=test_days-1)
    train_df = df[df['timestamp'].dt.date < test_start]
    test_df = df[df['timestamp'].dt.date >= test_start].reset_index(drop=True)
    print(f"Train period: {train_df['timestamp'].iloc[0].date()} to {train_df['timestamp'].iloc[-1].date()}")
    print(f"Test period: {test_df['timestamp'].iloc[0].date()} to {test_df['timestamp'].iloc[-1].date()}")

    # Train model on all train data
    features = [c for c in train_df.columns if c not in ['timestamp', 'next_output']]
    X_train = train_df[features]
    y_train = train_df['next_output']
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    # Simulate trading on test week
    cash = INITIAL_CASH
    asset = 0.0
    portfolio_values = []
    trades = []
    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        X = row[features].values.reshape(1, -1)
        pred = model.predict(X)[0]
        current_price = row['close']
        pred_move = (pred - current_price) / current_price
        # Simple strategy: buy if up, sell if down
        if pred_move > 0.002 and cash > 0:
            buy_price = current_price * (1 + SLIPPAGE_RATE)
            qty = cash / buy_price
            fee = qty * buy_price * FEE_RATE
            qty -= fee / buy_price
            asset += qty
            cash = 0
            trades.append(('BUY', row['timestamp'], buy_price, qty, fee))
        elif pred_move < -0.002 and asset > 0:
            sell_price = current_price * (1 - SLIPPAGE_RATE)
            fee = asset * sell_price * FEE_RATE
            cash += asset * sell_price - fee
            trades.append(('SELL', row['timestamp'], sell_price, asset, fee))
            asset = 0
        portfolio_values.append(cash + asset * current_price)
    # Final value
    final_price = test_df.iloc[-1]['close']
    final_value = cash + asset * final_price
    return portfolio_values, trades, final_value, test_df

# --- Main ---
def main():
    print("Fetching BTC/USD historical data from CoinGecko...")
    df = fetch_coingecko_ohlc(SYMBOL, VS_CURRENCY, DAYS)
    print(f"Fetched {len(df)} rows.")
    df = add_indicators(df)
    print("Running walk-forward out-of-sample backtest (last week)...")
    portfolio_values, trades, final_value, test_df = walkforward_backtest(df, test_days=TEST_DAYS)
    print(f"Final portfolio value: ${final_value:,.2f}")
    print(f"Total trades: {len(trades)}")
    # Print trade log
    for t in trades:
        print(f"{t[0]} | {t[1]} | Price: {t[2]:.2f} | Qty: {t[3]:.6f} | Fee: {t[4]:.2f}")
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(test_df['timestamp'][:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Test Week (BTC/USD)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.tight_layout()
    plt.savefig('crypto_walkforward_portfolio.png')
    print('Saved plot as crypto_walkforward_portfolio.png')
    # Stats
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    print(f"Sharpe ratio (test week): {sharpe:.2f}")
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    print(f"Total return (test week): {total_return:.2f}%")

if __name__ == '__main__':
    main() 