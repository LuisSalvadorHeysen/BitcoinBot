import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

# --- Simulation Settings ---
DATA_FILE = 'project3/data/bitcoin_usd.csv'
INITIAL_CASH = 500000  # USD
WINDOW = 60  # Rolling window size for training

# --- Realism Settings ---
FEE_RATE = 0.001  # 0.1% fee per trade
SLIPPAGE_RATE = 0.0005  # 0.05% random slippage
HOLD_THRESHOLD = 0.0005  # 0.05% threshold for trading

def calculate_technical_indicators(df):
    """Calculate technical indicators for feature engineering"""
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Price changes
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    
    # Volume (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df

def prepare_ml_data(df):
    """Prepare data with technical indicators for ML"""
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Select features for ML
    feature_columns = [
        'close', 'high', 'low', 'open',
        'sma_5', 'sma_20', 'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_middle',
        'price_change_1', 'price_change_5', 'price_change_10'
    ]
    
    # Add volume features if available
    if 'volume_ratio' in df.columns:
        feature_columns.append('volume_ratio')
    
    ml_df = df[['timestamp'] + feature_columns].copy()
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

def fetch_coingecko_data(symbol='bitcoin', vs_currency='usd', days=1):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; BitcoinBot/1.0; +https://github.com/yourusername/bitcoinbot)'
    }
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['close'].shift(1)
    df['high'] = df[['open', 'close']].max(axis=1)
    df['low'] = df[['open', 'close']].min(axis=1)
    df = df.dropna().reset_index(drop=True)
    return df

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
ml_df = prepare_ml_data(df)

# Initialize portfolio
cash = INITIAL_CASH
btc = 0.0
portfolio_values = []
trade_log = []
shap_values_list = []

print("Starting advanced backtest with technical indicators...")
print(f"Data points: {len(ml_df)}")
print(f"Features: {len(ml_df.columns) - 2}")  # -2 for timestamp and next_output

for i in range(WINDOW, len(ml_df) - 1):
    train = ml_df.iloc[i-WINDOW:i]
    test = ml_df.iloc[i+1:i+2]
    
    # Prepare features (exclude timestamp and next_output)
    feature_cols = [col for col in train.columns if col not in ['timestamp', 'next_output']]
    X_train = train[feature_cols]
    y_train = train['next_output']
    X_test = test[feature_cols]
    
    current_price = test['close'].values[0]
    current_time = test['timestamp'].values[0]

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)[0]
    pct_move = (pred - current_price) / current_price

    # Trading logic
    trade_action = 'HOLD'
    if pct_move > HOLD_THRESHOLD and cash > 0:
        # Buy with fees and slippage
        buy_price = current_price * (1 + SLIPPAGE_RATE * np.random.rand())
        btc_to_buy = (cash / buy_price)
        fee = btc_to_buy * FEE_RATE
        btc_to_buy -= fee
        
        btc += btc_to_buy
        cash = 0
        trade_action = 'BUY'
        trade_log.append((current_time, 'BUY', buy_price, btc_to_buy, fee * buy_price))

    elif pct_move < -HOLD_THRESHOLD and btc > 0:
        # Sell with fees and slippage
        sell_price = current_price * (1 - SLIPPAGE_RATE * np.random.rand())
        fee = btc * sell_price * FEE_RATE
        
        cash = btc * sell_price - fee
        btc = 0
        trade_action = 'SELL'
        trade_log.append((current_time, 'SELL', sell_price, btc, fee))

    # Print every 10th step to avoid spam
    if i % 10 == 0:
        print(f"{current_time} - Price: ${current_price:,.2f}, Predicted Move: {pct_move:+.4f}%, Action: {trade_action}")

    portfolio_value = cash + btc * current_price
    portfolio_values.append(portfolio_value)

print("Advanced backtest finished.")

# Performance Analysis
returns_df = pd.DataFrame(portfolio_values, columns=['value'])
returns_df['daily_return'] = returns_df['value'].pct_change()

std_return = returns_df['daily_return'].std()
if std_return and not np.isnan(std_return):
    sharpe_ratio = (returns_df['daily_return'].mean() / std_return) * np.sqrt(252)
    sharpe_str = f"{sharpe_ratio:.2f}"
else:
    sharpe_str = "N/A (insufficient return variance)"

peak = returns_df['value'].expanding(min_periods=1).max()
drawdown = (returns_df['value'] - peak) / peak
max_drawdown = drawdown.min()

print("\n--- Advanced Performance Metrics ---")
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Return: {(portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH * 100:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_str}")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
print(f"Total Trades: {len(trade_log)}")

if trade_log:
    log_df = pd.DataFrame(trade_log, columns=['Timestamp', 'Action', 'Price', 'Quantity', 'Fee (USD)'])
    log_df.to_csv('advanced_trade_log.csv', index=False)
    print("\nAdvanced trade log saved to advanced_trade_log.csv")
else:
    print("\nNo trades were executed.")

# Plotting
timestamps = df['timestamp'].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].values
df_plot = pd.DataFrame({'timestamp': timestamps, 'portfolio_value': portfolio_values})
df_plot.set_index('timestamp')['portfolio_value'].plot(title='Portfolio Value Over Time (Advanced Model)')
plt.ylabel('USD')
plt.savefig('portfolio_value_advanced.png')
print('Advanced portfolio value plot saved as portfolio_value_advanced.png')
plt.clf()

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print('Feature importance plot saved as feature_importance.png')
plt.close() 