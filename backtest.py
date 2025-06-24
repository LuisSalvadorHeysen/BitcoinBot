import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Simulation Settings ---
DATA_FILE = 'project3/data/bitcoin_usd.csv'
INITIAL_CASH = 10000  # USD
WINDOW = 30  # Rolling window size for training

# --- Realism Settings ---
FEE_RATE = 0.001  # 0.1% fee per trade
SLIPPAGE_RATE = 0.0005  # 0.05% random slippage
HOLD_THRESHOLD = 0.002 # Must predict a 0.2% move to overcome fees+slippage

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])

def prepare_ml_data(df):
    ml_df = df[['timestamp', 'close', 'high', 'low', 'open']].copy()
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

ml_df = prepare_ml_data(df)

# --- Portfolio & Logging Initialization ---
cash = INITIAL_CASH
btc = 0.0
portfolio_values = []
trade_log = []
alerts = []
shap_values_list = []
explainer = None

print("Starting realistic backtest...")
for i in range(WINDOW, len(ml_df) - 1):
    train = ml_df.iloc[i-WINDOW:i]
    test = ml_df.iloc[i+1:i+2]
    X_train = train[['close', 'high', 'low', 'open']]
    y_train = train['next_output']
    X_test = test[['close', 'high', 'low', 'open']]
    current_price = test['close'].values[0]
    current_time = test['timestamp'].values[0]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    pct_move = (pred - current_price) / current_price

    if explainer is None:
        explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values_list.append(shap_values.values[0])

    # --- Realistic Trading Logic ---
    trade_action = 'HOLD'
    if pct_move > HOLD_THRESHOLD and cash > 0:
        # --- Simulate BUY with fees and slippage ---
        buy_price = current_price * (1 + SLIPPAGE_RATE * np.random.rand())
        btc_to_buy = (cash / buy_price)
        fee = btc_to_buy * FEE_RATE
        btc_to_buy -= fee
        
        btc += btc_to_buy
        cash = 0
        trade_action = 'BUY'
        trade_log.append((current_time, 'BUY', buy_price, btc_to_buy, fee * buy_price))

    elif pct_move < -HOLD_THRESHOLD and btc > 0:
        # --- Simulate SELL with fees and slippage ---
        sell_price = current_price * (1 - SLIPPAGE_RATE * np.random.rand())
        fee = btc * sell_price * FEE_RATE
        
        cash = btc * sell_price - fee
        btc = 0
        trade_action = 'SELL'
        trade_log.append((current_time, 'SELL', sell_price, btc, fee))

    if i % 10 == 0:
        print(f"{current_time} - Price: ${current_price:,.2f}, Predicted Move: {pct_move:+.4f}%, Action: {trade_action}")

    portfolio_value = cash + btc * current_price
    portfolio_values.append(portfolio_value)

print("Backtest finished.")
# --- Performance Analysis ---
returns_df = pd.DataFrame(portfolio_values, columns=['value'])
returns_df['daily_return'] = returns_df['value'].pct_change()

sharpe_ratio = (returns_df['daily_return'].mean() / returns_df['daily_return'].std()) * np.sqrt(252) # Annualized

peak = returns_df['value'].expanding(min_periods=1).max()
drawdown = (returns_df['value'] - peak) / peak
max_drawdown = drawdown.min()

print("\n--- Performance Metrics ---")
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Return: {(portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH * 100:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
if trade_log:
    log_df = pd.DataFrame(trade_log, columns=['Timestamp', 'Action', 'Price', 'Quantity', 'Fee (USD)'])
    log_df.to_csv('trade_log.csv', index=False)
    print("\nTrade log saved to trade_log.csv")
else:
    print("\nNo trades were executed.")

# --- Plotting ---
timestamps = df['timestamp'].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].values
df_plot = pd.DataFrame({'timestamp': timestamps, 'portfolio_value': portfolio_values})
df_plot.set_index('timestamp')['portfolio_value'].plot(title='Portfolio Value Over Time (Realistic Simulation)')
plt.ylabel('USD')
plt.savefig('portfolio_value_realistic.png')
print('Portfolio value plot saved as portfolio_value_realistic.png')
plt.clf()

# --- SHAP Plot ---
feature_cols = ['close', 'high', 'low', 'open']
X_shap = ml_df[feature_cols].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].copy()
X_shap.index = range(len(X_shap))
shap.summary_plot(np.array(shap_values_list), X_shap, feature_names=feature_cols, show=False)
fig = plt.gcf()
fig.savefig('shap_summary.png')
plt.close(fig)
print('SHAP summary plot saved as shap_summary.png') 