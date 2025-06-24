import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_FILE = 'project3/data/bitcoin_usd.csv'
INITIAL_CASH = 10000  # USD
WINDOW = 30  # Rolling window size for training
ALERT_THRESHOLD = 0.01  # 1% predicted move

# Load data
df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])

# Prepare ML data
def prepare_ml_data(df):
    ml_df = df[['close', 'high', 'low', 'open']].copy()
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

ml_df = prepare_ml_data(df)

# Initialize portfolio
cash = INITIAL_CASH
btc = 0.0
portfolio_values = []
alerts = []
shap_values_list = []
explainer = None

for i in range(WINDOW, len(ml_df) - 1):
    train = ml_df.iloc[i-WINDOW:i]
    test = ml_df.iloc[i+1:i+2]
    X_train = train[['close', 'high', 'low', 'open']]
    y_train = train['next_output']
    X_test = test[['close', 'high', 'low', 'open']]
    current_price = test['close'].values[0]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)[0]
    pct_move = (pred - current_price) / current_price

    # SHAP explainability
    if explainer is None:
        explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values_list.append(shap_values.values[0])

    # Alerting
    if abs(pct_move) > ALERT_THRESHOLD:
        direction = 'UP' if pct_move > 0 else 'DOWN'
        alerts.append((test['timestamp'].values[0], direction, pct_move))
        print(f"ALERT: {test['timestamp'].values[0]} Model predicts {direction} move of {pct_move*100:.2f}%")

    # Paper trading logic: buy if up, sell if down
    if pct_move > 0 and cash > 0:
        # Buy BTC with all cash
        btc = cash / current_price
        cash = 0
    elif pct_move < 0 and btc > 0:
        # Sell all BTC
        cash = btc * current_price
        btc = 0
    # Track portfolio value
    portfolio_value = cash + btc * current_price
    portfolio_values.append(portfolio_value)

# Final stats
print(f"Final portfolio value: ${portfolio_values[-1]:.2f}")
returns = (portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH * 100
print(f"Total return: {returns:.2f}%")

# Get the correct timestamps from the original dataframe
timestamps = df['timestamp'].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].values
df_plot = pd.DataFrame({
    'timestamp': timestamps,
    'portfolio_value': portfolio_values
})
df_plot.set_index('timestamp')['portfolio_value'].plot(title='Portfolio Value Over Time')
plt.ylabel('USD')
plt.savefig('portfolio_value.png')
print('Portfolio value plot saved as portfolio_value.png')

# Only use the feature columns for SHAP summary plot
feature_cols = ['close', 'high', 'low', 'open']
X_shap = ml_df[feature_cols].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].copy()
X_shap.index = range(len(X_shap))
plt.clf()  # Clear any previous matplotlib state
shap.summary_plot(
    np.array(shap_values_list),
    X_shap,
    feature_names=feature_cols,
    show=False
)
fig = plt.gcf()
fig.savefig('shap_summary.png')
plt.close(fig)
print('SHAP summary plot saved as shap_summary.png') 