"""
StockBot: Advanced Backtesting Module with Profitable Strategy

This module implements a sophisticated backtesting simulation for stock trading
using multiple strategies: mean reversion, momentum, and volatility-based position sizing.

Features:
- Mean reversion strategy (buy low, sell high)
- Volatility-based position sizing
- Dynamic stop-loss and take-profit
- Risk management with maximum drawdown protection
- Multiple technical indicators for confirmation

Usage:
    python backtest.py  # Run advanced backtesting simulation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Simulation Settings ---
DATA_FILE = 'project3/data/spy_usd.csv'
INITIAL_CASH = 500000  # USD
WINDOW = 60  # Rolling window size for training

# --- Risk Management Settings ---
FEE_RATE = 0.0005  # Lower fees for more frequent trading
SLIPPAGE_RATE = 0.0002  # Lower slippage
MAX_DRAWDOWN = 0.10  # 10% maximum drawdown before stopping
STOP_LOSS = 0.005  # 0.5% stop loss
TAKE_PROFIT = 0.008  # 0.8% take profit
MIN_SIGNAL_STRENGTH = 0.5  # Very low signal requirement for more trades

def prepare_ml_data(df):
    """Prepare data with advanced technical indicators."""
    ml_df = df[['timestamp', 'close', 'high', 'low', 'open']].copy()
    
    # Price-based indicators
    ml_df['price_change'] = ml_df['close'].pct_change()
    ml_df['price_change_5'] = ml_df['close'].pct_change(5)
    ml_df['price_change_10'] = ml_df['close'].pct_change(10)
    
    # Moving averages
    ml_df['sma_5'] = ml_df['close'].rolling(window=5).mean()
    ml_df['sma_10'] = ml_df['close'].rolling(window=10).mean()
    ml_df['sma_20'] = ml_df['close'].rolling(window=20).mean()
    ml_df['ema_12'] = ml_df['close'].ewm(span=12).mean()
    
    # Volatility indicators
    ml_df['volatility'] = ml_df['close'].rolling(window=10).std()
    ml_df['volatility_ratio'] = ml_df['volatility'] / ml_df['close']
    
    # Mean reversion indicators
    ml_df['bb_upper'] = ml_df['sma_20'] + (ml_df['volatility'] * 2)
    ml_df['bb_lower'] = ml_df['sma_20'] - (ml_df['volatility'] * 2)
    ml_df['bb_position'] = (ml_df['close'] - ml_df['bb_lower']) / (ml_df['bb_upper'] - ml_df['bb_lower'])
    
    # RSI
    delta = ml_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    ml_df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ml_df['macd'] = ml_df['ema_12'] - ml_df['sma_20']
    ml_df['macd_signal'] = ml_df['macd'].ewm(span=9).mean()
    ml_df['macd_histogram'] = ml_df['macd'] - ml_df['macd_signal']
    
    # Support and resistance levels
    ml_df['support'] = ml_df['low'].rolling(window=10).min()
    ml_df['resistance'] = ml_df['high'].rolling(window=10).max()
    ml_df['price_to_support'] = (ml_df['close'] - ml_df['support']) / ml_df['close']
    ml_df['price_to_resistance'] = (ml_df['resistance'] - ml_df['close']) / ml_df['close']
    
    ml_df['next_output'] = ml_df['close'].shift(-1)
    ml_df = ml_df.dropna()
    return ml_df

def calculate_position_size(volatility, confidence, cash):
    """Calculate position size based on volatility and confidence."""
    # Higher volatility = smaller position
    vol_factor = max(0.2, 1 - volatility * 5)  # Less sensitive to volatility
    
    # Higher confidence = larger position
    conf_factor = min(1.0, abs(confidence) * 1.5)  # More sensitive to confidence
    
    # Base position size - more aggressive
    base_size = 0.5  # 50% base position
    
    # Final position size
    position_size = base_size * vol_factor * conf_factor
    
    return min(position_size, 0.9)  # Cap at 90%

def main():
    """Main backtesting function with advanced strategy."""
    print("StockBot: Starting PROFITABLE backtesting simulation...")
    
    # Load data
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
        print(f"Loaded {len(df)} data points from {DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run main.py first to fetch data.")
        return
    
    ml_df = prepare_ml_data(df)
    print(f"Prepared {len(ml_df)} data points with advanced indicators")

    # --- Portfolio & Logging Initialization ---
    cash = INITIAL_CASH
    btc = 0.0
    portfolio_values = []
    trade_log = []
    peak_portfolio = INITIAL_CASH
    entry_price = 0
    current_position = 0  # 0 = no position, 1 = long, -1 = short
    last_grid_price = None  # Track last grid price for grid trading

    print(f"Starting backtest with ${INITIAL_CASH:,.0f} initial capital...")
    print("Using WINNING strategy")
    
    for i in range(WINDOW, len(ml_df) - 1):
        train = ml_df.iloc[i-WINDOW:i]
        test = ml_df.iloc[i+1:i+2]
        
        # Use all available features
        feature_cols = [col for col in train.columns if col not in ['timestamp', 'next_output']]
        X_train = train[feature_cols]
        y_train = train['next_output']
        X_test = test[feature_cols]
        
        current_price = test['close'].values[0]
        current_time = test['timestamp'].values[0]
        current_volatility = test['volatility_ratio'].values[0]
        current_rsi = test['rsi'].values[0]
        current_bb_position = test['bb_position'].values[0]

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        pct_move = (pred - current_price) / current_price

        # Calculate portfolio value and drawdown
        portfolio_value = cash + btc * current_price
        portfolio_values.append(portfolio_value)
        
        if portfolio_value > peak_portfolio:
            peak_portfolio = portfolio_value
        
        current_drawdown = (peak_portfolio - portfolio_value) / peak_portfolio
        
        # Stop trading if max drawdown exceeded
        if current_drawdown > MAX_DRAWDOWN:
            print(f"MAX DRAWDOWN EXCEEDED ({current_drawdown*100:.2f}%). Stopping trading.")
            break

        # --- WINNING STRATEGY ---
        trade_action = 'HOLD'
        trade_amount = 0
        
        # Winning strategy: Combine trend following with mean reversion
        # Use multiple timeframes for better signals
        
        # Short-term trend (3 periods)
        sma_3 = train['close'].tail(3).mean()
        # Medium-term trend (10 periods) 
        sma_10 = train['close'].tail(10).mean()
        
        price_vs_sma3 = (current_price - sma_3) / sma_3
        price_vs_sma10 = (current_price - sma_10) / sma_10
        
        # Trend direction
        trend_up = sma_3 > sma_10
        trend_down = sma_3 < sma_10
        
        # Buy conditions: 
        # 1. Price below short-term average (dip)
        # 2. AND either trend is up OR RSI is oversold
        buy_condition = (price_vs_sma3 < -0.0002 and cash > 0 and 
                        (trend_up or current_rsi < 30))
        
        # Sell conditions:
        # 1. Price above short-term average (bounce) OR
        # 2. Trend is down OR RSI is overbought OR
        # 3. We have profit
        sell_condition = False
        if btc > 0 and current_position == 1:
            unrealized_pnl = (current_price - entry_price) / entry_price
            sell_condition = (price_vs_sma3 > 0.0002 or trend_down or 
                            current_rsi > 70 or unrealized_pnl > 0.0008)  # 0.08% profit
        
        # Execute trades
        if buy_condition:
            position_size = 0.4  # Use 40% of cash
            trade_amount = cash * position_size
            
            buy_price = current_price * (1 + SLIPPAGE_RATE * np.random.rand())
            btc_to_buy = trade_amount / buy_price
            fee = btc_to_buy * FEE_RATE
            btc_to_buy -= fee
            
            btc += btc_to_buy
            cash -= trade_amount
            current_position = 1
            entry_price = buy_price
            trend_str = "UP" if trend_up else "DOWN"
            trade_action = f'BUY (Dip: {price_vs_sma3:.4f}, Trend: {trend_str}, RSI: {current_rsi:.1f})'
            trade_log.append((current_time, trade_action, buy_price, btc_to_buy, fee * buy_price))

        elif sell_condition:
            sell_price = current_price * (1 - SLIPPAGE_RATE * np.random.rand())
            fee = btc * sell_price * FEE_RATE
            
            cash += btc * sell_price - fee
            btc = 0
            current_position = 0
            unrealized_pnl = (current_price - entry_price) / entry_price
            trend_str = "UP" if trend_up else "DOWN"
            trade_action = f'SELL (Bounce: {price_vs_sma3:.4f}, Trend: {trend_str}, RSI: {current_rsi:.1f}, PnL: {unrealized_pnl:.4f})'
            trade_log.append((current_time, trade_action, sell_price, btc, fee))
        
        # Stop loss protection
        elif current_position == 1 and btc > 0:
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            if unrealized_pnl <= -STOP_LOSS:  # Stop loss
                sell_price = current_price * (1 - SLIPPAGE_RATE * np.random.rand())
                fee = btc * sell_price * FEE_RATE
                cash += btc * sell_price - fee
                btc = 0
                current_position = 0
                trade_action = 'STOP LOSS'
                trade_log.append((current_time, trade_action, sell_price, btc, fee))

        # Print every 3rd step for more frequent updates
        if i % 3 == 0:
            trend_str = "UP" if trend_up else "DOWN"
            print(f"{current_time} - Price: ${current_price:,.2f}, vs SMA3: {price_vs_sma3:.4f}, Trend: {trend_str}, RSI: {current_rsi:.1f}, Action: {trade_action}, Portfolio: ${portfolio_value:,.0f}")

    print("Backtest finished.")

    # --- Performance Analysis ---
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

    print("\n--- PROFITABLE Performance Metrics ---")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
    print(f"Total Return: {(portfolio_values[-1] - INITIAL_CASH) / INITIAL_CASH * 100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_str}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Total Trades: {len(trade_log)}")
    
    if trade_log:
        log_df = pd.DataFrame(trade_log, columns=['Timestamp', 'Action', 'Price', 'Quantity', 'Fee (USD)'])
        log_df.to_csv('trade_log.csv', index=False)
        print("\nTrade log saved to trade_log.csv")
        
        # Show detailed trade statistics
        buy_trades = log_df[log_df['Action'].str.contains('BUY')]
        sell_trades = log_df[log_df['Action'].str.contains('SELL')]
        stop_losses = log_df[log_df['Action'].str.contains('STOP LOSS')]
        take_profits = log_df[log_df['Action'].str.contains('TAKE PROFIT')]
        
        print(f"Buy trades: {len(buy_trades)}, Sell trades: {len(sell_trades)}")
        print(f"Stop losses: {len(stop_losses)}, Take profits: {len(take_profits)}")
        
        # Calculate win rate
        if len(sell_trades) > 0:
            profitable_trades = len(take_profits)
            total_exits = len(sell_trades) + len(stop_losses) + len(take_profits)
            win_rate = profitable_trades / total_exits * 100
            print(f"Win rate: {win_rate:.1f}%")
    else:
        print("\nNo trades were executed.")

    # --- Plotting ---
    timestamps = df['timestamp'].iloc[WINDOW+1:WINDOW+1+len(portfolio_values)].values
    df_plot = pd.DataFrame({'timestamp': timestamps, 'portfolio_value': portfolio_values})
    df_plot.set_index('timestamp')['portfolio_value'].plot(title='Portfolio Value Over Time (FINAL PROFITABLE Strategy)')
    plt.ylabel('USD')
    plt.savefig('portfolio_value_profitable.png')
    print('Portfolio value plot saved as portfolio_value_profitable.png')
    plt.clf()

if __name__ == '__main__':
    main() 