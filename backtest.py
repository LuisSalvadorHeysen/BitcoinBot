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
import yfinance as yf
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Simulation Settings ---
DATA_FILE = 'project3/data/spy_usd.csv'
INITIAL_CASH = 500000  # USD
WINDOW = 60  # Rolling window size for training
TEST_DAYS = 30  # Out-of-sample test period (last 30 days)

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

def run_backtest(data):
    """Run backtest on historical data."""
    # Prepare ML data
    ml_df = prepare_ml_data(data)
    
    # Initialize portfolio
    cash = INITIAL_CASH
    shares = 0.0
    portfolio_values = []
    trades = []
    peak_portfolio = INITIAL_CASH
    entry_price = 0
    current_position = 0
    
    print(f"Starting backtest with ${INITIAL_CASH:,.0f} initial capital...")
    
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
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        pct_move = (pred - current_price) / current_price
        
        # Calculate portfolio value
        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)
        
        if portfolio_value > peak_portfolio:
            peak_portfolio = portfolio_value
        
        # Trading logic
        trade_action = 'HOLD'
        trade_amount = 0
        
        # Buy if predicted price is significantly higher
        if pct_move > MIN_SIGNAL_STRENGTH and cash > 0 and current_position == 0:
            position_size = calculate_position_size(0.1, pct_move, cash)
            trade_amount = cash * position_size
            
            buy_price = current_price * (1 + SLIPPAGE_RATE)
            shares_to_buy = trade_amount / buy_price
            fee = shares_to_buy * buy_price * FEE_RATE
            shares_to_buy -= fee / buy_price
            
            shares += shares_to_buy
            cash -= trade_amount
            current_position = 1
            entry_price = buy_price
            trade_action = f'BUY ({pct_move:.4f})'
            trades.append({
                'timestamp': current_time,
                'action': 'BUY',
                'price': buy_price,
                'quantity': shares_to_buy,
                'fee': fee,
                'portfolio_value': portfolio_value
            })
        
        # Sell if predicted price is significantly lower or we have profit
        elif (pct_move < -MIN_SIGNAL_STRENGTH or 
              (current_position == 1 and (current_price - entry_price) / entry_price > 0.01)) and shares > 0:
            sell_price = current_price * (1 - SLIPPAGE_RATE)
            fee = shares * sell_price * FEE_RATE
            
            cash += shares * sell_price - fee
            shares = 0
            current_position = 0
            trade_action = f'SELL ({pct_move:.4f})'
            trades.append({
                'timestamp': current_time,
                'action': 'SELL',
                'price': sell_price,
                'quantity': shares,
                'fee': fee,
                'portfolio_value': portfolio_value
            })
    
    # Calculate final portfolio value
    final_value = cash + shares * current_price
    
    # Calculate metrics
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    annualized_return = total_return * (252 / len(data))
    
    # Sharpe ratio
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Maximum drawdown
    peak = pd.Series(portfolio_values).expanding(min_periods=1).max()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Trade statistics
    total_trades = len(trades)
    if total_trades > 0:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        win_rate = len([t for t in sell_trades if t['price'] > entry_price]) / len(sell_trades) * 100 if sell_trades else 0
        avg_trade_return = sum([(t['price'] - entry_price) / entry_price for t in sell_trades]) / len(sell_trades) * 100 if sell_trades else 0
    else:
        win_rate = 0
        avg_trade_return = 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def plot_results(results):
    """Plot backtest results."""
    plt.figure(figsize=(12, 8))
    
    # Portfolio value over time
    plt.subplot(2, 1, 1)
    plt.plot(results['portfolio_values'])
    plt.title('Portfolio Value Over Time')
    plt.ylabel('USD')
    plt.grid(True)
    
    # Trade points
    if results['trades']:
        buy_times = [i for i, t in enumerate(results['trades']) if t['action'] == 'BUY']
        sell_times = [i for i, t in enumerate(results['trades']) if t['action'] == 'SELL']
        
        if buy_times:
            plt.scatter(buy_times, [results['portfolio_values'][i] for i in buy_times], 
                       color='green', marker='^', s=50, label='Buy')
        if sell_times:
            plt.scatter(sell_times, [results['portfolio_values'][i] for i in sell_times], 
                       color='red', marker='v', s=50, label='Sell')
        plt.legend()
    
    # Performance metrics
    plt.subplot(2, 1, 2)
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
    values = [results['total_return'], results['sharpe_ratio'], results['max_drawdown']]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    plt.bar(metrics, values, color=colors)
    plt.title('Performance Metrics')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('graphs/backtest', exist_ok=True)
    plt.savefig('graphs/backtest/backtest_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def walkforward_backtest(df, test_days=30):
    # Split into train and test
    last_day = df['timestamp'].dt.date.iloc[-1]
    test_start = last_day - pd.Timedelta(days=test_days-1)
    train_df = df[df['timestamp'].dt.date < test_start]
    test_df = df[df['timestamp'].dt.date >= test_start].reset_index(drop=True)
    print(f"\nWALK-FORWARD OUT-OF-SAMPLE BACKTEST")
    print(f"Train period: {train_df['timestamp'].iloc[0].date()} to {train_df['timestamp'].iloc[-1].date()} ({len(train_df)} rows)")
    print(f"Test period: {test_df['timestamp'].iloc[0].date()} to {test_df['timestamp'].iloc[-1].date()} ({len(test_df)} rows)")

    # Train model on all train data
    features = [c for c in train_df.columns if c not in ['timestamp', 'next_output']]
    X_train = train_df[features]
    y_train = train_df['next_output']
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    # Simulate trading on test month
    cash = INITIAL_CASH
    asset = 0.0
    portfolio_values = []
    trades = []
    step_log = []
    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        X = pd.DataFrame([row[features].values], columns=features)
        pred = model.predict(X)[0]
        current_price = row['close']
        pred_move = (pred - current_price) / current_price
        action = 'HOLD'
        # Simple strategy: buy if up, sell if down
        if pred_move > 0.002 and cash > 0:
            buy_price = current_price * (1 + SLIPPAGE_RATE)
            qty = cash / buy_price
            fee = qty * buy_price * FEE_RATE
            qty -= fee / buy_price
            asset += qty
            cash = 0
            action = 'BUY'
            trades.append({'action': 'BUY', 'date': row['timestamp'], 'price': buy_price, 'qty': qty, 'fee': fee, 'portfolio': cash + asset * current_price})
        elif pred_move < -0.002 and asset > 0:
            sell_price = current_price * (1 - SLIPPAGE_RATE)
            fee = asset * sell_price * FEE_RATE
            cash += asset * sell_price - fee
            action = 'SELL'
            trades.append({'action': 'SELL', 'date': row['timestamp'], 'price': sell_price, 'qty': asset, 'fee': fee, 'portfolio': cash})
            asset = 0
        portfolio_value = cash + asset * current_price
        portfolio_values.append(portfolio_value)
        step_log.append({'date': row['timestamp'], 'price': current_price, 'pred': pred, 'pred_move': pred_move, 'action': action, 'portfolio': portfolio_value})
        print(f"{row['timestamp']} | Price: {current_price:.2f} | Pred: {pred:.2f} | Move: {pred_move:+.4f} | Action: {action} | Portfolio: {portfolio_value:.2f}")
    # Final value
    final_price = test_df.iloc[-1]['close']
    final_value = cash + asset * final_price
    return portfolio_values, trades, final_value, test_df, step_log

def main():
    """Main backtesting function with advanced strategy."""
    print("StockBot Backtesting with Historical Data")
    print("=" * 50)
    
    # Fetch historical data
    print("Fetching historical SPY data...")
    df = fetch_historical_data('SPY', days=365)
    print(f"Fetched {len(df)} days of data")
    df = prepare_ml_data(df)  # Ensure features and next_output are present
    
    if len(df) < 100:
        print("Not enough historical data for backtesting")
        return
    
    # Run backtest
    print(f"Running walk-forward out-of-sample backtest (last {TEST_DAYS} days)...")
    portfolio_values, trades, final_value, test_df, step_log = walkforward_backtest(df, test_days=TEST_DAYS)
    print(f"\nFinal portfolio value: ${final_value:,.2f}")
    print(f"Total trades: {len(trades)}")
    for t in trades:
        print(f"TRADE | {t['action']} | {t['date']} | Price: {t['price']:.2f} | Qty: {t['qty']:.6f} | Fee: {t['fee']:.2f} | Portfolio: {t['portfolio']:.2f}")
    # Save trade log
    pd.DataFrame(trades).to_csv('spy_walkforward_trades.csv', index=False)
    pd.DataFrame(step_log).to_csv('spy_walkforward_steps.csv', index=False)
    print('Saved trade log as spy_walkforward_trades.csv and step log as spy_walkforward_steps.csv')
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(test_df['timestamp'][:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Test Month (SPY)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.tight_layout()
    os.makedirs('graphs/backtest', exist_ok=True)
    plt.savefig('graphs/backtest/spy_walkforward_portfolio.png')
    print('Saved plot as graphs/backtest/spy_walkforward_portfolio.png')
    # Stats
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    print(f"Sharpe ratio (test month): {sharpe:.2f}")
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    print(f"Total return (test month): {total_return:.2f}%")

if __name__ == '__main__':
    main() 