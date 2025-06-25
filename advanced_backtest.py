"""
StockBot: Advanced Backtesting Module

This module implements an advanced backtesting simulation for stock trading
using Random Forest with 18+ technical indicators and feature importance analysis.

Features:
- Advanced ML model (Random Forest) with 18+ technical indicators
- Realistic trading with fees and slippage
- Feature importance analysis
- Performance metrics and visualization
- Trade logging and portfolio tracking

Usage:
    python advanced_backtest.py  # Run advanced backtesting simulation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import shap

# --- Simulation Settings ---
DATA_FILE = 'project3/data/spy_usd.csv'
INITIAL_CASH = 500000  # USD
WINDOW = 60  # Rolling window size for training
TEST_DAYS = 30  # Out-of-sample test period (last 30 days)

# --- Realism Settings ---
FEE_RATE = 0.0005  # Lower fees for more frequent trading
SLIPPAGE_RATE = 0.0002  # Lower slippage
HOLD_THRESHOLD = 0.0001  # Lowered to 0.01% to allow more trades
POSITION_SIZE = 0.5  # Use 50% of available cash per trade

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

def run_advanced_backtest(ml_df):
    """Run advanced backtest with Random Forest model."""
    # Initialize portfolio
    cash = INITIAL_CASH
    shares = 0.0
    portfolio_values = []
    trades = []
    peak_portfolio = INITIAL_CASH
    entry_price = 0
    current_position = 0
    
    print(f"Starting advanced backtest with ${INITIAL_CASH:,.0f} initial capital...")
    
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
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
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
        
        if pct_move > HOLD_THRESHOLD and cash > 0 and current_position == 0:
            # Buy with position sizing
            trade_amount = cash * POSITION_SIZE
            buy_price = current_price * (1 + SLIPPAGE_RATE)
            shares_to_buy = trade_amount / buy_price
            fee = shares_to_buy * buy_price * FEE_RATE
            shares_to_buy -= fee / buy_price
            
            shares += shares_to_buy
            cash -= trade_amount
            current_position = 1
            entry_price = buy_price
            trade_action = 'BUY'
            trades.append({
                'timestamp': current_time,
                'action': 'BUY',
                'price': buy_price,
                'quantity': shares_to_buy,
                'fee': fee,
                'portfolio_value': portfolio_value
            })
        
        elif pct_move < -HOLD_THRESHOLD and shares > 0:
            # Sell with position sizing
            sell_price = current_price * (1 - SLIPPAGE_RATE)
            fee = shares * sell_price * FEE_RATE
            
            cash += shares * sell_price - fee
            shares = 0
            current_position = 0
            trade_action = 'SELL'
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
    annualized_return = total_return * (252 / len(ml_df))
    
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

def generate_shap_analysis(ml_df):
    """Generate SHAP analysis for model interpretability."""
    # Prepare data for SHAP
    feature_cols = [col for col in ml_df.columns if col not in ['timestamp', 'next_output']]
    X = ml_df[feature_cols].iloc[-100:]  # Use last 100 samples for SHAP
    y = ml_df['next_output'].iloc[-100:]
    
    # Train model on recent data
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X, y)
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Plot SHAP summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance Summary')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
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
    plt.close()

def plot_advanced_results(results):
    """Plot advanced backtest results."""
    plt.figure(figsize=(12, 8))
    
    # Portfolio value over time
    plt.subplot(2, 1, 1)
    plt.plot(results['portfolio_values'])
    plt.title('Portfolio Value Over Time (Advanced Model)')
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
    plt.title('Advanced Performance Metrics')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('portfolio_value_advanced.png', dpi=300, bbox_inches='tight')
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
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
    model.fit(X_train, y_train)

    # Simulate trading on test month
    cash = INITIAL_CASH
    asset = 0.0
    portfolio_values = []
    trades = []
    step_log = []
    for i in range(len(test_df) - 1):
        row = test_df.iloc[i]
        X = row[features].values.reshape(1, -1)
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
    """Main advanced backtesting function."""
    print("StockBot: Advanced Backtesting with SHAP Analysis")
    print("=" * 50)
    
    # Fetch historical data
    print("Fetching historical SPY data...")
    df = fetch_historical_data('SPY', days=365)
    print(f"Fetched {len(df)} days of data")
    df = prepare_ml_data(df)  # Ensure features and next_output are present
    
    # Run backtest
    print("Running advanced backtest...")
    results = run_advanced_backtest(df)
    
    # Display results
    print("\n" + "=" * 50)
    print("ADVANCED BACKTEST RESULTS")
    print("=" * 50)
    print(f"Initial Portfolio Value: ${INITIAL_CASH:,.2f}")
    print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Average Trade Return: {results['avg_trade_return']:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(results['trades'])
    results_df.to_csv('advanced_trade_log.csv', index=False)
    print(f"\nDetailed results saved to advanced_trade_log.csv")
    
    # Generate SHAP analysis
    print("Generating SHAP analysis...")
    generate_shap_analysis(df)
    
    # Plot results
    plot_advanced_results(results)
    print("Performance plot saved as portfolio_value_advanced.png")

    # Run walk-forward out-of-sample backtest
    print("Running walk-forward out-of-sample backtest (last 30 days)...")
    portfolio_values, trades, final_value, test_df, step_log = walkforward_backtest(df, test_days=TEST_DAYS)
    print(f"\nFinal portfolio value: ${final_value:,.2f}")
    print(f"Total trades: {len(trades)}")
    for t in trades:
        print(f"TRADE | {t['action']} | {t['date']} | Price: {t['price']:.2f} | Qty: {t['qty']:.6f} | Fee: {t['fee']:.2f} | Portfolio: {t['portfolio']:.2f}")
    # Save trade log
    pd.DataFrame(trades).to_csv('spy_advanced_walkforward_trades.csv', index=False)
    pd.DataFrame(step_log).to_csv('spy_advanced_walkforward_steps.csv', index=False)
    print('Saved trade log as spy_advanced_walkforward_trades.csv and step log as spy_advanced_walkforward_steps.csv')
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(test_df['timestamp'][:len(portfolio_values)], portfolio_values)
    plt.title('Portfolio Value Over Test Month (SPY, Advanced)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.tight_layout()
    plt.savefig('spy_advanced_walkforward_portfolio.png')
    print('Saved plot as spy_advanced_walkforward_portfolio.png')
    # Stats
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    print(f"Sharpe ratio (test month): {sharpe:.2f}")
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH * 100
    print(f"Total return (test month): {total_return:.2f}%")

if __name__ == '__main__':
    main() 