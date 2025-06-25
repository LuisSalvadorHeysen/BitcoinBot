# StockBot: Educational Algorithmic Trading Bot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-orange.svg)]()

A comprehensive, educational algorithmic trading bot for SPY (S&P 500 ETF) that demonstrates machine learning, backtesting, and explainable AI (XAI) in stock trading. This project is designed for students, educators, and anyone interested in learning about algorithmic trading and data science in finance.

---

## 🎯 Overview

This project walks you through building a realistic stock trading bot using:
- **Machine Learning**: Random Forest and Linear Regression models
- **Technical Analysis**: 18+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Historical Data**: Model training and backtesting on years of SPY data
- **Realistic Backtesting**: Transaction fees, slippage, and portfolio management
- **Explainable AI**: SHAP analysis for model interpretability
- **Performance Analytics**: Sharpe ratio, drawdown, and trade logging

---

## 🧑‍💻 How It Works

1. **Fetch Historical Data**: Download years of SPY price data from Yahoo Finance.
2. **Feature Engineering**: Calculate technical indicators and price-based features.
3. **Model Training**: Train a machine learning model (Random Forest or Linear Regression) on historical data.
4. **Backtesting**: Simulate trades using the trained model, including fees and slippage.
5. **Performance Analysis**: Visualize results, compute Sharpe ratio, drawdown, and analyze feature importance with SHAP.
6. **Live Prediction (Optional)**: Use the trained model to make live predictions on the latest SPY data.

---

## ✨ Features

- **Multiple ML Models**: Linear Regression, Random Forest, and more
- **18+ Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, etc.
- **Rolling Window Training**: Simulates real-time model updates
- **Realistic Simulation**: Includes transaction fees, slippage, and position sizing
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, trade logs
- **Explainable AI**: SHAP plots and feature importance
- **Educational Code**: Modular, well-documented, and easy to follow

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Typical Workflow
```bash
# 1. Fetch and save historical SPY data (automatically done in main.py)
python main.py

# 2. Run backtesting simulation on historical data
python backtest.py

# 3. Run advanced backtesting with technical indicators and SHAP analysis
python advanced_backtest.py
```

---

## 📁 Project Structure

```
BitcoinBot/
├── main.py                      # Fetches historical data, trains model, live prediction
├── backtest.py                  # Backtesting simulation on historical data
├── advanced_backtest.py         # Advanced ML backtesting + SHAP explainability
├── requirements.txt             # Python dependencies
├── project3/
│   └── data/
│       ├── spy_usd.csv          # Recent SPY data (live)
│       ├── spy_historical.csv   # Historical SPY data (for training/backtest)
├── *.png                        # Generated plots and visualizations
├── *.csv                        # Trade logs and performance data
├── LICENSE, CONTRIBUTING.md, .gitignore
```

---

## 📈 Results & Performance

### Sample Output
```
--- Performance Metrics ---
Final Portfolio Value: $512,345.67
Total Return: 2.47%
Annualized Sharpe Ratio: 1.85
Maximum Drawdown: -1.23%
Total Trades: 47
```

### Sample Visualizations
- `portfolio_value_advanced.png`: Portfolio value over time
- `feature_importance.png`: Feature importance (Random Forest)
- `shap_summary.png`: SHAP summary plot (explainable AI)

---

## 🛠️ Configuration

### Trading Parameters (edit in the scripts)
```python
FEE_RATE = 0.001        # 0.1% transaction fee
SLIPPAGE_RATE = 0.0005  # 0.05% price slippage
HOLD_THRESHOLD = 0.0005 # 0.05% minimum move to trade
INITIAL_CASH = 500000   # Starting capital
```

### Model Settings
```python
WINDOW = 60             # Training window size
FEATURE_COLUMNS = 18    # Number of technical indicators
MODEL_TYPE = "RandomForest"  # ML algorithm choice
```

---

## 📊 Technical Indicators

The bot uses 18+ technical indicators:
- **Moving Averages**: SMA (5, 20), EMA (12, 26)
- **Momentum**: RSI, MACD, MACD Signal, MACD Histogram
- **Volatility**: Bollinger Bands (Upper, Middle, Lower)
- **Price Action**: Price changes (1, 5, 10 periods)
- **Volume**: Volume ratios (when available)

---

## 🎓 Educational Value

This project demonstrates:
- **Data Science**: Feature engineering, model training, and validation
- **Financial Engineering**: Backtesting, risk management, and performance analysis
- **Software Engineering**: Clean code, modular design, and documentation
- **Machine Learning**: Model selection, hyperparameter tuning, and interpretability

---

## 🧹 Clean Up & Troubleshooting
- If you see errors about missing columns (e.g., 'timestamp'), ensure you are using the latest code and have run `main.py` at least once.
- If you want to start fresh, you can delete old CSVs in `project3/data/`.
- For Python version issues, use Python 3.8+.
- For plotting errors, ensure you have a working matplotlib installation (no GUI needed).

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork the repository
- Add new features or models
- Improve documentation
- Share your results and insights

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Stock trading involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 📞 Contact

- **GitHub**: [@LuisSalvadorHeysen](https://github.com/LuisSalvadorHeysen)
- **Project**: [StockBot Repository](https://github.com/LuisSalvadorHeysen/BitcoinBot)

---

**Built with ❤️ for educational purposes** 