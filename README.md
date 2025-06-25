# BitcoinBot: Educational Algorithmic Trading Bot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Educational-orange.svg)]()

A comprehensive, educational algorithmic trading bot for Bitcoin that demonstrates machine learning, backtesting, and explainable AI (XAI) in cryptocurrency trading.

## 🎯 Overview

This project showcases the development of a realistic Bitcoin trading bot using:
- **Machine Learning**: Random Forest and Linear Regression models
- **Technical Analysis**: 18+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Realistic Backtesting**: Transaction fees, slippage, and portfolio management
- **Explainable AI**: SHAP analysis for model interpretability
- **Performance Analytics**: Sharpe ratio, drawdown, and trade logging

## ✨ Features

### 🤖 Machine Learning
- **Multiple Models**: Linear Regression, Random Forest, and Gradient Boosting
- **Feature Engineering**: 18+ technical indicators and price-based features
- **Rolling Window Training**: Simulates real-time model updates
- **Prediction Confidence**: Model explains its trading decisions

### 📊 Data & Analysis
- **Real-time Data**: CoinGecko API integration for live Bitcoin prices
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Performance Metrics**: Sharpe ratio, maximum drawdown, total returns
- **Trade Logging**: Detailed CSV logs of all transactions

### 🎓 Educational Components
- **Jupyter Notebook**: Step-by-step tutorial with explanations
- **Modular Code**: Clean, well-documented functions
- **Visualizations**: Portfolio performance and feature importance plots
- **Best Practices**: Realistic trading simulation with fees and slippage

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib requests shap
```

### Basic Usage
```bash
# Fetch latest Bitcoin data
python main.py

# Run backtesting simulation
python backtest.py

# Run advanced backtesting with technical indicators
python advanced_backtest.py
```

### Jupyter Notebook
```bash
# Open the educational notebook
jupyter notebook BitcoinBot_Educational.ipynb
```

## 📁 Project Structure

```
BitcoinBot/
├── main.py                      # Data fetching and live prediction
├── backtest.py                  # Basic backtesting simulation
├── advanced_backtest.py         # Advanced ML backtesting
├── BitcoinBot_Educational.ipynb # Educational Jupyter notebook
├── requirements.txt             # Python dependencies
├── project3/
│   └── data/                   # Historical data storage
│       └── bitcoin_usd.csv     # Bitcoin price data
├── *.png                       # Generated plots and visualizations
└── *.csv                       # Trade logs and performance data
```

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

### Key Features
- **Realistic Trading**: Includes 0.1% transaction fees and 0.05% slippage
- **Risk Management**: Configurable hold thresholds and position sizing
- **Performance Tracking**: Real-time portfolio value and trade logging
- **Model Explainability**: SHAP analysis shows which features drive decisions

## 🛠️ Configuration

### Trading Parameters
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

## 📊 Technical Indicators

The bot uses 18+ technical indicators:
- **Moving Averages**: SMA (5, 20), EMA (12, 26)
- **Momentum**: RSI, MACD, MACD Signal, MACD Histogram
- **Volatility**: Bollinger Bands (Upper, Middle, Lower)
- **Price Action**: Price changes (1, 5, 10 periods)
- **Volume**: Volume ratios (when available)

## 🎓 Educational Value

This project demonstrates:
- **Data Science**: Feature engineering, model training, and validation
- **Financial Engineering**: Backtesting, risk management, and performance analysis
- **Software Engineering**: Clean code, modular design, and documentation
- **Machine Learning**: Model selection, hyperparameter tuning, and interpretability

## 🔮 Future Enhancements

- [ ] **Live Trading**: Integration with crypto exchanges
- [ ] **Advanced Models**: LSTM, Transformer models for time series
- [ ] **Multi-Asset**: Support for Ethereum, other cryptocurrencies
- [ ] **Web Dashboard**: Real-time monitoring and control interface
- [ ] **Risk Management**: Stop-loss, take-profit, and position sizing
- [ ] **Sentiment Analysis**: News and social media integration

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork the repository
- Add new features or models
- Improve documentation
- Share your results and insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Cryptocurrency trading involves substantial risk of loss. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 📞 Contact

- **GitHub**: [@LuisSalvadorHeysen](https://github.com/LuisSalvadorHeysen)
- **Project**: [BitcoinBot Repository](https://github.com/LuisSalvadorHeysen/BitcoinBot)

---

**Built with ❤️ for educational purposes** 