# BitcoinBot

## Introduction
This project is an algorithmic trading bot for cryptocurrencies. It currently uses the public CoinGecko API to fetch historical price data for Ethereum (ETH/USD) and applies a machine learning model (Linear Regression) to predict short-term price movements. **Trading logic is not implemented yet.**

## Features
- Retrieves crypto market data from CoinGecko (no API key required)
- Stores historical data in CSV files for each asset
- Trains a Linear Regression model on the most recent 100 minutes of data
- Prints current and predicted next price every minute
- Runs continuously, updating every minute

## Project Structure
- `main.py`: Main data retrieval and ML script
- `requirements.txt`: Python dependencies
- `project3/data/`: Directory where historical data CSVs are stored

## Setup
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the bot:**
   ```bash
   python main.py
   ```

## How It Works
- Fetches the last 100 minutes of ETH/USD price data from CoinGecko
- Saves the data to `project3/data/ethereum_usd.csv`
- Trains a Linear Regression model to predict the next minute's price
- Prints the current and predicted next price every iteration
- Handles errors gracefully and continues running

## Notes
- This bot is for educational purposes and uses only public market data.
- No trading or account integration is implemented yet.

## License
MIT 