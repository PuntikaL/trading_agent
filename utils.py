import pandas as pd
import requests

class TradingAgent:
    """Base class for trading agents."""
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0  # Number of units of the asset
        self.history = []  # To store trade history

    def generate_signals(self, data):
        """Generate trading signals. To be implemented by subclasses."""
        raise NotImplementedError

    def trade(self, price, signal):
        """Execute trades based on the signal."""
        if signal == 1:  # Buy
            self.position += self.cash / price
            self.cash = 0
            self.history.append(f"Buy at {price}")
        elif signal == -1:  # Sell
            self.cash += self.position * price
            self.position = 0
            self.history.append(f"Sell at {price}")
        # Hold: Do nothing
        self.history.append(f"Hold at {price}")

    def get_portfolio_value(self, price):
        """Calculate total portfolio value."""
        return self.cash + (self.position * price)

def fetch_historical_data(symbol, interval, limit=1000):
    """Fetch historical data from Binance API."""
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def calculate_performance_metrics(initial_cash, final_value):
    """Calculate performance metrics."""
    total_return = (final_value - initial_cash) / initial_cash
    return {"Total Return": total_return}