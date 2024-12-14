import pandas as pd

class Agent_1d:
    """Agent for daily interval trading based on SMA and RSI strategies."""
    def __init__(self, initial_cash=100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.trained = False
        self.feature_columns = []

    def add_technical_indicators(self, data):
        """Add SMA and RSI indicators to the data."""
        data['SMA_short'] = data['close'].rolling(window=10).mean()  # Short Moving Average
        data['SMA_long'] = data['close'].rolling(window=50).mean()  # Long Moving Average
        data['RSI'] = (100 - (100 / (1 + data['close'].pct_change().rolling(14).mean() / 
                                     data['close'].pct_change().rolling(14).std())))  # Simplified RSI
        data.dropna(inplace=True)  # Drop rows with NaN values
        return data

    def generate_signals(self, row):
        """Generate trading signals based on SMA and RSI."""
        if 'SMA_short' not in row or 'SMA_long' not in row or 'RSI' not in row:
            return 0  # Hold if indicators are missing
        if row['SMA_short'] > row['SMA_long'] and row['RSI'] < 70:
            return 1  # Buy
        elif row['SMA_short'] < row['SMA_long'] and row['RSI'] > 30:
            return -1  # Sell
        return 0  # Hold

    def trade(self, price, signal):
        """Execute trades based on the signal."""
        if signal == 1:  # Buy
            self.position += self.cash / price
            self.cash = 0
        elif signal == -1:  # Sell
            self.cash += self.position * price
            self.position = 0

    def get_portfolio_value(self, price):
        """Calculate total portfolio value."""
        return self.cash + (self.position * price)
