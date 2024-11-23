from utils import TradingAgent

class Agent_1h(TradingAgent):
    """Trading Agent for 1-hour interval."""
    def generate_signals(self, data):
        """Generate random signals for 1-hour interval."""
        import random
        return random.choice([-1, 0, 1])  # Randomly decide to Buy, Sell, or Hold
    
'''
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


class DummyAgent(TradingAgent):
    """Dummy agent implementing a simple random strategy."""
    def generate_signals(self, data):
        """Generate random signals."""
        import random
        return random.choice([-1, 0, 1])  # Randomly decide to Buy, Sell, or Hold


# example of MovingAverageCrossoverAgent
class MovingAverageCrossoverAgent(TradingAgent):
    def __init__(self, short_window=50, long_window=200, initial_cash=100000):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data['SMA50'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA200'] = data['Close'].rolling(window=self.long_window).mean()
        if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1]:
            return 1  # Buy signal
        elif data['SMA50'].iloc[-1] < data['SMA200'].iloc[-1]:
            return -1  # Sell signal
        return 0  # Hold
'''