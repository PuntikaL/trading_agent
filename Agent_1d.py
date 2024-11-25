from utils import *

class Agent_1d(TradingAgent):
    """Agent that trades on daily intervals using various technical indicators."""
    def __init__(self, short_window=20, long_window=50, initial_cash=100000):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window

    def preprocess_data(self, df_1d):
        """
        Preprocess the input data by adding indicators and signals.
        """
        df = df_1d[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        # Compute the rolling averages and other indicators before backtesting


        df = add_rsi(df)  # Add RSI indicator
        df = add_moving_averages(df)  # Add moving averages
        df = add_bollinger_bands(df)  # Add Bollinger Bands
        df = add_enhanced_signal(df)  # Add custom enhanced signals
        df = add_pointpos_column(df, signal_column='EnhancedSignal')  # Prepare for visualization
        return df

    def generate_signals(self, row):
        """Generate signals based on precomputed indicators (for each row)."""
        if row['SMA_short'] > row['SMA_long'] and row['RSI'] < 70:
            return 1  # Buy
        elif row['SMA_short'] < row['SMA_long'] and row['RSI'] > 30:
            return -1  # Sell
        return 0  # Hold


    def plot_data(self, df):
        """
        Plot the candlestick chart with signals.
        """
        plot_candlestick_with_signals(df, start_index=0, num_rows=100)
