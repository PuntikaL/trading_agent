from utils import *
import math

class Agent_1d(TradingAgent):

    def __init__(self, short_window=200, long_window=500, initial_cash=100000):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window

    def add_technical_indicators(self, df_1d):
        if len(df_1d) < 2000:
            self.short_window = math.floor(0.9 * len(df_1d))
            self.long_window = self.short_window + (self.short_window / 2)

        df_1d['SMA_short'] = df_1d['close'].rolling(window=self.short_window).mean()
        df_1d['SMA_long'] = df_1d['close'].rolling(window=self.long_window).mean()

        # Add RSI and other indicators
        df_1d = add_rsi(df_1d)
        df_1d = add_moving_averages(df_1d)
        df_1d = add_bollinger_bands(df_1d)
        df_1d = add_enhanced_signal(df_1d)
        return df_1d

    def preprocess_data(self, df_1d):
        df = df_1d[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)

        # Add technical indicators
        df = self.add_technical_indicators(df)
        df = add_pointpos_column(df, signal_column='EnhancedSignal')  # Prepare for visualization
        return df

    def generate_signals(self, row):
        if row['SMA_short'] > row['SMA_long'] and row['RSI'] < 70:
            return 1  # Buy
        elif row['SMA_short'] < row['SMA_long'] and row['RSI'] > 30:
            return -1  # Sell
        return 0  # Hold

    def plot_data(self, df):
        plot_candlestick_with_signals(df, start_index=0, num_rows=100)
