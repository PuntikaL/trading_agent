import pandas as pd
import requests
import pandas_ta as ta
from tqdm import tqdm
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Use a different renderer if you want to avoid opening a browser
pio.renderers.default = "notebook"
tqdm.pandas()

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


def total_signal(df, current_candle):
    current_pos = df.index.get_loc(current_candle)
    c0 = df['open'].iloc[current_pos] > df['close'].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df['high'].iloc[current_pos] > df['high'].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df['low'].iloc[current_pos] < df['low'].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df['close'].iloc[current_pos] < df['low'].iloc[current_pos - 1]

    if c0 and c1 and c2 and c3:
        return 2  # Signal for entering a Long trade at the open of the next bar
    
    c0 = df['open'].iloc[current_pos] < df['close'].iloc[current_pos]
    # Condition 1: The high is greater than the high of the previous day
    c1 = df['low'].iloc[current_pos] < df['low'].iloc[current_pos - 1]
    # Condition 2: The low is less than the low of the previous day
    c2 = df['high'].iloc[current_pos] > df['high'].iloc[current_pos - 1]
    # Condition 3: The close of the Outside Bar is less than the low of the previous day
    c3 = df['close'].iloc[current_pos] > df['high'].iloc[current_pos - 1]
    
    if c0 and c1 and c2 and c3:
        return 1

    return 0

def add_total_signal(df):
    df['TotalSignal'] = df.progress_apply(lambda row: total_signal(df, row.name), axis=1)#.shift(1)
    return df

def add_pointpos_column(df, signal_column):
    """
    Adds a 'pointpos' column to the DataFrame to indicate the position of support and resistance points.
    
    Parameters:
    df (DataFrame): DataFrame containing the stock data with the specified SR column, 'Low', and 'High' columns.
    sr_column (str): The name of the column to consider for the SR (support/resistance) points.
    
    Returns:
    DataFrame: The original DataFrame with an additional 'pointpos' column.
    """
    def pointpos(row):
        if row[signal_column] == 2:
            return row['low'] - 1e-4
        elif row[signal_column] == 1:
            return row['high'] + 1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    return df

def plot_candlestick_with_signals(df, start_index, num_rows):
    #pio.renderers.default = "notebook"
    #tqdm.pandas()
    df_subset = df[start_index:start_index + num_rows]

    # Create a subplot figure with 2 rows and 1 column
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # Share x-axis between rows
        vertical_spacing=0.1,  # Adjust spacing
        subplot_titles=("Candlestick Chart with Bollinger Bands", "RSI")
    )

    # Add Candlestick Chart (Row 1, Column 1)
    fig.add_trace(
        go.Candlestick(
            x=df_subset.index,
            open=df_subset['open'],
            high=df_subset['high'],
            low=df_subset['low'],
            close=df_subset['close'],
            name="Candlestick"
        ),
        row=1, col=1
    )

    # Add Bollinger Bands to Candlestick Chart (Row 1, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df_subset.index,
            y=df_subset['BB_Upper'],
            mode="lines",
            name="BB Upper",
            line=dict(color="blue", width=1)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_subset.index,
            y=df_subset['BB_Lower'],
            mode="lines",
            name="BB Lower",
            line=dict(color="blue", width=1)
        ),
        row=1, col=1
    )

    # Add RSI to the second row (Row 2, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df_subset.index,
            y=df_subset['RSI'],
            mode="lines",
            name="RSI",
            line=dict(color="orange", width=1)
        ),
        row=2, col=1
    )

    # Update Layout
    fig.update_layout(
        title_text="Candlestick Chart with Bollinger Bands and RSI",
        width=1200,
        height=800,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="white"
            ),
            bgcolor="black",
            bordercolor="gray",
            borderwidth=2
        )
    )

    fig.print_grid()
    fig.write_html("output.html")
    print("Chart saved to output.html")


def add_rsi(df, period=14):
    """
    Adds RSI to the DataFrame.
    """
    df['RSI'] = ta.rsi(df['close'], length=period)
    return df

def add_moving_averages(df, short_window=10, long_window=50):
    """
    Adds short and long moving averages to the DataFrame.
    """
    df['MA_Short'] = df['close'].rolling(window=short_window).mean()
    df['MA_Long'] = df['close'].rolling(window=long_window).mean()
    return df

def add_bollinger_bands(df, period=20, std_dev=2):
    """
    Adds Bollinger Bands to the DataFrame.
    """
    df['BB_Middle'] = df['close'].rolling(window=period).mean()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * df['close'].rolling(window=period).std())
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * df['close'].rolling(window=period).std())
    return df

def enhanced_signal(df, current_candle):
    """
    Combines candlestick signals with RSI and moving average conditions.
    """
    current_pos = df.index.get_loc(current_candle)
    base_signal = total_signal(df, current_candle)
    
    # Get additional conditions
    rsi = df['RSI'].iloc[current_pos]
    ma_short = df['MA_Short'].iloc[current_pos]
    ma_long = df['MA_Long'].iloc[current_pos]
    price = df['close'].iloc[current_pos]
    bb_lower = df['BB_Lower'].iloc[current_pos]
    bb_upper = df['BB_Upper'].iloc[current_pos]
    
    # Adjust signal with RSI
    if base_signal == 2 and rsi < 30:  # Buy signal
        return 2
    elif base_signal == 1 and rsi > 70:  # Sell signal
        return 1
    
    # Adjust signal with Moving Averages
    if base_signal == 2 and ma_short > ma_long:  # Confirm uptrend
        return 2
    elif base_signal == 1 and ma_short < ma_long:  # Confirm downtrend
        return 1

    # Adjust signal with Bollinger Bands
    if base_signal == 2 and price < bb_lower:  # Oversold
        return 2
    elif base_signal == 1 and price > bb_upper:  # Overbought
        return 1

    return 0  # No signal

def add_enhanced_signal(df):
    """
    Adds an enhanced signal column based on candlestick patterns and technical indicators.
    """
    df['EnhancedSignal'] = df.progress_apply(lambda row: enhanced_signal(df, row.name), axis=1)
    return df
