from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_1d import Agent_1d
from utils import *

def backtest(agent, data):
    """Backtest a trading agent."""
    for timestamp, row in data.iterrows():
        price = row['close']  # Use the closing price for trades
        signal = agent.generate_signals(row)
        agent.trade(price, signal)
    return agent.get_portfolio_value(data['close'].iloc[-1])

# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# Initialize agents
agent_1m = Agent_1m()
agent_1h = Agent_1h()
agent_1d = Agent_1d()

df_1d['SMA_short'] = df_1d['close'].rolling(window=agent_1d.short_window).mean()
df_1d['SMA_long'] = df_1d['close'].rolling(window=agent_1d.long_window).mean()

# Add RSI and other indicators (adjust these functions according to your setup)
df_1d = add_rsi(df_1d)  # Add RSI here
df_1d = add_moving_averages(df_1d)  # If you have other moving averages
df_1d = add_bollinger_bands(df_1d)  # If you are using Bollinger Bands
df_1d = add_enhanced_signal(df_1d) 

# Backtest each agent
portfolio_value_1m = backtest(agent_1m, df_1m)
portfolio_value_1h = backtest(agent_1h, df_1h)
portfolio_value_1d = backtest(agent_1d, df_1d)

# Print results
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d}")

# Evaluate performance
performance_1m = calculate_performance_metrics(agent_1m.initial_cash, portfolio_value_1m)
performance_1h = calculate_performance_metrics(agent_1h.initial_cash, portfolio_value_1h)
performance_1d = calculate_performance_metrics(agent_1d.initial_cash, portfolio_value_1d)

# Print performance metrics
print("Performance for 1m Interval:", performance_1m)
print("Performance for 1h Interval:", performance_1h)
print("Performance for 1d Interval:", performance_1d)
