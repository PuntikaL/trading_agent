from Agent_1m import Agent_1m
from Agent_1h import Agent_1h
from Agent_RL_1h import SARSA_Agent_1h
from Agent_1d import Agent_1d
from Agent_pca_1d import PCAAgent
from Agent_RL_1d import SARSA_Agent_1d
from Agent_VPA_1d import VanillaPolicyGradientAgent
from Agent_1m_DT import DecisionTreeAgent
from Agent_1m_Sarsa import SARSAAgent
from utils import *

def calculate_max_loss(initial_cash, portfolio_values):
    max_value = initial_cash
    max_loss = 0
    for value in portfolio_values:
        max_loss = max(max_loss, max_value - value)
        max_value = max(max_value, value)
    return max_loss

def backtest(agent, data):
    """Backtest a trading agent."""
    total_loss = 0
    portfolio_values = [agent.initial_cash]  # Initialize portfolio with starting cash
    prev_portfolio_value = agent.initial_cash

    for _, row in data.iterrows():
        price = row['close']  # Use the closing price for trades
        signal = agent.generate_signals(row)
        agent.trade(price, signal)

        portfolio_value = agent.get_portfolio_value(price)
        portfolio_values.append(portfolio_value)

        if portfolio_value < prev_portfolio_value:
            total_loss += prev_portfolio_value - portfolio_value

        prev_portfolio_value = portfolio_value

    max_loss = calculate_max_loss(agent.initial_cash, portfolio_values)
    return total_loss, max_loss, portfolio_values[-1]

# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# Initialize agents
agent_1m = Agent_1m()
agent_1m_DT = DecisionTreeAgent()
agent_1m_SARSA = SARSAAgent()
agent_1h = Agent_1h()
agent_1d = Agent_1d()
agent_1d_pca = PCAAgent()
agent_1d_RL = SARSA_Agent_1d()
agent_1h_RL = SARSA_Agent_1h()
agent_1d_VPA = VanillaPolicyGradientAgent()

# Preprocess data for Agent_1m
df_1m['target'] = df_1m['close'].shift(-1)  # Predict next close price
feature_columns = ['open', 'high', 'low', 'close', 'volume']
df_1m.dropna(inplace=True)  # Drop rows with NaN values

# Debugging data
print("Features shape (1m):", df_1m[feature_columns].shape)
print("Target shape (1m):", df_1m['target'].shape)

# Train agents
agent_1m.train_model(df_1m, feature_columns, target_column='target')
agent_1m_DT.train_model(df_1m, feature_columns, target_column='target')
agent_1m_SARSA.train_model(df_1m, feature_columns, target_column='target')

# Preprocess data for Agent_1h
df_1h['target'] = df_1h['close'].shift(-1)  # Predict next close price
df_1h.dropna(inplace=True)
agent_1h_RL.train_model(df_1h, feature_columns, target_column='target')

# Preprocess data for Agent_1d
df_1d['target'] = df_1d['close'].shift(-1)  # Predict next close price
df_1d.dropna(inplace=True)
agent_1d_pca.train(df_1d)
agent_1d_RL.train_model(df_1d, feature_columns, target_column='target')
agent_1d_VPA.train()
agent_1d_VPA.scaler.fit(df_1d[agent_1d_VPA.feature_columns].values)

import pandas_ta as ta

# Add SMA and RSI indicators to df_1d
df_1d['SMA_short'] = df_1d['close'].rolling(window=10).mean()  # Short Moving Average (10-day)
df_1d['SMA_long'] = df_1d['close'].rolling(window=50).mean()  # Long Moving Average (50-day)
df_1d['RSI'] = ta.rsi(df_1d['close'], length=14)  # RSI with a 14-day period

# Drop rows with NaN values caused by rolling calculations
df_1d.dropna(inplace=True)

# Debugging output to ensure indicators exist
print(df_1d[['SMA_short', 'SMA_long', 'RSI']].head())


# Backtest each agent
total_loss_1m, max_loss_1m, portfolio_value_1m = backtest(agent_1m, df_1m)
total_loss_1m_DT, max_loss_1m_DT, portfolio_value_1m_DT = backtest(agent_1m_DT, df_1m)
total_loss_1m_SARSA, max_loss_1m_SARSA, portfolio_value_1m_SARSA = backtest(agent_1m_SARSA, df_1m)
total_loss_1h, max_loss_1h, portfolio_value_1h = backtest(agent_1h, df_1h)
total_loss_1d, max_loss_1d, portfolio_value_1d = backtest(agent_1d, df_1d)
total_loss_1h_RL, max_loss_1h_RL, portfolio_value_1h_RL = backtest(agent_1h_RL, df_1h)
total_loss_1d_pca, max_loss_1d_pca, portfolio_value_1d_pca = backtest(agent_1d_pca, df_1d)
total_loss_1d_RL, max_loss_1d_RL, portfolio_value_1d_RL = backtest(agent_1d_RL, df_1d)
total_loss_1d_VPA, max_loss_1d_VPA, portfolio_value_1d_VPA = backtest(agent_1d_VPA, df_1d)

# Print results
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m:.2f}")
print(f"Portfolio Value for 1m Decision Tree Interval: {portfolio_value_1m_DT:.2f}")
print(f"Portfolio Value for 1m SARSA Interval: {portfolio_value_1m_SARSA:.2f}")
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h:.2f}")
print(f"Portfolio Value for 1h RL Interval: {portfolio_value_1h_RL:.2f}")
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d:.2f}")
print(f"Portfolio Value for 1d PCA Interval: {portfolio_value_1d_pca:.2f}")
print(f"Portfolio Value for 1d RL Interval: {portfolio_value_1d_RL:.2f}")
print(f"Portfolio Value for 1d VPA Interval: {portfolio_value_1d_VPA:.2f}")
