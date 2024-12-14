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
#agent_1d_VPA.train()
#agent_1d_VPA.scaler.fit(df_1d[agent_1d_VPA.feature_columns].values)

import pandas_ta as ta

# Add SMA and RSI indicators to df_1d
df_1d['SMA_short'] = df_1d['close'].rolling(window=10).mean()  # Short Moving Average (10-day)
df_1d['SMA_long'] = df_1d['close'].rolling(window=50).mean()  # Long Moving Average (50-day)
df_1d['RSI'] = ta.rsi(df_1d['close'], length=14)  # RSI with a 14-day period

# Drop rows with NaN values caused by rolling calculations
df_1d.dropna(inplace=True)

# Debugging output to ensure indicators exist
print(df_1d[['SMA_short', 'SMA_long', 'RSI']].head())


df_1d['daily_returns'] = df_1d['close'].pct_change()
daily_returns = df_1d['daily_returns'].dropna().tolist()

# Risk-free rate
risk_free_rate = 0.001
daily_returns_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
print(f"Sharpe Ratio: {daily_returns_ratio:.2f}")

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

print(f"Maximum Loss for 1m Interval: {max_loss_1m:.2f}")
print(f"Maximum Loss for 1m Decision Tree Interval: {max_loss_1m_DT:.2f}")
print(f"Maximum Loss for 1m SARSA Interval: {max_loss_1m_SARSA:.2f}")
print(f"Maximum Loss for 1h Interval: {max_loss_1h:.2f}")
print(f"Maximum Loss for 1h RL Interval: {max_loss_1h_RL:.2f}")
print(f"Maximum Loss for 1d Interval: {max_loss_1d:.2f}")
print(f"Maximum Loss for 1d PCA Interval: {max_loss_1d_pca:.2f}")
print(f"Maximum Loss for RL Interval: {max_loss_1d_RL:.2f}")
print(f"Maximum Loss for VPA Interval: {max_loss_1d_VPA:.2f}")


print(f"Total Loss for 1m Interval: {total_loss_1m:.2f}")
print(f"Total Loss for 1m Decision Tree Interval: {total_loss_1m_DT:.2f}")
print(f"Total Loss for 1m SARSA Interval: {total_loss_1m_SARSA:.2f}")
print(f"Total Loss for 1h Interval: {total_loss_1h:.2f}")
print(f"Total Loss for 1h RL Interval: {total_loss_1h_RL:.2f}")
print(f"Total Loss for 1d Interval: {total_loss_1d:.2f}")
print(f"Total Loss for 1d PCA Interval: {total_loss_1d_pca:.2f}")
print(f"Total Loss for RL Interval: {total_loss_1d_RL:.2f}")
print(f"Total Loss for VPA Interval: {total_loss_1d_VPA:.2f}")


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

performance_1m = calculate_performance_metrics(agent_1m.initial_cash, portfolio_value_1m)
performance_1m_DT = calculate_performance_metrics(agent_1m_DT.initial_cash, portfolio_value_1m_DT)
performance_1m_SARSA = calculate_performance_metrics(agent_1m_SARSA.initial_cash, portfolio_value_1m_SARSA)
performance_1h = calculate_performance_metrics(agent_1h.initial_cash, portfolio_value_1h)
performance_1h_RL = calculate_performance_metrics(agent_1d.initial_cash, portfolio_value_1h_RL)
performance_1d = calculate_performance_metrics(agent_1h_RL.initial_cash, portfolio_value_1d)
performance_1d_pca = calculate_performance_metrics(agent_1d_pca.initial_cash, portfolio_value_1d_pca)
performance_1d_RL = calculate_performance_metrics(agent_1d_RL.initial_cash, portfolio_value_1d_RL)
performance_1d_VPA = calculate_performance_metrics(agent_1d_VPA.initial_cash, portfolio_value_1d_VPA)

# Print performance metrics
print(f"Performance for 1m Interval: {performance_1m['Total Return']:.2f}")
print(f"Performance for 1m Decision Tree Interval: {performance_1m_DT['Total Return']:.2f}")
print(f"Performance for 1m SARSA Interval: {performance_1m_SARSA['Total Return']:.2f}")
print(f"Performance for 1h Interval: {performance_1h['Total Return']:.2f}")
print(f"Performance for 1h RL Interval: {performance_1h_RL['Total Return']:.2f}")
print(f"Performance for 1d Interval: {performance_1d['Total Return']:.2f}")
print(f"Performance for 1d PCA Interval: {performance_1d_pca['Total Return']:.2f}")
print(f"Performance for 1d RL Interval: {performance_1d_RL['Total Return']:.2f}")
print(f"Performance for 1d VPA Interval: {performance_1d_VPA['Total Return']:.2f}")