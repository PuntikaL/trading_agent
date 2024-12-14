from Agent_1m import Agent_1m


from Agent_1h import Agent_1h
from Agent_RL_1h import SARSA_Agent_1h

from Agent_1d import Agent_1d
from Agent_pca_1d import PCAAgent
from Agent_RL_1d import SARSA_Agent_1d
from Agent_VPA_1d import VanillaPolicyGradientAgent

from utils import *

'''
def backtest(agent, data):
    """Backtest a trading agent."""
    for timestamp, row in data.iterrows():
        price = row['close']  # Use the closing price for trades
        signal = agent.generate_signals(row)
        agent.trade(price, signal)
    return agent.get_portfolio_value(data['close'].iloc[-1])
'''

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
    portfolio_values = []
    portfolio_values = [agent.initial_cash]  # Initialize portfolio with starting cash
    prev_portfolio_value = agent.initial_cash
    for timestamp, row in data.iterrows():
        price = row['close']  # Use the closing price for trades
        signal = agent.generate_signals(row)
        agent.trade(price, signal)
        
        # After each trade, update the portfolio value
        portfolio_value = agent.get_portfolio_value(price)
        portfolio_values.append(portfolio_value)

        current_portfolio_value = agent.get_portfolio_value(price)
        portfolio_values.append(current_portfolio_value)

        # If there is a loss (portfolio value decreases), accumulate it
        if current_portfolio_value < prev_portfolio_value:
            total_loss += prev_portfolio_value - current_portfolio_value

        # Update the previous portfolio value for the next iteration
        prev_portfolio_value = current_portfolio_value
    
    # After the backtest, calculate the max loss during the trading period
    max_loss = calculate_max_loss(agent.initial_cash, portfolio_values)
    return total_loss, max_loss, portfolio_values[-1]

# Fetch data for different intervals
df_1m = fetch_historical_data('BTCUSDT', '1m')
df_1h = fetch_historical_data('BTCUSDT', '1h')
df_1d = fetch_historical_data('BTCUSDT', '1d')

# Initialize agents
agent_1m = Agent_1m()
agent_1h = Agent_1h()
agent_1d = Agent_1d()
agent_1d_pca = PCAAgent()
agent_1d_pca.train(df_1d)
#agent_1d_stoch_fib = Agent_1d_stoch_fib()
agent_1d_RL = SARSA_Agent_1d()
agent_1h_RL = SARSA_Agent_1h()
agent_1d_VPA = VanillaPolicyGradientAgent()


# Prepare data for Agent_1m (Linear Regression)
df_1m['target'] = df_1m['close'].shift(-1)  # Predict next close price
feature_columns = ['open', 'high', 'low', 'close', 'volume']  # Features for training
df_1m.dropna(inplace=True)  # Drop rows with NaN values due to target shift

# Training the Agent_1m (Linear Regression)
agent_1m.train_model(df_1m, feature_columns, target_column='target')
agent_1d.add_technical_indicators(df_1d)
#agent_1d_stoch_fib.add_technical_indicators(df_1d)

# Prepare data for Agent_1d (Linear Regression)
df_1h['target'] = df_1h['close'].shift(-1)  # Predict next close price
feature_columns = ['open', 'high', 'low', 'close', 'volume']  # Features for training
df_1h.dropna(inplace=True)  # Drop rows with NaN values due to target shift
agent_1h_RL.train_model(df_1h, feature_columns, target_column='target')

# Prepare data for Agent_1d (Linear Regression)
df_1d['target'] = df_1d['close'].shift(-1)  # Predict next close price
feature_columns = ['open', 'high', 'low', 'close', 'volume']  # Features for training
df_1d.dropna(inplace=True)  # Drop rows with NaN values due to target shift


agent_1d_RL.train_model(df_1d, feature_columns, target_column='target')

agent_1d_VPA.train()
agent_1d_VPA.scaler.fit(df_1d[agent_1d_VPA.feature_columns].values)

# Backtest each agent
total_loss_1m, max_loss_1m, portfolio_value_1m= backtest(agent_1m, df_1m)
total_loss_1h, max_loss_1h, portfolio_value_1h = backtest(agent_1h, df_1h)
total_loss_1d, max_loss_1d, portfolio_value_1d = backtest(agent_1d, df_1d)
total_loss_1h_RL, max_loss_1h_RL, portfolio_value_1h_RL = backtest(agent_1h_RL, df_1h)
total_loss_1d_pca, max_loss_1d_pca, portfolio_value_1d_pca = backtest(agent_1d_pca, df_1d)
#total_loss_1d_stoch_fib, max_loss_1d_stoch_fib, portfolio_value_1d_stoch_fib = backtest(agent_1d_stoch_fib, df_1d)
total_loss_1d_RL, max_loss_1d_RL, portfolio_value_1d_RL = backtest(agent_1d_RL, df_1d)
total_loss_1d_VPA, max_loss_1d_VPA, portfolio_value_1d_VPA = backtest(agent_1d_VPA, df_1d)

# Print results
print(f"Portfolio Value for 1m Interval: {portfolio_value_1m:.2f}")
#1h
print(f"Portfolio Value for 1h Interval: {portfolio_value_1h:.2f}")
print(f"Portfolio Value for 1h RL Interval: {portfolio_value_1h_RL:.2f}")
#1d
print(f"Portfolio Value for 1d Interval: {portfolio_value_1d:.2f}")
print(f"Portfolio Value for 1d PCA Interval: {portfolio_value_1d_pca:.2f}")
#print(f"Portfolio Value for 1d Stochastic+Fibonacci Interval: {portfolio_value_1d_stoch_fib:.2f}")
print(f"Portfolio Value for 1d RL Interval: {portfolio_value_1d_RL:.2f}")
print(f"Portfolio Value for 1d VPA Interval: {portfolio_value_1d_VPA:.2f}")

# Print maximum loss
print(f"Maximum Loss for 1m Interval: {max_loss_1m:.2f}")
print(f"Maximum Loss for 1h Interval: {max_loss_1h:.2f}")
print(f"Maximum Loss for 1h RL Interval: {max_loss_1h_RL:.2f}")
print(f"Maximum Loss for 1d Interval: {max_loss_1d:.2f}")
print(f"Maximum Loss for 1d PCA Interval: {max_loss_1d_pca:.2f}")
#print(f"Maximum Loss for 1d Stochastic+Fibonacci Interval: {max_loss_1d_stoch_fib:.2f}")
print(f"Maximum Loss for 1d RL Interval: {max_loss_1d_RL:.2f}")
print(f"Maximum Loss for 1d VPA Interval: {max_loss_1d_VPA:.2f}")

print(f"Total Loss for 1m Interval: {total_loss_1m:.2f}")
print(f"Total Loss for 1h Interval: {total_loss_1h:.2f}")
print(f"Total Loss for 1h RL Interval: {total_loss_1h_RL:.2f}")
print(f"Total Loss for 1d Interval: {total_loss_1d:.2f}")
print(f"Total Loss for 1d PCA Interval: {total_loss_1d_pca:.2f}")
#print(f"Total Loss for 1d Stochastic+Fibonacci Interval: {total_loss_1d_stoch_fib:.2f}")
print(f"Total Loss for 1d RL Interval: {total_loss_1d_RL:.2f}")
print(f"Total Loss for 1d VPA Interval: {total_loss_1d_VPA:.2f}")

df_1d['daily_returns'] = df_1d['close'].pct_change()
daily_returns = df_1d['daily_returns'].dropna().tolist()

# Risk-free rate
risk_free_rate = 0.001

#I didnt find the function for this so i added
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio.
    
    Parameters:
    - returns: A list or array of portfolio returns (daily, monthly, etc.).
    - risk_free_rate: The risk-free rate (e.g., the return of a risk-free asset, like a treasury bond).
    
    Returns:
    - Sharpe ratio: The risk-adjusted return.
    """
    # Convert returns to a numpy array for easy calculations
    returns = np.array(returns)
    
    # Calculate mean return and standard deviation of returns
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    
    # Calculate Sharpe ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    return sharpe_ratio
# Sharpe Ratio for candlestick-based method
daily_returns_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
print(f"Sharpe Ratio: {daily_returns_ratio:.2f}")

# Evaluate performance
performance_1m = calculate_performance_metrics(agent_1m.initial_cash, portfolio_value_1m)
performance_1h = calculate_performance_metrics(agent_1h.initial_cash, portfolio_value_1h)
performance_1h_RL = calculate_performance_metrics(agent_1h_RL.initial_cash, portfolio_value_1h_RL)
performance_1d = calculate_performance_metrics(agent_1d.initial_cash, portfolio_value_1d)
performance_1d_pca = calculate_performance_metrics(agent_1d_pca.initial_cash, portfolio_value_1d_pca)
#performance_1d_stoch_fib = calculate_performance_metrics(agent_1d_stoch_fib.initial_cash, portfolio_value_1d_stoch_fib)
performance_1d_RL = calculate_performance_metrics(agent_1d_RL.initial_cash, portfolio_value_1d_RL)
performance_1d_VPA = calculate_performance_metrics(agent_1d_VPA.initial_cash, portfolio_value_1d_VPA)

# Print performance metrics
print(f"Performance for 1m Interval: {performance_1m['Total Return']:.2f}")
print(f"Performance for 1h Interval: {performance_1h['Total Return']:.2f}")
print(f"Performance for 1h RL Interval: {performance_1h_RL['Total Return']:.2f}")
print(f"Performance for 1d Interval: {performance_1d['Total Return']:.2f}")
print(f"Performance for 1d PCA Interval: {performance_1d_pca['Total Return']:.2f}")
#print(f"Performance for 1d Stochastic+Fibonacci Interval: {performance_1d_stoch_fib['Total Return']:.2f}")
print(f"Performance for 1d RL Interval: {performance_1d_RL['Total Return']:.2f}")
print(f"Performance for 1d VPA Interval: {performance_1d_VPA['Total Return']:.2f}")

# Assuming these are the final portfolio values from the backtests
initial_cash_candlestick = agent_1d.initial_cash
final_value_candlestick = portfolio_value_1d

initial_cash_pca = agent_1d_pca.initial_cash
final_value_pca = portfolio_value_1d_pca