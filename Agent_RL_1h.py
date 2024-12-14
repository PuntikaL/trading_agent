import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils import TradingAgent  # Assuming TradingAgent is defined elsewhere

class SARSA_Agent_1h(TradingAgent):
    """1-day SARSA agent with linear regression for signal generation."""
    
    def __init__(self, initial_cash=100000, alpha=0.5, gamma=0.99, epsilon=0.5):
        super().__init__(initial_cash)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = []
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation tradeoff
        self.q_table = {}  # A dictionary to store Q-values (state-action pairs)
        
    def train_model(self, training_data, feature_columns, target_column):
        """Train the Linear Regression model."""
        self.feature_columns = feature_columns
        features = training_data[feature_columns]
        target = training_data[target_column]
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, target)
        self.trained = True

    def generate_signals(self, row):
        """Generate trading signals based on predictions from the trained model."""
        if not self.trained:
            raise ValueError("Model has not been trained. Call train_model() before using the agent.")
        
        # Convert the current row into a DataFrame with feature names
        current_features = pd.DataFrame([row[self.feature_columns]], columns=self.feature_columns)
        scaled_features = self.scaler.transform(current_features)
        predicted_price = self.model.predict(scaled_features)[0]
        current_price = row['close']
        
        if predicted_price > current_price:
            return 1  # Buy
        elif predicted_price < current_price:
            return -1  # Sell
        return 0  # Hold
    
    def epsilon_greedy(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, -1])  # Random action (exploration)
        else:
            return np.argmax(self.q_table.get(state, np.zeros(3)))  # Greedy action (exploitation)

    def update_q_table(self, state, action, reward, next_state, next_action):
        """Update Q-value based on SARSA update rule."""
        current_q_value = self.q_table.get(state, np.zeros(3))[action]
        next_q_value = self.q_table.get(next_state, np.zeros(3))[next_action]
        self.q_table[state][action] = current_q_value + self.alpha * (reward + self.gamma * next_q_value - current_q_value)

    def train(self, data, episodes=1000):
        """Train the agent using SARSA."""
        for episode in range(episodes):
            state = self.get_state(data.iloc[0])  # Initial state
            action = self.epsilon_greedy(state)  # Choose action using epsilon-greedy
            total_reward = 0
            for t in range(1, len(data)):
                row = data.iloc[t]
                next_state = self.get_state(row)  # Get next state
                reward = self.calculate_reward(state, action, row)  # Calculate reward for current action
                next_action = self.epsilon_greedy(next_state)  # Choose next action
                
                # Update Q-values using SARSA update rule
                self.update_q_table(state, action, reward, next_state, next_action)
                
                state = next_state  # Transition to next state
                action = next_action  # Take next action
                
                total_reward += reward  # Accumulate reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def get_state(self, row):
        """Generate the state representation from a row of data."""
        # Use a tuple of feature values as the state
        features = tuple(row[self.feature_columns])
        return features

    def calculate_reward(self, state, action, row):
        """Calculate the reward based on the action taken."""
        current_price = row['close']
        
        if action == 1:  # Buy
            reward = row['close'] - current_price  # Profit or loss from buying
        elif action == -1:  # Sell
            reward = current_price - row['close']  # Profit or loss from selling
        else:
            reward = 0  # Hold (no reward)
        
        return reward