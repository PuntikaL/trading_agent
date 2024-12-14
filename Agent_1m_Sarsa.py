from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class SARSAAgent:
    """SARSA-based Reinforcement Learning Agent"""
    def __init__(self, initial_cash=100000, learning_rate=0.01, discount_factor=0.99, epsilon=0.1):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # State-action mapping
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = []

    def train_model(self, training_data, feature_columns, target_column):
        """Preprocess training data and initialize Q-table."""
        self.feature_columns = feature_columns
        features = training_data[feature_columns]
        target = training_data[target_column]

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Initialize Q-table
        self.q_table = {tuple(row): {action: 0 for action in [-1, 0, 1]} for row in features_scaled}
        self.trained = True

    def generate_signals(self, row):
        """Generate trading signals using epsilon-greedy policy."""
        if not self.trained:
            raise ValueError("Model has not been trained. Call train_model() before using the agent.")

        state = tuple(row[self.feature_columns].values)
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in [-1, 0, 1]}  # Initialize state

        if np.random.rand() < self.epsilon:  # Explore
            return np.random.choice([-1, 0, 1])
        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

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
