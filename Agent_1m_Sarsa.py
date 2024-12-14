import numpy as np
from utils import TradingAgent

class SARSAAgent(TradingAgent):
    """SARSA-based Reinforcement Learning Agent for Algorithmic Trading"""
    
    def __init__(self, initial_cash=100000, learning_rate=0.01, discount_factor=0.99, epsilon=0.1):
        super().__init__(initial_cash)
        self.q_table = {}  # State-action mapping
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None
    
    def get_state(self, row):
        """Here we extract the state representation from the current row."""
        return (row['RSI'], row['BB_Lower'], row['BB_Upper'])  # Example state: RSI and Bollinger Bands
    
    def select_action(self, state):
        """Here we use the Epsilon-greedy policy for action selection."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in [-1, 0, 1]}  # Initialize actions

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])  # Explore: random action
        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit: best action
    
    def update_q_table(self, reward, new_state):
        """Here we update the Q-table using the SARSA algorithm."""
        if self.last_state is not None:
            old_value = self.q_table[self.last_state][self.last_action]
            next_action = self.select_action(new_state)
            next_value = self.q_table[new_state][next_action]
            self.q_table[self.last_state][self.last_action] = old_value + self.learning_rate * (
                reward + self.discount_factor * next_value - old_value)
    
    def generate_signals(self, row):
        """Here we generate the trading signals."""
        state = self.get_state(row)
        action = self.select_action(state)
        
        # Reward and Q-table update
        if self.last_state is not None and self.last_action is not None:
            reward = self.get_portfolio_value(row['close']) - self.initial_cash
            self.update_q_table(reward, state)
        
        # Update last state and action
        self.last_state = state
        self.last_action = action
        return action
