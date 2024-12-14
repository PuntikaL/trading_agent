import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import StandardScaler

class VanillaPolicyGradientAgent:
    def __init__(self, state_size=5, action_size=3, learning_rate=0.01, gamma=0.99, initial_cash=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.initial_cash = initial_cash

        # Portfolio management
        self.cash = initial_cash
        self.holdings = 0

        # Initialize the policy network
        self.policy_model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []

        # Scaler for normalizing input data
        self.scaler = StandardScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']

    def build_model(self):
        """Build the policy network."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')  # Output action probabilities
        ])
        return model

    def get_action(self, state):
        """Select an action based on the policy network's probabilities."""
        state = state.reshape(1, -1)  # Ensure input shape
        action_probs = self.policy_model(state).numpy()[0]
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def store_trajectory(self, state, action, reward):
        """Store state, action, and reward for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calculate_discounted_rewards(self):
        """Compute the discounted rewards for each time step."""
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        # Normalize rewards for stability
        discounted_rewards = np.array(discounted_rewards)
        return (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-10)

    def train(self):
        """Train the policy network using stored trajectories."""
        if not self.states:
            print("No trajectories to train on. Skipping training.")
            return

        # Convert lists to arrays
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        discounted_rewards = self.calculate_discounted_rewards()

        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = self.policy_model(states)
            indices = np.arange(len(actions))
            selected_action_probs = tf.gather_nd(action_probs, tf.expand_dims(indices, axis=1), batch_dims=1)[np.arange(len(actions)), actions]

            # Compute loss (negative log probability scaled by discounted rewards)
            loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * discounted_rewards)

        # Apply gradients
        grads = tape.gradient(loss, self.policy_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        # Clear trajectory storage
        self.states, self.actions, self.rewards = [], [], []

    def get_state(self, row):
        """Generate the state representation from a row of data."""
        features = np.array(row[self.feature_columns])
        return self.scaler.transform(features.reshape(1, -1))

    def calculate_reward(self, action, row):
        """Calculate the reward based on the action taken."""
        if action == 1:  # Buy
            return row['close'] - row['open']  # Profit/loss from buying
        elif action == 2:  # Sell
            return row['open'] - row['close']  # Profit/loss from selling
        else:  # Hold
            return 0  # No reward for holding

    def trade(self, price, signal):
        """Execute a trade based on the signal."""
        if signal == 1:  # Buy
            if self.cash > 0:  # Ensure there's cash to buy
                self.holdings += self.cash / price
                self.cash = 0
        elif signal == 2:  # Sell
            if self.holdings > 0:  # Ensure there are holdings to sell
                self.cash += self.holdings * price
                self.holdings = 0
        # Signal 0 is hold, no action required

    def get_portfolio_value(self, price):
        """Calculate the total portfolio value."""
        return self.cash + self.holdings * price

    def generate_signals(self, row):
        """Generate trading signals based on the policy."""
        state = self.get_state(row).reshape(1, -1)  # Convert row to state
        action = self.get_action(state)  # Get action from policy
        return action  # 0: Hold, 1: Buy, 2: Sell
