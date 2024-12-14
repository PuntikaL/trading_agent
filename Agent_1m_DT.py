from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DecisionTreeAgent:
    """Decision Tree Regressor Agent for 1m interval"""
    def __init__(self, initial_cash=100000):
        self.model = DecisionTreeRegressor()
        self.scaler = StandardScaler()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.trained = False
        self.feature_columns = []

    def train_model(self, training_data, feature_columns, target_column):
        """Train the Decision Tree Regressor model."""
        self.feature_columns = feature_columns
        features = training_data[feature_columns]
        target = training_data[target_column].astype(float)

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train the regressor
        self.model.fit(features_scaled, target)
        self.trained = True

    def generate_signals(self, row):
        """Generate trading signals based on predictions."""
        if not self.trained:
            raise ValueError("Model has not been trained. Call train_model() before using the agent.")

        # Scale features of the row for prediction
        features = pd.DataFrame([row[self.feature_columns]], columns=self.feature_columns)
        features_scaled = self.scaler.transform(features)

        predicted_price = self.model.predict(features_scaled)[0]
        current_price = row['close']

        # Generate signals
        if predicted_price > current_price:
            return 1  # Buy
        elif predicted_price < current_price:
            return -1  # Sell
        return 0  # Hold

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
