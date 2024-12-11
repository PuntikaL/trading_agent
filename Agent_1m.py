import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils import TradingAgent

class Agent_1m(TradingAgent):
    """1m agent with linear regression"""
    
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = []

    def train_model(self, training_data, feature_columns, target_column):
        """Train the Linear Regression model."""
        self.feature_columns = feature_columns
        features = training_data[feature_columns]
        target = training_data[target_column]
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, target)
        self.trained = True

    def generate_signals(self, row):
        """Generate trading signals based on predictions."""
        if not self.trained:
            raise ValueError("Model has not been trained. Call train_model() before using the agent.")

        # Here we are converting the current row into a DataFrame with feature names
        current_features = pd.DataFrame([row[self.feature_columns]], columns=self.feature_columns)
        
        # Here i scale features
        scaled_features = self.scaler.transform(current_features)
        
        # In order to predict the next price
        predicted_price = self.model.predict(scaled_features)[0]
        current_price = row['close']

        # Here we generate signals based on predicted price
        if predicted_price > current_price:
            return 1  # Buy
        elif predicted_price < current_price:
            return -1  # Sell
        return 0  # Hold
