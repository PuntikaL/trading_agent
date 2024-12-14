from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils import TradingAgent

class DecisionTreeAgent(TradingAgent):
    """Decision Tree Classifier Agent for Algorithmic Trading"""
    
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.model = DecisionTreeClassifier()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = []

    def train_model(self, training_data, feature_columns, target_column):
        """Here wrain the Decision Tree Classifier model."""
        self.feature_columns = feature_columns
        features = training_data[feature_columns]
        target = training_data[target_column]
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, target)
        self.trained = True

    def generate_signals(self, row):
        """Here we generate trading signals using the Decision Tree Classifier."""
        if not self.trained:
            raise ValueError("Model has not been trained. Call train_model() before using the agent.")

        current_features = pd.DataFrame([row[self.feature_columns]], columns=self.feature_columns)
        scaled_features = self.scaler.transform(current_features)
        prediction = self.model.predict(scaled_features)[0]
        return prediction  # The prediction corresponds to Buy (1), Sell (-1), or Hold (0)
