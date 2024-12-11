from utils import TradingAgent
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class PCAAgent(TradingAgent):
    """
    Agent that uses Principal Component Analysis (PCA) on rolling windows of close prices 
    to generate trading signals.
    """
    def __init__(self, short_window=200, long_window=500, initial_cash=100000, n_components=3):
        super().__init__(initial_cash)
        self.short_window = short_window
        self.long_window = long_window
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def compute_rolling(self, data):
        """Compute rolling features for PCA."""
        features = data[['close']].values
        if len(features) < self.short_window:
            return None  # Not enough data for PCA
        rolling_features = np.array(
            [features[i:i + self.short_window] for i in range(len(features) - self.short_window + 1)]
        )
        rolling_features = rolling_features.reshape(rolling_features.shape[0], -1)
        return rolling_features

    def train(self, data):
        """Train the model using PCA-transformed rolling features."""
        data = data.copy()
        data['Next_Close'] = data['close'].shift(-1)
        data['Target'] = np.where(data['Next_Close'] > data['close'], 1, 2)  # 1: Buy, 2: Sell
        data.dropna(inplace=True)

        rolling_features = self.compute_rolling(data)
        if rolling_features is None:
            raise ValueError("Not enough data for training.")

        # Fit PCA with rolling features
        pca_features = self.pca.fit_transform(rolling_features)
        target = data['Target'].iloc[self.short_window - 1:].values

        X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    def generate_signals(self, row):
        """
        Generate trading signals based on the latest rolling window of data.
        
        Parameters:
        - row: A single row of data passed during backtesting.
        
        Returns:
        - 1: Buy signal
        - 2: Sell signal
        - 0: No signal (if insufficient data)
        """
        if not hasattr(self, 'rolling_window'):
            self.rolling_window = []

        # Update the rolling window with the latest close price
        self.rolling_window.append(row['close'])
        if len(self.rolling_window) > self.short_window:
            self.rolling_window.pop(0)

        if len(self.rolling_window) < self.short_window:
            return 0  # Not enough data for prediction

        rolling_features = np.array(self.rolling_window).reshape(1, -1)

        # Ensure PCA is fitted before transforming
        if not hasattr(self.pca, 'components_'):
            raise ValueError("PCA must be fitted before generating signals.")

        # Apply PCA on rolling window features
        pca_features = self.pca.transform(rolling_features)
        prediction = self.model.predict(pca_features)[0]

        if prediction == 1:
            return 1  # Buy signal
        else:
            return 2  # Sell signal
