from utils import TradingAgent
import pandas as pd
import numpy as np

class Agent_1h(TradingAgent):
    """Trading Agent for 1-hour interval."""
    
    def __init__(self, initial_cash=100000):
        super().__init__(initial_cash)
        self.window_short = 6
        self.window_mid = 18
        self.window_long = 45
        self.rsi_window = 14
        self.min_periods = 55
        self.prices = []
        self.volumes = []
        self.last_action = 0
        self.entry_price = None
        self.stop_loss = 0.02
        
    def calculate_rsi(self, prices):
        prices = pd.Series(prices)
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_window, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(span=self.rsi_window, adjust=False).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
        
    def calculate_ema(self, prices, window):
        """Calculate EMA."""
        return pd.Series(prices).ewm(span=window, adjust=False).mean().iloc[-1]
    
    def should_exit_position(self, current_price):
        """Check stop loss."""
        if self.entry_price is None or self.position == 0:
            return False
            
        if self.position > 0:  # Long position
            return (current_price - self.entry_price) / self.entry_price <= -self.stop_loss
        else:  # Short position
            return (self.entry_price - current_price) / self.entry_price <= -self.stop_loss
    
    def get_trend_quality(self, ema_short, ema_mid, ema_long):
        """Calculate trend quality and alignment."""
        # Check EMA spacing
        spacing_short_mid = abs((ema_short - ema_mid) / ema_mid)
        spacing_mid_long = abs((ema_mid - ema_long) / ema_long)
        
        # Good trend has well-spaced EMAs
        return spacing_short_mid > 0.0008 and spacing_mid_long > 0.0012
        
    def generate_signals(self, data):
        """Generate trading signals."""
        try:
            current_price = float(data['close'])
            
            # Check stop loss first
            if self.should_exit_position(current_price):
                self.entry_price = None
                return -1 if self.position > 0 else 1
            
            # Store data
            self.prices.append(current_price)
            self.volumes.append(float(data['volume']))
            
            # Wait for enough data
            if len(self.prices) < self.min_periods:
                return 0
                
            # Calculate indicators
            ema_short = self.calculate_ema(self.prices, self.window_short)
            ema_mid = self.calculate_ema(self.prices, self.window_mid)
            ema_long = self.calculate_ema(self.prices, self.window_long)
            rsi = self.calculate_rsi(self.prices)
            
            # Volume analysis
            volume_ma = pd.Series(self.volumes).rolling(window=20).mean().iloc[-1]
            volume_condition = self.volumes[-1] > volume_ma * 1.2
            
            # Trend strength with quality check
            trend_strength = (ema_short - ema_long) / ema_long
            trend_quality = self.get_trend_quality(ema_short, ema_mid, ema_long)
            
            # Keep buffer size manageable
            if len(self.prices) > self.min_periods:
                self.prices.pop(0)
                self.volumes.pop(0)
            
            # No consecutive signals in the same direction
            if self.last_action != 0 and len(self.history) < 3:
                return 0
            
            # Strong uptrend conditions with quality check
            uptrend = (
                ema_short > ema_mid > ema_long and 
                trend_strength > 0.001 and
                trend_quality and
                rsi > 45 and rsi < 70
            )
                      
            # Strong downtrend conditions with quality check
            downtrend = (
                ema_short < ema_mid < ema_long and 
                trend_strength < -0.001 and
                trend_quality and
                rsi < 55 and rsi > 30
            )
            
            if self.position == 0:  # No position
                if uptrend and volume_condition:
                    self.last_action = 1
                    self.entry_price = current_price
                    return 1
                elif downtrend and volume_condition:
                    self.last_action = -1
                    self.entry_price = current_price
                    return -1
            else:  # Has position
                if self.position > 0 and downtrend:  # Exit long
                    self.last_action = -1
                    self.entry_price = None
                    return -1
                elif self.position < 0 and uptrend:  # Exit short
                    self.last_action = 1
                    self.entry_price = None
                    return 1
            
            return 0  # Hold
            
        except Exception as e:
            print(f"Error in signal generation: {str(e)}")
            return 0