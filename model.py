import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TrafficModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self._train_model()

    def _train_model(self):
        """
        Train the model with sample data.
        In a real application, this would use actual historical data.
        """
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000

        X = {
            'time_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'weather': np.random.randint(0, 4, n_samples),
            'temperature': np.random.normal(20, 10, n_samples),
            'special_event': np.random.randint(0, 2, n_samples)
        }

        # Store training data as DataFrame
        self.X_train = pd.DataFrame(X)

        # Generate synthetic target
        self.y_train = (
            X['time_of_day'] * 2 +
            X['day_of_week'] * 3 +
            X['weather'] * 5 +
            X['temperature'] * 0.5 +
            X['special_event'] * 10 +
            np.random.normal(0, 5, n_samples)
        )

        # Normalize to 0-100 range
        self.y_train = ((self.y_train - self.y_train.min()) / 
                        (self.y_train.max() - self.y_train.min())) * 100

        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.feature_names = list(X.keys())

    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)

    def get_metrics(self):
        """Return model evaluation metrics"""
        # Using training metrics for demonstration
        y_pred = self.model.predict(self.X_train)
        return {
            'MAE': mean_absolute_error(self.y_train, y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_train, y_pred)),
            'R²': r2_score(self.y_train, y_pred)
        }

    def get_feature_importance(self):
        """Return feature importance DataFrame"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)