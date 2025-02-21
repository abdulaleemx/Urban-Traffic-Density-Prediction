import pandas as pd
import numpy as np

def preprocess_data(data):
    """
    Preprocess input data for model prediction
    """
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Convert categorical variables
    if 'day_of_week' in df.columns:
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['day_of_week'] = df['day_of_week'].map(day_mapping)
    
    if 'weather' in df.columns:
        weather_mapping = {
            'Clear': 0, 'Rain': 1, 'Snow': 2, 'Cloudy': 3
        }
        df['weather'] = df['weather'].map(weather_mapping)
    
    # Ensure all required columns are present
    required_columns = ['time_of_day', 'day_of_week', 'weather', 
                       'temperature', 'special_event']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def load_sample_data():
    """
    Load sample data for demonstration
    """
    np.random.seed(42)
    n_samples = 24  # One day worth of hourly data
    
    sample_data = pd.DataFrame({
        'time_of_day': range(24),
        'day_of_week': ['Monday'] * n_samples,
        'weather': np.random.choice(['Clear', 'Rain', 'Snow', 'Cloudy'], n_samples),
        'temperature': np.random.normal(20, 5, n_samples),
        'special_event': np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    })
    
    return sample_data
