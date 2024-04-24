
import pandas as pd

class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def calculate_trip_duration(self):
        """
        Calculate the trip duration in seconds from the pickup and dropoff datetime.
        
        Returns:
            pandas.Series: The trip duration in seconds.
        """
        duration = self.data['tpep_dropoff_datetime'] - self.data['tpep_pickup_datetime']
        return duration.dt.total_seconds()

    def add_features(self):
        """
        Add new features to the data.
        
        Returns:
            pandas.DataFrame: The data with additional features.
        """
        self.data['trip_duration'] = self.calculate_trip_duration()
        # Add other feature engineering steps here
        return self.data