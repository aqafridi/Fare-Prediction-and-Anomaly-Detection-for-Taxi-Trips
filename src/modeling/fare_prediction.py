
from sklearn.ensemble import RandomForestRegressor

class FarePredictor:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestRegressor()

    def train(self, features, target):
        """
        Train the fare prediction model.
        
        Args:
            features (list): List of feature column names.
            target (str): Name of the target column.
        """
        X = self.data[features]
        y = self.data[target]
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the fare amount for new data.
        
        Args:
            X (pandas.DataFrame): The input data for prediction.
            
        Returns:
            numpy.ndarray: The predicted fare amounts.
        """
        return self.model.predict(X)