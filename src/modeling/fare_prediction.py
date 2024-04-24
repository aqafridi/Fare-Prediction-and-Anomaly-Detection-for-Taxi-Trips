# src/modeling/fare_prediction.py

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from .hyperparameter_tuning import tune_random_forest_hyperparams
from .model_persistence import ModelPersistence

class FarePredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.model_persistence = ModelPersistence('models/')

    def train(self, features, target, tune_hyperparams=False):
        """
        Train the fare prediction model.

        Args:
            features (list): List of feature column names.
            target (str): Name of the target column.
            tune_hyperparams (bool): Whether to tune hyperparameters or use default values.
        """
        X = self.data[features]
        y = self.data[target]

        if tune_hyperparams:
            study = optuna.create_study(direction='minimize')
            self.model = study.optimize(tune_random_forest_hyperparams, n_trials=100, X=X, y=y)
            self.model_persistence.save_model(self.model.user_attrs['best_model'], 'fare_prediction_model.joblib')
        else:
            self.model = RandomForestRegressor()
            self.model.fit(X, y)
            self.model_persistence.save_model(self.model, 'fare_prediction_model.joblib')

    

    def evaluate(self, features, target, cv=5):
        """
        Evaluate the fare prediction model using cross-validation.

        Args:
            features (list): List of feature column names.
            target (str): Name of the target column.
            cv (int): Number of cross-validation folds.

        Returns:
            float: The cross-validated score (R-squared) of the model.
        """
        X = self.data[features]
        y = self.data[target]
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        return scores.mean()

    def predict(self, X):
        """
        Predict the fare amount for new data.

        Args:
            X (pandas.DataFrame): The input data for prediction.

        Returns:
            numpy.ndarray: The predicted fare amounts.
        """
        return self.model.predict(X)