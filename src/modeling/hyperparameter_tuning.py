# src/modeling/hyperparameter_tuning.py

import optuna
import yaml
from sklearn.ensemble import RandomForestRegressor
from src.config import HYPERPARAMS_FILE

def tune_random_forest_hyperparams(trial, X, y):
    """
    Tune hyperparameters for the Random Forest Regressor using Optuna.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        X (pandas.DataFrame): The input features.
        y (pandas.Series): The target variable.

    Returns:
        RandomForestRegressor: The tuned Random Forest Regressor model.
    """
    with open(HYPERPARAMS_FILE, 'r') as file:
        hyperparams = yaml.safe_load(file)['fare_prediction']['random_forest']

    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'random_state': hyperparams['random_state']
    }

    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X, y)

    return rf_model