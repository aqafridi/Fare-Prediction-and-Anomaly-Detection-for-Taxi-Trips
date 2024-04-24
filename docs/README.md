# Fare Prediction and Anomaly Detection for Yellow Taxi Trips

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Model Persistence](#model-persistence)
9. [Model Deployment](#model-deployment)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project aims to build a machine learning system for predicting the total fare amount for yellow taxi trips in New York City and detecting potential anomalies or fraudulent trip records. The system leverages advanced regression modeling techniques for fare prediction and unsupervised anomaly detection algorithms. Additionally, it incorporates hyperparameter tuning and model persistence for optimizing and saving the trained models.

## Project Structure

The project follows a modular structure with separate modules for data loading, feature engineering, modeling, hyperparameter tuning, model persistence, and result saving. The main components are:

- `src/data/data_loader.py`: Module for loading the taxi trip data from a CSV file.
- `src/features/feature_engineering.py`: Module for creating new features from the existing data.
- `src/modeling/fare_prediction.py`: Module for training and evaluating the fare prediction model.
- `src/modeling/anomaly_detection.py`: Module for training and using the anomaly detection model.
- `src/modeling/hyperparameter_tuning.py`: Module for tuning the hyperparameters of the Random Forest Regressor using Optuna.
- `src/modeling/model_persistence.py`: Module for saving and loading trained models to/from disk.
- `src/modeling/result_saver.py`: Module for saving predictions and detected anomalies to CSV files.
- `src/config.py`: Module for managing configuration settings, such as file paths and hyperparameters.
- `src/app.py`: Entry point for the application, containing the main execution logic.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aqafridi/Fare-Prediction-and-Anomaly-Detection-for-Taxi-Trips.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing

The project uses the 2021 Yellow Taxi Trip Data dataset, which can be obtained from the [NYC Taxi and Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Place the downloaded CSV file in the `data/` directory.

The `DataLoader` class in `src/data/data_loader.py` is responsible for loading the data from the CSV file and converting the data types as necessary. New features are created using the `FeatureEngineer` class in `src/features/feature_engineering.py`.

## Model Training

The `FarePredictor` class in `src/modeling/fare_prediction.py` is used for training the Random Forest Regressor model for fare prediction. The `AnomalyDetector` class in `src/modeling/anomaly_detection.py` is used for training the Isolation Forest model for anomaly detection.

The `app.py` script demonstrates how to load the data, perform feature engineering, train the models, and make predictions and detect anomalies on new data.

## Model Evaluation

The `FarePredictor` class provides an `evaluate` method that performs cross-validation on the trained model and returns the cross-validated R-squared score. This method can be used to assess the performance of the fare prediction model.

## Hyperparameter Tuning

The project uses Optuna for tuning the hyperparameters of the Random Forest Regressor model. The `tune_random_forest_hyperparams` function in `src/modeling/hyperparameter_tuning.py` defines the hyperparameter search spaces and optimization objective.

The `FarePredictor` class in `src/modeling/fare_prediction.py` has a `tune_hyperparams` parameter that, when set to `True`, triggers the hyperparameter tuning process using Optuna.

## Model Persistence

The `ModelPersistence` class in `src/modeling/model_persistence.py` provides methods for saving and loading trained models to/from disk using `joblib`. The `FarePredictor` class in `src/modeling/fare_prediction.py` uses this class to save the trained model after training or hyperparameter tuning.

## Model Deployment

The trained models can be containerized using Docker and deployed to a cloud platform (e.g., AWS, Google Cloud, or Azure) with a RESTful API or serverless function for serving predictions. The necessary code and instructions for deployment are not included in this project but can be added as a separate component.

## Contributing

Contributions are welcome! Please follow the contribution guidelines outlined in the `CONTRIBUTING.md` file.

## License

This project is licensed under the [MIT License](LICENSE).