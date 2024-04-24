# Fare Prediction and Anomaly Detection for Yellow Taxi Trips

This project aims to build a machine learning system for predicting the total fare amount for yellow taxi trips in New York City and detecting potential anomalies or fraudulent trip records. The system leverages advanced regression modeling techniques for fare prediction and unsupervised anomaly detection algorithms. Additionally, it incorporates hyperparameter tuning and model persistence for optimizing and saving the trained models.

## Dataset

The project uses the 2021 Yellow Taxi Trip Data dataset, which includes information about taxi trips in NYC, such as pickup and dropoff times, locations, passenger counts, fare details, and more. The dataset can be obtained from the [NYC Taxi and Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Features

- **Fare Prediction:** A Random Forest Regressor is used to predict the total fare amount based on features like trip distance, passenger count, pickup and dropoff locations, and more.
- **Anomaly Detection:** An Isolation Forest algorithm is implemented to identify potential anomalies or outliers in the trip data, which could indicate fraudulent or erroneous records.
- **Feature Engineering:** New features are created by combining or transforming existing features, such as calculating trip duration, deriving day of the week, time of day, and geographic features.
- **Hyperparameter Tuning:** Optuna is used to tune the hyperparameters of the Random Forest Regressor for optimizing its performance.
- **Model Persistence:** The trained models are saved to disk using `joblib` for easy loading and deployment.
- **Model Deployment:** The trained models can be containerized using Docker and deployed to a cloud platform (e.g., AWS, Google Cloud, or Azure) with a RESTful API or serverless function for serving predictions.
- **Monitoring and Logging:** Mechanisms for monitoring the deployed models' performance, detecting issues, and facilitating model retraining or updates can be implemented.

## Project Structure
```bash
taxi-fare-prediction-anomaly-detection/
├── data/                            # Directory for storing data
├── dashboard/                       # For Dashboard of app
│   ├── index.html
│   ├── main.js
│   └── app.vue
├── docs/                            # Directory for documentation
├── models/                          # Directory for storing trained models
├── results/                         # Directory for storing results and logs
├── config/                          # Directory for configuration files
│   └── hyperparams.yml              # YAML file for hyperparameters
├── src/                             # Source code directory
│   ├── __init__.py                  # Initialization file
│   ├── config.py                    # Configuration module
│   ├── data/                        # Data-related modules
│   │   ├── __init__.py
│   │   └── data_loader.py           # Data loading module
│   ├── features/                    # Feature engineering modules
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # Feature engineering module
│   ├── modeling/                    # Modeling modules
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py     # Anomaly detection module
│   │   ├── fare_prediction.py       # Fare prediction module
│   │   ├── hyperparameter_tuning.py # Hyperparameter tuning module
│   │   ├── model_persistence.py     # Model persistence module
│   │   └── result_saver.py          # Result saving module
│   ├── utils/                       # Utility modules
│   │   ├── __init__.py
│   │   └── utils.py                 # Utility functions module
│   
├── app.py                       # Main application script
├── tests/                           # Directory for tests
├── .gitignore                       # Git ignore file
├── requirements.txt                 # Python dependencies file
└── README.md                        # Project README file

```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/aqafridi/Fare-Prediction-and-Anomaly-Detection-for-Taxi-Trips.git
```
2. Install the required dependencies (e.g., Python, scikit-learn, TensorFlow, Docker).
```bash
pip install -r requirements.txt
```

3. Follow the instructions in the `docs` directory to set up the project environment, preprocess the data, train the models, and deploy the system.


## Contributing

Contributions are welcome! Please follow the contribution guidelines outlined in the `CONTRIBUTING.md` file.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The NYC Taxi and Limousine Commission for providing the dataset.
- The open-source machine learning and data science communities for their valuable tools and resources.
