# app.py

from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.modeling.fare_prediction import FarePredictor
from src.modeling.anomaly_detection import AnomalyDetector

# Load data
data_loader = DataLoader('data/2021_Yellow_Taxi_Trip_Data_20240406.csv')
data = data_loader.load_data()

# Feature engineering
feature_engineer = FeatureEngineer(data)
data_with_features = feature_engineer.add_features()

# Fare prediction
fare_predictor = FarePredictor(data_with_features)
fare_features = ['passenger_count', 'trip_distance', 'trip_duration']
target = 'total_amount'
fare_predictor.train(fare_features, target)

# Anomaly detection
anomaly_detector = AnomalyDetector(data_with_features)
anomaly_features = fare_features + ['total_amount']
anomaly_detector.train(anomaly_features)

# Make predictions and detect anomalies on new data
new_data = data_with_features.sample(10)
predictions = fare_predictor.predict(new_data[fare_features])
anomaly_scores = anomaly_detector.detect_anomalies(new_data[anomaly_features])
threshold = anomaly_detector.get_anomaly_threshold(anomaly_scores)

# Classify instances as anomalous or normal
is_anomaly = anomaly_scores < threshold
print("Predictions:", predictions)
print("Anomalies:", is_anomaly)