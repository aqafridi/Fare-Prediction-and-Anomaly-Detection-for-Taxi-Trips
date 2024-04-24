# app.py

from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.modeling.fare_prediction import FarePredictor
from src.modeling.anomaly_detection import AnomalyDetector
from src.modeling.model_persistence import ModelPersistence
from src.modeling.result_saver import ResultSaver
from src.config import TAXI_DATA_PATH

# Load data
data_loader = DataLoader(TAXI_DATA_PATH)
data = data_loader.load_data()
data.dropna()
# Feature engineering
feature_engineer = FeatureEngineer(data)
data_with_features = feature_engineer.add_features()

# Fare prediction
fare_predictor = FarePredictor(data_with_features)
fare_features = ['passenger_count', 'trip_distance', 'trip_duration']
target = 'total_amount'
fare_predictor.train(fare_features, target,tune_hyperparams=True)
score = fare_predictor.evaluate(fare_features, target)
print(f"Cross-validated R-squared score: {score:.3f}")

# Anomaly detection
anomaly_detector = AnomalyDetector(data_with_features)
anomaly_features = fare_features + ['total_amount']
anomaly_detector.train(anomaly_features)

# Make predictions and detect anomalies on new data
new_data = data_with_features.sample(10)
predictions = fare_predictor.predict(new_data[fare_features])
anomaly_scores = anomaly_detector.detect_anomalies(new_data[anomaly_features])
threshold = anomaly_detector.get_anomaly_threshold(anomaly_scores)
is_anomaly = anomaly_scores < threshold

# Save models
model_persistence = ModelPersistence('models/')
model_persistence.save_model(fare_predictor.model, 'fare_prediction_model.pkl')
model_persistence.save_model(anomaly_detector.model, 'anomaly_detection_model.pkl')

# Save results
result_saver = ResultSaver('results/')
result_saver.save_predictions(predictions, 'predictions.csv')
result_saver.save_anomalies(is_anomaly, 'anomalies.csv')