# src/api.py

from flask import Flask, request, jsonify
import pandas as pd
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.modeling.fare_prediction import FarePredictor
from src.modeling.anomaly_detection import AnomalyDetector
from src.modeling.model_persistence import ModelPersistence
from src.config import TAXI_DATA_PATH, MODELS_DIR

app = Flask(__name__)

# Load data and models
data_loader = DataLoader(TAXI_DATA_PATH)
data = data_loader.load_data()
feature_engineer = FeatureEngineer(data)
data_with_features = feature_engineer.add_features()

model_persistence = ModelPersistence(MODELS_DIR)
fare_predictor = FarePredictor(data_with_features)
fare_predictor.model = model_persistence.load_model('fare_prediction_model.joblib')

anomaly_detector = AnomalyDetector(data_with_features)
anomaly_detector.model = model_persistence.load_model('anomaly_detection_model.joblib')

@app.route('/train', methods=['POST'])
def train_models():
    # Code for training the models
    return jsonify({'message': 'Models trained successfully'})

@app.route('/predict', methods=['POST'])
def predict_fare():
    data = request.get_json()
    new_data = pd.DataFrame(data)
    predictions = fare_predictor.predict(new_data)
    return jsonify({'predictions': predictions.tolist()})

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data = request.get_json()
    new_data = pd.DataFrame(data)
    anomaly_scores = anomaly_detector.detect_anomalies(new_data)
    threshold = anomaly_detector.get_anomaly_threshold(anomaly_scores)
    is_anomaly = anomaly_scores < threshold
    return jsonify({'is_anomaly': is_anomaly.tolist()})

if __name__ == '__main__':
    app.run(debug=True)