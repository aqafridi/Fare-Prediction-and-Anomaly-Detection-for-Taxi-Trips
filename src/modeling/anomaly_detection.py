

from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, data):
        self.data = data
        self.model = IsolationForest(contamination=0.1)  # Adjust contamination parameter as needed

    def train(self, features):
        """
        Train the anomaly detection model.

        Args:
            features (list): List of feature column names.
        """
        X = self.data[features]
        self.model.fit(X)

    def detect_anomalies(self, X):
        """
        Detect anomalies in the input data.

        Args:
            X (pandas.DataFrame): The input data for anomaly detection.

        Returns:
            numpy.ndarray: Array of anomaly scores for each instance.
        """
        return self.model.decision_function(X)

    def get_anomaly_threshold(self, anomaly_scores, contamination=0.1):
        """
        Calculate the anomaly score threshold based on the desired contamination level.

        Args:
            anomaly_scores (numpy.ndarray): Array of anomaly scores.
            contamination (float): The desired level of contamination (fraction of anomalies).

        Returns:
            float: The anomaly score threshold.
        """
        threshold = sorted(anomaly_scores)[int((1 - contamination) * len(anomaly_scores))]
        return threshold