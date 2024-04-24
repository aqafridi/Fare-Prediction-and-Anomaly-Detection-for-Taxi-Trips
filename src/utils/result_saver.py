# src/modeling/result_saver.py

import pandas as pd

class ResultSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save_predictions(self, predictions, filename):
        """
        Save the predictions to a CSV file.

        Args:
            predictions (numpy.ndarray or pandas.Series): The predictions to be saved.
            filename (str): The name of the CSV file to save the predictions.
        """
        file_path = f"{self.output_dir}/{filename}"
        pd.Series(predictions).to_csv(file_path, index=False, header=['prediction'])

    def save_anomalies(self, is_anomaly, filename):
        """
        Save the detected anomalies to a CSV file.

        Args:
            is_anomaly (numpy.ndarray or pandas.Series): Boolean array indicating whether each instance is an anomaly.
            filename (str): The name of the CSV file to save the anomalies.
        """
        file_path = f"{self.output_dir}/{filename}"
        pd.Series(is_anomaly).to_csv(file_path, index=False, header=['is_anomaly'])