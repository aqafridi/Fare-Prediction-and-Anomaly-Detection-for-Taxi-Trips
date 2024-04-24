# src/modeling/model_persistence.py

import pickle
import os
import joblib

class ModelPersistence:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, filename):
        """
        Save the trained model to a file.

        Args:
            model: The trained model object.
            filename (str): The name of the file to save the model.
        """
        file_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, file_path)

    def load_model(self, filename):
        """
        Load a saved model from a file.

        Args:
            filename (str): The name of the file containing the saved model.

        Returns:
            The loaded model object.
        """
        file_path = os.path.join(self.model_dir, filename)
        model = joblib.load(file_path)
        return model