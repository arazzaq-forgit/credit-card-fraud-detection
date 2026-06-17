import os
import joblib
import numpy as np


class FraudPredictor:
    def __init__(self, config):
        self.config = config

        model_path = os.path.join(
            self.config["model"]["model_dir"],
            "best_random_forest.pkl"
        )

        self.model = joblib.load(model_path)

    def predict(self, data):
        """
        Predict whether a transaction is fraudulent.

        Parameters:
            data: numpy array or list with transaction features

        Returns:
            prediction, probability
        """

        data = np.array(data).reshape(1, -1)

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0][1]

        return prediction, probability