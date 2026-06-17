import os
import joblib

from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        """
        Train a Random Forest classifier and save the trained model.
        """

        processed_dir = self.config["data"]["processed_data_dir"]
        model_dir = self.config["model"]["model_dir"]

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Load training data
        X_train = joblib.load(
            os.path.join(processed_dir, "X_train_selected.pkl")
        )

        y_train = joblib.load(
            os.path.join(processed_dir, "y_train.pkl")
        )

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config["preprocessing"]["random_state"],
            n_jobs=-1
        )

        # Train model
        model.fit(X_train, y_train)

        # Save trained model
        joblib.dump(
            model,
            os.path.join(model_dir, "random_forest.pkl")
        )

        print("✅ Model trained and saved successfully!")