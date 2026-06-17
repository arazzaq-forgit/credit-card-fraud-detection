import joblib
import os


class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def engineer_features(self):
        """
        Load processed data and perform feature engineering.
        Currently, no additional feature engineering is required
        because the dataset is already PCA transformed.
        """

        processed_dir = self.config["data"]["processed_data_dir"]

        # Load processed data
        X_train = joblib.load(os.path.join(processed_dir, "X_train.pkl"))
        X_test = joblib.load(os.path.join(processed_dir, "X_test.pkl"))

        # -----------------------------------------------------
        # Future Feature Engineering Can Be Added Here
        # Example:
        # - Polynomial Features
        # - Interaction Features
        # - Log Transformations
        # - Domain-specific Features
        # -----------------------------------------------------

        # Save engineered features
        joblib.dump(
            X_train,
            os.path.join(processed_dir, "X_train_engineered.pkl")
        )

        joblib.dump(
            X_test,
            os.path.join(processed_dir, "X_test_engineered.pkl")
        )

        print("✅ Feature engineering completed successfully!")