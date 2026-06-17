import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def preprocess(self):
        """
        Load raw data, preprocess it, and save processed files.
        """

        # Load dataset
        data = pd.read_csv(self.config["data"]["raw_data_path"])

        # Remove missing values (if any)
        data = data.dropna()

        # Separate features and target
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["preprocessing"]["test_size"],
            random_state=self.config["preprocessing"]["random_state"],
            stratify=y
        )

        # Feature Scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create processed data directory
        os.makedirs(
            self.config["data"]["processed_data_dir"],
            exist_ok=True
        )

        # Save processed datasets
        joblib.dump(
            X_train_scaled,
            os.path.join(
                self.config["data"]["processed_data_dir"],
                "X_train.pkl"
            )
        )

        joblib.dump(
            X_test_scaled,
            os.path.join(
                self.config["data"]["processed_data_dir"],
                "X_test.pkl"
            )
        )

        joblib.dump(
            y_train,
            os.path.join(
                self.config["data"]["processed_data_dir"],
                "y_train.pkl"
            )
        )

        joblib.dump(
            y_test,
            os.path.join(
                self.config["data"]["processed_data_dir"],
                "y_test.pkl"
            )
        )

        # Save scaler
        joblib.dump(
            scaler,
            os.path.join(
                self.config["data"]["processed_data_dir"],
                "scaler.pkl"
            )
        )

        print("✅ Data preprocessing completed successfully!")