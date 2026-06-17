import os
import joblib


class FeatureSelector:
    def __init__(self, config):
        self.config = config

    def select_features(self):
        """
        Load engineered features and perform feature selection.
        Currently, all features are retained.
        """

        processed_dir = self.config["data"]["processed_data_dir"]

        # Load engineered features
        X_train = joblib.load(
            os.path.join(processed_dir, "X_train_engineered.pkl")
        )

        X_test = joblib.load(
            os.path.join(processed_dir, "X_test_engineered.pkl")
        )

        # --------------------------------------------------
        # Future Feature Selection Methods
        #
        # Examples:
        # - SelectKBest
        # - Recursive Feature Elimination (RFE)
        # - PCA
        # - Mutual Information
        # - Feature Importance
        #
        # For this project, we keep all features.
        # --------------------------------------------------

        # Save selected features
        joblib.dump(
            X_train,
            os.path.join(processed_dir, "X_train_selected.pkl")
        )

        joblib.dump(
            X_test,
            os.path.join(processed_dir, "X_test_selected.pkl")
        )

        print("✅ Feature selection completed successfully!")