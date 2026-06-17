import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class ModelTuner:
    def __init__(self, config):
        self.config = config

    def tune(self):
        """
        Tune Random Forest hyperparameters using GridSearchCV.
        """

        processed_dir = self.config["data"]["processed_data_dir"]
        model_dir = self.config["model"]["model_dir"]

        # Load training data
        X_train = joblib.load(
            os.path.join(processed_dir, "X_train_selected.pkl")
        )

        y_train = joblib.load(
            os.path.join(processed_dir, "y_train.pkl")
        )

        # Base model
        rf = RandomForestClassifier(
            random_state=self.config["preprocessing"]["random_state"],
            n_jobs=-1
        )

        # Hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }

        # Grid Search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )

        # Train Grid Search
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Save tuned model
        joblib.dump(
            best_model,
            os.path.join(model_dir, "best_random_forest.pkl")
        )

        print("✅ Hyperparameter tuning completed!")
        print("Best Parameters:", grid_search.best_params_)