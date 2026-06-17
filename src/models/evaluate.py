import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self):
        """
        Evaluate the trained model on the test dataset.
        """

        processed_dir = self.config["data"]["processed_data_dir"]
        model_dir = self.config["model"]["model_dir"]

        # Load test data
        X_test = joblib.load(
            os.path.join(processed_dir, "X_test_selected.pkl")
        )

        y_test = joblib.load(
            os.path.join(processed_dir, "y_test.pkl")
        )

        # Load trained model
        model = joblib.load(
            os.path.join(model_dir, "best_random_forest.pkl")
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Print Results
        print("\n========== Model Evaluation ==========")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))