import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


class ModelVisualizer:
    def __init__(self, config):
        self.config = config

    def generate_plots(self):
        """
        Generate and save evaluation plots.
        """

        processed_dir = self.config["data"]["processed_data_dir"]
        model_dir = self.config["model"]["model_dir"]
        output_dir = self.config["visualization"]["output_dir"]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
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

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("Confusion Matrix")
        plt.savefig(
            os.path.join(output_dir, "confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # -----------------------------
        # ROC Curve
        # -----------------------------
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("ROC Curve")
        plt.savefig(
            os.path.join(output_dir, "roc_curve.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # -----------------------------
        # Precision-Recall Curve
        # -----------------------------
        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title("Precision-Recall Curve")
        plt.savefig(
            os.path.join(output_dir, "precision_recall_curve.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print("✅ Visualization plots saved successfully!")