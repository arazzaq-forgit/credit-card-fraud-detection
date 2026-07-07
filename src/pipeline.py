import yaml

from src.data.ingest import DataIngestion
from src.data.preprocess import DataPreprocessor

from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector

from src.models.train import ModelTrainer
from src.models.tune import ModelTuner
from src.models.evaluate import ModelEvaluator

from src.visualization.plots import ModelVisualizer


class FraudDetectionPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def run(self):
        print("=" * 60)
        print("Credit Card Fraud Detection Pipeline")
        print("=" * 60)

        # Step 1: Data Ingestion
        DataIngestion(self.config).ingest()

        # Step 2: Data Preprocessing
        DataPreprocessor(self.config).preprocess()

        # Step 3: Feature Engineering
        FeatureEngineer(self.config).engineer_features()

        # Step 4: Feature Selection
        FeatureSelector(self.config).select_features()

        # Step 5: Model Training
        ModelTrainer(self.config).train()

        # Step 6: Hyperparameter Tuning
        ModelTuner(self.config).tune()

        # Step 7: Model Evaluation
        ModelEvaluator(self.config).evaluate()

        # Step 8: Generate Visualizations
        ModelVisualizer(self.config).generate_plots()

        print("\n🎉 Pipeline executed successfully!")


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline("configs.yaml")
    pipeline.run()
