import os
import shutil


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def ingest(self):
        """
        Verify the dataset exists and copy it to the processed directory.
        """

        raw_data_path = self.config["data"]["raw_data_path"]
        processed_dir = self.config["data"]["processed_data_dir"]

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"Dataset not found: {raw_data_path}"
            )

        os.makedirs(processed_dir, exist_ok=True)

        destination = os.path.join(
            processed_dir,
            "creditcard.csv"
        )

        shutil.copy(raw_data_path, destination)

        print("✅ Data ingestion completed successfully!")