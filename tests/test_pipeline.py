import unittest

from src.pipeline import FraudDetectionPipeline


class TestPipeline(unittest.TestCase):

    def test_pipeline_initialization(self):
        """
        Test whether the pipeline initializes successfully.
        """
        pipeline = FraudDetectionPipeline()

        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()