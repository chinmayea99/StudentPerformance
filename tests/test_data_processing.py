import unittest
import pandas as pd
from src.processing.spark_processor import StudentDataProcessor

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = StudentDataProcessor()
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'student_id': ['S001', 'S002', 'S003'],
            'score': [85, 92, 78],
            'engagement': ['high', 'medium', 'high'],
            'time_spent': [120, 90, 150]
        })
    
    def test_data_quality_checks(self):
        """Test data quality validation"""
        result = self.processor.data_quality_checks(self.sample_data, "test_dataset")
        self.assertIsInstance(result, dict)
        self.assertIn('missing_values', result)
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        processed_data = self.processor.preprocess_data(self.sample_data, 'score')
        self.assertFalse(processed_data.isnull().any().any())
    
    def tearDown(self):
        self.processor.close()

if __name__ == '__main__':
    unittest.main()
