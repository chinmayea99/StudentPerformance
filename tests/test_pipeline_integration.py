import unittest
from src.pipeline.full_pipeline import FullPipeline

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.pipeline = FullPipeline()
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution"""
        result = self.pipeline.run_full_pipeline()
        self.assertTrue(result['success'])
        self.assertGreater(result['records_processed'], 0)
    
    def test_streaming_integration(self):
        """Test streaming pipeline"""
        stream_result = self.pipeline.test_streaming_pipeline()
        self.assertTrue(stream_result['kafka_connected'])
        self.assertTrue(stream_result['spark_streaming_active'])

if __name__ == '__main__':
    unittest.main()
