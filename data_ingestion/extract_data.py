# data_ingestion/extract_data.py
import requests
import pandas as pd
import os
from pathlib import Path

class DataExtractor:
    def __init__(self, base_url="https://github.com/chinmayea99/StudentPerformance/datasets/"):
        self.base_url = base_url
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_datasets(self):
        """Download all datasets from the repository"""
        datasets = {
            'performance_binary': 'Student Performance Prediction - Binary Scenario/',
            'performance_multiclass': 'Student Performance Prediction - Multiclass Case/',
            'engagement_binary': 'Student Engagement Level Prediction - Binary Case/',
            'engagement_multiclass': 'Student Engagement Level Prediction - Multiclass Case/'
        }
        
        for name, path in datasets.items():
            try:
                # This would need to be adjusted based on actual file structure
                csv_files = self.discover_csv_files(path)
                for csv_file in csv_files:
                    self.download_file(path + csv_file, name + '_' + csv_file)
            except Exception as e:
                print(f"Error downloading {name}: {e}")
    
    def download_file(self, remote_path, local_name):
        """Download individual file"""
        url = self.base_url + remote_path
        response = requests.get(url)
        if response.status_code == 200:
            file_path = self.data_dir / local_name
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {local_name}")
        else:
            print(f"Failed to download: {remote_path}")

if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.download_datasets()
