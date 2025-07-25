from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
sys.path.append('/opt/airflow/dags')

from processing.spark_processor import StudentDataProcessor
from analytics.advanced_analytics import AdvancedAnalytics

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'student_analytics_pipeline',
    default_args=default_args,
    description='Student Performance Analytics Pipeline',
    schedule_interval=timedelta(hours=6),
    catchup=False
)

def extract_data():
    """Extract data from sources"""
    from data_ingestion.extract_data import DataExtractor
    extractor = DataExtractor()
    extractor.download_datasets()

def process_batch_data():
    """Process batch data with Spark"""
    processor = StudentDataProcessor()
    datasets = processor.load_datasets()
    
    for name, df in datasets.items():
        print(f"Processing {name}")
        processor.data_quality_checks(df, name)
        # Additional processing logic here
    
    processor.close()

def run_advanced_analytics():
    """Run advanced analytics"""
    analytics = AdvancedAnalytics()
    # Load processed data and run analytics
    # This would load from your processed data store
    pass

def generate_reports():
    """Generate final reports"""
    print("Generating comprehensive reports...")
    # Report generation logic here

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

process_task = PythonOperator(
    task_id='process_batch_data',
    python_callable=process_batch_data,
    dag=dag
)

analytics_task = PythonOperator(
    task_id='run_advanced_analytics',
    python_callable=run_advanced_analytics,
    dag=dag
)

report_task = PythonOperator(
    task_id='generate_reports',
    python_callable=generate_reports,
    dag=dag
)

# Set task dependencies
extract_task >> process_task >> analytics_task >> report_task
