# StudentPerformance
Big data project on Student Performance in a learning management system

# Step 1: Environment Setup

## 1. Create project structure
mkdir -p {data/{raw,processed,models},src/{ingestion,processing,analytics,dashboard},config,logs,reports/{figures,documents},tests}

## 2. Set up Docker environment
docker-compose up -d

## 3. Install Python dependencies
pip install -r requirements.txt

## 4. Initialize Hadoop and create directories
docker exec -it namenode hdfs dfsadmin -safemode leave
docker exec -it namenode hdfs dfs -mkdir -p /user/student_analytics/{raw,processed,models}

# Step 2: Data Pipeline Development

## 1. Implement data extraction
python src/ingestion/extract_data.py

## 2. Set up Kafka for streaming
python src/streaming/kafka_producer.py &
python src/streaming/kafka_consumer.py &

## 3. Test Spark processing
python src/processing/spark_processor.py

## 4. Validate data quality
python src/analytics/data_validation.py

# Step 3: Analytics Implementation 

# 1. Run advanced analytics
python src/analytics/advanced_analytics.py

# 2. Train machine learning models
python src/models/train_models.py

# 3. Generate insights
python src/analytics/generate_insights.py

## Step 4: Dashboard Deployment 

# 1. Launch dashboard
python src/dashboard/app.py

# 2. Configure Airflow DAGs
cp dags/* /opt/airflow/dags/
docker exec -it airflow airflow dags list

# 3. Start scheduled workflows
docker exec -it airflow airflow dags unpause student_analytics_pipeline
