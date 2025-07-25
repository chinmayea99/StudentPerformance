from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

class StudentDataProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("StudentPerformanceAnalytics") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
    
    def load_datasets(self, data_path="data/raw"):
        """Load and combine all datasets"""
        # Performance datasets
        perf_binary = self.spark.read.csv(f"{data_path}/performance_binary.csv", 
                                         header=True, inferSchema=True)
        perf_multiclass = self.spark.read.csv(f"{data_path}/performance_multiclass.csv", 
                                             header=True, inferSchema=True)
        
        # Engagement datasets
        eng_binary = self.spark.read.csv(f"{data_path}/engagement_binary.csv", 
                                        header=True, inferSchema=True)
        eng_multiclass = self.spark.read.csv(f"{data_path}/engagement_multiclass.csv", 
                                            header=True, inferSchema=True)
        
        return {
            'performance_binary': perf_binary,
            'performance_multiclass': perf_multiclass,
            'engagement_binary': eng_binary,
            'engagement_multiclass': eng_multiclass
        }
    
    def data_quality_checks(self, df, dataset_name):
        """Perform comprehensive data quality checks"""
        print(f"\n=== Data Quality Report for {dataset_name} ===")
        
        # Basic statistics
        print(f"Total Records: {df.count()}")
        print(f"Total Columns: {len(df.columns)}")
        
        # Missing values analysis
        missing_analysis = df.select([
            (count(when(col(c).isNull(), c))/count('*')*100).alias(c) 
            for c in df.columns
        ]).collect()[0].asDict()
        
        print("\nMissing Values Percentage:")
        for col_name, missing_pct in missing_analysis.items():
            if missing_pct > 0:
                print(f"  {col_name}: {missing_pct:.2f}%")
        
        # Duplicate records
        total_records = df.count()
        unique_records = df.distinct().count()
        duplicate_pct = ((total_records - unique_records) / total_records) * 100
        print(f"\nDuplicate Records: {duplicate_pct:.2f}%")
        
        return missing_analysis
    
    def preprocess_data(self, df, target_column):
        """Advanced data preprocessing"""
        # Handle missing values
        df_processed = df.na.fill({
            col_name: df.select(mean(col(col_name))).collect()[0][0] 
            for col_name in df.columns 
            if df.schema[col_name].dataType in [IntegerType(), DoubleType(), FloatType()]
        })
        
        # Feature engineering
        df_processed = df_processed.withColumn(
            "engagement_score", 
            when(col("engagement_level") == "high", 3)
            .when(col("engagement_level") == "medium", 2)
            .otherwise(1)
        )
        
        # Create time-based features if timestamp exists
        if "timestamp" in df.columns:
            df_processed = df_processed.withColumn(
                "hour_of_day", hour(col("timestamp"))
            ).withColumn(
                "day_of_week", dayofweek(col("timestamp"))
            )
        
        return df_processed
    
    def build_ml_pipeline(self, df, target_column, feature_columns):
        """Build machine learning pipeline"""
        # Vector assembler
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features"
        )
        
        # Feature scaling
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features"
        )
        
        # Model selection based on target type
        if df.select(target_column).distinct().count() == 2:
            # Binary classification
            classifier = LogisticRegression(
                featuresCol="scaled_features",
                labelCol=target_column
            )
            evaluator = BinaryClassificationEvaluator(labelCol=target_column)
        else:
            # Multiclass classification
            classifier = RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol=target_column,
                numTrees=100
            )
            evaluator = MulticlassClassificationEvaluator(labelCol=target_column)
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler, classifier])
        
        return pipeline, evaluator
    
    def stream_processing(self):
        """Process streaming data from Kafka"""
        # Define schema for streaming data
        schema = StructType([
            StructField("student_id", StringType(), True),
            StructField("timestamp", LongType(), True),
            StructField("course_id", StringType(), True),
            StructField("activity_type", StringType(), True),
            StructField("duration_minutes", IntegerType(), True),
            StructField("score", DoubleType(), True),
            StructField("engagement_level", StringType(), True)
        ])
        
        # Read from Kafka
        df_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "student_activity") \
            .load()
        
        # Parse JSON data
        df_parsed = df_stream.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")
        
        # Real-time aggregations
        windowed_stats = df_parsed \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy(
                window(col("timestamp"), "5 minutes"),
                col("course_id")
            ) \
            .agg(
                count("*").alias("activity_count"),
                avg("duration_minutes").alias("avg_duration"),
                avg("score").alias("avg_score")
            )
        
        # Write to console (can be changed to other sinks)
        query = windowed_stats.writeStream \
            .outputMode("append") \
            .format("console") \
            .trigger(processingTime='30 seconds') \
            .start()
        
        return query
    
    def close(self):
        self.spark.stop()

if __name__ == "__main__":
    processor = StudentDataProcessor()
    
    # Load and process batch data
    datasets = processor.load_datasets()
    
    for name, df in datasets.items():
        print(f"\nProcessing {name}...")
        processor.data_quality_checks(df, name)
        
        # Preprocess data
        df_processed = processor.preprocess_data(df, "target_column")  # Adjust based on actual column
        
        # Build and train model
        feature_cols = [col for col in df_processed.columns if col != "target_column"]
        pipeline, evaluator = processor.build_ml_pipeline(df_processed, "target_column", feature_cols)
        
        # Train-test split
        train_df, test_df = df_processed.randomSplit([0.8, 0.2], seed=42)
        
        # Train model
        model = pipeline.fit(train_df)
        
        # Evaluate
        predictions = model.transform(test_df)
        accuracy = evaluator.evaluate(predictions)
        print(f"Model Accuracy for {name}: {accuracy:.4f}")
    
    # Start stream processing
    print("\nStarting stream processing...")
    stream_query = processor.stream_processing()
    
    try:
        stream_query.awaitTermination()
    except KeyboardInterrupt:
        stream_query.stop()
        processor.close()
