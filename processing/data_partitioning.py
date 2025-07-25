class DataPartitionStrategy:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def partition_by_date(self, df, date_column, partition_format="yyyy-MM"):
        """Partition data by date for efficient querying"""
        return df.withColumn("partition_date", 
                           date_format(col(date_column), partition_format)) \
                 .repartition(col("partition_date"))
    
    def partition_by_course(self, df, course_column):
        """Partition data by course for course-specific analytics"""
        return df.repartition(col(course_column))
    
    def optimize_partitions(self, df, target_partition_size_mb=128):
        """Optimize partition size for better performance"""
        total_size_mb = df.count() * len(df.columns) * 8 / (1024 * 1024)  # Rough estimate
        optimal_partitions = max(1, int(total_size_mb / target_partition_size_mb))
        return df.repartition(optimal_partitions)
