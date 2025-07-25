SPARK_CONFIG = {
    # Memory management
    "spark.executor.memory": "4g",
    "spark.executor.cores": "4",
    "spark.executor.instances": "8",
    "spark.driver.memory": "2g",
    "spark.driver.cores": "2",
    
    # Adaptive query execution
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    
    # Caching and persistence
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    
    # Dynamic allocation
    "spark.dynamicAllocation.enabled": "true",
    "spark.dynamicAllocation.minExecutors": "2",
    "spark.dynamicAllocation.maxExecutors": "20",
    
    # Shuffle optimization
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.adaptive.shuffle.targetPostShuffleInputSize": "64MB"
}

def optimize_spark_session(spark_session):
    """Apply optimization configurations to Spark session"""
    for key, value in SPARK_CONFIG.items():
        spark_session.conf.set(key, value)
    return spark_session
