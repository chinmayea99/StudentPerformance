import psutil
import logging
from datetime import datetime
import json

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def check_system_health(self):
        """Monitor system resources"""
        health_metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
        
        # Log metrics
        self.logger.info(f"System Health: {json.dumps(health_metrics, indent=2)}")
        
        # Check for alerts
        self.check_alerts(health_metrics)
        
        return health_metrics
    
    def check_alerts(self, metrics):
        """Check for system alerts"""
        alerts = []
        
        if metrics['cpu_percent'] > 80:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > 85:
            alerts.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
        
        if metrics['disk_usage'] > 90:
            alerts.append(f"High disk usage: {metrics['disk_usage']:.1f}%")
        
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
        
        return alerts
    
    def monitor_spark_jobs(self, spark_context):
        """Monitor Spark job execution"""
        status_tracker = spark_context.statusTracker()
        
        job_metrics = {
            'active_jobs': len(status_tracker.getActiveJobIds()),
            'active_stages': len(status_tracker.getActiveStageIds()),
            'executor_infos': len(status_tracker.getExecutorInfos())
        }
        
        self.logger.info(f"Spark Metrics: {json.dumps(job_metrics, indent=2)}")
        return job_metrics

if __name__ == "__main__":
    monitor = SystemMonitor()
    while True:
        import time
        monitor.check_system_health()
        time.sleep(60)  # Check every minute
