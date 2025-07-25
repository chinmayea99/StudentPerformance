from kafka import KafkaProducer
import json
import pandas as pd
import time
import random

class StudentDataProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
    
    def simulate_student_activity(self, topic='student_activity'):
        """Simulate real-time student activity data"""
        while True:
            # Generate synthetic student activity
            activity = {
                'student_id': f"STU_{random.randint(1000, 9999)}",
                'timestamp': int(time.time()),
                'course_id': f"COURSE_{random.randint(100, 999)}",
                'activity_type': random.choice(['login', 'video_watch', 'quiz_attempt', 'assignment_submit']),
                'duration_minutes': random.randint(1, 120),
                'score': random.uniform(0, 100) if random.random() > 0.3 else None,
                'engagement_level': random.choice(['low', 'medium', 'high'])
            }
            
            self.producer.send(topic, activity)
            time.sleep(random.uniform(0.1, 2.0))  # Random intervals
    
    def close(self):
        self.producer.close()

if __name__ == "__main__":
    producer = StudentDataProducer()
    try:
        producer.simulate_student_activity()
    except KeyboardInterrupt:
        producer.close()
