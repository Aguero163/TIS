"""
Streaming Data Pipeline for Real-time Telematics Processing
Handles high-volume, real-time telematics data ingestion and processing
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import boto3
from kafka import KafkaProducer, KafkaConsumer
import redis
import pandas as pd
from dataclasses import dataclass
import numpy as np

@dataclass
class TelematicsEvent:
    """Represents a single telematics data point"""
    driver_id: str
    trip_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_mph: float
    acceleration: float
    harsh_braking: bool
    harsh_acceleration: bool
    phone_usage: bool
    weather_condition: str
    road_type: str

class StreamProcessor:
    """
    Real-time stream processor for telematics data
    Handles high-throughput data ingestion and processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cloud services
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config['kafka']['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            batch_size=16384,  # Optimize for throughput
            linger_ms=100,     # Batch messages for efficiency
            compression_type='gzip'
        )
        
        self.redis_client = redis.Redis.from_url(config['redis']['url'])
        
        # AWS services
        self.kinesis_client = boto3.client('kinesis', region_name=config['aws']['region'])
        self.s3_client = boto3.client('s3', region_name=config['aws']['region'])
        
        # Processing state
        self.event_buffer = []
        self.buffer_size = config.get('buffer_size', 1000)
        
    async def ingest_telematics_data(self, event: TelematicsEvent) -> bool:
        """
        Ingest telematics data into the streaming pipeline
        """
        try:
            # Convert to dictionary for serialization
            event_data = {
                'driver_id': event.driver_id,
                'trip_id': event.trip_id,
                'timestamp': event.timestamp.isoformat(),
                'latitude': event.latitude,
                'longitude': event.longitude,
                'speed_mph': event.speed_mph,
                'acceleration': event.acceleration,
                'harsh_braking': event.harsh_braking,
                'harsh_acceleration': event.harsh_acceleration,
                'phone_usage': event.phone_usage,
                'weather_condition': event.weather_condition,
                'road_type': event.road_type,
                'ingestion_timestamp': datetime.now().isoformat()
            }
            
            # Send to Kafka for real-time processing
            self.kafka_producer.send(
                topic='telematics-events',
                key=event.driver_id,
                value=event_data
            )
            
            # Send to Kinesis for AWS processing
            self.kinesis_client.put_record(
                StreamName='telematics-data-stream',
                Data=json.dumps(event_data),
                PartitionKey=event.driver_id
            )
            
            # Cache recent data in Redis for fast access
            cache_key = f"driver:{event.driver_id}:recent"
            self.redis_client.lpush(cache_key, json.dumps(event_data))
            self.redis_client.ltrim(cache_key, 0, 100)  # Keep last 100 events
            self.redis_client.expire(cache_key, 3600)   # 1 hour expiry
            
            # Buffer for batch processing
            self.event_buffer.append(event_data)
            if len(self.event_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ingesting telematics data: {e}")
            return False
    
    async def _flush_buffer(self):
        """Flush buffered events to long-term storage"""
        if not self.event_buffer:
            return
        
        try:
            # Convert to DataFrame for efficient processing
            df = pd.DataFrame(self.event_buffer)
            
            # Partition by date for efficient storage
            current_date = datetime.now().strftime('%Y-%m-%d')
            partition_key = f"year={current_date[:4]}/month={current_date[5:7]}/day={current_date[8:10]}"
            
            # Save to S3 as Parquet for analytics
            file_key = f"telematics-raw-data/{partition_key}/batch_{datetime.now().strftime('%H%M%S')}.parquet"
            
            # Convert DataFrame to Parquet and upload
            parquet_buffer = df.to_parquet(index=False)
            self.s3_client.put_object(
                Bucket=self.config['aws']['s3_bucket'],
                Key=file_key,
                Body=parquet_buffer,
                ContentType='application/octet-stream'
            )
            
            self.logger.info(f"Flushed {len(self.event_buffer)} events to S3: {file_key}")
            self.event_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {e}")

class RealTimeRiskProcessor:
    """
    Real-time risk assessment processor
    Analyzes incoming telematics data for immediate risk evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis.from_url(config['redis']['url'])
        
        # Risk thresholds
        self.risk_thresholds = {
            'speed_limit_violation': 1.2,  # 20% over speed limit
            'harsh_braking_threshold': -8.0,  # m/s²
            'harsh_acceleration_threshold': 4.0,  # m/s²
            'consecutive_events_threshold': 3
        }
        
    async def process_real_time_event(self, event: TelematicsEvent) -> Dict[str, Any]:
        """
        Process individual telematics event for real-time risk assessment
        """
        risk_factors = {
            'driver_id': event.driver_id,
            'trip_id': event.trip_id,
            'timestamp': event.timestamp.isoformat(),
            'risk_events': [],
            'risk_score': 0.0,
            'immediate_alerts': []
        }
        
        try:
            # Check for harsh events
            if event.harsh_braking:
                risk_factors['risk_events'].append('harsh_braking')
                risk_factors['risk_score'] += 0.1
                risk_factors['immediate_alerts'].append('Harsh braking detected')
            
            if event.harsh_acceleration:
                risk_factors['risk_events'].append('harsh_acceleration')
                risk_factors['risk_score'] += 0.08
                risk_factors['immediate_alerts'].append('Harsh acceleration detected')
            
            # Check speed violations
            speed_limit = self._get_speed_limit(event.road_type)
            if event.speed_mph > speed_limit * self.risk_thresholds['speed_limit_violation']:
                risk_factors['risk_events'].append('speeding')
                risk_factors['risk_score'] += 0.15
                risk_factors['immediate_alerts'].append(f'Speeding: {event.speed_mph} mph in {speed_limit} mph zone')
            
            # Check phone usage
            if event.phone_usage:
                risk_factors['risk_events'].append('phone_usage')
                risk_factors['risk_score'] += 0.12
                risk_factors['immediate_alerts'].append('Phone usage while driving')
            
            # Check for patterns
            await self._check_risk_patterns(event.driver_id, risk_factors)
            
            # Store risk assessment
            await self._store_risk_assessment(risk_factors)
            
            # Send alerts if necessary
            if risk_factors['immediate_alerts']:
                await self._send_real_time_alerts(risk_factors)
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error processing real-time event: {e}")
            return risk_factors
    
    def _get_speed_limit(self, road_type: str) -> float:
        """Get speed limit based on road type"""
        speed_limits = {
            'residential': 25,
            'arterial': 35,
            'highway': 65,
            'interstate': 75
        }
        return speed_limits.get(road_type, 35)
    
    async def _check_risk_patterns(self, driver_id: str, risk_factors: Dict[str, Any]):
        """Check for concerning patterns in driver behavior"""
        
        # Get recent events from cache
        cache_key = f"driver:{driver_id}:recent"
        recent_events = self.redis_client.lrange(cache_key, 0, 10)
        
        if len(recent_events) < 3:
            return
        
        # Analyze recent events for patterns
        harsh_events_count = 0
        speeding_events_count = 0
        
        for event_json in recent_events:
            try:
                event_data = json.loads(event_json)
                if event_data.get('harsh_braking') or event_data.get('harsh_acceleration'):
                    harsh_events_count += 1
                if 'speeding' in event_data.get('risk_events', []):
                    speeding_events_count += 1
            except json.JSONDecodeError:
                continue
        
        # Check for concerning patterns
        if harsh_events_count >= self.risk_thresholds['consecutive_events_threshold']:
            risk_factors['risk_events'].append('pattern_harsh_driving')
            risk_factors['risk_score'] += 0.2
            risk_factors['immediate_alerts'].append('Pattern of harsh driving detected')
        
        if speeding_events_count >= self.risk_thresholds['consecutive_events_threshold']:
            risk_factors['risk_events'].append('pattern_speeding')
            risk_factors['risk_score'] += 0.25
            risk_factors['immediate_alerts'].append('Repeated speeding violations detected')
    
    async def _store_risk_assessment(self, risk_factors: Dict[str, Any]):
        """Store risk assessment for later analysis"""
        
        # Store in Redis for fast access
        risk_key = f"risk:{risk_factors['driver_id']}:{risk_factors['trip_id']}"
        self.redis_client.setex(
            risk_key,
            3600,  # 1 hour expiry
            json.dumps(risk_factors)
        )
        
        # Update driver's risk score rolling average
        driver_risk_key = f"driver_risk:{risk_factors['driver_id']}"
        current_scores = self.redis_client.lrange(driver_risk_key, 0, 99)
        scores = [float(score) for score in current_scores] + [risk_factors['risk_score']]
        
        # Keep rolling average of last 100 risk scores
        if len(scores) > 100:
            scores = scores[-100:]
        
        avg_risk = sum(scores) / len(scores)
        
        # Store updated average
        self.redis_client.delete(driver_risk_key)
        for score in scores:
            self.redis_client.lpush(driver_risk_key, score)
        self.redis_client.expire(driver_risk_key, 86400)  # 24 hours
        
        # Store average risk score
        self.redis_client.setex(
            f"driver_avg_risk:{risk_factors['driver_id']}",
            86400,
            avg_risk
        )
    
    async def _send_real_time_alerts(self, risk_factors: Dict[str, Any]):
        """Send real-time alerts for high-risk events"""
        
        alert_data = {
            'driver_id': risk_factors['driver_id'],
            'trip_id': risk_factors['trip_id'],
            'timestamp': risk_factors['timestamp'],
            'alerts': risk_factors['immediate_alerts'],
            'risk_score': risk_factors['risk_score'],
            'severity': 'high' if risk_factors['risk_score'] > 0.3 else 'medium'
        }
        
        # Send to notification system (could be SNS, email, push notifications)
        try:
            # Example: Send to AWS SNS for mobile push notifications
            sns_client = boto3.client('sns', region_name=self.config['aws']['region'])
            
            message = f"Risk Alert for Driver {risk_factors['driver_id']}: {', '.join(risk_factors['immediate_alerts'])}"
            
            sns_client.publish(
                TopicArn=self.config['aws']['sns_topic_arn'],
                Message=json.dumps(alert_data),
                Subject=f"Telematics Risk Alert - {alert_data['severity'].upper()}"
            )
            
            self.logger.info(f"Sent real-time alert for driver {risk_factors['driver_id']}")
            
        except Exception as e:
            self.logger.error(f"Error sending real-time alert: {e}")

class DataLakeManager:
    """
    Manages data lake operations for long-term storage and analytics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config['aws']['region'])
        self.glue_client = boto3.client('glue', region_name=config['aws']['region'])
        self.logger = logging.getLogger(__name__)
    
    async def create_data_partitions(self, date: str):
        """Create daily data partitions for efficient querying"""
        
        partition_path = f"year={date[:4]}/month={date[5:7]}/day={date[8:10]}"
        
        try:
            # Add partition to Glue Data Catalog
            self.glue_client.create_partition(
                DatabaseName='telematics_data_lake',
                TableName='telematics_events',
                PartitionInput={
                    'Values': [date[:4], date[5:7], date[8:10]],
                    'StorageDescriptor': {
                        'Columns': [
                            {'Name': 'driver_id', 'Type': 'string'},
                            {'Name': 'trip_id', 'Type': 'string'},
                            {'Name': 'timestamp', 'Type': 'timestamp'},
                            {'Name': 'latitude', 'Type': 'double'},
                            {'Name': 'longitude', 'Type': 'double'},
                            {'Name': 'speed_mph', 'Type': 'double'},
                            {'Name': 'harsh_braking', 'Type': 'boolean'},
                            {'Name': 'harsh_acceleration', 'Type': 'boolean'}
                        ],
                        'Location': f"s3://{self.config['aws']['s3_bucket']}/telematics-raw-data/{partition_path}/",
                        'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
                        }
                    }
                }
            )
            
            self.logger.info(f"Created partition for date: {date}")
            
        except Exception as e:
            if "AlreadyExistsException" not in str(e):
                self.logger.error(f"Error creating partition: {e}")

# Configuration example
STREAM_CONFIG = {
    'kafka': {
        'bootstrap_servers': ['kafka-cluster:9092']
    },
    'redis': {
        'url': 'redis://redis-cluster:6379'
    },
    'aws': {
        'region': 'us-west-2',
        's3_bucket': 'telematics-data-lake',
        'sns_topic_arn': 'arn:aws:sns:us-west-2:account:telematics-alerts'
    },
    'buffer_size': 1000
}

# Example usage
async def main():
    """Example usage of the streaming pipeline"""
    
    # Initialize processors
    stream_processor = StreamProcessor(STREAM_CONFIG)
    risk_processor = RealTimeRiskProcessor(STREAM_CONFIG)
    
    # Example telematics event
    event = TelematicsEvent(
        driver_id="driver_001",
        trip_id="trip_12345",
        timestamp=datetime.now(),
        latitude=40.7128,
        longitude=-74.0060,
        speed_mph=45.0,
        acceleration=2.5,
        harsh_braking=False,
        harsh_acceleration=True,
        phone_usage=False,
        weather_condition="clear",
        road_type="arterial"
    )
    
    # Process the event
    await stream_processor.ingest_telematics_data(event)
    risk_assessment = await risk_processor.process_real_time_event(event)
    
    print(f"Risk assessment: {risk_assessment}")

if __name__ == "__main__":
    asyncio.run(main())