
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict

@dataclass
class RiskFeatures:
    """Data class to hold calculated risk features"""
    driver_id: str
    period_start: datetime
    period_end: datetime

    # Mileage features
    total_mileage: float
    avg_trip_distance: float

    # Speed features
    avg_speed: float
    max_speed: float
    speed_variance: float
    over_speed_events: int
    over_speed_ratio: float

    # Acceleration features
    harsh_braking_count: int
    harsh_acceleration_count: int
    avg_acceleration: float
    max_deceleration: float

    # Time-based features
    night_driving_ratio: float
    weekend_driving_ratio: float
    rush_hour_ratio: float

    # Contextual features
    weather_risk_score: float
    road_type_risk_score: float

    # Distraction features
    phone_usage_events: int
    phone_usage_ratio: float

    # Trip pattern features
    trip_frequency: int
    avg_trip_duration: float
    long_trip_ratio: float

class TelematicsDataProcessor:
    """
    Processes raw telematics data into features for risk assessment
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()

        # Risk scoring weights for contextual factors
        self.weather_weights = {
            'clear': 1.0,
            'rain': 1.3,
            'snow': 1.8,
            'fog': 1.5
        }

        self.road_type_weights = {
            'highway': 1.2,
            'urban': 1.0,
            'suburban': 0.9,
            'rural': 1.1
        }

    def _default_config(self) -> Dict:
        """Default configuration for data processing"""
        return {
            'harsh_braking_threshold': -3.0,  # m/s²
            'harsh_acceleration_threshold': 3.0,  # m/s²
            'speed_limit_tolerance': 5.0,  # mph
            'night_hours': (22, 6),  # 10 PM to 6 AM
            'rush_hours': [(7, 9), (17, 19)],  # Morning and evening rush
            'long_trip_threshold': 60,  # minutes
            'processing_window': 7,  # days for feature calculation
            'min_trip_duration': 2  # minimum minutes for valid trip
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor"""
        logger = logging.getLogger('TelematicsProcessor')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def clean_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate raw telematics data"""
        self.logger.info(f"Cleaning {len(raw_data)} raw data points")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(raw_data['timestamp']):
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

        # Remove invalid GPS coordinates
        raw_data = raw_data[
            (raw_data['latitude'].between(-90, 90)) &
            (raw_data['longitude'].between(-180, 180))
        ]

        # Remove impossible speeds (negative or too high)
        raw_data = raw_data[
            (raw_data['speed_mph'] >= 0) &
            (raw_data['speed_mph'] <= 200)
        ]

        # Remove trips that are too short
        trip_durations = raw_data.groupby('trip_id')['timestamp'].agg(['min', 'max'])
        trip_durations['duration_minutes'] = (trip_durations['max'] - trip_durations['min']).dt.total_seconds() / 60
        valid_trips = trip_durations[trip_durations['duration_minutes'] >= self.config['min_trip_duration']].index
        raw_data = raw_data[raw_data['trip_id'].isin(valid_trips)]

        # Sort by driver and timestamp
        raw_data = raw_data.sort_values(['driver_id', 'timestamp']).reset_index(drop=True)

        self.logger.info(f"Cleaned data: {len(raw_data)} points remaining")
        return raw_data

    def extract_trip_features(self, trip_data: pd.DataFrame) -> Dict:
        """Extract features from a single trip"""
        if len(trip_data) == 0:
            return {}

        trip_duration = (trip_data['timestamp'].max() - trip_data['timestamp'].min()).total_seconds() / 60
        total_distance = self._calculate_distance(trip_data)

        features = {
            'trip_id': trip_data['trip_id'].iloc[0],
            'driver_id': trip_data['driver_id'].iloc[0],
            'start_time': trip_data['timestamp'].min(),
            'end_time': trip_data['timestamp'].max(),
            'duration_minutes': trip_duration,
            'distance_miles': total_distance,

            # Speed features
            'avg_speed': trip_data['speed_mph'].mean(),
            'max_speed': trip_data['speed_mph'].max(),
            'speed_variance': trip_data['speed_mph'].var(),
            'over_speed_events': (trip_data['speed_mph'] > trip_data['speed_limit_mph'] + self.config['speed_limit_tolerance']).sum(),
            'over_speed_ratio': (trip_data['speed_mph'] > trip_data['speed_limit_mph'] + self.config['speed_limit_tolerance']).mean(),

            # Acceleration features
            'harsh_braking_count': (trip_data['acceleration_ms2'] < self.config['harsh_braking_threshold']).sum(),
            'harsh_acceleration_count': (trip_data['acceleration_ms2'] > self.config['harsh_acceleration_threshold']).sum(),
            'avg_acceleration': trip_data['acceleration_ms2'].mean(),
            'max_deceleration': trip_data['acceleration_ms2'].min(),

            # Context features
            'weather_condition': trip_data['weather'].mode().iloc[0] if not trip_data['weather'].empty else 'clear',
            'road_type': trip_data['road_type'].mode().iloc[0] if not trip_data['road_type'].empty else 'urban',
            'time_of_day': trip_data['time_of_day'].mode().iloc[0] if not trip_data['time_of_day'].empty else 'day',

            # Distraction features
            'phone_usage_events': trip_data['phone_usage'].sum(),
            'phone_usage_ratio': trip_data['phone_usage'].mean(),
        }

        # Time-based classifications
        features['is_night'] = self._is_night_trip(trip_data['timestamp'].iloc[0])
        features['is_weekend'] = self._is_weekend_trip(trip_data['timestamp'].iloc[0])
        features['is_rush_hour'] = self._is_rush_hour_trip(trip_data['timestamp'].iloc[0])
        features['is_long_trip'] = trip_duration > self.config['long_trip_threshold']

        return features

    def _calculate_distance(self, trip_data: pd.DataFrame) -> float:
        """Calculate total distance traveled using GPS coordinates"""
        if len(trip_data) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(trip_data)):
            lat1, lon1 = trip_data.iloc[i-1][['latitude', 'longitude']]
            lat2, lon2 = trip_data.iloc[i][['latitude', 'longitude']]

            # Haversine formula for distance between two GPS points
            R = 3959.0  # Earth's radius in miles

            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)

            a = (np.sin(dlat/2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c

            total_distance += distance

        return total_distance

    def _is_night_trip(self, timestamp: datetime) -> bool:
        """Check if trip occurs during night hours"""
        hour = timestamp.hour
        night_start, night_end = self.config['night_hours']

        if night_start > night_end:  # Night spans midnight
            return hour >= night_start or hour < night_end
        else:
            return night_start <= hour < night_end

    def _is_weekend_trip(self, timestamp: datetime) -> bool:
        """Check if trip occurs on weekend"""
        return timestamp.weekday() >= 5  # Saturday=5, Sunday=6

    def _is_rush_hour_trip(self, timestamp: datetime) -> bool:
        """Check if trip occurs during rush hour"""
        hour = timestamp.hour
        for rush_start, rush_end in self.config['rush_hours']:
            if rush_start <= hour < rush_end:
                return True
        return False

    def aggregate_driver_features(self, trip_features: List[Dict], 
                                 period_days: int = 7) -> RiskFeatures:
        """Aggregate trip-level features into driver-level risk features"""
        if not trip_features:
            raise ValueError("No trip features provided")

        df = pd.DataFrame(trip_features)
        driver_id = df['driver_id'].iloc[0]

        # Calculate aggregated features
        total_mileage = df['distance_miles'].sum()
        avg_trip_distance = df['distance_miles'].mean()

        # Speed features
        avg_speed = df['avg_speed'].mean()
        max_speed = df['max_speed'].max()
        speed_variance = df['speed_variance'].mean()
        over_speed_events = df['over_speed_events'].sum()
        over_speed_ratio = df['over_speed_ratio'].mean()

        # Acceleration features
        harsh_braking_count = df['harsh_braking_count'].sum()
        harsh_acceleration_count = df['harsh_acceleration_count'].sum()
        avg_acceleration = df['avg_acceleration'].mean()
        max_deceleration = df['max_deceleration'].min()

        # Time-based features
        night_driving_ratio = df['is_night'].mean()
        weekend_driving_ratio = df['is_weekend'].mean()
        rush_hour_ratio = df['is_rush_hour'].mean()

        # Contextual risk scores
        weather_risk_score = df['weather_condition'].apply(
            lambda x: self.weather_weights.get(x, 1.0)
        ).mean()

        road_type_risk_score = df['road_type'].apply(
            lambda x: self.road_type_weights.get(x, 1.0)
        ).mean()

        # Distraction features
        phone_usage_events = df['phone_usage_events'].sum()
        phone_usage_ratio = df['phone_usage_ratio'].mean()

        # Trip pattern features
        trip_frequency = len(df)
        avg_trip_duration = df['duration_minutes'].mean()
        long_trip_ratio = df['is_long_trip'].mean()

        # Create RiskFeatures object
        risk_features = RiskFeatures(
            driver_id=driver_id,
            period_start=df['start_time'].min(),
            period_end=df['end_time'].max(),
            total_mileage=total_mileage,
            avg_trip_distance=avg_trip_distance,
            avg_speed=avg_speed,
            max_speed=max_speed,
            speed_variance=speed_variance,
            over_speed_events=over_speed_events,
            over_speed_ratio=over_speed_ratio,
            harsh_braking_count=harsh_braking_count,
            harsh_acceleration_count=harsh_acceleration_count,
            avg_acceleration=avg_acceleration,
            max_deceleration=max_deceleration,
            night_driving_ratio=night_driving_ratio,
            weekend_driving_ratio=weekend_driving_ratio,
            rush_hour_ratio=rush_hour_ratio,
            weather_risk_score=weather_risk_score,
            road_type_risk_score=road_type_risk_score,
            phone_usage_events=phone_usage_events,
            phone_usage_ratio=phone_usage_ratio,
            trip_frequency=trip_frequency,
            avg_trip_duration=avg_trip_duration,
            long_trip_ratio=long_trip_ratio
        )

        return risk_features

    def process_driver_data(self, raw_data: pd.DataFrame, driver_id: str) -> RiskFeatures:
        """Process raw data for a single driver into risk features"""
        self.logger.info(f"Processing data for driver {driver_id}")

        # Filter data for this driver
        driver_data = raw_data[raw_data['driver_id'] == driver_id]

        if len(driver_data) == 0:
            raise ValueError(f"No data found for driver {driver_id}")

        # Clean the data
        clean_data = self.clean_raw_data(driver_data)

        # Extract trip-level features
        trip_features = []
        for trip_id in clean_data['trip_id'].unique():
            trip_data = clean_data[clean_data['trip_id'] == trip_id]
            trip_feature = self.extract_trip_features(trip_data)
            if trip_feature:
                trip_features.append(trip_feature)

        # Aggregate into driver-level features
        risk_features = self.aggregate_driver_features(trip_features)

        self.logger.info(f"Processed {len(trip_features)} trips for driver {driver_id}")
        return risk_features

    def batch_process_drivers(self, raw_data: pd.DataFrame) -> List[RiskFeatures]:
        """Process raw data for all drivers"""
        unique_drivers = raw_data['driver_id'].unique()
        all_features = []

        self.logger.info(f"Processing data for {len(unique_drivers)} drivers")

        for i, driver_id in enumerate(unique_drivers):
            try:
                driver_features = self.process_driver_data(raw_data, driver_id)
                all_features.append(driver_features)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(unique_drivers)} drivers")

            except Exception as e:
                self.logger.error(f"Error processing driver {driver_id}: {str(e)}")
                continue

        self.logger.info(f"Successfully processed {len(all_features)} drivers")
        return all_features

    def features_to_dataframe(self, risk_features_list: List[RiskFeatures]) -> pd.DataFrame:
        """Convert list of RiskFeatures to pandas DataFrame for ML"""
        feature_dicts = []

        for features in risk_features_list:
            feature_dict = {
                'driver_id': features.driver_id,
                'total_mileage': features.total_mileage,
                'avg_trip_distance': features.avg_trip_distance,
                'avg_speed': features.avg_speed,
                'max_speed': features.max_speed,
                'speed_variance': features.speed_variance,
                'over_speed_events': features.over_speed_events,
                'over_speed_ratio': features.over_speed_ratio,
                'harsh_braking_count': features.harsh_braking_count,
                'harsh_acceleration_count': features.harsh_acceleration_count,
                'avg_acceleration': features.avg_acceleration,
                'max_deceleration': features.max_deceleration,
                'night_driving_ratio': features.night_driving_ratio,
                'weekend_driving_ratio': features.weekend_driving_ratio,
                'rush_hour_ratio': features.rush_hour_ratio,
                'weather_risk_score': features.weather_risk_score,
                'road_type_risk_score': features.road_type_risk_score,
                'phone_usage_events': features.phone_usage_events,
                'phone_usage_ratio': features.phone_usage_ratio,
                'trip_frequency': features.trip_frequency,
                'avg_trip_duration': features.avg_trip_duration,
                'long_trip_ratio': features.long_trip_ratio
            }
            feature_dicts.append(feature_dict)

        return pd.DataFrame(feature_dicts)

class RealTimeProcessor:
    """
    Real-time processing for streaming telematics data
    """

    def __init__(self, batch_processor: TelematicsDataProcessor):
        self.batch_processor = batch_processor
        self.driver_buffers = defaultdict(list)
        self.buffer_size = 1000  # Number of points to buffer per driver

    def process_real_time_data(self, data_point: Dict) -> Optional[Dict]:
        """Process a single real-time data point"""
        driver_id = data_point['driver_id']

        # Add to buffer
        self.driver_buffers[driver_id].append(data_point)

        # Check if buffer is full
        if len(self.driver_buffers[driver_id]) >= self.buffer_size:
            # Process buffer
            buffer_df = pd.DataFrame(self.driver_buffers[driver_id])

            try:
                risk_features = self.batch_processor.process_driver_data(buffer_df, driver_id)

                # Clear buffer
                self.driver_buffers[driver_id] = []

                return {
                    'driver_id': driver_id,
                    'timestamp': datetime.now(),
                    'risk_features': risk_features,
                    'status': 'processed'
                }

            except Exception as e:
                logging.error(f"Real-time processing error for driver {driver_id}: {str(e)}")
                return None

        return None

