
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Tuple
import uuid

class TelematicsDataSimulator:
    """
    Simulates realistic telematics data for auto insurance applications
    """

    def __init__(self):
        self.driver_profiles = ['safe', 'moderate', 'aggressive', 'elderly', 'young']
        self.vehicle_types = ['sedan', 'suv', 'truck', 'compact', 'luxury']
        self.road_types = ['highway', 'urban', 'suburban', 'rural']
        self.weather_conditions = ['clear', 'rain', 'snow', 'fog']

    def generate_driver_profile(self) -> Dict:
        """Generate a driver profile with static characteristics"""
        profile_type = random.choice(self.driver_profiles)

        # Define profile characteristics
        profiles = {
            'safe': {
                'avg_speed_ratio': 0.85,  # Ratio to speed limit
                'harsh_braking_prob': 0.02,
                'harsh_acceleration_prob': 0.01,
                'phone_usage_prob': 0.01,
                'night_driving_ratio': 0.1,
                'annual_mileage': random.randint(8000, 15000)
            },
            'moderate': {
                'avg_speed_ratio': 0.95,
                'harsh_braking_prob': 0.05,
                'harsh_acceleration_prob': 0.03,
                'phone_usage_prob': 0.05,
                'night_driving_ratio': 0.15,
                'annual_mileage': random.randint(12000, 20000)
            },
            'aggressive': {
                'avg_speed_ratio': 1.15,
                'harsh_braking_prob': 0.12,
                'harsh_acceleration_prob': 0.10,
                'phone_usage_prob': 0.15,
                'night_driving_ratio': 0.25,
                'annual_mileage': random.randint(18000, 30000)
            },
            'elderly': {
                'avg_speed_ratio': 0.8,
                'harsh_braking_prob': 0.03,
                'harsh_acceleration_prob': 0.02,
                'phone_usage_prob': 0.01,
                'night_driving_ratio': 0.05,
                'annual_mileage': random.randint(5000, 10000)
            },
            'young': {
                'avg_speed_ratio': 1.1,
                'harsh_braking_prob': 0.08,
                'harsh_acceleration_prob': 0.07,
                'phone_usage_prob': 0.20,
                'night_driving_ratio': 0.30,
                'annual_mileage': random.randint(15000, 25000)
            }
        }

        driver_id = str(uuid.uuid4())
        age = self._generate_age_for_profile(profile_type)

        return {
            'driver_id': driver_id,
            'profile_type': profile_type,
            'age': age,
            'gender': random.choice(['M', 'F']),
            'years_licensed': min(age - 16, random.randint(1, 40)),
            'vehicle_type': random.choice(self.vehicle_types),
            'vehicle_year': random.randint(2015, 2025),
            **profiles[profile_type]
        }

    def _generate_age_for_profile(self, profile_type: str) -> int:
        """Generate age based on profile type"""
        age_ranges = {
            'safe': (30, 55),
            'moderate': (25, 45),
            'aggressive': (20, 35),
            'elderly': (65, 80),
            'young': (18, 25)
        }
        min_age, max_age = age_ranges[profile_type]
        return random.randint(min_age, max_age)

    def generate_trip_data(self, driver_profile: Dict, duration_minutes: int = 30) -> pd.DataFrame:
        """Generate detailed trip data for a driver"""

        # Generate base trip parameters
        speed_limit = random.randint(25, 70)  # mph
        road_type = random.choice(self.road_types)
        weather = random.choice(self.weather_conditions)
        time_of_day = random.choice(['morning', 'afternoon', 'evening', 'night'])

        # Calculate number of data points (1 Hz frequency)
        num_points = duration_minutes * 60

        # Generate time series
        start_time = datetime.now() - timedelta(days=random.randint(0, 365))
        timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]

        # Generate realistic speed profile
        base_speed = speed_limit * driver_profile['avg_speed_ratio']
        speeds = self._generate_speed_profile(base_speed, num_points, driver_profile)

        # Generate acceleration data
        accelerations = self._calculate_acceleration(speeds)

        # Generate GPS coordinates (simplified simulation)
        start_lat, start_lon = 40.7589 + random.uniform(-0.5, 0.5), -73.9851 + random.uniform(-0.5, 0.5)
        gps_data = self._generate_gps_trajectory(start_lat, start_lon, speeds, num_points)

        # Generate events
        events = self._generate_events(driver_profile, num_points)

        # Create DataFrame
        trip_data = pd.DataFrame({
            'timestamp': timestamps,
            'driver_id': driver_profile['driver_id'],
            'trip_id': str(uuid.uuid4()),
            'latitude': [coord[0] for coord in gps_data],
            'longitude': [coord[1] for coord in gps_data],
            'speed_mph': speeds,
            'acceleration_ms2': accelerations,
            'speed_limit_mph': [speed_limit] * num_points,
            'road_type': [road_type] * num_points,
            'weather': [weather] * num_points,
            'time_of_day': [time_of_day] * num_points,
            'harsh_braking': events['harsh_braking'],
            'harsh_acceleration': events['harsh_acceleration'],
            'phone_usage': events['phone_usage'],
            'over_speed_limit': [1 if s > speed_limit else 0 for s in speeds]
        })

        return trip_data

    def _generate_speed_profile(self, base_speed: float, num_points: int, driver_profile: Dict) -> List[float]:
        """Generate realistic speed profile with variations"""
        speeds = []
        current_speed = 0  # Start from rest

        # Acceleration phase (first 20% of trip)
        accel_points = int(num_points * 0.2)
        for i in range(accel_points):
            current_speed = min(base_speed, current_speed + random.uniform(0.5, 2.0))
            speeds.append(max(0, current_speed))

        # Cruise phase (middle 60% of trip)
        cruise_points = int(num_points * 0.6)
        for i in range(cruise_points):
            # Add random variation
            variation = random.uniform(-5, 5) * (1 + driver_profile['harsh_acceleration_prob'] * 10)
            current_speed = max(0, min(base_speed * 1.3, current_speed + variation))
            speeds.append(current_speed)

        # Deceleration phase (last 20% of trip)
        decel_points = num_points - len(speeds)
        for i in range(decel_points):
            current_speed = max(0, current_speed - random.uniform(0.5, 2.0))
            speeds.append(current_speed)

        return speeds

    def _calculate_acceleration(self, speeds: List[float]) -> List[float]:
        """Calculate acceleration from speed data"""
        accelerations = [0]  # First point has no acceleration

        for i in range(1, len(speeds)):
            # Convert mph to m/s and calculate acceleration
            speed_diff_ms = (speeds[i] - speeds[i-1]) * 0.44704  # mph to m/s
            acceleration = speed_diff_ms  # Per second
            accelerations.append(acceleration)

        return accelerations

    def _generate_gps_trajectory(self, start_lat: float, start_lon: float, 
                                speeds: List[float], num_points: int) -> List[Tuple[float, float]]:
        """Generate GPS trajectory based on speed profile"""
        coords = [(start_lat, start_lon)]
        current_lat, current_lon = start_lat, start_lon

        # Rough conversion: 1 degree ≈ 69 miles
        for i in range(1, num_points):
            # Calculate movement based on speed (simplified)
            speed_ms = speeds[i] * 0.44704  # mph to m/s
            distance_m = speed_ms  # Distance in 1 second

            # Random direction with some consistency
            bearing = random.uniform(-0.1, 0.1)  # Small random changes in direction

            # Convert to lat/lon changes
            lat_change = (distance_m / 111000) * np.cos(bearing)  # 1 degree lat ≈ 111km
            lon_change = (distance_m / (111000 * np.cos(np.radians(current_lat)))) * np.sin(bearing)

            current_lat += lat_change
            current_lon += lon_change
            coords.append((current_lat, current_lon))

        return coords

    def _generate_events(self, driver_profile: Dict, num_points: int) -> Dict[str, List[int]]:
        """Generate driving events based on driver profile"""
        events = {
            'harsh_braking': [],
            'harsh_acceleration': [],
            'phone_usage': []
        }

        for i in range(num_points):
            # Generate events based on probabilities
            events['harsh_braking'].append(
                1 if random.random() < driver_profile['harsh_braking_prob'] / 3600 else 0
            )
            events['harsh_acceleration'].append(
                1 if random.random() < driver_profile['harsh_acceleration_prob'] / 3600 else 0
            )
            events['phone_usage'].append(
                1 if random.random() < driver_profile['phone_usage_prob'] / 3600 else 0
            )

        return events

    def generate_multiple_trips(self, num_drivers: int = 100, trips_per_driver: int = 50) -> pd.DataFrame:
        """Generate data for multiple drivers and trips"""
        all_trips = []

        print(f"Generating telematics data for {num_drivers} drivers...")

        for i in range(num_drivers):
            driver_profile = self.generate_driver_profile()

            for j in range(trips_per_driver):
                # Vary trip duration
                duration = random.randint(10, 60)  # 10-60 minutes
                trip_data = self.generate_trip_data(driver_profile, duration)
                all_trips.append(trip_data)

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} drivers...")

        combined_data = pd.concat(all_trips, ignore_index=True)
        print(f"Generated {len(combined_data)} data points across {num_drivers * trips_per_driver} trips")

        return combined_data

# Save the simulator to the appropriate directory
