"""
External Data Integration Module
Integrates additional risk-correlated data sources (crime data, traffic accidents, etc.)
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import asyncio
import aiohttp
from dataclasses import dataclass
import numpy as np
from geopy.distance import geodesic

@dataclass
class CrimeIncident:
    """Represents a crime incident near a route"""
    incident_type: str
    severity: str
    latitude: float
    longitude: float
    date: datetime
    distance_from_route: float

@dataclass
class TrafficAccident:
    """Represents a traffic accident"""
    accident_type: str
    severity: str
    latitude: float
    longitude: float
    date: datetime
    weather_conditions: str
    road_conditions: str
    casualties: int

class ExternalDataIntegrator:
    """
    Integrates external data sources for enhanced risk assessment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # API endpoints and keys
        self.crime_data_api = config.get('crime_data_api_url')
        self.traffic_api_key = config.get('traffic_api_key')
        self.weather_api_key = config.get('weather_api_key')
        self.census_api_key = config.get('census_api_key')
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_expiry = {}
    
    async def get_crime_risk_score(self, latitude: float, longitude: float, 
                                 radius_miles: float = 5.0) -> Dict[str, Any]:
        """
        Get crime risk score for a specific location
        """
        try:
            # Check cache first
            cache_key = f"crime_risk_{latitude}_{longitude}_{radius_miles}"
            if self._is_cached_valid(cache_key):
                return self.cache[cache_key]
            
            # Query crime data API (example using open data APIs)
            crime_data = await self._fetch_crime_data(latitude, longitude, radius_miles)
            
            # Calculate crime risk score
            risk_factors = self._analyze_crime_data(crime_data)
            
            result = {
                'location': {'latitude': latitude, 'longitude': longitude},
                'radius_miles': radius_miles,
                'crime_risk_score': risk_factors['overall_score'],
                'risk_breakdown': {
                    'violent_crime_risk': risk_factors['violent_score'],
                    'property_crime_risk': risk_factors['property_score'],
                    'vehicle_theft_risk': risk_factors['vehicle_theft_score'],
                    'frequency_factor': risk_factors['frequency_factor']
                },
                'recent_incidents_count': len(crime_data),
                'risk_level': self._categorize_risk(risk_factors['overall_score']),
                'last_updated': datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache_result(cache_key, result, hours=24)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching crime risk data: {e}")
            return self._default_crime_risk()
    
    async def get_traffic_accident_history(self, route_coordinates: List[tuple], 
                                         time_window_days: int = 365) -> Dict[str, Any]:
        """
        Get traffic accident history along a route
        """
        try:
            accident_data = []
            
            # Fetch accident data for route segments
            for i in range(len(route_coordinates) - 1):
                start_coord = route_coordinates[i]
                end_coord = route_coordinates[i + 1]
                
                segment_accidents = await self._fetch_traffic_accidents(
                    start_coord, end_coord, time_window_days
                )
                accident_data.extend(segment_accidents)
            
            # Analyze accident patterns
            analysis = self._analyze_traffic_accidents(accident_data, route_coordinates)
            
            return {
                'route_accident_score': analysis['route_score'],
                'accident_density': analysis['accidents_per_mile'],
                'severity_distribution': analysis['severity_breakdown'],
                'seasonal_patterns': analysis['seasonal_risk'],
                'weather_correlation': analysis['weather_risk'],
                'high_risk_segments': analysis['danger_zones'],
                'total_accidents': len(accident_data),
                'analysis_period_days': time_window_days,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching traffic accident data: {e}")
            return self._default_accident_data()
    
    async def get_demographic_risk_factors(self, zip_code: str) -> Dict[str, Any]:
        """
        Get demographic risk factors for insurance pricing
        """
        try:
            # Use US Census API for demographic data
            demographic_data = await self._fetch_census_data(zip_code)
            
            risk_analysis = {
                'zip_code': zip_code,
                'population_density': demographic_data.get('population_density', 0),
                'median_age': demographic_data.get('median_age', 0),
                'median_income': demographic_data.get('median_income', 0),
                'unemployment_rate': demographic_data.get('unemployment_rate', 0),
                'vehicle_density': demographic_data.get('vehicles_per_capita', 0),
                'education_level': demographic_data.get('college_educated_pct', 0),
                'risk_multiplier': self._calculate_demographic_risk(demographic_data),
                'risk_factors': {
                    'income_factor': self._income_risk_factor(demographic_data.get('median_income', 50000)),
                    'density_factor': self._density_risk_factor(demographic_data.get('population_density', 1000)),
                    'age_factor': self._age_demographic_factor(demographic_data.get('median_age', 35))
                }
            }
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error fetching demographic data: {e}")
            return self._default_demographic_data()
    
    async def get_contextual_risk_score(self, latitude: float, longitude: float,
                                      timestamp: datetime, route: List[tuple] = None) -> Dict[str, Any]:
        """
        Get comprehensive contextual risk score combining all external data sources
        """
        try:
            # Fetch all data sources concurrently
            tasks = [
                self.get_crime_risk_score(latitude, longitude),
                self.get_weather_risk_score(latitude, longitude, timestamp),
                self.get_traffic_congestion_risk(latitude, longitude, timestamp)
            ]
            
            if route:
                tasks.append(self.get_traffic_accident_history(route))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            crime_risk = results[0] if not isinstance(results[0], Exception) else self._default_crime_risk()
            weather_risk = results[1] if not isinstance(results[1], Exception) else self._default_weather_risk()
            traffic_risk = results[2] if not isinstance(results[2], Exception) else self._default_traffic_risk()
            
            accident_risk = None
            if len(results) > 3:
                accident_risk = results[3] if not isinstance(results[3], Exception) else self._default_accident_data()
            
            # Calculate combined contextual risk score
            contextual_score = self._calculate_combined_risk_score({
                'crime': crime_risk,
                'weather': weather_risk,
                'traffic': traffic_risk,
                'accidents': accident_risk
            })
            
            return {
                'location': {'latitude': latitude, 'longitude': longitude},
                'timestamp': timestamp.isoformat(),
                'contextual_risk_score': contextual_score['overall_score'],
                'risk_components': {
                    'crime_risk': crime_risk['crime_risk_score'],
                    'weather_risk': weather_risk['weather_risk_score'],
                    'traffic_risk': traffic_risk['congestion_risk_score'],
                    'accident_risk': accident_risk['route_accident_score'] if accident_risk else 0.5
                },
                'risk_weights': contextual_score['weights'],
                'risk_level': self._categorize_risk(contextual_score['overall_score']),
                'recommendations': self._generate_risk_recommendations(contextual_score),
                'confidence': contextual_score['confidence'],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating contextual risk score: {e}")
            return self._default_contextual_risk()
    
    async def _fetch_crime_data(self, latitude: float, longitude: float, 
                              radius_miles: float) -> List[CrimeIncident]:
        """Fetch crime data from external APIs"""
        
        crime_incidents = []
        
        try:
            # Example: Chicago Crime Data API (replace with actual API)
            async with aiohttp.ClientSession() as session:
                # Convert radius to approximate lat/lng bounds
                lat_delta = radius_miles / 69.0  # Approximate miles per degree
                lng_delta = radius_miles / (69.0 * np.cos(np.radians(latitude)))
                
                params = {
                    '$where': f'within_box(location, {latitude - lat_delta}, {longitude - lng_delta}, {latitude + lat_delta}, {longitude + lng_delta})',
                    '$limit': 1000,
                    '$order': 'date DESC'
                }
                
                # Mock crime data (replace with actual API call)
                # async with session.get(self.crime_data_api, params=params) as response:
                #     data = await response.json()
                
                # Simulate crime data
                for i in range(np.random.randint(5, 25)):
                    incident = CrimeIncident(
                        incident_type=np.random.choice(['theft', 'assault', 'burglary', 'vandalism']),
                        severity=np.random.choice(['minor', 'moderate', 'severe']),
                        latitude=latitude + np.random.normal(0, lat_delta/3),
                        longitude=longitude + np.random.normal(0, lng_delta/3),
                        date=datetime.now() - timedelta(days=np.random.randint(1, 365)),
                        distance_from_route=radius_miles * np.random.random()
                    )
                    crime_incidents.append(incident)
                    
        except Exception as e:
            self.logger.error(f"Error fetching crime data: {e}")
        
        return crime_incidents
    
    async def _fetch_traffic_accidents(self, start_coord: tuple, end_coord: tuple, 
                                     days: int) -> List[TrafficAccident]:
        """Fetch traffic accident data for route segment"""
        
        accidents = []
        
        try:
            # Mock traffic accident data (replace with actual API)
            for i in range(np.random.randint(0, 5)):
                # Random point along the route segment
                t = np.random.random()
                accident_lat = start_coord[0] + t * (end_coord[0] - start_coord[0])
                accident_lng = start_coord[1] + t * (end_coord[1] - start_coord[1])
                
                accident = TrafficAccident(
                    accident_type=np.random.choice(['collision', 'single_vehicle', 'pedestrian']),
                    severity=np.random.choice(['minor', 'major', 'fatal']),
                    latitude=accident_lat,
                    longitude=accident_lng,
                    date=datetime.now() - timedelta(days=np.random.randint(1, days)),
                    weather_conditions=np.random.choice(['clear', 'rain', 'snow', 'fog']),
                    road_conditions=np.random.choice(['dry', 'wet', 'icy']),
                    casualties=np.random.randint(0, 4)
                )
                accidents.append(accident)
                
        except Exception as e:
            self.logger.error(f"Error fetching traffic accident data: {e}")
        
        return accidents
    
    async def _fetch_census_data(self, zip_code: str) -> Dict[str, Any]:
        """Fetch demographic data from US Census API"""
        
        try:
            # Mock census data (replace with actual Census API call)
            return {
                'population_density': np.random.randint(500, 5000),
                'median_age': np.random.randint(25, 55),
                'median_income': np.random.randint(30000, 120000),
                'unemployment_rate': np.random.uniform(0.03, 0.15),
                'vehicles_per_capita': np.random.uniform(0.6, 1.2),
                'college_educated_pct': np.random.uniform(0.15, 0.7)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching census data: {e}")
            return {}
    
    async def get_weather_risk_score(self, latitude: float, longitude: float, 
                                   timestamp: datetime) -> Dict[str, Any]:
        """Get weather-based risk score"""
        
        try:
            # Mock weather data (replace with actual weather API)
            weather_conditions = np.random.choice(['clear', 'cloudy', 'rain', 'snow', 'fog'])
            visibility = np.random.uniform(0.1, 10.0)  # miles
            precipitation = np.random.uniform(0, 2.0)  # inches
            wind_speed = np.random.uniform(0, 40)  # mph
            
            # Calculate weather risk score
            risk_score = 0.0
            
            if weather_conditions in ['rain', 'snow']:
                risk_score += 0.3
            elif weather_conditions == 'fog':
                risk_score += 0.4
            
            if visibility < 1.0:
                risk_score += 0.2
            elif visibility < 3.0:
                risk_score += 0.1
            
            if precipitation > 0.5:
                risk_score += 0.2
            
            if wind_speed > 25:
                risk_score += 0.1
            
            risk_score = min(risk_score, 1.0)
            
            return {
                'weather_risk_score': risk_score,
                'conditions': weather_conditions,
                'visibility_miles': visibility,
                'precipitation_inches': precipitation,
                'wind_speed_mph': wind_speed,
                'risk_level': self._categorize_risk(risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching weather data: {e}")
            return self._default_weather_risk()
    
    async def get_traffic_congestion_risk(self, latitude: float, longitude: float, 
                                        timestamp: datetime) -> Dict[str, Any]:
        """Get traffic congestion risk score"""
        
        try:
            # Mock traffic data (replace with actual traffic API)
            congestion_level = np.random.uniform(0, 1)
            average_speed = np.random.uniform(15, 60)  # mph
            typical_speed = np.random.uniform(25, 65)  # mph
            
            # Calculate congestion risk
            speed_ratio = average_speed / typical_speed
            congestion_risk = 1.0 - speed_ratio
            
            # Adjust for time of day
            hour = timestamp.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                congestion_risk *= 1.3
            
            congestion_risk = min(congestion_risk, 1.0)
            
            return {
                'congestion_risk_score': congestion_risk,
                'congestion_level': congestion_level,
                'average_speed_mph': average_speed,
                'typical_speed_mph': typical_speed,
                'speed_ratio': speed_ratio,
                'is_rush_hour': 7 <= hour <= 9 or 17 <= hour <= 19,
                'risk_level': self._categorize_risk(congestion_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching traffic data: {e}")
            return self._default_traffic_risk()
    
    def _analyze_crime_data(self, crime_data: List[CrimeIncident]) -> Dict[str, float]:
        """Analyze crime data to calculate risk factors"""
        
        if not crime_data:
            return {'overall_score': 0.1, 'violent_score': 0.1, 'property_score': 0.1, 
                   'vehicle_theft_score': 0.1, 'frequency_factor': 0.1}
        
        # Categorize crimes
        violent_crimes = len([c for c in crime_data if c.incident_type in ['assault', 'robbery']])
        property_crimes = len([c for c in crime_data if c.incident_type in ['theft', 'burglary', 'vandalism']])
        vehicle_crimes = len([c for c in crime_data if 'vehicle' in c.incident_type.lower()])
        
        total_crimes = len(crime_data)
        
        # Calculate risk scores
        violent_score = min(violent_crimes / 20.0, 1.0)  # Max at 20 incidents
        property_score = min(property_crimes / 50.0, 1.0)  # Max at 50 incidents
        vehicle_theft_score = min(vehicle_crimes / 10.0, 1.0)  # Max at 10 incidents
        frequency_factor = min(total_crimes / 100.0, 1.0)  # Max at 100 incidents
        
        # Recent crimes get higher weight
        recent_crimes = len([c for c in crime_data if (datetime.now() - c.date).days <= 30])
        recency_factor = min(recent_crimes / 20.0, 1.0)
        
        overall_score = (
            violent_score * 0.4 +
            property_score * 0.3 +
            vehicle_theft_score * 0.2 +
            frequency_factor * 0.1 +
            recency_factor * 0.2
        ) / 1.2  # Normalize
        
        return {
            'overall_score': overall_score,
            'violent_score': violent_score,
            'property_score': property_score,
            'vehicle_theft_score': vehicle_theft_score,
            'frequency_factor': frequency_factor
        }
    
    def _analyze_traffic_accidents(self, accidents: List[TrafficAccident], 
                                 route: List[tuple]) -> Dict[str, Any]:
        """Analyze traffic accident data for route risk assessment"""
        
        if not accidents:
            return {
                'route_score': 0.1,
                'accidents_per_mile': 0,
                'severity_breakdown': {'minor': 0, 'major': 0, 'fatal': 0},
                'seasonal_risk': 0.1,
                'weather_risk': 0.1,
                'danger_zones': []
            }
        
        # Calculate route distance
        route_distance = 0
        for i in range(len(route) - 1):
            route_distance += geodesic(route[i], route[i+1]).miles
        
        # Accident density
        accident_density = len(accidents) / max(route_distance, 1)
        
        # Severity analysis
        severity_counts = {'minor': 0, 'major': 0, 'fatal': 0}
        for accident in accidents:
            severity_counts[accident.severity] += 1
        
        # Seasonal patterns (simplified)
        seasonal_accidents = [a for a in accidents if (datetime.now() - a.date).days <= 90]
        seasonal_risk = min(len(seasonal_accidents) / 10.0, 1.0)
        
        # Weather correlation
        weather_accidents = [a for a in accidents if a.weather_conditions in ['rain', 'snow', 'fog']]
        weather_risk = min(len(weather_accidents) / max(len(accidents), 1) * 2, 1.0)
        
        # Calculate overall route score
        severity_weight = (
            severity_counts['minor'] * 0.1 +
            severity_counts['major'] * 0.3 +
            severity_counts['fatal'] * 1.0
        )
        
        route_score = min(
            (accident_density * 0.3 + 
             severity_weight / max(len(accidents), 1) * 0.4 +
             seasonal_risk * 0.2 +
             weather_risk * 0.1), 1.0
        )
        
        return {
            'route_score': route_score,
            'accidents_per_mile': accident_density,
            'severity_breakdown': severity_counts,
            'seasonal_risk': seasonal_risk,
            'weather_risk': weather_risk,
            'danger_zones': []  # Could identify high-risk segments
        }
    
    def _calculate_demographic_risk(self, demographic_data: Dict[str, Any]) -> float:
        """Calculate risk multiplier based on demographic factors"""
        
        risk_multiplier = 1.0
        
        # Income factor
        median_income = demographic_data.get('median_income', 50000)
        if median_income < 30000:
            risk_multiplier *= 1.2
        elif median_income > 80000:
            risk_multiplier *= 0.9
        
        # Population density factor
        density = demographic_data.get('population_density', 1000)
        if density > 3000:
            risk_multiplier *= 1.1
        elif density < 500:
            risk_multiplier *= 0.95
        
        # Age factor
        median_age = demographic_data.get('median_age', 35)
        if median_age < 30:
            risk_multiplier *= 1.05
        elif median_age > 45:
            risk_multiplier *= 0.95
        
        return min(max(risk_multiplier, 0.7), 1.5)  # Bound between 0.7 and 1.5
    
    def _calculate_combined_risk_score(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined contextual risk score from all sources"""
        
        # Risk weights
        weights = {
            'crime': 0.25,
            'weather': 0.30,
            'traffic': 0.25,
            'accidents': 0.20
        }
        
        # Extract individual scores
        crime_score = risk_data['crime']['crime_risk_score']
        weather_score = risk_data['weather']['weather_risk_score']
        traffic_score = risk_data['traffic']['congestion_risk_score']
        accident_score = risk_data['accidents']['route_accident_score'] if risk_data['accidents'] else 0.5
        
        # Calculate weighted score
        overall_score = (
            crime_score * weights['crime'] +
            weather_score * weights['weather'] +
            traffic_score * weights['traffic'] +
            accident_score * weights['accidents']
        )
        
        # Calculate confidence based on data availability
        confidence = 0.8  # Base confidence
        if not risk_data['accidents']:
            confidence -= 0.1
        
        return {
            'overall_score': overall_score,
            'weights': weights,
            'confidence': confidence
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk score into levels"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _generate_risk_recommendations(self, contextual_score: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        score = contextual_score['overall_score']
        
        if score > 0.7:
            recommendations.append("Consider avoiding this area during peak hours")
            recommendations.append("Use extra caution due to high risk factors")
        elif score > 0.4:
            recommendations.append("Moderate risk area - drive defensively")
        
        return recommendations
    
    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        if cache_key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any, hours: int = 24):
        """Cache result with expiration"""
        self.cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=hours)
    
    # Default data methods for error handling
    def _default_crime_risk(self) -> Dict[str, Any]:
        return {
            'crime_risk_score': 0.3,
            'risk_level': 'medium',
            'risk_breakdown': {
                'violent_crime_risk': 0.2,
                'property_crime_risk': 0.3,
                'vehicle_theft_risk': 0.25,
                'frequency_factor': 0.3
            }
        }
    
    def _default_weather_risk(self) -> Dict[str, Any]:
        return {
            'weather_risk_score': 0.2,
            'risk_level': 'low',
            'conditions': 'unknown'
        }
    
    def _default_traffic_risk(self) -> Dict[str, Any]:
        return {
            'congestion_risk_score': 0.3,
            'risk_level': 'medium'
        }
    
    def _default_accident_data(self) -> Dict[str, Any]:
        return {
            'route_accident_score': 0.4,
            'accident_density': 0,
            'severity_distribution': {'minor': 0, 'major': 0, 'fatal': 0}
        }
    
    def _default_demographic_data(self) -> Dict[str, Any]:
        return {
            'risk_multiplier': 1.0,
            'zip_code': 'unknown',
            'risk_factors': {
                'income_factor': 1.0,
                'density_factor': 1.0,
                'age_factor': 1.0
            }
        }
    
    def _default_contextual_risk(self) -> Dict[str, Any]:
        return {
            'contextual_risk_score': 0.5,
            'risk_level': 'medium',
            'confidence': 0.6,
            'risk_components': {
                'crime_risk': 0.3,
                'weather_risk': 0.2,
                'traffic_risk': 0.3,
                'accident_risk': 0.4
            }
        }
    
    def _income_risk_factor(self, income: float) -> float:
        """Calculate risk factor based on income"""
        if income < 30000:
            return 1.2
        elif income > 80000:
            return 0.9
        else:
            return 1.0
    
    def _density_risk_factor(self, density: float) -> float:
        """Calculate risk factor based on population density"""
        if density > 3000:
            return 1.1
        elif density < 500:
            return 0.95
        else:
            return 1.0
    
    def _age_demographic_factor(self, age: float) -> float:
        """Calculate risk factor based on median age"""
        if age < 30:
            return 1.05
        elif age > 45:
            return 0.95
        else:
            return 1.0

# Configuration example
EXTERNAL_DATA_CONFIG = {
    'crime_data_api_url': 'https://data.cityofchicago.org/resource/crimes.json',
    'traffic_api_key': 'your-traffic-api-key',
    'weather_api_key': 'your-weather-api-key',
    'census_api_key': 'your-census-api-key'
}

# Example usage
async def main():
    """Example usage of external data integration"""
    
    integrator = ExternalDataIntegrator(EXTERNAL_DATA_CONFIG)
    
    # Get contextual risk for a location
    risk_data = await integrator.get_contextual_risk_score(
        latitude=41.8781,
        longitude=-87.6298,
        timestamp=datetime.now()
    )
    
    print(f"Contextual Risk Score: {risk_data['contextual_risk_score']}")
    print(f"Risk Level: {risk_data['risk_level']}")

if __name__ == "__main__":
    asyncio.run(main())