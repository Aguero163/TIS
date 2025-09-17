
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PremiumCalculation:
    """Data class to hold premium calculation details"""
    driver_id: str
    base_premium: float
    risk_score: float
    risk_level: RiskLevel
    risk_multiplier: float
    mileage_adjustment: float
    temporal_adjustment: float
    contextual_adjustment: float
    final_premium: float
    calculation_date: datetime
    details: Dict

@dataclass
class DriverProfile:
    """Enhanced driver profile for pricing"""
    driver_id: str
    age: int
    gender: str
    years_licensed: int
    vehicle_type: str
    vehicle_year: int
    location_zip: str
    credit_score: Optional[int] = None
    claims_history: List[Dict] = None
    policy_start_date: Optional[datetime] = None

class DynamicPricingEngine:
    """
    Dynamic insurance pricing engine using telematics data
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()

        # Load external data sources
        self.weather_risk_data = self._load_weather_risk_data()
        self.crime_data = self._load_crime_data()
        self.traffic_data = self._load_traffic_data()

    def _default_config(self) -> Dict:
        """Default pricing configuration"""
        return {
            'base_premiums': {
                'sedan': 800,
                'suv': 900,
                'truck': 1000,
                'compact': 700,
                'luxury': 1200
            },
            'risk_multipliers': {
                RiskLevel.VERY_LOW: 0.6,
                RiskLevel.LOW: 0.8,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 1.4,
                RiskLevel.VERY_HIGH: 2.0
            },
            'age_adjustments': {
                (16, 25): 1.5,
                (26, 35): 1.1,
                (36, 55): 1.0,
                (56, 65): 0.9,
                (66, 100): 1.2
            },
            'mileage_tiers': {
                (0, 5000): 0.8,
                (5001, 10000): 0.9,
                (10001, 15000): 1.0,
                (15001, 20000): 1.1,
                (20001, 30000): 1.3,
                (30001, float('inf')): 1.5
            },
            'vehicle_year_adjustment': 0.02,  # 2% increase per year older
            'claims_penalty': 0.3,  # 30% increase per claim
            'max_premium': 5000,
            'min_premium': 300,
            'update_frequency_days': 30
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pricing engine"""
        logger = logging.getLogger('PricingEngine')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_weather_risk_data(self) -> Dict:
        """Load weather risk data (simulated)"""
        # In production, this would connect to weather APIs
        return {
            'zip_codes': {
                '10001': {'annual_rain_days': 120, 'snow_days': 30, 'fog_days': 15},
                '90210': {'annual_rain_days': 35, 'snow_days': 0, 'fog_days': 40},
                '60601': {'annual_rain_days': 100, 'snow_days': 45, 'fog_days': 20}
            },
            'risk_multipliers': {
                'rain': 1.2,
                'snow': 1.6,
                'fog': 1.4
            }
        }

    def _load_crime_data(self) -> Dict:
        """Load crime statistics (simulated)"""
        # In production, this would connect to crime databases
        return {
            'zip_codes': {
                '10001': {'theft_rate': 5.2, 'vandalism_rate': 3.1},
                '90210': {'theft_rate': 2.1, 'vandalism_rate': 1.5},
                '60601': {'theft_rate': 4.8, 'vandalism_rate': 2.9}
            }
        }

    def _load_traffic_data(self) -> Dict:
        """Load traffic accident data (simulated)"""
        # In production, this would connect to traffic APIs
        return {
            'zip_codes': {
                '10001': {'accidents_per_1000': 12.5, 'congestion_index': 0.8},
                '90210': {'accidents_per_1000': 8.2, 'congestion_index': 0.6},
                '60601': {'accidents_per_1000': 11.1, 'congestion_index': 0.7}
            }
        }

    def calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert continuous risk score to risk level"""
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def calculate_base_premium(self, driver_profile: DriverProfile) -> float:
        """Calculate base premium based on traditional factors"""
        # Start with vehicle type base
        base_premium = self.config['base_premiums'].get(driver_profile.vehicle_type, 800)

        # Age adjustment
        age_multiplier = 1.0
        for age_range, multiplier in self.config['age_adjustments'].items():
            if age_range[0] <= driver_profile.age <= age_range[1]:
                age_multiplier = multiplier
                break

        base_premium *= age_multiplier

        # Vehicle year adjustment (older cars cost more to insure)
        current_year = datetime.now().year
        vehicle_age = current_year - driver_profile.vehicle_year
        vehicle_adjustment = 1 + (vehicle_age * self.config['vehicle_year_adjustment'])
        base_premium *= vehicle_adjustment

        # Claims history adjustment
        if driver_profile.claims_history:
            claims_count = len(driver_profile.claims_history)
            claims_adjustment = 1 + (claims_count * self.config['claims_penalty'])
            base_premium *= claims_adjustment

        return base_premium

    def calculate_mileage_adjustment(self, annual_mileage: float) -> float:
        """Calculate mileage-based adjustment (PAYD component)"""
        for mileage_range, multiplier in self.config['mileage_tiers'].items():
            if mileage_range[0] <= annual_mileage <= mileage_range[1]:
                return multiplier

        return 1.0

    def calculate_temporal_adjustment(self, telematics_features: Dict) -> float:
        """Calculate temporal risk adjustments"""
        adjustment = 1.0

        # Night driving increases risk
        night_ratio = telematics_features.get('night_driving_ratio', 0)
        adjustment += night_ratio * 0.3  # 30% increase for full night driving

        # Weekend driving pattern
        weekend_ratio = telematics_features.get('weekend_driving_ratio', 0)
        adjustment += weekend_ratio * 0.1  # 10% increase for weekend driving

        # Rush hour driving
        rush_hour_ratio = telematics_features.get('rush_hour_ratio', 0)
        adjustment += rush_hour_ratio * 0.2  # 20% increase for rush hour driving

        return adjustment

    def calculate_contextual_adjustment(self, driver_profile: DriverProfile, 
                                      telematics_features: Dict) -> float:
        """Calculate contextual risk adjustments based on external data"""
        adjustment = 1.0
        zip_code = driver_profile.location_zip

        # Weather risk adjustment
        if zip_code in self.weather_risk_data['zip_codes']:
            weather_data = self.weather_risk_data['zip_codes'][zip_code]
            weather_score = telematics_features.get('weather_risk_score', 1.0)

            # Combine weather exposure with telematics weather risk
            weather_adjustment = 1 + ((weather_score - 1) * 0.5)  # Moderate the effect
            adjustment *= weather_adjustment

        # Crime risk adjustment
        if zip_code in self.crime_data['zip_codes']:
            crime_data = self.crime_data['zip_codes'][zip_code]
            theft_risk = min(crime_data['theft_rate'] / 10, 0.2)  # Cap at 20%
            adjustment += theft_risk

        # Traffic risk adjustment
        if zip_code in self.traffic_data['zip_codes']:
            traffic_data = self.traffic_data['zip_codes'][zip_code]
            accident_risk = min(traffic_data['accidents_per_1000'] / 100, 0.15)  # Cap at 15%
            adjustment += accident_risk

        return adjustment

    def calculate_premium(self, driver_profile: DriverProfile, 
                         risk_score: float, telematics_features: Dict) -> PremiumCalculation:
        """Calculate complete premium with all adjustments"""

        # Calculate base premium
        base_premium = self.calculate_base_premium(driver_profile)

        # Determine risk level and multiplier
        risk_level = self.calculate_risk_level(risk_score)
        risk_multiplier = self.config['risk_multipliers'][risk_level]

        # Calculate adjustments
        mileage_adjustment = self.calculate_mileage_adjustment(
            telematics_features.get('total_mileage', 12000)
        )

        temporal_adjustment = self.calculate_temporal_adjustment(telematics_features)

        contextual_adjustment = self.calculate_contextual_adjustment(
            driver_profile, telematics_features
        )

        # Apply all adjustments
        adjusted_premium = base_premium * risk_multiplier * mileage_adjustment * temporal_adjustment * contextual_adjustment

        # Apply bounds
        final_premium = max(self.config['min_premium'], 
                          min(adjusted_premium, self.config['max_premium']))

        # Create detailed calculation
        calculation_details = {
            'base_premium_factors': {
                'vehicle_type': driver_profile.vehicle_type,
                'age_group': self._get_age_group(driver_profile.age),
                'vehicle_age': datetime.now().year - driver_profile.vehicle_year,
                'claims_count': len(driver_profile.claims_history) if driver_profile.claims_history else 0
            },
            'telematics_factors': {
                'harsh_braking_events': telematics_features.get('harsh_braking_count', 0),
                'harsh_acceleration_events': telematics_features.get('harsh_acceleration_count', 0),
                'over_speed_ratio': telematics_features.get('over_speed_ratio', 0),
                'phone_usage_ratio': telematics_features.get('phone_usage_ratio', 0)
            },
            'contextual_factors': {
                'location_zip': driver_profile.location_zip,
                'weather_risk': telematics_features.get('weather_risk_score', 1.0),
                'annual_mileage': telematics_features.get('total_mileage', 12000)
            }
        }

        premium_calc = PremiumCalculation(
            driver_id=driver_profile.driver_id,
            base_premium=base_premium,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_multiplier=risk_multiplier,
            mileage_adjustment=mileage_adjustment,
            temporal_adjustment=temporal_adjustment,
            contextual_adjustment=contextual_adjustment,
            final_premium=final_premium,
            calculation_date=datetime.now(),
            details=calculation_details
        )

        self.logger.info(f"Premium calculated for driver {driver_profile.driver_id}: ${final_premium:.2f}")

        return premium_calc

    def _get_age_group(self, age: int) -> str:
        """Get age group for reporting"""
        for age_range, _ in self.config['age_adjustments'].items():
            if age_range[0] <= age <= age_range[1]:
                return f"{age_range[0]}-{age_range[1]}"
        return "unknown"

    def calculate_premium_change(self, current_premium: float, new_premium: float) -> Dict:
        """Calculate and categorize premium changes"""
        change_amount = new_premium - current_premium
        change_percentage = (change_amount / current_premium) * 100

        if abs(change_percentage) < 5:
            change_category = "minimal"
        elif change_percentage >= 5:
            change_category = "increase"
        else:
            change_category = "decrease"

        return {
            'previous_premium': current_premium,
            'new_premium': new_premium,
            'change_amount': change_amount,
            'change_percentage': change_percentage,
            'change_category': change_category,
            'effective_date': datetime.now() + timedelta(days=self.config['update_frequency_days'])
        }

    def generate_premium_explanation(self, premium_calc: PremiumCalculation) -> str:
        """Generate human-readable explanation of premium calculation"""
        explanation = f"Premium Calculation for Driver {premium_calc.driver_id}\n"
        explanation += "=" * 50 + "\n\n"

        explanation += f"Base Premium: ${premium_calc.base_premium:.2f}\n"
        explanation += f"Risk Level: {premium_calc.risk_level.value.title()} (Score: {premium_calc.risk_score:.3f})\n"
        explanation += f"Risk Multiplier: {premium_calc.risk_multiplier}x\n\n"

        explanation += "Adjustments:\n"
        explanation += f"  Mileage: {premium_calc.mileage_adjustment:.3f}x\n"
        explanation += f"  Temporal: {premium_calc.temporal_adjustment:.3f}x\n"
        explanation += f"  Contextual: {premium_calc.contextual_adjustment:.3f}x\n\n"

        explanation += f"Final Premium: ${premium_calc.final_premium:.2f}\n"

        # Add driving behavior insights
        details = premium_calc.details['telematics_factors']
        explanation += "\nDriving Behavior Highlights:\n"
        explanation += f"  Harsh Braking Events: {details['harsh_braking_events']}\n"
        explanation += f"  Harsh Acceleration Events: {details['harsh_acceleration_events']}\n"
        explanation += f"  Over Speed Ratio: {details['over_speed_ratio']:.1%}\n"
        explanation += f"  Phone Usage Ratio: {details['phone_usage_ratio']:.1%}\n"

        return explanation

    def batch_calculate_premiums(self, driver_profiles: List[DriverProfile], 
                               risk_scores: List[float], 
                               telematics_features: List[Dict]) -> List[PremiumCalculation]:
        """Calculate premiums for multiple drivers"""
        if len(driver_profiles) != len(risk_scores) or len(driver_profiles) != len(telematics_features):
            raise ValueError("Input lists must have the same length")

        premium_calculations = []

        self.logger.info(f"Calculating premiums for {len(driver_profiles)} drivers...")

        for i, (profile, risk_score, features) in enumerate(zip(driver_profiles, risk_scores, telematics_features)):
            try:
                premium_calc = self.calculate_premium(profile, risk_score, features)
                premium_calculations.append(premium_calc)

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(driver_profiles)} premiums")

            except Exception as e:
                self.logger.error(f"Error calculating premium for driver {profile.driver_id}: {str(e)}")
                continue

        self.logger.info(f"Completed premium calculations for {len(premium_calculations)} drivers")
        return premium_calculations

    def save_premium_calculations(self, premium_calculations: List[PremiumCalculation], 
                                filepath: str):
        """Save premium calculations to CSV"""
        data = []

        for calc in premium_calculations:
            row = {
                'driver_id': calc.driver_id,
                'base_premium': calc.base_premium,
                'risk_score': calc.risk_score,
                'risk_level': calc.risk_level.value,
                'risk_multiplier': calc.risk_multiplier,
                'mileage_adjustment': calc.mileage_adjustment,
                'temporal_adjustment': calc.temporal_adjustment,
                'contextual_adjustment': calc.contextual_adjustment,
                'final_premium': calc.final_premium,
                'calculation_date': calc.calculation_date.isoformat()
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Premium calculations saved to {filepath}")

class PremiumOptimizer:
    """
    Optimize pricing strategies based on market conditions
    """

    def __init__(self, pricing_engine: DynamicPricingEngine):
        self.pricing_engine = pricing_engine
        self.logger = logging.getLogger('PremiumOptimizer')

    def analyze_market_position(self, premium_calculations: List[PremiumCalculation]) -> Dict:
        """Analyze competitive position and pricing distribution"""
        premiums = [calc.final_premium for calc in premium_calculations]

        analysis = {
            'total_drivers': len(premiums),
            'average_premium': np.mean(premiums),
            'median_premium': np.median(premiums),
            'premium_std': np.std(premiums),
            'min_premium': min(premiums),
            'max_premium': max(premiums),
            'risk_distribution': self._analyze_risk_distribution(premium_calculations),
            'pricing_recommendations': self._generate_pricing_recommendations(premium_calculations)
        }

        return analysis

    def _analyze_risk_distribution(self, premium_calculations: List[PremiumCalculation]) -> Dict:
        """Analyze distribution of risk levels"""
        risk_counts = {}
        for calc in premium_calculations:
            risk_level = calc.risk_level.value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

        total = len(premium_calculations)
        risk_distribution = {level: count/total for level, count in risk_counts.items()}

        return risk_distribution

    def _generate_pricing_recommendations(self, premium_calculations: List[PremiumCalculation]) -> List[str]:
        """Generate strategic pricing recommendations"""
        recommendations = []

        # Analyze risk distribution
        risk_dist = self._analyze_risk_distribution(premium_calculations)

        if risk_dist.get('high', 0) + risk_dist.get('very_high', 0) > 0.3:
            recommendations.append("Consider additional driver training programs to reduce high-risk population")

        if risk_dist.get('very_low', 0) + risk_dist.get('low', 0) > 0.6:
            recommendations.append("Opportunity to attract more low-risk drivers with competitive pricing")

        # Analyze premium distribution
        premiums = [calc.final_premium for calc in premium_calculations]
        if np.std(premiums) / np.mean(premiums) > 0.5:
            recommendations.append("High premium variance suggests good risk differentiation")

        return recommendations

