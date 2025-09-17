
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
import sys
from typing import Dict, List, Optional
import uuid
import hashlib
import secrets

# Simple password hashing that works on all systems
def simple_password_hash(password: str) -> str:
    """Simple but secure password hashing using SHA256 and salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password_hash(password: str, hash_string: str) -> bool:
    """Verify password against hash"""
    try:
        salt, stored_hash = hash_string.split(':')
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash == stored_hash
    except:
        return False

# Import our custom modules (would normally be proper imports)
sys.path.append('../')
try:
    from data_processing.data_processor import TelematicsDataProcessor
    from ml_models.risk_scorer import RiskScorer
    from pricing_engine.dynamic_pricing import DynamicPricingEngine, DriverProfile
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Running in demo mode with limited functionality")

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'telematics-insurance-jwt-secret-key'  # Change in production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Extensions
CORS(app)
jwt = JWTManager(app)

# Global components (in production, these would use dependency injection)
try:
    data_processor = TelematicsDataProcessor()
    risk_scorer = RiskScorer()
    pricing_engine = DynamicPricingEngine()
except:
    print("Warning: Running in demo mode - some features may be limited")
    data_processor = None
    risk_scorer = None
    pricing_engine = None

# In-memory storage (in production, use proper database)
users_db = {}
telematics_data_cache = {}
risk_scores_cache = {}
premium_calculations_cache = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility functions
def create_response(data=None, message="Success", status_code=200):
    """Create standardized API response"""
    return make_response(jsonify({
        'status': 'success' if status_code < 400 else 'error',
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }), status_code)

def validate_required_fields(data, required_fields):
    """Validate required fields in request data"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    return None

# Authentication endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()

        validation_error = validate_required_fields(data, ['username', 'password', 'email', 'user_type'])
        if validation_error:
            return create_response(message=validation_error, status_code=400)

        username = data['username']
        if username in users_db:
            return create_response(message="Username already exists", status_code=409)

        # Create user
        user_id = str(uuid.uuid4())
        users_db[username] = {
            'user_id': user_id,
            'username': username,
            'password_hash': simple_password_hash(data['password']),
            'email': data['email'],
            'user_type': data['user_type'],  # 'driver', 'admin', 'insurer'
            'created_at': datetime.now().isoformat(),
            'active': True
        }

        # Create access token
        access_token = create_access_token(identity=username)

        return create_response(data={
            'user_id': user_id,
            'username': username,
            'user_type': data['user_type'],
            'access_token': access_token
        }, message="User registered successfully")

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return create_response(message="Registration failed", status_code=500)

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()

        validation_error = validate_required_fields(data, ['username', 'password'])
        if validation_error:
            return create_response(message=validation_error, status_code=400)

        username = data['username']
        if username not in users_db:
            return create_response(message="Invalid credentials", status_code=401)

        user = users_db[username]
        if not verify_password_hash(data['password'], user['password_hash']):
            return create_response(message="Invalid credentials", status_code=401)

        if not user['active']:
            return create_response(message="Account deactivated", status_code=401)

        access_token = create_access_token(identity=username)

        return create_response(data={
            'user_id': user['user_id'],
            'username': username,
            'user_type': user['user_type'],
            'access_token': access_token
        }, message="Login successful")

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return create_response(message="Login failed", status_code=500)

# Telematics data endpoints
@app.route('/api/telematics/upload', methods=['POST'])
@jwt_required()
def upload_telematics_data():
    """Upload telematics data"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()

        validation_error = validate_required_fields(data, ['driver_id', 'trip_data'])
        if validation_error:
            return create_response(message=validation_error, status_code=400)

        driver_id = data['driver_id']
        trip_data = data['trip_data']

        # Convert to DataFrame
        df = pd.DataFrame(trip_data)

        # Store in cache (in production, save to database)
        if driver_id not in telematics_data_cache:
            telematics_data_cache[driver_id] = []
        telematics_data_cache[driver_id].append(df)

        # Process data and update risk score if possible
        try:
            if data_processor and risk_scorer:
                risk_features = data_processor.process_driver_data(df, driver_id)
                features_df = data_processor.features_to_dataframe([risk_features])

                # Calculate risk score
                if risk_scorer.is_trained:
                    risk_score = risk_scorer.predict_risk_score(features_df)[0]
                    risk_scores_cache[driver_id] = {
                        'score': float(risk_score),
                        'updated_at': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.warning(f"Could not process risk score for {driver_id}: {str(e)}")

        return create_response(data={
            'driver_id': driver_id,
            'records_uploaded': len(trip_data),
            'status': 'processed'
        }, message="Telematics data uploaded successfully")

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return create_response(message="Upload failed", status_code=500)

@app.route('/api/telematics/driver/<driver_id>', methods=['GET'])
@jwt_required()
def get_driver_telematics(driver_id):
    """Get telematics data for a specific driver"""
    try:
        current_user = get_jwt_identity()

        if driver_id not in telematics_data_cache:
            return create_response(message="No data found for driver", status_code=404)

        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        driver_data = telematics_data_cache[driver_id]

        # Combine all trips for this driver
        combined_df = pd.concat(driver_data, ignore_index=True)

        # Filter by date if provided
        if start_date or end_date:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

            if start_date:
                combined_df = combined_df[combined_df['timestamp'] >= start_date]
            if end_date:
                combined_df = combined_df[combined_df['timestamp'] <= end_date]

        # Calculate summary statistics
        summary = {
            'total_trips': combined_df['trip_id'].nunique(),
            'total_distance': combined_df.groupby('trip_id')['speed_mph'].count().sum() * 0.01,  # Rough estimate
            'avg_speed': combined_df['speed_mph'].mean(),
            'max_speed': combined_df['speed_mph'].max(),
            'harsh_braking_events': combined_df['harsh_braking'].sum(),
            'harsh_acceleration_events': combined_df['harsh_acceleration'].sum(),
            'phone_usage_events': combined_df['phone_usage'].sum(),
            'date_range': {
                'start': combined_df['timestamp'].min().isoformat() if not combined_df.empty else None,
                'end': combined_df['timestamp'].max().isoformat() if not combined_df.empty else None
            }
        }

        return create_response(data={
            'driver_id': driver_id,
            'summary': summary,
            'records_count': len(combined_df)
        })

    except Exception as e:
        logger.error(f"Get telematics error: {str(e)}")
        return create_response(message="Failed to retrieve telematics data", status_code=500)

# Risk scoring endpoints
@app.route('/api/risk/score/<driver_id>', methods=['GET'])
@jwt_required()
def get_risk_score(driver_id):
    """Get risk score for a specific driver"""
    try:
        current_user = get_jwt_identity()

        if driver_id not in risk_scores_cache:
            # Generate a demo risk score
            demo_score = 0.3 + (hash(driver_id) % 100) / 200.0  # Generate consistent demo score
            risk_scores_cache[driver_id] = {
                'score': demo_score,
                'updated_at': datetime.now().isoformat()
            }

        risk_data = risk_scores_cache[driver_id]

        # Add risk category
        score = risk_data['score']
        if score < 0.3:
            category = 'low'
        elif score < 0.7:
            category = 'medium'
        else:
            category = 'high'

        return create_response(data={
            'driver_id': driver_id,
            'risk_score': score,
            'risk_category': category,
            'updated_at': risk_data['updated_at']
        })

    except Exception as e:
        logger.error(f"Get risk score error: {str(e)}")
        return create_response(message="Failed to retrieve risk score", status_code=500)

# Premium calculation endpoints
@app.route('/api/premium/calculate', methods=['POST'])
@jwt_required()
def calculate_premium():
    """Calculate premium for a driver"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()

        validation_error = validate_required_fields(data, ['driver_id', 'driver_profile'])
        if validation_error:
            return create_response(message=validation_error, status_code=400)

        driver_id = data['driver_id']
        profile_data = data['driver_profile']

        # Get or create risk score
        if driver_id not in risk_scores_cache:
            # Generate demo risk score based on driver profile
            age_factor = max(0.1, min(1.0, (profile_data['age'] - 16) / 70))
            mileage_factor = min(1.0, profile_data.get('annual_mileage', 12000) / 30000)
            risk_score = (age_factor + mileage_factor) / 2 + np.random.uniform(-0.1, 0.1)
            risk_score = max(0.1, min(0.9, risk_score))

            risk_scores_cache[driver_id] = {
                'score': risk_score,
                'updated_at': datetime.now().isoformat()
            }

        risk_score = risk_scores_cache[driver_id]['score']

        # Calculate premium using simplified logic
        base_premium = {
            'sedan': 800,
            'suv': 900,
            'truck': 1000,
            'compact': 700,
            'luxury': 1200
        }.get(profile_data.get('vehicle_type', 'sedan'), 800)

        # Age adjustment
        age = profile_data['age']
        if age < 25:
            age_multiplier = 1.5
        elif age < 35:
            age_multiplier = 1.1
        elif age < 65:
            age_multiplier = 1.0
        else:
            age_multiplier = 1.2

        # Risk multiplier
        if risk_score < 0.3:
            risk_multiplier = 0.7
        elif risk_score < 0.7:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 1.5

        # Mileage adjustment
        annual_mileage = profile_data.get('annual_mileage', 12000)
        if annual_mileage < 5000:
            mileage_adjustment = 0.8
        elif annual_mileage < 15000:
            mileage_adjustment = 1.0
        else:
            mileage_adjustment = 1.2

        final_premium = base_premium * age_multiplier * risk_multiplier * mileage_adjustment
        final_premium = max(300, min(5000, final_premium))  # Clamp between $300-$5000

        # Store calculation
        premium_calculations_cache[driver_id] = {
            'base_premium': base_premium,
            'risk_score': risk_score,
            'final_premium': final_premium,
            'updated_at': datetime.now().isoformat()
        }

        # Prepare response
        response_data = {
            'driver_id': driver_id,
            'base_premium': base_premium,
            'risk_score': risk_score,
            'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high',
            'final_premium': final_premium,
            'adjustments': {
                'risk_multiplier': risk_multiplier,
                'mileage_adjustment': mileage_adjustment,
                'age_multiplier': age_multiplier,
                'contextual_adjustment': 1.0
            },
            'calculation_date': datetime.now().isoformat()
        }

        return create_response(data=response_data)

    except Exception as e:
        logger.error(f"Premium calculation error: {str(e)}")
        return create_response(message="Premium calculation failed", status_code=500)

# Dashboard data endpoints
@app.route('/api/dashboard/overview', methods=['GET'])
@jwt_required()
def get_dashboard_overview():
    """Get dashboard overview data"""
    try:
        current_user = get_jwt_identity()

        # Calculate overview statistics
        total_drivers = len(telematics_data_cache) if telematics_data_cache else 10
        total_trips = sum(len(trips) for trips in telematics_data_cache.values()) if telematics_data_cache else 50

        # Generate demo risk distribution
        risk_distribution = {
            'low': 6,
            'medium': 8,
            'high': 3
        }

        avg_premium = 850.0
        if premium_calculations_cache:
            premiums = [data['final_premium'] for data in premium_calculations_cache.values()]
            avg_premium = sum(premiums) / len(premiums)

        overview = {
            'total_drivers': total_drivers,
            'total_trips': total_trips,
            'average_premium': avg_premium,
            'risk_distribution': risk_distribution,
            'active_policies': len(premium_calculations_cache) if premium_calculations_cache else 15,
            'last_updated': datetime.now().isoformat()
        }

        return create_response(data=overview)

    except Exception as e:
        logger.error(f"Dashboard overview error: {str(e)}")
        return create_response(message="Failed to retrieve dashboard data", status_code=500)

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return create_response(data={
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'data_processor': 'active' if data_processor else 'demo_mode',
            'risk_scorer': 'trained' if risk_scorer and risk_scorer.is_trained else 'demo_mode',
            'pricing_engine': 'active' if pricing_engine else 'demo_mode'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return create_response(message="Endpoint not found", status_code=404)

@app.errorhandler(500)
def internal_error(error):
    return create_response(message="Internal server error", status_code=500)

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return create_response(message="Token has expired", status_code=401)

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return create_response(message="Invalid token", status_code=401)

@jwt.unauthorized_loader
def missing_token_callback(error):
    return create_response(message="Access token required", status_code=401)

if __name__ == '__main__':
    # Create default admin user
    admin_username = 'admin'
    if admin_username not in users_db:
        users_db[admin_username] = {
            'user_id': str(uuid.uuid4()),
            'username': admin_username,
            'password_hash': simple_password_hash('admin123'),  # Fixed password hashing
            'email': 'admin@telematics-insurance.com',
            'user_type': 'admin',
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        logger.info("Default admin user created (username: admin, password: admin123)")

    logger.info("Starting Telematics Insurance API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
