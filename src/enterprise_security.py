"""
Enterprise API Security and Integration Module
Secure APIs for integration with existing insurance platforms
"""

from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import jwt
import hashlib
import hmac
import time
from datetime import datetime, timedelta
import redis
import logging
from typing import Dict, Any, Optional, List
import secrets
import bcrypt
from cryptography.fernet import Fernet
import uuid
import requests
from dataclasses import dataclass

@dataclass
class APIKey:
    """Represents an API key for partner integration"""
    key_id: str
    key_secret: str
    partner_name: str
    permissions: List[str]
    rate_limit: int
    expires_at: Optional[datetime]
    is_active: bool

class SecurityManager:
    """
    Advanced security manager for enterprise API integrations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis.from_url(config['redis_url'])
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self.fernet = Fernet(config['encryption_key'].encode())
        
        # JWT settings
        self.jwt_secret = config['jwt_secret']
        self.jwt_algorithm = 'HS256'
        self.jwt_expiration = timedelta(hours=24)
        
        # API versioning
        self.current_version = "v2.1"
        self.supported_versions = ["v1.0", "v1.1", "v2.0", "v2.1"]
        
    def generate_api_key(self, partner_name: str, permissions: List[str], 
                        rate_limit: int = 1000, expires_days: int = 365) -> APIKey:
        """Generate a new API key for partner integration"""
        
        key_id = f"tk_{secrets.token_urlsafe(16)}"
        key_secret = secrets.token_urlsafe(32)
        
        # Hash the secret for storage
        secret_hash = bcrypt.hashpw(key_secret.encode(), bcrypt.gensalt())
        
        expires_at = datetime.now() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_secret=key_secret,
            partner_name=partner_name,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at,
            is_active=True
        )
        
        # Store in Redis with encryption
        key_data = {
            'partner_name': partner_name,
            'secret_hash': secret_hash.decode(),
            'permissions': permissions,
            'rate_limit': rate_limit,
            'expires_at': expires_at.isoformat(),
            'is_active': True,
            'created_at': datetime.now().isoformat()
        }
        
        encrypted_data = self.fernet.encrypt(str(key_data).encode())
        self.redis_client.setex(f"api_key:{key_id}", 86400 * expires_days, encrypted_data)
        
        self.logger.info(f"Generated API key for partner: {partner_name}")
        return api_key
    
    def validate_api_key(self, key_id: str, key_secret: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return partner information"""
        
        try:
            # Retrieve from Redis
            encrypted_data = self.redis_client.get(f"api_key:{key_id}")
            if not encrypted_data:
                return None
            
            # Decrypt and parse
            key_data = eval(self.fernet.decrypt(encrypted_data).decode())
            
            # Check if key is active and not expired
            if not key_data.get('is_active'):
                return None
            
            expires_at = datetime.fromisoformat(key_data['expires_at'])
            if datetime.now() > expires_at:
                return None
            
            # Verify secret
            if not bcrypt.checkpw(key_secret.encode(), key_data['secret_hash'].encode()):
                return None
            
            return {
                'key_id': key_id,
                'partner_name': key_data['partner_name'],
                'permissions': key_data['permissions'],
                'rate_limit': key_data['rate_limit']
            }
            
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            return None
    
    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated users"""
        
        payload = {
            'user_id': user_data['user_id'],
            'user_type': user_data['user_type'],
            'permissions': user_data.get('permissions', []),
            'exp': datetime.utcnow() + self.jwt_expiration,
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())  # Unique token ID
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Store token for revocation capability
        self.redis_client.setex(
            f"jwt_token:{payload['jti']}", 
            int(self.jwt_expiration.total_seconds()),
            token
        )
        
        return token
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if token is blacklisted
            jti = payload.get('jti')
            if jti and not self.redis_client.exists(f"jwt_token:{jti}"):
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid JWT token")
            return None

class EnterpriseAPIGateway:
    """
    Enterprise API Gateway with advanced security and integration features
    """
    
    def __init__(self, app: Flask, security_manager: SecurityManager):
        self.app = app
        self.security_manager = security_manager
        self.redis_client = security_manager.redis_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiter
        self.limiter = Limiter(
            app,
            key_func=get_remote_address,
            default_limits=["1000 per hour"],
            storage_uri=security_manager.config['redis_url']
        )
        
        self._register_middleware()
        self._register_enterprise_endpoints()
    
    def _register_middleware(self):
        """Register security middleware"""
        
        @self.app.before_request
        def security_middleware():
            """Apply security checks to all requests"""
            
            # Skip security for health checks
            if request.endpoint == 'health_check':
                return
            
            # API versioning
            api_version = request.headers.get('API-Version', 'v1.0')
            if api_version not in self.security_manager.supported_versions:
                return jsonify({
                    'status': 'error',
                    'error': {
                        'code': 'UNSUPPORTED_VERSION',
                        'message': f'API version {api_version} is not supported',
                        'supported_versions': self.security_manager.supported_versions
                    }
                }), 400
            
            g.api_version = api_version
            
            # CORS handling
            if request.method == 'OPTIONS':
                response = jsonify({'status': 'ok'})
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, API-Key, API-Secret'
                return response
            
            # Authentication
            auth_header = request.headers.get('Authorization')
            api_key = request.headers.get('API-Key')
            api_secret = request.headers.get('API-Secret')
            
            if api_key and api_secret:
                # API Key authentication
                key_info = self.security_manager.validate_api_key(api_key, api_secret)
                if not key_info:
                    return jsonify({
                        'status': 'error',
                        'error': {
                            'code': 'INVALID_API_KEY',
                            'message': 'Invalid or expired API key'
                        }
                    }), 401
                
                g.auth_type = 'api_key'
                g.partner_info = key_info
                
            elif auth_header and auth_header.startswith('Bearer '):
                # JWT token authentication
                token = auth_header.split(' ')[1]
                payload = self.security_manager.validate_jwt_token(token)
                if not payload:
                    return jsonify({
                        'status': 'error',
                        'error': {
                            'code': 'INVALID_TOKEN',
                            'message': 'Invalid or expired JWT token'
                        }
                    }), 401
                
                g.auth_type = 'jwt'
                g.user_info = payload
                
            else:
                return jsonify({
                    'status': 'error',
                    'error': {
                        'code': 'AUTHENTICATION_REQUIRED',
                        'message': 'Valid authentication is required'
                    }
                }), 401
            
            # Request signature validation (for high-security endpoints)
            if request.endpoint in ['enterprise_risk_calculation', 'enterprise_bulk_operations']:
                if not self._validate_request_signature():
                    return jsonify({
                        'status': 'error',
                        'error': {
                            'code': 'INVALID_SIGNATURE',
                            'message': 'Request signature validation failed'
                        }
                    }), 401
            
            # Log API usage
            self._log_api_usage()
        
        @self.app.after_request
        def security_headers(response):
            """Add security headers to all responses"""
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            response.headers['Content-Security-Policy'] = "default-src 'self'"
            response.headers['API-Version'] = self.security_manager.current_version
            return response
    
    def _validate_request_signature(self) -> bool:
        """Validate HMAC request signature for high-security endpoints"""
        
        signature = request.headers.get('X-Signature')
        timestamp = request.headers.get('X-Timestamp')
        
        if not signature or not timestamp:
            return False
        
        # Check timestamp (prevent replay attacks)
        try:
            request_time = int(timestamp)
            if abs(time.time() - request_time) > 300:  # 5 minutes tolerance
                return False
        except ValueError:
            return False
        
        # Construct signature string
        if hasattr(g, 'partner_info'):
            secret = g.partner_info['key_secret']
        else:
            return False
        
        signature_string = f"{request.method}{request.path}{timestamp}{request.get_data().decode()}"
        expected_signature = hmac.new(
            secret.encode(),
            signature_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _log_api_usage(self):
        """Log API usage for monitoring and billing"""
        
        usage_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': request.endpoint,
            'method': request.method,
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent'),
            'api_version': g.get('api_version'),
            'auth_type': g.get('auth_type'),
            'partner_name': g.get('partner_info', {}).get('partner_name'),
            'user_id': g.get('user_info', {}).get('user_id')
        }
        
        # Store usage data for analytics
        self.redis_client.lpush('api_usage_log', str(usage_data))
        self.redis_client.ltrim('api_usage_log', 0, 10000)  # Keep last 10k requests
    
    def _register_enterprise_endpoints(self):
        """Register enterprise-specific API endpoints"""
        
        @self.app.route('/api/enterprise/risk/batch-calculate', methods=['POST'])
        @self.limiter.limit("100 per minute")
        def enterprise_bulk_risk_calculation():
            """Bulk risk calculation for insurance platforms"""
            
            try:
                data = request.get_json()
                
                if not data or 'drivers' not in data:
                    return jsonify({
                        'status': 'error',
                        'error': {
                            'code': 'INVALID_INPUT',
                            'message': 'drivers array is required'
                        }
                    }), 400
                
                drivers = data['drivers']
                if len(drivers) > 1000:
                    return jsonify({
                        'status': 'error',
                        'error': {
                            'code': 'BATCH_SIZE_EXCEEDED',
                            'message': 'Maximum batch size is 1000 drivers'
                        }
                    }), 400
                
                # Process batch risk calculation
                results = []
                for driver in drivers:
                    # Simulate risk calculation
                    risk_result = {
                        'driver_id': driver['driver_id'],
                        'risk_score': 0.35,  # Would be calculated by ML model
                        'risk_category': 'medium',
                        'confidence': 0.87,
                        'factors': {
                            'driving_behavior': 0.6,
                            'demographics': 0.3,
                            'vehicle_type': 0.1
                        }
                    }
                    results.append(risk_result)
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'results': results,
                        'processed_count': len(results),
                        'processing_time_ms': 150,
                        'batch_id': str(uuid.uuid4())
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error in bulk risk calculation: {e}")
                return jsonify({
                    'status': 'error',
                    'error': {
                        'code': 'INTERNAL_ERROR',
                        'message': 'Internal server error'
                    }
                }), 500
        
        @self.app.route('/api/enterprise/webhooks/register', methods=['POST'])
        def register_webhook():
            """Register webhook for real-time notifications"""
            
            try:
                data = request.get_json()
                
                required_fields = ['url', 'events', 'secret']
                for field in required_fields:
                    if field not in data:
                        return jsonify({
                            'status': 'error',
                            'error': {
                                'code': 'MISSING_FIELD',
                                'message': f'{field} is required'
                            }
                        }), 400
                
                # Store webhook configuration
                webhook_id = str(uuid.uuid4())
                webhook_config = {
                    'id': webhook_id,
                    'url': data['url'],
                    'events': data['events'],
                    'secret': data['secret'],
                    'partner_name': g.partner_info.get('partner_name'),
                    'created_at': datetime.now().isoformat(),
                    'is_active': True
                }
                
                self.redis_client.setex(
                    f"webhook:{webhook_id}",
                    86400 * 30,  # 30 days
                    str(webhook_config)
                )
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'webhook_id': webhook_id,
                        'events': data['events'],
                        'status': 'active'
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error registering webhook: {e}")
                return jsonify({
                    'status': 'error',
                    'error': {
                        'code': 'INTERNAL_ERROR',
                        'message': 'Internal server error'
                    }
                }), 500
        
        @self.app.route('/api/enterprise/integration/policy-sync', methods=['POST'])
        def sync_policy_data():
            """Synchronize policy data with insurance platform"""
            
            try:
                data = request.get_json()
                
                # Validate policy data
                required_fields = ['policy_number', 'driver_id', 'coverage_details']
                for field in required_fields:
                    if field not in data:
                        return jsonify({
                            'status': 'error',
                            'error': {
                                'code': 'MISSING_FIELD',
                                'message': f'{field} is required'
                            }
                        }), 400
                
                # Process policy synchronization
                policy_data = {
                    'policy_number': data['policy_number'],
                    'driver_id': data['driver_id'],
                    'coverage_details': data['coverage_details'],
                    'partner_name': g.partner_info.get('partner_name'),
                    'sync_timestamp': datetime.now().isoformat(),
                    'status': 'synchronized'
                }
                
                # Store policy data
                self.redis_client.setex(
                    f"policy:{data['policy_number']}",
                    86400 * 365,  # 1 year
                    str(policy_data)
                )
                
                return jsonify({
                    'status': 'success',
                    'data': {
                        'policy_number': data['policy_number'],
                        'sync_status': 'completed',
                        'sync_timestamp': policy_data['sync_timestamp']
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error syncing policy data: {e}")
                return jsonify({
                    'status': 'error',
                    'error': {
                        'code': 'INTERNAL_ERROR',
                        'message': 'Internal server error'
                    }
                }), 500

class WebhookManager:
    """
    Manages webhook delivery for real-time notifications
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def send_webhook(self, event_type: str, event_data: Dict[str, Any]):
        """Send webhook notification to registered endpoints"""
        
        # Get all active webhooks
        webhook_keys = self.redis_client.keys("webhook:*")
        
        for webhook_key in webhook_keys:
            try:
                webhook_config = eval(self.redis_client.get(webhook_key).decode())
                
                if not webhook_config.get('is_active'):
                    continue
                
                if event_type not in webhook_config.get('events', []):
                    continue
                
                # Prepare webhook payload
                payload = {
                    'event_type': event_type,
                    'timestamp': datetime.now().isoformat(),
                    'data': event_data,
                    'webhook_id': webhook_config['id']
                }
                
                # Create signature
                secret = webhook_config['secret']
                signature = hmac.new(
                    secret.encode(),
                    str(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                
                # Send webhook
                headers = {
                    'Content-Type': 'application/json',
                    'X-Telematics-Signature': signature,
                    'X-Telematics-Event': event_type
                }
                
                response = requests.post(
                    webhook_config['url'],
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    self.logger.info(f"Webhook delivered successfully: {webhook_config['id']}")
                else:
                    self.logger.warning(f"Webhook delivery failed: {webhook_config['id']} - {response.status_code}")
                
            except Exception as e:
                self.logger.error(f"Error sending webhook: {e}")

# Configuration example
ENTERPRISE_CONFIG = {
    'redis_url': 'redis://localhost:6379',
    'jwt_secret': 'super-secret-jwt-key-enterprise-2025',
    'encryption_key': 'fernet-encryption-key-32-characters',
    'rate_limits': {
        'default': '1000 per hour',
        'enterprise': '10000 per hour',
        'bulk_operations': '100 per minute'
    }
}

# Example usage for partner integration
def setup_enterprise_partner():
    """Example: Setup enterprise partner with API access"""
    
    security_manager = SecurityManager(ENTERPRISE_CONFIG)
    
    # Generate API key for insurance partner
    api_key = security_manager.generate_api_key(
        partner_name="StateWide Insurance Co.",
        permissions=["risk_calculation", "policy_sync", "webhook_management"],
        rate_limit=5000,
        expires_days=365
    )
    
    print(f"Partner API Key ID: {api_key.key_id}")
    print(f"Partner API Secret: {api_key.key_secret}")
    
    return api_key