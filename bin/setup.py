#!/usr/bin/env python3
"""
Telematics Insurance System Setup Script
Automated environment setup and system initialization
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import shutil
import platform

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/setup.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create all necessary project directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'src/data_collection',
        'src/data_processing', 
        'src/ml_models',
        'src/pricing_engine',
        'src/api',
        'src/integrations',
        'src/dashboard',
        'models',
        'data/raw',
        'data/processed',
        'data/samples',
        'logs',
        'docs',
        'tests',
        'config',
        'kubernetes',
        'terraform',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    python_packages = [
        'src',
        'src/data_collection',
        'src/data_processing',
        'src/ml_models', 
        'src/pricing_engine',
        'src/api',
        'src/integrations',
        'tests'
    ]
    
    for package in python_packages:
        init_file = Path(package) / '__init__.py'
        init_file.touch()
        logger.info(f"✓ Created __init__.py: {package}")

def check_python_version():
    """Check Python version compatibility"""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error(f"Python 3.8+ required. Current version: {sys.version}")
        return False
    
    logger.info(f"✓ Python version compatible: {sys.version}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("📦 Installing Python dependencies...")
        
        # Upgrade pip first
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        
        # Install core dependencies
        core_deps = [
            "flask>=2.3.0",
            "pandas>=2.0.0", 
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "redis>=5.0.0",
            "requests>=2.31.0",
            "python-dotenv>=1.0.0",
            "pytest>=7.4.0"
        ]
        
        for dep in core_deps:
            logger.info(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ])
        
        # Install from requirements.txt if it exists
        if os.path.exists('requirements.txt'):
            logger.info("Installing from requirements.txt...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
        
        logger.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.info("Try installing manually: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        logger.error("pip not found. Please ensure Python and pip are installed")
        return False

def create_config_files():
    """Create configuration files"""
    logger = logging.getLogger(__name__)
    
    # Main configuration
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "telematics_insurance",
            "user": "telematics_user"
        },
        "api": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        },
        "ml_models": {
            "model_path": "models/",
            "retrain_interval_days": 30,
            "min_training_samples": 1000,
            "performance_threshold": 0.85
        },
        "features": {
            "speed_thresholds": {
                "residential": 25,
                "arterial": 35,
                "highway": 65,
                "interstate": 75
            },
            "risk_weights": {
                "harsh_braking_rate": 0.25,
                "harsh_acceleration_rate": 0.20,
                "over_speed_ratio": 0.18,
                "night_driving_ratio": 0.12,
                "phone_usage_ratio": 0.10,
                "total_mileage": 0.08,
                "weather_risk_score": 0.07
            }
        },
        "security": {
            "jwt_expiration_hours": 24,
            "api_rate_limit_per_hour": 1000,
            "max_login_attempts": 5
        },
        "external_apis": {
            "weather_api_timeout": 10,
            "crime_data_timeout": 15,
            "traffic_api_timeout": 10
        }
    }
    
    with open('config/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Environment template
    env_template = """# Telematics Insurance System Environment Variables

# Database Configuration
DATABASE_URL=postgresql://telematics_user:password@localhost:5432/telematics_insurance
REDIS_URL=redis://localhost:6379/0

# Security Settings (To be changed accordingly)
JWT_SECRET_KEY=dev-secret-key-change-in-production
ENCRYPTION_KEY=dev-encryption-key-32-chars-here
API_RATE_LIMIT=1000

# External APIs (Just placeholder)
WEATHER_API_KEY=your-openweather-api-key
CRIME_DATA_API_KEY=your-crime-data-api-key
TRAFFIC_API_KEY=your-traffic-api-key

# AWS Configuration (Just placeholder)
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
S3_BUCKET_NAME=telematics-data-lake

# Development Settings
DEBUG=True
LOG_LEVEL=INFO
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        shutil.copy('.env.template', '.env')
        logger.info("Created .env file from template")
    
    logger.info("Created configuration files")

def check_system_requirements():
    """Check system requirements and optional dependencies"""
    logger = logging.getLogger(__name__)
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            logger.warning(f"Low memory: {memory_gb:.1f}GB. Recommended: 4GB+")
        else:
            logger.info(f"Memory: {memory_gb:.1f}GB")
    except ImportError:
        logger.info("Install psutil for system monitoring: pip install psutil")
    
    # Check disk space
    total, used, free = shutil.disk_usage('.')
    free_gb = free / (1024**3)
    if free_gb < 2:
        logger.warning(f"Low disk space: {free_gb:.1f}GB free. Recommended: 2GB+")
    else:
        logger.info(f"Disk space: {free_gb:.1f}GB free")
    
    # Check optional services
    services = {
        'docker': 'docker --version',
        'git': 'git --version',
        'curl': 'curl --version'
    }
    
    for service, command in services.items():
        try:
            subprocess.run(command.split(), 
                         capture_output=True, 
                         check=True, 
                         timeout=5)
            logger.info(f"{service.title()} available")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.info(f"{service.title()} not found (optional)")

def verify_installation():
    """Verify that the installation is working correctly"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test core imports
        logger.info("Testing core dependencies...")
        
        import pandas
        import numpy
        import sklearn
        import flask
        
        logger.info("Core dependencies working")
        
        # Test file structure
        required_files = [
            'src/data_collection/telematics_simulator.py',
            'src/data_processing/data_processor.py',
            'src/ml_models/risk_scorer.py',
            'src/pricing_engine/dynamic_pricing.py',
            'src/api/api_server.py',
            'bin/generate_data.py',
            'bin/train_models.py',
            'bin/start_api.py',
            'bin/start_dashboard.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
            logger.info("Make sure all source files are in place")
        else:
            logger.info("All source files present")
        
        return len(missing_files) == 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False

def create_sample_data():
    """Create initial sample data if possible"""
    logger = logging.getLogger(__name__)
    
    try:
        # Only create sample data if the simulator exists
        if os.path.exists('src/data_collection/telematics_simulator.py'):
            logger.info("Generating initial sample data...")
            
            # Import and run data generator
            sys.path.insert(0, os.path.abspath('.'))
            
            result = subprocess.run([
                sys.executable, 'bin/generate_data.py', 
                '--drivers', '10', 
                '--trips', '5',
                '--output', 'data/samples/initial_sample_data.csv'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("Initial sample data created")
            else:
                logger.warning(f"Could not create sample data: {result.stderr}")
                
    except Exception as e:
        logger.warning(f"Could not create sample data: {e}")
        logger.info("You can generate data later with: python bin/generate_data.py")

def print_next_steps():
    """Print next steps for the user"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE! Next steps:")
    print("="*60)
    print()
    print("1. Generate sample data:")
    print("   python bin/generate_data.py --drivers 50 --trips 15")
    print()
    print("2. Train ML models:")
    print("   python bin/train_models.py")
    print()
    print("3. Start the system:")
    print("   # Terminal 1:")
    print("   python bin/start_api.py")
    print()
    print("   # Terminal 2:")
    print("   python bin/start_dashboard.py")
    print()
    print("4. Access the system:")
    print("   • Dashboard: http://localhost:8080")
    print("   • API Health: http://localhost:5000/api/health")
    print()
    print("Documentation:")
    print("   • Technical: docs/TECHNICAL.md")
    print("   • API Guide: docs/API.md")  
    print("   • Deployment: docs/DEPLOYMENT.md")
    print()
    print("Troubleshooting:")
    print("   • Check logs in: logs/")
    print("   • Run tests: pytest tests/")
    print("   • Verify install: python -c \"import pandas, numpy, sklearn, flask\"")
    print()
    print("System ready for development and testing!")
    print("="*60)

def main():
    """Main setup function"""
    
    print("Telematics Insurance System Setup")
    print("=" * 50)
    
    # Create logs directory first
    Path('logs').mkdir(exist_ok=True)
    
    logger = setup_logging()
    logger.info("Starting system setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    logger.info("Creating directory structure...")
    create_directories()
    
    # Check system requirements
    logger.info("Checking system requirements...")
    check_system_requirements()
    
    # Install dependencies
    logger.info("Installing dependencies...")
    if not install_dependencies():
        logger.warning("Some dependencies failed to install")
        logger.info("You may need to install them manually")
    
    # Create configuration files
    logger.info("Creating configuration files...")
    create_config_files()
    
    # Verify installation
    logger.info("Verifying installation...")
    installation_ok = verify_installation()
    
    # Create sample data
    if installation_ok:
        create_sample_data()
    
    # Print next steps
    print_next_steps()
    
    if installation_ok:
        logger.info("Setup completed successfully!")
        return True
    else:
        logger.warning("Setup completed with warnings")
        logger.info("Check the logs and install missing dependencies")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)