# Deployment Guide - Telematics Insurance System

## Quick Deployment

### Prerequisites
- Python 3.8+
- 4GB RAM minimum
- 10GB disk space
- Internet connection for dependencies

### 1. System Setup
```bash
# Clone/extract the project
cd telematics_insurance_system

# Run setup script
python bin/setup.py
```

### 2. Generate Sample Data
```bash
# Generate test data for 100 drivers
python bin/generate_data.py --drivers 100 --trips 50
```

### 3. Train Models
```bash
# Train machine learning models
python bin/train_models.py
```

### 4. Start Services
```bash
# Terminal 1: API Server
python bin/start_api.py

# Terminal 2: Dashboard
python bin/start_dashboard.py
```

### 5. Access the System
- Dashboard: http://localhost:8080
- API: http://localhost:5000
- Default Login: admin / admin123

## Production Deployment

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Database Setup (PostgreSQL)
```sql
-- Create database and user
CREATE DATABASE telematics_insurance;
CREATE USER telematics_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE telematics_insurance TO telematics_user;
```

### Environment Variables
```bash
export DATABASE_URL="postgresql://telematics_user:secure_password@localhost:5432/telematics_insurance"
export JWT_SECRET_KEY="your-very-secure-jwt-secret-key"
export REDIS_URL="redis://localhost:6379/0"
export FLASK_ENV="production"
```

### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: telematics_insurance
      POSTGRES_USER: telematics_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://telematics_user:secure_password@db:5432/telematics_insurance
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

```bash
# Deploy with Docker
docker-compose up -d
```

### Health Checks
```bash
# API Health
curl http://localhost:5000/api/health

# Expected Response
{
  "status": "success",
  "data": {
    "status": "healthy",
    "services": {
      "data_processor": "active",
      "risk_scorer": "trained",
      "pricing_engine": "active"
    }
  }
}
```

### Monitoring Setup
```bash
# Install monitoring tools
pip install prometheus-client grafana-api

# Start monitoring services
python -m prometheus_client --port 8000
```

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
lsof -i :5000  # Mac/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows
```

**Database Connection Error**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # Mac

# Test connection
psql -h localhost -U telematics_user -d telematics_insurance
```

**Model Training Fails**
```bash
# Check data requirements
python -c "
import pandas as pd
df = pd.read_csv('data/samples/sample_telematics_data.csv')
print(f'Drivers: {df["driver_id"].nunique()}')
print(f'Trips: {df["trip_id"].nunique()}')
print('Minimum 10 drivers required')
"
```

### Performance Tuning

**API Server**
```python
# In production, use Gunicorn
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 src.api.api_server:app
```

**Database Optimization**
```sql
-- Create indexes for performance
CREATE INDEX idx_telematics_driver_timestamp ON telematics_raw(driver_id, timestamp);
CREATE INDEX idx_risk_features_driver ON risk_features(driver_id);
```

## Security Checklist

- [ ] Change default admin password
- [ ] Use strong JWT secret key
- [ ] Enable HTTPS in production
- [ ] Set up database backups
- [ ] Configure firewall rules
- [ ] Enable API rate limiting
- [ ] Set up logging and monitoring
- [ ] Regular security updates

## Backup and Recovery

### Database Backup
```bash
# Create backup
pg_dump -h localhost -U telematics_user telematics_insurance > backup.sql

# Restore backup
psql -h localhost -U telematics_user telematics_insurance < backup.sql
```

### Model Backup
```bash
# Backup trained models
tar -czf models_backup.tar.gz models/

# Restore models
tar -xzf models_backup.tar.gz
```

---
