# Cloud Infrastructure Configuration for Telematics Insurance System

## AWS Cloud Architecture

### EKS Kubernetes Deployment
```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: telematics-insurance
  labels:
    name: telematics-insurance
---
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: telematics-api
  namespace: telematics-insurance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: telematics-api
  template:
    metadata:
      labels:
        app: telematics-api
    spec:
      containers:
      - name: api
        image: telematics/api:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# kubernetes/api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: telematics-api-service
  namespace: telematics-insurance
spec:
  selector:
    app: telematics-api
  ports:
  - port: 80
    targetPort: 5000
  type: ClusterIP
---
# kubernetes/api-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: telematics-api-hpa
  namespace: telematics-insurance
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: telematics-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: telematics-ingress
  namespace: telematics-insurance
  annotations:
    kubernetes.io/ingress.class: "alb"
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS-1-2-2017-01
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.telematics-insurance.com
    secretName: telematics-tls
  rules:
  - host: api.telematics-insurance.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: telematics-api-service
            port:
              number: 80
```

### Terraform Infrastructure as Code
```hcl
# terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "telematics-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "telematics-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-west-2a", "us-west-2b", "us-west-2c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "telematics-cluster"
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  
  eks_managed_node_groups = {
    telematics_nodes = {
      min_size     = 3
      max_size     = 20
      desired_size = 5
      
      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = "production"
        NodeGroup   = "telematics-nodes"
      }
    }
  }
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# RDS Database Cluster
resource "aws_rds_cluster" "telematics_db" {
  cluster_identifier      = "telematics-postgres-cluster"
  engine                  = "aurora-postgresql"
  engine_version          = "13.13"
  availability_zones      = ["us-west-2a", "us-west-2b", "us-west-2c"]
  database_name          = "telematics_insurance"
  master_username        = "telematics_admin"
  master_password        = var.db_password
  backup_retention_period = 7
  preferred_backup_window = "07:00-09:00"
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.telematics.name
  
  encryption_at_rest_enabled = true
  storage_encrypted         = true
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# Redis ElastiCache Cluster
resource "aws_elasticache_replication_group" "telematics_redis" {
  replication_group_id       = "telematics-redis"
  description                = "Redis cluster for telematics caching"
  
  node_type                 = "cache.t3.micro"
  port                      = 6379
  parameter_group_name      = "default.redis7"
  
  num_cache_clusters        = 3
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.telematics.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# S3 Buckets for Data Storage
resource "aws_s3_bucket" "telematics_data_lake" {
  bucket = "telematics-insurance-data-lake-${random_id.bucket_suffix.hex}"
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
    Purpose     = "data-lake"
  }
}

resource "aws_s3_bucket_versioning" "telematics_data_lake" {
  bucket = aws_s3_bucket.telematics_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "telematics_data_lake" {
  bucket = aws_s3_bucket.telematics_data_lake.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.telematics_s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

# Kinesis Data Streams for Real-time Processing
resource "aws_kinesis_stream" "telematics_stream" {
  name             = "telematics-data-stream"
  shard_count      = 3
  retention_period = 168  # 7 days
  
  shard_level_metrics = [
    "IncomingRecords",
    "OutgoingRecords"
  ]
  
  encryption_type = "KMS"
  kms_key_id     = aws_kms_key.telematics_kinesis.arn
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# Lambda Functions for Serverless Processing
resource "aws_lambda_function" "telematics_processor" {
  filename         = "telematics_processor.zip"
  function_name    = "telematics-data-processor"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 300
  
  environment {
    variables = {
      DATABASE_URL = "postgresql://${aws_rds_cluster.telematics_db.master_username}:${var.db_password}@${aws_rds_cluster.telematics_db.endpoint}:5432/${aws_rds_cluster.telematics_db.database_name}"
      REDIS_URL    = "redis://${aws_elasticache_replication_group.telematics_redis.primary_endpoint_address}:6379"
      S3_BUCKET    = aws_s3_bucket.telematics_data_lake.bucket
    }
  }
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}

# API Gateway for Enterprise Integration
resource "aws_api_gateway_rest_api" "telematics_enterprise_api" {
  name        = "telematics-enterprise-api"
  description = "Enterprise API for insurance platform integration"
  
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

# CloudFront CDN
resource "aws_cloudfront_distribution" "telematics_cdn" {
  origin {
    domain_name = aws_s3_bucket.telematics_static.bucket_regional_domain_name
    origin_id   = "telematics-static-origin"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.telematics.cloudfront_access_identity_path
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "telematics-static-origin"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.telematics.arn
    ssl_support_method  = "sni-only"
  }
  
  tags = {
    Environment = "production"
    Project     = "telematics-insurance"
  }
}
```

### Auto-scaling Configuration
```yaml
# kubernetes/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/telematics-cluster
```