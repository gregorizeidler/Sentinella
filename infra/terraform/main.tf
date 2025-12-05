terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    # Configure backend for state management
    # bucket = "your-terraform-state-bucket"
    # key    = "sentinella/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC for Sentinella infrastructure
resource "aws_vpc" "sentinella_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "sentinella-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "sentinella_igw" {
  vpc_id = aws_vpc.sentinella_vpc.id

  tags = {
    Name = "sentinella-igw"
  }
}

# Public Subnet
resource "aws_subnet" "sentinella_public" {
  vpc_id                  = aws_vpc.sentinella_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "sentinella-public-subnet"
  }
}

# Private Subnet (for EKS)
resource "aws_subnet" "sentinella_private" {
  vpc_id            = aws_vpc.sentinella_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name = "sentinella-private-subnet"
  }
}

# Route Table for Public Subnet
resource "aws_route_table" "sentinella_public" {
  vpc_id = aws_vpc.sentinella_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.sentinella_igw.id
  }

  tags = {
    Name = "sentinella-public-rt"
  }
}

resource "aws_route_table_association" "sentinella_public" {
  subnet_id      = aws_subnet.sentinella_public.id
  route_table_id = aws_route_table.sentinella_public.id
}

# Elasticache Redis for caching
resource "aws_elasticache_subnet_group" "sentinella_redis" {
  name       = "sentinella-redis-subnet-group"
  subnet_ids = [aws_subnet.sentinella_private.id]
}

resource "aws_elasticache_replication_group" "sentinella_redis" {
  replication_group_id       = "sentinella-redis"
  description                = "Redis cache for Sentinella Gateway"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  automatic_failover_enabled = true
  num_cache_clusters         = 2
  subnet_group_name          = aws_elasticache_subnet_group.sentinella_redis.name
  security_group_ids         = [aws_security_group.sentinella_redis.id]

  tags = {
    Name = "sentinella-redis"
  }
}

# Security Group for Redis
resource "aws_security_group" "sentinella_redis" {
  name        = "sentinella-redis-sg"
  description = "Security group for Sentinella Redis"
  vpc_id      = aws_vpc.sentinella_vpc.id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.sentinella_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sentinella-redis-sg"
  }
}

# RDS PostgreSQL for LangFuse
resource "aws_db_subnet_group" "sentinella_db" {
  name       = "sentinella-db-subnet-group"
  subnet_ids = [aws_subnet.sentinella_private.id]

  tags = {
    Name = "sentinella-db-subnet-group"
  }
}

resource "aws_db_instance" "sentinella_postgres" {
  identifier             = "sentinella-postgres"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  storage_type           = "gp3"
  db_name                = "langfuse"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.sentinella_db.name
  vpc_security_group_ids = [aws_security_group.sentinella_db.id]
  skip_final_snapshot    = true
  publicly_accessible    = false

  tags = {
    Name = "sentinella-postgres"
  }
}

# Security Group for PostgreSQL
resource "aws_security_group" "sentinella_db" {
  name        = "sentinella-db-sg"
  description = "Security group for Sentinella PostgreSQL"
  vpc_id      = aws_vpc.sentinella_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.sentinella_vpc.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sentinella-db-sg"
  }
}

# EKS Cluster (simplified - full EKS setup would require more resources)
# For production, use eks module or eksctl
resource "aws_eks_cluster" "sentinella" {
  name     = "sentinella-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = [
      aws_subnet.sentinella_private.id,
      aws_subnet.sentinella_public.id,
    ]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = {
    Name = "sentinella-eks-cluster"
  }
}

# IAM Role for EKS Cluster
resource "aws_iam_role" "eks_cluster" {
  name = "sentinella-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster.name
}

# Secrets Manager for API keys
resource "aws_secretsmanager_secret" "sentinella_secrets" {
  name = "sentinella/api-keys"

  tags = {
    Name = "sentinella-secrets"
  }
}

# Outputs
output "vpc_id" {
  value = aws_vpc.sentinella_vpc.id
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.sentinella_redis.configuration_endpoint_address
}

output "postgres_endpoint" {
  value = aws_db_instance.sentinella_postgres.endpoint
}

output "eks_cluster_name" {
  value = aws_eks_cluster.sentinella.name
}

