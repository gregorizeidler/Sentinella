output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.sentinella_vpc.id
}

output "redis_endpoint" {
  description = "Redis endpoint address"
  value       = aws_elasticache_replication_group.sentinella_redis.configuration_endpoint_address
  sensitive   = true
}

output "postgres_endpoint" {
  description = "PostgreSQL endpoint"
  value       = aws_db_instance.sentinella_postgres.endpoint
  sensitive   = true
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.sentinella.name
}

