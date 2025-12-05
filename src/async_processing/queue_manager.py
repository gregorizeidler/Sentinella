"""Async queue manager for background processing"""

from typing import Optional, Dict, Any, Callable
import asyncio
import json
import redis.asyncio as redis
from datetime import datetime


class QueueManager:
    """Manage async job queues"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.workers: Dict[str, Callable] = {}
    
    async def connect(self, redis_host: str = "localhost", redis_port: int = 6379):
        """Connect to Redis"""
        self.redis_client = await redis.from_url(
            f"redis://{redis_host}:{redis_port}/6",
            decode_responses=True,
        )
    
    async def disconnect(self):
        """Disconnect"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def enqueue(
        self,
        queue_name: str,
        job_data: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """Enqueue a job"""
        if not self.redis_client:
            return ""
        
        job_id = f"{queue_name}:{datetime.now().timestamp()}"
        job = {
            "id": job_id,
            "queue": queue_name,
            "data": job_data,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }
        
        # Use sorted set for priority queue
        await self.redis_client.zadd(
            f"queue:{queue_name}",
            {json.dumps(job): priority},
        )
        
        return job_id
    
    async def register_worker(self, queue_name: str, worker_func: Callable):
        """Register a worker function for a queue"""
        self.workers[queue_name] = worker_func
    
    async def process_queue(self, queue_name: str):
        """Process jobs from a queue"""
        if not self.redis_client or queue_name not in self.workers:
            return
        
        worker = self.workers[queue_name]
        
        while True:
            # Get highest priority job
            jobs = await self.redis_client.zrange(
                f"queue:{queue_name}",
                0,
                0,
                withscores=True,
            )
            
            if not jobs:
                await asyncio.sleep(1)
                continue
            
            job_json, _ = jobs[0]
            job = json.loads(job_json)
            
            # Remove from queue
            await self.redis_client.zrem(f"queue:{queue_name}", job_json)
            
            # Process job
            try:
                await worker(job["data"])
                job["status"] = "completed"
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
            
            # Store result
            await self.redis_client.setex(
                f"job_result:{job['id']}",
                3600,
                json.dumps(job),
            )

