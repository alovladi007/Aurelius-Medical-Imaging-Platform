"""Rate limiting and quota management middleware.

Implements per-user and per-tenant rate limiting with Redis backend.
"""
import os
import time
import logging
from typing import Callable, Optional
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter with tenant-aware quotas."""
    
    def __init__(self, redis_url: str):
        """Initialize rate limiter with Redis connection."""
        self.redis_url = redis_url
        self.redis = None
        
        # Default rate limits (per user)
        self.DEFAULT_LIMITS = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }
        
        # Tenant-specific quotas (monthly)
        self.TENANT_QUOTAS = {
            "api_calls": 1000000,  # 1M API calls per month
            "storage_gb": 1000,    # 1TB storage
            "gpu_hours": 100       # 100 GPU hours per month
        }
    
    async def connect(self):
        """Connect to Redis."""
        if self.redis is None:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("✅ Rate limiter connected to Redis")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    def get_user_key(self, user_id: str, window: str) -> str:
        """Get Redis key for user rate limit."""
        return f"ratelimit:user:{user_id}:{window}"
    
    def get_tenant_key(self, tenant_id: str, resource: str, month: str) -> str:
        """Get Redis key for tenant quota."""
        return f"quota:tenant:{tenant_id}:{resource}:{month}"
    
    async def check_rate_limit(
        self,
        user_id: str,
        limits: Optional[dict] = None
    ) -> tuple[bool, dict]:
        """
        Check if user is within rate limits.
        
        Returns:
            (allowed, info) where info contains current usage and limits
        """
        if not self.redis:
            await self.connect()
        
        if limits is None:
            limits = self.DEFAULT_LIMITS
        
        now = time.time()
        current_minute = int(now / 60)
        current_hour = int(now / 3600)
        current_day = int(now / 86400)
        
        # Check minute limit
        minute_key = self.get_user_key(user_id, f"minute:{current_minute}")
        minute_count = await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        if minute_count > limits["requests_per_minute"]:
            return False, {
                "window": "minute",
                "limit": limits["requests_per_minute"],
                "current": minute_count,
                "reset_at": (current_minute + 1) * 60
            }
        
        # Check hour limit
        hour_key = self.get_user_key(user_id, f"hour:{current_hour}")
        hour_count = await self.redis.incr(hour_key)
        await self.redis.expire(hour_key, 3600)
        
        if hour_count > limits["requests_per_hour"]:
            return False, {
                "window": "hour",
                "limit": limits["requests_per_hour"],
                "current": hour_count,
                "reset_at": (current_hour + 1) * 3600
            }
        
        # Check day limit
        day_key = self.get_user_key(user_id, f"day:{current_day}")
        day_count = await self.redis.incr(day_key)
        await self.redis.expire(day_key, 86400)
        
        if day_count > limits["requests_per_day"]:
            return False, {
                "window": "day",
                "limit": limits["requests_per_day"],
                "current": day_count,
                "reset_at": (current_day + 1) * 86400
            }
        
        return True, {
            "requests_per_minute": {
                "limit": limits["requests_per_minute"],
                "remaining": limits["requests_per_minute"] - minute_count
            },
            "requests_per_hour": {
                "limit": limits["requests_per_hour"],
                "remaining": limits["requests_per_hour"] - hour_count
            },
            "requests_per_day": {
                "limit": limits["requests_per_day"],
                "remaining": limits["requests_per_day"] - day_count
            }
        }
    
    async def check_tenant_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: float = 1.0
    ) -> tuple[bool, dict]:
        """
        Check and increment tenant quota usage.
        
        Args:
            tenant_id: Tenant identifier
            resource: Resource type ("api_calls", "storage_gb", "gpu_hours")
            amount: Amount to add (default 1.0)
        
        Returns:
            (allowed, info) where info contains usage and quota details
        """
        if not self.redis:
            await self.connect()
        
        # Get current month
        month = datetime.now().strftime("%Y-%m")
        
        # Get quota limit
        quota_limit = self.TENANT_QUOTAS.get(resource, float('inf'))
        
        # Get current usage
        quota_key = self.get_tenant_key(tenant_id, resource, month)
        current_usage = await self.redis.get(quota_key)
        current_usage = float(current_usage) if current_usage else 0.0
        
        # Check if adding amount would exceed quota
        new_usage = current_usage + amount
        
        if new_usage > quota_limit:
            return False, {
                "resource": resource,
                "quota": quota_limit,
                "current": current_usage,
                "requested": amount,
                "would_be": new_usage,
                "month": month
            }
        
        # Increment usage
        await self.redis.incrbyfloat(quota_key, amount)
        
        # Set expiry to end of next month
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1)
        expire_seconds = int((next_month - datetime.now()).total_seconds())
        await self.redis.expire(quota_key, expire_seconds)
        
        return True, {
            "resource": resource,
            "quota": quota_limit,
            "current": new_usage,
            "remaining": quota_limit - new_usage,
            "percentage": (new_usage / quota_limit) * 100 if quota_limit > 0 else 0,
            "month": month
        }
    
    async def get_tenant_usage(self, tenant_id: str) -> dict:
        """Get current usage for all resources for a tenant."""
        if not self.redis:
            await self.connect()
        
        month = datetime.now().strftime("%Y-%m")
        usage = {}
        
        for resource, quota in self.TENANT_QUOTAS.items():
            key = self.get_tenant_key(tenant_id, resource, month)
            current = await self.redis.get(key)
            current = float(current) if current else 0.0
            
            usage[resource] = {
                "quota": quota,
                "current": current,
                "remaining": quota - current,
                "percentage": (current / quota) * 100 if quota > 0 else 0
            }
        
        return {
            "tenant_id": tenant_id,
            "month": month,
            "resources": usage
        }


async def rate_limit_middleware(request: Request, call_next: Callable):
    """
    Middleware to enforce rate limits and quotas.
    
    Adds headers:
    - X-RateLimit-Limit: Request limit for current window
    - X-RateLimit-Remaining: Remaining requests
    - X-RateLimit-Reset: Unix timestamp when limit resets
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/ready", "/live"]:
        return await call_next(request)
    
    # Get rate limiter from app state
    rate_limiter: RateLimiter = request.app.state.rate_limiter
    
    # Get user ID from request (assumes authentication middleware ran first)
    user_id = getattr(request.state, "user_id", None)
    tenant_id = getattr(request.state, "tenant_id", None)
    
    if not user_id:
        # Use IP address as fallback
        user_id = f"ip:{request.client.host}"
    
    # Check rate limit
    try:
        allowed, info = await rate_limiter.check_rate_limit(user_id)
        
        if not allowed:
            # Rate limit exceeded
            reset_at = info.get("reset_at", 0)
            retry_after = int(reset_at - time.time())
            
            logger.warning(
                f"Rate limit exceeded for user {user_id}: "
                f"{info['current']}/{info['limit']} in {info['window']}"
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "window": info["window"],
                    "limit": info["limit"],
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at)
                }
            )
        
        # Check tenant quota (API calls)
        if tenant_id:
            quota_ok, quota_info = await rate_limiter.check_tenant_quota(
                tenant_id, "api_calls", 1.0
            )
            
            if not quota_ok:
                logger.warning(
                    f"Tenant {tenant_id} exceeded API quota: "
                    f"{quota_info['current']}/{quota_info['quota']}"
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Monthly API quota exceeded. Please upgrade your plan.",
                        "resource": "api_calls",
                        "quota": quota_info["quota"],
                        "month": quota_info["month"]
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        minute_info = info.get("requests_per_minute", {})
        response.headers["X-RateLimit-Limit"] = str(minute_info.get("limit", 60))
        response.headers["X-RateLimit-Remaining"] = str(minute_info.get("remaining", 60))
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        
        return response
        
    except Exception as e:
        logger.error(f"Rate limiting error: {e}", exc_info=True)
        # Allow request to proceed on rate limiter errors
        return await call_next(request)


# Create global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _rate_limiter = RateLimiter(redis_url)
    return _rate_limiter
