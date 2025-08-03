"""
Production middleware components for the Fraud Detection API.

This module provides:
- Rate limiting middleware
- Security middleware
- Logging middleware
- Error handling middleware
"""

import time
import logging
from typing import Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import uuid
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class RateLimitMiddleware:
    """Advanced rate limiting middleware with multiple strategies."""
    
    def __init__(self, 
                 requests_per_minute: int = 100,
                 requests_per_hour: int = 1000,
                 burst_limit: int = 20):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit
        
        # Storage for different time windows
        self.minute_storage = defaultdict(lambda: deque())
        self.hour_storage = defaultdict(lambda: deque())
        self.burst_storage = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def _clean_old_requests(self, storage: Dict, window_seconds: int):
        """Clean old requests from storage."""
        now = time.time()
        cutoff = now - window_seconds
        
        for client_id in list(storage.keys()):
            while storage[client_id] and storage[client_id][0] < cutoff:
                storage[client_id].popleft()
            
            # Remove empty deques
            if not storage[client_id]:
                del storage[client_id]
    
    def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed and return rate limit info."""
        now = time.time()
        
        with self.lock:
            # Clean old requests
            self._clean_old_requests(self.minute_storage, 60)
            self._clean_old_requests(self.hour_storage, 3600)
            self._clean_old_requests(self.burst_storage, 10)  # 10-second burst window
            
            # Check limits
            minute_count = len(self.minute_storage[client_id])
            hour_count = len(self.hour_storage[client_id])
            burst_count = len(self.burst_storage[client_id])
            
            # Rate limit info
            rate_limit_info = {
                "requests_per_minute": minute_count,
                "requests_per_hour": hour_count,
                "burst_requests": burst_count,
                "limits": {
                    "per_minute": self.requests_per_minute,
                    "per_hour": self.requests_per_hour,
                    "burst": self.burst_limit
                }
            }
            
            # Check if any limit is exceeded
            if (minute_count >= self.requests_per_minute or
                hour_count >= self.requests_per_hour or
                burst_count >= self.burst_limit):
                return False, rate_limit_info
            
            # Add current request to all storages
            self.minute_storage[client_id].append(now)
            self.hour_storage[client_id].append(now)
            self.burst_storage[client_id].append(now)
            
            return True, rate_limit_info


class SecurityMiddleware:
    """Security middleware for API protection."""
    
    def __init__(self):
        self.blocked_ips = set()
        self.suspicious_patterns = [
            "script", "alert", "javascript:", "vbscript:",
            "<script", "</script>", "onload=", "onerror="
        ]
    
    def check_request_security(self, request: Request) -> bool:
        """Check if request passes security checks."""
        client_ip = self.get_client_ip(request)
        
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            return False
        
        # Check for suspicious patterns in query parameters
        for param_value in request.query_params.values():
            if any(pattern.lower() in param_value.lower() 
                   for pattern in self.suspicious_patterns):
                logger.warning(f"Suspicious pattern detected from {client_ip}: {param_value}")
                return False
        
        return True
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class RequestLoggingMiddleware:
    """Middleware for comprehensive request logging."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.requests")
    
    def log_request(self, request: Request, response_time: float, status_code: int):
        """Log request details."""
        client_ip = request.headers.get("X-Forwarded-For", 
                                       request.client.host if request.client else "unknown")
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", "unknown"),
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
        
        if status_code >= 400:
            self.logger.warning(f"Request failed: {log_data}")
        else:
            self.logger.info(f"Request processed: {log_data}")


class ErrorHandlingMiddleware:
    """Comprehensive error handling middleware."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.errors")
    
    def handle_validation_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle Pydantic validation errors."""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        error_detail = {
            "error": "Validation Error",
            "message": "Invalid input data",
            "details": str(exc),
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        self.logger.warning(f"Validation error: {error_detail}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_detail
        )
    
    def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        error_detail = {
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        self.logger.error(f"HTTP exception: {error_detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_detail
        )
    
    def handle_general_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions."""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        error_detail = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        self.logger.error(f"Unexpected error: {error_detail}, Exception: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_detail
        )


# Global middleware instances
rate_limiter = RateLimitMiddleware()
security_middleware = SecurityMiddleware()
logging_middleware = RequestLoggingMiddleware()
error_handler = ErrorHandlingMiddleware()
