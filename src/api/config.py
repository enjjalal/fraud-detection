"""
Configuration management for the Production Fraud Detection API.

This module provides:
- Environment-based configuration
- Security settings
- Rate limiting configuration
- Database and logging configuration
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Basic API settings
    title: str = "Fraud Detection API"
    description: str = "Production-Ready Credit Card Fraud Detection API"
    version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=1000, env="RATE_LIMIT_PER_HOUR")
    burst_limit: int = Field(default=20, env="BURST_LIMIT")
    
    # Model settings
    models_dir: str = Field(default="models/saved", env="MODELS_DIR")
    model_timeout_seconds: int = Field(default=30, env="MODEL_TIMEOUT")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_endpoint: str = Field(default="/metrics", env="METRICS_ENDPOINT")
    
    # Health check settings
    health_check_timeout: int = Field(default=5, env="HEALTH_CHECK_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class SecurityConfig:
    """Security configuration and utilities."""
    
    def __init__(self, settings: APISettings):
        self.settings = settings
        self.blocked_ips = set()
        self.api_keys = set()  # For future API key authentication
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.settings.debug
    
    def get_allowed_hosts(self) -> List[str]:
        """Get allowed hosts for the API."""
        return self.settings.allowed_hosts
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS allowed origins."""
        return self.settings.cors_origins
    
    def add_blocked_ip(self, ip: str):
        """Add IP to blocked list."""
        self.blocked_ips.add(ip)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        return ip in self.blocked_ips


class ModelConfig:
    """Model configuration and management."""
    
    def __init__(self, settings: APISettings):
        self.settings = settings
        self.models_dir = Path(settings.models_dir)
        self.timeout = settings.model_timeout_seconds
    
    def get_models_directory(self) -> Path:
        """Get models directory path."""
        return self.models_dir
    
    def get_model_files(self) -> dict:
        """Get available model files."""
        model_files = {
            'xgboost': ['XGBoost_optimized.joblib', 'XGBoost.joblib'],
            'lightgbm': ['LightGBM_optimized.joblib', 'LightGBM.joblib'],
            'catboost': ['CatBoost_optimized.joblib', 'CatBoost.joblib']
        }
        return model_files
    
    def get_scaler_path(self) -> Path:
        """Get scaler file path."""
        return self.models_dir / "scaler.joblib"


class LoggingConfig:
    """Logging configuration."""
    
    def __init__(self, settings: APISettings):
        self.settings = settings
    
    def get_log_config(self) -> dict:
        """Get logging configuration dictionary."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.settings.log_level,
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "": {
                    "level": self.settings.log_level,
                    "handlers": ["console"]
                },
                "api.requests": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False
                },
                "api.errors": {
                    "level": "ERROR",
                    "handlers": ["console"],
                    "propagate": False
                }
            }
        }
        
        # Add file handler if log file is specified
        if self.settings.log_file:
            config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "level": self.settings.log_level,
                "formatter": "detailed",
                "filename": self.settings.log_file
            }
            
            # Add file handler to all loggers
            for logger_config in config["loggers"].values():
                logger_config["handlers"].append("file")
        
        return config


# Global settings instance
settings = APISettings()

# Configuration instances
security_config = SecurityConfig(settings)
model_config = ModelConfig(settings)
logging_config = LoggingConfig(settings)


def get_settings() -> APISettings:
    """Get API settings instance."""
    return settings


def get_security_config() -> SecurityConfig:
    """Get security configuration instance."""
    return security_config


def get_model_config() -> ModelConfig:
    """Get model configuration instance."""
    return model_config


def get_logging_config() -> LoggingConfig:
    """Get logging configuration instance."""
    return logging_config
