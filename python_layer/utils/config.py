"""
配置管理工具
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional

class APISettings(BaseSettings):
    """API配置"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    mongodb_url: str = Field(..., env="MONGODB_URL")
    redis_url: str = Field(..., env="REDIS_URL")

class AISettings(BaseSettings):
    """AI配置"""
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")

class TradingSettings(BaseSettings):
    """交易配置"""
    okx_api_key: Optional[str] = Field(None, env="OKX_API_KEY")
    okx_secret_key: Optional[str] = Field(None, env="OKX_SECRET_KEY")
    okx_passphrase: Optional[str] = Field(None, env="OKX_PASSPHRASE")
    binance_api_key: Optional[str] = Field(None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(None, env="BINANCE_SECRET_KEY")
    max_position_size: float = Field(default=0.8, env="MAX_POSITION_SIZE")
    hard_stop_loss: int = Field(default=300, env="HARD_STOP_LOSS")

class Settings(BaseSettings):
    """主配置"""
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    api: APISettings = APISettings()
    database: DatabaseSettings = DatabaseSettings()
    ai: AISettings = AISettings()
    trading: TradingSettings = TradingSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    """获取设置实例"""
    return Settings()