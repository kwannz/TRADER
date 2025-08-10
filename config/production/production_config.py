"""
生产环境配置管理器
提供生产级别的配置管理、环境变量处理、安全设置等
支持多环境配置、配置验证和动态配置更新
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import hashlib
from cryptography.fernet import Fernet
import motor.motor_asyncio
import redis.asyncio as aioredis
from loguru import logger
import yaml

class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class SecurityConfig:
    """安全配置"""
    secret_key: str
    jwt_secret: str
    encryption_key: str
    api_key_hash_salt: str
    session_timeout: int = 3600
    max_login_attempts: int = 5
    password_min_length: int = 8
    require_2fa: bool = True
    cors_origins: List[str] = None
    trusted_proxies: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = []
        if self.trusted_proxies is None:
            self.trusted_proxies = []

@dataclass
class DatabaseConfig:
    """数据库配置"""
    mongodb_url: str
    redis_url: str
    mongodb_max_pool_size: int = 100
    mongodb_min_pool_size: int = 10
    redis_max_connections: int = 50
    connection_timeout: int = 30
    query_timeout: int = 30
    enable_ssl: bool = True
    replica_set: Optional[str] = None
    read_preference: str = "primary"

@dataclass
class AIServiceConfig:
    """AI服务配置"""
    deepseek_api_key: str
    gemini_api_key: str
    max_requests_per_minute: int = 60
    request_timeout: int = 30
    retry_attempts: int = 3
    cache_duration: int = 300
    enable_rate_limiting: bool = True
    fallback_enabled: bool = True

@dataclass
class MonitoringConfig:
    """监控配置"""
    health_check_interval: int = 30
    metrics_retention_days: int = 30
    alert_webhook_url: Optional[str] = None
    enable_performance_tracking: bool = True
    log_level: str = "INFO"
    enable_distributed_tracing: bool = True
    metrics_export_interval: int = 60

@dataclass
class PerformanceConfig:
    """性能配置"""
    max_concurrent_requests: int = 1000
    request_timeout: int = 30
    websocket_max_connections: int = 10000
    cache_max_size: int = 1000000
    worker_processes: int = 4
    enable_compression: bool = True
    static_file_cache_max_age: int = 86400

@dataclass
class BackupConfig:
    """备份配置"""
    enable_automated_backup: bool = True
    backup_interval_hours: int = 6
    backup_retention_days: int = 30
    backup_storage_path: str = "/backups"
    enable_encryption: bool = True
    compress_backups: bool = True
    remote_backup_enabled: bool = False
    remote_backup_url: Optional[str] = None

class ProductionConfigManager:
    """生产环境配置管理器"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.config_path = Path("config/production")
        self.config_cache = {}
        self.encryption_cipher = None
        self.last_config_update = None
        
        # 配置文件路径
        self.config_files = {
            "security": self.config_path / "security.yaml",
            "database": self.config_path / "database.yaml", 
            "ai_services": self.config_path / "ai_services.yaml",
            "monitoring": self.config_path / "monitoring.yaml",
            "performance": self.config_path / "performance.yaml",
            "backup": self.config_path / "backup.yaml"
        }
        
        # 确保配置目录存在
        self.config_path.mkdir(parents=True, exist_ok=True)
        
    def _detect_environment(self) -> Environment:
        """检测当前环境"""
        env_name = os.getenv("TRADING_ENV", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"未知环境类型: {env_name}，使用development环境")
            return Environment.DEVELOPMENT
    
    def initialize_encryption(self, key: Optional[str] = None):
        """初始化加密"""
        if key:
            self.encryption_cipher = Fernet(key.encode())
        else:
            # 从环境变量或生成新密钥
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                encryption_key = Fernet.generate_key().decode()
                logger.warning("未找到ENCRYPTION_KEY环境变量，生成临时密钥")
            
            self.encryption_cipher = Fernet(encryption_key.encode())
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        if not self.encryption_cipher:
            raise RuntimeError("加密未初始化")
        
        return self.encryption_cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        if not self.encryption_cipher:
            raise RuntimeError("加密未初始化")
        
        return self.encryption_cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_security_config(self) -> SecurityConfig:
        """生成安全配置"""
        return SecurityConfig(
            secret_key=secrets.token_urlsafe(64),
            jwt_secret=secrets.token_urlsafe(64),
            encryption_key=Fernet.generate_key().decode(),
            api_key_hash_salt=secrets.token_urlsafe(32),
            session_timeout=3600 if self.environment == Environment.DEVELOPMENT else 1800,
            max_login_attempts=10 if self.environment == Environment.DEVELOPMENT else 5,
            require_2fa=False if self.environment == Environment.DEVELOPMENT else True,
            cors_origins=self._get_cors_origins(),
            trusted_proxies=self._get_trusted_proxies()
        )
    
    def _get_cors_origins(self) -> List[str]:
        """获取CORS原点配置"""
        if self.environment == Environment.DEVELOPMENT:
            return ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]
        elif self.environment == Environment.STAGING:
            return ["https://staging.trading-system.com"]
        else:  # PRODUCTION
            return ["https://trading-system.com", "https://www.trading-system.com"]
    
    def _get_trusted_proxies(self) -> List[str]:
        """获取信任的代理配置"""
        if self.environment == Environment.PRODUCTION:
            return ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]  # 内网IP
        return []
    
    def generate_database_config(self) -> DatabaseConfig:
        """生成数据库配置"""
        base_config = {
            "connection_timeout": 30,
            "query_timeout": 30,
            "enable_ssl": True,
            "read_preference": "primary"
        }
        
        if self.environment == Environment.DEVELOPMENT:
            return DatabaseConfig(
                mongodb_url="mongodb://localhost:27017/trading_dev",
                redis_url="redis://localhost:6379/0",
                mongodb_max_pool_size=20,
                mongodb_min_pool_size=2,
                redis_max_connections=10,
                enable_ssl=False,
                **base_config
            )
        elif self.environment == Environment.STAGING:
            return DatabaseConfig(
                mongodb_url=os.getenv("MONGODB_STAGING_URL", "mongodb://staging-db:27017/trading_staging"),
                redis_url=os.getenv("REDIS_STAGING_URL", "redis://staging-redis:6379/0"),
                mongodb_max_pool_size=50,
                mongodb_min_pool_size=5,
                redis_max_connections=25,
                **base_config
            )
        else:  # PRODUCTION
            return DatabaseConfig(
                mongodb_url=os.getenv("MONGODB_PRODUCTION_URL", "mongodb://prod-cluster:27017/trading_prod"),
                redis_url=os.getenv("REDIS_PRODUCTION_URL", "redis://prod-redis:6379/0"),
                mongodb_max_pool_size=100,
                mongodb_min_pool_size=10,
                redis_max_connections=50,
                replica_set="rs0",
                **base_config
            )
    
    def generate_ai_service_config(self) -> AIServiceConfig:
        """生成AI服务配置"""
        return AIServiceConfig(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            max_requests_per_minute=30 if self.environment == Environment.DEVELOPMENT else 60,
            request_timeout=30,
            retry_attempts=3,
            cache_duration=300 if self.environment != Environment.PRODUCTION else 600,
            enable_rate_limiting=True,
            fallback_enabled=True
        )
    
    def generate_monitoring_config(self) -> MonitoringConfig:
        """生成监控配置"""
        return MonitoringConfig(
            health_check_interval=60 if self.environment == Environment.DEVELOPMENT else 30,
            metrics_retention_days=7 if self.environment == Environment.DEVELOPMENT else 30,
            alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
            enable_performance_tracking=True,
            log_level="DEBUG" if self.environment == Environment.DEVELOPMENT else "INFO",
            enable_distributed_tracing=self.environment == Environment.PRODUCTION,
            metrics_export_interval=60
        )
    
    def generate_performance_config(self) -> PerformanceConfig:
        """生成性能配置"""
        if self.environment == Environment.DEVELOPMENT:
            return PerformanceConfig(
                max_concurrent_requests=100,
                worker_processes=1,
                websocket_max_connections=100,
                cache_max_size=100000,
                enable_compression=False
            )
        elif self.environment == Environment.STAGING:
            return PerformanceConfig(
                max_concurrent_requests=500,
                worker_processes=2,
                websocket_max_connections=1000,
                cache_max_size=500000,
                enable_compression=True
            )
        else:  # PRODUCTION
            return PerformanceConfig(
                max_concurrent_requests=1000,
                worker_processes=4,
                websocket_max_connections=10000,
                cache_max_size=1000000,
                enable_compression=True
            )
    
    def generate_backup_config(self) -> BackupConfig:
        """生成备份配置"""
        return BackupConfig(
            enable_automated_backup=True,
            backup_interval_hours=24 if self.environment == Environment.DEVELOPMENT else 6,
            backup_retention_days=7 if self.environment == Environment.DEVELOPMENT else 30,
            backup_storage_path=f"/backups/{self.environment.value}",
            enable_encryption=self.environment != Environment.DEVELOPMENT,
            compress_backups=True,
            remote_backup_enabled=self.environment == Environment.PRODUCTION,
            remote_backup_url=os.getenv("REMOTE_BACKUP_URL")
        )
    
    def save_config(self, config_type: str, config_data: Any):
        """保存配置到文件"""
        try:
            config_file = self.config_files[config_type]
            
            # 转换为字典
            if hasattr(config_data, '__dict__'):
                data = asdict(config_data)
            else:
                data = config_data
            
            # 加密敏感字段
            if config_type == "security":
                sensitive_fields = ["secret_key", "jwt_secret", "encryption_key", "api_key_hash_salt"]
                for field in sensitive_fields:
                    if field in data and self.encryption_cipher:
                        data[f"{field}_encrypted"] = self.encrypt_sensitive_data(data[field])
                        del data[field]
            
            elif config_type == "ai_services":
                sensitive_fields = ["deepseek_api_key", "gemini_api_key"]
                for field in sensitive_fields:
                    if field in data and data[field] and self.encryption_cipher:
                        data[f"{field}_encrypted"] = self.encrypt_sensitive_data(data[field])
                        del data[field]
            
            # 添加元数据
            data["_metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "environment": self.environment.value,
                "version": "1.0.0"
            }
            
            # 保存到文件
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存: {config_type} -> {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {config_type} - {e}")
            raise
    
    def load_config(self, config_type: str) -> Optional[Dict[str, Any]]:
        """从文件加载配置"""
        try:
            config_file = self.config_files[config_type]
            
            if not config_file.exists():
                logger.warning(f"配置文件不存在: {config_file}")
                return None
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 解密敏感字段
            if config_type == "security":
                sensitive_fields = ["secret_key", "jwt_secret", "encryption_key", "api_key_hash_salt"]
                for field in sensitive_fields:
                    encrypted_field = f"{field}_encrypted"
                    if encrypted_field in data and self.encryption_cipher:
                        data[field] = self.decrypt_sensitive_data(data[encrypted_field])
                        del data[encrypted_field]
            
            elif config_type == "ai_services":
                sensitive_fields = ["deepseek_api_key", "gemini_api_key"]
                for field in sensitive_fields:
                    encrypted_field = f"{field}_encrypted"
                    if encrypted_field in data and self.encryption_cipher:
                        data[field] = self.decrypt_sensitive_data(data[encrypted_field])
                        del data[encrypted_field]
            
            # 移除元数据
            data.pop("_metadata", None)
            
            return data
            
        except Exception as e:
            logger.error(f"加载配置失败: {config_type} - {e}")
            return None
    
    def initialize_all_configs(self):
        """初始化所有配置"""
        try:
            # 初始化加密
            self.initialize_encryption()
            
            # 生成并保存所有配置
            configs = {
                "security": self.generate_security_config(),
                "database": self.generate_database_config(),
                "ai_services": self.generate_ai_service_config(),
                "monitoring": self.generate_monitoring_config(),
                "performance": self.generate_performance_config(),
                "backup": self.generate_backup_config()
            }
            
            for config_type, config_data in configs.items():
                self.save_config(config_type, config_data)
                self.config_cache[config_type] = config_data
            
            logger.success(f"所有生产配置初始化完成 (环境: {self.environment.value})")
            
        except Exception as e:
            logger.error(f"初始化配置失败: {e}")
            raise
    
    def get_config(self, config_type: str) -> Optional[Any]:
        """获取配置"""
        # 首先从缓存获取
        if config_type in self.config_cache:
            return self.config_cache[config_type]
        
        # 从文件加载
        config_data = self.load_config(config_type)
        if config_data:
            # 转换为对应的配置对象
            if config_type == "security":
                config_obj = SecurityConfig(**config_data)
            elif config_type == "database":
                config_obj = DatabaseConfig(**config_data)
            elif config_type == "ai_services":
                config_obj = AIServiceConfig(**config_data)
            elif config_type == "monitoring":
                config_obj = MonitoringConfig(**config_data)
            elif config_type == "performance":
                config_obj = PerformanceConfig(**config_data)
            elif config_type == "backup":
                config_obj = BackupConfig(**config_data)
            else:
                config_obj = config_data
            
            self.config_cache[config_type] = config_obj
            return config_obj
        
        return None
    
    def validate_config(self, config_type: str, config_data: Any) -> List[str]:
        """验证配置"""
        errors = []
        
        try:
            if config_type == "security":
                if not config_data.secret_key or len(config_data.secret_key) < 32:
                    errors.append("secret_key必须至少32个字符")
                if not config_data.jwt_secret or len(config_data.jwt_secret) < 32:
                    errors.append("jwt_secret必须至少32个字符")
                if config_data.session_timeout < 300:
                    errors.append("session_timeout不应少于5分钟")
                    
            elif config_type == "database":
                if not config_data.mongodb_url:
                    errors.append("mongodb_url不能为空")
                if not config_data.redis_url:
                    errors.append("redis_url不能为空")
                if config_data.mongodb_max_pool_size < config_data.mongodb_min_pool_size:
                    errors.append("mongodb_max_pool_size必须大于等于mongodb_min_pool_size")
                    
            elif config_type == "ai_services":
                if not config_data.deepseek_api_key and not config_data.gemini_api_key:
                    errors.append("至少需要配置一个AI服务API密钥")
                if config_data.max_requests_per_minute <= 0:
                    errors.append("max_requests_per_minute必须大于0")
                    
            elif config_type == "performance":
                if config_data.max_concurrent_requests <= 0:
                    errors.append("max_concurrent_requests必须大于0")
                if config_data.worker_processes <= 0:
                    errors.append("worker_processes必须大于0")
                    
        except Exception as e:
            errors.append(f"配置验证异常: {e}")
        
        return errors
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """验证所有配置"""
        all_errors = {}
        
        for config_type in self.config_files.keys():
            config_data = self.get_config(config_type)
            if config_data:
                errors = self.validate_config(config_type, config_data)
                if errors:
                    all_errors[config_type] = errors
        
        return all_errors
    
    def export_config_template(self, output_path: str):
        """导出配置模板"""
        try:
            template_path = Path(output_path)
            template_path.mkdir(parents=True, exist_ok=True)
            
            # 生成示例配置
            example_configs = {
                "security": {
                    "secret_key": "your-secret-key-here",
                    "jwt_secret": "your-jwt-secret-here",
                    "encryption_key": "your-encryption-key-here",
                    "api_key_hash_salt": "your-salt-here",
                    "session_timeout": 3600,
                    "max_login_attempts": 5,
                    "require_2fa": True,
                    "cors_origins": ["https://your-domain.com"],
                    "trusted_proxies": ["10.0.0.0/8"]
                },
                "database": {
                    "mongodb_url": "mongodb://username:password@host:port/database",
                    "redis_url": "redis://username:password@host:port/0",
                    "mongodb_max_pool_size": 100,
                    "mongodb_min_pool_size": 10,
                    "redis_max_connections": 50,
                    "enable_ssl": True,
                    "replica_set": "rs0"
                },
                "ai_services": {
                    "deepseek_api_key": "your-deepseek-api-key",
                    "gemini_api_key": "your-gemini-api-key",
                    "max_requests_per_minute": 60,
                    "request_timeout": 30,
                    "enable_rate_limiting": True
                }
            }
            
            for config_name, config_data in example_configs.items():
                template_file = template_path / f"{config_name}.template.yaml"
                with open(template_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            logger.success(f"配置模板已导出到: {template_path}")
            
        except Exception as e:
            logger.error(f"导出配置模板失败: {e}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            "environment": self.environment.value,
            "config_path": str(self.config_path),
            "python_version": os.sys.version,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            "platform": os.sys.platform,
            "config_files_status": {
                name: path.exists() for name, path in self.config_files.items()
            },
            "environment_variables": {
                key: "***" if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower() 
                else value
                for key, value in os.environ.items()
                if key.startswith(("TRADING_", "MONGODB_", "REDIS_", "DEEPSEEK_", "GEMINI_"))
            }
        }

# 全局生产配置管理器
production_config_manager = ProductionConfigManager()

# 便捷函数
def get_production_config(config_type: str) -> Any:
    """获取生产配置"""
    return production_config_manager.get_config(config_type)

def initialize_production_environment():
    """初始化生产环境"""
    try:
        production_config_manager.initialize_all_configs()
        
        # 验证配置
        validation_errors = production_config_manager.validate_all_configs()
        if validation_errors:
            logger.error(f"配置验证失败: {validation_errors}")
            raise ValueError(f"配置验证失败: {validation_errors}")
        
        logger.success("生产环境初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"生产环境初始化失败: {e}")
        raise