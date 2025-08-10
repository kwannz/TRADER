"""
AI量化交易系统 - 核心配置管理
支持环境变量、配置文件和运行时配置
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

class DatabaseSettings(BaseSettings):
    """数据库配置"""
    mongodb_url: str = "mongodb://admin:quantum_2025@localhost:27017/quantum_trader?authSource=admin"
    redis_url: str = "redis://:quantum_redis_2025@localhost:6379/0"
    
    # 数据库连接池配置
    mongodb_max_pool_size: int = 50
    mongodb_min_pool_size: int = 10
    redis_max_connections: int = 20

class AIAPISettings(BaseSettings):
    """AI API配置"""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 2000
    deepseek_timeout: int = 30
    
    gemini_api_key: str = ""
    gemini_model: str = "gemini-pro"
    gemini_max_tokens: int = 2000
    gemini_timeout: int = 30

class ExchangeSettings(BaseSettings):
    """交易所API配置"""
    # OKX配置
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_sandbox: bool = True
    
    # Binance配置
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_sandbox: bool = True
    
    # 交易配置
    max_position_size: float = 0.8  # 最大仓位80%
    hard_stop_loss: float = 300.0   # 硬止损300 USDT
    initial_balance: float = 500.0  # 初始资金500 USDT

class TradingSettings(BaseSettings):
    """交易系统配置"""
    # 策略配置
    max_concurrent_strategies: int = 5
    strategy_check_interval: int = 1  # 秒
    risk_check_interval: int = 5      # 秒
    
    # 性能配置
    order_timeout: int = 30           # 订单超时30秒
    max_order_latency_ms: int = 500   # 最大下单延迟500ms
    websocket_reconnect_delay: int = 5 # WebSocket重连间隔
    
    # 数据配置
    historical_days: int = 180        # 历史数据180天
    refresh_rate_hz: int = 4          # 数据刷新频率4Hz

class CLISettings(BaseSettings):
    """CLI界面配置"""
    terminal_theme: str = "bloomberg"
    min_terminal_width: int = 120
    min_terminal_height: int = 40
    
    # 颜色配置
    primary_color: str = "#0D1B2A"      # 深海蓝背景
    secondary_color: str = "#1B263B"    # 面板背景
    accent_color: str = "#277DA1"       # 品牌蓝
    success_color: str = "#52B788"      # 成功绿
    warning_color: str = "#F2CC8F"      # 警告黄
    error_color: str = "#E07A5F"        # 错误橙红

class CTBenchSettings(BaseSettings):
    """CTBench时间序列生成配置"""
    # 模型配置
    enabled_models: List[str] = [
        "TimeVAE", "KoVAE", "QuantGAN", 
        "COSCI-GAN", "DiffusionTS", "FIDE",
        "FourierFlow", "LS4"
    ]
    
    # 训练配置
    sequence_length: int = 100
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    device: str = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
    
    # 因子配置
    alpha101_factors: int = 30
    custom_factors_enabled: bool = True
    factor_ic_threshold: float = 0.02
    factor_pvalue_threshold: float = 0.05

class LoggingSettings(BaseSettings):
    """日志配置"""
    log_level: str = "INFO"
    log_file: str = "logs/quantum_trader.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

class Settings(BaseSettings):
    """主配置类 - 聚合所有配置"""
    
    # 基本信息
    app_name: str = "AI量化交易系统"
    version: str = "1.0.0"
    description: str = "AI驱动的个人量化交易CLI终端"
    debug: bool = True
    
    # 数据库配置
    mongodb_url: str = "mongodb://admin:quantum_2025@localhost:27017/quantum_trader?authSource=admin"
    redis_url: str = "redis://:quantum_redis_2025@localhost:6379/0"
    mongodb_max_pool_size: int = 50
    mongodb_min_pool_size: int = 10
    redis_max_connections: int = 20
    
    # AI API配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 2000
    deepseek_timeout: int = 30
    gemini_api_key: str = ""
    gemini_model: str = "gemini-pro"
    gemini_max_tokens: int = 2000
    gemini_timeout: int = 30
    
    # 交易所配置
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_sandbox: bool = True
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_sandbox: bool = True
    max_position_size: float = 0.8
    hard_stop_loss: float = 300.0
    initial_balance: float = 500.0
    
    # 数据源API配置
    coinglass_api_key: str = ""
    jin10_api_key: str = ""
    
    # 交易系统配置
    max_concurrent_strategies: int = 5
    strategy_check_interval: int = 1
    risk_check_interval: int = 5
    order_timeout: int = 30
    max_order_latency_ms: int = 500
    websocket_reconnect_delay: int = 5
    historical_days: int = 180
    refresh_rate_hz: int = 4
    
    # CLI界面配置
    terminal_theme: str = "bloomberg"
    min_terminal_width: int = 120
    min_terminal_height: int = 40
    primary_color: str = "#0D1B2A"
    secondary_color: str = "#1B263B"
    accent_color: str = "#277DA1"
    success_color: str = "#52B788"
    warning_color: str = "#F2CC8F"
    error_color: str = "#E07A5F"
    
    # CTBench配置
    enabled_models: List[str] = [
        "TimeVAE", "KoVAE", "QuantGAN", 
        "COSCI-GAN", "DiffusionTS", "FIDE",
        "FourierFlow", "LS4"
    ]
    sequence_length: int = 100
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    device: str = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
    alpha101_factors: int = 30
    custom_factors_enabled: bool = True
    factor_ic_threshold: float = 0.02
    factor_pvalue_threshold: float = 0.05
    
    # 日志配置
    log_level: str = "DEBUG"
    log_file: str = "logs/quantum_trader.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置loguru日志系统"""
        logger.remove()  # 移除默认配置
        
        # 创建日志目录
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 控制台输出
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=self.log_level,
            format=self.log_format,
            colorize=True
        )
        
        # 文件输出
        logger.add(
            sink=self.log_file,
            level=self.log_level,
            format=self.log_format,
            rotation=self.log_rotation,
            retention=self.log_retention,
            compression="zip"
        )
    
    @validator("debug", pre=True)
    def parse_debug(cls, v):
        """解析debug模式"""
        if isinstance(v, str):
            return v.lower() in ["true", "1", "yes", "on"]
        return v
    
    def get_api_config(self, exchange: str) -> Dict[str, Any]:
        """获取交易所API配置"""
        if exchange.lower() == "okx":
            return {
                "apiKey": self.okx_api_key,
                "secret": self.okx_secret_key,
                "password": self.okx_passphrase,
                "sandbox": self.okx_sandbox,
            }
        elif exchange.lower() == "binance":
            return {
                "apiKey": self.binance_api_key,
                "secret": self.binance_secret_key,
                "sandbox": self.binance_sandbox,
            }
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    
    def get_ai_config(self, provider: str) -> Dict[str, Any]:
        """获取AI API配置"""
        if provider.lower() == "deepseek":
            return {
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model,
                "max_tokens": self.deepseek_max_tokens,
                "timeout": self.deepseek_timeout,
            }
        elif provider.lower() == "gemini":
            return {
                "api_key": self.gemini_api_key,
                "model": self.gemini_model,
                "max_tokens": self.gemini_max_tokens,
                "timeout": self.gemini_timeout,
            }
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
    
    def validate_config(self) -> List[str]:
        """验证配置完整性，返回缺失的配置项"""
        missing = []
        
        # 检查AI API密钥
        if not self.deepseek_api_key:
            missing.append("DeepSeek API Key")
        if not self.gemini_api_key:
            missing.append("Gemini API Key")
            
        # 检查交易所API密钥
        if not self.okx_api_key:
            missing.append("OKX API credentials")
        if not self.binance_api_key:
            missing.append("Binance API credentials")
            
        return missing

# 全局设置实例
settings = Settings()

# 导出常用配置
__all__ = [
    "settings",
    "Settings", 
    "DatabaseSettings",
    "AIAPISettings", 
    "ExchangeSettings",
    "TradingSettings",
    "CLISettings",
    "CTBenchSettings",
    "LoggingSettings"
]