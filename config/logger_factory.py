"""
日志工厂 - 基于配置创建logger实例
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from core.unified_logger import (
    UnifiedLogger, 
    LogLevel, 
    LogCategory,
    ConsoleOutput,
    FileOutput, 
    JsonFileOutput,
    DatabaseOutput,
    LogFilter
)


class LoggerFactory:
    """日志记录器工厂"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path(__file__).parent / "logging_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载日志配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 环境变量替换
            config = self._substitute_env_vars(config)
            return config
            
        except Exception as e:
            print(f"加载日志配置失败: {e}")
            return self._get_default_config()
    
    def _substitute_env_vars(self, obj):
        """递归替换环境变量"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # 替换 ${VAR_NAME} 格式的环境变量
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replace_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(pattern, replace_var, obj)
        else:
            return obj
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "logging": {
                "level": "INFO",
                "console": {"enabled": True, "colorize": True},
                "file": {
                    "enabled": True,
                    "directory": "logs",
                    "system_log": {
                        "filename": "system.log",
                        "max_bytes": 50 * 1024 * 1024,
                        "backup_count": 5,
                        "level": "DEBUG"
                    }
                },
                "json_file": {
                    "enabled": True,
                    "directory": "logs",
                    "structured_log": {
                        "filename": "system_structured.jsonl",
                        "max_bytes": 50 * 1024 * 1024,
                        "backup_count": 5
                    }
                }
            }
        }
    
    def create_logger(self, name: str = "trader_system") -> UnifiedLogger:
        """创建配置化的日志记录器"""
        logger = UnifiedLogger(name)
        logging_config = self.config.get("logging", {})
        
        # 添加控制台输出
        if logging_config.get("console", {}).get("enabled", True):
            console_config = logging_config["console"]
            console_output = ConsoleOutput(
                colorize=console_config.get("colorize", True)
            )
            
            # 设置控制台过滤器
            if "level" in console_config:
                console_output.filter.min_level = LogLevel(console_config["level"])
            
            logger.add_output(console_output)
        
        # 添加文件输出
        if logging_config.get("file", {}).get("enabled", True):
            file_config = logging_config["file"]
            log_dir = Path(file_config.get("directory", "logs"))
            log_dir.mkdir(exist_ok=True)
            
            # 系统日志文件
            if "system_log" in file_config:
                sys_config = file_config["system_log"]
                file_output = FileOutput(
                    log_dir / sys_config["filename"],
                    max_bytes=sys_config.get("max_bytes", 50 * 1024 * 1024),
                    backup_count=sys_config.get("backup_count", 5)
                )
                
                if "level" in sys_config:
                    file_output.filter.min_level = LogLevel(sys_config["level"])
                
                logger.add_output(file_output)
            
            # 错误日志文件
            if "error_log" in file_config:
                error_config = file_config["error_log"]
                error_output = FileOutput(
                    log_dir / error_config["filename"],
                    max_bytes=error_config.get("max_bytes", 10 * 1024 * 1024),
                    backup_count=error_config.get("backup_count", 10)
                )
                error_output.filter.min_level = LogLevel.ERROR
                logger.add_output(error_output)
            
            # 分类日志文件
            for log_type in ["trading_log", "api_log", "ai_log", "performance_log"]:
                if log_type in file_config:
                    type_config = file_config[log_type]
                    type_output = FileOutput(
                        log_dir / type_config["filename"],
                        max_bytes=type_config.get("max_bytes", 20 * 1024 * 1024),
                        backup_count=type_config.get("backup_count", 3)
                    )
                    
                    # 设置分类过滤器
                    if "category" in type_config:
                        try:
                            category = LogCategory(type_config["category"])
                            type_output.filter.categories = [category]
                        except ValueError:
                            pass
                    
                    if "level" in type_config:
                        type_output.filter.min_level = LogLevel(type_config["level"])
                    
                    logger.add_output(type_output)
        
        # 添加JSON文件输出
        if logging_config.get("json_file", {}).get("enabled", True):
            json_config = logging_config["json_file"]
            log_dir = Path(json_config.get("directory", "logs"))
            log_dir.mkdir(exist_ok=True)
            
            # 结构化日志文件
            if "structured_log" in json_config:
                struct_config = json_config["structured_log"]
                json_output = JsonFileOutput(
                    log_dir / struct_config["filename"],
                    max_bytes=struct_config.get("max_bytes", 50 * 1024 * 1024),
                    backup_count=struct_config.get("backup_count", 5)
                )
                
                if "level" in struct_config:
                    json_output.filter.min_level = LogLevel(struct_config["level"])
                
                logger.add_output(json_output)
            
            # 交易结构化日志
            if "trading_structured" in json_config:
                trading_config = json_config["trading_structured"]
                trading_json_output = JsonFileOutput(
                    log_dir / trading_config["filename"],
                    max_bytes=trading_config.get("max_bytes", 20 * 1024 * 1024),
                    backup_count=trading_config.get("backup_count", 3)
                )
                trading_json_output.filter.categories = [LogCategory.TRADING]
                logger.add_output(trading_json_output)
        
        # 添加数据库输出
        if logging_config.get("database", {}).get("enabled", False):
            db_config = logging_config["database"]
            mongodb_config = db_config.get("mongodb", {})
            redis_config = db_config.get("redis", {})
            
            if mongodb_config.get("url") and redis_config.get("url"):
                db_output = DatabaseOutput(
                    mongodb_config["url"],
                    redis_config["url"]
                )
                # 数据库输出需要异步初始化
                import asyncio
                asyncio.create_task(db_output.initialize())
                logger.add_output(db_output)
        
        return logger
    
    def get_category_config(self, category: str) -> Dict[str, Any]:
        """获取特定分类的配置"""
        categories = self.config.get("logging", {}).get("categories", {})
        return categories.get(category, {})
    
    def get_filter_config(self, filter_name: str) -> Optional[LogFilter]:
        """根据配置创建过滤器"""
        filters_config = self.config.get("logging", {}).get("filters", {})
        if filter_name not in filters_config:
            return None
        
        filter_config = filters_config[filter_name]
        filter_obj = LogFilter()
        
        if "min_level" in filter_config:
            filter_obj.min_level = LogLevel(filter_config["min_level"])
        
        if "max_level" in filter_config:
            filter_obj.max_level = LogLevel(filter_config["max_level"])
        
        if "categories" in filter_config:
            filter_obj.categories = [LogCategory(cat) for cat in filter_config["categories"]]
        
        if "modules" in filter_config:
            filter_obj.modules = filter_config["modules"]
        
        if "tags" in filter_config:
            filter_obj.tags = filter_config["tags"]
        
        return filter_obj
    
    def is_feature_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        features = self.config.get("features", {})
        return features.get(feature, False)
    
    def get_retention_config(self) -> Dict[str, Any]:
        """获取日志保留配置"""
        return self.config.get("retention", {})
    
    def get_alerts_config(self) -> Dict[str, Any]:
        """获取告警配置"""
        return self.config.get("alerts", {})


# 全局工厂实例
logger_factory = LoggerFactory()

# 便捷函数
def create_configured_logger(name: str = "trader_system") -> UnifiedLogger:
    """创建配置化的日志记录器"""
    return logger_factory.create_logger(name)

def get_category_level(category: str) -> LogLevel:
    """获取分类的默认日志级别"""
    config = logger_factory.get_category_config(category)
    level_str = config.get("default_level", "INFO")
    return LogLevel(level_str)

def is_logging_feature_enabled(feature: str) -> bool:
    """检查日志功能是否启用"""
    return logger_factory.is_feature_enabled(feature)