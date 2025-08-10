"""
简单的日志工具 - 兼容性日志解决方案
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = "trader", level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # 设置级别
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
        console_handler.setLevel(log_level)
        
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = __name__) -> logging.Logger:
    """获取日志器实例"""
    return setup_logger(name)

# 创建默认日志器
logger = get_logger()