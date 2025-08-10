"""
日志工具
"""

import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """获取标准化的日志记录器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # 创建处理器
        handler = logging.StreamHandler(sys.stdout)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # 防止重复日志
        logger.propagate = False
    
    return logger