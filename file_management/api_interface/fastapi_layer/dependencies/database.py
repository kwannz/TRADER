"""
数据库依赖注入
提供数据库连接和管理器
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_manager import data_manager
from core.real_data_manager import real_data_manager

async def get_database_manager():
    """获取数据库管理器"""
    if not data_manager._initialized:
        await data_manager.initialize()
    return data_manager

async def get_real_data_manager():
    """获取真实数据管理器"""
    if not real_data_manager.websocket_manager:
        await real_data_manager.initialize()
    return real_data_manager

def get_mongodb_client():
    """获取MongoDB客户端"""
    return data_manager.client

def get_mongodb_database():
    """获取MongoDB数据库"""
    return data_manager.db

def get_redis_client():
    """获取Redis客户端"""
    if hasattr(data_manager, 'cache_manager') and data_manager.cache_manager:
        return data_manager.cache_manager.redis_client
    return None