"""
Python层数据管理器
封装和代理核心数据管理器功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_manager import data_manager as core_data_manager

class DataManager:
    """Python层数据管理器"""
    
    def __init__(self):
        self.core_manager = core_data_manager
        self._initialized = False
    
    async def initialize(self):
        """初始化数据管理器"""
        if not self._initialized:
            await self.core_manager.initialize()
            self._initialized = True
    
    async def close(self):
        """关闭数据管理器"""
        if hasattr(self.core_manager, 'close'):
            await self.core_manager.close()
        self._initialized = False
    
    async def get_stats(self):
        """获取统计信息"""
        if hasattr(self.core_manager, 'get_stats'):
            return await self.core_manager.get_stats()
        return {}
    
    async def reset_stats(self):
        """重置统计信息"""
        if hasattr(self.core_manager, 'reset_stats'):
            await self.core_manager.reset_stats()
    
    async def get_latest_market_data(self):
        """获取最新市场数据"""
        # 这里可以调用real_data_manager或其他数据源
        from core.real_data_manager import real_data_manager
        return await real_data_manager.get_latest_prices()
    
    @property
    def is_initialized(self):
        return self._initialized