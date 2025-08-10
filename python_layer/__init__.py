"""
Python业务逻辑层

集成Rust高性能引擎，提供完整的量化交易业务逻辑
"""

__version__ = "1.0.0"
__author__ = "Fullstack Engineer Agent"

# 导入核心模块
from . import core
from . import integrations
from . import models
from . import utils

# 导入主要类
from .core.ai_engine import AIEngine
from .core.data_manager import DataManager
from .core.strategy_manager import StrategyManager
from .core.system_monitor import SystemMonitor

__all__ = [
    "core",
    "integrations", 
    "models",
    "utils",
    "AIEngine",
    "DataManager",
    "StrategyManager",
    "SystemMonitor",
]