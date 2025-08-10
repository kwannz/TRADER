"""
Data Management Module
数据管理模块 - 统一数据访问接口
"""

from .unified_data_reader import (
    UnifiedDataReader, unified_data_reader,
    FactorDataReader, DatabaseHandler
)

__all__ = [
    'UnifiedDataReader',
    'unified_data_reader',
    'FactorDataReader',
    'DatabaseHandler'
]