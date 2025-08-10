"""
市场模拟器模块
导出MarketTick, KLineData类和market_simulator实例
"""

from .trading_simulator import MarketTick, KLineData, TradingSimulator

# 创建全局market_simulator实例
market_simulator = TradingSimulator(mode="simulation")

# 导出类和实例
__all__ = ['MarketTick', 'KLineData', 'market_simulator']