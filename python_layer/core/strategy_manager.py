"""
Python层策略管理器
管理交易策略的执行和监控
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum

class StrategyStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"

class Strategy:
    """交易策略基类"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.status = StrategyStatus.INACTIVE
        self.created_at = datetime.utcnow()
        self.last_execution = None
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_pnl = 0.0
    
    async def execute(self, market_data: Dict) -> Dict[str, Any]:
        """执行策略逻辑"""
        self.execution_count += 1
        self.last_execution = datetime.utcnow()
        
        try:
            # 子类应该重写此方法
            result = await self._run_strategy(market_data)
            self.success_count += 1
            return result
        except Exception as e:
            self.error_count += 1
            return {"error": str(e), "success": False}
    
    async def _run_strategy(self, market_data: Dict) -> Dict[str, Any]:
        """策略具体实现（子类重写）"""
        return {"signal": "hold", "confidence": 0.5, "success": True}

class MACDStrategy(Strategy):
    """MACD策略示例"""
    
    def __init__(self):
        super().__init__("MACD Strategy", "基于MACD指标的交易策略")
    
    async def _run_strategy(self, market_data: Dict) -> Dict[str, Any]:
        """MACD策略实现"""
        # 模拟MACD计算和信号生成
        await asyncio.sleep(0.1)  # 模拟计算时间
        
        # 简化的信号逻辑
        price_trend = market_data.get("trend", "neutral")
        
        if price_trend == "bullish":
            return {
                "signal": "buy",
                "confidence": 0.75,
                "reason": "MACD金叉信号",
                "success": True
            }
        elif price_trend == "bearish":
            return {
                "signal": "sell", 
                "confidence": 0.70,
                "reason": "MACD死叉信号",
                "success": True
            }
        else:
            return {
                "signal": "hold",
                "confidence": 0.50,
                "reason": "MACD中性信号",
                "success": True
            }

class RSIStrategy(Strategy):
    """RSI策略示例"""
    
    def __init__(self):
        super().__init__("RSI Strategy", "基于RSI指标的交易策略")
    
    async def _run_strategy(self, market_data: Dict) -> Dict[str, Any]:
        """RSI策略实现"""
        await asyncio.sleep(0.05)
        
        # 模拟RSI值
        rsi_value = 45 + (hash(str(market_data)) % 40)  # 模拟RSI 45-85
        
        if rsi_value > 70:
            return {
                "signal": "sell",
                "confidence": 0.80,
                "reason": f"RSI超买 ({rsi_value})",
                "rsi": rsi_value,
                "success": True
            }
        elif rsi_value < 30:
            return {
                "signal": "buy",
                "confidence": 0.85,
                "reason": f"RSI超卖 ({rsi_value})",
                "rsi": rsi_value,
                "success": True
            }
        else:
            return {
                "signal": "hold",
                "confidence": 0.40,
                "reason": f"RSI中性 ({rsi_value})",
                "rsi": rsi_value,
                "success": True
            }

class StrategyManager:
    """策略管理器"""
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: List[str] = []
        self.execution_history: List[Dict] = []
        self.is_running = False
    
    async def initialize(self):
        """初始化策略管理器"""
        # 注册默认策略
        self.register_strategy(MACDStrategy())
        self.register_strategy(RSIStrategy())
    
    def register_strategy(self, strategy: Strategy):
        """注册策略"""
        self.strategies[strategy.name] = strategy
    
    def activate_strategy(self, strategy_name: str):
        """激活策略"""
        if strategy_name in self.strategies:
            if strategy_name not in self.active_strategies:
                self.active_strategies.append(strategy_name)
                self.strategies[strategy_name].status = StrategyStatus.ACTIVE
                return True
        return False
    
    def deactivate_strategy(self, strategy_name: str):
        """停用策略"""
        if strategy_name in self.active_strategies:
            self.active_strategies.remove(strategy_name)
            if strategy_name in self.strategies:
                self.strategies[strategy_name].status = StrategyStatus.INACTIVE
            return True
        return False
    
    async def execute_strategies(self, market_data: Dict) -> Dict[str, Any]:
        """执行所有活跃策略"""
        results = {}
        
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                try:
                    result = await strategy.execute(market_data)
                    results[strategy_name] = result
                    
                    # 记录执行历史
                    self.execution_history.append({
                        "strategy": strategy_name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "result": result,
                        "market_data": market_data
                    })
                    
                    # 限制历史记录长度
                    if len(self.execution_history) > 1000:
                        self.execution_history = self.execution_history[-500:]
                        
                except Exception as e:
                    results[strategy_name] = {
                        "error": str(e),
                        "success": False
                    }
        
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取策略统计"""
        stats = {
            "total_strategies": len(self.strategies),
            "active_strategies": len(self.active_strategies),
            "execution_history_count": len(self.execution_history)
        }
        
        # 各策略统计
        strategy_stats = {}
        for name, strategy in self.strategies.items():
            strategy_stats[name] = {
                "status": strategy.status.value,
                "execution_count": strategy.execution_count,
                "success_count": strategy.success_count,
                "error_count": strategy.error_count,
                "success_rate": (
                    strategy.success_count / strategy.execution_count * 100
                    if strategy.execution_count > 0 else 0
                ),
                "total_pnl": strategy.total_pnl,
                "last_execution": (
                    strategy.last_execution.isoformat() 
                    if strategy.last_execution else None
                ),
                "created_at": strategy.created_at.isoformat()
            }
        
        stats["strategies"] = strategy_stats
        return stats
    
    async def reset_stats(self):
        """重置统计信息"""
        for strategy in self.strategies.values():
            strategy.execution_count = 0
            strategy.success_count = 0
            strategy.error_count = 0
            strategy.total_pnl = 0.0
            strategy.last_execution = None
        
        self.execution_history.clear()
    
    def get_strategy_list(self) -> List[Dict[str, Any]]:
        """获取策略列表"""
        return [
            {
                "name": strategy.name,
                "description": strategy.description,
                "status": strategy.status.value,
                "execution_count": strategy.execution_count,
                "success_rate": (
                    strategy.success_count / strategy.execution_count * 100
                    if strategy.execution_count > 0 else 0
                )
            }
            for strategy in self.strategies.values()
        ]
    
    def get_execution_history(self, limit: int = 50) -> List[Dict]:
        """获取执行历史"""
        return self.execution_history[-limit:] if self.execution_history else []