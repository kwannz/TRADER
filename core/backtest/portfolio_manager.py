"""
投资组合管理器

提供投资组合管理和资产配置功能：
- 持仓管理和跟踪
- 资产配置和再平衡
- 风险控制和限额管理
- 投资组合价值计算
- 绩效归因分析
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from .trading_environment import Position, PositionSide
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class AssetAllocation:
    """资产配置"""
    symbol: str
    target_weight: float  # 目标权重
    current_weight: float = 0.0  # 当前权重
    deviation: float = 0.0  # 偏离度
    rebalance_threshold: float = 0.05  # 再平衡阈值5%

@dataclass
class PortfolioMetrics:
    """投资组合指标"""
    total_value: float
    cash_balance: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    total_return: float
    daily_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    exposure_ratio: float  # 仓位使用率

@dataclass
class RiskLimits:
    """风险限额"""
    max_position_size: float = 0.10  # 单一持仓最大占比10%
    max_sector_exposure: float = 0.30  # 单一行业最大敞口30%
    max_total_leverage: float = 1.0  # 最大杠杆1倍
    max_daily_loss: float = 0.02  # 最大日亏损2%
    max_drawdown_limit: float = 0.10  # 最大回撤限制10%
    var_limit: float = 0.05  # VaR限额5%

class RebalanceStrategy(Enum):
    """再平衡策略"""
    NONE = "none"  # 不再平衡
    PERIODIC = "periodic"  # 定期再平衡
    THRESHOLD = "threshold"  # 阈值再平衡
    VOLATILITY = "volatility"  # 波动率再平衡

class PortfolioManager:
    """投资组合管理器主类"""
    
    def __init__(self, initial_balance: float = 10000.0,
                 currency: str = "USDT",
                 rebalance_strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD):
        
        # 基础配置
        self.initial_balance = initial_balance
        self.currency = currency
        self.rebalance_strategy = rebalance_strategy
        
        # 投资组合状态
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.asset_allocations: Dict[str, AssetAllocation] = {}
        
        # 历史数据
        self.value_history: deque = deque(maxlen=1000)
        self.return_history: deque = deque(maxlen=252)  # 一年的交易日
        self.drawdown_history: deque = deque(maxlen=1000)
        
        # 风险管理
        self.risk_limits = RiskLimits()
        self.daily_pnl = 0.0
        self.peak_value = initial_balance
        self.current_drawdown = 0.0
        
        # 市场数据缓存
        self.current_prices: Dict[str, float] = {}
        self.historical_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # 再平衡控制
        self.last_rebalance_time: Optional[datetime] = None
        self.rebalance_frequency = timedelta(days=7)  # 每周检查一次
        
        logger.info(f"投资组合管理器初始化: 初始资金={initial_balance} {currency}")
    
    async def initialize(self) -> None:
        """初始化投资组合管理器"""
        try:
            # 重置所有状态
            self.cash_balance = self.initial_balance
            self.positions.clear()
            self.asset_allocations.clear()
            self.current_prices.clear()
            self.historical_returns.clear()
            
            # 清空历史数据
            self.value_history.clear()
            self.return_history.clear()
            self.drawdown_history.clear()
            
            # 重置统计数据
            self.daily_pnl = 0.0
            self.peak_value = self.initial_balance
            self.current_drawdown = 0.0
            self.last_rebalance_time = None
            
            logger.info("投资组合管理器初始化完成")
            
        except Exception as e:
            logger.error(f"投资组合管理器初始化失败: {e}")
            raise
    
    # 投资组合管理
    async def update_positions(self, positions: Dict[str, Position]) -> None:
        """更新持仓信息"""
        try:
            self.positions = positions.copy()
            
            # 记录当前组合价值
            total_value = await self.calculate_total_value()
            await self._record_portfolio_value(total_value)
            
            # 更新资产配置权重
            await self._update_allocation_weights()
            
            # 检查风险限额
            await self._check_risk_limits()
            
        except Exception as e:
            logger.error(f"更新持仓信息失败: {e}")
    
    async def update_portfolio_value(self, market_data: Dict[str, Any]) -> None:
        """更新投资组合价值"""
        try:
            # 更新市场价格
            symbol = market_data['symbol']
            if market_data['data_type'] == 'kline':
                self.current_prices[symbol] = market_data['data']['close']
            elif 'price' in market_data['data']:
                self.current_prices[symbol] = market_data['data']['price']
            
            # 更新历史收益率
            await self._update_historical_returns(symbol, market_data)
            
            # 重新计算投资组合价值
            total_value = await self.calculate_total_value()
            await self._record_portfolio_value(total_value)
            
        except Exception as e:
            logger.error(f"更新投资组合价值失败: {e}")
    
    async def calculate_total_value(self) -> float:
        """计算投资组合总价值"""
        try:
            total_value = self.cash_balance
            
            # 加上所有持仓的市值
            for symbol, position in self.positions.items():
                if symbol in self.current_prices:
                    current_price = self.current_prices[symbol]
                    if position.side == PositionSide.LONG:
                        market_value = position.size * current_price
                    else:
                        # 空头持仓价值计算
                        market_value = position.size * (2 * position.entry_price - current_price)
                    
                    total_value += market_value
            
            return total_value
            
        except Exception as e:
            logger.error(f"计算投资组合价值失败: {e}")
            return self.cash_balance
    
    async def set_asset_allocation(self, allocations: Dict[str, float]) -> None:
        """设置资产配置"""
        try:
            # 验证权重总和
            total_weight = sum(allocations.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"资产配置权重总和不为1: {total_weight}")
            
            # 更新资产配置
            self.asset_allocations.clear()
            for symbol, weight in allocations.items():
                self.asset_allocations[symbol] = AssetAllocation(
                    symbol=symbol,
                    target_weight=weight
                )
            
            logger.info(f"资产配置已更新: {allocations}")
            
            # 立即更新当前权重
            await self._update_allocation_weights()
            
        except Exception as e:
            logger.error(f"设置资产配置失败: {e}")
    
    async def _update_allocation_weights(self) -> None:
        """更新资产配置当前权重"""
        try:
            total_value = await self.calculate_total_value()
            
            if total_value <= 0:
                return
            
            for symbol, allocation in self.asset_allocations.items():
                if symbol in self.positions and symbol in self.current_prices:
                    position = self.positions[symbol]
                    current_price = self.current_prices[symbol]
                    
                    if position.side == PositionSide.LONG:
                        position_value = position.size * current_price
                    else:
                        position_value = position.size * (2 * position.entry_price - current_price)
                    
                    allocation.current_weight = position_value / total_value
                else:
                    allocation.current_weight = 0.0
                
                # 计算偏离度
                allocation.deviation = allocation.current_weight - allocation.target_weight
                
        except Exception as e:
            logger.error(f"更新资产配置权重失败: {e}")
    
    async def check_rebalance_needed(self) -> bool:
        """检查是否需要再平衡"""
        try:
            if self.rebalance_strategy == RebalanceStrategy.NONE:
                return False
            
            current_time = datetime.utcnow()
            
            # 定期再平衡
            if self.rebalance_strategy == RebalanceStrategy.PERIODIC:
                if (self.last_rebalance_time is None or 
                    current_time - self.last_rebalance_time >= self.rebalance_frequency):
                    return True
            
            # 阈值再平衡
            elif self.rebalance_strategy == RebalanceStrategy.THRESHOLD:
                for allocation in self.asset_allocations.values():
                    if abs(allocation.deviation) > allocation.rebalance_threshold:
                        logger.info(f"需要再平衡: {allocation.symbol} 偏离{allocation.deviation:.2%}")
                        return True
            
            # 波动率再平衡
            elif self.rebalance_strategy == RebalanceStrategy.VOLATILITY:
                portfolio_volatility = await self._calculate_portfolio_volatility()
                if portfolio_volatility > 0.30:  # 年化波动率超过30%
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查再平衡需求失败: {e}")
            return False
    
    async def generate_rebalance_orders(self) -> List[Dict[str, Any]]:
        """生成再平衡订单"""
        try:
            orders = []
            total_value = await self.calculate_total_value()
            
            if total_value <= 0:
                return orders
            
            for symbol, allocation in self.asset_allocations.items():
                if symbol not in self.current_prices:
                    continue
                
                target_value = total_value * allocation.target_weight
                current_price = self.current_prices[symbol]
                
                # 计算当前持仓价值
                if symbol in self.positions:
                    position = self.positions[symbol]
                    if position.side == PositionSide.LONG:
                        current_value = position.size * current_price
                    else:
                        current_value = position.size * (2 * position.entry_price - current_price)
                else:
                    current_value = 0.0
                
                # 计算需要调整的价值
                value_diff = target_value - current_value
                
                # 如果差异超过最小交易额度
                min_trade_amount = total_value * 0.001  # 0.1%最小调整
                if abs(value_diff) > min_trade_amount:
                    # 计算交易数量
                    quantity = abs(value_diff) / current_price
                    side = "buy" if value_diff > 0 else "sell"
                    
                    orders.append({
                        'symbol': symbol,
                        'side': side,
                        'type': 'market',
                        'quantity': quantity,
                        'reason': 'rebalance',
                        'target_weight': allocation.target_weight,
                        'current_weight': allocation.current_weight
                    })
            
            if orders:
                self.last_rebalance_time = datetime.utcnow()
                logger.info(f"生成再平衡订单: {len(orders)}个")
            
            return orders
            
        except Exception as e:
            logger.error(f"生成再平衡订单失败: {e}")
            return []
    
    # 风险管理
    async def _check_risk_limits(self) -> Dict[str, Any]:
        """检查风险限额"""
        try:
            violations = []
            total_value = await self.calculate_total_value()
            
            if total_value <= 0:
                return {'violations': violations}
            
            # 检查单一持仓限额
            for symbol, position in self.positions.items():
                if symbol in self.current_prices:
                    position_value = position.size * self.current_prices[symbol]
                    weight = position_value / total_value
                    
                    if weight > self.risk_limits.max_position_size:
                        violations.append({
                            'type': 'position_size',
                            'symbol': symbol,
                            'current': weight,
                            'limit': self.risk_limits.max_position_size
                        })
            
            # 检查总杠杆
            total_exposure = sum(abs(pos.size * self.current_prices.get(symbol, 0))
                               for symbol, pos in self.positions.items()
                               if symbol in self.current_prices)
            leverage = total_exposure / total_value if total_value > 0 else 0
            
            if leverage > self.risk_limits.max_total_leverage:
                violations.append({
                    'type': 'leverage',
                    'current': leverage,
                    'limit': self.risk_limits.max_total_leverage
                })
            
            # 检查日亏损
            daily_return = self.daily_pnl / self.initial_balance
            if daily_return < -self.risk_limits.max_daily_loss:
                violations.append({
                    'type': 'daily_loss',
                    'current': daily_return,
                    'limit': -self.risk_limits.max_daily_loss
                })
            
            # 检查最大回撤
            if self.current_drawdown > self.risk_limits.max_drawdown_limit:
                violations.append({
                    'type': 'drawdown',
                    'current': self.current_drawdown,
                    'limit': self.risk_limits.max_drawdown_limit
                })
            
            if violations:
                logger.warning(f"风险限额违规: {len(violations)}项")
                
            return {
                'violations': violations,
                'total_value': total_value,
                'leverage': leverage,
                'daily_return': daily_return,
                'drawdown': self.current_drawdown
            }
            
        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return {'violations': []}
    
    async def calculate_var(self, confidence_level: float = 0.95, 
                           lookback_days: int = 30) -> float:
        """计算风险价值(VaR)"""
        try:
            if len(self.return_history) < lookback_days:
                return 0.0
            
            # 获取最近的收益率数据
            recent_returns = list(self.return_history)[-lookback_days:]
            
            if not recent_returns:
                return 0.0
            
            # 计算VaR
            returns_array = np.array(recent_returns)
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(returns_array, var_percentile)
            
            # 转换为货币金额
            current_value = await self.calculate_total_value()
            var_amount = abs(var * current_value)
            
            return var_amount
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0
    
    # 性能分析
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """计算投资组合指标"""
        try:
            total_value = await self.calculate_total_value()
            
            # 计算PnL
            invested_value = total_value - self.cash_balance
            total_pnl = total_value - self.initial_balance
            total_return = total_pnl / self.initial_balance
            
            # 未实现和已实现盈亏
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            # 计算日收益率
            daily_return = 0.0
            if len(self.return_history) > 0:
                daily_return = self.return_history[-1]
            
            # 计算波动率
            volatility = 0.0
            if len(self.return_history) > 1:
                returns_array = np.array(list(self.return_history))
                volatility = np.std(returns_array) * np.sqrt(252)  # 年化波动率
            
            # 计算夏普比率
            sharpe_ratio = 0.0
            if volatility > 0 and len(self.return_history) > 0:
                avg_return = np.mean(list(self.return_history)) * 252  # 年化收益
                sharpe_ratio = avg_return / volatility
            
            # 胜率计算
            win_rate = 0.0
            if len(self.return_history) > 0:
                positive_returns = [r for r in self.return_history if r > 0]
                win_rate = len(positive_returns) / len(self.return_history)
            
            # 盈亏比
            profit_factor = 0.0
            if len(self.return_history) > 0:
                positive_returns = [r for r in self.return_history if r > 0]
                negative_returns = [abs(r) for r in self.return_history if r < 0]
                
                if negative_returns:
                    avg_win = np.mean(positive_returns) if positive_returns else 0
                    avg_loss = np.mean(negative_returns)
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 仓位使用率
            exposure_ratio = invested_value / total_value if total_value > 0 else 0
            
            return PortfolioMetrics(
                total_value=total_value,
                cash_balance=self.cash_balance,
                invested_value=invested_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                total_return=total_return,
                daily_return=daily_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max(self.drawdown_history) if self.drawdown_history else 0,
                win_rate=win_rate,
                profit_factor=profit_factor,
                exposure_ratio=exposure_ratio
            )
            
        except Exception as e:
            logger.error(f"计算投资组合指标失败: {e}")
            return PortfolioMetrics(
                total_value=self.initial_balance, cash_balance=self.cash_balance,
                invested_value=0, unrealized_pnl=0, realized_pnl=0, total_pnl=0,
                total_return=0, daily_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, exposure_ratio=0
            )
    
    async def _calculate_portfolio_volatility(self) -> float:
        """计算投资组合波动率"""
        try:
            if len(self.return_history) < 20:  # 需要至少20个数据点
                return 0.0
            
            returns_array = np.array(list(self.return_history))
            volatility = np.std(returns_array) * np.sqrt(252)  # 年化波动率
            
            return volatility
            
        except Exception as e:
            logger.error(f"计算组合波动率失败: {e}")
            return 0.0
    
    # 数据记录和更新
    async def _record_portfolio_value(self, total_value: float) -> None:
        """记录投资组合价值"""
        try:
            timestamp = datetime.utcnow()
            
            # 记录价值历史
            self.value_history.append({
                'timestamp': timestamp,
                'total_value': total_value,
                'cash_balance': self.cash_balance
            })
            
            # 计算收益率
            if len(self.value_history) > 1:
                prev_value = self.value_history[-2]['total_value']
                daily_return = (total_value - prev_value) / prev_value
                self.return_history.append(daily_return)
                self.daily_pnl = total_value - prev_value
            
            # 计算回撤
            if total_value > self.peak_value:
                self.peak_value = total_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_value - total_value) / self.peak_value
            
            self.drawdown_history.append(self.current_drawdown)
            
        except Exception as e:
            logger.error(f"记录投资组合价值失败: {e}")
    
    async def _update_historical_returns(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """更新历史收益率"""
        try:
            if market_data['data_type'] != 'kline':
                return
            
            kline_data = market_data['data']
            close_price = kline_data['close']
            
            # 计算收益率
            if symbol in self.historical_returns and len(self.historical_returns[symbol]) > 0:
                prev_price = list(self.historical_returns[symbol])[-1]
                if prev_price > 0:
                    return_rate = (close_price - prev_price) / prev_price
                    self.historical_returns[symbol].append(return_rate)
            else:
                self.historical_returns[symbol].append(close_price)
                
        except Exception as e:
            logger.error(f"更新历史收益率失败: {e}")
    
    # 查询方法
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        return {
            'cash_balance': self.cash_balance,
            'positions_count': len(self.positions),
            'asset_allocations': {
                symbol: {
                    'target_weight': alloc.target_weight,
                    'current_weight': alloc.current_weight,
                    'deviation': alloc.deviation
                }
                for symbol, alloc in self.asset_allocations.items()
            }
        }
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """获取持仓摘要"""
        positions_data = {}
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position.current_price)
            
            positions_data[symbol] = {
                'side': position.side.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'entry_time': position.entry_time,
                'last_update': position.last_update
            }
        
        return positions_data
    
    def get_total_value(self) -> float:
        """同步获取总价值(用于快速查询)"""
        try:
            if self.value_history:
                return self.value_history[-1]['total_value']
            return self.initial_balance
        except Exception:
            return self.initial_balance
    
    async def finalize_portfolio(self) -> None:
        """完成投资组合(回测结束时调用)"""
        try:
            # 最后一次更新
            total_value = await self.calculate_total_value()
            await self._record_portfolio_value(total_value)
            
            logger.info(f"投资组合最终价值: {total_value:.2f} {self.currency}")
            
        except Exception as e:
            logger.error(f"完成投资组合失败: {e}")