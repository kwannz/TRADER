"""
滑点模型

提供各种滑点计算模型：
- 线性滑点模型
- 平方根滑点模型
- 市场影响滑点模型
- 时间相关滑点模型
- 流动性滑点模型
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

from ..backtest.trading_environment import Order, OrderType, OrderSide
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class SlippageType(Enum):
    """滑点类型"""
    LINEAR = "linear"              # 线性滑点
    SQUARE_ROOT = "square_root"    # 平方根滑点  
    MARKET_IMPACT = "market_impact" # 市场影响滑点
    TIME_DEPENDENT = "time_dependent" # 时间相关滑点
    LIQUIDITY_BASED = "liquidity_based" # 流动性滑点

@dataclass
class SlippageParameters:
    """滑点参数"""
    base_slippage_bps: float = 2.0    # 基础滑点(基点)
    size_impact_factor: float = 0.1   # 规模影响因子
    volatility_factor: float = 1.0    # 波动率因子
    liquidity_factor: float = 1.0     # 流动性因子
    time_penalty: float = 0.0         # 时间惩罚因子
    max_slippage_pct: float = 5.0     # 最大滑点百分比

@dataclass  
class SlippageResult:
    """滑点计算结果"""
    original_price: float
    adjusted_price: float
    slippage_amount: float
    slippage_bps: float
    slippage_pct: float
    breakdown: Dict[str, float]

class SlippageModel:
    """滑点模型主类"""
    
    def __init__(self, slippage_type: SlippageType = SlippageType.LINEAR,
                 parameters: Optional[SlippageParameters] = None):
        
        self.slippage_type = slippage_type
        self.parameters = parameters or SlippageParameters()
        
        # 市场状态缓存
        self.market_volatility: Dict[str, deque] = {}
        self.order_flow: Dict[str, deque] = {}
        self.liquidity_metrics: Dict[str, float] = {}
        
        # 滑点历史
        self.slippage_history: Dict[str, deque] = {}
        
        # 时间相关状态
        self.last_order_times: Dict[str, datetime] = {}
        
        logger.info(f"滑点模型初始化: {slippage_type.value}")
    
    async def calculate_slippage(self, order: Order, current_price: float,
                               market_data: Optional[Dict[str, Any]] = None) -> SlippageResult:
        """计算订单滑点"""
        try:
            symbol = order.symbol
            quantity = order.quantity
            order_value = quantity * current_price
            
            # 根据滑点类型选择计算方法
            if self.slippage_type == SlippageType.LINEAR:
                slippage = await self._calculate_linear_slippage(
                    symbol, order_value, current_price, market_data
                )
            elif self.slippage_type == SlippageType.SQUARE_ROOT:
                slippage = await self._calculate_sqrt_slippage(
                    symbol, order_value, current_price, market_data
                )
            elif self.slippage_type == SlippageType.MARKET_IMPACT:
                slippage = await self._calculate_market_impact_slippage(
                    symbol, quantity, current_price, order, market_data
                )
            elif self.slippage_type == SlippageType.TIME_DEPENDENT:
                slippage = await self._calculate_time_dependent_slippage(
                    symbol, order_value, current_price, order, market_data
                )
            elif self.slippage_type == SlippageType.LIQUIDITY_BASED:
                slippage = await self._calculate_liquidity_slippage(
                    symbol, order_value, current_price, market_data
                )
            else:
                slippage = await self._calculate_linear_slippage(
                    symbol, order_value, current_price, market_data
                )
            
            # 应用方向性滑点
            if order.side == OrderSide.BUY:
                adjusted_price = current_price + slippage['total_slippage']
            else:
                adjusted_price = current_price - slippage['total_slippage']
            
            # 限制最大滑点
            max_slippage = current_price * (self.parameters.max_slippage_pct / 100)
            if abs(adjusted_price - current_price) > max_slippage:
                if order.side == OrderSide.BUY:
                    adjusted_price = current_price + max_slippage
                else:
                    adjusted_price = current_price - max_slippage
            
            # 计算最终滑点
            final_slippage = abs(adjusted_price - current_price)
            slippage_bps = (final_slippage / current_price) * 10000
            slippage_pct = (final_slippage / current_price) * 100
            
            result = SlippageResult(
                original_price=current_price,
                adjusted_price=adjusted_price,
                slippage_amount=final_slippage,
                slippage_bps=slippage_bps,
                slippage_pct=slippage_pct,
                breakdown=slippage
            )
            
            # 记录滑点历史
            await self._record_slippage(symbol, result)
            
            return result
            
        except Exception as e:
            logger.error(f"计算滑点失败: {e}")
            # 返回基础滑点
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return SlippageResult(
                original_price=current_price,
                adjusted_price=current_price + base_slippage,
                slippage_amount=base_slippage,
                slippage_bps=self.parameters.base_slippage_bps,
                slippage_pct=self.parameters.base_slippage_bps / 100,
                breakdown={'base': base_slippage}
            )
    
    async def _calculate_linear_slippage(self, symbol: str, order_value: float,
                                       current_price: float, 
                                       market_data: Optional[Dict]) -> Dict[str, float]:
        """线性滑点模型"""
        try:
            # 基础滑点
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            
            # 规模影响 - 与订单金额线性相关
            size_factor = order_value / 10000  # 每10k增加影响
            size_slippage = size_factor * self.parameters.size_impact_factor * base_slippage
            
            # 波动率影响
            volatility_slippage = 0.0
            if market_data and 'volatility' in market_data:
                volatility = market_data['volatility']
                volatility_slippage = volatility * self.parameters.volatility_factor * base_slippage
            
            total_slippage = base_slippage + size_slippage + volatility_slippage
            
            return {
                'base': base_slippage,
                'size_impact': size_slippage,
                'volatility_impact': volatility_slippage,
                'total_slippage': total_slippage
            }
            
        except Exception as e:
            logger.error(f"计算线性滑点失败: {e}")
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return {'base': base_slippage, 'total_slippage': base_slippage}
    
    async def _calculate_sqrt_slippage(self, symbol: str, order_value: float,
                                     current_price: float,
                                     market_data: Optional[Dict]) -> Dict[str, float]:
        """平方根滑点模型 - 适用于流动性较好的市场"""
        try:
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            
            # 规模影响使用平方根函数
            size_factor = np.sqrt(order_value / 1000)  # 每1k基准
            size_slippage = size_factor * self.parameters.size_impact_factor * base_slippage
            
            # 波动率影响
            volatility_slippage = 0.0
            if market_data and 'volatility' in market_data:
                volatility = market_data['volatility']
                volatility_slippage = np.sqrt(volatility) * self.parameters.volatility_factor * base_slippage
            
            total_slippage = base_slippage + size_slippage + volatility_slippage
            
            return {
                'base': base_slippage,
                'size_impact_sqrt': size_slippage,
                'volatility_impact_sqrt': volatility_slippage,
                'total_slippage': total_slippage
            }
            
        except Exception as e:
            logger.error(f"计算平方根滑点失败: {e}")
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return {'base': base_slippage, 'total_slippage': base_slippage}
    
    async def _calculate_market_impact_slippage(self, symbol: str, quantity: float,
                                              current_price: float, order: Order,
                                              market_data: Optional[Dict]) -> Dict[str, float]:
        """市场影响滑点模型"""
        try:
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            
            # 计算市场影响
            # 永久影响 - 订单对价格的持久影响
            permanent_impact = 0.0
            if market_data and 'daily_volume' in market_data:
                daily_volume = market_data['daily_volume']
                participation_rate = (quantity * current_price) / daily_volume
                permanent_impact = participation_rate * 0.5 * current_price  # 简化模型
            
            # 临时影响 - 订单执行时的临时价格冲击
            temporary_impact = 0.0
            if market_data and 'bid_ask_spread' in market_data:
                spread = market_data['bid_ask_spread']
                # 市价订单承担一半价差
                if order.order_type == OrderType.MARKET:
                    temporary_impact = spread / 2
                else:
                    temporary_impact = spread / 4  # 限价订单影响较小
            
            # 时机影响 - 基于订单紧急程度
            timing_impact = 0.0
            if order.order_type == OrderType.MARKET:
                timing_impact = base_slippage * 0.5  # 市价订单时机成本
            
            total_slippage = base_slippage + permanent_impact + temporary_impact + timing_impact
            
            return {
                'base': base_slippage,
                'permanent_impact': permanent_impact,
                'temporary_impact': temporary_impact,
                'timing_impact': timing_impact,
                'total_slippage': total_slippage
            }
            
        except Exception as e:
            logger.error(f"计算市场影响滑点失败: {e}")
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return {'base': base_slippage, 'total_slippage': base_slippage}
    
    async def _calculate_time_dependent_slippage(self, symbol: str, order_value: float,
                                               current_price: float, order: Order,
                                               market_data: Optional[Dict]) -> Dict[str, float]:
        """时间相关滑点模型"""
        try:
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            
            # 时间惩罚 - 连续订单的时间间隔影响
            time_penalty = 0.0
            current_time = order.created_time or datetime.utcnow()
            
            if symbol in self.last_order_times:
                time_diff = (current_time - self.last_order_times[symbol]).total_seconds()
                
                # 短时间内连续订单增加滑点
                if time_diff < 60:  # 1分钟内
                    time_penalty = base_slippage * self.parameters.time_penalty * (60 - time_diff) / 60
            
            # 更新最后订单时间
            self.last_order_times[symbol] = current_time
            
            # 市场时段影响
            session_impact = 0.0
            hour = current_time.hour
            
            # 开盘和收盘时段波动较大
            if hour in [0, 1, 8, 9, 16, 17]:  # UTC时间
                session_impact = base_slippage * 0.5
            elif hour in [2, 3, 4, 5]:  # 流动性较低时段
                session_impact = base_slippage * 0.3
            
            total_slippage = base_slippage + time_penalty + session_impact
            
            return {
                'base': base_slippage,
                'time_penalty': time_penalty,
                'session_impact': session_impact,
                'total_slippage': total_slippage
            }
            
        except Exception as e:
            logger.error(f"计算时间相关滑点失败: {e}")
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return {'base': base_slippage, 'total_slippage': base_slippage}
    
    async def _calculate_liquidity_slippage(self, symbol: str, order_value: float,
                                          current_price: float,
                                          market_data: Optional[Dict]) -> Dict[str, float]:
        """流动性滑点模型"""
        try:
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            
            # 订单簿深度影响
            depth_impact = 0.0
            if market_data and 'order_book_depth' in market_data:
                depth = market_data['order_book_depth']
                # 订单价值相对于深度的比例
                depth_ratio = order_value / max(depth, 1000)  # 避免除零
                depth_impact = depth_ratio * base_slippage * 2
            
            # 流动性指标影响
            liquidity_impact = 0.0
            liquidity_score = self.liquidity_metrics.get(symbol, 1.0)
            
            # 流动性越低，滑点越大
            if liquidity_score < 1.0:
                liquidity_impact = base_slippage * (1 - liquidity_score) * self.parameters.liquidity_factor
            
            # 最近交易量影响
            volume_impact = 0.0
            if market_data and 'recent_volume' in market_data:
                recent_volume = market_data['recent_volume']
                avg_volume = market_data.get('average_volume', recent_volume)
                
                # 成交量异常低时增加滑点
                if recent_volume < avg_volume * 0.5:
                    volume_ratio = recent_volume / avg_volume
                    volume_impact = base_slippage * (1 - volume_ratio)
            
            total_slippage = base_slippage + depth_impact + liquidity_impact + volume_impact
            
            return {
                'base': base_slippage,
                'depth_impact': depth_impact,
                'liquidity_impact': liquidity_impact,
                'volume_impact': volume_impact,
                'total_slippage': total_slippage
            }
            
        except Exception as e:
            logger.error(f"计算流动性滑点失败: {e}")
            base_slippage = current_price * (self.parameters.base_slippage_bps / 10000)
            return {'base': base_slippage, 'total_slippage': base_slippage}
    
    async def _record_slippage(self, symbol: str, result: SlippageResult) -> None:
        """记录滑点历史"""
        try:
            if symbol not in self.slippage_history:
                self.slippage_history[symbol] = deque(maxlen=1000)
            
            record = {
                'timestamp': datetime.utcnow(),
                'slippage_bps': result.slippage_bps,
                'slippage_pct': result.slippage_pct,
                'price': result.original_price,
                'breakdown': result.breakdown
            }
            
            self.slippage_history[symbol].append(record)
            
        except Exception as e:
            logger.error(f"记录滑点历史失败: {e}")
    
    def update_market_state(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """更新市场状态"""
        try:
            # 更新波动率
            if symbol not in self.market_volatility:
                self.market_volatility[symbol] = deque(maxlen=100)
            
            if 'price' in market_data:
                price_changes = []
                if len(self.market_volatility[symbol]) > 0:
                    prev_data = self.market_volatility[symbol][-1]
                    if 'price' in prev_data:
                        price_change = (market_data['price'] - prev_data['price']) / prev_data['price']
                        price_changes.append(abs(price_change))
                
                market_data['volatility'] = np.std(price_changes) if len(price_changes) > 10 else 0.01
                
            self.market_volatility[symbol].append(market_data)
            
            # 更新流动性指标
            if 'volume' in market_data and 'bid_ask_spread' in market_data:
                # 简单的流动性评分：成交量高、价差小 = 流动性好
                volume = market_data['volume']
                spread = market_data['bid_ask_spread']
                price = market_data.get('price', 1.0)
                
                # 标准化流动性评分
                volume_score = min(1.0, volume / 1000000)  # 1M作为基准
                spread_score = max(0.1, 1.0 - (spread / price) * 1000)  # 价差越小分数越高
                
                self.liquidity_metrics[symbol] = (volume_score + spread_score) / 2
            
        except Exception as e:
            logger.error(f"更新市场状态失败: {e}")
    
    def get_slippage_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取滑点统计"""
        try:
            if symbol:
                symbols = [symbol] if symbol in self.slippage_history else []
            else:
                symbols = list(self.slippage_history.keys())
            
            stats = {}
            
            for sym in symbols:
                history = list(self.slippage_history[sym])
                if not history:
                    continue
                
                slippages = [record['slippage_bps'] for record in history]
                
                stats[sym] = {
                    'count': len(slippages),
                    'avg_slippage_bps': np.mean(slippages),
                    'median_slippage_bps': np.median(slippages),
                    'std_slippage_bps': np.std(slippages),
                    'max_slippage_bps': np.max(slippages),
                    'min_slippage_bps': np.min(slippages),
                    'percentile_95_bps': np.percentile(slippages, 95),
                    'percentile_99_bps': np.percentile(slippages, 99)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取滑点统计失败: {e}")
            return {}
    
    def set_parameters(self, parameters: SlippageParameters) -> None:
        """设置滑点参数"""
        self.parameters = parameters
        logger.info(f"滑点参数已更新: 基础滑点={parameters.base_slippage_bps}bps")
    
    def get_current_parameters(self) -> SlippageParameters:
        """获取当前滑点参数"""
        return self.parameters