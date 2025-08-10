"""
流动性提供者模拟器

模拟流动性提供者行为：
- 深度流动性提供
- 动态价格调整
- 大额订单处理
- 流动性挖矿模拟
- 激励机制
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import deque, defaultdict

from ..backtest.trading_environment import Order, OrderType, OrderSide
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class LiquidityStrategy(Enum):
    """流动性策略类型"""
    DEEP_LIQUIDITY = "deep_liquidity"      # 深度流动性
    TIERED_PRICING = "tiered_pricing"      # 分层定价
    VOLUME_INCENTIVE = "volume_incentive"  # 成交量激励
    DYNAMIC_SPREAD = "dynamic_spread"      # 动态价差
    MARKET_NEUTRAL = "market_neutral"      # 市场中性

@dataclass
class LiquidityTier:
    """流动性层级"""
    tier_level: int                        # 层级
    min_order_size: float                  # 最小订单规模
    max_order_size: float                  # 最大订单规模
    spread_discount_bps: float             # 价差折扣(基点)
    volume_threshold: float                # 成交量阈值
    rebate_rate_bps: float                 # 返佣费率

@dataclass
class LiquidityProviderConfig:
    """流动性提供者配置"""
    symbol: str
    base_liquidity_amount: float = 10000.0    # 基础流动性金额
    max_liquidity_amount: float = 100000.0    # 最大流动性金额
    min_spread_bps: float = 1.0              # 最小价差
    base_spread_bps: float = 10.0            # 基础价差
    max_levels: int = 10                     # 最大报价档数
    tier_count: int = 3                      # 流动性层级数
    rebalance_threshold: float = 0.2         # 再平衡阈值
    risk_limit_ratio: float = 0.8           # 风险限制比例
    
@dataclass
class LiquidityMetrics:
    """流动性指标"""
    total_provided_liquidity: float = 0.0
    active_orders_count: int = 0
    total_volume_matched: float = 0.0
    average_spread_bps: float = 0.0
    liquidity_utilization: float = 0.0
    rebates_earned: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0

class LiquidityProvider:
    """流动性提供者主类"""
    
    def __init__(self, config: LiquidityProviderConfig,
                 strategy: LiquidityStrategy = LiquidityStrategy.DEEP_LIQUIDITY):
        
        self.config = config
        self.strategy = strategy
        self.metrics = LiquidityMetrics()
        
        # 流动性层级设置
        self.liquidity_tiers = self._setup_liquidity_tiers()
        
        # 市场状态
        self.current_price = 0.0
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.market_volatility = 0.01
        self.order_flow_imbalance = 0.0
        
        # 活跃订单管理
        self.active_orders: Dict[str, Order] = {}
        self.order_levels: Dict[int, List[str]] = defaultdict(list)  # 档位 -> 订单列表
        
        # 流动性状态
        self.total_committed_liquidity = 0.0
        self.available_liquidity = config.max_liquidity_amount
        self.position_exposure = 0.0
        
        # 性能跟踪
        self.volume_history = deque(maxlen=1000)
        self.spread_history = deque(maxlen=1000)
        self.utilization_history = deque(maxlen=100)
        
        # 激励和返佣
        self.current_tier = 0
        self.volume_milestone = 0.0
        self.rebate_accumulator = 0.0
        
        logger.info(f"流动性提供者初始化: {config.symbol}, 策略={strategy.value}")
    
    def _setup_liquidity_tiers(self) -> List[LiquidityTier]:
        """设置流动性层级"""
        tiers = []
        
        for i in range(self.config.tier_count):
            tier = LiquidityTier(
                tier_level=i + 1,
                min_order_size=1000 * (i + 1),       # 递增最小订单规模
                max_order_size=10000 * (i + 1),      # 递增最大订单规模
                spread_discount_bps=i * 0.5,         # 更高层级价差更优
                volume_threshold=50000 * (i + 1),    # 递增成交量阈值
                rebate_rate_bps=0.5 + i * 0.2       # 递增返佣率
            )
            tiers.append(tier)
        
        return tiers
    
    async def initialize(self, initial_price: float) -> None:
        """初始化流动性提供者"""
        try:
            self.current_price = initial_price
            self.best_bid = initial_price * 0.999
            self.best_ask = initial_price * 1.001
            
            # 初始化流动性层级
            await self._update_liquidity_tier()
            
            logger.info(f"流动性提供者初始化完成: {self.config.symbol} @ {initial_price}")
            
        except Exception as e:
            logger.error(f"流动性提供者初始化失败: {e}")
            raise
    
    async def update_market_state(self, market_data: Dict[str, Any]) -> None:
        """更新市场状态"""
        try:
            # 更新价格信息
            if 'price' in market_data:
                self.current_price = market_data['price']
            
            if 'best_bid' in market_data and 'best_ask' in market_data:
                self.best_bid = market_data['best_bid']
                self.best_ask = market_data['best_ask']
                
                # 计算当前价差
                if self.best_bid > 0 and self.best_ask > 0:
                    spread = self.best_ask - self.best_bid
                    mid_price = (self.best_bid + self.best_ask) / 2
                    spread_bps = (spread / mid_price) * 10000
                    self.spread_history.append(spread_bps)
            
            # 更新订单流不平衡度
            if 'buy_volume' in market_data and 'sell_volume' in market_data:
                total_volume = market_data['buy_volume'] + market_data['sell_volume']
                if total_volume > 0:
                    self.order_flow_imbalance = (market_data['buy_volume'] - market_data['sell_volume']) / total_volume
            
            # 更新波动率
            if 'volatility' in market_data:
                self.market_volatility = market_data['volatility']
            
            # 更新流动性层级
            await self._update_liquidity_tier()
            
        except Exception as e:
            logger.error(f"更新市场状态失败: {e}")
    
    async def _update_liquidity_tier(self) -> None:
        """更新当前流动性层级"""
        try:
            total_volume = sum(self.volume_history) if self.volume_history else 0
            
            # 根据历史成交量确定当前层级
            new_tier = 0
            for i, tier in enumerate(self.liquidity_tiers):
                if total_volume >= tier.volume_threshold:
                    new_tier = i
            
            if new_tier != self.current_tier:
                logger.info(f"流动性层级升级: {self.current_tier} -> {new_tier}")
                self.current_tier = new_tier
            
        except Exception as e:
            logger.error(f"更新流动性层级失败: {e}")
    
    async def provide_liquidity(self) -> List[Order]:
        """提供流动性"""
        try:
            if self.current_price <= 0:
                return []
            
            # 根据策略生成流动性订单
            if self.strategy == LiquidityStrategy.DEEP_LIQUIDITY:
                orders = await self._provide_deep_liquidity()
            elif self.strategy == LiquidityStrategy.TIERED_PRICING:
                orders = await self._provide_tiered_pricing()
            elif self.strategy == LiquidityStrategy.VOLUME_INCENTIVE:
                orders = await self._provide_volume_incentive()
            elif self.strategy == LiquidityStrategy.DYNAMIC_SPREAD:
                orders = await self._provide_dynamic_spread()
            elif self.strategy == LiquidityStrategy.MARKET_NEUTRAL:
                orders = await self._provide_market_neutral()
            else:
                orders = await self._provide_deep_liquidity()
            
            # 更新活跃订单
            for order in orders:
                self.active_orders[order.order_id] = order
            
            # 更新流动性指标
            await self._update_liquidity_metrics()
            
            return orders
            
        except Exception as e:
            logger.error(f"提供流动性失败: {e}")
            return []
    
    async def _provide_deep_liquidity(self) -> List[Order]:
        """深度流动性策略"""
        try:
            orders = []
            current_tier = self.liquidity_tiers[min(self.current_tier, len(self.liquidity_tiers) - 1)]
            
            # 计算基础价差
            base_spread = self.current_price * (self.config.base_spread_bps / 10000)
            tier_spread = base_spread * (1 - current_tier.spread_discount_bps / 10000)
            
            # 可用流动性分配
            liquidity_per_level = self.available_liquidity / self.config.max_levels
            
            for level in range(self.config.max_levels):
                # 价格计算
                level_spread_multiplier = 1 + level * 0.1  # 每档增加10%
                level_spread = tier_spread * level_spread_multiplier
                
                bid_price = self.current_price - level_spread / 2
                ask_price = self.current_price + level_spread / 2
                
                # 数量计算 - 深度流动性意味着大订单
                base_quantity = liquidity_per_level / self.current_price
                level_quantity = base_quantity * (0.9 ** level)  # 递减但保持较大规模
                
                # 确保订单满足最小规模要求
                if level_quantity * self.current_price >= current_tier.min_order_size:
                    current_time = datetime.utcnow()
                    
                    # 买单
                    bid_order = Order(
                        order_id=f"lp_deep_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=level_quantity,
                        price=bid_price,
                        strategy_id="liquidity_provider_deep"
                    )
                    
                    # 卖单
                    ask_order = Order(
                        order_id=f"lp_deep_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=level_quantity,
                        price=ask_price,
                        strategy_id="liquidity_provider_deep"
                    )
                    
                    orders.extend([bid_order, ask_order])
                    self.order_levels[level].extend([bid_order.order_id, ask_order.order_id])
            
            return orders
            
        except Exception as e:
            logger.error(f"提供深度流动性失败: {e}")
            return []
    
    async def _provide_tiered_pricing(self) -> List[Order]:
        """分层定价策略"""
        try:
            orders = []
            
            for tier in self.liquidity_tiers:
                # 每层级提供不同价格的流动性
                tier_spread_bps = self.config.base_spread_bps - tier.spread_discount_bps
                tier_spread = self.current_price * (tier_spread_bps / 10000)
                
                # 分层数量
                tier_quantity = min(tier.max_order_size, self.available_liquidity * 0.3) / self.current_price
                
                if tier_quantity > 0:
                    # 计算分层价格
                    bid_price = self.current_price - tier_spread / 2
                    ask_price = self.current_price + tier_spread / 2
                    
                    current_time = datetime.utcnow()
                    
                    bid_order = Order(
                        order_id=f"lp_tier_bid_{self.config.symbol}_{tier.tier_level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=tier_quantity,
                        price=bid_price,
                        strategy_id=f"liquidity_provider_tier_{tier.tier_level}"
                    )
                    
                    ask_order = Order(
                        order_id=f"lp_tier_ask_{self.config.symbol}_{tier.tier_level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=tier_quantity,
                        price=ask_price,
                        strategy_id=f"liquidity_provider_tier_{tier.tier_level}"
                    )
                    
                    orders.extend([bid_order, ask_order])
            
            return orders
            
        except Exception as e:
            logger.error(f"提供分层定价流动性失败: {e}")
            return []
    
    async def _provide_volume_incentive(self) -> List[Order]:
        """成交量激励策略"""
        try:
            orders = []
            
            # 根据历史成交量调整激励
            recent_volume = sum(list(self.volume_history)[-10:]) if len(self.volume_history) >= 10 else 0
            volume_factor = min(2.0, 1 + recent_volume / 100000)  # 成交量越大，激励越多
            
            # 激励价差 - 成交量高时价差更紧
            incentive_spread_bps = self.config.base_spread_bps / volume_factor
            incentive_spread = self.current_price * (incentive_spread_bps / 10000)
            
            # 激励数量 - 成交量高时提供更多流动性
            incentive_liquidity = self.config.base_liquidity_amount * volume_factor
            
            levels_to_create = min(self.config.max_levels, int(volume_factor * 5))
            
            for level in range(levels_to_create):
                level_spread = incentive_spread * (1 + level * 0.15)
                bid_price = self.current_price - level_spread / 2
                ask_price = self.current_price + level_spread / 2
                
                level_quantity = (incentive_liquidity / levels_to_create) / self.current_price
                level_quantity *= (0.85 ** level)
                
                current_time = datetime.utcnow()
                
                bid_order = Order(
                    order_id=f"lp_vol_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=bid_price,
                    strategy_id="liquidity_provider_volume"
                )
                
                ask_order = Order(
                    order_id=f"lp_vol_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=ask_price,
                    strategy_id="liquidity_provider_volume"
                )
                
                orders.extend([bid_order, ask_order])
            
            return orders
            
        except Exception as e:
            logger.error(f"提供成交量激励流动性失败: {e}")
            return []
    
    async def _provide_dynamic_spread(self) -> List[Order]:
        """动态价差策略"""
        try:
            orders = []
            
            # 根据市场波动率动态调整价差
            volatility_multiplier = 1 + self.market_volatility * 20
            dynamic_spread_bps = self.config.base_spread_bps * volatility_multiplier
            
            # 根据订单流不平衡调整
            imbalance_adjustment = abs(self.order_flow_imbalance) * 0.5
            adjusted_spread_bps = dynamic_spread_bps * (1 + imbalance_adjustment)
            
            # 限制价差范围
            adjusted_spread_bps = max(self.config.min_spread_bps, adjusted_spread_bps)
            adjusted_spread = self.current_price * (adjusted_spread_bps / 10000)
            
            # 订单流不平衡时调整价格偏移
            price_skew = self.order_flow_imbalance * adjusted_spread * 0.3
            
            for level in range(self.config.max_levels):
                level_multiplier = 1 + level * 0.2
                level_spread = adjusted_spread * level_multiplier
                
                # 应用价格偏移
                bid_price = self.current_price - level_spread / 2 + price_skew
                ask_price = self.current_price + level_spread / 2 + price_skew
                
                # 动态数量调整
                base_quantity = self.config.base_liquidity_amount / self.config.max_levels / self.current_price
                level_quantity = base_quantity * (0.8 ** level) / volatility_multiplier
                
                current_time = datetime.utcnow()
                
                if level_quantity > 0:
                    bid_order = Order(
                        order_id=f"lp_dyn_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=level_quantity,
                        price=max(0.01, bid_price),
                        strategy_id="liquidity_provider_dynamic"
                    )
                    
                    ask_order = Order(
                        order_id=f"lp_dyn_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=level_quantity,
                        price=ask_price,
                        strategy_id="liquidity_provider_dynamic"
                    )
                    
                    orders.extend([bid_order, ask_order])
            
            return orders
            
        except Exception as e:
            logger.error(f"提供动态价差流动性失败: {e}")
            return []
    
    async def _provide_market_neutral(self) -> List[Order]:
        """市场中性策略"""
        try:
            orders = []
            
            # 市场中性意味着买卖订单完全对称
            neutral_spread = self.current_price * (self.config.base_spread_bps / 10000)
            
            # 总流动性平分到各档位
            liquidity_per_side = self.available_liquidity / 2  # 买卖各一半
            liquidity_per_level = liquidity_per_side / self.config.max_levels
            
            for level in range(self.config.max_levels):
                level_multiplier = 1 + level * 0.12
                level_spread = neutral_spread * level_multiplier
                
                # 严格对称定价
                bid_price = self.current_price - level_spread / 2
                ask_price = self.current_price + level_spread / 2
                
                # 完全相同的数量
                level_quantity = liquidity_per_level / self.current_price
                level_quantity *= (0.9 ** level)  # 每档略微递减
                
                current_time = datetime.utcnow()
                
                bid_order = Order(
                    order_id=f"lp_neutral_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=bid_price,
                    strategy_id="liquidity_provider_neutral"
                )
                
                ask_order = Order(
                    order_id=f"lp_neutral_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=ask_price,
                    strategy_id="liquidity_provider_neutral"
                )
                
                orders.extend([bid_order, ask_order])
            
            return orders
            
        except Exception as e:
            logger.error(f"提供市场中性流动性失败: {e}")
            return []
    
    def on_order_filled(self, order_id: str, filled_quantity: float, fill_price: float) -> None:
        """处理订单成交"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                
                # 记录成交量
                trade_volume = filled_quantity * fill_price
                self.volume_history.append(trade_volume)
                
                # 更新流动性指标
                self.metrics.total_volume_matched += trade_volume
                
                # 计算返佣
                current_tier = self.liquidity_tiers[min(self.current_tier, len(self.liquidity_tiers) - 1)]
                rebate = trade_volume * (current_tier.rebate_rate_bps / 10000)
                self.metrics.rebates_earned += rebate
                self.rebate_accumulator += rebate
                
                # 更新仓位敞口
                if order.side == OrderSide.BUY:
                    self.position_exposure += filled_quantity
                else:
                    self.position_exposure -= filled_quantity
                
                # 移除完全成交的订单
                if filled_quantity >= order.quantity:
                    del self.active_orders[order_id]
                    # 从档位记录中移除
                    for level_orders in self.order_levels.values():
                        if order_id in level_orders:
                            level_orders.remove(order_id)
                else:
                    # 更新部分成交订单
                    order.filled_quantity += filled_quantity
                
                logger.debug(f"流动性订单成交: {order_id}, 数量={filled_quantity}, 价格={fill_price}")
                
        except Exception as e:
            logger.error(f"处理订单成交失败: {e}")
    
    async def _update_liquidity_metrics(self) -> None:
        """更新流动性指标"""
        try:
            # 统计活跃订单
            self.metrics.active_orders_count = len(self.active_orders)
            
            # 计算已提供的总流动性
            total_liquidity = 0.0
            for order in self.active_orders.values():
                remaining_quantity = order.quantity - order.filled_quantity
                total_liquidity += remaining_quantity * (order.price or self.current_price)
            
            self.metrics.total_provided_liquidity = total_liquidity
            
            # 计算流动性利用率
            if self.config.max_liquidity_amount > 0:
                self.metrics.liquidity_utilization = total_liquidity / self.config.max_liquidity_amount
                self.utilization_history.append(self.metrics.liquidity_utilization)
            
            # 计算平均价差
            if self.spread_history:
                self.metrics.average_spread_bps = np.mean(list(self.spread_history))
            
            # 更新可用流动性
            self.available_liquidity = max(0, self.config.max_liquidity_amount - total_liquidity)
            
            # 计算净PnL (简化计算)
            self.metrics.net_pnl = self.metrics.rebates_earned - self.metrics.fees_paid
            
        except Exception as e:
            logger.error(f"更新流动性指标失败: {e}")
    
    def get_liquidity_status(self) -> Dict[str, Any]:
        """获取流动性状态"""
        try:
            return {
                'symbol': self.config.symbol,
                'strategy': self.strategy.value,
                'current_tier': self.current_tier + 1,  # 显示从1开始
                'tier_info': {
                    'level': self.liquidity_tiers[self.current_tier].tier_level,
                    'volume_threshold': self.liquidity_tiers[self.current_tier].volume_threshold,
                    'rebate_rate_bps': self.liquidity_tiers[self.current_tier].rebate_rate_bps
                },
                'liquidity_metrics': {
                    'total_provided': self.metrics.total_provided_liquidity,
                    'available': self.available_liquidity,
                    'utilization': self.metrics.liquidity_utilization,
                    'active_orders': self.metrics.active_orders_count,
                    'total_volume_matched': self.metrics.total_volume_matched,
                    'average_spread_bps': self.metrics.average_spread_bps,
                    'rebates_earned': self.metrics.rebates_earned,
                    'net_pnl': self.metrics.net_pnl
                },
                'position_exposure': self.position_exposure,
                'market_state': {
                    'current_price': self.current_price,
                    'volatility': self.market_volatility,
                    'order_flow_imbalance': self.order_flow_imbalance
                }
            }
            
        except Exception as e:
            logger.error(f"获取流动性状态失败: {e}")
            return {}
    
    async def rebalance_liquidity(self) -> List[str]:
        """重新平衡流动性"""
        try:
            cancelled_orders = []
            
            # 检查是否需要重新平衡
            if abs(self.position_exposure) > self.config.max_liquidity_amount * self.config.rebalance_threshold:
                logger.info(f"触发流动性重新平衡: 敞口={self.position_exposure}")
                
                # 取消所有活跃订单
                for order_id in list(self.active_orders.keys()):
                    cancelled_orders.append(order_id)
                    del self.active_orders[order_id]
                
                # 清空档位记录
                self.order_levels.clear()
                
                # 重置敞口(在实际实现中应该通过对冲来平衡)
                self.position_exposure = 0.0
                
                logger.info(f"流动性重新平衡完成: 取消{len(cancelled_orders)}个订单")
            
            return cancelled_orders
            
        except Exception as e:
            logger.error(f"重新平衡流动性失败: {e}")
            return []
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self.metrics = LiquidityMetrics()
        self.volume_history.clear()
        self.spread_history.clear()
        self.utilization_history.clear()
        self.rebate_accumulator = 0.0
        logger.info(f"流动性提供者指标已重置: {self.config.symbol}")