"""
做市商模拟器

模拟做市商行为和市场微观结构：
- 双边报价策略
- 库存管理
- 价格发现
- 流动性提供
- 风险控制
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
from collections import deque, defaultdict

from ..backtest.trading_environment import Order, OrderType, OrderSide, OrderStatus
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class MarketMakerStrategy(Enum):
    """做市策略类型"""
    SIMPLE_SPREAD = "simple_spread"        # 简单价差策略
    INVENTORY_BASED = "inventory_based"    # 库存驱动策略  
    VOLUME_WEIGHTED = "volume_weighted"    # 成交量加权策略
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 波动率调整策略
    ADAPTIVE = "adaptive"                  # 自适应策略

@dataclass
class MarketMakerConfig:
    """做市商配置"""
    symbol: str
    base_spread_bps: float = 20.0         # 基础价差(基点)
    max_inventory: float = 1000.0         # 最大库存
    target_inventory: float = 0.0         # 目标库存
    quote_size: float = 100.0             # 报价数量
    max_quote_levels: int = 5             # 最大报价档数
    inventory_penalty_rate: float = 0.1   # 库存惩罚率
    risk_limit: float = 5000.0           # 风险限额
    min_spread_bps: float = 5.0          # 最小价差
    max_spread_bps: float = 200.0        # 最大价差

@dataclass
class QuoteLevel:
    """报价档位"""
    level: int
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None

@dataclass
class MarketMakerState:
    """做市商状态"""
    symbol: str
    current_inventory: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    bid_trades: int = 0
    ask_trades: int = 0
    active_quotes: Dict[str, Order] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.utcnow)

class MarketMaker:
    """做市商模拟器主类"""
    
    def __init__(self, config: MarketMakerConfig,
                 strategy: MarketMakerStrategy = MarketMakerStrategy.SIMPLE_SPREAD):
        
        self.config = config
        self.strategy = strategy
        self.state = MarketMakerState(symbol=config.symbol)
        
        # 市场数据
        self.current_price = 0.0
        self.price_history = deque(maxlen=1000)
        self.volatility = 0.01
        
        # 订单簿状态
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_depth = 0.0
        self.ask_depth = 0.0
        
        # 做市策略参数
        self.quote_levels: List[QuoteLevel] = []
        self.last_quote_time = datetime.utcnow()
        self.quote_frequency = timedelta(seconds=1)  # 报价频率
        
        # 风险控制
        self.position_limit_reached = False
        self.risk_limit_reached = False
        
        # 性能统计
        self.quote_count = 0
        self.fill_count = 0
        self.quote_hit_ratio = 0.0
        
        # 市场指标
        self.mid_price_changes = deque(maxlen=100)
        self.spread_history = deque(maxlen=1000)
        
        logger.info(f"做市商初始化: {config.symbol}, 策略={strategy.value}")
    
    async def initialize(self, initial_price: float) -> None:
        """初始化做市商"""
        try:
            self.current_price = initial_price
            self.price_history.append(initial_price)
            
            # 设置初始最优价格
            initial_spread = initial_price * (self.config.base_spread_bps / 10000)
            self.best_bid = initial_price - initial_spread / 2
            self.best_ask = initial_price + initial_spread / 2
            
            logger.info(f"做市商初始化完成: {self.config.symbol} @ {initial_price}")
            
        except Exception as e:
            logger.error(f"做市商初始化失败: {e}")
            raise
    
    async def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """更新市场数据"""
        try:
            # 更新价格
            if 'price' in market_data:
                old_price = self.current_price
                self.current_price = market_data['price']
                self.price_history.append(self.current_price)
                
                # 记录价格变化
                if old_price > 0:
                    price_change = (self.current_price - old_price) / old_price
                    self.mid_price_changes.append(price_change)
            
            # 更新订单簿信息
            if 'best_bid' in market_data and 'best_ask' in market_data:
                self.best_bid = market_data['best_bid']
                self.best_ask = market_data['best_ask']
                
                # 记录价差历史
                if self.best_bid > 0 and self.best_ask > 0:
                    spread = self.best_ask - self.best_bid
                    spread_bps = (spread / ((self.best_bid + self.best_ask) / 2)) * 10000
                    self.spread_history.append(spread_bps)
            
            # 更新流动性深度
            if 'bid_depth' in market_data:
                self.bid_depth = market_data['bid_depth']
            if 'ask_depth' in market_data:
                self.ask_depth = market_data['ask_depth']
            
            # 更新波动率
            await self._update_volatility()
            
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
    
    async def _update_volatility(self) -> None:
        """更新波动率估计"""
        try:
            if len(self.mid_price_changes) > 20:
                returns = np.array(list(self.mid_price_changes))
                self.volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # 年化波动率
                
        except Exception as e:
            logger.error(f"更新波动率失败: {e}")
    
    async def generate_quotes(self) -> List[Order]:
        """生成做市报价"""
        try:
            current_time = datetime.utcnow()
            
            # 检查报价频率
            if current_time - self.last_quote_time < self.quote_frequency:
                return []
            
            # 检查风险限额
            if await self._check_risk_limits():
                return []
            
            # 根据策略生成报价
            if self.strategy == MarketMakerStrategy.SIMPLE_SPREAD:
                quotes = await self._generate_simple_spread_quotes()
            elif self.strategy == MarketMakerStrategy.INVENTORY_BASED:
                quotes = await self._generate_inventory_based_quotes()
            elif self.strategy == MarketMakerStrategy.VOLUME_WEIGHTED:
                quotes = await self._generate_volume_weighted_quotes()
            elif self.strategy == MarketMakerStrategy.VOLATILITY_ADJUSTED:
                quotes = await self._generate_volatility_adjusted_quotes()
            elif self.strategy == MarketMakerStrategy.ADAPTIVE:
                quotes = await self._generate_adaptive_quotes()
            else:
                quotes = await self._generate_simple_spread_quotes()
            
            self.last_quote_time = current_time
            self.quote_count += len(quotes)
            
            return quotes
            
        except Exception as e:
            logger.error(f"生成报价失败: {e}")
            return []
    
    async def _generate_simple_spread_quotes(self) -> List[Order]:
        """简单价差策略"""
        try:
            if self.current_price <= 0:
                return []
            
            quotes = []
            base_spread = self.current_price * (self.config.base_spread_bps / 10000)
            
            for level in range(self.config.max_quote_levels):
                # 计算价格
                spread_multiplier = 1 + level * 0.2  # 每档增加20%价差
                level_spread = base_spread * spread_multiplier
                
                bid_price = self.current_price - level_spread / 2
                ask_price = self.current_price + level_spread / 2
                
                # 数量随档位递减
                level_quantity = self.config.quote_size * (0.8 ** level)
                
                # 创建买单
                bid_order = Order(
                    order_id=f"mm_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=bid_price,
                    strategy_id="market_maker"
                )
                
                # 创建卖单
                ask_order = Order(
                    order_id=f"mm_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=ask_price,
                    strategy_id="market_maker"
                )
                
                quotes.extend([bid_order, ask_order])
            
            return quotes
            
        except Exception as e:
            logger.error(f"生成简单价差报价失败: {e}")
            return []
    
    async def _generate_inventory_based_quotes(self) -> List[Order]:
        """库存驱动策略"""
        try:
            if self.current_price <= 0:
                return []
            
            quotes = []
            
            # 计算库存偏离
            inventory_skew = (self.state.current_inventory - self.config.target_inventory) / self.config.max_inventory
            
            # 基础价差
            base_spread = self.current_price * (self.config.base_spread_bps / 10000)
            
            # 库存调整 - 库存过多时提高卖价降低买价，库存过少时相反
            inventory_adjustment = inventory_skew * self.config.inventory_penalty_rate * self.current_price
            
            for level in range(self.config.max_quote_levels):
                spread_multiplier = 1 + level * 0.3
                level_spread = base_spread * spread_multiplier
                
                # 应用库存偏斜
                bid_price = self.current_price - level_spread / 2 - inventory_adjustment
                ask_price = self.current_price + level_spread / 2 + inventory_adjustment
                
                # 库存影响报价数量
                if inventory_skew > 0:  # 库存过多，增加卖单数量
                    bid_quantity = self.config.quote_size * (0.7 ** level) * (1 - abs(inventory_skew) * 0.5)
                    ask_quantity = self.config.quote_size * (0.7 ** level) * (1 + abs(inventory_skew) * 0.5)
                else:  # 库存过少，增加买单数量
                    bid_quantity = self.config.quote_size * (0.7 ** level) * (1 + abs(inventory_skew) * 0.5)
                    ask_quantity = self.config.quote_size * (0.7 ** level) * (1 - abs(inventory_skew) * 0.5)
                
                current_time = datetime.utcnow()
                
                if bid_quantity > 10:  # 最小数量限制
                    bid_order = Order(
                        order_id=f"mm_inv_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=bid_quantity,
                        price=max(0.01, bid_price),  # 价格不能为负
                        strategy_id="market_maker_inventory"
                    )
                    quotes.append(bid_order)
                
                if ask_quantity > 10:
                    ask_order = Order(
                        order_id=f"mm_inv_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=ask_quantity,
                        price=ask_price,
                        strategy_id="market_maker_inventory"
                    )
                    quotes.append(ask_order)
            
            return quotes
            
        except Exception as e:
            logger.error(f"生成库存驱动报价失败: {e}")
            return []
    
    async def _generate_volatility_adjusted_quotes(self) -> List[Order]:
        """波动率调整策略"""
        try:
            if self.current_price <= 0:
                return []
            
            quotes = []
            
            # 根据波动率调整价差
            volatility_multiplier = 1 + self.volatility * 10  # 波动率越高，价差越大
            adjusted_spread_bps = self.config.base_spread_bps * volatility_multiplier
            adjusted_spread_bps = max(self.config.min_spread_bps, 
                                    min(adjusted_spread_bps, self.config.max_spread_bps))
            
            base_spread = self.current_price * (adjusted_spread_bps / 10000)
            
            # 波动率影响报价数量
            volatility_quantity_factor = max(0.3, 1 - self.volatility * 5)
            
            for level in range(self.config.max_quote_levels):
                spread_multiplier = 1 + level * 0.25
                level_spread = base_spread * spread_multiplier
                
                bid_price = self.current_price - level_spread / 2
                ask_price = self.current_price + level_spread / 2
                
                level_quantity = self.config.quote_size * (0.8 ** level) * volatility_quantity_factor
                
                current_time = datetime.utcnow()
                
                bid_order = Order(
                    order_id=f"mm_vol_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=bid_price,
                    strategy_id="market_maker_volatility"
                )
                
                ask_order = Order(
                    order_id=f"mm_vol_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                    symbol=self.config.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=level_quantity,
                    price=ask_price,
                    strategy_id="market_maker_volatility"
                )
                
                quotes.extend([bid_order, ask_order])
            
            return quotes
            
        except Exception as e:
            logger.error(f"生成波动率调整报价失败: {e}")
            return []
    
    async def _generate_adaptive_quotes(self) -> List[Order]:
        """自适应策略 - 结合多种因素"""
        try:
            if self.current_price <= 0:
                return []
            
            # 综合考虑库存、波动率、市场流动性等因素
            inventory_skew = (self.state.current_inventory - self.config.target_inventory) / self.config.max_inventory
            volatility_factor = 1 + self.volatility * 8
            
            # 流动性因素
            liquidity_factor = 1.0
            if self.bid_depth > 0 and self.ask_depth > 0:
                avg_depth = (self.bid_depth + self.ask_depth) / 2
                liquidity_factor = max(0.5, min(2.0, 1000 / max(avg_depth, 100)))
            
            # 动态调整价差
            adaptive_spread_bps = (self.config.base_spread_bps * volatility_factor * liquidity_factor)
            adaptive_spread_bps = max(self.config.min_spread_bps, 
                                    min(adaptive_spread_bps, self.config.max_spread_bps))
            
            base_spread = self.current_price * (adaptive_spread_bps / 10000)
            
            # 库存调整
            inventory_adjustment = inventory_skew * self.config.inventory_penalty_rate * self.current_price
            
            quotes = []
            current_time = datetime.utcnow()
            
            for level in range(self.config.max_quote_levels):
                spread_multiplier = 1 + level * 0.4
                level_spread = base_spread * spread_multiplier
                
                bid_price = self.current_price - level_spread / 2 - inventory_adjustment
                ask_price = self.current_price + level_spread / 2 + inventory_adjustment
                
                # 自适应数量调整
                quantity_factor = (1 / volatility_factor) * (1 / liquidity_factor) * (0.75 ** level)
                
                if inventory_skew > 0:
                    bid_quantity = self.config.quote_size * quantity_factor * (1 - abs(inventory_skew) * 0.7)
                    ask_quantity = self.config.quote_size * quantity_factor * (1 + abs(inventory_skew) * 0.3)
                else:
                    bid_quantity = self.config.quote_size * quantity_factor * (1 + abs(inventory_skew) * 0.3)
                    ask_quantity = self.config.quote_size * quantity_factor * (1 - abs(inventory_skew) * 0.7)
                
                if bid_quantity > 5 and bid_price > 0:
                    bid_order = Order(
                        order_id=f"mm_adapt_bid_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=bid_quantity,
                        price=bid_price,
                        strategy_id="market_maker_adaptive"
                    )
                    quotes.append(bid_order)
                
                if ask_quantity > 5:
                    ask_order = Order(
                        order_id=f"mm_adapt_ask_{self.config.symbol}_{level}_{current_time.timestamp()}",
                        symbol=self.config.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=ask_quantity,
                        price=ask_price,
                        strategy_id="market_maker_adaptive"
                    )
                    quotes.append(ask_order)
            
            return quotes
            
        except Exception as e:
            logger.error(f"生成自适应报价失败: {e}")
            return []
    
    async def _check_risk_limits(self) -> bool:
        """检查风险限额"""
        try:
            # 检查仓位限额
            if abs(self.state.current_inventory) > self.config.max_inventory:
                if not self.position_limit_reached:
                    logger.warning(f"做市商仓位限额触发: {self.state.current_inventory}")
                    self.position_limit_reached = True
                return True
            else:
                self.position_limit_reached = False
            
            # 检查风险限额
            total_risk = abs(self.state.current_inventory * self.current_price)
            if total_risk > self.config.risk_limit:
                if not self.risk_limit_reached:
                    logger.warning(f"做市商风险限额触发: {total_risk}")
                    self.risk_limit_reached = True
                return True
            else:
                self.risk_limit_reached = False
            
            return False
            
        except Exception as e:
            logger.error(f"检查风险限额失败: {e}")
            return True
    
    def on_trade_executed(self, trade_data: Dict[str, Any]) -> None:
        """处理成交事件"""
        try:
            if trade_data.get('strategy_id', '').startswith('market_maker'):
                self.fill_count += 1
                
                # 更新库存
                if trade_data['side'] == 'buy':
                    self.state.current_inventory += trade_data['quantity']
                    self.state.bid_trades += 1
                else:
                    self.state.current_inventory -= trade_data['quantity']
                    self.state.ask_trades += 1
                
                self.state.total_trades += 1
                
                # 更新PnL (简化计算)
                trade_pnl = 0.0
                if 'pnl' in trade_data:
                    trade_pnl = trade_data['pnl']
                    self.state.realized_pnl += trade_pnl
                    self.state.total_pnl += trade_pnl
                
                self.state.last_update = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"处理成交事件失败: {e}")
    
    def get_market_maker_metrics(self) -> Dict[str, Any]:
        """获取做市商指标"""
        try:
            # 计算报价命中率
            if self.quote_count > 0:
                self.quote_hit_ratio = self.fill_count / (self.quote_count / 2)  # 除以2因为每次报价生成买卖两单
            
            # 计算当前未实现盈亏
            self.state.unrealized_pnl = self.state.current_inventory * self.current_price
            self.state.total_pnl = self.state.realized_pnl + self.state.unrealized_pnl
            
            # 计算平均价差
            avg_spread_bps = np.mean(list(self.spread_history)) if self.spread_history else 0
            
            return {
                'symbol': self.config.symbol,
                'strategy': self.strategy.value,
                'current_inventory': self.state.current_inventory,
                'inventory_ratio': self.state.current_inventory / self.config.max_inventory,
                'total_pnl': self.state.total_pnl,
                'realized_pnl': self.state.realized_pnl,
                'unrealized_pnl': self.state.unrealized_pnl,
                'total_trades': self.state.total_trades,
                'bid_trades': self.state.bid_trades,
                'ask_trades': self.state.ask_trades,
                'quote_count': self.quote_count,
                'fill_count': self.fill_count,
                'quote_hit_ratio': self.quote_hit_ratio,
                'avg_spread_bps': avg_spread_bps,
                'current_volatility': self.volatility,
                'position_limit_reached': self.position_limit_reached,
                'risk_limit_reached': self.risk_limit_reached,
                'last_update': self.state.last_update
            }
            
        except Exception as e:
            logger.error(f"获取做市商指标失败: {e}")
            return {}
    
    def reset_state(self) -> None:
        """重置做市商状态"""
        self.state = MarketMakerState(symbol=self.config.symbol)
        self.quote_count = 0
        self.fill_count = 0
        self.quote_hit_ratio = 0.0
        self.position_limit_reached = False
        self.risk_limit_reached = False
        logger.info(f"做市商状态已重置: {self.config.symbol}")