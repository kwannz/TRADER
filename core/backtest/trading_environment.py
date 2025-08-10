"""
交易环境模拟

提供完整的虚拟交易环境：
- 虚拟账户管理
- 订单管理系统
- 成交模拟
- 手续费和滑点计算
- 风险控制
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open" 
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    time_in_force: str = "GTC"  # Good Till Cancelled

@dataclass
class Position:
    """持仓"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    commission_paid: float
    entry_time: datetime
    last_update: datetime

@dataclass
class Trade:
    """交易记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    strategy_id: Optional[str] = None
    pnl: Optional[float] = None

class TradingEnvironment:
    """交易环境主类"""
    
    def __init__(self, initial_balance: float = 10000.0,
                 commission_rate: float = 0.001,
                 slippage_bps: float = 2.0,
                 max_slippage_pct: float = 0.05):
        
        # 账户设置
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000.0  # 基点转换为小数
        self.max_slippage_pct = max_slippage_pct / 100.0
        
        # 交易数据
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.open_orders: Dict[str, Order] = {}
        
        # 市场数据缓存
        self.current_prices: Dict[str, float] = {}
        self.bid_ask_spreads: Dict[str, Tuple[float, float]] = {}
        
        # 事件订阅
        self.trade_event_handlers: List[Callable] = []
        self.order_event_handlers: List[Callable] = []
        self.position_event_handlers: List[Callable] = []
        
        # 风险控制参数
        self.max_position_size = {}  # 单个品种最大仓位
        self.max_total_exposure = initial_balance * 0.95  # 最大总敞口
        self.margin_requirement = 0.1  # 保证金要求
        
        # 统计数据
        self.total_commission = 0.0
        self.total_trades = 0
        self.total_volume = 0.0
        
        logger.info(f"交易环境初始化: 初始资金={initial_balance}, 手续费率={commission_rate}")
    
    async def initialize(self) -> None:
        """初始化交易环境"""
        try:
            # 重置所有状态
            self.balance = self.initial_balance
            self.orders.clear()
            self.positions.clear()
            self.trades.clear()
            self.open_orders.clear()
            self.current_prices.clear()
            
            logger.info("交易环境初始化完成")
            
        except Exception as e:
            logger.error(f"交易环境初始化失败: {e}")
            raise
    
    # 订阅方法
    def subscribe_trade_event(self, handler: Callable) -> None:
        """订阅交易事件"""
        self.trade_event_handlers.append(handler)
    
    def subscribe_order_event(self, handler: Callable) -> None:
        """订阅订单事件"""
        self.order_event_handlers.append(handler)
    
    def subscribe_position_event(self, handler: Callable) -> None:
        """订阅仓位事件"""
        self.position_event_handlers.append(handler)
    
    # 市场数据更新
    async def update_market_data(self, symbol: str, price: float,
                               bid: Optional[float] = None,
                               ask: Optional[float] = None) -> None:
        """更新市场数据"""
        try:
            self.current_prices[symbol] = price
            
            if bid and ask:
                self.bid_ask_spreads[symbol] = (bid, ask)
            else:
                # 估算买卖价差
                spread = price * 0.0001  # 0.01%价差
                self.bid_ask_spreads[symbol] = (price - spread/2, price + spread/2)
            
            # 更新持仓未实现盈亏
            await self._update_position_pnl(symbol, price)
            
            # 检查停止订单
            await self._check_stop_orders(symbol, price)
            
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")
    
    # 订单管理
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         strategy_id: Optional[str] = None,
                         client_order_id: Optional[str] = None) -> str:
        """下单"""
        try:
            # 生成订单ID
            order_id = str(uuid.uuid4())
            
            # 创建订单
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                created_time=datetime.utcnow(),
                updated_time=datetime.utcnow(),
                strategy_id=strategy_id,
                client_order_id=client_order_id
            )
            
            # 验证订单
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                order.status = OrderStatus.REJECTED
                await self._notify_order_event(order)
                logger.warning(f"订单被拒绝: {validation_result['reason']}")
                return order_id
            
            # 保存订单
            self.orders[order_id] = order
            
            # 处理订单
            if order_type == OrderType.MARKET:
                await self._process_market_order(order)
            else:
                order.status = OrderStatus.OPEN
                self.open_orders[order_id] = order
                await self._notify_order_event(order)
            
            logger.debug(f"订单已提交: {symbol} {side.value} {quantity} @ {price}")
            return order_id
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            if order_id not in self.orders:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            order = self.orders[order_id]
            
            if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIAL_FILLED]:
                logger.warning(f"订单状态不允许取消: {order.status}")
                return False
            
            # 取消订单
            order.status = OrderStatus.CANCELLED
            order.updated_time = datetime.utcnow()
            
            # 从活动订单中移除
            if order_id in self.open_orders:
                del self.open_orders[order_id]
            
            await self._notify_order_event(order)
            logger.debug(f"订单已取消: {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    async def _validate_order(self, order: Order) -> Dict[str, Any]:
        """验证订单"""
        try:
            # 检查symbol价格
            if order.symbol not in self.current_prices:
                return {'valid': False, 'reason': f'未知品种: {order.symbol}'}
            
            # 检查数量
            if order.quantity <= 0:
                return {'valid': False, 'reason': '订单数量必须大于0'}
            
            # 检查资金
            current_price = self.current_prices[order.symbol]
            estimated_cost = order.quantity * (order.price or current_price)
            
            if order.side == OrderSide.BUY:
                # 买入检查资金
                if estimated_cost > self.balance:
                    return {'valid': False, 'reason': '资金不足'}
            else:
                # 卖出检查持仓
                position = self.positions.get(order.symbol)
                if not position or position.size < order.quantity:
                    return {'valid': False, 'reason': '持仓不足'}
            
            # 检查价格合理性
            if order.price:
                price_deviation = abs(order.price - current_price) / current_price
                if price_deviation > 0.1:  # 10%偏差限制
                    return {'valid': False, 'reason': '价格偏差过大'}
            
            return {'valid': True, 'reason': ''}
            
        except Exception as e:
            logger.error(f"订单验证失败: {e}")
            return {'valid': False, 'reason': f'验证异常: {e}'}
    
    async def _process_market_order(self, order: Order) -> None:
        """处理市价订单"""
        try:
            current_price = self.current_prices[order.symbol]
            bid, ask = self.bid_ask_spreads.get(order.symbol, (current_price, current_price))
            
            # 确定成交价格
            if order.side == OrderSide.BUY:
                execution_price = ask
            else:
                execution_price = bid
            
            # 计算滑点
            slippage = self._calculate_slippage(order.quantity, current_price)
            if order.side == OrderSide.BUY:
                execution_price += slippage
            else:
                execution_price -= slippage
            
            # 执行交易
            await self._execute_trade(order, order.quantity, execution_price)
            
        except Exception as e:
            logger.error(f"处理市价订单失败: {e}")
            order.status = OrderStatus.REJECTED
            await self._notify_order_event(order)
    
    def _calculate_slippage(self, quantity: float, price: float) -> float:
        """计算滑点"""
        try:
            # 基于数量的线性滑点模型
            base_slippage = price * self.slippage_bps
            
            # 大订单额外滑点
            if quantity * price > 1000:  # 大于1000USDT的订单
                size_impact = (quantity * price / 10000) * price * 0.0001
                base_slippage += size_impact
            
            # 限制最大滑点
            max_slippage = price * self.max_slippage_pct
            return min(base_slippage, max_slippage)
            
        except Exception as e:
            logger.error(f"计算滑点失败: {e}")
            return price * self.slippage_bps
    
    async def _execute_trade(self, order: Order, quantity: float, price: float) -> None:
        """执行交易"""
        try:
            # 计算手续费
            commission = quantity * price * self.commission_rate
            
            # 创建交易记录
            trade_id = str(uuid.uuid4())
            trade = Trade(
                trade_id=trade_id,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=price,
                commission=commission,
                timestamp=datetime.utcnow(),
                strategy_id=order.strategy_id
            )
            
            # 更新订单状态
            order.filled_quantity += quantity
            order.average_fill_price = ((order.average_fill_price * (order.filled_quantity - quantity) + 
                                      price * quantity) / order.filled_quantity)
            order.commission += commission
            order.updated_time = datetime.utcnow()
            
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                if order.order_id in self.open_orders:
                    del self.open_orders[order.order_id]
            else:
                order.status = OrderStatus.PARTIAL_FILLED
            
            # 更新账户余额
            if order.side == OrderSide.BUY:
                self.balance -= (quantity * price + commission)
            else:
                self.balance += (quantity * price - commission)
            
            # 更新持仓
            await self._update_position(order, quantity, price, commission)
            
            # 保存交易记录
            self.trades.append(trade)
            self.total_commission += commission
            self.total_trades += 1
            self.total_volume += quantity * price
            
            # 通知事件
            await self._notify_trade_event(trade)
            await self._notify_order_event(order)
            
            logger.debug(f"交易执行: {order.symbol} {order.side.value} {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
    
    async def _update_position(self, order: Order, quantity: float, 
                              price: float, commission: float) -> None:
        """更新持仓"""
        try:
            symbol = order.symbol
            
            if symbol not in self.positions:
                # 新建持仓
                side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=quantity if order.side == OrderSide.BUY else -quantity,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    commission_paid=commission,
                    entry_time=datetime.utcnow(),
                    last_update=datetime.utcnow()
                )
            else:
                # 更新现有持仓
                position = self.positions[symbol]
                
                if order.side == OrderSide.BUY:
                    if position.size >= 0:  # 同方向
                        # 加仓
                        total_cost = position.size * position.entry_price + quantity * price
                        position.size += quantity
                        position.entry_price = total_cost / position.size
                    else:  # 反方向
                        # 平仓或开反向仓
                        if quantity <= abs(position.size):
                            # 部分或完全平仓
                            realized_pnl = quantity * (position.entry_price - price) if position.side == PositionSide.SHORT else quantity * (price - position.entry_price)
                            position.realized_pnl += realized_pnl
                            position.size += quantity
                        else:
                            # 平仓后开反向仓
                            close_quantity = abs(position.size)
                            realized_pnl = close_quantity * (position.entry_price - price) if position.side == PositionSide.SHORT else close_quantity * (price - position.entry_price)
                            position.realized_pnl += realized_pnl
                            
                            remaining_quantity = quantity - close_quantity
                            position.size = remaining_quantity
                            position.side = PositionSide.LONG
                            position.entry_price = price
                            position.entry_time = datetime.utcnow()
                else:  # SELL
                    if position.size <= 0:  # 同方向
                        # 加仓
                        total_cost = abs(position.size) * position.entry_price + quantity * price
                        position.size -= quantity
                        position.entry_price = total_cost / abs(position.size)
                    else:  # 反方向
                        # 平仓或开反向仓
                        if quantity <= position.size:
                            # 部分或完全平仓
                            realized_pnl = quantity * (price - position.entry_price) if position.side == PositionSide.LONG else quantity * (position.entry_price - price)
                            position.realized_pnl += realized_pnl
                            position.size -= quantity
                        else:
                            # 平仓后开反向仓
                            close_quantity = position.size
                            realized_pnl = close_quantity * (price - position.entry_price) if position.side == PositionSide.LONG else close_quantity * (position.entry_price - price)
                            position.realized_pnl += realized_pnl
                            
                            remaining_quantity = quantity - close_quantity
                            position.size = -remaining_quantity
                            position.side = PositionSide.SHORT
                            position.entry_price = price
                            position.entry_time = datetime.utcnow()
                
                position.commission_paid += commission
                position.last_update = datetime.utcnow()
                
                # 如果持仓为0，移除持仓
                if abs(position.size) < 1e-8:
                    del self.positions[symbol]
            
            await self._notify_position_event(self.positions.get(symbol))
            
        except Exception as e:
            logger.error(f"更新持仓失败: {e}")
    
    async def _update_position_pnl(self, symbol: str, current_price: float) -> None:
        """更新持仓未实现盈亏"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = position.size * (current_price - position.entry_price)
            else:
                position.unrealized_pnl = abs(position.size) * (position.entry_price - current_price)
            
            position.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"更新持仓盈亏失败: {e}")
    
    async def _check_stop_orders(self, symbol: str, current_price: float) -> None:
        """检查停止订单"""
        try:
            orders_to_trigger = []
            
            for order_id, order in self.open_orders.items():
                if order.symbol != symbol or not order.stop_price:
                    continue
                
                should_trigger = False
                
                if order.order_type == OrderType.STOP_LOSS:
                    if order.side == OrderSide.BUY and current_price >= order.stop_price:
                        should_trigger = True
                    elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                        should_trigger = True
                
                elif order.order_type == OrderType.TAKE_PROFIT:
                    if order.side == OrderSide.BUY and current_price <= order.stop_price:
                        should_trigger = True
                    elif order.side == OrderSide.SELL and current_price >= order.stop_price:
                        should_trigger = True
                
                if should_trigger:
                    orders_to_trigger.append(order)
            
            # 触发停止订单
            for order in orders_to_trigger:
                if order.order_type == OrderType.STOP_LIMIT:
                    # 转换为限价订单
                    order.order_type = OrderType.LIMIT
                    order.price = order.stop_price
                else:
                    # 转换为市价订单执行
                    del self.open_orders[order.order_id]
                    await self._process_market_order(order)
                
        except Exception as e:
            logger.error(f"检查停止订单失败: {e}")
    
    # 账户查询
    def get_balance(self) -> float:
        """获取账户余额"""
        return self.balance
    
    def get_total_equity(self) -> float:
        """获取总权益"""
        try:
            total_equity = self.balance
            
            # 加上持仓市值
            for position in self.positions.values():
                if position.current_price > 0:
                    if position.side == PositionSide.LONG:
                        total_equity += position.size * position.current_price
                    else:
                        total_equity += abs(position.size) * position.current_price
            
            return total_equity
            
        except Exception as e:
            logger.error(f"计算总权益失败: {e}")
            return self.balance
    
    def get_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取特定持仓"""
        return self.positions.get(symbol)
    
    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """获取订单"""
        if status:
            return [order for order in self.orders.values() if order.status == status]
        return list(self.orders.values())
    
    def get_open_orders(self) -> List[Order]:
        """获取活动订单"""
        return list(self.open_orders.values())
    
    async def get_trade_history(self, symbol: Optional[str] = None, 
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取交易历史"""
        try:
            trades = self.trades
            
            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            
            if limit:
                trades = trades[-limit:]
            
            # 计算每笔交易的PnL
            trade_dicts = []
            for trade in trades:
                trade_dict = asdict(trade)
                trade_dict['timestamp'] = trade.timestamp
                
                # 简化的PnL计算（实际应该考虑持仓成本）
                if trade.pnl is None:
                    trade_dict['pnl'] = 0.0  # 需要更复杂的计算
                
                trade_dicts.append(trade_dict)
            
            return trade_dicts
            
        except Exception as e:
            logger.error(f"获取交易历史失败: {e}")
            return []
    
    def get_account_summary(self) -> Dict[str, Any]:
        """获取账户摘要"""
        try:
            total_equity = self.get_total_equity()
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            
            return {
                'balance': self.balance,
                'total_equity': total_equity,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': total_realized_pnl,
                'total_pnl': total_realized_pnl + total_unrealized_pnl,
                'total_return': (total_equity - self.initial_balance) / self.initial_balance,
                'total_commission': self.total_commission,
                'total_trades': self.total_trades,
                'total_volume': self.total_volume,
                'open_positions': len(self.positions),
                'open_orders': len(self.open_orders)
            }
            
        except Exception as e:
            logger.error(f"获取账户摘要失败: {e}")
            return {}
    
    # 风险管理
    async def close_position(self, symbol: str, quantity: Optional[float] = None) -> str:
        """平仓"""
        try:
            if symbol not in self.positions:
                raise ValueError(f"无持仓: {symbol}")
            
            position = self.positions[symbol]
            close_quantity = quantity or abs(position.size)
            
            # 确定平仓方向
            if position.side == PositionSide.LONG:
                close_side = OrderSide.SELL
            else:
                close_side = OrderSide.BUY
            
            # 下市价单平仓
            order_id = await self.place_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=close_quantity,
                strategy_id="close_position"
            )
            
            return order_id
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            raise
    
    async def close_all_positions(self) -> List[str]:
        """平掉所有仓位"""
        try:
            order_ids = []
            
            for symbol in list(self.positions.keys()):
                order_id = await self.close_position(symbol)
                order_ids.append(order_id)
            
            return order_ids
            
        except Exception as e:
            logger.error(f"平掉所有仓位失败: {e}")
            return []
    
    # 事件通知
    async def _notify_trade_event(self, trade: Trade) -> None:
        """通知交易事件"""
        for handler in self.trade_event_handlers:
            try:
                trade_data = asdict(trade)
                trade_data['timestamp'] = trade.timestamp
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(trade_data)
                else:
                    handler(trade_data)
            except Exception as e:
                logger.error(f"交易事件通知失败: {e}")
    
    async def _notify_order_event(self, order: Order) -> None:
        """通知订单事件"""
        for handler in self.order_event_handlers:
            try:
                order_data = asdict(order)
                order_data['created_time'] = order.created_time
                order_data['updated_time'] = order.updated_time
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(order_data)
                else:
                    handler(order_data)
            except Exception as e:
                logger.error(f"订单事件通知失败: {e}")
    
    async def _notify_position_event(self, position: Optional[Position]) -> None:
        """通知持仓事件"""
        if not position:
            return
            
        for handler in self.position_event_handlers:
            try:
                position_data = asdict(position)
                position_data['entry_time'] = position.entry_time
                position_data['last_update'] = position.last_update
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(position_data)
                else:
                    handler(position_data)
            except Exception as e:
                logger.error(f"持仓事件通知失败: {e}")