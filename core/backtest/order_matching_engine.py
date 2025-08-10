"""
订单撮合引擎

提供订单簿管理和订单撮合功能：
- 限价订单簿维护
- 市价订单即时撮合
- 价格时间优先撮合算法
- 部分成交支持
- 订单簿深度计算
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import bisect
from collections import defaultdict, deque

from .trading_environment import Order, OrderType, OrderSide, OrderStatus, Trade
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class OrderBookEntry:
    """订单簿条目"""
    order_id: str
    price: float
    quantity: float
    timestamp: datetime
    order: Order

@dataclass  
class OrderBook:
    """订单簿"""
    symbol: str
    bids: List[OrderBookEntry] = field(default_factory=list)  # 买单按价格降序
    asks: List[OrderBookEntry] = field(default_factory=list)  # 卖单按价格升序
    last_update: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MarketDepth:
    """市场深度"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(价格, 数量)]
    asks: List[Tuple[float, float]]  # [(价格, 数量)]
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    mid_price: Optional[float] = None

class MatchingAlgorithm(Enum):
    """撮合算法"""
    PRICE_TIME = "price_time"  # 价格时间优先
    PRO_RATA = "pro_rata"      # 比例分配
    SIZE_TIME = "size_time"    # 数量时间优先

class OrderMatchingEngine:
    """订单撮合引擎主类"""
    
    def __init__(self, matching_algorithm: MatchingAlgorithm = MatchingAlgorithm.PRICE_TIME,
                 tick_size: float = 0.01,
                 enable_partial_fills: bool = True):
        
        # 撮合设置
        self.matching_algorithm = matching_algorithm
        self.tick_size = tick_size
        self.enable_partial_fills = enable_partial_fills
        
        # 订单簿存储
        self.order_books: Dict[str, OrderBook] = {}
        self.order_index: Dict[str, OrderBookEntry] = {}  # 订单ID -> 订单簿条目
        
        # 撮合统计
        self.total_matches = 0
        self.total_volume = 0.0
        self.match_latency_stats = deque(maxlen=10000)
        
        # 事件处理
        self.match_event_handlers: List[Callable] = []
        self.depth_update_handlers: List[Callable] = []
        self.trade_event_handlers: List[Callable] = []
        
        # 市场数据缓存
        self.market_depths: Dict[str, MarketDepth] = {}
        self.recent_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info(f"订单撮合引擎初始化: {matching_algorithm.value}算法, 最小变动价位={tick_size}")
    
    async def initialize(self) -> None:
        """初始化撮合引擎"""
        try:
            # 清空所有状态
            self.order_books.clear()
            self.order_index.clear()
            self.market_depths.clear()
            self.recent_trades.clear()
            
            self.total_matches = 0
            self.total_volume = 0.0
            self.match_latency_stats.clear()
            
            logger.info("订单撮合引擎初始化完成")
            
        except Exception as e:
            logger.error(f"撮合引擎初始化失败: {e}")
            raise
    
    # 事件订阅
    def subscribe_match_event(self, handler: Callable) -> None:
        """订阅撮合事件"""
        self.match_event_handlers.append(handler)
    
    def subscribe_depth_update(self, handler: Callable) -> None:
        """订阅深度更新事件"""
        self.depth_update_handlers.append(handler)
    
    def subscribe_trade_event(self, handler: Callable) -> None:
        """订阅交易事件"""
        self.trade_event_handlers.append(handler)
    
    # 订单处理
    async def add_order(self, order: Order) -> List[Trade]:
        """添加订单到撮合引擎"""
        try:
            start_time = datetime.utcnow()
            trades = []
            
            # 确保订单簿存在
            if order.symbol not in self.order_books:
                self.order_books[order.symbol] = OrderBook(symbol=order.symbol)
            
            order_book = self.order_books[order.symbol]
            
            # 市价订单立即撮合
            if order.order_type == OrderType.MARKET:
                trades = await self._match_market_order(order, order_book)
            else:
                # 限价订单先尝试撮合，剩余部分进入订单簿
                trades = await self._match_limit_order(order, order_book)
                
                # 如果订单未完全成交，加入订单簿
                if order.filled_quantity < order.quantity:
                    await self._add_to_order_book(order, order_book)
            
            # 更新市场深度
            await self._update_market_depth(order.symbol)
            
            # 记录撮合延迟
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.match_latency_stats.append(latency)
            
            # 通知事件
            for trade in trades:
                await self._notify_trade_event(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"添加订单失败: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            if order_id not in self.order_index:
                logger.warning(f"订单不存在: {order_id}")
                return False
            
            # 获取订单信息
            entry = self.order_index[order_id]
            order_book = self.order_books[entry.order.symbol]
            
            # 从订单簿移除
            if entry.order.side == OrderSide.BUY:
                order_book.bids.remove(entry)
            else:
                order_book.asks.remove(entry)
            
            # 从索引移除
            del self.order_index[order_id]
            
            # 更新订单状态
            entry.order.status = OrderStatus.CANCELLED
            entry.order.updated_time = datetime.utcnow()
            
            # 更新市场深度
            await self._update_market_depth(entry.order.symbol)
            
            logger.debug(f"订单已取消: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败: {e}")
            return False
    
    async def _match_market_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """撮合市价订单"""
        try:
            trades = []
            remaining_quantity = order.quantity
            
            # 选择对手盘
            counterparty_orders = order_book.asks if order.side == OrderSide.BUY else order_book.bids
            
            if not counterparty_orders:
                # 没有对手盘，订单被拒绝
                order.status = OrderStatus.REJECTED
                logger.warning(f"市价订单无法撮合: {order.symbol} 缺少对手盘")
                return trades
            
            # 按价格优先顺序撮合
            for entry in counterparty_orders[:]:  # 创建副本避免修改时的问题
                if remaining_quantity <= 0:
                    break
                
                # 计算成交数量
                match_quantity = min(remaining_quantity, entry.quantity)
                match_price = entry.price
                
                # 创建交易记录
                trade = await self._create_trade(
                    order, entry.order, match_quantity, match_price
                )
                trades.append(trade)
                
                # 更新订单状态
                await self._update_order_after_match(order, match_quantity, match_price)
                await self._update_order_after_match(entry.order, match_quantity, match_price)
                
                # 更新订单簿条目
                entry.quantity -= match_quantity
                remaining_quantity -= match_quantity
                
                # 如果对手订单完全成交，从订单簿移除
                if entry.quantity <= 0:
                    counterparty_orders.remove(entry)
                    del self.order_index[entry.order_id]
                    entry.order.status = OrderStatus.FILLED
                
            # 更新市价订单状态
            if remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIAL_FILLED
            else:
                order.status = OrderStatus.REJECTED  # 无法成交
            
            return trades
            
        except Exception as e:
            logger.error(f"撮合市价订单失败: {e}")
            return []
    
    async def _match_limit_order(self, order: Order, order_book: OrderBook) -> List[Trade]:
        """撮合限价订单"""
        try:
            trades = []
            remaining_quantity = order.quantity
            
            # 选择对手盘并确定撮合条件
            if order.side == OrderSide.BUY:
                counterparty_orders = order_book.asks
                can_match = lambda entry_price: entry_price <= order.price
            else:
                counterparty_orders = order_book.bids  
                can_match = lambda entry_price: entry_price >= order.price
            
            # 撮合循环
            for entry in counterparty_orders[:]:
                if remaining_quantity <= 0:
                    break
                
                # 检查价格是否能撮合
                if not can_match(entry.price):
                    break  # 由于排序，后面的价格更不利，直接跳出
                
                # 计算成交数量和价格
                match_quantity = min(remaining_quantity, entry.quantity)
                match_price = entry.price  # 使用对手盘价格
                
                # 创建交易记录
                trade = await self._create_trade(
                    order, entry.order, match_quantity, match_price
                )
                trades.append(trade)
                
                # 更新订单状态
                await self._update_order_after_match(order, match_quantity, match_price)
                await self._update_order_after_match(entry.order, match_quantity, match_price)
                
                # 更新订单簿条目
                entry.quantity -= match_quantity
                remaining_quantity -= match_quantity
                
                # 处理完全成交的对手订单
                if entry.quantity <= 0:
                    counterparty_orders.remove(entry)
                    del self.order_index[entry.order_id]
                    entry.order.status = OrderStatus.FILLED
            
            # 更新限价订单状态
            if remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
            elif order.filled_quantity > 0:
                order.status = OrderStatus.PARTIAL_FILLED
            else:
                order.status = OrderStatus.OPEN  # 未成交，进入订单簿
            
            return trades
            
        except Exception as e:
            logger.error(f"撮合限价订单失败: {e}")
            return []
    
    async def _add_to_order_book(self, order: Order, order_book: OrderBook) -> None:
        """添加订单到订单簿"""
        try:
            remaining_quantity = order.quantity - order.filled_quantity
            
            if remaining_quantity <= 0:
                return
            
            # 创建订单簿条目
            entry = OrderBookEntry(
                order_id=order.order_id,
                price=order.price,
                quantity=remaining_quantity,
                timestamp=order.created_time or datetime.utcnow(),
                order=order
            )
            
            # 添加到相应的订单簿侧
            if order.side == OrderSide.BUY:
                # 买单按价格降序插入
                bisect.insort(order_book.bids, entry, key=lambda x: (-x.price, x.timestamp))
            else:
                # 卖单按价格升序插入
                bisect.insort(order_book.asks, entry, key=lambda x: (x.price, x.timestamp))
            
            # 添加到索引
            self.order_index[order.order_id] = entry
            
            # 更新订单状态
            order.status = OrderStatus.OPEN
            order_book.last_update = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"添加订单到订单簿失败: {e}")
    
    async def _create_trade(self, taker_order: Order, maker_order: Order, 
                           quantity: float, price: float) -> Trade:
        """创建交易记录"""
        try:
            trade_id = str(uuid.uuid4())
            
            trade = Trade(
                trade_id=trade_id,
                order_id=taker_order.order_id,  # 主要使用taker订单ID
                symbol=taker_order.symbol,
                side=taker_order.side,
                quantity=quantity,
                price=price,
                commission=0.0,  # 手续费将在TradingEnvironment中计算
                timestamp=datetime.utcnow(),
                strategy_id=taker_order.strategy_id
            )
            
            # 记录交易到历史
            self.recent_trades[taker_order.symbol].append({
                'trade_id': trade_id,
                'timestamp': trade.timestamp,
                'price': price,
                'quantity': quantity,
                'taker_side': taker_order.side.value,
                'taker_order_id': taker_order.order_id,
                'maker_order_id': maker_order.order_id
            })
            
            # 更新统计
            self.total_matches += 1
            self.total_volume += quantity * price
            
            return trade
            
        except Exception as e:
            logger.error(f"创建交易记录失败: {e}")
            raise
    
    async def _update_order_after_match(self, order: Order, quantity: float, price: float) -> None:
        """撮合后更新订单"""
        try:
            # 更新成交数量和均价
            old_filled = order.filled_quantity
            order.filled_quantity += quantity
            
            # 计算成交均价
            if old_filled == 0:
                order.average_fill_price = price
            else:
                total_value = old_filled * order.average_fill_price + quantity * price
                order.average_fill_price = total_value / order.filled_quantity
            
            # 更新时间
            order.updated_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"更新订单状态失败: {e}")
    
    async def _update_market_depth(self, symbol: str) -> None:
        """更新市场深度"""
        try:
            if symbol not in self.order_books:
                return
            
            order_book = self.order_books[symbol]
            timestamp = datetime.utcnow()
            
            # 聚合买盘深度
            bid_levels = {}
            for entry in order_book.bids:
                if entry.price in bid_levels:
                    bid_levels[entry.price] += entry.quantity
                else:
                    bid_levels[entry.price] = entry.quantity
            
            # 聚合卖盘深度  
            ask_levels = {}
            for entry in order_book.asks:
                if entry.price in ask_levels:
                    ask_levels[entry.price] += entry.quantity
                else:
                    ask_levels[entry.price] = entry.quantity
            
            # 排序并转换格式
            bids = sorted(bid_levels.items(), key=lambda x: x[0], reverse=True)[:20]  # 前20档
            asks = sorted(ask_levels.items(), key=lambda x: x[0])[:20]
            
            # 计算最优买卖价和价差
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            spread = (best_ask - best_bid) if (best_bid and best_ask) else None
            mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
            
            # 创建市场深度对象
            market_depth = MarketDepth(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                mid_price=mid_price
            )
            
            # 保存并通知
            self.market_depths[symbol] = market_depth
            await self._notify_depth_update(market_depth)
            
        except Exception as e:
            logger.error(f"更新市场深度失败: {e}")
    
    # 查询方法
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """获取订单簿"""
        return self.order_books.get(symbol)
    
    def get_market_depth(self, symbol: str, levels: int = 10) -> Optional[MarketDepth]:
        """获取市场深度"""
        depth = self.market_depths.get(symbol)
        if depth and levels < len(depth.bids):
            # 限制返回档数
            limited_depth = MarketDepth(
                symbol=depth.symbol,
                timestamp=depth.timestamp,
                bids=depth.bids[:levels],
                asks=depth.asks[:levels],
                best_bid=depth.best_bid,
                best_ask=depth.best_ask,
                spread=depth.spread,
                mid_price=depth.mid_price
            )
            return limited_depth
        return depth
    
    def get_best_prices(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """获取最优买卖价"""
        depth = self.market_depths.get(symbol)
        if depth:
            return depth.best_bid, depth.best_ask
        return None, None
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近交易记录"""
        if symbol in self.recent_trades:
            trades = list(self.recent_trades[symbol])
            return trades[-limit:] if len(trades) > limit else trades
        return []
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """获取引擎统计"""
        avg_latency = sum(self.match_latency_stats) / len(self.match_latency_stats) if self.match_latency_stats else 0
        
        return {
            'total_matches': self.total_matches,
            'total_volume': self.total_volume,
            'average_match_latency_ms': avg_latency,
            'active_symbols': len(self.order_books),
            'total_active_orders': sum(len(book.bids) + len(book.asks) for book in self.order_books.values()),
            'matching_algorithm': self.matching_algorithm.value
        }
    
    def get_order_book_summary(self, symbol: str) -> Dict[str, Any]:
        """获取订单簿摘要"""
        if symbol not in self.order_books:
            return {}
        
        order_book = self.order_books[symbol]
        depth = self.market_depths.get(symbol)
        
        return {
            'symbol': symbol,
            'bid_orders': len(order_book.bids),
            'ask_orders': len(order_book.asks),
            'best_bid': depth.best_bid if depth else None,
            'best_ask': depth.best_ask if depth else None,
            'spread': depth.spread if depth else None,
            'mid_price': depth.mid_price if depth else None,
            'last_update': order_book.last_update,
            'total_bid_volume': sum(entry.quantity for entry in order_book.bids),
            'total_ask_volume': sum(entry.quantity for entry in order_book.asks)
        }
    
    # 事件通知
    async def _notify_trade_event(self, trade: Trade) -> None:
        """通知交易事件"""
        for handler in self.trade_event_handlers:
            try:
                trade_data = {
                    'trade_id': trade.trade_id,
                    'order_id': trade.order_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'timestamp': trade.timestamp,
                    'strategy_id': trade.strategy_id
                }
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(trade_data)
                else:
                    handler(trade_data)
                    
            except Exception as e:
                logger.error(f"交易事件通知失败: {e}")
    
    async def _notify_depth_update(self, market_depth: MarketDepth) -> None:
        """通知深度更新"""
        for handler in self.depth_update_handlers:
            try:
                depth_data = {
                    'symbol': market_depth.symbol,
                    'timestamp': market_depth.timestamp,
                    'bids': market_depth.bids,
                    'asks': market_depth.asks,
                    'best_bid': market_depth.best_bid,
                    'best_ask': market_depth.best_ask,
                    'spread': market_depth.spread,
                    'mid_price': market_depth.mid_price
                }
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(depth_data)
                else:
                    handler(depth_data)
                    
            except Exception as e:
                logger.error(f"深度更新通知失败: {e}")
    
    # 高级功能
    async def batch_cancel_orders(self, symbol: Optional[str] = None, 
                                 side: Optional[OrderSide] = None) -> int:
        """批量取消订单"""
        try:
            cancelled_count = 0
            orders_to_cancel = []
            
            # 收集要取消的订单
            for order_id, entry in self.order_index.items():
                order = entry.order
                
                # 筛选条件
                if symbol and order.symbol != symbol:
                    continue
                if side and order.side != side:
                    continue
                
                orders_to_cancel.append(order_id)
            
            # 批量取消
            for order_id in orders_to_cancel:
                if await self.cancel_order(order_id):
                    cancelled_count += 1
            
            logger.info(f"批量取消订单完成: {cancelled_count}个")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"批量取消订单失败: {e}")
            return 0
    
    def get_price_levels(self, symbol: str, side: OrderSide, levels: int = 5) -> List[Tuple[float, float]]:
        """获取指定方向的价格档位"""
        try:
            if symbol not in self.order_books:
                return []
            
            order_book = self.order_books[symbol]
            
            if side == OrderSide.BUY:
                entries = order_book.bids[:levels]
            else:
                entries = order_book.asks[:levels]
            
            # 按价格聚合数量
            price_levels = {}
            for entry in entries:
                if entry.price in price_levels:
                    price_levels[entry.price] += entry.quantity
                else:
                    price_levels[entry.price] = entry.quantity
            
            # 排序返回
            if side == OrderSide.BUY:
                return sorted(price_levels.items(), key=lambda x: x[0], reverse=True)
            else:
                return sorted(price_levels.items(), key=lambda x: x[0])
                
        except Exception as e:
            logger.error(f"获取价格档位失败: {e}")
            return []