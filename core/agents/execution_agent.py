"""
执行Agent - 智能交易执行和订单管理
负责交易信号执行、订单管理、执行优化和交易监控
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage, MessageType

# 订单状态枚举
class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

# 订单类型枚举
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

# 执行策略枚举
class ExecutionStrategy(Enum):
    IMMEDIATE = "immediate"
    TWAP = "twap"  # 时间加权平均价格
    VWAP = "vwap"  # 成交量加权平均价格
    ICEBERG = "iceberg"  # 冰山订单
    SMART = "smart"  # 智能执行

@dataclass
class Order:
    """订单对象"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # 订单状态
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # 执行参数
    execution_strategy: ExecutionStrategy = ExecutionStrategy.IMMEDIATE
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    
    # 元数据
    client_order_id: Optional[str] = None
    strategy_name: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # 执行统计
    execution_cost: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "execution_strategy": self.execution_strategy.value,
            "time_in_force": self.time_in_force,
            "execution_cost": self.execution_cost,
            "slippage": self.slippage
        }

@dataclass
class Trade:
    """成交记录"""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    commission: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "commission": self.commission
        }

@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    
    total_volume: float = 0.0
    total_commission: float = 0.0
    average_slippage: float = 0.0
    average_execution_time: float = 0.0
    
    fill_rate: float = 0.0
    success_rate: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.utcnow)

class MockExchange:
    """模拟交易所接口"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.market_data: Dict[str, Dict[str, float]] = {}
        self.balance: Dict[str, float] = {"USDT": 100000.0}  # 模拟初始余额
        self.commission_rate = 0.001  # 0.1%手续费
        
    async def submit_order(self, order: Order) -> bool:
        """提交订单"""
        try:
            # 检查余额
            if not self._check_balance(order):
                order.status = OrderStatus.REJECTED
                return False
            
            # 模拟订单提交延迟
            await asyncio.sleep(0.1)
            
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            self.orders[order.order_id] = order
            
            # 启动订单执行模拟
            asyncio.create_task(self._simulate_order_execution(order))
            
            return True
            
        except Exception:
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def update_market_data(self, symbol: str, price: float, volume: float = 1000000):
        """更新市场数据"""
        self.market_data[symbol] = {
            "price": price,
            "bid": price * 0.9995,  # 模拟买价
            "ask": price * 1.0005,  # 模拟卖价
            "volume": volume,
            "timestamp": datetime.utcnow().timestamp()
        }
    
    def _check_balance(self, order: Order) -> bool:
        """检查余额是否足够"""
        if order.side == "buy":
            required = order.quantity * (order.price or self.market_data.get(order.symbol, {}).get("ask", 0))
            return self.balance.get("USDT", 0) >= required
        else:
            return self.balance.get(order.symbol, 0) >= order.quantity
    
    async def _simulate_order_execution(self, order: Order):
        """模拟订单执行"""
        try:
            # 市价单立即执行
            if order.order_type == OrderType.MARKET:
                await asyncio.sleep(0.2)  # 模拟执行延迟
                await self._fill_order(order, order.quantity)
            
            # 限价单等待价格匹配
            elif order.order_type == OrderType.LIMIT:
                await self._wait_for_limit_fill(order)
                
        except Exception as e:
            logging.error(f"模拟订单执行失败: {e}")
    
    async def _fill_order(self, order: Order, fill_quantity: float):
        """执行订单"""
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            return
        
        market_data = self.market_data.get(order.symbol, {})
        if not market_data:
            return
        
        # 计算成交价格（考虑滑点）
        base_price = market_data["ask"] if order.side == "buy" else market_data["bid"]
        slippage = 0.0005 * (1 if order.side == "buy" else -1)  # 0.05%滑点
        fill_price = base_price * (1 + slippage)
        
        # 更新订单状态
        order.filled_quantity += fill_quantity
        order.remaining_quantity -= fill_quantity
        
        # 计算平均成交价格
        if order.filled_quantity > 0:
            order.average_price = ((order.average_price * (order.filled_quantity - fill_quantity)) + 
                                 (fill_price * fill_quantity)) / order.filled_quantity
        
        # 计算手续费
        commission = fill_quantity * fill_price * self.commission_rate
        order.execution_cost += commission
        
        # 计算滑点
        expected_price = order.price or base_price
        order.slippage = (fill_price - expected_price) / expected_price
        
        # 更新余额
        self._update_balance(order, fill_quantity, fill_price, commission)
        
        # 更新订单状态
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
    
    async def _wait_for_limit_fill(self, order: Order):
        """等待限价单成交"""
        timeout = 300  # 5分钟超时
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            if order.status == OrderStatus.CANCELLED:
                break
            
            market_data = self.market_data.get(order.symbol, {})
            if market_data:
                current_price = market_data["price"]
                
                # 检查是否满足成交条件
                if ((order.side == "buy" and current_price <= order.price) or
                    (order.side == "sell" and current_price >= order.price)):
                    await self._fill_order(order, order.remaining_quantity)
                    break
            
            await asyncio.sleep(1)
        
        # 超时处理
        if order.status == OrderStatus.SUBMITTED:
            order.status = OrderStatus.EXPIRED
    
    def _update_balance(self, order: Order, quantity: float, price: float, commission: float):
        """更新账户余额"""
        if order.side == "buy":
            # 买入：减少USDT，增加标的
            cost = quantity * price + commission
            self.balance["USDT"] = self.balance.get("USDT", 0) - cost
            self.balance[order.symbol] = self.balance.get(order.symbol, 0) + quantity
        else:
            # 卖出：减少标的，增加USDT
            proceeds = quantity * price - commission
            self.balance[order.symbol] = self.balance.get(order.symbol, 0) - quantity
            self.balance["USDT"] = self.balance.get("USDT", 0) + proceeds

class ExecutionEngine:
    """执行引擎"""
    
    def __init__(self, exchange: MockExchange):
        self.exchange = exchange
        self.pending_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.metrics = ExecutionMetrics()
        self.logger = logging.getLogger("ExecutionEngine")
        
    async def execute_order(self, order: Order) -> str:
        """执行订单"""
        try:
            self.pending_orders[order.order_id] = order
            
            if order.execution_strategy == ExecutionStrategy.IMMEDIATE:
                success = await self.exchange.submit_order(order)
                if success:
                    self.logger.info(f"订单已提交: {order.order_id}")
                else:
                    self.logger.error(f"订单提交失败: {order.order_id}")
            
            elif order.execution_strategy == ExecutionStrategy.TWAP:
                await self._execute_twap_order(order)
            
            elif order.execution_strategy == ExecutionStrategy.ICEBERG:
                await self._execute_iceberg_order(order)
            
            elif order.execution_strategy == ExecutionStrategy.SMART:
                await self._execute_smart_order(order)
            
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"执行订单失败: {e}")
            return ""
    
    async def _execute_twap_order(self, parent_order: Order):
        """执行TWAP订单"""
        # 将大订单分割为多个小订单，在指定时间内均匀执行
        duration_minutes = 30  # 30分钟内执行
        slice_count = 10  # 分为10个切片
        slice_size = parent_order.quantity / slice_count
        interval = duration_minutes * 60 / slice_count
        
        for i in range(slice_count):
            if parent_order.status == OrderStatus.CANCELLED:
                break
            
            # 创建切片订单
            slice_order = Order(
                symbol=parent_order.symbol,
                side=parent_order.side,
                order_type=OrderType.MARKET,
                quantity=slice_size,
                parent_order_id=parent_order.order_id,
                execution_strategy=ExecutionStrategy.IMMEDIATE
            )
            
            await self.exchange.submit_order(slice_order)
            
            if i < slice_count - 1:  # 最后一个切片不需要等待
                await asyncio.sleep(interval)
    
    async def _execute_iceberg_order(self, parent_order: Order):
        """执行冰山订单"""
        # 只显示部分数量，成交后继续显示
        visible_size = min(parent_order.quantity * 0.1, 1000)  # 10%或1000，取较小值
        remaining = parent_order.quantity
        
        while remaining > 0 and parent_order.status != OrderStatus.CANCELLED:
            current_size = min(visible_size, remaining)
            
            slice_order = Order(
                symbol=parent_order.symbol,
                side=parent_order.side,
                order_type=parent_order.order_type,
                quantity=current_size,
                price=parent_order.price,
                parent_order_id=parent_order.order_id
            )
            
            await self.exchange.submit_order(slice_order)
            
            # 等待切片订单完成
            while slice_order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                await asyncio.sleep(1)
                slice_order = self.exchange.get_order_status(slice_order.order_id)
                if not slice_order:
                    break
            
            if slice_order and slice_order.status == OrderStatus.FILLED:
                remaining -= slice_order.filled_quantity
            else:
                break
    
    async def _execute_smart_order(self, order: Order):
        """智能执行订单"""
        # 根据市场条件选择最佳执行策略
        market_data = self.exchange.market_data.get(order.symbol, {})
        
        if not market_data:
            # 没有市场数据，使用市价单
            order.order_type = OrderType.MARKET
            await self.exchange.submit_order(order)
            return
        
        volume = market_data.get("volume", 0)
        order_impact = order.quantity / volume if volume > 0 else 1
        
        if order_impact < 0.01:  # 小订单，直接市价执行
            order.order_type = OrderType.MARKET
            await self.exchange.submit_order(order)
        
        elif order_impact < 0.05:  # 中等订单，使用限价单
            current_price = market_data["price"]
            if order.side == "buy":
                order.price = current_price * 1.001  # 略高于市价
            else:
                order.price = current_price * 0.999  # 略低于市价
            order.order_type = OrderType.LIMIT
            await self.exchange.submit_order(order)
        
        else:  # 大订单，使用TWAP策略
            order.execution_strategy = ExecutionStrategy.TWAP
            await self._execute_twap_order(order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        success = await self.exchange.cancel_order(order_id)
        if success:
            self.logger.info(f"订单已取消: {order_id}")
        return success
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        # 先查询交易所
        order = self.exchange.get_order_status(order_id)
        if order:
            # 更新本地记录
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                self.pending_orders.pop(order_id, None)
                self.completed_orders[order_id] = order
                self._update_metrics(order)
        
        return order
    
    def _update_metrics(self, order: Order):
        """更新执行指标"""
        self.metrics.total_orders += 1
        
        if order.status == OrderStatus.FILLED:
            self.metrics.filled_orders += 1
            self.metrics.total_volume += order.filled_quantity
            self.metrics.total_commission += order.execution_cost
        elif order.status == OrderStatus.CANCELLED:
            self.metrics.cancelled_orders += 1
        elif order.status == OrderStatus.REJECTED:
            self.metrics.rejected_orders += 1
        
        # 计算比率
        if self.metrics.total_orders > 0:
            self.metrics.fill_rate = self.metrics.filled_orders / self.metrics.total_orders
            self.metrics.success_rate = (self.metrics.filled_orders + self.metrics.cancelled_orders) / self.metrics.total_orders
        
        self.metrics.last_updated = datetime.utcnow()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        return {
            "total_orders": self.metrics.total_orders,
            "filled_orders": self.metrics.filled_orders,
            "cancelled_orders": self.metrics.cancelled_orders,
            "rejected_orders": self.metrics.rejected_orders,
            "total_volume": self.metrics.total_volume,
            "total_commission": self.metrics.total_commission,
            "fill_rate": self.metrics.fill_rate,
            "success_rate": self.metrics.success_rate,
            "last_updated": self.metrics.last_updated.isoformat()
        }

class ExecutionAgent(BaseAgent):
    """执行Agent - 智能交易执行和订单管理"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="execution_agent",
            agent_type=AgentType.EXECUTION,
            capabilities=[
                AgentCapability.ORDER_EXECUTION,
                AgentCapability.MARKET_MONITORING
            ],
            config=config or {}
        )
        
        # 执行引擎
        self.exchange = MockExchange()
        self.execution_engine = ExecutionEngine(self.exchange)
        
        # 订单管理
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.max_history = config.get("max_order_history", 1000)
        
        # 执行配置
        self.default_execution_strategy = ExecutionStrategy(
            config.get("default_execution_strategy", ExecutionStrategy.SMART.value)
        )
        self.risk_check_enabled = config.get("risk_check_enabled", True)
        
        # 监控配置
        self.order_monitoring_interval = config.get("order_monitoring_interval", 1)  # 秒
        
    async def _initialize(self):
        """初始化执行Agent"""
        # 启动订单监控循环
        asyncio.create_task(self._order_monitoring_loop())
        
        # 启动市场数据更新循环
        asyncio.create_task(self._market_data_update_loop())
        
        self.logger.info("⚡ 执行Agent初始化完成")
    
    async def _handle_command(self, message: AgentMessage):
        """处理命令消息"""
        command = message.content.get("command")
        params = message.content.get("params", {})
        
        if command == "submit_order":
            await self._handle_submit_order(message, params)
        elif command == "cancel_order":
            await self._handle_cancel_order(message, params)
        elif command == "execute_signal":
            await self._handle_execute_signal(message, params)
        elif command == "update_market_data":
            await self._handle_update_market_data(message, params)
        elif command == "modify_order":
            await self._handle_modify_order(message, params)
        else:
            await self.send_response(message, {"error": f"未知命令: {command}"})
    
    async def _handle_query(self, message: AgentMessage):
        """处理查询消息"""
        query = message.content.get("query")
        params = message.content.get("params", {})
        
        if query == "get_order_status":
            response = await self._get_order_status(params)
        elif query == "get_active_orders":
            response = await self._get_active_orders()
        elif query == "get_execution_metrics":
            response = await self._get_execution_metrics()
        elif query == "get_order_history":
            response = await self._get_order_history(params)
        elif query == "get_balance":
            response = await self._get_balance()
        else:
            response = {"error": f"未知查询: {query}"}
        
        await self.send_response(message, response)
    
    async def _handle_submit_order(self, message: AgentMessage, params: Dict[str, Any]):
        """处理订单提交"""
        try:
            # 创建订单
            order = Order(
                symbol=params.get("symbol", ""),
                side=params.get("side", ""),
                order_type=OrderType(params.get("order_type", OrderType.MARKET.value)),
                quantity=params.get("quantity", 0.0),
                price=params.get("price"),
                stop_price=params.get("stop_price"),
                execution_strategy=ExecutionStrategy(
                    params.get("execution_strategy", self.default_execution_strategy.value)
                ),
                time_in_force=params.get("time_in_force", "GTC"),
                client_order_id=params.get("client_order_id"),
                strategy_name=params.get("strategy_name")
            )
            
            # 风险检查
            if self.risk_check_enabled:
                risk_check = await self._perform_risk_check(order)
                if not risk_check["approved"]:
                    await self.send_response(message, {
                        "success": False,
                        "error": "风险检查未通过",
                        "risk_issues": risk_check.get("blocks", [])
                    })
                    return
            
            # 执行订单
            order_id = await self.execution_engine.execute_order(order)
            
            if order_id:
                self.active_orders[order_id] = order
                
                await self.send_response(message, {
                    "success": True,
                    "order_id": order_id,
                    "status": order.status.value
                })
                
                self.logger.info(f"📝 提交订单: {order.symbol} {order.side} {order.quantity}")
            else:
                await self.send_response(message, {
                    "success": False,
                    "error": "订单执行失败"
                })
                
        except Exception as e:
            await self.send_response(message, {
                "success": False,
                "error": str(e)
            })
    
    async def _handle_execute_signal(self, message: AgentMessage, params: Dict[str, Any]):
        """处理交易信号执行"""
        try:
            signal_data = params.get("signal", {})
            
            # 从交易信号创建订单
            order = Order(
                symbol=signal_data.get("symbol", ""),
                side=signal_data.get("action", ""),
                order_type=OrderType.MARKET,  # 信号默认使用市价单
                quantity=self._calculate_position_size(signal_data),
                execution_strategy=self.default_execution_strategy,
                strategy_name=signal_data.get("strategy_name")
            )
            
            # 根据信号置信度调整执行策略
            confidence = signal_data.get("confidence", 0.5)
            if confidence >= 0.9:
                order.execution_strategy = ExecutionStrategy.IMMEDIATE
            elif confidence >= 0.7:
                order.execution_strategy = ExecutionStrategy.SMART
            else:
                order.execution_strategy = ExecutionStrategy.TWAP
            
            # 风险检查
            if self.risk_check_enabled:
                risk_check = await self._perform_risk_check(order)
                if not risk_check["approved"]:
                    await self.send_response(message, {
                        "success": False,
                        "signal_executed": False,
                        "error": "信号执行被风险控制阻止",
                        "risk_issues": risk_check.get("blocks", [])
                    })
                    return
            
            # 执行订单
            order_id = await self.execution_engine.execute_order(order)
            
            if order_id:
                self.active_orders[order_id] = order
                
                await self.send_response(message, {
                    "success": True,
                    "signal_executed": True,
                    "order_id": order_id,
                    "order_details": order.to_dict()
                })
                
                self.logger.info(f"🎯 执行交易信号: {order.symbol} {order.side} {order.quantity} (置信度: {confidence})")
            else:
                await self.send_response(message, {
                    "success": False,
                    "signal_executed": False,
                    "error": "信号执行失败"
                })
                
        except Exception as e:
            await self.send_response(message, {
                "success": False,
                "signal_executed": False,
                "error": str(e)
            })
    
    async def _handle_cancel_order(self, message: AgentMessage, params: Dict[str, Any]):
        """处理订单取消"""
        try:
            order_id = params.get("order_id")
            
            if not order_id:
                await self.send_response(message, {
                    "success": False,
                    "error": "缺少order_id参数"
                })
                return
            
            success = await self.execution_engine.cancel_order(order_id)
            
            await self.send_response(message, {
                "success": success,
                "order_id": order_id,
                "message": "订单已取消" if success else "订单取消失败"
            })
            
        except Exception as e:
            await self.send_response(message, {
                "success": False,
                "error": str(e)
            })
    
    async def _handle_update_market_data(self, message: AgentMessage, params: Dict[str, Any]):
        """处理市场数据更新"""
        try:
            symbol = params.get("symbol")
            price = params.get("price")
            volume = params.get("volume", 1000000)
            
            if symbol and price:
                self.exchange.update_market_data(symbol, price, volume)
            
            await self.send_response(message, {"status": "updated"})
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _perform_risk_check(self, order: Order) -> Dict[str, Any]:
        """执行风险检查"""
        try:
            # 向风险Agent请求验证
            risk_query = AgentMessage(
                receiver_id="risk_agent",
                message_type=MessageType.QUERY,
                content={
                    "query": "validate_trade",
                    "params": {
                        "trade": {
                            "symbol": order.symbol,
                            "size": order.quantity,
                            "action": order.side
                        }
                    }
                }
            )
            
            # 等待风险检查结果
            response = await self.send_query(
                "risk_agent",
                "validate_trade",
                {
                    "trade": {
                        "symbol": order.symbol,
                        "size": order.quantity,
                        "action": order.side
                    }
                },
                timeout=10
            )
            
            if response:
                return response
            else:
                # 风险Agent未响应，使用默认检查
                return self._default_risk_check(order)
                
        except Exception as e:
            self.logger.error(f"风险检查失败: {e}")
            return self._default_risk_check(order)
    
    def _default_risk_check(self, order: Order) -> Dict[str, Any]:
        """默认风险检查"""
        # 简单的本地风险检查
        warnings = []
        blocks = []
        
        # 检查订单大小
        if order.quantity <= 0:
            blocks.append("订单数量必须大于0")
        
        # 检查余额
        if order.side == "buy":
            market_data = self.exchange.market_data.get(order.symbol, {})
            required_balance = order.quantity * market_data.get("ask", 0)
            current_balance = self.exchange.balance.get("USDT", 0)
            
            if required_balance > current_balance:
                blocks.append(f"余额不足: 需要{required_balance:.2f} USDT，当前{current_balance:.2f} USDT")
        
        return {
            "approved": len(blocks) == 0,
            "risk_level": "MODERATE",
            "warnings": warnings,
            "blocks": blocks
        }
    
    def _calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        """根据信号计算仓位大小"""
        # 简化的仓位计算逻辑
        confidence = signal_data.get("confidence", 0.5)
        base_size = 1000  # 基础仓位大小
        
        # 根据置信度调整仓位
        position_size = base_size * confidence
        
        # 根据账户余额调整
        balance = self.exchange.balance.get("USDT", 0)
        max_position = balance * 0.1  # 最大10%仓位
        
        return min(position_size, max_position)
    
    async def _order_monitoring_loop(self):
        """订单监控循环"""
        self.logger.info("🔄 启动订单监控循环")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.order_monitoring_interval)
                
                # 检查活跃订单状态
                completed_orders = []
                
                for order_id in list(self.active_orders.keys()):
                    order = self.execution_engine.get_order_status(order_id)
                    
                    if order and order.status in [
                        OrderStatus.FILLED, OrderStatus.CANCELLED, 
                        OrderStatus.REJECTED, OrderStatus.EXPIRED
                    ]:
                        completed_orders.append(order_id)
                        
                        # 通知订单状态变化
                        await self._notify_order_update(order)
                
                # 移除已完成的订单
                for order_id in completed_orders:
                    order = self.active_orders.pop(order_id, None)
                    if order:
                        self.order_history.append(order)
                        
                        # 限制历史记录长度
                        if len(self.order_history) > self.max_history:
                            self.order_history = self.order_history[-self.max_history//2:]
                
            except Exception as e:
                self.logger.error(f"订单监控循环错误: {e}")
    
    async def _market_data_update_loop(self):
        """市场数据更新循环（模拟）"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # 每5秒更新一次模拟数据
                
                # 模拟价格变动
                import random
                for symbol in ["BTC/USDT", "ETH/USDT", "BNB/USDT"]:
                    current_data = self.exchange.market_data.get(symbol, {"price": 50000})
                    current_price = current_data["price"]
                    
                    # 随机价格变动 (-0.5% 到 +0.5%)
                    price_change = random.uniform(-0.005, 0.005)
                    new_price = current_price * (1 + price_change)
                    
                    self.exchange.update_market_data(symbol, new_price)
                
            except Exception as e:
                self.logger.error(f"市场数据更新错误: {e}")
    
    async def _notify_order_update(self, order: Order):
        """通知订单状态更新"""
        try:
            notification = AgentMessage(
                receiver_id="*",  # 广播
                message_type=MessageType.EVENT,
                priority=6,
                content={
                    "event_type": "order_update",
                    "order": order.to_dict()
                }
            )
            
            await self.send_message(notification)
            
            self.logger.info(f"📋 订单状态更新: {order.order_id} -> {order.status.value}")
            
        except Exception as e:
            self.logger.error(f"通知订单更新失败: {e}")
    
    async def _get_order_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取订单状态"""
        order_id = params.get("order_id")
        
        if order_id:
            order = self.execution_engine.get_order_status(order_id)
            if order:
                return {
                    "order_id": order_id,
                    "order": order.to_dict()
                }
            else:
                return {"error": f"订单不存在: {order_id}"}
        else:
            return {"error": "缺少order_id参数"}
    
    async def _get_active_orders(self) -> Dict[str, Any]:
        """获取活跃订单"""
        active_orders = []
        
        for order in self.active_orders.values():
            # 获取最新状态
            updated_order = self.execution_engine.get_order_status(order.order_id)
            if updated_order:
                active_orders.append(updated_order.to_dict())
            else:
                active_orders.append(order.to_dict())
        
        return {
            "active_orders": active_orders,
            "count": len(active_orders)
        }
    
    async def _get_execution_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        return self.execution_engine.get_metrics()
    
    async def _get_order_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取订单历史"""
        limit = params.get("limit", 50)
        
        recent_history = self.order_history[-limit:]
        
        return {
            "order_history": [order.to_dict() for order in recent_history],
            "count": len(recent_history),
            "total_history_count": len(self.order_history)
        }
    
    async def _get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        return {
            "balance": dict(self.exchange.balance),
            "timestamp": datetime.utcnow().isoformat()
        }