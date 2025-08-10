"""
交易路由模块
处理交易执行、历史记录、持仓管理等API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from decimal import Decimal

router = APIRouter()

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class CreateOrderRequest(BaseModel):
    symbol: str = Field(..., description="交易对，如BTC/USDT")
    side: OrderSide = Field(..., description="买卖方向")
    order_type: OrderType = Field(..., description="订单类型")
    amount: float = Field(..., gt=0, description="交易数量")
    price: Optional[float] = Field(None, description="限价单价格")
    stop_price: Optional[float] = Field(None, description="止损价格")

class Order(BaseModel):
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float]
    stop_price: Optional[float]
    filled_amount: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime
    updated_at: datetime

class Position(BaseModel):
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str
    created_at: datetime
    updated_at: datetime

class BatchOrderRequest(BaseModel):
    """批量订单请求"""
    orders: List[CreateOrderRequest] = Field(..., description="订单列表")
    execution_mode: str = Field("parallel", description="执行模式: parallel(并行), sequential(串行)")
    
    @validator('orders')
    def validate_orders_count(cls, v):
        if len(v) > 50:
            raise ValueError("单次最多提交50个订单")
        if len(v) == 0:
            raise ValueError("订单列表不能为空")
        return v

class RiskCheckRequest(BaseModel):
    """风控检查请求"""
    symbol: str
    side: OrderSide
    amount: float
    price: Optional[float] = None
    check_types: List[str] = Field(["position_limit", "exposure", "balance"], description="检查项目")

class RiskLimitConfig(BaseModel):
    """风控限制配置"""
    max_position_size: float = Field(100000.0, description="最大持仓金额")
    max_daily_loss: float = Field(5000.0, description="最大日损失")
    max_order_value: float = Field(10000.0, description="单笔订单最大价值")
    max_leverage: float = Field(10.0, description="最大杠杆倍数")
    blacklist_symbols: List[str] = Field([], description="禁止交易品种")

class BatchOperationResult(BaseModel):
    """批量操作结果"""
    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    execution_time: float

# 模拟订单数据库
orders_db: Dict[str, Order] = {}
positions_db: Dict[str, Position] = {}
trade_counter = 1

# 风控配置
risk_limits = RiskLimitConfig()

# 当日交易统计
daily_stats = {
    "total_loss": 0.0,
    "total_volume": 0.0,
    "trade_count": 0,
    "last_reset": datetime.utcnow().date()
}

def generate_order_id() -> str:
    """生成订单ID"""
    global trade_counter
    order_id = f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{trade_counter:04d}"
    trade_counter += 1
    return order_id

def reset_daily_stats_if_needed():
    """如果需要，重置当日统计"""
    global daily_stats
    current_date = datetime.utcnow().date()
    if daily_stats["last_reset"] != current_date:
        daily_stats = {
            "total_loss": 0.0,
            "total_volume": 0.0,
            "trade_count": 0,
            "last_reset": current_date
        }

async def risk_check(request: RiskCheckRequest) -> Dict[str, Any]:
    """执行风控检查"""
    reset_daily_stats_if_needed()
    
    risks = []
    warnings = []
    
    # 检查品种黑名单
    if request.symbol in risk_limits.blacklist_symbols:
        risks.append(f"品种 {request.symbol} 在交易黑名单中")
    
    # 检查订单价值
    order_value = request.amount * (request.price or 50000)  # 使用模拟价格
    if order_value > risk_limits.max_order_value:
        risks.append(f"订单价值 ${order_value:.2f} 超过限制 ${risk_limits.max_order_value:.2f}")
    
    # 检查持仓限制
    if "position_limit" in request.check_types:
        current_position = positions_db.get(request.symbol)
        if current_position:
            new_position_value = (current_position.size + request.amount) * (request.price or current_position.current_price)
            if new_position_value > risk_limits.max_position_size:
                risks.append(f"持仓价值将超过限制 ${risk_limits.max_position_size:.2f}")
    
    # 检查当日损失
    if "exposure" in request.check_types:
        if daily_stats["total_loss"] > risk_limits.max_daily_loss:
            risks.append(f"当日损失 ${daily_stats['total_loss']:.2f} 已超过限制 ${risk_limits.max_daily_loss:.2f}")
    
    # 检查余额（简化）
    if "balance" in request.check_types:
        if order_value > 100000:  # 模拟余额检查
            warnings.append("订单价值较大，请确认账户余额充足")
    
    return {
        "passed": len(risks) == 0,
        "risks": risks,
        "warnings": warnings,
        "order_value": order_value,
        "risk_score": len(risks) * 20 + len(warnings) * 5
    }

@router.post("/orders", response_model=Order)
async def create_order(order_request: CreateOrderRequest):
    """创建新订单"""
    try:
        # 生成订单ID
        order_id = generate_order_id()
        
        # 验证订单参数
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order_request.price is None:
            raise HTTPException(status_code=400, detail="限价单必须指定价格")
        
        if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order_request.stop_price is None:
            raise HTTPException(status_code=400, detail="止损单必须指定止损价格")
        
        # 创建订单
        order = Order(
            id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            amount=order_request.amount,
            price=order_request.price,
            stop_price=order_request.stop_price,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # 保存到数据库
        orders_db[order_id] = order
        
        # 模拟订单执行（在实际环境中这里会连接到交易所）
        if order_request.order_type == OrderType.MARKET:
            # 市价单立即成交
            order.status = OrderStatus.FILLED
            order.filled_amount = order.amount
            order.updated_at = datetime.utcnow()
        
        return order
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建订单失败: {str(e)}")

@router.get("/orders", response_model=List[Order])
async def get_orders(
    symbol: Optional[str] = Query(None, description="按交易对过滤"),
    status: Optional[OrderStatus] = Query(None, description="按状态过滤"),
    limit: int = Query(50, ge=1, le=100, description="返回数量限制")
):
    """获取订单列表"""
    try:
        orders = list(orders_db.values())
        
        # 按条件过滤
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        # 按创建时间倒序排列
        orders.sort(key=lambda x: x.created_at, reverse=True)
        
        # 限制数量
        orders = orders[:limit]
        
        return orders
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取订单失败: {str(e)}")

@router.get("/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: str,
):
    """获取特定订单详情"""
    if order_id not in orders_db:
        raise HTTPException(status_code=404, detail="订单不存在")
    
    return orders_db[order_id]

@router.delete("/orders/{order_id}")
async def cancel_order(
    order_id: str,
):
    """取消订单"""
    try:
        if order_id not in orders_db:
            raise HTTPException(status_code=404, detail="订单不存在")
        
        order = orders_db[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="订单已完成或已取消，无法取消")
        
        # 更新订单状态
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        
        return {
            "success": True,
            "message": f"订单 {order_id} 已取消",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取消订单失败: {str(e)}")

@router.get("/positions", response_model=List[Position])
async def get_positions(
    symbol: Optional[str] = Query(None, description="按交易对过滤"),
):
    """获取持仓列表"""
    try:
        positions = list(positions_db.values())
        
        # 按条件过滤
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        
        # 更新未实现盈亏（这里应该根据最新价格计算）
        for position in positions:
            # 模拟价格更新（实际应从市场数据获取）
            position.current_price = position.entry_price * (1 + 0.02)  # 模拟2%涨幅
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
            position.unrealized_pnl_pct = (position.current_price - position.entry_price) / position.entry_price * 100
            position.updated_at = datetime.utcnow()
        
        return positions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取持仓失败: {str(e)}")

@router.get("/history")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="按交易对过滤"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    limit: int = Query(50, ge=1, le=500, description="返回数量限制"),
):
    """获取交易历史"""
    try:
        # 获取已成交的订单
        trades = [o for o in orders_db.values() if o.status == OrderStatus.FILLED]
        
        # 按条件过滤
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_date:
            trades = [t for t in trades if t.created_at >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.created_at <= end_date]
        
        # 按时间倒序排列
        trades.sort(key=lambda x: x.created_at, reverse=True)
        
        # 限制数量
        trades = trades[:limit]
        
        # 转换为交易历史格式
        trade_history = []
        for trade in trades:
            trade_history.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "amount": trade.filled_amount,
                "price": trade.price,
                "value": trade.filled_amount * (trade.price or 0),
                "timestamp": trade.updated_at.isoformat()
            })
        
        return {
            "success": True,
            "data": trade_history,
            "total": len(trade_history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取交易历史失败: {str(e)}")

@router.get("/stats")
async def get_trading_stats(
    period: str = Query("7d", description="统计周期：1d, 7d, 30d"),
):
    """获取交易统计"""
    try:
        # 计算统计周期
        period_map = {"1d": 1, "7d": 7, "30d": 30}
        days = period_map.get(period, 7)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # 获取周期内的交易
        trades = [
            o for o in orders_db.values() 
            if o.status == OrderStatus.FILLED and o.created_at >= start_date
        ]
        
        # 计算统计数据
        total_trades = len(trades)
        total_volume = sum(t.filled_amount * (t.price or 0) for t in trades)
        buy_trades = len([t for t in trades if t.side == OrderSide.BUY])
        sell_trades = len([t for t in trades if t.side == OrderSide.SELL])
        
        # 模拟盈亏计算
        total_pnl = sum(p.unrealized_pnl for p in positions_db.values())
        
        return {
            "success": True,
            "data": {
                "period": period,
                "total_trades": total_trades,
                "total_volume": total_volume,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "win_rate": 0.65,  # 模拟胜率
                "total_pnl": total_pnl,
                "active_positions": len(positions_db)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计数据失败: {str(e)}")

# ============ 批量操作和风控API ============

@router.post("/risk-check")
async def check_trading_risk(
    risk_request: RiskCheckRequest,
):
    """风控检查"""
    try:
        result = await risk_check(risk_request, current_user)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"风控检查失败: {str(e)}")

@router.post("/orders/batch", response_model=BatchOperationResult)
async def create_batch_orders(
    batch_request: BatchOrderRequest,
    background_tasks: BackgroundTasks,
):
    """批量创建订单"""
    try:
        start_time = datetime.utcnow()
        results = []
        successful = 0
        failed = 0
        
        if batch_request.execution_mode == "parallel":
            # 并行执行
            tasks = []
            for order_req in batch_request.orders:
                tasks.append(create_single_order_async(order_req))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "index": i,
                        "success": False,
                        "error": str(result),
                        "order_request": batch_request.orders[i].dict()
                    })
                    failed += 1
                else:
                    results.append({
                        "index": i,
                        "success": True,
                        "order": result.dict()
                    })
                    successful += 1
        
        else:  # sequential
            # 串行执行
            for i, order_req in enumerate(batch_request.orders):
                try:
                    order = await create_single_order_async(order_req)
                    results.append({
                        "index": i,
                        "success": True,
                        "order": order.dict()
                    })
                    successful += 1
                except Exception as e:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": str(e),
                        "order_request": order_req.dict()
                    })
                    failed += 1
                    
                    # 如果是串行模式且某个订单失败，可选择是否继续
                    if "stop_on_error" in batch_request.dict().get("options", {}):
                        break
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return BatchOperationResult(
            total=len(batch_request.orders),
            successful=successful,
            failed=failed,
            results=results,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量创建订单失败: {str(e)}")

async def create_single_order_async(order_request: CreateOrderRequest) -> Order:
    """异步创建单个订单（用于批量操作）"""
    # 风控检查
    risk_request = RiskCheckRequest(
        symbol=order_request.symbol,
        side=order_request.side,
        amount=order_request.amount,
        price=order_request.price
    )
    
    risk_result = await risk_check(risk_request)
    if not risk_result["passed"]:
        raise HTTPException(status_code=400, detail=f"风控检查失败: {'; '.join(risk_result['risks'])}")
    
    # 生成订单ID
    order_id = generate_order_id()
    
    # 验证订单参数
    if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order_request.price is None:
        raise ValueError("限价单必须指定价格")
    
    if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order_request.stop_price is None:
        raise ValueError("止损单必须指定止损价格")
    
    # 创建订单
    order = Order(
        id=order_id,
        symbol=order_request.symbol,
        side=order_request.side,
        order_type=order_request.order_type,
        amount=order_request.amount,
        price=order_request.price,
        stop_price=order_request.stop_price,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # 保存到数据库
    orders_db[order_id] = order
    
    # 模拟订单执行
    if order_request.order_type == OrderType.MARKET:
        order.status = OrderStatus.FILLED
        order.filled_amount = order.amount
        order.updated_at = datetime.utcnow()
        
        # 更新统计
        daily_stats["trade_count"] += 1
        daily_stats["total_volume"] += order.amount * (order.price or 50000)
    
    return order

@router.delete("/orders/batch")
async def cancel_batch_orders(order_ids: List[str]):
    """批量取消订单"""
    try:
        start_time = datetime.utcnow()
        results = []
        successful = 0
        failed = 0
        
        for order_id in order_ids:
            try:
                if order_id not in orders_db:
                    results.append({
                        "order_id": order_id,
                        "success": False,
                        "error": "订单不存在"
                    })
                    failed += 1
                    continue
                
                order = orders_db[order_id]
                
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                    results.append({
                        "order_id": order_id,
                        "success": False,
                        "error": "订单已完成或已取消"
                    })
                    failed += 1
                    continue
                
                # 取消订单
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.utcnow()
                
                results.append({
                    "order_id": order_id,
                    "success": True,
                    "message": "订单已取消"
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    "order_id": order_id,
                    "success": False,
                    "error": str(e)
                })
                failed += 1
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "success": True,
            "data": BatchOperationResult(
                total=len(order_ids),
                successful=successful,
                failed=failed,
                results=results,
                execution_time=execution_time
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量取消订单失败: {str(e)}")

@router.get("/risk-limits")
async def get_risk_limits(current_user: str = Depends(verify_token)):
    """获取风控限制配置"""
    return {
        "success": True,
        "data": risk_limits.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.put("/risk-limits")
async def update_risk_limits(new_limits: RiskLimitConfig):
    """更新风控限制配置"""
    try:
        global risk_limits
        risk_limits = new_limits
        
        return {
            "success": True,
            "message": "风控限制已更新",
            "data": risk_limits.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新风控限制失败: {str(e)}")

@router.get("/daily-stats")
async def get_daily_trading_stats(current_user: str = Depends(verify_token)):
    """获取当日交易统计"""
    reset_daily_stats_if_needed()
    
    return {
        "success": True,
        "data": {
            **daily_stats,
            "last_reset": daily_stats["last_reset"].isoformat(),
            "risk_utilization": {
                "loss_usage_pct": (daily_stats["total_loss"] / risk_limits.max_daily_loss) * 100,
                "remaining_loss_limit": max(0, risk_limits.max_daily_loss - daily_stats["total_loss"])
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/positions/close-all")
async def close_all_positions(
    symbol: Optional[str] = Query(None, description="指定品种，为空时关闭所有持仓"),
):
    """平仓所有持仓"""
    try:
        start_time = datetime.utcnow()
        results = []
        successful = 0
        failed = 0
        
        positions_to_close = []
        if symbol:
            if symbol in positions_db:
                positions_to_close.append(positions_db[symbol])
        else:
            positions_to_close = list(positions_db.values())
        
        for position in positions_to_close:
            try:
                # 创建平仓订单
                close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
                close_order_request = CreateOrderRequest(
                    symbol=position.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    amount=abs(position.size)
                )
                
                order = await create_single_order_async(close_order_request)
                
                # 移除持仓
                del positions_db[position.symbol]
                
                results.append({
                    "symbol": position.symbol,
                    "success": True,
                    "close_order_id": order.id,
                    "closed_size": position.size
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    "symbol": position.symbol,
                    "success": False,
                    "error": str(e)
                })
                failed += 1
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "success": True,
            "data": {
                "total_positions": len(positions_to_close),
                "successful_closures": successful,
                "failed_closures": failed,
                "results": results,
                "execution_time": execution_time
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"平仓操作失败: {str(e)}")

@router.get("/orders/summary")
async def get_orders_summary(
    timeframe: str = Query("1d", description="时间范围: 1h, 1d, 7d, 30d"),
):
    """获取订单摘要统计"""
    try:
        # 计算时间范围
        timeframe_map = {"1h": 1/24, "1d": 1, "7d": 7, "30d": 30}
        days = timeframe_map.get(timeframe, 1)
        start_time = datetime.utcnow() - timedelta(days=days)
        
        # 过滤订单
        filtered_orders = [
            order for order in orders_db.values()
            if order.created_at >= start_time
        ]
        
        # 统计
        total_orders = len(filtered_orders)
        status_counts = {}
        for status in OrderStatus:
            status_counts[status.value] = len([o for o in filtered_orders if o.status == status])
        
        type_counts = {}
        for order_type in OrderType:
            type_counts[order_type.value] = len([o for o in filtered_orders if o.order_type == order_type])
        
        side_counts = {
            "buy": len([o for o in filtered_orders if o.side == OrderSide.BUY]),
            "sell": len([o for o in filtered_orders if o.side == OrderSide.SELL])
        }
        
        avg_order_value = sum(
            o.amount * (o.price or 50000) for o in filtered_orders if o.price
        ) / max(1, len([o for o in filtered_orders if o.price]))
        
        return {
            "success": True,
            "data": {
                "timeframe": timeframe,
                "total_orders": total_orders,
                "status_breakdown": status_counts,
                "type_breakdown": type_counts,
                "side_breakdown": side_counts,
                "average_order_value": avg_order_value,
                "fill_rate": (status_counts.get("filled", 0) / max(1, total_orders)) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取订单摘要失败: {str(e)}")