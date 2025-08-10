"""
市场数据路由模块
处理实时行情、历史数据、技术指标等API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.real_data_manager import real_data_manager

router = APIRouter()

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    change_24h_pct: float
    high_24h: float
    low_24h: float
    timestamp: datetime

class CoinglassAnalysisResponse(BaseModel):
    enabled: bool
    composite_signal: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    funding_rates: Optional[Dict[str, Any]] = None
    open_interest: Optional[Dict[str, Any]] = None
    etf_flows: Optional[Dict[str, Any]] = None

@router.get("/latest")
async def get_latest_market_data(
    symbols: Optional[str] = Query(None, description="逗号分隔的交易对列表，如：BTC/USDT,ETH/USDT")
):
    """获取最新市场数据"""
    try:
        # 获取最新价格数据
        latest_prices = await real_data_manager.get_latest_prices()
        
        # 获取连接状态
        connection_status = real_data_manager.get_connection_status()
        
        # 如果指定了特定交易对，进行过滤
        if symbols:
            requested_symbols = [s.strip() for s in symbols.split(",")]
            filtered_prices = {
                symbol: data for symbol, data in latest_prices.items() 
                if symbol in requested_symbols
            }
            latest_prices = filtered_prices
        
        return {
            "success": True,
            "data": {
                "prices": latest_prices,
                "connections": connection_status,
                "update_time": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场数据失败: {str(e)}")

@router.get("/coinglass")
async def get_coinglass_analysis():
    """获取Coinglass市场情绪分析"""
    try:
        analysis = await real_data_manager.get_coinglass_analysis()
        
        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取Coinglass分析失败: {str(e)}")

@router.get("/health")
async def get_market_data_health():
    """获取市场数据服务健康状态"""
    try:
        health = await real_data_manager.health_check()
        
        return {
            "success": True,
            "data": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取健康状态失败: {str(e)}")

@router.post("/start-stream")
async def start_real_data_stream():
    """启动实时数据流"""
    try:
        if real_data_manager.is_running:
            return {
                "success": True,
                "message": "实时数据流已在运行",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await real_data_manager.start_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流启动成功",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动实时数据流失败: {str(e)}")

@router.post("/stop-stream")
async def stop_real_data_stream():
    """停止实时数据流"""
    try:
        if not real_data_manager.is_running:
            return {
                "success": True,
                "message": "实时数据流未在运行",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await real_data_manager.stop_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流停止成功",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止实时数据流失败: {str(e)}")

@router.get("/stream-status")
async def get_stream_status():
    """获取数据流状态"""
    try:
        connection_status = real_data_manager.get_connection_status()
        
        return {
            "success": True,
            "data": {
                "is_running": real_data_manager.is_running,
                "connections": connection_status,
                "coinglass_enabled": real_data_manager.coinglass_enabled
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/fetch-historical")
async def fetch_historical_data(
    days_back: int = Query(30, ge=1, le=365, description="获取多少天的历史数据")
):
    """获取历史训练数据"""
    try:
        training_data = await real_data_manager.fetch_historical_training_data(days_back)
        
        return {
            "success": True,
            "message": f"成功获取{days_back}天历史数据",
            "data": {
                "days_requested": days_back,
                "data_points": len(training_data) if training_data else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史数据失败: {str(e)}")

@router.post("/migrate-coinglass")
async def migrate_coinglass_data(
    source_path: str = Query(..., description="Coinglass数据源路径")
):
    """迁移Coinglass历史数据"""
    try:
        result = await real_data_manager.migrate_coinglass_historical_data(source_path)
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据迁移失败: {str(e)}")

# WebSocket和技术指标功能扩展
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import numpy as np

class TechnicalIndicatorRequest(BaseModel):
    symbol: str
    indicator: str  # RSI, MACD, BOLLINGER_BANDS, SMA, EMA
    period: int = 14
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    limit: int = 100

class SubscriptionRequest(BaseModel):
    symbols: List[str]
    channels: List[str]  # ticker, trades, orderbook, kline

@router.websocket("/ws/realtime")
async def websocket_realtime_data(websocket: WebSocket):
    """WebSocket实时数据推送"""
    await websocket.accept()
    
    try:
        # WebSocket连接建立
        client_info = {
            "client": websocket.client,
            "connect_time": datetime.utcnow(),
            "subscriptions": set()
        }
        
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "WebSocket连接已建立",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            try:
                # 接收客户端消息
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "subscribe":
                    # 处理订阅请求
                    symbols = message.get("symbols", [])
                    channels = message.get("channels", ["ticker"])
                    
                    for symbol in symbols:
                        for channel in channels:
                            subscription = f"{symbol}:{channel}"
                            client_info["subscriptions"].add(subscription)
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbols": symbols,
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message["type"] == "unsubscribe":
                    # 处理取消订阅
                    symbols = message.get("symbols", [])
                    channels = message.get("channels", [])
                    
                    for symbol in symbols:
                        for channel in channels:
                            subscription = f"{symbol}:{channel}"
                            client_info["subscriptions"].discard(subscription)
                    
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "symbols": symbols,
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # 推送实时数据 (这里应该从real_data_manager获取实际数据)
                if client_info["subscriptions"]:
                    market_data = await real_data_manager.get_latest_prices()
                    
                    for subscription in client_info["subscriptions"]:
                        symbol, channel = subscription.split(":", 1)
                        
                        # 模拟推送对应的数据类型
                        if channel == "ticker" and symbol in market_data:
                            await websocket.send_json({
                                "type": "ticker",
                                "symbol": symbol,
                                "data": market_data[symbol],
                                "timestamp": datetime.utcnow().isoformat()
                            })
                
                await asyncio.sleep(0.1)  # 控制推送频率
                
            except asyncio.TimeoutError:
                # 发送心跳包
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
    finally:
        # 清理连接
        pass

@router.post("/technical-indicators")
async def get_technical_indicators(request: TechnicalIndicatorRequest):
    """计算技术指标"""
    try:
        # 获取历史数据 (这里应该从数据库获取真实的K线数据)
        # 为演示目的，使用模拟数据
        np.random.seed(42)  # 为了结果一致性
        
        # 模拟价格数据
        base_price = 50000 if request.symbol.startswith("BTC") else 3000
        prices = base_price + np.random.randn(request.limit) * base_price * 0.02
        high_prices = prices * (1 + np.random.rand(request.limit) * 0.01)
        low_prices = prices * (1 - np.random.rand(request.limit) * 0.01)
        
        result = {}
        
        if request.indicator.upper() == "RSI":
            # 计算RSI
            rsi_values = calculate_rsi(prices, request.period)
            result = {
                "indicator": "RSI",
                "period": request.period,
                "values": rsi_values.tolist()[-50:],  # 返回最近50个值
                "current": float(rsi_values[-1]),
                "signal": "OVERSOLD" if rsi_values[-1] < 30 else "OVERBOUGHT" if rsi_values[-1] > 70 else "NEUTRAL"
            }
            
        elif request.indicator.upper() == "MACD":
            # 计算MACD
            macd_line, signal_line, histogram = calculate_macd(prices)
            result = {
                "indicator": "MACD",
                "values": {
                    "macd": macd_line.tolist()[-50:],
                    "signal": signal_line.tolist()[-50:],
                    "histogram": histogram.tolist()[-50:]
                },
                "current": {
                    "macd": float(macd_line[-1]),
                    "signal": float(signal_line[-1]),
                    "histogram": float(histogram[-1])
                }
            }
            
        elif request.indicator.upper() == "BOLLINGER_BANDS":
            # 计算布林带
            upper, middle, lower = calculate_bollinger_bands(prices, request.period)
            result = {
                "indicator": "BOLLINGER_BANDS",
                "period": request.period,
                "values": {
                    "upper": upper.tolist()[-50:],
                    "middle": middle.tolist()[-50:],
                    "lower": lower.tolist()[-50:]
                },
                "current": {
                    "upper": float(upper[-1]),
                    "middle": float(middle[-1]),
                    "lower": float(lower[-1]),
                    "price": float(prices[-1])
                }
            }
            
        elif request.indicator.upper() == "SMA":
            # 简单移动平均线
            sma_values = calculate_sma(prices, request.period)
            result = {
                "indicator": "SMA",
                "period": request.period,
                "values": sma_values.tolist()[-50:],
                "current": float(sma_values[-1])
            }
            
        elif request.indicator.upper() == "EMA":
            # 指数移动平均线
            ema_values = calculate_ema(prices, request.period)
            result = {
                "indicator": "EMA",
                "period": request.period,
                "values": ema_values.tolist()[-50:],
                "current": float(ema_values[-1])
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"不支持的技术指标: {request.indicator}")
        
        return {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技术指标计算失败: {str(e)}")

@router.get("/kline/{symbol}")
async def get_kline_data(
    symbol: str,
    timeframe: str = Query("1h", description="时间周期"),
    limit: int = Query(100, description="返回数量", le=1000),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间")
):
    """获取K线数据"""
    try:
        # 这里应该从数据库获取真实的K线数据
        # 为演示目的，生成模拟K线数据
        
        klines = []
        current_time = datetime.utcnow()
        
        # 根据时间周期计算间隔
        interval_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }.get(timeframe, 60)
        
        base_price = 50000 if symbol.upper().startswith("BTC") else 3000
        
        for i in range(limit):
            timestamp = current_time - timedelta(minutes=interval_minutes * (limit - i - 1))
            
            # 生成OHLCV数据
            open_price = base_price + np.random.randn() * base_price * 0.01
            close_price = open_price + np.random.randn() * base_price * 0.005
            high_price = max(open_price, close_price) + abs(np.random.randn()) * base_price * 0.003
            low_price = min(open_price, close_price) - abs(np.random.randn()) * base_price * 0.003
            volume = abs(np.random.randn()) * 100
            
            klines.append({
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 6)
            })
        
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": klines,
            "count": len(klines),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")

@router.get("/market-depth/{symbol}")
async def get_market_depth(
    symbol: str,
    limit: int = Query(20, description="深度档数", le=100)
):
    """获取市场深度数据"""
    try:
        # 生成模拟市场深度数据
        base_price = 50000 if symbol.upper().startswith("BTC") else 3000
        
        bids = []
        asks = []
        
        # 生成买单深度
        for i in range(limit):
            price = base_price * (1 - (i + 1) * 0.0001)
            quantity = abs(np.random.randn()) * 10
            bids.append([round(price, 2), round(quantity, 6)])
        
        # 生成卖单深度
        for i in range(limit):
            price = base_price * (1 + (i + 1) * 0.0001)
            quantity = abs(np.random.randn()) * 10
            asks.append([round(price, 2), round(quantity, 6)])
        
        return {
            "success": True,
            "symbol": symbol,
            "data": {
                "bids": bids,  # [[price, quantity], ...]
                "asks": asks,  # [[price, quantity], ...]
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场深度失败: {str(e)}")

# 技术指标计算函数
def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI指标"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = avg_gain / (avg_loss + 1e-10)  # 避免除零
    rsi = 100 - (100 / (1 + rs))
    
    return rsi[period:]

def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    """计算MACD指标"""
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2):
    """计算布林带"""
    sma = calculate_sma(prices, period)
    
    # 计算标准差
    std = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1])
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return upper, sma, lower

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均线"""
    sma = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    return sma

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均线"""
    ema = np.zeros_like(prices)
    multiplier = 2 / (period + 1)
    
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema