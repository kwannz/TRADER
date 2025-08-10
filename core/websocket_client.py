"""
WebSocket客户端管理器
支持OKX、Binance实时数据流和自动重连
"""

import asyncio
import json
import hmac
import hashlib
import base64
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import websockets
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

# 简化设置管理
class Settings:
    def get_api_config(self, exchange: str) -> dict:
        """获取交易所API配置"""
        if exchange.lower() == "okx":
            return {
                "api_key": os.getenv("OKX_API_KEY", ""),
                "secret_key": os.getenv("OKX_SECRET_KEY", ""),
                "passphrase": os.getenv("OKX_PASSPHRASE", ""),
                "sandbox": os.getenv("OKX_SANDBOX", "true").lower() == "true"
            }
        elif exchange.lower() == "binance":
            return {
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "secret_key": os.getenv("BINANCE_SECRET_KEY", ""),
                "sandbox": os.getenv("BINANCE_SANDBOX", "true").lower() == "true"
            }
        return {}

settings = Settings()
from .data_manager import data_manager

class BaseWebSocketClient:
    """WebSocket客户端基类"""
    
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        self.subscriptions: List[Dict] = []
        self.callbacks: Dict[str, Callable] = {}
        self._running = False
        self._reconnect_task = None
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"{self.name} WebSocket连接成功")
            
            # 重新订阅
            if self.subscriptions:
                await self._resubscribe()
                
        except Exception as e:
            logger.error(f"{self.name} WebSocket连接失败: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """断开WebSocket连接"""
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self.is_connected = False
        logger.info(f"{self.name} WebSocket已断开")
    
    async def _resubscribe(self):
        """重新订阅所有频道"""
        for subscription in self.subscriptions:
            await self.send(subscription)
            await asyncio.sleep(0.1)  # 避免频率限制
    
    async def send(self, message: Dict):
        """发送消息"""
        if not self.is_connected or not self.websocket:
            logger.warning(f"{self.name} WebSocket未连接，无法发送消息")
            return
            
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"{self.name} 发送消息: {message}")
        except Exception as e:
            logger.error(f"{self.name} 发送消息失败: {e}")
    
    async def listen(self):
        """监听WebSocket消息"""
        self._running = True
        while self._running:
            try:
                if not self.is_connected:
                    await self._handle_reconnect()
                    continue
                    
                # 监听消息
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=30
                )
                
                # 处理消息
                await self._handle_message(message)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"{self.name} WebSocket连接断开")
                self.is_connected = False
            except asyncio.TimeoutError:
                logger.warning(f"{self.name} WebSocket接收超时")
                self.is_connected = False
            except Exception as e:
                logger.error(f"{self.name} WebSocket监听错误: {e}")
                self.is_connected = False
                
            # 短暂延迟避免过度重连
            await asyncio.sleep(0.1)
    
    async def _handle_reconnect(self):
        """处理重连逻辑"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"{self.name} 超过最大重连次数，停止重连")
            self._running = False
            return
            
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * self.reconnect_attempts, 60)
        
        logger.info(f"{self.name} 尝试重连 ({self.reconnect_attempts}/{self.max_reconnect_attempts})，{delay}秒后重试")
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"{self.name} 重连失败: {e}")
    
    async def _handle_message(self, message: str):
        """处理接收到的消息 - 子类实现"""
        pass
    
    def add_callback(self, event: str, callback: Callable):
        """添加事件回调"""
        self.callbacks[event] = callback

class OKXWebSocketClient(BaseWebSocketClient):
    """OKX WebSocket客户端"""
    
    def __init__(self):
        super().__init__(
            "OKX", 
            "wss://ws.okx.com:8443/ws/v5/public"
        )
        self.api_config = settings.get_api_config("okx")
        
    async def subscribe_tickers(self, symbols: List[str]):
        """订阅行情数据"""
        args = []
        for symbol in symbols:
            args.append({
                "channel": "tickers",
                "instId": symbol
            })
            
        message = {
            "op": "subscribe",
            "args": args
        }
        
        self.subscriptions.append(message)
        if self.is_connected:
            await self.send(message)
    
    async def subscribe_candles(self, symbols: List[str], timeframe: str = "1m"):
        """订阅K线数据"""
        args = []
        for symbol in symbols:
            # OKX K线频道名称格式修正
            candle_channel = f"candle{timeframe}"
            if timeframe == "1m":
                candle_channel = "candle1m"
            elif timeframe == "5m":
                candle_channel = "candle5m"
            elif timeframe == "15m":
                candle_channel = "candle15m"
            elif timeframe == "1h":
                candle_channel = "candle1H"
            elif timeframe == "4h":
                candle_channel = "candle4H"
            elif timeframe == "1d":
                candle_channel = "candle1D"
            
            args.append({
                "channel": candle_channel,
                "instId": symbol
            })
            
        message = {
            "op": "subscribe", 
            "args": args
        }
        
        self.subscriptions.append(message)
        if self.is_connected:
            await self.send(message)
    
    async def _handle_message(self, message: str):
        """处理OKX消息"""
        try:
            data = json.loads(message)
            
            # 处理订阅确认
            if data.get("event") == "subscribe":
                logger.info(f"OKX订阅成功: {data}")
                return
            
            # 处理错误信息
            if data.get("event") == "error":
                logger.error(f"OKX错误: {data}")
                return
            
            # 处理数据
            if "data" in data and data["data"]:
                await self._process_data(data)
                
        except Exception as e:
            logger.error(f"OKX消息处理失败: {e}")
    
    async def _process_data(self, data: Dict):
        """处理OKX数据"""
        try:
            channel = data.get("arg", {}).get("channel", "")
            symbol = data.get("arg", {}).get("instId", "")
            
            if channel == "tickers":
                await self._process_ticker(symbol, data["data"][0])
            elif channel.startswith("candle"):
                await self._process_candle(symbol, data["data"])
                
        except Exception as e:
            logger.error(f"OKX数据处理失败: {e}")
    
    async def _process_ticker(self, symbol: str, ticker_data: Dict):
        """处理行情数据"""
        try:
            # 字段兼容处理
            last_price = float(ticker_data.get("last", 0) or 0)
            # OKX 文档：open24h / sodUtc0 / sodUtc8 可作为参考开盘价
            open_24h = ticker_data.get("open24h") or ticker_data.get("sodUtc0") or ticker_data.get("sodUtc8")
            open_24h = float(open_24h) if open_24h is not None else 0.0

            if open_24h > 0:
                change_abs = last_price - open_24h
                change_pct = (change_abs / open_24h) * 100.0
            else:
                # 回退到服务端可能提供的其他变化字段（若存在）
                change_abs = float(ticker_data.get("chgUtc", ticker_data.get("change24h", 0)) or 0)
                change_pct = float(ticker_data.get("chgUtcPct", ticker_data.get("change24h_pct", 0)) or 0)

            # 24h 成交量：优先使用报价货币体量，其次基础货币体量
            volume_24h = ticker_data.get("volCcy24h")
            if volume_24h is None:
                volume_24h = ticker_data.get("vol24h", 0)
            volume_24h = float(volume_24h or 0)

            processed_data = {
                "symbol": symbol,
                "exchange": "OKX",
                "price": last_price,
                "volume_24h": volume_24h,
                "change_24h": change_abs,
                "change_24h_pct": change_pct,
                "high_24h": float(ticker_data.get("high24h", 0) or 0),
                "low_24h": float(ticker_data.get("low24h", 0) or 0),
                "timestamp": int(ticker_data.get("ts", time.time() * 1000))
            }
            
            # 缓存到Redis
            if data_manager.cache_manager:
                await data_manager.cache_manager.set_market_data(
                    f"OKX:{symbol}", 
                    processed_data
                )
            
            # 触发回调
            if "ticker" in self.callbacks:
                await self.callbacks["ticker"](processed_data)
                
        except Exception as e:
            logger.error(f"处理OKX行情数据失败: {e}")
    
    async def _process_candle(self, symbol: str, candle_data: List):
        """处理K线数据"""
        try:
            processed_candles = []
            for candle in candle_data:
                processed_candle = {
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                }
                processed_candles.append(processed_candle)
            
            # 保存到MongoDB
            if data_manager.time_series_manager:
                await data_manager.time_series_manager.insert_kline_data(
                    symbol, "1m", processed_candles
                )
            
            # 触发回调
            if "candle" in self.callbacks:
                await self.callbacks["candle"](symbol, processed_candles)
                
        except Exception as e:
            logger.error(f"处理OKX K线数据失败: {e}")

class BinanceWebSocketClient(BaseWebSocketClient):
    """Binance WebSocket客户端"""
    
    def __init__(self):
        super().__init__(
            "Binance",
            "wss://stream.binance.com:9443/ws"
        )
        self.api_config = settings.get_api_config("binance")
        
    async def subscribe_tickers(self, symbols: List[str]):
        """订阅行情数据"""
        streams = []
        for symbol in symbols:
            streams.append(f"{symbol.lower()}@ticker")
            
        message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        self.subscriptions.append(message)
        if self.is_connected:
            await self.send(message)
    
    async def subscribe_candles(self, symbols: List[str], timeframe: str = "1m"):
        """订阅K线数据"""
        streams = []
        for symbol in symbols:
            streams.append(f"{symbol.lower()}@kline_{timeframe}")
            
        message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        self.subscriptions.append(message)
        if self.is_connected:
            await self.send(message)
    
    async def _handle_message(self, message: str):
        """处理Binance消息"""
        try:
            data = json.loads(message)
            
            # 处理订阅响应
            if "result" in data:
                if data["result"] is None:
                    logger.info(f"Binance订阅成功: {data}")
                else:
                    logger.error(f"Binance订阅失败: {data}")
                return
            
            # 处理流数据
            if "stream" in data:
                await self._process_stream_data(data)
                
        except Exception as e:
            logger.error(f"Binance消息处理失败: {e}")
    
    async def _process_stream_data(self, data: Dict):
        """处理Binance流数据"""
        try:
            stream = data["stream"]
            payload = data["data"]
            
            if "@ticker" in stream:
                await self._process_ticker(payload)
            elif "@kline" in stream:
                await self._process_kline(payload)
                
        except Exception as e:
            logger.error(f"Binance流数据处理失败: {e}")
    
    async def _process_ticker(self, ticker_data: Dict):
        """处理行情数据"""
        try:
            processed_data = {
                "symbol": ticker_data["s"],
                "exchange": "Binance",
                "price": float(ticker_data["c"]),
                "volume_24h": float(ticker_data["v"]),
                "change_24h": float(ticker_data["P"]),
                "change_24h_pct": float(ticker_data["P"]),
                "high_24h": float(ticker_data["h"]),
                "low_24h": float(ticker_data["l"]),
                "timestamp": int(ticker_data["E"])
            }
            
            # 缓存到Redis
            if data_manager.cache_manager:
                await data_manager.cache_manager.set_market_data(
                    f"Binance:{processed_data['symbol']}", 
                    processed_data
                )
            
            # 触发回调
            if "ticker" in self.callbacks:
                await self.callbacks["ticker"](processed_data)
                
        except Exception as e:
            logger.error(f"处理Binance行情数据失败: {e}")
    
    async def _process_kline(self, kline_data: Dict):
        """处理K线数据"""
        try:
            k = kline_data["k"]
            
            # 只处理已完成的K线
            if not k["x"]:
                return
                
            processed_candle = {
                "timestamp": int(k["t"]),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"])
            }
            
            # 保存到MongoDB
            if data_manager.time_series_manager:
                await data_manager.time_series_manager.insert_kline_data(
                    k["s"], "1m", [processed_candle]
                )
            
            # 触发回调
            if "candle" in self.callbacks:
                await self.callbacks["candle"](k["s"], [processed_candle])
                
        except Exception as e:
            logger.error(f"处理Binance K线数据失败: {e}")

class WebSocketManager:
    """WebSocket管理器"""
    
    def __init__(self):
        self.clients: Dict[str, BaseWebSocketClient] = {}
        self.is_running = False
        
    def add_client(self, name: str, client: BaseWebSocketClient):
        """添加WebSocket客户端"""
        self.clients[name] = client
        logger.info(f"添加WebSocket客户端: {name}")
    
    async def start_all(self):
        """启动所有WebSocket客户端"""
        if self.is_running:
            return
            
        tasks = []
        for name, client in self.clients.items():
            try:
                await client.connect()
                task = asyncio.create_task(client.listen())
                tasks.append(task)
                logger.info(f"启动WebSocket客户端: {name}")
            except Exception as e:
                logger.error(f"启动WebSocket客户端失败 {name}: {e}")
        
        if tasks:
            self.is_running = True
            # 并发运行所有客户端
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self):
        """停止所有WebSocket客户端"""
        self.is_running = False
        
        for name, client in self.clients.items():
            try:
                await client.disconnect()
                logger.info(f"停止WebSocket客户端: {name}")
            except Exception as e:
                logger.error(f"停止WebSocket客户端失败 {name}: {e}")
    
    def get_client(self, name: str) -> Optional[BaseWebSocketClient]:
        """获取WebSocket客户端"""
        return self.clients.get(name)
    
    def get_connection_status(self) -> Dict[str, bool]:
        """获取连接状态"""
        return {
            name: client.is_connected 
            for name, client in self.clients.items()
        }

# 全局WebSocket管理器实例
websocket_manager = WebSocketManager()

# 为了向后兼容，提供WebSocketClient别名
WebSocketClient = WebSocketManager