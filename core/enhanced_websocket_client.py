"""
增强版WebSocket客户端管理器
修复Binance连接问题，实现智能重连和数据验证
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, InvalidHandshake
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class ConnectionStats:
    """连接统计信息"""
    def __init__(self):
        self.connect_time = None
        self.total_messages = 0
        self.successful_messages = 0
        self.failed_messages = 0
        self.last_message_time = None
        self.reconnect_count = 0
        self.total_downtime = 0.0
        
    def record_message(self, success: bool = True):
        """记录消息统计"""
        self.total_messages += 1
        self.last_message_time = datetime.utcnow()
        if success:
            self.successful_messages += 1
        else:
            self.failed_messages += 1
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_messages == 0:
            return 0.0
        return self.successful_messages / self.total_messages * 100

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_price_data(data: Dict) -> bool:
        """验证价格数据"""
        try:
            price = float(data.get("price", 0))
            volume = float(data.get("volume_24h", 0))
            
            # 基本范围检查
            if price <= 0 or price > 10000000:  # 价格范围检查
                logger.warning(f"异常价格: {price}")
                return False
                
            if volume < 0:  # 成交量检查
                logger.warning(f"异常成交量: {volume}")
                return False
                
            # 时间戳检查
            timestamp = data.get("timestamp", 0)
            if timestamp > 0:
                data_time = datetime.fromtimestamp(timestamp / 1000)
                age = (datetime.utcnow() - data_time).total_seconds()
                if age > 300:  # 数据不能超过5分钟
                    logger.warning(f"数据过期: {age}秒")
                    return False
                    
            return True
            
        except (ValueError, TypeError) as e:
            logger.error(f"数据验证失败: {e}")
            return False
    
    @staticmethod
    def validate_candle_data(data: List[Dict]) -> bool:
        """验证K线数据"""
        try:
            for candle in data:
                if not all(key in candle for key in ["o", "h", "l", "c", "v"]):
                    return False
                    
                open_price = float(candle["o"])
                high_price = float(candle["h"])
                low_price = float(candle["l"])
                close_price = float(candle["c"])
                volume = float(candle["v"])
                
                # 价格关系检查
                if not (low_price <= open_price <= high_price and 
                       low_price <= close_price <= high_price):
                    logger.warning(f"K线价格关系异常: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
                    return False
                    
                if volume < 0:
                    return False
                    
            return True
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"K线数据验证失败: {e}")
            return False

class EnhancedWebSocketClient:
    """增强版WebSocket客户端基类"""
    
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.websocket = None
        self.is_connected = False
        self.subscriptions: List[Dict] = []
        self.callbacks: Dict[str, Callable] = {}
        
        # 连接管理
        self._running = False
        self._reconnect_task = None
        self._heartbeat_task = None
        self._message_task = None
        
        # 重连配置
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_reconnect_delay = 5
        self.max_reconnect_delay = 300
        
        # 统计信息
        self.stats = ConnectionStats()
        self.validator = DataValidator()
        
        # 健康检查
        self.last_pong_time = None
        self.ping_interval = 30
        self.ping_timeout = 10
        
    async def connect(self):
        """建立WebSocket连接"""
        if self.is_connected:
            return
            
        disconnect_time = datetime.utcnow() if self.stats.connect_time else None
        
        try:
            logger.info(f"🔌 连接{self.name} WebSocket: {self.url}")
            
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=10,
                max_size=1024*1024,  # 1MB消息限制
                compression=None     # 禁用压缩提高性能
            )
            
            self.is_connected = True
            self.stats.connect_time = datetime.utcnow()
            self.reconnect_attempts = 0
            
            # 计算下线时间
            if disconnect_time:
                downtime = (self.stats.connect_time - disconnect_time).total_seconds()
                self.stats.total_downtime += downtime
                logger.info(f"📊 {self.name}重连成功，下线时长: {downtime:.1f}秒")
            
            logger.info(f"✅ {self.name} WebSocket连接成功")
            
            # 启动消息处理和健康检查
            self._running = True
            self._message_task = asyncio.create_task(self._message_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # 重新订阅
            if self.subscriptions:
                await self._resubscribe_all()
                
        except Exception as e:
            logger.error(f"❌ {self.name} WebSocket连接失败: {e}")
            self.is_connected = False
            
            if not self._running:
                return
                
            # 启动重连任务
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            raise
    
    async def disconnect(self):
        """断开WebSocket连接"""
        logger.info(f"🔴 断开{self.name} WebSocket连接...")
        
        self._running = False
        
        # 取消所有任务
        for task in [self._reconnect_task, self._heartbeat_task, self._message_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 关闭连接
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            
        self.websocket = None
        self.is_connected = False
        
        logger.info(f"✅ {self.name} WebSocket已断开")
    
    async def send(self, message: Dict):
        """发送消息"""
        if not self.is_connected or not self.websocket:
            logger.warning(f"⚠️ {self.name} 未连接，无法发送消息")
            return False
            
        try:
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            logger.debug(f"📤 {self.name} 发送: {message_str}")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.name} 发送消息失败: {e}")
            return False
    
    async def _message_loop(self):
        """消息接收循环"""
        while self._running and self.is_connected:
            try:
                if not self.websocket:
                    break
                    
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=60
                )
                
                self.stats.record_message(True)
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ {self.name} 消息接收超时")
                continue
                
            except ConnectionClosed:
                logger.warning(f"🔌 {self.name} WebSocket连接关闭")
                self.is_connected = False
                break
                
            except Exception as e:
                logger.error(f"❌ {self.name} 消息循环错误: {e}")
                self.stats.record_message(False)
                await asyncio.sleep(1)
    
    async def _heartbeat_loop(self):
        """心跳检查循环"""
        while self._running and self.is_connected:
            try:
                if self.websocket:
                    # 发送ping
                    pong_waiter = await self.websocket.ping()
                    try:
                        await asyncio.wait_for(pong_waiter, timeout=self.ping_timeout)
                        self.last_pong_time = datetime.utcnow()
                        logger.debug(f"💗 {self.name} 心跳正常")
                    except asyncio.TimeoutError:
                        logger.warning(f"💔 {self.name} 心跳超时")
                        self.is_connected = False
                        break
                        
                await asyncio.sleep(self.ping_interval)
                
            except Exception as e:
                logger.error(f"❌ {self.name} 心跳检查错误: {e}")
                await asyncio.sleep(5)
    
    async def _reconnect_loop(self):
        """重连循环"""
        while self._running and self.reconnect_attempts < self.max_reconnect_attempts:
            if self.is_connected:
                break
                
            self.reconnect_attempts += 1
            
            # 指数退避延迟
            delay = min(
                self.base_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                self.max_reconnect_delay
            )
            
            logger.info(f"🔄 {self.name} 第{self.reconnect_attempts}次重连，等待{delay}秒...")
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                if self.is_connected:
                    self.stats.reconnect_count += 1
                    logger.info(f"✅ {self.name} 重连成功")
                    break
                    
            except Exception as e:
                logger.error(f"❌ {self.name} 重连失败: {e}")
                
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"🚫 {self.name} 重连次数超限，停止重连")
    
    async def _resubscribe_all(self):
        """重新订阅所有频道"""
        logger.info(f"🔄 {self.name} 重新订阅{len(self.subscriptions)}个频道...")
        
        for subscription in self.subscriptions:
            try:
                success = await self.send(subscription)
                if success:
                    logger.debug(f"✅ 重新订阅成功: {subscription}")
                else:
                    logger.warning(f"⚠️ 重新订阅失败: {subscription}")
                    
                await asyncio.sleep(0.1)  # 避免频率限制
                
            except Exception as e:
                logger.error(f"❌ 重新订阅错误: {e}")
    
    def add_callback(self, event_type: str, callback: Callable):
        """添加事件回调"""
        self.callbacks[event_type] = callback
    
    def get_connection_info(self) -> Dict:
        """获取连接信息"""
        return {
            "name": self.name,
            "connected": self.is_connected,
            "connect_time": self.stats.connect_time.isoformat() if self.stats.connect_time else None,
            "total_messages": self.stats.total_messages,
            "success_rate": self.stats.get_success_rate(),
            "reconnect_count": self.stats.reconnect_count,
            "total_downtime": self.stats.total_downtime,
            "subscriptions": len(self.subscriptions)
        }
    
    async def _handle_message(self, message: str):
        """处理消息 - 需要子类实现"""
        raise NotImplementedError("子类必须实现_handle_message方法")

class EnhancedBinanceWebSocketClient(EnhancedWebSocketClient):
    """增强版Binance WebSocket客户端"""
    
    def __init__(self):
        super().__init__("Binance", "wss://stream.binance.com:9443/ws")
        from .data_manager import data_manager
        self.data_manager = data_manager
        
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
            success = await self.send(message)
            if success:
                logger.info(f"📊 Binance订阅行情: {symbols}")
            return success
        return False
    
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
            success = await self.send(message)
            if success:
                logger.info(f"📈 Binance订阅K线: {symbols} ({timeframe})")
            return success
        return False
    
    async def _handle_message(self, message: str):
        """处理Binance消息"""
        try:
            data = json.loads(message)
            
            # 处理订阅响应
            if "result" in data:
                if data["result"] is None:
                    logger.info(f"✅ Binance订阅成功: ID {data.get('id')}")
                else:
                    logger.error(f"❌ Binance订阅失败: {data}")
                return
            
            # 处理错误信息
            if "error" in data:
                logger.error(f"❌ Binance API错误: {data['error']}")
                return
            
            # 处理流数据
            if "stream" in data and "data" in data:
                await self._process_stream_data(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Binance JSON解析失败: {e}")
            self.stats.record_message(False)
        except Exception as e:
            logger.error(f"❌ Binance消息处理失败: {e}")
            self.stats.record_message(False)
    
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
            logger.error(f"❌ Binance流数据处理失败: {e}")
    
    async def _process_ticker(self, ticker_data: Dict):
        """处理行情数据"""
        try:
            # 修复字段映射问题
            processed_data = {
                "symbol": ticker_data["s"],  # 交易对
                "exchange": "Binance",
                "price": float(ticker_data["c"]),  # 最新价格
                "volume_24h": float(ticker_data["q"]),  # 24小时成交额(报价货币)
                "change_24h": float(ticker_data["p"]),  # 24小时价格变化
                "change_24h_pct": float(ticker_data["P"]),  # 24小时价格变化百分比
                "high_24h": float(ticker_data["h"]),  # 24小时最高价
                "low_24h": float(ticker_data["l"]),   # 24小时最低价
                "timestamp": int(ticker_data["E"])    # 事件时间
            }
            
            # 数据验证
            if not self.validator.validate_price_data(processed_data):
                logger.warning(f"⚠️ Binance数据验证失败: {processed_data['symbol']}")
                return
            
            # 缓存到Redis
            if self.data_manager.cache_manager:
                cache_key = f"Binance:{processed_data['symbol']}"
                await self.data_manager.cache_manager.set_market_data(
                    cache_key, 
                    processed_data
                )
                logger.debug(f"💾 缓存Binance数据: {cache_key}")
            
            # 触发回调
            if "ticker" in self.callbacks:
                try:
                    callback = self.callbacks["ticker"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed_data["symbol"], processed_data)
                    else:
                        callback(processed_data["symbol"], processed_data)
                except Exception as e:
                    logger.error(f"❌ Binance ticker回调失败: {e}")
            
            logger.debug(f"📊 Binance {processed_data['symbol']}: ${processed_data['price']} ({processed_data['change_24h_pct']:+.2f}%)")
                
        except (KeyError, ValueError) as e:
            logger.error(f"❌ 处理Binance行情数据失败: {e}")
            logger.debug(f"原始数据: {ticker_data}")
    
    async def _process_kline(self, kline_data: Dict):
        """处理K线数据"""
        try:
            k = kline_data["k"]
            
            processed_data = {
                "symbol": k["s"],
                "exchange": "Binance",
                "timestamp": int(k["t"]),  # 开盘时间
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),   # 成交量
                "close_time": int(k["T"]), # 收盘时间
                "is_closed": k["x"]        # K线是否完结
            }
            
            # 只处理完结的K线
            if not processed_data["is_closed"]:
                return
            
            # 数据验证
            if not self.validator.validate_candle_data([{
                "o": processed_data["open"],
                "h": processed_data["high"],
                "l": processed_data["low"],
                "c": processed_data["close"],
                "v": processed_data["volume"]
            }]):
                logger.warning(f"⚠️ Binance K线数据验证失败: {processed_data['symbol']}")
                return
            
            # 触发回调
            if "candle" in self.callbacks:
                try:
                    callback = self.callbacks["candle"]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed_data["symbol"], [processed_data])
                    else:
                        callback(processed_data["symbol"], [processed_data])
                except Exception as e:
                    logger.error(f"❌ Binance candle回调失败: {e}")
                    
            logger.debug(f"📈 Binance K线 {processed_data['symbol']}: OHLC({processed_data['open']}, {processed_data['high']}, {processed_data['low']}, {processed_data['close']})")
                
        except (KeyError, ValueError) as e:
            logger.error(f"❌ 处理Binance K线数据失败: {e}")
            logger.debug(f"原始K线数据: {kline_data}")

# 创建全局实例
enhanced_binance_client = EnhancedBinanceWebSocketClient()