"""
数据回放系统

提供历史数据的时间序列回放功能：
- 多数据源支持
- 时间精确控制
- 事件驱动架构
- 高效数据加载和预处理
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from collections import deque
import bisect

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class MarketData:
    """市场数据"""
    timestamp: datetime
    symbol: str
    data_type: str  # 'kline', 'tick', 'orderbook', 'trade'
    data: Dict[str, Any]

class DataSource:
    """数据源基类"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
    
    async def load_data(self, symbols: List[str], timeframes: List[str],
                       start_date: datetime, end_date: datetime) -> List[MarketData]:
        """加载数据 - 需要子类实现"""
        raise NotImplementedError

class CoinglassDataSource(DataSource):
    """CoinGlass数据源"""
    
    def __init__(self):
        super().__init__("coinglass")
        self.data_path = Path("coinglass_副本/data")
    
    async def load_data(self, symbols: List[str], timeframes: List[str],
                       start_date: datetime, end_date: datetime) -> List[MarketData]:
        """从CoinGlass数据加载"""
        try:
            market_data = []
            
            for symbol in symbols:
                # 转换symbol格式 (BTC/USDT -> BTC)
                coin_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                
                # 加载现货数据
                spot_file = self.data_path / "coinglass" / "spot" / f"{coin_symbol.lower()}-spot-markets.json"
                if spot_file.exists():
                    with open(spot_file, 'r') as f:
                        spot_data = json.load(f)
                    
                    for record in spot_data.get('data', []):
                        if 'price' in record:
                            market_data.append(MarketData(
                                timestamp=pd.to_datetime(record.get('timestamp', datetime.utcnow())),
                                symbol=symbol,
                                data_type='spot_price',
                                data=record
                            ))
                
                # 加载期货数据
                futures_file = self.data_path / "coinglass" / "futures" / f"{coin_symbol.lower()}-futures-markets.json"
                if futures_file.exists():
                    with open(futures_file, 'r') as f:
                        futures_data = json.load(f)
                    
                    for record in futures_data.get('data', []):
                        if 'price' in record:
                            market_data.append(MarketData(
                                timestamp=pd.to_datetime(record.get('timestamp', datetime.utcnow())),
                                symbol=symbol,
                                data_type='futures_price',
                                data=record
                            ))
            
            # 按时间排序
            market_data.sort(key=lambda x: x.timestamp)
            
            # 筛选时间范围
            filtered_data = [
                data for data in market_data
                if start_date <= data.timestamp <= end_date
            ]
            
            logger.info(f"CoinGlass数据加载完成: {len(filtered_data)}条记录")
            return filtered_data
            
        except Exception as e:
            logger.error(f"CoinGlass数据加载失败: {e}")
            return []

class OKXDataSource(DataSource):
    """OKX数据源"""
    
    def __init__(self):
        super().__init__("okx")
        self.data_path = Path("coinglass_副本/data/okx")
    
    async def load_data(self, symbols: List[str], timeframes: List[str],
                       start_date: datetime, end_date: datetime) -> List[MarketData]:
        """从OKX数据加载K线数据"""
        try:
            market_data = []
            
            for symbol in symbols:
                # 转换symbol格式
                coin_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                
                for timeframe in timeframes:
                    # 构建文件路径
                    symbol_dir = self.data_path / coin_symbol.upper() / timeframe
                    data_file = symbol_dir / "data.json"
                    
                    if not data_file.exists():
                        logger.warning(f"OKX数据文件不存在: {data_file}")
                        continue
                    
                    try:
                        with open(data_file, 'r') as f:
                            kline_data = json.load(f)
                        
                        for record in kline_data:
                            # OKX数据格式: [timestamp, open, high, low, close, volume, ...]
                            if len(record) >= 6:
                                timestamp = pd.to_datetime(int(record[0]), unit='ms')
                                
                                # 跳过时间范围外的数据
                                if not (start_date <= timestamp <= end_date):
                                    continue
                                
                                kline_dict = {
                                    'timestamp': timestamp,
                                    'open': float(record[1]),
                                    'high': float(record[2]),
                                    'low': float(record[3]),
                                    'close': float(record[4]),
                                    'volume': float(record[5]),
                                    'timeframe': timeframe
                                }
                                
                                market_data.append(MarketData(
                                    timestamp=timestamp,
                                    symbol=symbol,
                                    data_type='kline',
                                    data=kline_dict
                                ))
                    
                    except Exception as e:
                        logger.error(f"加载OKX文件失败 {data_file}: {e}")
                        continue
            
            # 按时间排序
            market_data.sort(key=lambda x: x.timestamp)
            
            logger.info(f"OKX数据加载完成: {len(market_data)}条记录")
            return market_data
            
        except Exception as e:
            logger.error(f"OKX数据加载失败: {e}")
            return []

class SimulatedDataSource(DataSource):
    """模拟数据源"""
    
    def __init__(self):
        super().__init__("simulated")
    
    async def load_data(self, symbols: List[str], timeframes: List[str],
                       start_date: datetime, end_date: datetime) -> List[MarketData]:
        """生成模拟K线数据"""
        try:
            market_data = []
            
            for symbol in symbols:
                # 设定初始价格
                initial_prices = {
                    'BTC/USDT': 45000.0,
                    'ETH/USDT': 2800.0,
                    'ADA/USDT': 1.2,
                    'DOT/USDT': 35.0,
                    'LINK/USDT': 28.0
                }
                
                initial_price = initial_prices.get(symbol, 1000.0)
                current_price = initial_price
                
                # 生成时间序列
                current_time = start_date
                time_delta = timedelta(minutes=1)  # 1分钟K线
                
                while current_time <= end_date:
                    # 随机价格变动 (几何布朗运动)
                    returns = np.random.normal(0.0001, 0.02)  # 均值0.01%, 标准差2%
                    price_change = current_price * returns
                    current_price = max(0.01, current_price + price_change)
                    
                    # 生成OHLC数据
                    volatility = current_price * 0.001  # 0.1%内波动
                    high = current_price + np.random.uniform(0, volatility)
                    low = current_price - np.random.uniform(0, volatility)
                    open_price = current_price + np.random.uniform(-volatility/2, volatility/2)
                    close_price = current_price
                    volume = np.random.uniform(100, 10000)
                    
                    kline_dict = {
                        'timestamp': current_time,
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close_price,
                        'volume': volume,
                        'timeframe': '1m'
                    }
                    
                    market_data.append(MarketData(
                        timestamp=current_time,
                        symbol=symbol,
                        data_type='kline',
                        data=kline_dict
                    ))
                    
                    current_time += time_delta
            
            logger.info(f"模拟数据生成完成: {len(market_data)}条记录")
            return market_data
            
        except Exception as e:
            logger.error(f"模拟数据生成失败: {e}")
            return []

class DataReplaySystem:
    """数据回放系统主类"""
    
    def __init__(self, symbols: List[str], timeframes: List[str],
                 start_date: datetime, end_date: datetime,
                 data_source: str = "okx"):
        
        self.symbols = symbols
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        self.data_source_name = data_source
        
        # 数据存储
        self.market_data: List[MarketData] = []
        self.current_index = 0
        self.is_initialized = False
        
        # 数据源
        self.data_sources = {
            'coinglass': CoinglassDataSource(),
            'okx': OKXDataSource(),
            'simulated': SimulatedDataSource()
        }
        
        # 事件订阅
        self.data_event_handlers: List[Callable] = []
        self.bar_event_handlers: List[Callable] = []
        self.tick_event_handlers: List[Callable] = []
        
        # 缓存和优化
        self.data_cache: Dict[str, Any] = {}
        self.preload_buffer_size = 1000
        self.current_buffer = deque(maxlen=self.preload_buffer_size)
        
        # 时间控制
        self.current_time: Optional[datetime] = None
        self.time_step = timedelta(seconds=1)  # 默认1秒时间步长
        
        logger.info(f"数据回放系统初始化: {data_source} 数据源, "
                   f"{len(symbols)}个品种, 时间范围: {start_date} - {end_date}")
    
    async def initialize(self) -> None:
        """初始化数据回放系统"""
        try:
            if self.data_source_name not in self.data_sources:
                raise ValueError(f"不支持的数据源: {self.data_source_name}")
            
            # 选择数据源
            data_source = self.data_sources[self.data_source_name]
            
            # 加载数据
            logger.info(f"开始加载数据: {self.data_source_name}")
            self.market_data = await data_source.load_data(
                symbols=self.symbols,
                timeframes=self.timeframes,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if not self.market_data:
                logger.warning("没有加载到任何数据，使用模拟数据源")
                simulated_source = self.data_sources['simulated']
                self.market_data = await simulated_source.load_data(
                    symbols=self.symbols,
                    timeframes=self.timeframes,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
            
            # 重置索引
            self.current_index = 0
            self.current_time = self.start_date if self.market_data else None
            
            # 预加载缓冲区
            await self._preload_buffer()
            
            self.is_initialized = True
            logger.info(f"数据回放系统初始化完成: {len(self.market_data)}条记录")
            
        except Exception as e:
            logger.error(f"数据回放系统初始化失败: {e}")
            raise
    
    async def _preload_buffer(self) -> None:
        """预加载数据到缓冲区"""
        try:
            self.current_buffer.clear()
            
            end_index = min(self.current_index + self.preload_buffer_size, len(self.market_data))
            for i in range(self.current_index, end_index):
                self.current_buffer.append(self.market_data[i])
                
        except Exception as e:
            logger.error(f"预加载缓冲区失败: {e}")
    
    # 事件订阅
    def subscribe_data_event(self, handler: Callable) -> None:
        """订阅数据事件"""
        self.data_event_handlers.append(handler)
    
    def subscribe_bar_event(self, handler: Callable) -> None:
        """订阅K线事件"""
        self.bar_event_handlers.append(handler)
    
    def subscribe_tick_event(self, handler: Callable) -> None:
        """订阅Tick事件"""
        self.tick_event_handlers.append(handler)
    
    # 主要回放方法
    async def replay_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """回放数据生成器"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"开始数据回放: {len(self.market_data)}条记录")
            
            while self.current_index < len(self.market_data):
                # 获取当前数据点
                current_data = self.market_data[self.current_index]
                self.current_time = current_data.timestamp
                
                # 构建市场数据字典
                market_data_dict = {
                    'timestamp': current_data.timestamp,
                    'symbol': current_data.symbol,
                    'data_type': current_data.data_type,
                    'data': current_data.data
                }
                
                # 触发事件
                await self._trigger_events(current_data)
                
                # 更新索引
                self.current_index += 1
                
                # 更新缓冲区
                if self.current_index % 100 == 0:  # 每100条记录更新一次
                    await self._preload_buffer()
                
                # 返回数据
                yield market_data_dict
                
                # 可选的时间控制
                await asyncio.sleep(0.001)  # 1ms延迟，防止占用太多CPU
            
            logger.info("数据回放完成")
            
        except Exception as e:
            logger.error(f"数据回放失败: {e}")
            raise
    
    async def _trigger_events(self, data: MarketData) -> None:
        """触发数据事件"""
        try:
            # 通用数据事件
            data_dict = {
                'timestamp': data.timestamp,
                'symbol': data.symbol,
                'data_type': data.data_type,
                'data': data.data
            }
            
            for handler in self.data_event_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data_dict)
                    else:
                        handler(data_dict)
                except Exception as e:
                    logger.error(f"数据事件处理失败: {e}")
            
            # K线事件
            if data.data_type == 'kline':
                for handler in self.bar_event_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data_dict)
                        else:
                            handler(data_dict)
                    except Exception as e:
                        logger.error(f"K线事件处理失败: {e}")
            
            # Tick事件  
            elif data.data_type in ['tick', 'trade']:
                for handler in self.tick_event_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data_dict)
                        else:
                            handler(data_dict)
                    except Exception as e:
                        logger.error(f"Tick事件处理失败: {e}")
                        
        except Exception as e:
            logger.error(f"触发事件失败: {e}")
    
    # 数据查询和辅助方法
    async def get_data_at_time(self, timestamp: datetime) -> List[MarketData]:
        """获取指定时间的数据"""
        try:
            # 使用二分查找
            timestamps = [data.timestamp for data in self.market_data]
            index = bisect.bisect_left(timestamps, timestamp)
            
            # 获取该时间点前后的数据
            result = []
            for i in range(max(0, index-1), min(len(self.market_data), index+2)):
                if abs((self.market_data[i].timestamp - timestamp).total_seconds()) < 60:  # 1分钟内
                    result.append(self.market_data[i])
            
            return result
            
        except Exception as e:
            logger.error(f"获取指定时间数据失败: {e}")
            return []
    
    async def get_price_at_time(self, symbol: str, timestamp: datetime) -> Optional[float]:
        """获取指定时间的价格"""
        try:
            data_points = await self.get_data_at_time(timestamp)
            
            for data in data_points:
                if data.symbol == symbol and data.data_type == 'kline':
                    return data.data.get('close', 0.0)
                elif data.symbol == symbol and 'price' in data.data:
                    return data.data['price']
            
            return None
            
        except Exception as e:
            logger.error(f"获取价格失败: {e}")
            return None
    
    async def get_benchmark_data(self, benchmark_symbol: str) -> List[Dict[str, Any]]:
        """获取基准数据"""
        try:
            benchmark_data = []
            
            for data in self.market_data:
                if (data.symbol == benchmark_symbol and 
                    data.data_type == 'kline' and 
                    'close' in data.data):
                    
                    benchmark_data.append({
                        'timestamp': data.timestamp,
                        'close': data.data['close'],
                        'volume': data.data.get('volume', 0)
                    })
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"获取基准数据失败: {e}")
            return []
    
    def get_data_range(self, start_time: datetime, end_time: datetime) -> List[MarketData]:
        """获取时间范围内的数据"""
        try:
            return [
                data for data in self.market_data
                if start_time <= data.timestamp <= end_time
            ]
            
        except Exception as e:
            logger.error(f"获取时间范围数据失败: {e}")
            return []
    
    def get_current_prices(self) -> Dict[str, float]:
        """获取当前价格"""
        try:
            current_prices = {}
            
            # 从当前位置向后查找最新价格
            for i in range(max(0, self.current_index-100), self.current_index):
                data = self.market_data[i]
                
                if data.data_type == 'kline' and 'close' in data.data:
                    current_prices[data.symbol] = data.data['close']
                elif 'price' in data.data:
                    current_prices[data.symbol] = data.data['price']
            
            return current_prices
            
        except Exception as e:
            logger.error(f"获取当前价格失败: {e}")
            return {}
    
    # 控制方法
    def seek_to_time(self, timestamp: datetime) -> None:
        """跳转到指定时间"""
        try:
            timestamps = [data.timestamp for data in self.market_data]
            self.current_index = bisect.bisect_left(timestamps, timestamp)
            self.current_time = timestamp
            
            logger.info(f"跳转到时间: {timestamp}, 索引: {self.current_index}")
            
        except Exception as e:
            logger.error(f"时间跳转失败: {e}")
    
    def get_progress(self) -> float:
        """获取回放进度"""
        if not self.market_data:
            return 0.0
        return self.current_index / len(self.market_data)
    
    def get_status(self) -> Dict[str, Any]:
        """获取回放状态"""
        return {
            'is_initialized': self.is_initialized,
            'total_records': len(self.market_data),
            'current_index': self.current_index,
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'progress': self.get_progress(),
            'data_source': self.data_source_name,
            'symbols': self.symbols,
            'timeframes': self.timeframes
        }