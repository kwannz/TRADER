"""
优化数据流水线管理器
实现高性能数据流处理、批量操作和智能缓冲
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

# 延迟导入智能清洗引擎以避免循环导入
try:
    from .intelligent_data_cleaner import intelligent_cleaner
    CLEANING_ENABLED = True
    logger.info("✅ 智能数据清洗引擎已加载")
except ImportError as e:
    intelligent_cleaner = None
    CLEANING_ENABLED = False
    logger.warning(f"⚠️ 智能数据清洗引擎加载失败: {e}")

@dataclass
class PipelineMetrics:
    """数据流水线性能指标"""
    processed_count: int = 0
    error_count: int = 0
    avg_processing_time: float = 0.0
    throughput: float = 0.0  # 每秒处理数量
    buffer_size: int = 0
    last_update: datetime = None
    
class DataBuffer:
    """智能数据缓冲器"""
    
    def __init__(self, max_size: int = 1000, flush_interval: float = 1.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer = deque()
        self.last_flush = time.time()
        self.lock = asyncio.Lock()
        
    async def add(self, data: Dict[str, Any]) -> bool:
        """添加数据到缓冲区"""
        async with self.lock:
            self.buffer.append(data)
            
            # 检查是否需要刷新缓冲区
            current_time = time.time()
            should_flush = (
                len(self.buffer) >= self.max_size or
                current_time - self.last_flush >= self.flush_interval
            )
            
            return should_flush
    
    async def flush(self) -> List[Dict[str, Any]]:
        """刷新缓冲区，返回所有数据"""
        async with self.lock:
            if not self.buffer:
                return []
                
            data = list(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            return data
    
    def size(self) -> int:
        """获取缓冲区大小"""
        return len(self.buffer)

class DataProcessor:
    """数据处理器基类"""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.metrics = PipelineMetrics()
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条数据"""
        start_time = time.time()
        
        try:
            result = await self._process_data(data)
            
            # 更新指标
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success=True)
            
            return result
            
        except Exception as e:
            self._update_metrics(time.time() - start_time, success=False)
            logger.error(f"数据处理失败 [{self.processor_id}]: {e}")
            raise
    
    async def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理数据（集成智能清洗）"""
        start_time = time.time()
        results = []
        
        try:
            # 1. 智能数据清洗（如果启用）
            cleaned_batch = batch_data
            cleaning_result = None
            
            if CLEANING_ENABLED and intelligent_cleaner:
                try:
                    if self.processor_id == "market_data":
                        cleaning_result = await intelligent_cleaner.clean_market_data(batch_data)
                        cleaned_batch = [batch_data[i] for i in range(len(batch_data)) 
                                       if i < cleaning_result.cleaned_count]
                    elif self.processor_id == "candle_data":
                        cleaning_result = await intelligent_cleaner.clean_candle_data(batch_data)
                        cleaned_batch = [batch_data[i] for i in range(len(batch_data)) 
                                       if i < cleaning_result.cleaned_count]
                    
                    if cleaning_result:
                        logger.debug(f"数据清洗完成 [{self.processor_id}]: "
                                   f"原始{cleaning_result.original_count}条 -> "
                                   f"清洗{cleaning_result.cleaned_count}条 "
                                   f"(质量分: {cleaning_result.quality_score})")
                        
                except Exception as e:
                    logger.error(f"智能清洗失败 [{self.processor_id}]: {e}")
                    # 清洗失败时使用原始数据
                    cleaned_batch = batch_data
            
            # 2. 并行处理清洗后的数据
            if cleaned_batch:
                tasks = [self._process_data(data) for data in cleaned_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 统计成功和失败
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                error_count = len(results) - success_count
                
                # 更新批量指标
                processing_time = time.time() - start_time
                self.metrics.processed_count += success_count
                self.metrics.error_count += error_count
                
                if success_count > 0:
                    avg_time = processing_time / len(cleaned_batch)
                    self._update_metrics(avg_time, success=True, batch_size=len(cleaned_batch))
                
                # 过滤掉异常结果
                results = [r for r in results if not isinstance(r, Exception)]
            
            return results
            
        except Exception as e:
            logger.error(f"批量数据处理失败 [{self.processor_id}]: {e}")
            return []
    
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """子类需要实现的具体处理逻辑"""
        raise NotImplementedError
    
    def _update_metrics(self, processing_time: float, success: bool, batch_size: int = 1):
        """更新性能指标"""
        if success:
            self.metrics.processed_count += batch_size
            # 计算平均处理时间
            if self.metrics.avg_processing_time == 0:
                self.metrics.avg_processing_time = processing_time
            else:
                self.metrics.avg_processing_time = (
                    self.metrics.avg_processing_time * 0.9 + processing_time * 0.1
                )
        else:
            self.metrics.error_count += batch_size
        
        self.metrics.last_update = datetime.utcnow()

class MarketDataProcessor(DataProcessor):
    """市场数据处理器"""
    
    def __init__(self):
        super().__init__("market_data")
        
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理市场数据"""
        try:
            # 数据标准化
            processed_data = {
                "symbol": data.get("symbol", ""),
                "exchange": data.get("exchange", ""),
                "price": float(data.get("price", 0)),
                "volume": float(data.get("volume", 0)),
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
                "processed_at": datetime.utcnow(),
                "data_type": "market_data"
            }
            
            # 数据验证
            if processed_data["price"] <= 0:
                raise ValueError(f"无效价格: {processed_data['price']}")
            
            if not processed_data["symbol"]:
                raise ValueError("缺少交易对信息")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"市场数据处理失败: {e}")
            raise

class CandleDataProcessor(DataProcessor):
    """K线数据处理器"""
    
    def __init__(self):
        super().__init__("candle_data")
        
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理K线数据"""
        try:
            processed_data = {
                "symbol": data.get("symbol", ""),
                "exchange": data.get("exchange", ""),
                "timeframe": data.get("timeframe", "1m"),
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "close": float(data.get("close", 0)),
                "volume": float(data.get("volume", 0)),
                "processed_at": datetime.utcnow(),
                "data_type": "candle_data"
            }
            
            # K线数据验证
            if processed_data["high"] < processed_data["low"]:
                raise ValueError(f"K线数据异常: high={processed_data['high']} < low={processed_data['low']}")
            
            if processed_data["close"] <= 0:
                raise ValueError(f"无效收盘价: {processed_data['close']}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"K线数据处理失败: {e}")
            raise

class CoinglaasDataProcessor(DataProcessor):
    """Coinglass数据处理器"""
    
    def __init__(self):
        super().__init__("coinglass_data")
        
    async def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理Coinglass数据"""
        try:
            processed_data = {
                "data_type": data.get("data_type", "coinglass"),
                "symbol": data.get("symbol", ""),
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
                "processed_at": datetime.utcnow()
            }
            
            # 根据数据类型添加特定字段
            data_type = data.get("data_type", "")
            
            if data_type == "fear_greed":
                processed_data.update({
                    "value": int(data.get("value", 50)),
                    "classification": data.get("classification", "Neutral")
                })
            elif data_type == "funding_rate":
                processed_data.update({
                    "exchange": data.get("exchange", ""),
                    "funding_rate": float(data.get("funding_rate", 0)),
                    "next_funding_time": data.get("next_funding_time", 0)
                })
            elif data_type == "open_interest":
                processed_data.update({
                    "exchange": data.get("exchange", ""),
                    "open_interest": float(data.get("open_interest", 0)),
                    "change_24h": float(data.get("change_24h", 0))
                })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Coinglass数据处理失败: {e}")
            raise

class OptimizedDataPipeline:
    """优化数据流水线"""
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 flush_interval: float = 1.0,
                 max_workers: int = 10):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_workers = max_workers
        
        # 数据缓冲区
        self.buffers = {
            "market_data": DataBuffer(buffer_size, flush_interval),
            "candle_data": DataBuffer(buffer_size, flush_interval),
            "coinglass_data": DataBuffer(buffer_size // 10, flush_interval * 5)  # Coinglass数据更新频率低
        }
        
        # 数据处理器
        self.processors = {
            "market_data": MarketDataProcessor(),
            "candle_data": CandleDataProcessor(),
            "coinglass_data": CoinglaasDataProcessor()
        }
        
        # 回调函数
        self.callbacks = {
            "market_data": [],
            "candle_data": [],
            "coinglass_data": []
        }
        
        # 任务队列和工作池
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
        
        # 性能监控
        self.pipeline_metrics = {
            "total_processed": 0,
            "total_errors": 0,
            "start_time": None,
            "throughput": 0.0
        }
    
    def register_callback(self, data_type: str, callback: Callable):
        """注册数据处理完成的回调函数"""
        if data_type in self.callbacks:
            self.callbacks[data_type].append(callback)
    
    async def start(self):
        """启动数据流水线"""
        if self.running:
            logger.warning("数据流水线已在运行")
            return
            
        try:
            self.running = True
            self.pipeline_metrics["start_time"] = datetime.utcnow()
            
            # 启动工作协程
            self.workers = []
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker(f"worker_{i}"))
                self.workers.append(worker)
            
            # 启动缓冲区监控
            self.buffer_monitor_task = asyncio.create_task(self._monitor_buffers())
            
            logger.info(f"优化数据流水线已启动，工作协程数: {self.max_workers}")
            
        except Exception as e:
            logger.error(f"启动数据流水线失败: {e}")
            self.running = False
            raise
    
    async def stop(self):
        """停止数据流水线"""
        if not self.running:
            return
            
        try:
            self.running = False
            
            # 停止缓冲区监控
            if hasattr(self, 'buffer_monitor_task'):
                self.buffer_monitor_task.cancel()
            
            # 停止所有工作协程
            for worker in self.workers:
                worker.cancel()
            
            # 处理剩余缓冲区数据
            await self._flush_all_buffers()
            
            logger.info("数据流水线已停止")
            
        except Exception as e:
            logger.error(f"停止数据流水线失败: {e}")
    
    async def add_data(self, data_type: str, data: Dict[str, Any]):
        """添加数据到流水线"""
        if not self.running:
            logger.warning("数据流水线未运行，无法处理数据")
            return
            
        if data_type not in self.buffers:
            logger.error(f"不支持的数据类型: {data_type}")
            return
        
        try:
            # 添加数据到对应缓冲区
            should_flush = await self.buffers[data_type].add(data)
            
            # 如果需要刷新，立即处理
            if should_flush:
                await self.task_queue.put(("flush", data_type))
                
        except Exception as e:
            logger.error(f"添加数据失败 [{data_type}]: {e}")
    
    async def _worker(self, worker_id: str):
        """工作协程"""
        logger.info(f"工作协程 {worker_id} 已启动")
        
        while self.running:
            try:
                # 从任务队列获取任务
                task_data = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                task_type, data_type = task_data
                
                if task_type == "flush":
                    await self._process_buffer(data_type)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"工作协程 {worker_id} 处理任务失败: {e}")
        
        logger.info(f"工作协程 {worker_id} 已停止")
    
    async def _monitor_buffers(self):
        """缓冲区监控协程"""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                # 检查所有缓冲区
                for data_type, buffer in self.buffers.items():
                    if buffer.size() > 0:
                        current_time = time.time()
                        if current_time - buffer.last_flush >= self.flush_interval:
                            await self.task_queue.put(("flush", data_type))
                
            except Exception as e:
                logger.error(f"缓冲区监控失败: {e}")
    
    async def _process_buffer(self, data_type: str):
        """处理缓冲区数据"""
        try:
            # 获取缓冲区数据
            buffer_data = await self.buffers[data_type].flush()
            if not buffer_data:
                return
            
            # 批量处理数据
            processor = self.processors[data_type]
            processed_data = await processor.process_batch(buffer_data)
            
            # 更新总体指标
            self.pipeline_metrics["total_processed"] += len(processed_data)
            
            # 调用回调函数
            if processed_data and data_type in self.callbacks:
                for callback in self.callbacks[data_type]:
                    try:
                        await callback(processed_data)
                    except Exception as e:
                        logger.error(f"回调函数执行失败 [{data_type}]: {e}")
            
            # 计算吞吐量
            if self.pipeline_metrics["start_time"]:
                elapsed = (datetime.utcnow() - self.pipeline_metrics["start_time"]).total_seconds()
                if elapsed > 0:
                    self.pipeline_metrics["throughput"] = self.pipeline_metrics["total_processed"] / elapsed
            
            logger.debug(f"处理完成 [{data_type}]: {len(processed_data)}条数据")
            
        except Exception as e:
            logger.error(f"处理缓冲区数据失败 [{data_type}]: {e}")
            self.pipeline_metrics["total_errors"] += 1
    
    async def _flush_all_buffers(self):
        """刷新所有缓冲区"""
        for data_type in self.buffers.keys():
            try:
                await self._process_buffer(data_type)
            except Exception as e:
                logger.error(f"刷新缓冲区失败 [{data_type}]: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取流水线性能指标"""
        processor_metrics = {}
        for data_type, processor in self.processors.items():
            processor_metrics[data_type] = {
                "processed_count": processor.metrics.processed_count,
                "error_count": processor.metrics.error_count,
                "avg_processing_time": processor.metrics.avg_processing_time,
                "last_update": processor.metrics.last_update
            }
        
        # 添加智能清洗统计
        cleaning_stats = {}
        if CLEANING_ENABLED and intelligent_cleaner:
            cleaning_stats = intelligent_cleaner.get_cleaning_stats()
        
        return {
            "pipeline": self.pipeline_metrics,
            "processors": processor_metrics,
            "buffers": {
                data_type: buffer.size() 
                for data_type, buffer in self.buffers.items()
            },
            "cleaning": cleaning_stats,
            "running": self.running
        }

# 全局优化数据流水线实例
optimized_pipeline = OptimizedDataPipeline(
    buffer_size=1000,
    flush_interval=1.0,
    max_workers=10
)