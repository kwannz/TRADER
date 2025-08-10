"""
统一日志系统 - 全面的日志解决方案
整合所有现有日志功能，提供统一、高性能、功能完整的日志系统
"""

import asyncio
import json
import os
import sys
import traceback
import threading
import uuid
import time
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from collections import deque, defaultdict
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor

# 第三方依赖
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import motor.motor_asyncio
    import redis.asyncio as aioredis
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


class LogLevel(Enum):
    """日志级别"""
    TRACE = "TRACE"
    DEBUG = "DEBUG" 
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"


class LogCategory(Enum):
    """日志分类"""
    SYSTEM = "system"
    API = "api"
    TRADING = "trading"
    AI = "ai"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER = "user"
    WORKFLOW = "workflow"
    COINGLASS = "coinglass"
    BACKTEST = "backtest"
    STRATEGY = "strategy"
    RISK = "risk"
    MONITORING = "monitoring"


class LogOutput(Enum):
    """日志输出方式"""
    CONSOLE = "console"
    FILE = "file"
    JSON_FILE = "json_file"
    DATABASE = "database"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    WEBHOOK = "webhook"


@dataclass
class LogMetadata:
    """日志元数据"""
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    hostname: Optional[str] = None
    service_name: Optional[str] = None
    version: Optional[str] = None


@dataclass 
class PerformanceMetrics:
    """性能指标"""
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    io_reads: Optional[int] = None
    io_writes: Optional[int] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_received: Optional[int] = None


@dataclass
class LogRecord:
    """统一日志记录"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line: int
    filename: str
    metadata: LogMetadata = field(default_factory=LogMetadata)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    extra_data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    exception_info: Optional[Dict[str, Any]] = None
    stack_trace: Optional[List[str]] = None


class LogFilter:
    """日志过滤器"""
    
    def __init__(self,
                 min_level: Optional[LogLevel] = None,
                 max_level: Optional[LogLevel] = None,
                 categories: Optional[List[LogCategory]] = None,
                 modules: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None):
        self.min_level = min_level
        self.max_level = max_level
        self.categories = categories or []
        self.modules = modules or []
        self.tags = tags or []
    
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        # 级别过滤
        if self.min_level and record.level.value < self.min_level.value:
            return False
        if self.max_level and record.level.value > self.max_level.value:
            return False
        
        # 分类过滤
        if self.categories and record.category not in self.categories:
            return False
        
        # 模块过滤
        if self.modules and not any(module in record.module for module in self.modules):
            return False
        
        # 标签过滤
        if self.tags and record.tags:
            if not any(tag in record.tags for tag in self.tags):
                return False
        
        return True


class LogFormatter:
    """统一日志格式化器"""
    
    @staticmethod
    def format_console(record: LogRecord, colorize: bool = True) -> str:
        """格式化控制台输出"""
        timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 颜色映射
        color_map = {
            LogLevel.TRACE: "\033[90m",     # 灰色
            LogLevel.DEBUG: "\033[36m",     # 青色
            LogLevel.INFO: "\033[32m",      # 绿色
            LogLevel.SUCCESS: "\033[92m",   # 亮绿色
            LogLevel.WARNING: "\033[33m",   # 黄色
            LogLevel.ERROR: "\033[31m",     # 红色
            LogLevel.CRITICAL: "\033[95m",  # 品红色
            LogLevel.FATAL: "\033[41m",     # 红色背景
        }
        reset = "\033[0m"
        
        if colorize and record.level in color_map:
            level_colored = f"{color_map[record.level]}{record.level.value:<8}{reset}"
        else:
            level_colored = f"{record.level.value:<8}"
        
        # 基础信息
        base_info = f"[{timestamp}] {level_colored} {record.category.value:<12}"
        
        # 位置信息
        location = f"{record.module}:{record.function}:{record.line}"
        
        # 构建消息
        msg = f"{base_info} {location:<50} | {record.message}"
        
        # 添加元数据
        metadata_parts = []
        if record.metadata.correlation_id:
            metadata_parts.append(f"correlation_id={record.metadata.correlation_id}")
        if record.metadata.user_id:
            metadata_parts.append(f"user_id={record.metadata.user_id}")
        if record.performance.execution_time_ms is not None:
            metadata_parts.append(f"duration={record.performance.execution_time_ms:.2f}ms")
        if record.tags:
            metadata_parts.append(f"tags={','.join(record.tags)}")
        
        if metadata_parts:
            msg += f" [{' | '.join(metadata_parts)}]"
        
        # 异常信息
        if record.exception_info:
            msg += f"\n  Exception: {record.exception_info['type']} - {record.exception_info['message']}"
        
        # 额外数据
        if record.extra_data:
            msg += f"\n  Extra: {json.dumps(record.extra_data, ensure_ascii=False, indent=2)}"
        
        return msg
    
    @staticmethod
    def format_json(record: LogRecord) -> str:
        """格式化为JSON"""
        record_dict = asdict(record)
        record_dict["timestamp"] = record.timestamp.isoformat()
        record_dict["level"] = record.level.value
        record_dict["category"] = record.category.value
        return json.dumps(record_dict, ensure_ascii=False, separators=(',', ':'))
    
    @staticmethod
    def format_structured(record: LogRecord) -> Dict[str, Any]:
        """格式化为结构化数据"""
        return {
            "@timestamp": record.timestamp.isoformat(),
            "level": record.level.value,
            "category": record.category.value,
            "message": record.message,
            "location": {
                "module": record.module,
                "function": record.function,
                "line": record.line,
                "filename": record.filename
            },
            "metadata": asdict(record.metadata),
            "performance": asdict(record.performance),
            "extra_data": record.extra_data,
            "tags": record.tags,
            "exception": record.exception_info,
            "stack_trace": record.stack_trace
        }


class LogOutput:
    """日志输出处理器基类"""
    
    def __init__(self, name: str, formatter: Callable[[LogRecord], str] = None):
        self.name = name
        self.formatter = formatter or LogFormatter.format_console
        self.filter = LogFilter()
        self.enabled = True
    
    async def write(self, record: LogRecord):
        """写入日志记录"""
        raise NotImplementedError
    
    async def flush(self):
        """刷新缓冲区"""
        pass
    
    async def close(self):
        """关闭输出处理器"""
        pass


class ConsoleOutput(LogOutput):
    """控制台输出处理器"""
    
    def __init__(self, colorize: bool = True):
        super().__init__("console")
        self.colorize = colorize
        self.formatter = lambda record: LogFormatter.format_console(record, self.colorize)
    
    async def write(self, record: LogRecord):
        if not self.filter.should_log(record):
            return
        
        formatted = self.formatter(record)
        print(formatted, flush=True)


class FileOutput(LogOutput):
    """文件输出处理器"""
    
    def __init__(self, 
                 filepath: Union[str, Path],
                 max_bytes: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5,
                 encoding: str = "utf-8"):
        super().__init__("file")
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        self.handler = logging.handlers.RotatingFileHandler(
            self.filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        # 设置格式
        formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        
        self.formatter = LogFormatter.format_console
    
    async def write(self, record: LogRecord):
        if not self.filter.should_log(record):
            return
        
        formatted = self.formatter(record)
        
        # 创建日志记录
        log_record = logging.LogRecord(
            name=record.module,
            level=self._get_logging_level(record.level),
            pathname=record.filename,
            lineno=record.line,
            msg=formatted,
            args=(),
            exc_info=None
        )
        
        self.handler.emit(log_record)
    
    def _get_logging_level(self, level: LogLevel) -> int:
        """转换日志级别"""
        level_map = {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.SUCCESS: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.FATAL: logging.CRITICAL
        }
        return level_map.get(level, logging.INFO)
    
    async def flush(self):
        self.handler.flush()
    
    async def close(self):
        self.handler.close()


class JsonFileOutput(LogOutput):
    """JSON文件输出处理器"""
    
    def __init__(self, 
                 filepath: Union[str, Path],
                 max_bytes: int = 100 * 1024 * 1024,
                 backup_count: int = 5):
        super().__init__("json_file")
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.handler = logging.handlers.RotatingFileHandler(
            self.filepath,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        
        formatter = logging.Formatter('%(message)s')
        self.handler.setFormatter(formatter)
        
        self.formatter = LogFormatter.format_json
    
    async def write(self, record: LogRecord):
        if not self.filter.should_log(record):
            return
        
        formatted = self.formatter(record)
        
        log_record = logging.LogRecord(
            name=record.module,
            level=logging.INFO,
            pathname=record.filename,
            lineno=record.line,
            msg=formatted,
            args=(),
            exc_info=None
        )
        
        self.handler.emit(log_record)


class DatabaseOutput(LogOutput):
    """数据库输出处理器"""
    
    def __init__(self, mongodb_url: str, redis_url: str):
        super().__init__("database")
        self.mongodb_url = mongodb_url
        self.redis_url = redis_url
        self.mongodb_client = None
        self.redis_client = None
        self.db = None
        self._buffer = deque(maxlen=1000)
        self._flush_task = None
        self._initialized = False
    
    async def initialize(self):
        """初始化数据库连接"""
        if not DATABASE_AVAILABLE:
            return
        
        try:
            # MongoDB连接
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongodb_client.get_default_database()
            
            # Redis连接
            self.redis_client = aioredis.from_url(self.redis_url)
            
            # 创建索引
            await self._setup_indexes()
            
            # 启动刷新任务
            self._flush_task = asyncio.create_task(self._flush_loop())
            
            self._initialized = True
            
        except Exception as e:
            print(f"数据库日志输出初始化失败: {e}")
    
    async def _setup_indexes(self):
        """设置数据库索引"""
        try:
            await self.db.logs.create_index("timestamp")
            await self.db.logs.create_index("level")
            await self.db.logs.create_index("category")
            await self.db.logs.create_index("module")
            await self.db.logs.create_index([("timestamp", -1)])
            await self.db.logs.create_index("timestamp", expireAfterSeconds=2592000)  # 30天
        except Exception as e:
            print(f"创建数据库索引失败: {e}")
    
    async def write(self, record: LogRecord):
        if not self._initialized or not self.filter.should_log(record):
            return
        
        self._buffer.append(record)
    
    async def _flush_loop(self):
        """定期刷新缓冲区"""
        try:
            while True:
                await asyncio.sleep(5)
                await self._flush_buffer()
        except asyncio.CancelledError:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """刷新缓冲区到数据库"""
        if not self._buffer:
            return
        
        try:
            # 获取待写入记录
            records = list(self._buffer)
            self._buffer.clear()
            
            # 转换为字典格式
            docs = [LogFormatter.format_structured(record) for record in records]
            
            # 批量写入MongoDB
            if docs:
                await self.db.logs.insert_many(docs)
            
            # 更新Redis统计
            await self._update_redis_stats(records)
            
        except Exception as e:
            print(f"刷新数据库缓冲区失败: {e}")
    
    async def _update_redis_stats(self, records: List[LogRecord]):
        """更新Redis统计"""
        try:
            pipeline = self.redis_client.pipeline()
            
            level_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for record in records:
                level_counts[record.level.value] += 1
                category_counts[record.category.value] += 1
            
            # 更新统计
            for level, count in level_counts.items():
                pipeline.hincrby("log_stats:levels", level, count)
            
            for category, count in category_counts.items():
                pipeline.hincrby("log_stats:categories", category, count)
            
            pipeline.hincrby("log_stats:total", "count", len(records))
            pipeline.hset("log_stats:total", "last_update", datetime.utcnow().isoformat())
            
            await pipeline.execute()
            
        except Exception as e:
            print(f"更新Redis统计失败: {e}")
    
    async def close(self):
        """关闭数据库连接"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self._flush_buffer()
        
        if self.mongodb_client:
            self.mongodb_client.close()
        if self.redis_client:
            await self.redis_client.close()


class UnifiedLogger:
    """统一日志记录器"""
    
    def __init__(self, name: str = "unified_logger"):
        self.name = name
        self.outputs: List[LogOutput] = []
        self._context_stack = []
        self._context_lock = threading.Lock()
        self._performance_timers = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # 默认元数据
        self.default_metadata = LogMetadata(
            hostname=self._get_hostname(),
            service_name=self.name,
            version=self._get_version()
        )
    
    def _get_hostname(self) -> str:
        """获取主机名"""
        try:
            import socket
            return socket.gethostname()
        except:
            return "unknown"
    
    def _get_version(self) -> str:
        """获取版本信息"""
        try:
            # 可以从环境变量或配置文件读取
            return os.getenv("APP_VERSION", "1.0.0")
        except:
            return "unknown"
    
    def _get_memory_usage(self) -> Optional[float]:
        """获取内存使用量（MB）"""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return None
    
    def _get_cpu_usage(self) -> Optional[float]:
        """获取CPU使用率"""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            return psutil.cpu_percent()
        except:
            return None
    
    def _get_caller_info(self, skip_frames: int = 3) -> Tuple[str, str, int, str]:
        """获取调用者信息"""
        try:
            frame = inspect.currentframe()
            for _ in range(skip_frames):
                frame = frame.f_back
                if not frame:
                    break
            
            if frame:
                filename = frame.f_code.co_filename
                module = frame.f_globals.get("__name__", "unknown")
                function = frame.f_code.co_name
                line = frame.f_lineno
                return module, function, line, filename
            else:
                return "unknown", "unknown", 0, "unknown"
                
        except:
            return "unknown", "unknown", 0, "unknown"
    
    def _get_current_context(self) -> LogMetadata:
        """获取当前上下文"""
        with self._context_lock:
            if not self._context_stack:
                return self.default_metadata
            
            # 合并上下文
            merged = asdict(self.default_metadata)
            
            # LogMetadata的有效字段
            valid_fields = {field.name for field in LogMetadata.__dataclass_fields__.values()}
            
            for context in self._context_stack:
                for k, v in context.items():
                    if v is not None and k in valid_fields:
                        merged[k] = v
            
            return LogMetadata(**merged)
    
    @contextmanager
    def context(self, **kwargs):
        """上下文管理器"""
        with self._context_lock:
            self._context_stack.append(kwargs)
        
        try:
            yield
        finally:
            with self._context_lock:
                if self._context_stack:
                    self._context_stack.pop()
    
    @contextmanager
    def performance_context(self, operation: str, **context_kwargs):
        """性能监控上下文"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # 将操作信息加入额外数据，而不是上下文
        extra_context_data = {"operation": operation}
        extra_context_data.update(context_kwargs)
        
        with self.context(**context_kwargs):
            try:
                yield
                # 成功完成
                duration = (time.time() - start_time) * 1000
                memory_usage = self._get_memory_usage()
                
                self.success(
                    f"操作完成: {operation}",
                    category=LogCategory.PERFORMANCE,
                    extra_data=extra_context_data,
                    performance=PerformanceMetrics(
                        execution_time_ms=duration,
                        memory_usage_mb=memory_usage
                    )
                )
                
            except Exception as e:
                # 操作失败
                duration = (time.time() - start_time) * 1000
                memory_usage = self._get_memory_usage()
                
                self.error(
                    f"操作失败: {operation}",
                    category=LogCategory.PERFORMANCE,
                    exception=e,
                    extra_data=extra_context_data,
                    performance=PerformanceMetrics(
                        execution_time_ms=duration,
                        memory_usage_mb=memory_usage
                    )
                )
                raise
    
    def add_output(self, output: LogOutput):
        """添加日志输出处理器"""
        self.outputs.append(output)
    
    def remove_output(self, output_name: str):
        """移除日志输出处理器"""
        self.outputs = [o for o in self.outputs if o.name != output_name]
    
    def _create_log_record(self,
                          level: LogLevel,
                          category: LogCategory,
                          message: str,
                          exception: Optional[Exception] = None,
                          extra_data: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None,
                          performance: Optional[PerformanceMetrics] = None) -> LogRecord:
        """创建日志记录"""
        
        # 获取调用信息
        module, function, line, filename = self._get_caller_info()
        
        # 获取上下文
        metadata = self._get_current_context()
        metadata.thread_id = threading.current_thread().name
        metadata.process_id = os.getpid()
        
        # 处理异常信息
        exception_info = None
        stack_trace = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exception_only(type(exception), exception)
            }
            stack_trace = traceback.format_exc().split('\n')
        
        # 性能指标
        if performance is None:
            performance = PerformanceMetrics(
                memory_usage_mb=self._get_memory_usage(),
                cpu_usage_percent=self._get_cpu_usage()
            )
        
        return LogRecord(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            message=message,
            module=module,
            function=function,
            line=line,
            filename=filename,
            metadata=metadata,
            performance=performance,
            extra_data=extra_data,
            tags=tags,
            exception_info=exception_info,
            stack_trace=stack_trace
        )
    
    async def _write_to_outputs(self, record: LogRecord):
        """写入到所有输出处理器"""
        write_tasks = []
        
        for output in self.outputs:
            if output.enabled:
                try:
                    task = asyncio.create_task(output.write(record))
                    write_tasks.append(task)
                except Exception as e:
                    # 输出处理器错误不应该影响主程序
                    print(f"日志输出处理器 {output.name} 写入失败: {e}")
        
        if write_tasks:
            try:
                await asyncio.gather(*write_tasks, return_exceptions=True)
            except Exception as e:
                print(f"批量写入日志失败: {e}")
    
    def log(self, level: LogLevel, category: LogCategory, message: str, **kwargs):
        """记录日志"""
        try:
            record = self._create_log_record(level, category, message, **kwargs)
            
            # 异步写入
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._write_to_outputs(record))
                else:
                    loop.run_until_complete(self._write_to_outputs(record))
            except RuntimeError:
                # 没有事件循环，使用线程池
                self._executor.submit(
                    asyncio.run, 
                    self._write_to_outputs(record)
                )
                
        except Exception as e:
            # 日志记录失败时的降级方案
            print(f"日志记录失败: {e} | 原始消息: {message}")
    
    # 便捷方法
    def trace(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        self.log(LogLevel.TRACE, category, message, **kwargs)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        self.log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        self.log(LogLevel.INFO, category, message, **kwargs)
    
    def success(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        self.log(LogLevel.SUCCESS, category, message, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        self.log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
              exception: Optional[Exception] = None, **kwargs):
        self.log(LogLevel.ERROR, category, message, exception=exception, **kwargs)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                exception: Optional[Exception] = None, **kwargs):
        self.log(LogLevel.CRITICAL, category, message, exception=exception, **kwargs)
    
    def fatal(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             exception: Optional[Exception] = None, **kwargs):
        self.log(LogLevel.FATAL, category, message, exception=exception, **kwargs)
    
    # 分类便捷方法
    def api_info(self, message: str, **kwargs):
        self.info(message, LogCategory.API, **kwargs)
    
    def api_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        self.error(message, LogCategory.API, exception=exception, **kwargs)
    
    def trading_info(self, message: str, **kwargs):
        self.info(message, LogCategory.TRADING, **kwargs)
    
    def trading_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        self.error(message, LogCategory.TRADING, exception=exception, **kwargs)
    
    def ai_info(self, message: str, **kwargs):
        self.info(message, LogCategory.AI, **kwargs)
    
    def ai_error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        self.error(message, LogCategory.AI, exception=exception, **kwargs)
    
    def security_warning(self, message: str, **kwargs):
        self.warning(message, LogCategory.SECURITY, **kwargs)
    
    def security_critical(self, message: str, **kwargs):
        self.critical(message, LogCategory.SECURITY, **kwargs)
    
    def performance_info(self, message: str, duration: Optional[float] = None, **kwargs):
        performance = kwargs.get("performance", PerformanceMetrics())
        if duration is not None:
            performance.execution_time_ms = duration
        kwargs["performance"] = performance
        self.info(message, LogCategory.PERFORMANCE, **kwargs)
    
    async def flush_all(self):
        """刷新所有输出缓冲区"""
        flush_tasks = []
        for output in self.outputs:
            if hasattr(output, 'flush'):
                flush_tasks.append(asyncio.create_task(output.flush()))
        
        if flush_tasks:
            await asyncio.gather(*flush_tasks, return_exceptions=True)
    
    async def close(self):
        """关闭日志记录器"""
        try:
            await self.flush_all()
            
            close_tasks = []
            for output in self.outputs:
                if hasattr(output, 'close'):
                    close_tasks.append(asyncio.create_task(output.close()))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self._executor.shutdown(wait=True)
            
        except Exception as e:
            print(f"关闭日志记录器失败: {e}")


# 日志装饰器
def log_performance(category: LogCategory = LogCategory.PERFORMANCE,
                   operation_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with get_logger().performance_context(op_name):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with get_logger().performance_context(op_name):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def log_errors(category: LogCategory = LogCategory.SYSTEM,
               reraise: bool = True):
    """错误日志装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                get_logger().error(
                    f"函数执行失败: {func.__module__}.{func.__name__}",
                    category=category,
                    exception=e,
                    extra_data={"function_args": str(args)[:500]}
                )
                if reraise:
                    raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                get_logger().error(
                    f"函数执行失败: {func.__module__}.{func.__name__}",
                    category=category,
                    exception=e,
                    extra_data={"function_args": str(args)[:500]}
                )
                if reraise:
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# 全局日志实例
_global_logger: Optional[UnifiedLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> UnifiedLogger:
    """获取全局日志实例"""
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = create_default_logger()
    
    return _global_logger


def create_default_logger() -> UnifiedLogger:
    """创建默认配置的日志记录器"""
    logger = UnifiedLogger("trader_system")
    
    # 添加控制台输出
    console_output = ConsoleOutput(colorize=True)
    logger.add_output(console_output)
    
    # 添加文件输出
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 系统日志文件
    file_output = FileOutput(
        logs_dir / "system.log",
        max_bytes=50 * 1024 * 1024,  # 50MB
        backup_count=5
    )
    logger.add_output(file_output)
    
    # JSON结构化日志文件
    json_output = JsonFileOutput(
        logs_dir / "system_structured.jsonl",
        max_bytes=50 * 1024 * 1024,
        backup_count=5
    )
    logger.add_output(json_output)
    
    # 错误专用日志文件
    error_output = FileOutput(
        logs_dir / "errors.log",
        max_bytes=10 * 1024 * 1024,  # 10MB
        backup_count=10
    )
    error_output.filter = LogFilter(min_level=LogLevel.ERROR)
    logger.add_output(error_output)
    
    return logger


def setup_database_logging(mongodb_url: str, redis_url: str):
    """设置数据库日志"""
    logger = get_logger()
    
    db_output = DatabaseOutput(mongodb_url, redis_url)
    asyncio.create_task(db_output.initialize())
    logger.add_output(db_output)


# 兼容性函数
def setup_logger(name: str = "trader", level: str = "INFO") -> UnifiedLogger:
    """兼容旧版本的setup_logger函数"""
    return get_logger()


# 模块级别日志实例
logger = get_logger()