"""
结构化日志系统
提供统一的、结构化的日志记录功能
支持多种输出格式、日志轮转、性能监控和分析
"""

import asyncio
import json
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import threading
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging
import logging.handlers
from enum import Enum
import motor.motor_asyncio
import redis.asyncio as aioredis
from loguru import logger
import uuid
import inspect

from config.settings import settings

class LogLevel(Enum):
    """日志级别"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

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
    USER_ACTION = "user_action"
    WORKFLOW = "workflow"

@dataclass
class LogRecord:
    """结构化日志记录"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    module: str
    function: str
    line: int
    thread_id: str
    process_id: int
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    exception: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    tags: Optional[List[str]] = None

class LogFormatter:
    """日志格式化器"""
    
    @staticmethod
    def format_json(record: LogRecord) -> str:
        """格式化为JSON"""
        record_dict = asdict(record)
        record_dict["timestamp"] = record.timestamp.isoformat()
        record_dict["level"] = record.level.value
        record_dict["category"] = record.category.value
        return json.dumps(record_dict, ensure_ascii=False)
    
    @staticmethod
    def format_text(record: LogRecord) -> str:
        """格式化为可读文本"""
        timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 基础信息
        base_info = f"[{timestamp}] {record.level.value:<8} {record.category.value:<12}"
        
        # 位置信息
        location = f"{record.module}:{record.function}:{record.line}"
        
        # 核心消息
        core_msg = f"{base_info} {location:<40} | {record.message}"
        
        # 附加信息
        extras = []
        if record.correlation_id:
            extras.append(f"correlation_id={record.correlation_id}")
        if record.user_id:
            extras.append(f"user_id={record.user_id}")
        if record.duration is not None:
            extras.append(f"duration={record.duration:.3f}s")
        if record.tags:
            extras.append(f"tags={','.join(record.tags)}")
        
        if extras:
            core_msg += f" [{' '.join(extras)}]"
        
        # 异常信息
        if record.exception:
            core_msg += f"\n  Exception: {record.exception['type']} - {record.exception['message']}"
            if record.exception.get('traceback'):
                core_msg += f"\n  Traceback: {record.exception['traceback']}"
        
        # 额外数据
        if record.extra_data:
            core_msg += f"\n  Extra: {json.dumps(record.extra_data, ensure_ascii=False, indent=2)}"
        
        return core_msg

class LogStorage:
    """日志存储"""
    
    def __init__(self):
        self.mongodb_client = None
        self.redis_client = None
        self.mongodb_db = None
        self._buffer = deque(maxlen=1000)  # 内存缓冲区
        self._buffer_lock = threading.Lock()
        self._flush_task = None
        self._initialized = False
    
    async def initialize(self):
        """初始化存储"""
        try:
            db_config = settings.get_database_config()
            
            # MongoDB连接
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                db_config["mongodb_url"],
                maxPoolSize=10,
                minPoolSize=2
            )
            self.mongodb_db = self.mongodb_client.get_default_database()
            
            # Redis连接
            self.redis_client = aioredis.from_url(
                db_config["redis_url"],
                max_connections=10
            )
            
            # 创建日志集合索引
            await self._setup_log_indexes()
            
            # 启动异步刷新任务
            self._flush_task = asyncio.create_task(self._flush_buffer_loop())
            
            self._initialized = True
            logger.debug("日志存储初始化成功")
            
        except Exception as e:
            logger.error(f"日志存储初始化失败: {e}")
    
    async def _setup_log_indexes(self):
        """设置日志集合索引"""
        try:
            # 创建日志索引
            await self.mongodb_db.logs.create_index("timestamp")
            await self.mongodb_db.logs.create_index("level")
            await self.mongodb_db.logs.create_index("category") 
            await self.mongodb_db.logs.create_index("module")
            await self.mongodb_db.logs.create_index("correlation_id")
            await self.mongodb_db.logs.create_index("user_id")
            await self.mongodb_db.logs.create_index([("timestamp", -1)])
            
            # TTL索引 - 30天后自动删除
            await self.mongodb_db.logs.create_index(
                "timestamp", 
                expireAfterSeconds=2592000
            )
            
            # 复合索引
            await self.mongodb_db.logs.create_index([
                ("level", 1),
                ("timestamp", -1)
            ])
            
            await self.mongodb_db.logs.create_index([
                ("category", 1),
                ("timestamp", -1)
            ])
            
        except Exception as e:
            logger.warning(f"创建日志索引失败: {e}")
    
    def store_log(self, record: LogRecord):
        """存储日志记录"""
        if not self._initialized:
            return
        
        with self._buffer_lock:
            self._buffer.append(record)
    
    async def _flush_buffer_loop(self):
        """异步刷新缓冲区"""
        try:
            while True:
                await asyncio.sleep(5)  # 每5秒刷新一次
                await self._flush_buffer()
        except asyncio.CancelledError:
            # 最后一次刷新
            await self._flush_buffer()
        except Exception as e:
            logger.error(f"日志缓冲区刷新异常: {e}")
    
    async def _flush_buffer(self):
        """刷新缓冲区到数据库"""
        if not self._initialized:
            return
        
        try:
            # 获取待刷新的记录
            records_to_flush = []
            with self._buffer_lock:
                if self._buffer:
                    records_to_flush = list(self._buffer)
                    self._buffer.clear()
            
            if not records_to_flush:
                return
            
            # 准备MongoDB文档
            mongo_docs = []
            for record in records_to_flush:
                doc = asdict(record)
                doc["level"] = record.level.value
                doc["category"] = record.category.value
                mongo_docs.append(doc)
            
            # 批量插入到MongoDB
            if mongo_docs:
                await self.mongodb_db.logs.insert_many(mongo_docs)
            
            # 更新Redis统计
            await self._update_log_stats(records_to_flush)
            
        except Exception as e:
            logger.error(f"刷新日志缓冲区失败: {e}")
    
    async def _update_log_stats(self, records: List[LogRecord]):
        """更新Redis中的日志统计"""
        try:
            if not records:
                return
            
            # 统计各级别日志数量
            level_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for record in records:
                level_counts[record.level.value] += 1
                category_counts[record.category.value] += 1
            
            # 更新Redis统计
            pipeline = self.redis_client.pipeline()
            
            for level, count in level_counts.items():
                pipeline.hincrby("log_stats:levels", level, count)
            
            for category, count in category_counts.items():
                pipeline.hincrby("log_stats:categories", category, count)
            
            # 更新总计数
            pipeline.hincrby("log_stats:total", "count", len(records))
            pipeline.hset("log_stats:total", "last_update", datetime.utcnow().isoformat())
            
            # 设置过期时间
            pipeline.expire("log_stats:levels", 86400)  # 1天
            pipeline.expire("log_stats:categories", 86400)
            pipeline.expire("log_stats:total", 86400)
            
            await pipeline.execute()
            
        except Exception as e:
            logger.warning(f"更新日志统计失败: {e}")
    
    async def search_logs(self, 
                         level: Optional[LogLevel] = None,
                         category: Optional[LogCategory] = None,
                         module: Optional[str] = None,
                         correlation_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        try:
            if not self._initialized:
                return []
            
            # 构建查询条件
            query = {}
            
            if level:
                query["level"] = level.value
            if category:
                query["category"] = category.value
            if module:
                query["module"] = {"$regex": module, "$options": "i"}
            if correlation_id:
                query["correlation_id"] = correlation_id
            if user_id:
                query["user_id"] = user_id
            
            # 时间范围
            if start_time or end_time:
                time_query = {}
                if start_time:
                    time_query["$gte"] = start_time
                if end_time:
                    time_query["$lte"] = end_time
                query["timestamp"] = time_query
            
            # 执行查询
            cursor = self.mongodb_db.logs.find(query).sort("timestamp", -1).limit(limit)
            results = []
            
            async for doc in cursor:
                doc.pop("_id", None)  # 移除MongoDB的_id字段
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索日志失败: {e}")
            return []
    
    async def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        try:
            if not self._initialized:
                return {}
            
            # 从Redis获取统计
            level_stats = await self.redis_client.hgetall("log_stats:levels")
            category_stats = await self.redis_client.hgetall("log_stats:categories")
            total_stats = await self.redis_client.hgetall("log_stats:total")
            
            # 转换数据类型
            level_stats = {k.decode(): int(v.decode()) for k, v in level_stats.items()}
            category_stats = {k.decode(): int(v.decode()) for k, v in category_stats.items()}
            total_count = int(total_stats.get(b"count", b"0").decode())
            last_update = total_stats.get(b"last_update", b"").decode()
            
            return {
                "level_distribution": level_stats,
                "category_distribution": category_stats,
                "total_logs": total_count,
                "last_update": last_update
            }
            
        except Exception as e:
            logger.error(f"获取日志统计失败: {e}")
            return {}
    
    async def close(self):
        """关闭存储"""
        try:
            # 停止刷新任务
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # 最后一次刷新
            await self._flush_buffer()
            
            # 关闭数据库连接
            if self.mongodb_client:
                self.mongodb_client.close()
            if self.redis_client:
                await self.redis_client.close()
            
        except Exception as e:
            logger.error(f"关闭日志存储失败: {e}")

class ContextualLogger:
    """上下文日志记录器"""
    
    def __init__(self, storage: LogStorage):
        self.storage = storage
        self._context_stack = []
        self._context_lock = threading.Lock()
        self._performance_timers = {}
        
        # 文件处理器
        self._setup_file_handlers()
    
    def _setup_file_handlers(self):
        """设置文件处理器"""
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # 删除现有的loguru处理器
            logger.remove()
            
            # 控制台处理器
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                level="INFO",
                colorize=True
            )
            
            # JSON文件处理器
            logger.add(
                logs_dir / "system_structured.json",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
                level="DEBUG",
                rotation="100 MB",
                retention="30 days",
                compression="gzip",
                serialize=True
            )
            
            # 错误文件处理器
            logger.add(
                logs_dir / "errors.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
                level="ERROR",
                rotation="10 MB",
                retention="30 days",
                compression="gzip"
            )
            
        except Exception as e:
            print(f"设置文件处理器失败: {e}")
    
    def _get_caller_info(self) -> Tuple[str, str, int]:
        """获取调用者信息"""
        try:
            frame = inspect.currentframe()
            # 回溯到实际调用位置
            for _ in range(3):  # 跳过当前函数、log方法和包装函数
                frame = frame.f_back
                if not frame:
                    break
            
            if frame:
                module = frame.f_globals.get("__name__", "unknown")
                function = frame.f_code.co_name
                line = frame.f_lineno
                return module, function, line
            else:
                return "unknown", "unknown", 0
                
        except Exception:
            return "unknown", "unknown", 0
    
    def _get_current_context(self) -> Dict[str, Any]:
        """获取当前上下文"""
        with self._context_lock:
            if not self._context_stack:
                return {}
            
            # 合并所有上下文
            merged_context = {}
            for context in self._context_stack:
                merged_context.update(context)
            
            return merged_context
    
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
    def performance_timer(self, operation_name: str, **context):
        """性能计时器"""
        timer_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        with self.context(operation=operation_name, timer_id=timer_id, **context):
            self.info(f"开始操作: {operation_name}")
            
            try:
                yield
                # 成功完成
                duration = (datetime.utcnow() - start_time).total_seconds()
                self.success(f"操作完成: {operation_name}", extra_data={"duration": duration})
                
            except Exception as e:
                # 操作失败
                duration = (datetime.utcnow() - start_time).total_seconds()
                self.error(f"操作失败: {operation_name}", 
                          exception=e,
                          extra_data={"duration": duration})
                raise
    
    def _create_log_record(self, 
                          level: LogLevel, 
                          category: LogCategory, 
                          message: str, 
                          exception: Optional[Exception] = None,
                          extra_data: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> LogRecord:
        """创建日志记录"""
        
        # 获取调用者信息
        module, function, line = self._get_caller_info()
        
        # 获取上下文
        context = self._get_current_context()
        
        # 处理异常信息
        exception_info = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # 合并额外数据
        final_extra_data = {}
        if extra_data:
            final_extra_data.update(extra_data)
        
        return LogRecord(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            message=message,
            module=module,
            function=function,
            line=line,
            thread_id=threading.current_thread().name,
            process_id=threading.current_thread().ident or 0,
            correlation_id=context.get("correlation_id"),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            request_id=context.get("request_id"),
            extra_data=final_extra_data if final_extra_data else None,
            exception=exception_info,
            tags=tags
        )
    
    def log(self, level: LogLevel, category: LogCategory, message: str, **kwargs):
        """通用日志记录方法"""
        try:
            record = self._create_log_record(level, category, message, **kwargs)
            
            # 存储到数据库
            self.storage.store_log(record)
            
            # 同时使用loguru输出
            loguru_level = level.value
            loguru_message = f"[{category.value}] {message}"
            
            # 添加额外信息到loguru消息
            if record.extra_data:
                loguru_message += f" | Extra: {json.dumps(record.extra_data, ensure_ascii=False)}"
            
            if record.exception:
                logger.opt(exception=kwargs.get('exception')).log(loguru_level, loguru_message)
            else:
                logger.log(loguru_level, loguru_message)
                
        except Exception as e:
            # 日志记录失败时使用基础logger
            logger.error(f"结构化日志记录失败: {e}")
            logger.log(level.value, f"[{category.value}] {message}")
    
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
    
    # 分类便捷方法
    def api_log(self, level: LogLevel, message: str, **kwargs):
        self.log(level, LogCategory.API, message, **kwargs)
    
    def trading_log(self, level: LogLevel, message: str, **kwargs):
        self.log(level, LogCategory.TRADING, message, **kwargs)
    
    def ai_log(self, level: LogLevel, message: str, **kwargs):
        self.log(level, LogCategory.AI, message, **kwargs)
    
    def db_log(self, level: LogLevel, message: str, **kwargs):
        self.log(level, LogCategory.DATABASE, message, **kwargs)
    
    def security_log(self, level: LogLevel, message: str, **kwargs):
        self.log(level, LogCategory.SECURITY, message, **kwargs)
    
    def performance_log(self, level: LogLevel, message: str, duration: float = None, **kwargs):
        extra_data = kwargs.get("extra_data", {})
        if duration is not None:
            extra_data["duration"] = duration
        kwargs["extra_data"] = extra_data
        self.log(level, LogCategory.PERFORMANCE, message, **kwargs)

class StructuredLoggingManager:
    """结构化日志管理器"""
    
    def __init__(self):
        self.storage = LogStorage()
        self.logger = None
        self._initialized = False
    
    async def initialize(self):
        """初始化日志管理器"""
        try:
            await self.storage.initialize()
            self.logger = ContextualLogger(self.storage)
            self._initialized = True
            
            # 记录初始化成功
            self.logger.success("结构化日志系统初始化成功", LogCategory.SYSTEM)
            
        except Exception as e:
            # 使用基础logger记录初始化失败
            logger.error(f"结构化日志系统初始化失败: {e}")
            raise
    
    def get_logger(self) -> ContextualLogger:
        """获取日志记录器"""
        if not self._initialized:
            raise RuntimeError("日志管理器未初始化")
        return self.logger
    
    async def search_logs(self, **kwargs) -> List[Dict[str, Any]]:
        """搜索日志"""
        return await self.storage.search_logs(**kwargs)
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return await self.storage.get_log_stats()
    
    async def close(self):
        """关闭日志管理器"""
        try:
            if self._initialized and self.logger:
                self.logger.info("结构化日志系统正在关闭", LogCategory.SYSTEM)
            
            await self.storage.close()
            self._initialized = False
            
        except Exception as e:
            logger.error(f"关闭结构化日志系统失败: {e}")

# 全局结构化日志管理器
structured_logging_manager = StructuredLoggingManager()

# 便捷函数
async def get_structured_logger() -> ContextualLogger:
    """获取结构化日志记录器"""
    if not structured_logging_manager._initialized:
        await structured_logging_manager.initialize()
    return structured_logging_manager.get_logger()

# 装饰器
def log_performance(category: LogCategory = LogCategory.PERFORMANCE, 
                   operation_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger_instance = await get_structured_logger()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with logger_instance.performance_timer(op_name):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # 对于同步函数，创建简单的计时
            import time
            start_time = time.time()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.success(f"操作完成: {op_name} (耗时: {duration:.3f}秒)")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"操作失败: {op_name} (耗时: {duration:.3f}秒) - {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def log_errors(category: LogCategory = LogCategory.SYSTEM):
    """错误日志装饰器"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger_instance = await get_structured_logger()
                logger_instance.error(
                    f"函数执行失败: {func.__module__}.{func.__name__}",
                    category=category,
                    exception=e,
                    extra_data={"args": str(args)[:500], "kwargs": str(kwargs)[:500]}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"函数执行失败: {func.__module__}.{func.__name__} - {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator