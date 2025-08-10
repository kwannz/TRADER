"""
增强日志系统
提供结构化日志、错误分级、自动告警和日志分析功能
"""

import asyncio
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import logging.handlers
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertSeverity(Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    """结构化日志条目"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    exception_info: Optional[Dict] = None

@dataclass
class ErrorAlert:
    """错误告警"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    error_type: str
    error_message: str
    occurrence_count: int = 1
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

class LogBuffer:
    """日志缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: List[LogEntry] = []
        self.lock = asyncio.Lock()
    
    async def add(self, log_entry: LogEntry):
        """添加日志条目"""
        async with self.lock:
            self.buffer.append(log_entry)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)
    
    async def get_logs(self, 
                      level: Optional[LogLevel] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      limit: int = 100) -> List[LogEntry]:
        """获取日志条目"""
        async with self.lock:
            filtered_logs = self.buffer.copy()
        
        # 按级别过滤
        if level:
            filtered_logs = [log for log in filtered_logs if log.level == level]
        
        # 按时间范围过滤
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # 按时间倒序排列并限制数量
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]
    
    async def clear(self):
        """清空缓冲区"""
        async with self.lock:
            self.buffer.clear()

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.active_alerts: Dict[str, ErrorAlert] = {}
        self.alert_history: List[ErrorAlert] = []
        self.max_history = 1000
        self.alert_callbacks: List[Callable] = []
        
        # 告警规则配置
        self.alert_rules = {
            "error_rate": {"threshold": 10, "window_minutes": 5},  # 5分钟内超过10个错误
            "critical_errors": {"threshold": 1, "window_minutes": 1},  # 1分钟内1个严重错误
            "memory_usage": {"threshold": 80, "window_minutes": 5},  # 内存使用率超过80%
            "response_time": {"threshold": 5000, "window_minutes": 5}  # 响应时间超过5秒
        }
    
    def register_alert_callback(self, callback: Callable[[ErrorAlert], None]):
        """注册告警回调"""
        self.alert_callbacks.append(callback)
    
    async def process_log_entry(self, log_entry: LogEntry):
        """处理日志条目，检查是否需要告警"""
        try:
            # 错误级别告警
            if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                await self._handle_error_alert(log_entry)
            
            # 性能告警
            if log_entry.execution_time_ms and log_entry.execution_time_ms > self.alert_rules["response_time"]["threshold"]:
                await self._handle_performance_alert(log_entry, "high_response_time")
            
            # 内存使用告警
            if log_entry.memory_usage_mb and log_entry.memory_usage_mb > self.alert_rules["memory_usage"]["threshold"]:
                await self._handle_performance_alert(log_entry, "high_memory_usage")
                
        except Exception as e:
            print(f"处理告警时发生错误: {e}")
    
    async def _handle_error_alert(self, log_entry: LogEntry):
        """处理错误告警"""
        error_type = log_entry.exception_info.get("type", "UnknownError") if log_entry.exception_info else "GeneralError"
        alert_key = f"{log_entry.logger_name}:{error_type}"
        
        # 确定告警严重程度
        if log_entry.level == LogLevel.CRITICAL:
            severity = AlertSeverity.CRITICAL
        elif "timeout" in log_entry.message.lower() or "connection" in log_entry.message.lower():
            severity = AlertSeverity.HIGH
        elif "warning" in log_entry.message.lower():
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.HIGH
        
        if alert_key in self.active_alerts:
            # 更新现有告警
            alert = self.active_alerts[alert_key]
            alert.occurrence_count += 1
            alert.last_occurrence = log_entry.timestamp
        else:
            # 创建新告警
            alert = ErrorAlert(
                alert_id=f"alert_{int(log_entry.timestamp.timestamp())}_{hash(alert_key) % 10000}",
                timestamp=log_entry.timestamp,
                severity=severity,
                component=log_entry.logger_name,
                error_type=error_type,
                error_message=log_entry.message,
                first_occurrence=log_entry.timestamp,
                last_occurrence=log_entry.timestamp,
                context={
                    "module": log_entry.module,
                    "function": log_entry.function,
                    "line_number": log_entry.line_number,
                    "extra_data": log_entry.extra_data
                }
            )
            
            self.active_alerts[alert_key] = alert
            
            # 触发告警回调
            await self._trigger_alert_callbacks(alert)
    
    async def _handle_performance_alert(self, log_entry: LogEntry, alert_type: str):
        """处理性能告警"""
        alert_key = f"{log_entry.logger_name}:{alert_type}"
        
        if alert_key not in self.active_alerts:
            alert = ErrorAlert(
                alert_id=f"perf_{int(log_entry.timestamp.timestamp())}_{hash(alert_key) % 10000}",
                timestamp=log_entry.timestamp,
                severity=AlertSeverity.MEDIUM,
                component=log_entry.logger_name,
                error_type=alert_type,
                error_message=f"性能告警: {alert_type}",
                context={
                    "execution_time_ms": log_entry.execution_time_ms,
                    "memory_usage_mb": log_entry.memory_usage_mb,
                    "extra_data": log_entry.extra_data
                }
            )
            
            self.active_alerts[alert_key] = alert
            await self._trigger_alert_callbacks(alert)
    
    async def _trigger_alert_callbacks(self, alert: ErrorAlert):
        """触发告警回调"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                print(f"告警回调执行失败: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.active_alerts.values():
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.alert_history.append(alert)
                
                # 从活跃告警中移除
                key_to_remove = None
                for key, active_alert in self.active_alerts.items():
                    if active_alert.alert_id == alert_id:
                        key_to_remove = key
                        break
                
                if key_to_remove:
                    del self.active_alerts[key_to_remove]
                
                # 限制历史记录大小
                if len(self.alert_history) > self.max_history:
                    self.alert_history.pop(0)
                
                break
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[ErrorAlert]:
        """获取活跃告警"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_stats(self, hours: int = 24) -> Dict[str, Any]:
        """获取告警统计"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ] + list(self.active_alerts.values())
        
        severity_counts = {}
        component_counts = {}
        
        for alert in recent_alerts:
            # 按严重程度统计
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # 按组件统计
            component = alert.component
            component_counts[component] = component_counts.get(component, 0) + 1
        
        return {
            "time_range_hours": hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len([a for a in recent_alerts if a.resolved]),
            "severity_breakdown": severity_counts,
            "component_breakdown": component_counts,
            "alert_rate": len(recent_alerts) / max(hours, 1)
        }

class EnhancedLogger:
    """增强日志记录器"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 创建标准logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        
        # 文件处理器（按大小轮转）
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # JSON文件处理器（结构化日志）
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}_structured.json",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        self.logger.addHandler(json_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 日志缓冲区和告警管理器
        self.log_buffer = LogBuffer()
        self.alert_manager = AlertManager()
        
        # 性能监控
        self.execution_times: Dict[str, float] = {}
        self.memory_tracker = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _create_log_entry(self, level: LogLevel, message: str, **kwargs) -> LogEntry:
        """创建日志条目"""
        # 获取调用信息
        frame = sys._getframe(3)  # 跳过内部调用层级
        
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            logger_name=self.name,
            message=message,
            module=frame.f_globals.get('__name__'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            memory_usage_mb=self._get_memory_usage(),
            extra_data=kwargs.get('extra', {}),
            tags=kwargs.get('tags', []),
            user_id=kwargs.get('user_id'),
            session_id=kwargs.get('session_id'),
            request_id=kwargs.get('request_id'),
            execution_time_ms=kwargs.get('execution_time_ms'),
            exception_info=kwargs.get('exception_info')
        )
    
    async def _log_async(self, level: LogLevel, message: str, **kwargs):
        """异步日志记录"""
        # 创建结构化日志条目
        log_entry = self._create_log_entry(level, message, **kwargs)
        
        # 添加到缓冲区
        await self.log_buffer.add(log_entry)
        
        # 处理告警
        await self.alert_manager.process_log_entry(log_entry)
        
        # 写入结构化日志文件
        json_log = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level.value,
            "logger": log_entry.logger_name,
            "message": log_entry.message,
            "module": log_entry.module,
            "function": log_entry.function,
            "line": log_entry.line_number,
            "memory_mb": log_entry.memory_usage_mb,
            "execution_time_ms": log_entry.execution_time_ms,
            "extra": log_entry.extra_data,
            "tags": log_entry.tags
        }
        
        if log_entry.exception_info:
            json_log["exception"] = log_entry.exception_info
        
        # 找到JSON处理器并写入
        for handler in self.logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler) and "structured" in handler.baseFilename:
                handler.emit(logging.LogRecord(
                    name=self.name,
                    level=getattr(logging, level.value),
                    pathname="",
                    lineno=0,
                    msg=json.dumps(json_log, ensure_ascii=False),
                    args=(),
                    exc_info=None
                ))
                break
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message)
        asyncio.create_task(self._log_async(LogLevel.DEBUG, message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message)
        asyncio.create_task(self._log_async(LogLevel.INFO, message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message)
        asyncio.create_task(self._log_async(LogLevel.WARNING, message, **kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """错误日志"""
        self.logger.error(message)
        
        # 处理异常信息
        exception_info = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
            }
        
        kwargs["exception_info"] = exception_info
        asyncio.create_task(self._log_async(LogLevel.ERROR, message, **kwargs))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """严重错误日志"""
        self.logger.critical(message)
        
        exception_info = None
        if exception:
            exception_info = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exception(type(exception), exception, exception.__traceback__)
            }
        
        kwargs["exception_info"] = exception_info
        asyncio.create_task(self._log_async(LogLevel.CRITICAL, message, **kwargs))
    
    async def log_performance(self, operation_name: str, execution_time_ms: float, **kwargs):
        """记录性能日志"""
        message = f"性能监控: {operation_name} 耗时 {execution_time_ms:.2f}ms"
        kwargs["execution_time_ms"] = execution_time_ms
        kwargs["tags"] = kwargs.get("tags", []) + ["performance"]
        
        await self._log_async(LogLevel.INFO, message, **kwargs)
    
    def start_timer(self, operation_name: str) -> str:
        """开始计时"""
        timer_key = f"{operation_name}_{int(datetime.utcnow().timestamp() * 1000)}"
        self.execution_times[timer_key] = time.time()
        return timer_key
    
    async def end_timer(self, timer_key: str, operation_name: str, **kwargs):
        """结束计时并记录"""
        if timer_key in self.execution_times:
            start_time = self.execution_times.pop(timer_key)
            execution_time_ms = (time.time() - start_time) * 1000
            await self.log_performance(operation_name, execution_time_ms, **kwargs)
    
    async def get_recent_logs(self, level: Optional[LogLevel] = None, hours: int = 1, limit: int = 100) -> List[Dict]:
        """获取最近日志"""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        log_entries = await self.log_buffer.get_logs(level=level, start_time=start_time, limit=limit)
        
        return [asdict(entry) for entry in log_entries]
    
    async def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误摘要"""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        error_logs = await self.log_buffer.get_logs(level=LogLevel.ERROR, start_time=start_time)
        critical_logs = await self.log_buffer.get_logs(level=LogLevel.CRITICAL, start_time=start_time)
        
        all_errors = error_logs + critical_logs
        
        # 按错误类型分组
        error_types = {}
        for error in all_errors:
            error_type = error.exception_info.get("type", "UnknownError") if error.exception_info else "GeneralError"
            if error_type not in error_types:
                error_types[error_type] = {"count": 0, "latest_message": "", "latest_time": None}
            
            error_types[error_type]["count"] += 1
            if not error_types[error_type]["latest_time"] or error.timestamp > error_types[error_type]["latest_time"]:
                error_types[error_type]["latest_message"] = error.message
                error_types[error_type]["latest_time"] = error.timestamp
        
        return {
            "time_range_hours": hours,
            "total_errors": len(error_logs),
            "total_critical": len(critical_logs),
            "error_rate": len(all_errors) / max(hours, 1),
            "error_types": error_types,
            "alert_stats": self.alert_manager.get_alert_stats(hours)
        }
    
    def register_alert_callback(self, callback: Callable):
        """注册告警回调"""
        self.alert_manager.register_alert_callback(callback)

# 创建增强日志记录器工厂
def create_enhanced_logger(name: str, log_dir: str = "logs") -> EnhancedLogger:
    """创建增强日志记录器"""
    return EnhancedLogger(name, log_dir)

# 全局日志记录器实例
system_logger = create_enhanced_logger("system")
trading_logger = create_enhanced_logger("trading") 
data_logger = create_enhanced_logger("data_processing")
api_logger = create_enhanced_logger("api_requests")
workflow_logger = create_enhanced_logger("workflows")