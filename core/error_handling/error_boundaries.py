"""
综合错误边界和降级处理系统
提供完整的错误处理、系统降级、故障恢复和业务连续性保障
"""

import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from functools import wraps
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import psutil
from loguru import logger
import json

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemState(Enum):
    """系统状态"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    PARTIAL_FAILURE = "partial_failure"
    CRITICAL_FAILURE = "critical_failure"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"

class FallbackStrategy(Enum):
    """降级策略"""
    CACHE_FALLBACK = "cache_fallback"
    DEFAULT_RESPONSE = "default_response"
    ALTERNATIVE_SERVICE = "alternative_service"
    SIMPLIFIED_PROCESSING = "simplified_processing"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class ErrorContext:
    """错误上下文"""
    error_id: str
    timestamp: datetime
    service_name: str
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    system_state: Optional[SystemState] = None
    recovery_attempts: int = 0
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class FallbackConfig:
    """降级配置"""
    strategy: FallbackStrategy
    timeout: float
    max_retries: int
    retry_delay: float
    cache_duration: int
    enabled: bool = True
    priority: int = 1

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        with self._lock:
            if self.state == "open":
                if self.last_failure_time and \
                   (time.time() - self.last_failure_time) > self.timeout:
                    self.state = "half_open"
                else:
                    raise Exception("熔断器开启状态，拒绝请求")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.reset()
                
                return result
                
            except Exception as e:
                self.record_failure()
                raise e
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"熔断器开启: 失败次数 {self.failure_count}")
    
    def reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        logger.info("熔断器已重置")

class ErrorBoundary:
    """错误边界"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.error_history = deque(maxlen=1000)
        self.fallback_configs = {}
        self.circuit_breakers = {}
        self.system_state = SystemState.NORMAL
        self.error_counters = defaultdict(int)
        self.last_health_check = datetime.utcnow()
        
        # 监控线程
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
    
    def register_fallback(self, function_name: str, config: FallbackConfig):
        """注册降级配置"""
        self.fallback_configs[function_name] = config
        
        if config.strategy == FallbackStrategy.CIRCUIT_BREAKER:
            self.circuit_breakers[function_name] = CircuitBreaker()
    
    def wrap_function(self, function_name: str, fallback_func: Optional[Callable] = None):
        """函数装饰器"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_boundary(
                    func, function_name, args, kwargs, fallback_func
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_boundary(
                    func, function_name, args, kwargs, fallback_func
                ))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def _execute_with_boundary(self, func: Callable, function_name: str, 
                                   args: tuple, kwargs: dict, 
                                   fallback_func: Optional[Callable] = None):
        """在错误边界内执行函数"""
        start_time = time.time()
        error_context = None
        
        try:
            # 检查熔断器状态
            if function_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[function_name]
                if circuit_breaker.state == "open":
                    logger.warning(f"熔断器开启，使用降级策略: {function_name}")
                    return await self._execute_fallback(function_name, args, kwargs, fallback_func)
            
            # 执行主函数
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 记录成功
            execution_time = time.time() - start_time
            self._record_success(function_name, execution_time)
            
            return result
            
        except Exception as e:
            # 记录错误
            execution_time = time.time() - start_time
            error_context = self._create_error_context(
                function_name, e, args, kwargs, execution_time
            )
            
            # 记录到错误历史
            self.error_history.append(error_context)
            self.error_counters[function_name] += 1
            
            # 更新熔断器
            if function_name in self.circuit_breakers:
                self.circuit_breakers[function_name].record_failure()
            
            # 尝试降级处理
            try:
                fallback_result = await self._execute_fallback(
                    function_name, args, kwargs, fallback_func, error_context
                )
                
                logger.info(f"降级处理成功: {function_name}")
                return fallback_result
                
            except Exception as fallback_error:
                logger.error(f"降级处理失败: {function_name} - {fallback_error}")
                
                # 如果降级也失败，根据严重程度决定是否重新抛出异常
                if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    self._trigger_system_degradation(error_context)
                
                raise e
    
    def _create_error_context(self, function_name: str, error: Exception, 
                            args: tuple, kwargs: dict, execution_time: float) -> ErrorContext:
        """创建错误上下文"""
        error_id = f"{self.service_name}_{function_name}_{int(time.time())}"
        
        # 判断错误严重程度
        severity = self._determine_error_severity(error, function_name)
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            service_name=self.service_name,
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            stack_trace=traceback.format_exc(),
            system_state=self.system_state,
            additional_data={
                "execution_time": execution_time,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
        )
    
    def _determine_error_severity(self, error: Exception, function_name: str) -> ErrorSeverity:
        """判断错误严重程度"""
        # 数据库相关错误
        if "database" in function_name.lower() or "db" in function_name.lower():
            if "connection" in str(error).lower():
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        # AI服务相关错误
        if "ai" in function_name.lower() or "deepseek" in function_name.lower() or "gemini" in function_name.lower():
            if "timeout" in str(error).lower() or "rate limit" in str(error).lower():
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.LOW
        
        # 交易相关错误
        if "trade" in function_name.lower() or "order" in function_name.lower():
            return ErrorSeverity.HIGH
        
        # 系统级错误
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # 网络错误
        if "network" in str(error).lower() or "connection" in str(error).lower():
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    async def _execute_fallback(self, function_name: str, args: tuple, kwargs: dict,
                               fallback_func: Optional[Callable] = None,
                               error_context: Optional[ErrorContext] = None):
        """执行降级策略"""
        config = self.fallback_configs.get(function_name)
        if not config or not config.enabled:
            raise Exception("没有可用的降级策略")
        
        logger.info(f"执行降级策略: {function_name} - {config.strategy.value}")
        
        if config.strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return await self._default_response_fallback(function_name, args, kwargs)
        
        elif config.strategy == FallbackStrategy.CACHE_FALLBACK:
            return await self._cache_fallback(function_name, args, kwargs)
        
        elif config.strategy == FallbackStrategy.ALTERNATIVE_SERVICE:
            return await self._alternative_service_fallback(function_name, args, kwargs)
        
        elif config.strategy == FallbackStrategy.SIMPLIFIED_PROCESSING:
            return await self._simplified_processing_fallback(function_name, args, kwargs)
        
        elif config.strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_fallback(function_name, args, kwargs)
        
        elif fallback_func:
            if asyncio.iscoroutinefunction(fallback_func):
                return await fallback_func(*args, **kwargs)
            else:
                return fallback_func(*args, **kwargs)
        
        else:
            raise Exception("未实现的降级策略")
    
    async def _default_response_fallback(self, function_name: str, args: tuple, kwargs: dict):
        """默认响应降级"""
        # 根据函数名返回合适的默认响应
        if "analyze" in function_name.lower():
            return {
                "status": "fallback",
                "message": "分析服务暂时不可用，返回默认结果",
                "result": {},
                "confidence": 0.0
            }
        elif "predict" in function_name.lower():
            return {
                "status": "fallback", 
                "prediction": "neutral",
                "confidence": 0.0,
                "message": "预测服务暂时不可用"
            }
        elif "data" in function_name.lower():
            return {
                "status": "fallback",
                "data": [],
                "message": "数据服务暂时不可用"
            }
        else:
            return {
                "status": "fallback",
                "message": f"服务 {function_name} 暂时不可用"
            }
    
    async def _cache_fallback(self, function_name: str, args: tuple, kwargs: dict):
        """缓存降级"""
        # 这里应该从Redis或内存缓存中获取最近的结果
        # 简化实现
        return {
            "status": "cache_fallback",
            "message": "使用缓存数据",
            "data": {},
            "cached_at": datetime.utcnow().isoformat()
        }
    
    async def _alternative_service_fallback(self, function_name: str, args: tuple, kwargs: dict):
        """备用服务降级"""
        # 这里应该调用备用服务
        # 简化实现
        return {
            "status": "alternative_service",
            "message": "使用备用服务",
            "data": {}
        }
    
    async def _simplified_processing_fallback(self, function_name: str, args: tuple, kwargs: dict):
        """简化处理降级"""
        # 使用简化的算法或逻辑
        return {
            "status": "simplified",
            "message": "使用简化处理逻辑",
            "data": {}
        }
    
    async def _graceful_degradation_fallback(self, function_name: str, args: tuple, kwargs: dict):
        """优雅降级"""
        # 保持核心功能，去除非关键功能
        return {
            "status": "degraded",
            "message": "系统运行在降级模式",
            "core_data": {},
            "disabled_features": ["advanced_analysis", "real_time_updates"]
        }
    
    def _record_success(self, function_name: str, execution_time: float):
        """记录成功执行"""
        # 重置相关的熔断器
        if function_name in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[function_name]
            if circuit_breaker.state == "half_open":
                circuit_breaker.reset()
    
    def _trigger_system_degradation(self, error_context: ErrorContext):
        """触发系统降级"""
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.system_state = SystemState.CRITICAL_FAILURE
            logger.critical(f"系统进入严重故障状态: {error_context.error_message}")
        elif error_context.severity == ErrorSeverity.HIGH:
            if self.system_state == SystemState.NORMAL:
                self.system_state = SystemState.DEGRADED
                logger.warning(f"系统进入降级状态: {error_context.error_message}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > datetime.utcnow() - timedelta(minutes=10)
        ]
        
        error_rate = len(recent_errors) / 10  # 每分钟错误数
        
        # 分析错误模式
        error_patterns = defaultdict(int)
        for error in recent_errors:
            error_patterns[error.error_type] += 1
        
        return {
            "service_name": self.service_name,
            "system_state": self.system_state.value,
            "recent_error_count": len(recent_errors),
            "error_rate_per_minute": error_rate,
            "most_common_errors": dict(error_patterns),
            "circuit_breakers": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            "fallback_configs": {
                name: {"strategy": config.strategy.value, "enabled": config.enabled}
                for name, config in self.fallback_configs.items()
            },
            "total_error_count": sum(self.error_counters.values()),
            "last_health_check": self.last_health_check.isoformat()
        }
    
    def get_error_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误报告"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        relevant_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
        
        # 按严重程度分组
        errors_by_severity = defaultdict(list)
        for error in relevant_errors:
            errors_by_severity[error.severity.value].append(asdict(error))
        
        # 按函数分组
        errors_by_function = defaultdict(int)
        for error in relevant_errors:
            errors_by_function[error.function_name] += 1
        
        return {
            "report_period_hours": hours,
            "total_errors": len(relevant_errors),
            "errors_by_severity": dict(errors_by_severity),
            "errors_by_function": dict(errors_by_function),
            "system_state_changes": self._get_system_state_changes(cutoff_time),
            "recovery_attempts": sum(error.recovery_attempts for error in relevant_errors),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_system_state_changes(self, since: datetime) -> List[Dict[str, Any]]:
        """获取系统状态变化"""
        # 简化实现，实际应该跟踪状态变化历史
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "previous_state": "normal",
                "new_state": self.system_state.value,
                "trigger": "error_threshold_exceeded"
            }
        ]
    
    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info(f"错误边界监控已启动: {self.service_name}")
    
    def stop_monitoring_thread(self):
        """停止监控"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_monitoring.wait(60):  # 每分钟检查一次
            try:
                self._health_check()
                self._cleanup_old_errors()
                self._auto_recovery()
            except Exception as e:
                logger.error(f"错误边界监控异常: {e}")
    
    def _health_check(self):
        """健康检查"""
        self.last_health_check = datetime.utcnow()
        
        # 检查错误率
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        error_rate = len(recent_errors) / 5  # 每分钟错误数
        
        if error_rate > 10:  # 每分钟超过10个错误
            if self.system_state == SystemState.NORMAL:
                self.system_state = SystemState.DEGRADED
                logger.warning(f"系统进入降级状态，错误率: {error_rate}/分钟")
        elif error_rate < 1:  # 错误率降低
            if self.system_state == SystemState.DEGRADED:
                self.system_state = SystemState.RECOVERY
                logger.info("系统开始恢复")
    
    def _cleanup_old_errors(self):
        """清理旧错误记录"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        original_count = len(self.error_history)
        
        # deque不支持按条件过滤，需要重建
        filtered_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
        
        self.error_history.clear()
        self.error_history.extend(filtered_errors)
        
        cleaned_count = original_count - len(self.error_history)
        if cleaned_count > 0:
            logger.debug(f"清理了 {cleaned_count} 个旧错误记录")
    
    def _auto_recovery(self):
        """自动恢复"""
        if self.system_state in [SystemState.RECOVERY, SystemState.DEGRADED]:
            # 检查是否可以恢复正常状态
            recent_errors = [
                error for error in self.error_history
                if error.timestamp > datetime.utcnow() - timedelta(minutes=10)
            ]
            
            if len(recent_errors) == 0:
                self.system_state = SystemState.NORMAL
                logger.info("系统已恢复正常状态")
                
                # 重置熔断器
                for cb in self.circuit_breakers.values():
                    if cb.state == "open":
                        cb.reset()

class GlobalErrorBoundaryManager:
    """全局错误边界管理器"""
    
    def __init__(self):
        self.error_boundaries = {}
        self.global_fallback_configs = {}
        self.system_health = SystemState.NORMAL
        self.monitoring_enabled = True
        
    def register_service(self, service_name: str) -> ErrorBoundary:
        """注册服务"""
        if service_name not in self.error_boundaries:
            error_boundary = ErrorBoundary(service_name)
            self.error_boundaries[service_name] = error_boundary
            
            # 应用全局降级配置
            for func_name, config in self.global_fallback_configs.items():
                error_boundary.register_fallback(func_name, config)
            
            # 启动监控
            if self.monitoring_enabled:
                error_boundary.start_monitoring()
        
        return self.error_boundaries[service_name]
    
    def register_global_fallback(self, function_name: str, config: FallbackConfig):
        """注册全局降级配置"""
        self.global_fallback_configs[function_name] = config
        
        # 应用到所有已注册的服务
        for boundary in self.error_boundaries.values():
            boundary.register_fallback(function_name, config)
    
    def get_global_health_status(self) -> Dict[str, Any]:
        """获取全局健康状态"""
        service_statuses = {}
        total_errors = 0
        critical_services = []
        
        for service_name, boundary in self.error_boundaries.items():
            health = boundary.get_health_status()
            service_statuses[service_name] = health
            total_errors += health["total_error_count"]
            
            if health["system_state"] in ["critical_failure", "partial_failure"]:
                critical_services.append(service_name)
        
        # 计算全局系统状态
        if critical_services:
            global_state = SystemState.CRITICAL_FAILURE
        elif any(
            status["system_state"] == "degraded" 
            for status in service_statuses.values()
        ):
            global_state = SystemState.DEGRADED
        else:
            global_state = SystemState.NORMAL
        
        return {
            "global_system_state": global_state.value,
            "total_services": len(self.error_boundaries),
            "critical_services": critical_services,
            "total_errors_24h": total_errors,
            "service_details": service_statuses,
            "monitoring_enabled": self.monitoring_enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        global_health = self.get_global_health_status()
        
        # 收集各服务的错误报告
        service_reports = {}
        for service_name, boundary in self.error_boundaries.items():
            service_reports[service_name] = boundary.get_error_report()
        
        return {
            "global_health": global_health,
            "service_reports": service_reports,
            "recommendations": self._generate_recommendations(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        for service_name, boundary in self.error_boundaries.items():
            health = boundary.get_health_status()
            
            if health["error_rate_per_minute"] > 5:
                recommendations.append(
                    f"服务 {service_name} 错误率较高，建议检查相关组件"
                )
            
            if health["system_state"] != "normal":
                recommendations.append(
                    f"服务 {service_name} 处于 {health['system_state']} 状态，需要关注"
                )
            
            # 检查熔断器状态
            for cb_name, cb_state in health["circuit_breakers"].items():
                if cb_state == "open":
                    recommendations.append(
                        f"服务 {service_name} 的熔断器 {cb_name} 处于开启状态，建议检查"
                    )
        
        return recommendations
    
    def shutdown(self):
        """关闭所有错误边界"""
        for boundary in self.error_boundaries.values():
            boundary.stop_monitoring_thread()
        
        logger.info("全局错误边界管理器已关闭")

# 全局错误边界管理器实例
global_error_boundary_manager = GlobalErrorBoundaryManager()

# 便捷函数
def get_error_boundary(service_name: str) -> ErrorBoundary:
    """获取错误边界"""
    return global_error_boundary_manager.register_service(service_name)

def error_boundary(service_name: str, fallback_func: Optional[Callable] = None):
    """错误边界装饰器"""
    def decorator(func):
        boundary = get_error_boundary(service_name)
        return boundary.wrap_function(func.__name__, fallback_func)(func)
    
    return decorator

def configure_fallback(service_name: str, function_name: str, 
                      strategy: FallbackStrategy, **kwargs):
    """配置降级策略"""
    config = FallbackConfig(
        strategy=strategy,
        timeout=kwargs.get("timeout", 30.0),
        max_retries=kwargs.get("max_retries", 3),
        retry_delay=kwargs.get("retry_delay", 1.0),
        cache_duration=kwargs.get("cache_duration", 300),
        enabled=kwargs.get("enabled", True),
        priority=kwargs.get("priority", 1)
    )
    
    boundary = get_error_boundary(service_name)
    boundary.register_fallback(function_name, config)