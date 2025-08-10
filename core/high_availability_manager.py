"""
高可用架构管理器
实现系统容错、故障恢复、负载均衡和服务监控
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    OFFLINE = "offline"

class FailoverStrategy(Enum):
    """故障转移策略"""
    IMMEDIATE = "immediate"     # 立即切换
    GRACEFUL = "graceful"      # 优雅切换
    MANUAL = "manual"          # 手动切换

@dataclass
class ServiceEndpoint:
    """服务端点配置"""
    service_id: str
    endpoint_url: str
    priority: int = 1           # 优先级，1最高
    weight: int = 100          # 负载均衡权重
    max_failures: int = 3      # 最大失败次数
    timeout: float = 10.0      # 超时时间
    health_check_interval: float = 30.0
    enabled: bool = True
    
    # 运行时状态
    status: ServiceStatus = ServiceStatus.HEALTHY
    failure_count: int = 0
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5     # 失败阈值
    recovery_timeout: float = 60.0  # 恢复超时
    half_open_max_calls: int = 3   # 半开状态最大调用次数

class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭状态，正常处理请求
    OPEN = "open"          # 开启状态，拒绝请求
    HALF_OPEN = "half_open" # 半开状态，试探性处理请求

@dataclass
class CircuitBreaker:
    """熔断器"""
    service_id: str
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.health_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_health_callback(self, service_id: str, callback: Callable):
        """注册健康状态变化回调"""
        self.health_callbacks[service_id].append(callback)
    
    async def start_health_check(self, endpoint: ServiceEndpoint):
        """启动健康检查"""
        if endpoint.service_id in self.check_tasks:
            return
        
        task = asyncio.create_task(
            self._health_check_loop(endpoint)
        )
        self.check_tasks[endpoint.service_id] = task
        logger.info(f"启动健康检查: {endpoint.service_id}")
    
    async def stop_health_check(self, service_id: str):
        """停止健康检查"""
        if service_id in self.check_tasks:
            self.check_tasks[service_id].cancel()
            del self.check_tasks[service_id]
            logger.info(f"停止健康检查: {service_id}")
    
    async def _health_check_loop(self, endpoint: ServiceEndpoint):
        """健康检查循环"""
        while endpoint.enabled:
            try:
                await asyncio.sleep(endpoint.health_check_interval)
                
                # 执行健康检查
                start_time = time.time()
                is_healthy = await self._perform_health_check(endpoint)
                response_time = time.time() - start_time
                
                # 更新响应时间
                endpoint.response_times.append(response_time)
                endpoint.last_health_check = datetime.utcnow()
                
                # 更新状态
                old_status = endpoint.status
                if is_healthy:
                    endpoint.failure_count = 0
                    if endpoint.status != ServiceStatus.HEALTHY:
                        endpoint.status = ServiceStatus.RECOVERING
                        # 给一些时间恢复，然后标记为健康
                        await asyncio.sleep(5)
                        endpoint.status = ServiceStatus.HEALTHY
                else:
                    endpoint.failure_count += 1
                    if endpoint.failure_count >= endpoint.max_failures:
                        endpoint.status = ServiceStatus.UNHEALTHY
                    else:
                        endpoint.status = ServiceStatus.DEGRADED
                
                # 状态变化时触发回调
                if old_status != endpoint.status:
                    await self._notify_status_change(endpoint, old_status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查异常 {endpoint.service_id}: {e}")
                endpoint.last_error = str(e)
                endpoint.status = ServiceStatus.UNHEALTHY
    
    async def _perform_health_check(self, endpoint: ServiceEndpoint) -> bool:
        """执行具体的健康检查"""
        try:
            # 这里可以根据服务类型实现不同的健康检查逻辑
            # 目前简化为基本连通性检查
            
            if "mongodb" in endpoint.service_id.lower():
                return await self._check_mongodb_health(endpoint.endpoint_url)
            elif "redis" in endpoint.service_id.lower():
                return await self._check_redis_health(endpoint.endpoint_url)
            elif "binance" in endpoint.service_id.lower():
                return await self._check_api_health(endpoint.endpoint_url)
            elif "okx" in endpoint.service_id.lower():
                return await self._check_api_health(endpoint.endpoint_url)
            else:
                return True  # 默认健康
                
        except Exception as e:
            logger.error(f"健康检查失败 {endpoint.service_id}: {e}")
            return False
    
    async def _check_mongodb_health(self, connection_url: str) -> bool:
        """检查MongoDB健康状态"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            client = AsyncIOMotorClient(connection_url, serverSelectionTimeoutMS=5000)
            await client.admin.command('ping')
            client.close()
            return True
        except:
            return False
    
    async def _check_redis_health(self, connection_url: str) -> bool:
        """检查Redis健康状态"""
        try:
            import redis.asyncio as aioredis
            redis = aioredis.from_url(connection_url)
            await redis.ping()
            await redis.close()
            return True
        except:
            return False
    
    async def _check_api_health(self, api_url: str) -> bool:
        """检查API健康状态"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(api_url) as response:
                    return response.status < 500
        except:
            return False
    
    async def _notify_status_change(self, endpoint: ServiceEndpoint, old_status: ServiceStatus):
        """通知状态变化"""
        logger.info(f"服务状态变化: {endpoint.service_id} {old_status.value} -> {endpoint.status.value}")
        
        for callback in self.health_callbacks[endpoint.service_id]:
            try:
                await callback(endpoint, old_status, endpoint.status)
            except Exception as e:
                logger.error(f"健康状态回调执行失败: {e}")

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
    
    def select_endpoint(self, service_id: str, endpoints: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """选择服务端点"""
        # 过滤健康的端点
        healthy_endpoints = [
            ep for ep in endpoints 
            if ep.enabled and ep.status in [ServiceStatus.HEALTHY, ServiceStatus.RECOVERING]
        ]
        
        if not healthy_endpoints:
            # 如果没有健康端点，尝试使用降级端点
            degraded_endpoints = [
                ep for ep in endpoints 
                if ep.enabled and ep.status == ServiceStatus.DEGRADED
            ]
            if degraded_endpoints:
                healthy_endpoints = degraded_endpoints
        
        if not healthy_endpoints:
            return None
        
        if self.strategy == "priority":
            return self._priority_select(healthy_endpoints)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_select(service_id, healthy_endpoints)
        elif self.strategy == "least_response_time":
            return self._least_response_time_select(healthy_endpoints)
        else:
            return healthy_endpoints[0]
    
    def _priority_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """优先级选择"""
        return min(endpoints, key=lambda ep: ep.priority)
    
    def _weighted_round_robin_select(self, service_id: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """加权轮询选择"""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return endpoints[0]
        
        # 简化的加权轮询实现
        self.round_robin_counters[service_id] += 1
        counter = self.round_robin_counters[service_id]
        
        weighted_index = counter % total_weight
        current_weight = 0
        
        for endpoint in endpoints:
            current_weight += endpoint.weight
            if weighted_index < current_weight:
                return endpoint
        
        return endpoints[0]
    
    def _least_response_time_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """最小响应时间选择"""
        def avg_response_time(endpoint):
            if not endpoint.response_times:
                return 0.0
            return sum(endpoint.response_times) / len(endpoint.response_times)
        
        return min(endpoints, key=avg_response_time)

class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, strategy: FailoverStrategy = FailoverStrategy.GRACEFUL):
        self.strategy = strategy
        self.failover_history: Dict[str, List[Dict]] = defaultdict(list)
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
    
    async def handle_service_failure(self, failed_endpoint: ServiceEndpoint, 
                                   available_endpoints: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """处理服务故障"""
        logger.warning(f"处理服务故障: {failed_endpoint.service_id}")
        
        # 记录故障
        failure_record = {
            "timestamp": datetime.utcnow(),
            "endpoint": failed_endpoint.endpoint_url,
            "error": failed_endpoint.last_error,
            "strategy": self.strategy.value
        }
        self.failover_history[failed_endpoint.service_id].append(failure_record)
        
        # 选择替代端点
        backup_endpoint = None
        for endpoint in available_endpoints:
            if (endpoint.service_id == failed_endpoint.service_id and 
                endpoint != failed_endpoint and
                endpoint.status == ServiceStatus.HEALTHY):
                backup_endpoint = endpoint
                break
        
        if backup_endpoint:
            if self.strategy == FailoverStrategy.IMMEDIATE:
                return backup_endpoint
            elif self.strategy == FailoverStrategy.GRACEFUL:
                # 优雅切换：给一些时间处理现有请求
                await asyncio.sleep(2)
                return backup_endpoint
        
        return None
    
    async def start_recovery_monitor(self, failed_endpoint: ServiceEndpoint):
        """启动恢复监控"""
        if failed_endpoint.service_id not in self.recovery_tasks:
            task = asyncio.create_task(
                self._recovery_monitor_loop(failed_endpoint)
            )
            self.recovery_tasks[failed_endpoint.service_id] = task
    
    async def _recovery_monitor_loop(self, endpoint: ServiceEndpoint):
        """恢复监控循环"""
        while endpoint.status == ServiceStatus.UNHEALTHY:
            try:
                await asyncio.sleep(30)  # 30秒检查一次
                
                # 尝试恢复检查
                logger.info(f"尝试恢复服务: {endpoint.service_id}")
                # 这里可以实现具体的恢复逻辑
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"恢复监控异常: {e}")

class HighAvailabilityManager:
    """高可用架构管理器"""
    
    def __init__(self):
        self.service_endpoints: Dict[str, List[ServiceEndpoint]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer("weighted_round_robin")
        self.failover_manager = FailoverManager(FailoverStrategy.GRACEFUL)
        
        # 系统监控
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "failovers": 0
        }
        
        self.monitoring_enabled = True
        self.monitoring_task = None
    
    def register_service(self, service_config: Dict[str, Any]):
        """注册服务"""
        service_id = service_config["service_id"]
        
        # 创建服务端点
        for endpoint_config in service_config.get("endpoints", []):
            endpoint = ServiceEndpoint(
                service_id=service_id,
                endpoint_url=endpoint_config["url"],
                priority=endpoint_config.get("priority", 1),
                weight=endpoint_config.get("weight", 100),
                max_failures=endpoint_config.get("max_failures", 3),
                timeout=endpoint_config.get("timeout", 10.0),
                health_check_interval=endpoint_config.get("health_check_interval", 30.0)
            )
            self.service_endpoints[service_id].append(endpoint)
        
        # 创建熔断器
        circuit_config = CircuitBreakerConfig(
            failure_threshold=service_config.get("circuit_breaker", {}).get("failure_threshold", 5),
            recovery_timeout=service_config.get("circuit_breaker", {}).get("recovery_timeout", 60.0),
            half_open_max_calls=service_config.get("circuit_breaker", {}).get("half_open_max_calls", 3)
        )
        self.circuit_breakers[service_id] = CircuitBreaker(service_id, circuit_config)
        
        logger.info(f"注册服务: {service_id} ({len(self.service_endpoints[service_id])}个端点)")
    
    async def initialize(self):
        """初始化高可用管理器"""
        try:
            # 注册核心服务
            await self._register_core_services()
            
            # 启动健康检查
            for service_id, endpoints in self.service_endpoints.items():
                for endpoint in endpoints:
                    await self.health_checker.start_health_check(endpoint)
            
            # 注册健康状态回调
            for service_id in self.service_endpoints.keys():
                self.health_checker.register_health_callback(
                    service_id, self._handle_health_status_change
                )
            
            # 启动监控
            if self.monitoring_enabled:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("高可用架构管理器初始化完成")
            
        except Exception as e:
            logger.error(f"高可用管理器初始化失败: {e}")
            raise
    
    async def _register_core_services(self):
        """注册核心服务"""
        # MongoDB服务
        mongodb_config = {
            "service_id": "mongodb_primary",
            "endpoints": [
                {"url": "mongodb://localhost:27017", "priority": 1, "weight": 100}
            ],
            "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 30.0}
        }
        self.register_service(mongodb_config)
        
        # Redis服务
        redis_config = {
            "service_id": "redis_primary",
            "endpoints": [
                {"url": "redis://localhost:6379", "priority": 1, "weight": 100}
            ],
            "circuit_breaker": {"failure_threshold": 3, "recovery_timeout": 30.0}
        }
        self.register_service(redis_config)
        
        # Binance API服务
        binance_config = {
            "service_id": "binance_api",
            "endpoints": [
                {"url": "https://api.binance.com/api/v3/ping", "priority": 1, "weight": 100},
                {"url": "https://api1.binance.com/api/v3/ping", "priority": 2, "weight": 80},
                {"url": "https://api2.binance.com/api/v3/ping", "priority": 3, "weight": 60}
            ],
            "circuit_breaker": {"failure_threshold": 5, "recovery_timeout": 60.0}
        }
        self.register_service(binance_config)
        
        # OKX API服务
        okx_config = {
            "service_id": "okx_api",
            "endpoints": [
                {"url": "https://www.okx.com/api/v5/public/time", "priority": 1, "weight": 100}
            ],
            "circuit_breaker": {"failure_threshold": 5, "recovery_timeout": 60.0}
        }
        self.register_service(okx_config)
    
    async def get_service_endpoint(self, service_id: str) -> Optional[ServiceEndpoint]:
        """获取可用的服务端点"""
        if service_id not in self.service_endpoints:
            return None
        
        endpoints = self.service_endpoints[service_id]
        circuit_breaker = self.circuit_breakers.get(service_id)
        
        # 检查熔断器状态
        if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
            if await self._should_attempt_reset(circuit_breaker):
                circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                circuit_breaker.half_open_calls = 0
            else:
                logger.warning(f"服务熔断中: {service_id}")
                return None
        
        # 负载均衡选择端点
        selected_endpoint = self.load_balancer.select_endpoint(service_id, endpoints)
        
        return selected_endpoint
    
    async def record_request_result(self, service_id: str, success: bool, response_time: float = 0.0):
        """记录请求结果"""
        self.system_metrics["total_requests"] += 1
        
        if success:
            self.system_metrics["successful_requests"] += 1
            await self._handle_successful_request(service_id)
        else:
            self.system_metrics["failed_requests"] += 1
            await self._handle_failed_request(service_id)
        
        # 更新平均响应时间
        if response_time > 0:
            current_avg = self.system_metrics["average_response_time"]
            total_requests = self.system_metrics["total_requests"]
            self.system_metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    async def _handle_successful_request(self, service_id: str):
        """处理成功请求"""
        circuit_breaker = self.circuit_breakers.get(service_id)
        if circuit_breaker:
            if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                circuit_breaker.half_open_calls += 1
                if circuit_breaker.half_open_calls >= circuit_breaker.config.half_open_max_calls:
                    circuit_breaker.state = CircuitBreakerState.CLOSED
                    circuit_breaker.failure_count = 0
                    logger.info(f"熔断器恢复: {service_id}")
            elif circuit_breaker.state == CircuitBreakerState.CLOSED:
                circuit_breaker.failure_count = max(0, circuit_breaker.failure_count - 1)
    
    async def _handle_failed_request(self, service_id: str):
        """处理失败请求"""
        circuit_breaker = self.circuit_breakers.get(service_id)
        if circuit_breaker:
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.utcnow()
            
            if circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold:
                circuit_breaker.state = CircuitBreakerState.OPEN
                logger.warning(f"熔断器开启: {service_id} (失败次数: {circuit_breaker.failure_count})")
    
    async def _should_attempt_reset(self, circuit_breaker: CircuitBreaker) -> bool:
        """判断是否应该尝试重置熔断器"""
        if not circuit_breaker.last_failure_time:
            return True
        
        time_elapsed = (datetime.utcnow() - circuit_breaker.last_failure_time).total_seconds()
        return time_elapsed >= circuit_breaker.config.recovery_timeout
    
    async def _handle_health_status_change(self, endpoint: ServiceEndpoint, 
                                         old_status: ServiceStatus, new_status: ServiceStatus):
        """处理健康状态变化"""
        logger.info(f"服务健康状态变化: {endpoint.service_id} {old_status.value} -> {new_status.value}")
        
        if new_status == ServiceStatus.UNHEALTHY:
            # 启动故障转移
            available_endpoints = self.service_endpoints[endpoint.service_id]
            backup_endpoint = await self.failover_manager.handle_service_failure(
                endpoint, available_endpoints
            )
            
            if backup_endpoint:
                self.system_metrics["failovers"] += 1
                logger.info(f"故障转移完成: {endpoint.service_id} -> {backup_endpoint.endpoint_url}")
            
            # 启动恢复监控
            await self.failover_manager.start_recovery_monitor(endpoint)
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 生成监控报告
                report = await self.generate_health_report()
                logger.info(f"系统健康报告: {report['summary']}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        service_status = {}
        overall_healthy = True
        
        for service_id, endpoints in self.service_endpoints.items():
            healthy_count = sum(1 for ep in endpoints if ep.status == ServiceStatus.HEALTHY)
            total_count = len(endpoints)
            
            service_health = {
                "healthy_endpoints": healthy_count,
                "total_endpoints": total_count,
                "availability": (healthy_count / total_count * 100) if total_count > 0 else 0,
                "circuit_breaker_state": self.circuit_breakers[service_id].state.value if service_id in self.circuit_breakers else "none"
            }
            
            if healthy_count == 0:
                overall_healthy = False
            
            service_status[service_id] = service_health
        
        total_requests = self.system_metrics["total_requests"]
        success_rate = (
            (self.system_metrics["successful_requests"] / total_requests * 100) 
            if total_requests > 0 else 100.0
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy" if overall_healthy else "degraded",
            "services": service_status,
            "system_metrics": {
                **self.system_metrics,
                "success_rate": success_rate
            },
            "summary": f"整体状态: {'健康' if overall_healthy else '降级'}, 成功率: {success_rate:.1f}%"
        }
    
    async def shutdown(self):
        """关闭高可用管理器"""
        try:
            # 停止监控
            self.monitoring_enabled = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            # 停止健康检查
            for service_id in self.service_endpoints.keys():
                await self.health_checker.stop_health_check(service_id)
            
            logger.info("高可用架构管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭高可用管理器失败: {e}")

# 全局高可用管理器实例
ha_manager = HighAvailabilityManager()