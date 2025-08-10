"""
系统健康监控器
提供全方位的系统监控、健康检查和性能指标收集
包括数据库、AI服务、网络连接、资源使用等监控
"""

import asyncio
import time
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import motor.motor_asyncio
import redis.asyncio as aioredis
from loguru import logger
from collections import deque, defaultdict
import json
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from config.settings import settings
from services.ai_clients.deepseek_client import get_deepseek_client
from services.ai_clients.gemini_client import get_gemini_client

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    ERROR = "error"

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_name: str
    status: HealthStatus
    response_time: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    process_count: int
    load_average: List[float]

class SystemHealthMonitor:
    """系统健康监控器"""
    
    def __init__(self):
        self.mongodb_client = None
        self.redis_client = None
        self.is_monitoring = False
        self.monitoring_task = None
        self.health_history = deque(maxlen=1000)  # 保留最近1000次检查
        self.metrics_history = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # 监控配置
        self.check_interval = 30  # 检查间隔30秒
        self.health_check_timeout = 10  # 健康检查超时10秒
        self.critical_thresholds = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 95.0,
            "response_time": 5.0,
            "error_rate": 10.0
        }
        
        # 服务状态缓存
        self.service_status_cache = {}
        self.last_alert_time = {}
        
        # 性能计数器
        self.performance_counters = defaultdict(int)
        self.response_times = defaultdict(list)
        
    async def initialize(self):
        """初始化监控器"""
        try:
            # 初始化数据库连接
            await self._initialize_database_connections()
            
            # 初始化监控数据库集合
            await self._setup_monitoring_collections()
            
            logger.info("系统健康监控器初始化成功")
            
        except Exception as e:
            logger.error(f"系统健康监控器初始化失败: {e}")
            raise
    
    async def _initialize_database_connections(self):
        """初始化数据库连接"""
        try:
            db_config = settings.get_database_config()
            
            # MongoDB连接
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                db_config["mongodb_url"],
                maxPoolSize=5,
                minPoolSize=2,
                serverSelectionTimeoutMS=5000
            )
            self.mongodb_db = self.mongodb_client.get_default_database()
            
            # Redis连接
            self.redis_client = aioredis.from_url(
                db_config["redis_url"],
                max_connections=5,
                retry_on_timeout=True
            )
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    async def _setup_monitoring_collections(self):
        """设置监控数据库集合"""
        try:
            # 创建健康检查历史集合
            await self.mongodb_db.health_checks.create_index("timestamp")
            await self.mongodb_db.health_checks.create_index("service_name")
            await self.mongodb_db.health_checks.create_index("status")
            await self.mongodb_db.health_checks.create_index(
                "timestamp", expireAfterSeconds=604800  # 7天后过期
            )
            
            # 创建系统指标集合
            await self.mongodb_db.system_metrics.create_index("timestamp")
            await self.mongodb_db.system_metrics.create_index(
                "timestamp", expireAfterSeconds=2592000  # 30天后过期
            )
            
            # 创建告警历史集合
            await self.mongodb_db.alerts.create_index("timestamp")
            await self.mongodb_db.alerts.create_index("severity")
            await self.mongodb_db.alerts.create_index("service_name")
            await self.mongodb_db.alerts.create_index(
                "timestamp", expireAfterSeconds=2592000  # 30天后过期
            )
            
        except Exception as e:
            logger.warning(f"设置监控集合失败: {e}")
    
    async def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            logger.warning("监控已经在运行中")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("系统健康监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("系统健康监控已停止")
    
    async def _monitoring_loop(self):
        """监控主循环"""
        try:
            while self.is_monitoring:
                start_time = time.time()
                
                # 执行健康检查
                await self._run_health_checks()
                
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 分析和告警
                await self._analyze_and_alert()
                
                # 清理历史数据
                await self._cleanup_old_data()
                
                # 计算循环耗时并等待
                loop_time = time.time() - start_time
                sleep_time = max(0, self.check_interval - loop_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"监控循环耗时过长: {loop_time:.2f}秒")
                    
        except asyncio.CancelledError:
            logger.info("监控循环已取消")
        except Exception as e:
            logger.error(f"监控循环异常: {e}")
            # 监控循环出错，尝试重启
            await asyncio.sleep(60)  # 等待1分钟后重试
            if self.is_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _run_health_checks(self):
        """运行所有健康检查"""
        try:
            # 并发执行所有健康检查
            check_tasks = [
                self._check_database_health(),
                self._check_ai_services_health(),
                self._check_external_apis_health(),
                self._check_system_resources_health(),
                self._check_network_connectivity_health()
            ]
            
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # 处理检查结果
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"健康检查异常: {result}")
                elif isinstance(result, list):
                    for health_result in result:
                        await self._process_health_result(health_result)
                elif result:
                    await self._process_health_result(result)
                    
        except Exception as e:
            logger.error(f"运行健康检查失败: {e}")
    
    async def _check_database_health(self) -> List[HealthCheckResult]:
        """检查数据库健康状况"""
        results = []
        
        # MongoDB健康检查
        try:
            start_time = time.time()
            await self.mongodb_db.command("ping")
            response_time = (time.time() - start_time) * 1000
            
            # 获取更详细的状态
            server_status = await self.mongodb_db.command("serverStatus")
            connection_stats = server_status.get("connections", {})
            
            status = HealthStatus.HEALTHY
            message = "MongoDB连接正常"
            
            # 检查连接数
            current_connections = connection_stats.get("current", 0)
            available_connections = connection_stats.get("available", 0)
            
            if available_connections > 0:
                connection_usage = (current_connections / (current_connections + available_connections)) * 100
                if connection_usage > 80:
                    status = HealthStatus.WARNING
                    message = f"MongoDB连接使用率较高: {connection_usage:.1f}%"
                elif connection_usage > 95:
                    status = HealthStatus.CRITICAL
                    message = f"MongoDB连接使用率危险: {connection_usage:.1f}%"
            
            results.append(HealthCheckResult(
                service_name="mongodb",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    "version": server_status.get("version"),
                    "uptime": server_status.get("uptime"),
                    "connections": connection_stats,
                    "memory": server_status.get("mem", {})
                },
                timestamp=datetime.utcnow()
            ))
            
        except Exception as e:
            results.append(HealthCheckResult(
                service_name="mongodb",
                status=HealthStatus.ERROR,
                response_time=self.health_check_timeout * 1000,
                message="MongoDB连接失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            ))
        
        # Redis健康检查
        try:
            start_time = time.time()
            await self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # 获取Redis信息
            info = await self.redis_client.info()
            
            status = HealthStatus.HEALTHY
            message = "Redis连接正常"
            
            # 检查内存使用
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            
            if max_memory > 0:
                memory_usage = (used_memory / max_memory) * 100
                if memory_usage > 80:
                    status = HealthStatus.WARNING
                    message = f"Redis内存使用率较高: {memory_usage:.1f}%"
                elif memory_usage > 95:
                    status = HealthStatus.CRITICAL
                    message = f"Redis内存使用率危险: {memory_usage:.1f}%"
            
            results.append(HealthCheckResult(
                service_name="redis",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    "version": info.get("redis_version"),
                    "uptime": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": used_memory,
                    "max_memory": max_memory
                },
                timestamp=datetime.utcnow()
            ))
            
        except Exception as e:
            results.append(HealthCheckResult(
                service_name="redis",
                status=HealthStatus.ERROR,
                response_time=self.health_check_timeout * 1000,
                message="Redis连接失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            ))
        
        return results
    
    async def _check_ai_services_health(self) -> List[HealthCheckResult]:
        """检查AI服务健康状况"""
        results = []
        
        # DeepSeek健康检查
        try:
            deepseek_client = await get_deepseek_client()
            start_time = time.time()
            health_result = await deepseek_client.health_check()
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if health_result.get("api_available", False) else HealthStatus.ERROR
            
            results.append(HealthCheckResult(
                service_name="deepseek_ai",
                status=status,
                response_time=response_time,
                message=health_result.get("error", "DeepSeek API正常") if status == HealthStatus.ERROR else "DeepSeek API正常",
                details=health_result.get("stats", {}),
                timestamp=datetime.utcnow(),
                error=health_result.get("error")
            ))
            
        except Exception as e:
            results.append(HealthCheckResult(
                service_name="deepseek_ai",
                status=HealthStatus.ERROR,
                response_time=self.health_check_timeout * 1000,
                message="DeepSeek AI服务检查失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            ))
        
        # Gemini健康检查
        try:
            gemini_client = await get_gemini_client()
            start_time = time.time()
            health_result = await gemini_client.health_check()
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if health_result.get("api_available", False) else HealthStatus.ERROR
            
            results.append(HealthCheckResult(
                service_name="gemini_ai",
                status=status,
                response_time=response_time,
                message=health_result.get("error", "Gemini API正常") if status == HealthStatus.ERROR else "Gemini API正常",
                details=health_result.get("stats", {}),
                timestamp=datetime.utcnow(),
                error=health_result.get("error")
            ))
            
        except Exception as e:
            results.append(HealthCheckResult(
                service_name="gemini_ai",
                status=HealthStatus.ERROR,
                response_time=self.health_check_timeout * 1000,
                message="Gemini AI服务检查失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            ))
        
        return results
    
    async def _check_external_apis_health(self) -> List[HealthCheckResult]:
        """检查外部API健康状况"""
        results = []
        
        # 检查主要的加密货币API
        apis_to_check = [
            {
                "name": "binance_api",
                "url": "https://api.binance.com/api/v3/ping",
                "timeout": 5
            },
            {
                "name": "coinglass_api", 
                "url": "https://open-api.coinglass.com/public/v2/indicator/fear_greed_index",
                "timeout": 10
            }
        ]
        
        for api_config in apis_to_check:
            try:
                start_time = time.time()
                
                timeout = aiohttp.ClientTimeout(total=api_config["timeout"])
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(api_config["url"]) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        status = HealthStatus.HEALTHY if response.status == 200 else HealthStatus.WARNING
                        message = f"API响应正常 (HTTP {response.status})" if response.status == 200 else f"API响应异常 (HTTP {response.status})"
                        
                        results.append(HealthCheckResult(
                            service_name=api_config["name"],
                            status=status,
                            response_time=response_time,
                            message=message,
                            details={
                                "http_status": response.status,
                                "url": api_config["url"]
                            },
                            timestamp=datetime.utcnow()
                        ))
                        
            except asyncio.TimeoutError:
                results.append(HealthCheckResult(
                    service_name=api_config["name"],
                    status=HealthStatus.CRITICAL,
                    response_time=api_config["timeout"] * 1000,
                    message="API请求超时",
                    details={"url": api_config["url"]},
                    timestamp=datetime.utcnow(),
                    error="请求超时"
                ))
                
            except Exception as e:
                results.append(HealthCheckResult(
                    service_name=api_config["name"],
                    status=HealthStatus.ERROR,
                    response_time=self.health_check_timeout * 1000,
                    message="API连接失败",
                    details={"url": api_config["url"]},
                    timestamp=datetime.utcnow(),
                    error=str(e)
                ))
        
        return results
    
    async def _check_system_resources_health(self) -> HealthCheckResult:
        """检查系统资源健康状况"""
        try:
            # 收集系统资源信息
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # 评估健康状况
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_usage > self.critical_thresholds["cpu_usage"]:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU使用率过高: {cpu_usage:.1f}%")
            elif cpu_usage > self.critical_thresholds["cpu_usage"] - 10:
                status = HealthStatus.WARNING
                issues.append(f"CPU使用率较高: {cpu_usage:.1f}%")
            
            if memory.percent > self.critical_thresholds["memory_usage"]:
                status = HealthStatus.CRITICAL
                issues.append(f"内存使用率过高: {memory.percent:.1f}%")
            elif memory.percent > self.critical_thresholds["memory_usage"] - 10:
                status = HealthStatus.WARNING  
                issues.append(f"内存使用率较高: {memory.percent:.1f}%")
            
            disk_usage = (disk.used / disk.total) * 100
            if disk_usage > self.critical_thresholds["disk_usage"]:
                status = HealthStatus.CRITICAL
                issues.append(f"磁盘使用率过高: {disk_usage:.1f}%")
            elif disk_usage > self.critical_thresholds["disk_usage"] - 10:
                status = HealthStatus.WARNING
                issues.append(f"磁盘使用率较高: {disk_usage:.1f}%")
            
            message = "系统资源正常" if not issues else "; ".join(issues)
            
            return HealthCheckResult(
                service_name="system_resources",
                status=status,
                response_time=0,  # 本地检查无网络延迟
                message=message,
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available,
                    "disk_usage": disk_usage,
                    "disk_free": disk.free,
                    "load_average": load_avg,
                    "process_count": len(psutil.pids())
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name="system_resources",
                status=HealthStatus.ERROR,
                response_time=0,
                message="系统资源检查失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_network_connectivity_health(self) -> HealthCheckResult:
        """检查网络连通性健康状况"""
        try:
            # 检查网络IO
            net_io = psutil.net_io_counters()
            
            # 简单的网络连通性检查
            start_time = time.time()
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("https://www.google.com") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    status = HealthStatus.HEALTHY if response.status == 200 else HealthStatus.WARNING
                    message = "网络连接正常" if response.status == 200 else f"网络连接异常 (HTTP {response.status})"
                    
                    return HealthCheckResult(
                        service_name="network_connectivity",
                        status=status,
                        response_time=response_time,
                        message=message,
                        details={
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "packets_sent": net_io.packets_sent,
                            "packets_recv": net_io.packets_recv,
                            "errin": net_io.errin,
                            "errout": net_io.errout
                        },
                        timestamp=datetime.utcnow()
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                service_name="network_connectivity",
                status=HealthStatus.ERROR,
                response_time=5000,
                message="网络连通性检查失败",
                details={},
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # 收集系统指标
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # 获取活跃连接数
            active_connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                },
                active_connections=active_connections,
                process_count=len(psutil.pids()),
                load_average=list(load_avg)
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            # 存储到数据库
            await self._store_metrics(metrics)
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    async def _store_metrics(self, metrics: SystemMetrics):
        """存储指标到数据库"""
        try:
            metrics_doc = asdict(metrics)
            await self.mongodb_db.system_metrics.insert_one(metrics_doc)
        except Exception as e:
            logger.warning(f"存储系统指标失败: {e}")
    
    async def _process_health_result(self, result: HealthCheckResult):
        """处理健康检查结果"""
        try:
            # 添加到历史记录
            self.health_history.append(result)
            
            # 更新服务状态缓存
            self.service_status_cache[result.service_name] = result
            
            # 记录响应时间
            self.response_times[result.service_name].append(result.response_time)
            if len(self.response_times[result.service_name]) > 100:
                self.response_times[result.service_name] = self.response_times[result.service_name][-100:]
            
            # 存储到数据库
            await self._store_health_result(result)
            
            # 检查是否需要告警
            await self._check_for_alerts(result)
            
        except Exception as e:
            logger.error(f"处理健康检查结果失败: {e}")
    
    async def _store_health_result(self, result: HealthCheckResult):
        """存储健康检查结果到数据库"""
        try:
            result_doc = asdict(result)
            result_doc["status"] = result.status.value  # 转换枚举为字符串
            await self.mongodb_db.health_checks.insert_one(result_doc)
        except Exception as e:
            logger.warning(f"存储健康检查结果失败: {e}")
    
    async def _check_for_alerts(self, result: HealthCheckResult):
        """检查是否需要告警"""
        try:
            # 只对警告和错误状态告警
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.ERROR]:
                # 检查是否在冷却期内
                last_alert = self.last_alert_time.get(result.service_name)
                if last_alert and (datetime.utcnow() - last_alert).total_seconds() < 300:  # 5分钟冷却
                    return
                
                # 创建告警
                alert = {
                    "service_name": result.service_name,
                    "severity": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": datetime.utcnow(),
                    "resolved": False
                }
                
                # 存储告警
                await self.mongodb_db.alerts.insert_one(alert)
                
                # 更新最后告警时间
                self.last_alert_time[result.service_name] = datetime.utcnow()
                
                # 触发告警回调
                for callback in self.alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(alert)
                        else:
                            callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调执行失败: {e}")
                
                logger.warning(f"触发告警: {result.service_name} - {result.message}")
                
        except Exception as e:
            logger.error(f"检查告警失败: {e}")
    
    async def _analyze_and_alert(self):
        """分析健康状况并告警"""
        try:
            # 分析系统整体健康状况
            overall_status = await self._calculate_overall_health()
            
            # 分析性能趋势
            performance_trends = await self._analyze_performance_trends()
            
            # 更新Redis中的系统状态
            await self._update_system_status(overall_status, performance_trends)
            
        except Exception as e:
            logger.error(f"分析和告警失败: {e}")
    
    async def _calculate_overall_health(self) -> Dict[str, Any]:
        """计算系统整体健康状况"""
        try:
            if not self.service_status_cache:
                return {"status": "unknown", "message": "暂无健康检查数据"}
            
            # 统计各状态的服务数量
            status_counts = defaultdict(int)
            for service, result in self.service_status_cache.items():
                status_counts[result.status] += 1
            
            total_services = len(self.service_status_cache)
            
            # 计算整体状态
            if status_counts[HealthStatus.ERROR] > 0:
                overall_status = "critical"
            elif status_counts[HealthStatus.CRITICAL] > 0:
                overall_status = "critical"
            elif status_counts[HealthStatus.WARNING] > total_services * 0.3:
                overall_status = "warning"
            elif status_counts[HealthStatus.HEALTHY] == total_services:
                overall_status = "healthy"
            else:
                overall_status = "warning"
            
            return {
                "status": overall_status,
                "total_services": total_services,
                "healthy_services": status_counts[HealthStatus.HEALTHY],
                "warning_services": status_counts[HealthStatus.WARNING],
                "critical_services": status_counts[HealthStatus.CRITICAL],
                "error_services": status_counts[HealthStatus.ERROR],
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"计算整体健康状况失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        try:
            if len(self.metrics_history) < 2:
                return {"status": "insufficient_data"}
            
            # 取最近的指标进行趋势分析
            recent_metrics = list(self.metrics_history)[-10:]  # 最近10个数据点
            
            # 计算平均值和趋势
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_usage for m in recent_metrics) / len(recent_metrics)
            
            # 简单的趋势计算（最新值与平均值比较）
            latest = recent_metrics[-1]
            cpu_trend = "increasing" if latest.cpu_usage > avg_cpu * 1.1 else "stable" if latest.cpu_usage > avg_cpu * 0.9 else "decreasing"
            memory_trend = "increasing" if latest.memory_usage > avg_memory * 1.1 else "stable" if latest.memory_usage > avg_memory * 0.9 else "decreasing"
            
            return {
                "cpu": {
                    "average": avg_cpu,
                    "current": latest.cpu_usage,
                    "trend": cpu_trend
                },
                "memory": {
                    "average": avg_memory,
                    "current": latest.memory_usage,
                    "trend": memory_trend
                },
                "disk": {
                    "average": avg_disk,
                    "current": latest.disk_usage,
                    "trend": "stable"  # 磁盘使用变化通常较慢
                }
            }
            
        except Exception as e:
            logger.error(f"分析性能趋势失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _update_system_status(self, overall_status: Dict, performance_trends: Dict):
        """更新系统状态到Redis"""
        try:
            status_data = {
                "overall_health": json.dumps(overall_status),
                "performance_trends": json.dumps(performance_trends),
                "last_update": datetime.utcnow().isoformat(),
                "monitoring_active": str(self.is_monitoring)
            }
            
            await self.redis_client.hset("system:health_status", mapping=status_data)
            await self.redis_client.expire("system:health_status", 300)  # 5分钟过期
            
        except Exception as e:
            logger.warning(f"更新系统状态失败: {e}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            # 每小时执行一次清理
            if int(time.time()) % 3600 == 0:
                # 清理内存中的历史数据
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # 清理健康检查历史
                self.health_history = deque([
                    h for h in self.health_history 
                    if h.timestamp > cutoff_time
                ], maxlen=1000)
                
                # 清理指标历史
                self.metrics_history = deque([
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ], maxlen=1000)
                
                logger.debug("旧数据清理完成")
                
        except Exception as e:
            logger.error(f"清理旧数据失败: {e}")
    
    def add_alert_callback(self, callback):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback):
        """移除告警回调函数"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        try:
            overall_health = await self._calculate_overall_health()
            performance_trends = await self._analyze_performance_trends()
            
            # 获取服务状态
            services_status = {}
            for service_name, result in self.service_status_cache.items():
                services_status[service_name] = {
                    "status": result.status.value,
                    "response_time": result.response_time,
                    "message": result.message,
                    "last_check": result.timestamp.isoformat()
                }
            
            # 获取最新指标
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                "overall_health": overall_health,
                "services_status": services_status,
                "performance_trends": performance_trends,
                "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
                "monitoring_active": self.is_monitoring,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取当前状态失败: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_health_history(self, service_name: Optional[str] = None, 
                                hours: int = 24) -> List[Dict[str, Any]]:
        """获取健康检查历史"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # 从内存中获取最近的历史
            history = [
                asdict(h) for h in self.health_history
                if h.timestamp > cutoff_time and (not service_name or h.service_name == service_name)
            ]
            
            # 如果内存中数据不够，从数据库获取
            if len(history) < 10:
                try:
                    query = {"timestamp": {"$gte": cutoff_time}}
                    if service_name:
                        query["service_name"] = service_name
                    
                    cursor = self.mongodb_db.health_checks.find(query).sort("timestamp", -1)
                    async for doc in cursor:
                        doc.pop("_id", None)  # 移除MongoDB的_id字段
                        history.append(doc)
                        
                except Exception as e:
                    logger.warning(f"从数据库获取健康历史失败: {e}")
            
            return history
            
        except Exception as e:
            logger.error(f"获取健康历史失败: {e}")
            return []
    
    async def close(self):
        """关闭监控器"""
        try:
            # 停止监控
            await self.stop_monitoring()
            
            # 关闭数据库连接
            if self.mongodb_client:
                self.mongodb_client.close()
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("系统健康监控器已关闭")
            
        except Exception as e:
            logger.error(f"关闭系统健康监控器失败: {e}")

# 全局系统健康监控器实例
system_health_monitor = SystemHealthMonitor()