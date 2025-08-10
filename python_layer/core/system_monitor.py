"""
Python层系统监控器
监控系统性能和健康状态
"""

import asyncio
import psutil
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import deque

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.is_running = False
        self.start_time = None
        
        # 性能历史数据（最近1小时，每分钟记录一次）
        self.cpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.disk_history = deque(maxlen=60)
        self.network_history = deque(maxlen=60)
        
        # 监控任务
        self.monitor_task = None
        self.collection_interval = 60  # 60秒收集一次
        
        # 告警配置
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time": 5.0  # 秒
        }
        
        self.alerts = []
        self.max_alerts = 100
    
    async def start(self):
        """启动系统监控"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # 启动监控任务
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """停止系统监控"""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._add_alert("system_monitor", f"监控循环错误: {e}", "error")
                await asyncio.sleep(5)  # 出错后短暂等待
    
    async def _collect_metrics(self):
        """收集系统指标"""
        timestamp = datetime.utcnow()
        
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_history.append({
            "timestamp": timestamp,
            "percent": cpu_percent
        })
        
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            await self._add_alert("cpu", f"CPU使用率过高: {cpu_percent:.1f}%", "warning")
        
        # 内存指标
        memory = psutil.virtual_memory()
        self.memory_history.append({
            "timestamp": timestamp,
            "percent": memory.percent,
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3)
        })
        
        if memory.percent > self.alert_thresholds["memory_percent"]:
            await self._add_alert("memory", f"内存使用率过高: {memory.percent:.1f}%", "warning")
        
        # 磁盘指标
        disk = psutil.disk_usage('/')
        disk_percent = disk.used / disk.total * 100
        self.disk_history.append({
            "timestamp": timestamp,
            "percent": disk_percent,
            "used_gb": disk.used / (1024**3),
            "total_gb": disk.total / (1024**3)
        })
        
        if disk_percent > self.alert_thresholds["disk_percent"]:
            await self._add_alert("disk", f"磁盘使用率过高: {disk_percent:.1f}%", "warning")
        
        # 网络指标
        network = psutil.net_io_counters()
        self.network_history.append({
            "timestamp": timestamp,
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        })
    
    async def _add_alert(self, component: str, message: str, level: str):
        """添加告警"""
        alert = {
            "timestamp": datetime.utcnow(),
            "component": component,
            "message": message,
            "level": level
        }
        
        self.alerts.append(alert)
        
        # 限制告警数量
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts//2:]
        
        print(f"[ALERT-{level.upper()}] {component}: {message}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取当前系统指标"""
        if not self.is_running:
            return {"error": "系统监控未启动"}
        
        # 当前系统状态
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # 计算运行时间
        uptime_seconds = (
            (datetime.utcnow() - self.start_time).total_seconds()
            if self.start_time else 0
        )
        
        metrics = {
            "uptime_seconds": uptime_seconds,
            "current": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "percent": memory.percent,
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                },
                "disk": {
                    "percent": disk.used / disk.total * 100,
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            },
            "history": {
                "cpu": list(self.cpu_history)[-10:],  # 最近10个数据点
                "memory": list(self.memory_history)[-10:],
                "disk": list(self.disk_history)[-10:],
                "network": list(self.network_history)[-10:]
            },
            "alerts": {
                "recent": self.alerts[-10:],  # 最近10个告警
                "total_count": len(self.alerts)
            },
            "thresholds": self.alert_thresholds
        }
        
        return metrics
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统基本信息"""
        import platform
        
        return {
            "platform": {
                "system": platform.system(),
                "node": platform.node(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3)
            }
        }
    
    def get_process_info(self) -> Dict[str, Any]:
        """获取当前进程信息"""
        process = psutil.Process()
        
        return {
            "pid": process.pid,
            "name": process.name(),
            "status": process.status(),
            "create_time": datetime.fromtimestamp(process.create_time()),
            "cpu_percent": process.cpu_percent(),
            "memory_info": {
                "rss_mb": process.memory_info().rss / (1024**2),
                "vms_mb": process.memory_info().vms / (1024**2)
            },
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "connections": len(process.connections()) if hasattr(process, 'connections') else 0
        }
    
    def clear_alerts(self):
        """清除所有告警"""
        self.alerts.clear()
    
    def set_alert_threshold(self, metric: str, value: float):
        """设置告警阈值"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = value
            return True
        return False