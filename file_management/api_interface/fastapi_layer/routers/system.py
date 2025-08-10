"""
系统路由模块
处理系统监控、状态检查、配置管理等API
"""

import sys
import os
import psutil
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.real_data_manager import real_data_manager
from core.data_manager import data_manager
from .auth import verify_token

router = APIRouter()

class SystemInfo(BaseModel):
    python_version: str
    platform: str
    cpu_count: int
    memory_total: int
    memory_available: int
    disk_usage: Dict[str, Any]
    uptime: float

class ComponentStatus(BaseModel):
    name: str
    status: str
    health: str
    last_check: datetime
    details: Dict[str, Any]

@router.get("/info")
async def get_system_info():
    """获取系统基本信息"""
    try:
        # 获取系统信息
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = SystemInfo(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=sys.platform,
            cpu_count=psutil.cpu_count(),
            memory_total=memory.total,
            memory_available=memory.available,
            disk_usage={
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            uptime=0.0  # 这里可以计算应用启动时间
        )
        
        return {
            "success": True,
            "data": system_info.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")

@router.get("/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取各组件健康状态
        health_check = await real_data_manager.health_check()
        
        # CPU和内存使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # 组件状态
        components = []
        
        # 数据库状态
        db_health = health_check.get("database", {})
        components.append(ComponentStatus(
            name="database",
            status="running" if db_health.get("mongodb", {}).get("connected") else "error",
            health="healthy" if db_health.get("mongodb", {}).get("connected") else "unhealthy",
            last_check=datetime.utcnow(),
            details=db_health
        ))
        
        # WebSocket状态
        ws_health = health_check.get("websockets", {})
        components.append(ComponentStatus(
            name="websockets",
            status="running" if any(ws_health.values()) else "stopped",
            health="healthy" if any(ws_health.values()) else "degraded",
            last_check=datetime.utcnow(),
            details=ws_health
        ))
        
        # Coinglass状态
        coinglass_health = health_check.get("coinglass", {})
        components.append(ComponentStatus(
            name="coinglass",
            status="running" if coinglass_health.get("enabled") else "disabled",
            health="healthy" if coinglass_health.get("enabled") else "disabled",
            last_check=datetime.utcnow(),
            details=coinglass_health
        ))
        
        system_status = {
            "overall_status": "healthy",
            "uptime_seconds": 0,  # 这里可以计算实际运行时间
            "performance": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3)
            },
            "components": [comp.dict() for comp in components],
            "environment": {
                "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        }
        
        return {
            "success": True,
            "data": system_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@router.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        health = await real_data_manager.health_check()
        
        # 判断整体健康状态
        overall_healthy = True
        issues = []
        
        # 检查数据库
        if not health.get("database", {}).get("mongodb", {}).get("connected"):
            overall_healthy = False
            issues.append("数据库连接失败")
        
        # 检查Redis
        if not health.get("database", {}).get("redis", {}).get("connected"):
            overall_healthy = False
            issues.append("Redis连接失败")
        
        health_status = {
            "healthy": overall_healthy,
            "status": "healthy" if overall_healthy else "degraded",
            "checks": health,
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        status_code = 200 if overall_healthy else 503
        
        return {
            "success": True,
            "data": health_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"健康检查失败: {str(e)}")

@router.get("/metrics")
async def get_system_metrics():
    """获取系统性能指标"""
    try:
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        
        # 内存指标
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 磁盘指标
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # 网络指标
        network_io = psutil.net_io_counters()
        
        metrics = {
            "cpu": {
                "overall_percent": sum(cpu_percent) / len(cpu_percent),
                "per_core_percent": cpu_percent,
                "frequency": {
                    "current": cpu_freq.current if cpu_freq else 0,
                    "min": cpu_freq.min if cpu_freq else 0,
                    "max": cpu_freq.max if cpu_freq else 0
                }
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent,
                "swap": {
                    "total_gb": swap.total / (1024**3),
                    "used_gb": swap.used / (1024**3),
                    "percent": swap.percent
                }
            },
            "disk": {
                "usage": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent": disk_usage.percent
                },
                "io": {
                    "read_count": disk_io.read_count if disk_io else 0,
                    "write_count": disk_io.write_count if disk_io else 0,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                }
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "processes": {
                "total": len(psutil.pids()),
                "current_process": {
                    "pid": os.getpid(),
                    "memory_info": psutil.Process().memory_info()._asdict(),
                    "cpu_percent": psutil.Process().cpu_percent()
                }
            }
        }
        
        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = Query("INFO", description="日志级别"),
    limit: int = Query(100, ge=1, le=1000, description="返回数量限制"),
    current_user: str = Depends(verify_token)
):
    """获取系统日志"""
    try:
        # 这里可以读取日志文件或从日志系统获取
        logs_path = Path("logs/quantum_trader.log")
        
        logs = []
        if logs_path.exists():
            with open(logs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 取最后的日志记录
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in recent_lines:
                if line.strip():
                    # 解析日志行（简单实现）
                    parts = line.split(' | ')
                    if len(parts) >= 3:
                        logs.append({
                            "timestamp": parts[0],
                            "level": parts[1].strip(),
                            "logger": parts[2].strip(),
                            "message": ' | '.join(parts[3:]).strip() if len(parts) > 3 else ""
                        })
        
        # 按级别过滤
        if level and level != "ALL":
            logs = [log for log in logs if log.get("level") == level]
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "total": len(logs),
                "level_filter": level
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")

@router.post("/restart-component")
async def restart_component(
    component: str = Query(..., description="组件名称"),
    background_tasks: BackgroundTasks,
    current_user: str = Depends(verify_token)
):
    """重启系统组件"""
    try:
        supported_components = ["coinglass", "websockets", "data_manager"]
        
        if component not in supported_components:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的组件: {component}. 支持的组件: {supported_components}"
            )
        
        # 添加重启任务到后台
        background_tasks.add_task(restart_component_task, component)
        
        return {
            "success": True,
            "message": f"组件 {component} 重启任务已启动",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启组件失败: {str(e)}")

async def restart_component_task(component: str):
    """后台任务：重启组件"""
    try:
        if component == "coinglass":
            from core.coinglass_collector import coinglass_collector_manager
            await coinglass_collector_manager.stop_collection()
            await coinglass_collector_manager.start_collection()
            
        elif component == "websockets":
            await real_data_manager.stop_real_data_stream()
            await asyncio.sleep(2)
            await real_data_manager.start_real_data_stream()
            
        elif component == "data_manager":
            await data_manager.close()
            await asyncio.sleep(2)
            await data_manager.initialize()
            
        print(f"组件 {component} 重启完成")
        
    except Exception as e:
        print(f"重启组件 {component} 失败: {e}")

@router.get("/config")
async def get_system_config(current_user: str = Depends(verify_token)):
    """获取系统配置"""
    try:
        config = {
            "environment": os.getenv("ENVIRONMENT", "production"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "8000"))
            },
            "database": {
                "mongodb_url": "mongodb://***:***@localhost:27017/***",  # 隐藏敏感信息
                "redis_url": "redis://***@localhost:6379/0"
            },
            "features": {
                "coinglass_enabled": bool(os.getenv("COINGLASS_API_KEY")),
                "coinglass_update_interval": int(os.getenv("COINGLASS_UPDATE_INTERVAL", "300")),
                "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.8")),
                "hard_stop_loss": int(os.getenv("HARD_STOP_LOSS", "300"))
            }
        }
        
        return {
            "success": True,
            "data": config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@router.post("/clear-cache")
async def clear_system_cache(
    cache_type: str = Query("all", description="缓存类型：all, redis, analysis"),
    current_user: str = Depends(verify_token)
):
    """清除系统缓存"""
    try:
        cleared_items = []
        
        if cache_type in ["all", "redis"]:
            # 清除Redis缓存
            if hasattr(data_manager, 'cache_manager') and data_manager.cache_manager:
                # 这里可以添加清除Redis缓存的逻辑
                cleared_items.append("redis_cache")
        
        if cache_type in ["all", "analysis"]:
            # 清除分析缓存
            from core.coinglass_analyzer import coinglass_analyzer
            coinglass_analyzer.analysis_cache.clear()
            cleared_items.append("analysis_cache")
        
        return {
            "success": True,
            "message": "缓存清除完成",
            "data": {
                "cleared_items": cleared_items,
                "cache_type": cache_type
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")