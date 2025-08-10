"""
AI量化交易系统 - Web服务器
基于核心组件测试成功，提供完整的Web API服务
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from core.real_data_manager import real_data_manager
from core.data_manager import data_manager

# 创建FastAPI应用
app = FastAPI(
    title="AI量化交易系统",
    description="""
    ## 🤖 AI驱动的量化交易系统
    
    ### 核心功能
    - 🧠 **AI智能分析**: 基于Coinglass市场情绪和技术指标
    - 📊 **实时数据**: OKX和Binance实时行情数据
    - 🛡️ **智能风控**: 多维度风险评估和管理
    - 📈 **策略执行**: 自动化交易策略执行
    
    ### 技术特色
    - Python 3.13 + FastAPI 高性能API
    - MongoDB 8.0 + Redis 8.0 数据存储
    - WebSocket 实时数据流
    - AI驱动的市场分析
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 应用状态
class AppState:
    def __init__(self):
        self.initialized = False
        self.startup_time = None
        self.components_status = {}

app_state = AppState()

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print("🚀 启动AI量化交易系统...")
    app_state.startup_time = datetime.utcnow()
    
    try:
        # 初始化数据管理器
        print("📊 初始化数据管理器...")
        await data_manager.initialize()
        app_state.components_status["data_manager"] = "initialized"
        
        # 初始化真实数据管理器
        print("📡 初始化真实数据管理器...")
        await real_data_manager.initialize()
        app_state.components_status["real_data_manager"] = "initialized"
        
        # 初始化Coinglass分析器
        from core.coinglass_analyzer import coinglass_analyzer
        await coinglass_analyzer.initialize()
        app_state.components_status["coinglass_analyzer"] = "initialized"
        
        app_state.initialized = True
        print("✅ AI量化交易系统启动完成!")
        print(f"📍 Web服务: http://0.0.0.0:8000")
        print(f"📚 API文档: http://0.0.0.0:8000/docs")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        app_state.initialized = False

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    print("🔴 关闭AI量化交易系统...")
    
    try:
        if real_data_manager.is_running:
            await real_data_manager.stop_real_data_stream()
        
        if hasattr(data_manager, 'close'):
            await data_manager.close()
        
        print("✅ 系统已安全关闭")
        
    except Exception as e:
        print(f"❌ 关闭过程出错: {e}")

# ============ 异常处理器 ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "internal_server_error",
                "message": str(exc)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============ 核心API端点 ============

@app.get("/")
async def root():
    """API根端点"""
    return {
        "success": True,
        "data": {
            "name": "AI量化交易系统",
            "version": "1.0.0",
            "description": "高性能AI驱动的量化交易系统",
            "features": [
                "🤖 AI智能分析（Coinglass + 技术指标）",
                "📊 实时数据流（OKX + Binance）",
                "🛡️ 智能风险管理",
                "📈 自动化策略执行",
                "🔌 WebSocket实时推送"
            ],
            "status": "ready" if app_state.initialized else "initializing",
            "uptime_seconds": (
                (datetime.utcnow() - app_state.startup_time).total_seconds()
                if app_state.startup_time else 0
            ),
            "api_docs": "/docs",
            "websocket_endpoint": "/ws/market"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        if not app_state.initialized:
            return {
                "success": False,
                "data": {
                    "healthy": False,
                    "status": "initializing",
                    "message": "系统正在初始化中..."
                }
            }
        
        # 获取系统健康状态
        health = await real_data_manager.health_check()
        
        # 检查各组件状态
        components_health = {
            "data_manager": app_state.components_status.get("data_manager") == "initialized",
            "real_data_manager": app_state.components_status.get("real_data_manager") == "initialized",
            "coinglass_analyzer": app_state.components_status.get("coinglass_analyzer") == "initialized",
            "mongodb": health.get("database", {}).get("mongodb", False),
            "redis": health.get("database", {}).get("redis", False),
            "coinglass_enabled": health.get("coinglass", {}).get("enabled", False)
        }
        
        overall_healthy = all(components_health.values())
        
        return {
            "success": True,
            "data": {
                "healthy": overall_healthy,
                "status": "healthy" if overall_healthy else "degraded",
                "uptime_seconds": (
                    (datetime.utcnow() - app_state.startup_time).total_seconds()
                    if app_state.startup_time else 0
                ),
                "components": components_health,
                "details": health
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "data": {
                "healthy": False,
                "status": "error",
                "error": str(e)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

# ============ 市场数据API ============

@app.get("/api/v1/market/latest")
async def get_latest_market_data():
    """获取最新市场数据"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        prices = await real_data_manager.get_latest_prices()
        connections = real_data_manager.get_connection_status()
        
        return {
            "success": True,
            "data": {
                "prices": prices,
                "connections": connections,
                "data_stream_running": real_data_manager.is_running,
                "symbols_available": list(prices.keys()) if prices else []
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场数据失败: {str(e)}")

@app.get("/api/v1/market/coinglass")
async def get_coinglass_analysis():
    """获取Coinglass市场情绪分析"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        analysis = await real_data_manager.get_coinglass_analysis()
        
        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取Coinglass分析失败: {str(e)}")

# ============ 数据流控制API ============

@app.post("/api/v1/data/start-stream")
async def start_real_data_stream():
    """启动实时数据流"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        if real_data_manager.is_running:
            return {
                "success": True,
                "message": "实时数据流已在运行",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await real_data_manager.start_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流启动成功",
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动数据流失败: {str(e)}")

@app.post("/api/v1/data/stop-stream")
async def stop_real_data_stream():
    """停止实时数据流"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        if not real_data_manager.is_running:
            return {
                "success": True,
                "message": "实时数据流未在运行",
                "status": "stopped",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await real_data_manager.stop_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流已停止",
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止数据流失败: {str(e)}")

@app.get("/api/v1/data/stream-status")
async def get_stream_status():
    """获取数据流状态"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        connections = real_data_manager.get_connection_status()
        
        return {
            "success": True,
            "data": {
                "is_running": real_data_manager.is_running,
                "connections": connections,
                "coinglass_enabled": real_data_manager.coinglass_enabled,
                "trading_pairs": real_data_manager.trading_pairs
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

# 高可用架构相关端点
@app.get("/api/v1/ha/health")
async def get_system_health():
    """获取系统健康状态"""
    try:
        from core.high_availability_manager import ha_manager
        health_report = await ha_manager.generate_health_report()
        
        return {
            "success": True,
            "data": health_report,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取健康状态失败: {str(e)}")

@app.get("/api/v1/ha/services")
async def get_service_status():
    """获取所有服务状态"""
    try:
        from core.high_availability_manager import ha_manager
        
        service_info = {}
        for service_id, endpoints in ha_manager.service_endpoints.items():
            service_info[service_id] = {
                "endpoints": [
                    {
                        "url": ep.endpoint_url,
                        "status": ep.status.value,
                        "priority": ep.priority,
                        "weight": ep.weight,
                        "failure_count": ep.failure_count,
                        "last_health_check": ep.last_health_check.isoformat() if ep.last_health_check else None,
                        "avg_response_time": sum(ep.response_times) / len(ep.response_times) if ep.response_times else 0.0
                    }
                    for ep in endpoints
                ],
                "circuit_breaker": {
                    "state": ha_manager.circuit_breakers[service_id].state.value,
                    "failure_count": ha_manager.circuit_breakers[service_id].failure_count
                } if service_id in ha_manager.circuit_breakers else None
            }
        
        return {
            "success": True,
            "data": {
                "services": service_info,
                "system_metrics": ha_manager.system_metrics
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取服务状态失败: {str(e)}")

@app.get("/api/v1/pipeline/metrics")
async def get_pipeline_metrics():
    """获取数据流水线性能指标"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        metrics = real_data_manager.get_pipeline_metrics()
        
        return {
            "success": True,
            "data": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取流水线指标失败: {str(e)}")

# ============ N8N工作流API ============

@app.get("/api/v1/workflows")
async def get_workflows():
    """获取所有工作流"""
    try:
        from core.n8n_workflow_manager import n8n_workflow_manager
        
        workflows = []
        for workflow_def in n8n_workflow_manager.workflows.values():
            workflows.append({
                "workflow_id": workflow_def.workflow_id,
                "name": workflow_def.name,
                "description": workflow_def.description,
                "enabled": workflow_def.enabled,
                "trigger_type": workflow_def.trigger_type.value,
                "schedule": workflow_def.schedule,
                "webhook_path": workflow_def.webhook_path
            })
        
        return {
            "success": True,
            "data": workflows,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工作流失败: {str(e)}")

@app.post("/api/v1/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request_data: dict = None):
    """执行工作流"""
    try:
        from core.n8n_workflow_manager import n8n_workflow_manager
        from core.enhanced_logger import workflow_logger
        
        input_data = request_data.get("data") if request_data else None
        
        workflow_logger.info(f"开始执行工作流: {workflow_id}", extra={
            "workflow_id": workflow_id,
            "input_data": input_data
        })
        
        execution_id = await n8n_workflow_manager.execute_workflow(workflow_id, input_data)
        
        if execution_id:
            workflow_logger.info(f"工作流执行启动成功: {workflow_id}", extra={
                "workflow_id": workflow_id,
                "execution_id": execution_id
            })
            
            return {
                "success": True,
                "data": {
                    "execution_id": execution_id,
                    "workflow_id": workflow_id,
                    "status": "running"
                },
                "message": "工作流执行已启动",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            workflow_logger.error(f"工作流执行启动失败: {workflow_id}")
            raise HTTPException(status_code=500, detail="工作流执行启动失败")
        
    except Exception as e:
        from core.enhanced_logger import workflow_logger
        workflow_logger.error(f"工作流执行异常: {workflow_id}", exception=e)
        raise HTTPException(status_code=500, detail=f"执行工作流失败: {str(e)}")

@app.get("/api/v1/workflows/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """获取执行状态"""
    try:
        from core.n8n_workflow_manager import n8n_workflow_manager
        
        execution = await n8n_workflow_manager.get_execution_status(execution_id)
        
        if execution:
            return {
                "success": True,
                "data": {
                    "execution_id": execution.execution_id,
                    "workflow_id": execution.workflow_id,
                    "status": execution.status.value,
                    "start_time": execution.start_time.isoformat(),
                    "end_time": execution.end_time.isoformat() if execution.end_time else None,
                    "duration": execution.duration,
                    "error_message": execution.error_message,
                    "retry_count": execution.retry_count
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="执行记录未找到")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取执行状态失败: {str(e)}")

@app.get("/api/v1/workflows/stats")
async def get_workflow_stats():
    """获取工作流统计"""
    try:
        from core.n8n_workflow_manager import n8n_workflow_manager
        
        stats = n8n_workflow_manager.get_workflow_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取工作流统计失败: {str(e)}")

# ============ 日志管理API ============

@app.get("/api/v1/logs/recent")
async def get_recent_logs(
    level: str = None,
    hours: int = 1,
    limit: int = 100,
    logger_name: str = "system"
):
    """获取最近日志"""
    try:
        from core.enhanced_logger import system_logger, trading_logger, data_logger, api_logger, workflow_logger
        
        # 选择日志记录器
        loggers = {
            "system": system_logger,
            "trading": trading_logger,
            "data": data_logger,
            "api": api_logger,
            "workflow": workflow_logger
        }
        
        selected_logger = loggers.get(logger_name, system_logger)
        
        # 解析日志级别
        log_level = None
        if level:
            from core.enhanced_logger import LogLevel
            try:
                log_level = LogLevel(level.upper())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的日志级别: {level}")
        
        logs = await selected_logger.get_recent_logs(level=log_level, hours=hours, limit=limit)
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "total_count": len(logs),
                "logger_name": logger_name,
                "level_filter": level,
                "time_range_hours": hours
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")

@app.get("/api/v1/logs/errors")
async def get_error_summary(hours: int = 24, logger_name: str = "system"):
    """获取错误摘要"""
    try:
        from core.enhanced_logger import system_logger, trading_logger, data_logger, api_logger, workflow_logger
        
        loggers = {
            "system": system_logger,
            "trading": trading_logger, 
            "data": data_logger,
            "api": api_logger,
            "workflow": workflow_logger
        }
        
        selected_logger = loggers.get(logger_name, system_logger)
        error_summary = await selected_logger.get_error_summary(hours=hours)
        
        return {
            "success": True,
            "data": error_summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取错误摘要失败: {str(e)}")

@app.get("/api/v1/alerts/active")
async def get_active_alerts(severity: str = None):
    """获取活跃告警"""
    try:
        from core.enhanced_logger import system_logger, AlertSeverity
        
        # 解析严重程度
        alert_severity = None
        if severity:
            try:
                alert_severity = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效的告警严重程度: {severity}")
        
        active_alerts = system_logger.alert_manager.get_active_alerts(severity=alert_severity)
        
        # 转换为字典格式
        alerts_data = []
        for alert in active_alerts:
            alerts_data.append({
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity.value,
                "component": alert.component,
                "error_type": alert.error_type,
                "error_message": alert.error_message,
                "occurrence_count": alert.occurrence_count,
                "first_occurrence": alert.first_occurrence.isoformat() if alert.first_occurrence else None,
                "last_occurrence": alert.last_occurrence.isoformat() if alert.last_occurrence else None,
                "context": alert.context,
                "resolved": alert.resolved
            })
        
        return {
            "success": True,
            "data": {
                "alerts": alerts_data,
                "total_count": len(alerts_data),
                "severity_filter": severity
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取活跃告警失败: {str(e)}")

@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """解决告警"""
    try:
        from core.enhanced_logger import system_logger
        
        await system_logger.alert_manager.resolve_alert(alert_id)
        
        return {
            "success": True,
            "message": f"告警 {alert_id} 已标记为已解决",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解决告警失败: {str(e)}")

@app.get("/api/v1/alerts/stats")
async def get_alert_stats(hours: int = 24):
    """获取告警统计"""
    try:
        from core.enhanced_logger import system_logger
        
        stats = system_logger.alert_manager.get_alert_stats(hours=hours)
        
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取告警统计失败: {str(e)}")

# ============ 数据质量API ============

@app.get("/api/v1/data/quality-report")
async def get_data_quality_report(hours: int = 1):
    """获取数据质量报告"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        report = await real_data_manager.get_data_quality_report(hours)
        
        return {
            "success": True,
            "data": report,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据质量报告失败: {str(e)}")

@app.get("/api/v1/data/database-optimization")
async def get_database_optimization_report():
    """获取数据库优化报告"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        from core.data_manager import data_manager
        
        if data_manager.db_optimizer:
            report = await data_manager.db_optimizer.get_optimization_report()
            
            return {
                "success": True,
                "data": report,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "数据库优化器未启用",
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据库优化报告失败: {str(e)}")

@app.get("/api/v1/data/cache-performance")
async def get_cache_performance():
    """获取缓存性能统计"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        from core.data_manager import data_manager
        
        if hasattr(data_manager.cache_manager, 'get_comprehensive_stats'):
            # 增强缓存管理器统计
            stats = await data_manager.cache_manager.get_comprehensive_stats()
            
            return {
                "success": True,
                "data": {
                    "cache_type": "enhanced_multi_tier",
                    "stats": stats,
                    "recommendations": _get_cache_recommendations(stats)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # 基础缓存统计
            return {
                "success": True,
                "data": {
                    "cache_type": "basic_redis",
                    "message": "基础缓存管理器，统计功能有限"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存性能统计失败: {str(e)}")

def _get_cache_recommendations(stats: Dict) -> List[str]:
    """根据缓存统计生成优化建议"""
    recommendations = []
    
    try:
        overall = stats.get("overall", {})
        l1_memory = stats.get("l1_memory", {})
        l2_redis = stats.get("l2_redis", {})
        
        # 总体命中率建议
        hit_rate = overall.get("hit_rate", 0)
        if hit_rate < 70:
            recommendations.append("🔴 总体缓存命中率低于70%，建议增加L1内存缓存大小")
        elif hit_rate < 85:
            recommendations.append("🟡 缓存命中率可以进一步提升，考虑优化预热策略")
        else:
            recommendations.append("✅ 缓存命中率良好")
        
        # L1缓存建议
        l1_hit_rate = l1_memory.get("hit_rate", 0)
        l1_size = l1_memory.get("size", 0)
        l1_max_size = l1_memory.get("max_size", 0)
        evictions = l1_memory.get("evictions", 0)
        
        if l1_hit_rate < 40:
            recommendations.append("🔴 L1内存缓存命中率过低，建议检查数据访问模式")
        
        if l1_size / l1_max_size > 0.9 and evictions > 100:
            recommendations.append("🟡 L1缓存空间紧张且淘汰频繁，建议增加内存缓存大小")
        
        # 响应时间建议
        avg_response_time = overall.get("avg_response_time_ms", 0)
        if avg_response_time > 50:
            recommendations.append("🟡 平均响应时间较高，考虑优化数据结构或增加缓存层级")
        elif avg_response_time < 10:
            recommendations.append("✅ 缓存响应时间优秀")
        
        # Redis连接建议
        connected_clients = l2_redis.get("connected_clients", 0)
        if connected_clients > 50:
            recommendations.append("⚠️ Redis连接数较多，注意监控连接池状态")
        
    except Exception:
        recommendations.append("⚠️ 无法生成详细建议，建议检查缓存统计数据")
    
    return recommendations if recommendations else ["✅ 缓存性能正常"]

# ============ AI分析API ============

@app.get("/api/v1/ai/market-sentiment")
async def get_market_sentiment():
    """获取AI市场情绪分析"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        from core.coinglass_analyzer import coinglass_analyzer
        
        # 获取市场情绪
        sentiment = await coinglass_analyzer.analyze_market_sentiment()
        
        # 获取综合信号
        composite = await coinglass_analyzer.generate_composite_signal()
        
        return {
            "success": True,
            "data": {
                "sentiment": {
                    "score": sentiment.sentiment_score,
                    "trend": sentiment.trend,
                    "strength": sentiment.strength,
                    "confidence": sentiment.confidence,
                    "components": sentiment.components
                },
                "composite_signal": {
                    "overall_score": composite.overall_score,
                    "signal_strength": composite.signal_strength,
                    "market_regime": composite.market_regime,
                    "risk_assessment": composite.risk_assessment,
                    "confidence": composite.confidence,
                    "key_factors": composite.key_factors
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI分析失败: {str(e)}")

# ============ 系统管理API ============

@app.get("/api/v1/system/status")
async def get_system_status():
    """获取系统详细状态"""
    try:
        health = await real_data_manager.health_check() if app_state.initialized else {}
        
        return {
            "success": True,
            "data": {
                "initialized": app_state.initialized,
                "uptime_seconds": (
                    (datetime.utcnow() - app_state.startup_time).total_seconds()
                    if app_state.startup_time else 0
                ),
                "components": app_state.components_status,
                "health": health,
                "system_info": {
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "platform": sys.platform
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")

@app.post("/api/v1/system/refresh-analysis")
async def refresh_analysis(background_tasks: BackgroundTasks):
    """刷新AI分析缓存"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    # 添加后台任务刷新分析
    background_tasks.add_task(refresh_analysis_cache)
    
    return {
        "success": True,
        "message": "分析缓存刷新任务已启动",
        "timestamp": datetime.utcnow().isoformat()
    }

async def refresh_analysis_cache():
    """后台任务：刷新分析缓存"""
    try:
        from core.coinglass_analyzer import coinglass_analyzer
        coinglass_analyzer.analysis_cache.clear()
        print("✅ 分析缓存已刷新")
    except Exception as e:
        print(f"❌ 刷新分析缓存失败: {e}")

# ============ WebSocket支持 ============

@app.websocket("/ws/market")
async def websocket_market_data(websocket):
    """市场数据WebSocket端点"""
    await websocket.accept()
    print("🔌 WebSocket连接建立: 市场数据")
    
    try:
        while True:
            if app_state.initialized:
                # 获取最新数据
                prices = await real_data_manager.get_latest_prices()
                coinglass = await real_data_manager.get_coinglass_analysis()
                
                # 发送数据
                await websocket.send_json({
                    "type": "market_update",
                    "data": {
                        "prices": prices,
                        "sentiment": coinglass.get("sentiment", {}),
                        "composite_signal": coinglass.get("composite_signal", {}),
                        "connections": real_data_manager.get_connection_status(),
                        "is_running": real_data_manager.is_running
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                # 系统未初始化，发送状态信息
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "status": "initializing",
                        "message": "系统正在初始化中..."
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(2)  # 每2秒更新一次
            
    except Exception as e:
        print(f"❌ WebSocket错误: {e}")
    finally:
        print("🔌 WebSocket连接关闭: 市场数据")

async def initialize_system():
    """初始化系统组件"""
    try:
        from core.enhanced_logger import system_logger
        system_logger.info("🔧 开始初始化系统组件...")
        
        # 初始化高可用管理器
        from core.high_availability_manager import ha_manager
        timer_key = system_logger.start_timer("ha_manager_init")
        await ha_manager.initialize()
        await system_logger.end_timer(timer_key, "高可用管理器初始化")
        app_state.components_status["ha_manager"] = True
        system_logger.info("✅ 高可用管理器已初始化")
        
        # 初始化N8N工作流管理器
        from core.n8n_workflow_manager import n8n_workflow_manager
        try:
            timer_key = system_logger.start_timer("n8n_workflow_init")
            await n8n_workflow_manager.initialize()
            await system_logger.end_timer(timer_key, "N8N工作流管理器初始化")
            app_state.components_status["n8n_workflow_manager"] = True
            system_logger.info("✅ N8N工作流管理器已初始化")
        except Exception as e:
            system_logger.error("⚠️ N8N工作流管理器初始化失败，将以降级模式运行", exception=e)
            app_state.components_status["n8n_workflow_manager"] = False
        
        # 初始化真实数据管理器
        timer_key = system_logger.start_timer("data_manager_init")
        await real_data_manager.initialize()
        await system_logger.end_timer(timer_key, "真实数据管理器初始化")
        app_state.components_status["real_data_manager"] = True
        system_logger.info("✅ 真实数据管理器已初始化")
        
        app_state.initialized = True
        app_state.startup_time = datetime.utcnow()
        
        system_logger.info("🎉 系统初始化完成!", extra={
            "components": app_state.components_status,
            "startup_time": app_state.startup_time.isoformat()
        })
        
    except Exception as e:
        from core.enhanced_logger import system_logger
        system_logger.critical("❌ 系统初始化失败", exception=e, extra={
            "components_status": app_state.components_status
        })
        import traceback
        traceback.print_exc()

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    await initialize_system()

@app.on_event("shutdown") 
async def shutdown_event():
    """应用关闭事件"""
    try:
        from core.enhanced_logger import system_logger
        system_logger.info("🛑 开始关闭系统组件...")
        
        # 关闭N8N工作流管理器
        if app_state.components_status.get("n8n_workflow_manager", False):
            from core.n8n_workflow_manager import n8n_workflow_manager
            await n8n_workflow_manager.shutdown()
            system_logger.info("✅ N8N工作流管理器已关闭")
        
        # 关闭高可用管理器
        from core.high_availability_manager import ha_manager
        await ha_manager.shutdown()
        system_logger.info("✅ 高可用管理器已关闭")
        
        system_logger.info("✅ 系统组件已安全关闭")
        
    except Exception as e:
        try:
            from core.enhanced_logger import system_logger
            system_logger.critical("❌ 系统关闭异常", exception=e)
        except:
            print(f"❌ 系统关闭异常: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动AI量化交易系统Web服务器...")
    
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )