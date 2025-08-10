"""
FastAPI主应用

提供统一的API接口层，集成Rust引擎和Python业务逻辑
使用Python 3.13和FastAPI最新特性优化性能
"""

import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

# FastAPI 2025核心组件
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Python 3.13性能优化
if sys.version_info >= (3, 13):
    # 启用free-threading模式（如果可用）
    if hasattr(sys, '_is_free_threading') and sys._is_free_threading:
        import concurrent.futures
        # 使用并发执行器优化CPU密集型任务
        CPU_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    else:
        CPU_EXECUTOR = None
else:
    CPU_EXECUTOR = None

# 添加项目根目录到Python路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 应用模块导入
from fastapi_layer.routers import strategies, trades, market_data, ai_analysis, system, backtest, simulation, data_management, performance
from fastapi_layer.middleware.logging import LoggingMiddleware
from fastapi_layer.middleware.cors import CustomCORSMiddleware
from fastapi_layer.dependencies.database import get_database_manager
from python_layer.core.ai_engine import AIEngine
from python_layer.core.data_manager import DataManager
from python_layer.core.strategy_manager import StrategyManager
from python_layer.core.system_monitor import SystemMonitor
from python_layer.utils.config import get_settings
from python_layer.utils.logger import get_logger

logger = get_logger(__name__)

# 全局应用状态
class AppState:
    def __init__(self):
        self.ai_engine: Optional[AIEngine] = None
        self.data_manager: Optional[DataManager] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.startup_time: Optional[datetime] = None
        self.is_ready: bool = False

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动阶段
    logger.info("FastAPI应用启动中...")
    app_state.startup_time = datetime.utcnow()
    
    try:
        # 初始化核心组件
        settings = get_settings()
        
        logger.info("初始化数据管理器...")
        app_state.data_manager = DataManager()
        await app_state.data_manager.initialize()
        
        logger.info("初始化AI引擎...")
        app_state.ai_engine = AIEngine()
        
        logger.info("初始化策略管理器...")
        app_state.strategy_manager = StrategyManager()
        await app_state.strategy_manager.initialize()
        
        logger.info("初始化系统监控器...")
        app_state.system_monitor = SystemMonitor()
        await app_state.system_monitor.start()
        
        # 健康检查
        health_status = await perform_health_check()
        if not health_status["healthy"]:
            logger.error(f"健康检查失败: {health_status['issues']}")
            raise RuntimeError("应用初始化失败")
        
        app_state.is_ready = True
        logger.info("FastAPI应用启动完成 ✅")
        
        yield
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise
    
    finally:
        # 关闭阶段
        logger.info("FastAPI应用关闭中...")
        
        if app_state.system_monitor:
            await app_state.system_monitor.stop()
        
        if app_state.data_manager:
            await app_state.data_manager.close()
        
        logger.info("FastAPI应用已关闭")

# 创建FastAPI应用实例
def create_app() -> FastAPI:
    """创建配置完整的FastAPI应用"""
    
    settings = get_settings()
    
    # 自定义OpenAPI配置
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
            
        openapi_schema = get_openapi(
            title="AI量化交易系统API",
            version="1.0.0",
            description="""
            ## AI驱动的量化交易系统API
            
            ### 核心功能
            - 🤖 AI智能分析（DeepSeek + Gemini）
            - 📊 实时市场数据处理
            - 🚀 高性能策略执行（Rust引擎）
            - 📈 Alpha因子发现
            - 🛡️ 智能风控管理
            - 🔄 量化回测系统（历史数据回放）
            - 🎡 实时仿真交易（虚拟市场环境）
            
            ### 技术栈
            - **后端**: Python 3.13 + FastAPI + Rust
            - **数据库**: MongoDB 8.0 + Redis 8.0
            - **AI**: DeepSeek Reasoner + Gemini Pro
            - **性能**: JIT编译 + 异步优化
            
            ### API特性
            - 实时WebSocket数据流
            - 智能缓存策略
            - 自动文档生成
            - 完整类型检查
            """,
            routes=app.routes,
            contact={
                "name": "Trading System Support",
                "email": "support@tradingsystem.ai"
            },
            license_info={
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        )
        
        # 自定义标签
        openapi_schema["tags"] = [
            {"name": "market-data", "description": "市场数据和实时行情"},
            {"name": "strategies", "description": "交易策略管理"},
            {"name": "trades", "description": "交易执行和记录"},
            {"name": "ai-analysis", "description": "AI智能分析"},
            {"name": "system", "description": "系统监控和管理"},
            {"name": "backtest", "description": "量化回测系统"},
            {"name": "simulation", "description": "实时仿真交易系统"},
            {"name": "data-management", "description": "数据管理和分析"},
            {"name": "performance", "description": "性能分析和报告"},
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    # 创建应用实例
    app = FastAPI(
        title="AI量化交易系统",
        description="高性能AI驱动的量化交易系统API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        # Python 3.13优化
        debug=settings.debug,
    )
    
    app.openapi = custom_openapi
    
    # 中间件配置（按顺序添加）
    
    # 1. 受信任主机中间件（安全）
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.api.allowed_hosts
        )
    
    # 2. GZIP压缩中间件（性能）
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=6
    )
    
    # 3. 自定义CORS中间件
    app.add_middleware(
        CustomCORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 4. 日志中间件
    app.add_middleware(LoggingMiddleware)
    
    # 路由注册    
    app.include_router(
        market_data.router,
        prefix="/api/v1/market",
        tags=["market-data"]
    )
    
    app.include_router(
        strategies.router,
        prefix="/api/v1/strategies",
        tags=["strategies"]
    )
    
    app.include_router(
        trades.router,
        prefix="/api/v1/trades",
        tags=["trades"]
    )
    
    app.include_router(
        ai_analysis.router,
        prefix="/api/v1/ai",
        tags=["ai-analysis"]
    )
    
    app.include_router(
        system.router,
        prefix="/api/v1/system",
        tags=["system"]
    )
    
    app.include_router(
        backtest.router,
        prefix="/api/v1/backtest",
        tags=["backtest"]
    )
    
    app.include_router(
        simulation.router,
        prefix="/api/v1/simulation",
        tags=["simulation"]
    )
    
    app.include_router(
        data_management.router,
        prefix="/api/v1/data",
        tags=["data-management"]
    )
    
    app.include_router(
        performance.router,
        prefix="/api/v1/performance",
        tags=["performance"]
    )
    
    return app

# 创建应用实例
app = create_app()

# ============ 全局异常处理器 ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
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
            "path": str(request.url.path),
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """数值错误处理器"""
    logger.warning(f"值错误: {exc}, 路径: {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "type": "validation_error",
                "message": str(exc)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "internal_server_error",
                "message": "服务器内部错误",
                "detail": str(exc) if app.debug else None
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============ 核心API端点 ============

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API根端点"""
    return {
        "success": True,
        "data": {
            "name": "AI量化交易系统API",
            "version": "1.0.0",
            "description": "高性能AI驱动的量化交易系统",
            "features": [
                "🤖 AI智能分析",
                "🚀 Rust高性能引擎", 
                "📊 实时数据处理",
                "🛡️ 智能风控",
                "📈 Alpha因子发现",
                "🔄 量化回测系统",
                "🎡 实时仿真交易"
            ],
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json",
            "status": "ready" if app_state.is_ready else "initializing"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """健康检查端点"""
    return await perform_health_check()

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """系统指标端点"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    # 收集各组件指标
    metrics = {
        "system": {
            "uptime_seconds": (
                (datetime.utcnow() - app_state.startup_time).total_seconds()
                if app_state.startup_time else 0
            ),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "jit_available": hasattr(sys, 'implementation') and getattr(sys.implementation, 'name', '') == 'cpython' and sys.version_info >= (3, 13),
            "free_threading": hasattr(sys, '_is_free_threading') and sys._is_free_threading,
        }
    }
    
    # AI引擎指标
    if app_state.ai_engine:
        metrics["ai_engine"] = app_state.ai_engine.get_stats()
    
    # 数据管理器指标
    if app_state.data_manager:
        metrics["data_manager"] = await app_state.data_manager.get_stats()
    
    # 策略管理器指标
    if app_state.strategy_manager:
        metrics["strategy_manager"] = await app_state.strategy_manager.get_stats()
    
    # 系统监控指标
    if app_state.system_monitor:
        metrics["system_monitor"] = await app_state.system_monitor.get_metrics()
    
    return {
        "success": True,
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/reset-stats", response_model=Dict[str, Any])
async def reset_statistics():
    """重置统计信息"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    reset_results = {}
    
    # 重置AI引擎统计
    if app_state.ai_engine:
        app_state.ai_engine.stats = {
            "total_requests": 0,
            "deepseek_requests": 0,
            "gemini_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "errors": 0,
        }
        reset_results["ai_engine"] = "已重置"
    
    # 重置其他组件统计
    if app_state.data_manager:
        await app_state.data_manager.reset_stats()
        reset_results["data_manager"] = "已重置"
    
    if app_state.strategy_manager:
        await app_state.strategy_manager.reset_stats()
        reset_results["strategy_manager"] = "已重置"
    
    logger.info("系统统计信息已重置")
    
    return {
        "success": True,
        "data": reset_results,
        "message": "统计信息重置完成",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============ WebSocket支持 ============

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket):
    """市场数据WebSocket端点"""
    from fastapi import WebSocket
    
    await websocket.accept()
    logger.info(f"WebSocket连接建立: {websocket.client}")
    
    try:
        while True:
            # 这里实现实时市场数据推送
            if app_state.data_manager:
                market_data = await app_state.data_manager.get_latest_market_data()
                await websocket.send_json({
                    "type": "market_data",
                    "data": market_data,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(0.25)  # 4Hz更新频率
            
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        logger.info(f"WebSocket连接关闭: {websocket.client}")

@app.websocket("/ws/ai-analysis")
async def websocket_ai_analysis(websocket):
    """AI分析WebSocket端点"""
    from fastapi import WebSocket
    
    await websocket.accept()
    logger.info(f"AI分析WebSocket连接建立: {websocket.client}")
    
    try:
        while True:
            # 实时AI分析结果推送
            if app_state.ai_engine and app_state.data_manager:
                # 这里应该实现智能推送逻辑
                await asyncio.sleep(5)  # 每5秒推送一次AI分析
                
    except Exception as e:
        logger.error(f"AI分析WebSocket错误: {e}")
    finally:
        logger.info(f"AI分析WebSocket连接关闭: {websocket.client}")

# ============ 后台任务 ============

async def background_data_sync():
    """后台数据同步任务"""
    if app_state.data_manager:
        await app_state.data_manager.sync_market_data()

async def background_ai_analysis():
    """后台AI分析任务"""
    if app_state.ai_engine and app_state.data_manager:
        market_data = await app_state.data_manager.get_latest_market_data()
        if market_data:
            # 执行定期AI分析
            sentiment = await app_state.ai_engine.analyze_market_sentiment([])
            logger.debug(f"后台AI分析完成，情绪得分: {sentiment.score}")

# ============ 工具函数 ============

async def perform_health_check() -> Dict[str, Any]:
    """执行健康检查"""
    health_status = {
        "healthy": True,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {},
        "issues": []
    }
    
    # 检查AI引擎
    if app_state.ai_engine:
        try:
            ai_stats = app_state.ai_engine.get_stats()
            health_status["components"]["ai_engine"] = {
                "status": "healthy",
                "error_rate": ai_stats.get("error_rate", 0)
            }
        except Exception as e:
            health_status["healthy"] = False
            health_status["components"]["ai_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["issues"].append(f"AI引擎故障: {e}")
    
    # 检查数据管理器
    if app_state.data_manager:
        try:
            data_stats = await app_state.data_manager.get_stats()
            health_status["components"]["data_manager"] = {
                "status": "healthy",
                "connections": data_stats.get("active_connections", 0)
            }
        except Exception as e:
            health_status["healthy"] = False
            health_status["components"]["data_manager"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            health_status["issues"].append(f"数据管理器故障: {e}")
    
    # 检查策略管理器
    if app_state.strategy_manager:
        try:
            strategy_stats = await app_state.strategy_manager.get_stats()
            health_status["components"]["strategy_manager"] = {
                "status": "healthy",
                "active_strategies": strategy_stats.get("active_strategies", 0)
            }
        except Exception as e:
            health_status["healthy"] = False
            health_status["components"]["strategy_manager"] = {
                "status": "unhealthy",
                "error": str(e) 
            }
            health_status["issues"].append(f"策略管理器故障: {e}")
    
    return {
        "success": True,
        "data": health_status
    }

def get_app_state() -> AppState:
    """获取应用状态（依赖注入）"""
    return app_state

# ============ 应用启动配置 ============

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    # Python 3.13优化的服务器配置
    server_config = {
        "host": settings.api.host,
        "port": settings.api.port,
        "log_level": "info",
        "access_log": True,
        "reload": settings.debug,
        "workers": 1 if settings.debug else 4,
    }
    
    # 如果支持free-threading，启用更多worker
    if hasattr(sys, '_is_free_threading') and sys._is_free_threading:
        server_config["workers"] = min(8, server_config["workers"] * 2)
    
    logger.info(f"启动FastAPI服务器: {server_config}")
    
    uvicorn.run(
        "fastapi_layer.main:app",
        **server_config
    )