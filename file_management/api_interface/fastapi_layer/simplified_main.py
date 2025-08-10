"""
简化版FastAPI主应用
专注于核心功能，集成已有的AI量化交易系统组件
"""

import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# FastAPI核心组件
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入已有的系统组件
from core.real_data_manager import real_data_manager
from core.ai_trading_engine import ai_trading_engine
from core.data_manager import data_manager
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局应用状态
class AppState:
    def __init__(self):
        self.startup_time: Optional[datetime] = None
        self.is_ready: bool = False
        self.real_data_manager = real_data_manager
        self.ai_engine = ai_trading_engine
        self.data_manager = data_manager

app_state = AppState()

# 创建FastAPI应用
app = FastAPI(
    title="AI量化交易系统",
    description="高性能AI驱动的量化交易系统API",
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

# 添加GZIP压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============ 启动和关闭事件 ============

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("AI量化交易系统 - FastAPI服务启动中...")
    app_state.startup_time = datetime.utcnow()
    
    try:
        # 初始化数据管理器
        logger.info("初始化数据管理器...")
        await app_state.data_manager.initialize()
        
        # 初始化真实数据管理器
        logger.info("初始化真实数据管理器...")
        await app_state.real_data_manager.initialize()
        
        # 初始化AI交易引擎
        logger.info("初始化AI交易引擎...")
        await app_state.ai_engine.initialize()
        
        app_state.is_ready = True
        logger.info("✅ AI量化交易系统FastAPI服务启动完成")
        
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        app_state.is_ready = False
        # 不抛出异常，允许系统以降级模式运行

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("AI量化交易系统 - FastAPI服务关闭中...")
    
    try:
        # 停止真实数据流
        if hasattr(app_state.real_data_manager, 'stop_real_data_stream'):
            await app_state.real_data_manager.stop_real_data_stream()
        
        # 关闭数据连接
        if hasattr(app_state.data_manager, 'close'):
            await app_state.data_manager.close()
        
        logger.info("✅ AI量化交易系统FastAPI服务已关闭")
        
    except Exception as e:
        logger.error(f"关闭过程出错: {e}")

# ============ 异常处理器 ============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "internal_server_error",
                "message": "服务器内部错误",
                "detail": str(exc)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# ============ API端点 ============

@app.get("/")
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
                "📊 实时数据处理", 
                "🛡️ 智能风控",
                "📈 Coinglass市场情绪分析"
            ],
            "docs_url": "/docs",
            "status": "ready" if app_state.is_ready else "initializing"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    health_status = {
        "healthy": app_state.is_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": (
            (datetime.utcnow() - app_state.startup_time).total_seconds()
            if app_state.startup_time else 0
        ),
        "components": {}
    }
    
    # 检查数据管理器
    try:
        db_health = await app_state.data_manager.health_check()
        health_status["components"]["database"] = {
            "status": "healthy" if db_health.get("mongodb", {}).get("connected") else "unhealthy",
            "details": db_health
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["healthy"] = False
    
    # 检查真实数据管理器
    try:
        real_data_health = await app_state.real_data_manager.health_check()
        health_status["components"]["real_data"] = {
            "status": "healthy",
            "details": real_data_health
        }
    except Exception as e:
        health_status["components"]["real_data"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    return {
        "success": True,
        "data": health_status
    }

@app.get("/api/v1/market/latest")
async def get_latest_market_data():
    """获取最新市场数据"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    try:
        # 获取最新价格
        latest_prices = await app_state.real_data_manager.get_latest_prices()
        
        # 获取连接状态
        connection_status = app_state.real_data_manager.get_connection_status()
        
        return {
            "success": True,
            "data": {
                "prices": latest_prices,
                "connections": connection_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ai/coinglass-analysis")
async def get_coinglass_analysis():
    """获取Coinglass分析结果"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    try:
        analysis = await app_state.real_data_manager.get_coinglass_analysis()
        
        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取Coinglass分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ai/analyze-market")
async def analyze_market():
    """执行AI市场分析"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    try:
        # 获取市场数据进行分析
        market_data = await app_state.real_data_manager.get_latest_prices()
        coinglass_data = await app_state.real_data_manager.get_coinglass_analysis()
        
        # 执行AI分析（这里可以集成更复杂的AI逻辑）
        analysis_result = {
            "market_sentiment": coinglass_data.get("sentiment", {}),
            "technical_analysis": {
                "trend": "neutral",
                "strength": "moderate",
                "recommendation": "hold"
            },
            "risk_assessment": {
                "level": coinglass_data.get("composite_signal", {}).get("risk_assessment", "moderate"),
                "factors": coinglass_data.get("composite_signal", {}).get("key_factors", [])
            }
        }
        
        return {
            "success": True,
            "data": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI市场分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/status")
async def get_system_status():
    """获取系统状态"""
    try:
        # 获取各组件状态
        health = await app_state.real_data_manager.health_check()
        
        system_status = {
            "service_ready": app_state.is_ready,
            "uptime_seconds": (
                (datetime.utcnow() - app_state.startup_time).total_seconds()
                if app_state.startup_time else 0
            ),
            "components": {
                "real_data_manager": {
                    "running": health.get("running", False),
                    "websockets": health.get("websockets", {}),
                    "coinglass": health.get("coinglass", {})
                },
                "database": health.get("database", {}),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }
        
        return {
            "success": True,
            "data": system_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/start-real-stream")
async def start_real_data_stream():
    """启动实时数据流"""
    if not app_state.is_ready:
        raise HTTPException(status_code=503, detail="系统尚未就绪")
    
    try:
        await app_state.real_data_manager.start_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流已启动",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动实时数据流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/data/stop-real-stream")
async def stop_real_data_stream():
    """停止实时数据流"""
    try:
        await app_state.real_data_manager.stop_real_data_stream()
        
        return {
            "success": True,
            "message": "实时数据流已停止",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"停止实时数据流失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ WebSocket支持 ============

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket):
    """市场数据WebSocket端点"""
    await websocket.accept()
    logger.info(f"WebSocket连接建立: market-data")
    
    try:
        while True:
            if app_state.is_ready:
                # 获取最新市场数据
                market_data = await app_state.real_data_manager.get_latest_prices()
                coinglass_data = await app_state.real_data_manager.get_coinglass_analysis()
                
                await websocket.send_json({
                    "type": "market_update",
                    "data": {
                        "prices": market_data,
                        "sentiment": coinglass_data.get("sentiment", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            
            await asyncio.sleep(1)  # 1秒更新一次
            
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        logger.info("市场数据WebSocket连接关闭")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动简化版FastAPI服务器...")
    
    uvicorn.run(
        "simplified_main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )