"""
简化的FastAPI测试版本
验证核心系统功能
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from core.real_data_manager import real_data_manager
from core.ai_trading_engine import ai_trading_engine
from core.data_manager import data_manager

# 创建应用
app = FastAPI(
    title="AI量化交易系统 - 测试版", 
    description="核心功能测试API",
    version="1.0.0"
)

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
class AppState:
    def __init__(self):
        self.initialized = False
        self.startup_time = None

app_state = AppState()

@app.on_event("startup")
async def startup_event():
    """应用启动"""
    print("🚀 启动AI量化交易系统...")
    app_state.startup_time = datetime.utcnow()
    
    try:
        # 初始化数据管理器
        print("📊 初始化数据管理器...")
        await data_manager.initialize()
        
        # 初始化真实数据管理器
        print("📡 初始化真实数据管理器...")
        await real_data_manager.initialize()
        
        # 初始化AI交易引擎
        print("🤖 初始化AI交易引擎...")
        await ai_trading_engine.initialize()
        
        app_state.initialized = True
        print("✅ 系统启动完成!")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        app_state.initialized = False

@app.on_event("shutdown")  
async def shutdown_event():
    """应用关闭"""
    print("🔴 关闭AI量化交易系统...")
    
    try:
        if hasattr(real_data_manager, 'stop_real_data_stream'):
            await real_data_manager.stop_real_data_stream()
        
        if hasattr(data_manager, 'close'):
            await data_manager.close()
        
        print("✅ 系统已安全关闭")
        
    except Exception as e:
        print(f"❌ 关闭过程出错: {e}")

@app.get("/")
async def root():
    """根端点"""
    return {
        "name": "AI量化交易系统",
        "version": "1.0.0",
        "status": "ready" if app_state.initialized else "initializing",
        "features": [
            "🤖 AI智能分析",
            "📊 实时数据处理",
            "🛡️ 智能风控", 
            "📈 Coinglass市场情绪"
        ],
        "docs": "/docs",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        health = await real_data_manager.health_check()
        
        return {
            "healthy": True,
            "uptime_seconds": (
                (datetime.utcnow() - app_state.startup_time).total_seconds()
                if app_state.startup_time else 0
            ),
            "components": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/market/latest")
async def get_market_data():
    """获取市场数据"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        prices = await real_data_manager.get_latest_prices()
        connections = real_data_manager.get_connection_status()
        
        return {
            "success": True,
            "data": {
                "prices": prices,
                "connections": connections
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/coinglass")
async def get_coinglass_analysis():
    """获取Coinglass分析"""
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/start-stream")
async def start_data_stream():
    """启动数据流"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        if real_data_manager.is_running:
            return {
                "success": True,
                "message": "数据流已在运行",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await real_data_manager.start_real_data_stream()
        
        return {
            "success": True,
            "message": "数据流启动成功",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/stop-stream")
async def stop_data_stream():
    """停止数据流"""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    try:
        await real_data_manager.stop_real_data_stream()
        
        return {
            "success": True,
            "message": "数据流已停止",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态"""
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
                "health": health
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/market")
async def websocket_market_data(websocket):
    """市场数据WebSocket"""
    await websocket.accept()
    print("WebSocket连接建立")
    
    try:
        while True:
            if app_state.initialized:
                # 获取最新数据
                prices = await real_data_manager.get_latest_prices()
                coinglass = await real_data_manager.get_coinglass_analysis()
                
                await websocket.send_json({
                    "type": "market_update",
                    "data": {
                        "prices": prices,
                        "sentiment": coinglass.get("sentiment", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
            
            await asyncio.sleep(2)  # 2秒更新一次
            
    except Exception as e:
        print(f"WebSocket错误: {e}")
    finally:
        print("WebSocket连接关闭")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 启动简化版AI量化交易系统...")
    
    uvicorn.run(
        "test_fastapi_simple:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )