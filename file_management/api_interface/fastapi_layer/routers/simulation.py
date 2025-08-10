"""
仿真交易系统API路由
提供实时市场数据仿真和交易模拟接口
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio
import json

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.trading_simulator import trading_simulator
from core.simulation.slippage_model import SlippageType
from core.simulation.latency_simulator import LatencyType
from core.simulation.market_maker import MarketMakerStrategy
from core.simulation.liquidity_provider import LiquidityStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ============ Pydantic Models ============

class SimulationConfigRequest(BaseModel):
    """仿真配置请求"""
    symbols: List[str] = Field(["BTC-USDT", "ETH-USDT"], description="仿真品种")
    initial_prices: Optional[Dict[str, float]] = Field(None, description="初始价格")
    tick_frequency_hz: float = Field(10.0, description="Tick频率", gt=0, le=100)
    enable_market_maker: bool = Field(True, description="启用做市商")
    enable_liquidity_provider: bool = Field(True, description="启用流动性提供者")
    enable_volatility_events: bool = Field(True, description="启用波动率事件")
    enable_news_events: bool = Field(False, description="启用新闻事件")

class SlippageConfigRequest(BaseModel):
    """滑点配置请求"""
    model: str = Field("linear", description="滑点模型")
    base_bps: float = Field(1.0, description="基础滑点(基点)", ge=0)
    size_impact_factor: float = Field(0.1, description="规模影响因子", ge=0)

class LatencyConfigRequest(BaseModel):
    """延迟配置请求"""
    model: str = Field("fixed", description="延迟模型")
    base_latency_ms: float = Field(10.0, description="基础延迟(毫秒)", ge=0)
    processing_latency_ms: float = Field(5.0, description="处理延迟", ge=0)

class MarketEventRequest(BaseModel):
    """市场事件请求"""
    event_type: str = Field(..., description="事件类型")
    symbol: str = Field(..., description="影响品种")
    intensity: float = Field(1.0, description="事件强度", ge=0, le=10)
    duration_seconds: int = Field(60, description="持续时间", gt=0)

class SimulationStatusResponse(BaseModel):
    """仿真状态响应"""
    is_running: bool
    mode: str
    uptime_seconds: float
    symbols_count: int
    active_symbols: List[str]
    tick_frequency_hz: float
    total_ticks_generated: int
    market_makers_active: int
    liquidity_providers_active: int
    current_prices: Dict[str, float]

# ============ API端点 ============

@router.get("/status", response_model=SimulationStatusResponse)
async def get_simulation_status():
    """获取仿真系统状态"""
    try:
        # 获取市场摘要
        market_summary = trading_simulator.get_market_summary()
        
        return SimulationStatusResponse(
            is_running=trading_simulator.is_running,
            mode=trading_simulator.mode,
            uptime_seconds=trading_simulator.get_uptime_seconds(),
            symbols_count=len(trading_simulator.symbols),
            active_symbols=list(trading_simulator.symbols.keys()),
            tick_frequency_hz=10.0,  # 从配置获取
            total_ticks_generated=trading_simulator.get_total_ticks_generated(),
            market_makers_active=len(trading_simulator.market_makers),
            liquidity_providers_active=len(trading_simulator.liquidity_providers),
            current_prices={symbol: data.get('price', 0) for symbol, data in market_summary.items()}
        )
        
    except Exception as e:
        logger.error(f"获取仿真状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/configure")
async def configure_simulation(config: SimulationConfigRequest):
    """配置仿真参数"""
    try:
        # 切换到仿真模式
        await trading_simulator.set_mode("simulation")
        
        # 配置仿真参数
        await trading_simulator.configure_simulation({
            "symbols": config.symbols,
            "initial_prices": config.initial_prices or {},
            "tick_frequency_hz": config.tick_frequency_hz,
            "enable_market_maker": config.enable_market_maker,
            "enable_liquidity_provider": config.enable_liquidity_provider,
            "enable_volatility_events": config.enable_volatility_events,
            "enable_news_events": config.enable_news_events
        })
        
        logger.info(f"仿真配置成功: {config.symbols}, 频率={config.tick_frequency_hz}Hz")
        
        return {
            "success": True,
            "message": "仿真配置成功",
            "config": {
                "symbols": config.symbols,
                "tick_frequency_hz": config.tick_frequency_hz,
                "features_enabled": {
                    "market_maker": config.enable_market_maker,
                    "liquidity_provider": config.enable_liquidity_provider,
                    "volatility_events": config.enable_volatility_events,
                    "news_events": config.enable_news_events
                }
            }
        }
        
    except Exception as e:
        logger.error(f"配置仿真失败: {e}")
        raise HTTPException(status_code=400, detail=f"配置失败: {str(e)}")

@router.post("/start")
async def start_simulation():
    """启动仿真"""
    try:
        if trading_simulator.is_running:
            return {
                "success": False,
                "message": "仿真已在运行中"
            }
        
        await trading_simulator.start_simulation()
        
        logger.info("仿真交易启动成功")
        
        return {
            "success": True,
            "message": "仿真交易已启动",
            "symbols": list(trading_simulator.symbols.keys()),
            "start_time": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"启动仿真失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动仿真失败: {str(e)}")

@router.post("/stop")
async def stop_simulation():
    """停止仿真"""
    try:
        if not trading_simulator.is_running:
            return {
                "success": False,
                "message": "仿真未在运行"
            }
        
        await trading_simulator.stop_simulation()
        
        logger.info("仿真交易已停止")
        
        return {
            "success": True,
            "message": "仿真交易已停止",
            "stop_time": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"停止仿真失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止仿真失败: {str(e)}")

@router.get("/market-summary")
async def get_market_summary():
    """获取市场数据摘要"""
    try:
        market_summary = trading_simulator.get_market_summary()
        
        return {
            "success": True,
            "market_data": market_summary,
            "symbols_count": len(market_summary),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取市场摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取市场摘要失败: {str(e)}")

@router.get("/market-data/{symbol}")
async def get_symbol_data(symbol: str):
    """获取特定品种的市场数据"""
    try:
        market_summary = trading_simulator.get_market_summary()
        
        if symbol not in market_summary:
            raise HTTPException(status_code=404, detail=f"品种 {symbol} 不存在")
        
        symbol_data = market_summary[symbol]
        
        return {
            "success": True,
            "symbol": symbol,
            "data": symbol_data,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取品种数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")

@router.post("/slippage/configure")
async def configure_slippage(config: SlippageConfigRequest):
    """配置滑点模型"""
    try:
        # 解析滑点模型
        slippage_model = SlippageType[config.model.upper()]
        
        # 配置滑点参数
        slippage_config = {
            "model": slippage_model,
            "base_bps": config.base_bps,
            "size_impact_factor": config.size_impact_factor
        }
        
        await trading_simulator.configure_slippage(slippage_config)
        
        logger.info(f"滑点配置成功: {config.model}, 基础={config.base_bps}bps")
        
        return {
            "success": True,
            "message": "滑点模型配置成功",
            "config": slippage_config
        }
        
    except KeyError:
        raise HTTPException(status_code=400, detail=f"不支持的滑点模型: {config.model}")
    except Exception as e:
        logger.error(f"配置滑点失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置滑点失败: {str(e)}")

@router.post("/latency/configure")
async def configure_latency(config: LatencyConfigRequest):
    """配置延迟模型"""
    try:
        # 解析延迟模型
        latency_model = LatencyType[config.model.upper()]
        
        # 配置延迟参数
        latency_config = {
            "model": latency_model,
            "base_latency_ms": config.base_latency_ms,
            "processing_latency_ms": config.processing_latency_ms
        }
        
        await trading_simulator.configure_latency(latency_config)
        
        logger.info(f"延迟配置成功: {config.model}, 基础={config.base_latency_ms}ms")
        
        return {
            "success": True,
            "message": "延迟模型配置成功",
            "config": latency_config
        }
        
    except KeyError:
        raise HTTPException(status_code=400, detail=f"不支持的延迟模型: {config.model}")
    except Exception as e:
        logger.error(f"配置延迟失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置延迟失败: {str(e)}")

@router.get("/market-makers")
async def get_market_makers_status():
    """获取做市商状态"""
    try:
        market_makers_status = []
        
        for mm_id, market_maker in trading_simulator.market_makers.items():
            status = market_maker.get_market_maker_metrics()
            market_makers_status.append({
                "mm_id": mm_id,
                **status
            })
        
        return {
            "success": True,
            "market_makers": market_makers_status,
            "total_count": len(market_makers_status)
        }
        
    except Exception as e:
        logger.error(f"获取做市商状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.get("/liquidity-providers")
async def get_liquidity_providers_status():
    """获取流动性提供者状态"""
    try:
        lp_status = []
        
        for lp_id, liquidity_provider in trading_simulator.liquidity_providers.items():
            status = liquidity_provider.get_liquidity_status()
            lp_status.append({
                "lp_id": lp_id,
                **status
            })
        
        return {
            "success": True,
            "liquidity_providers": lp_status,
            "total_count": len(lp_status)
        }
        
    except Exception as e:
        logger.error(f"获取流动性提供者状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/market-event")
async def trigger_market_event(event: MarketEventRequest):
    """触发市场事件"""
    try:
        # 触发市场事件
        event_id = await trading_simulator.trigger_market_event({
            "event_type": event.event_type,
            "symbol": event.symbol,
            "intensity": event.intensity,
            "duration_seconds": event.duration_seconds
        })
        
        logger.info(f"市场事件已触发: {event.event_type} for {event.symbol}")
        
        return {
            "success": True,
            "message": "市场事件已触发",
            "event_id": event_id,
            "event_info": {
                "type": event.event_type,
                "symbol": event.symbol,
                "intensity": event.intensity,
                "duration": event.duration_seconds
            }
        }
        
    except Exception as e:
        logger.error(f"触发市场事件失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发事件失败: {str(e)}")

@router.get("/statistics")
async def get_simulation_statistics():
    """获取仿真统计数据"""
    try:
        stats = {
            "general": {
                "is_running": trading_simulator.is_running,
                "uptime_seconds": trading_simulator.get_uptime_seconds(),
                "mode": trading_simulator.mode,
                "symbols_count": len(trading_simulator.symbols)
            },
            "market_data": {
                "total_ticks_generated": trading_simulator.get_total_ticks_generated(),
                "tick_frequency_hz": 10.0,
                "active_price_generators": len(trading_simulator.price_generators)
            },
            "market_participants": {
                "market_makers_count": len(trading_simulator.market_makers),
                "liquidity_providers_count": len(trading_simulator.liquidity_providers)
            },
            "trading": {
                "total_orders": 0,  # 需要实现
                "total_trades": 0,  # 需要实现
                "total_volume": 0.0  # 需要实现
            }
        }
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")

@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """实时市场数据WebSocket"""
    await websocket.accept()
    logger.info(f"仿真市场数据WebSocket连接建立: {websocket.client}")
    
    try:
        while True:
            if trading_simulator.is_running:
                # 获取最新市场数据
                market_summary = trading_simulator.get_market_summary()
                
                # 发送数据
                await websocket.send_json({
                    "type": "market_data",
                    "data": market_summary,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(0.1)  # 10Hz 更新频率
            
    except WebSocketDisconnect:
        logger.info("仿真市场数据WebSocket连接断开")
    except Exception as e:
        logger.error(f"仿真市场数据WebSocket错误: {e}")
    finally:
        logger.info("仿真市场数据WebSocket连接关闭")

@router.get("/health")
async def simulation_health_check():
    """仿真系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "mode": trading_simulator.mode,
            "is_running": trading_simulator.is_running,
            "symbols_configured": len(trading_simulator.symbols) > 0,
            "price_generators_active": len(trading_simulator.price_generators) > 0,
            "timestamp": datetime.utcnow()
        }
        
        # 检查是否有组件出现问题
        if trading_simulator.mode != "simulation":
            health_status["status"] = "warning"
            health_status["message"] = "不在仿真模式"
        
        return {
            "success": True,
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"仿真系统健康检查失败: {e}")
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
        }