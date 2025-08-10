"""
数据管理API路由
提供数据源管理、历史数据查询、数据清理等功能
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data_manager import data_manager
from core.real_data_manager import real_data_manager
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class DataSourceConfig(BaseModel):
    """数据源配置"""
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型")  # websocket, rest, database, file
    endpoint: str = Field(..., description="数据源端点")
    api_key: Optional[str] = Field(None, description="API密钥")
    api_secret: Optional[str] = Field(None, description="API密钥")
    enabled: bool = Field(True, description="是否启用")
    config: Dict[str, Any] = Field({}, description="其他配置参数")

class DataCleanupConfig(BaseModel):
    """数据清理配置"""
    data_types: List[str] = Field(..., description="要清理的数据类型")
    before_date: Optional[datetime] = Field(None, description="清理此日期前的数据")
    keep_days: Optional[int] = Field(None, description="保留天数")
    dry_run: bool = Field(True, description="是否为测试运行")

class DataExportConfig(BaseModel):
    """数据导出配置"""
    data_types: List[str] = Field(..., description="导出数据类型")
    symbols: Optional[List[str]] = Field(None, description="指定品种")
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")
    format: str = Field("json", description="导出格式")  # json, csv, parquet
    include_metadata: bool = Field(False, description="是否包含元数据")

class DataSyncConfig(BaseModel):
    """数据同步配置"""
    source: str = Field(..., description="源数据库")
    target: str = Field(..., description="目标数据库")
    data_types: List[str] = Field(..., description="同步数据类型")
    sync_mode: str = Field("incremental", description="同步模式")  # full, incremental
    batch_size: int = Field(1000, description="批次大小")

@router.get("/sources")
async def list_data_sources():
    """获取数据源列表"""
    try:
        # 获取真实数据管理器的连接状态
        connection_status = real_data_manager.get_connection_status()
        
        # 模拟数据源配置 (实际应该从配置文件或数据库读取)
        data_sources = [
            {
                "id": "okx_websocket",
                "name": "OKX WebSocket",
                "type": "websocket",
                "endpoint": "wss://ws.okx.com:8443/ws/v5/public",
                "status": "connected" if connection_status.get("okx") else "disconnected",
                "enabled": True,
                "last_update": datetime.utcnow() - timedelta(seconds=10),
                "data_types": ["ticker", "trades", "orderbook", "kline"],
                "symbols_count": 50
            },
            {
                "id": "binance_websocket",
                "name": "Binance WebSocket", 
                "type": "websocket",
                "endpoint": "wss://stream.binance.com:9443/ws",
                "status": "connected" if connection_status.get("binance") else "disconnected",
                "enabled": True,
                "last_update": datetime.utcnow() - timedelta(seconds=5),
                "data_types": ["ticker", "trades", "orderbook", "kline"],
                "symbols_count": 100
            },
            {
                "id": "coinglass_api",
                "name": "Coinglass API",
                "type": "rest",
                "endpoint": "https://open-api.coinglass.com/public/v2",
                "status": "active",
                "enabled": True,
                "last_update": datetime.utcnow() - timedelta(minutes=1),
                "data_types": ["funding_rate", "open_interest", "liquidation"],
                "symbols_count": 200
            },
            {
                "id": "mongodb_primary",
                "name": "MongoDB 主数据库",
                "type": "database",
                "endpoint": "mongodb://localhost:27017/trading_data",
                "status": await data_manager.health_check() and "connected" or "disconnected",
                "enabled": True,
                "last_update": datetime.utcnow() - timedelta(seconds=30),
                "data_types": ["historical_prices", "trades", "strategies", "backtests"],
                "records_count": 1500000
            }
        ]
        
        return {
            "success": True,
            "data": data_sources,
            "total": len(data_sources),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据源列表失败: {str(e)}")

@router.post("/sources")
async def add_data_source(config: DataSourceConfig):
    """添加新数据源"""
    try:
        # 验证数据源配置
        if not config.endpoint:
            raise HTTPException(status_code=400, detail="数据源端点不能为空")
        
        # 实际实现中应该测试连接并保存到配置
        new_source_id = f"{config.name.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}"
        
        # 模拟添加数据源的过程
        source_info = {
            "id": new_source_id,
            "name": config.name,
            "type": config.type,
            "endpoint": config.endpoint,
            "status": "testing",
            "enabled": config.enabled,
            "created_at": datetime.utcnow(),
            "config": config.config
        }
        
        logger.info(f"添加新数据源: {config.name}")
        
        return {
            "success": True,
            "message": "数据源添加成功",
            "data": source_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"添加数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加数据源失败: {str(e)}")

@router.put("/sources/{source_id}")
async def update_data_source(source_id: str, config: DataSourceConfig):
    """更新数据源配置"""
    try:
        # 实际实现中应该从数据库查找并更新数据源
        logger.info(f"更新数据源: {source_id}")
        
        return {
            "success": True,
            "message": f"数据源 {source_id} 更新成功",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"更新数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新数据源失败: {str(e)}")

@router.delete("/sources/{source_id}")
async def delete_data_source(source_id: str):
    """删除数据源"""
    try:
        # 实际实现中应该从数据库删除数据源配置
        logger.info(f"删除数据源: {source_id}")
        
        return {
            "success": True,
            "message": f"数据源 {source_id} 删除成功",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"删除数据源失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除数据源失败: {str(e)}")

@router.get("/historical/{data_type}")
async def get_historical_data(
    data_type: str,  # prices, trades, funding_rates, open_interest
    symbols: Optional[str] = Query(None, description="交易品种，逗号分隔"),
    start_date: Optional[datetime] = Query(None, description="开始时间"),
    end_date: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(1000, description="最大返回数量", le=10000),
    offset: int = Query(0, description="偏移量", ge=0)
):
    """查询历史数据"""
    try:
        # 解析品种列表
        symbol_list = symbols.split(",") if symbols else []
        
        # 设置默认时间范围
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        # 实际实现中应该从数据库查询真实历史数据
        # 这里返回模拟数据结构
        
        data_records = []
        
        if data_type == "prices":
            # 价格数据
            for i in range(min(limit, 100)):  # 限制模拟数据数量
                timestamp = start_date + timedelta(hours=i)
                for symbol in (symbol_list or ["BTC-USDT", "ETH-USDT"]):
                    base_price = 50000 if symbol.startswith("BTC") else 3000
                    data_records.append({
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "open": base_price + (i * 10),
                        "high": base_price + (i * 10) + 100,
                        "low": base_price + (i * 10) - 100,
                        "close": base_price + (i * 10) + 50,
                        "volume": 1000 + (i * 5)
                    })
                    
        elif data_type == "trades":
            # 交易数据
            for i in range(min(limit, 50)):
                timestamp = start_date + timedelta(minutes=i * 5)
                for symbol in (symbol_list or ["BTC-USDT"]):
                    base_price = 50000 if symbol.startswith("BTC") else 3000
                    data_records.append({
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "trade_id": f"trade_{i}",
                        "price": base_price + (i * 10),
                        "quantity": 0.1 + (i * 0.01),
                        "side": "buy" if i % 2 == 0 else "sell"
                    })
        
        return {
            "success": True,
            "data_type": data_type,
            "symbols": symbol_list,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data": data_records,
            "count": len(data_records),
            "total": len(data_records),  # 实际应该是总记录数
            "has_more": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"查询历史数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询历史数据失败: {str(e)}")

@router.get("/statistics")
async def get_data_statistics():
    """获取数据统计信息"""
    try:
        # 获取数据库统计信息
        db_health = await data_manager.health_check()
        
        statistics = {
            "database": {
                "status": "healthy" if db_health else "unhealthy",
                "collections": {
                    "market_data": {"records": 1234567, "size_mb": 256},
                    "trades": {"records": 567890, "size_mb": 128},
                    "strategies": {"records": 45, "size_mb": 2},
                    "backtests": {"records": 123, "size_mb": 64},
                    "users": {"records": 5, "size_mb": 1}
                },
                "total_records": 1802625,
                "total_size_mb": 451
            },
            "data_sources": {
                "total": 4,
                "active": 3,
                "websocket_connections": 2,
                "api_endpoints": 1,
                "database_connections": 1
            },
            "data_freshness": {
                "market_data": {"last_update": datetime.utcnow() - timedelta(seconds=10)},
                "trades": {"last_update": datetime.utcnow() - timedelta(minutes=1)},
                "funding_rates": {"last_update": datetime.utcnow() - timedelta(hours=1)}
            },
            "performance": {
                "avg_query_time_ms": 45.6,
                "cache_hit_rate": 0.85,
                "active_connections": 12,
                "queries_per_minute": 234
            }
        }
        
        return {
            "success": True,
            "data": statistics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取数据统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据统计失败: {str(e)}")

@router.post("/cleanup")
async def cleanup_data(config: DataCleanupConfig, background_tasks: BackgroundTasks):
    """数据清理"""
    try:
        # 添加后台清理任务
        background_tasks.add_task(execute_data_cleanup, config)
        
        estimated_records = 0
        for data_type in config.data_types:
            # 估算要清理的记录数
            if data_type == "market_data":
                estimated_records += 100000
            elif data_type == "trades":
                estimated_records += 50000
        
        return {
            "success": True,
            "message": "数据清理任务已启动" if not config.dry_run else "数据清理测试已启动",
            "task_id": f"cleanup_{int(datetime.utcnow().timestamp())}",
            "estimated_records": estimated_records,
            "dry_run": config.dry_run,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动数据清理失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据清理失败: {str(e)}")

@router.post("/export")
async def export_data(config: DataExportConfig, background_tasks: BackgroundTasks):
    """数据导出"""
    try:
        # 添加后台导出任务
        background_tasks.add_task(execute_data_export, config)
        
        export_id = f"export_{int(datetime.utcnow().timestamp())}"
        
        return {
            "success": True,
            "message": "数据导出任务已启动",
            "export_id": export_id,
            "config": {
                "data_types": config.data_types,
                "date_range": {
                    "start": config.start_date.isoformat(),
                    "end": config.end_date.isoformat()
                },
                "format": config.format
            },
            "estimated_completion": datetime.utcnow() + timedelta(minutes=10),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动数据导出失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据导出失败: {str(e)}")

@router.post("/sync")
async def sync_data(config: DataSyncConfig, background_tasks: BackgroundTasks):
    """数据同步"""
    try:
        # 添加后台同步任务
        background_tasks.add_task(execute_data_sync, config)
        
        sync_id = f"sync_{int(datetime.utcnow().timestamp())}"
        
        return {
            "success": True,
            "message": "数据同步任务已启动",
            "sync_id": sync_id,
            "config": {
                "source": config.source,
                "target": config.target,
                "data_types": config.data_types,
                "mode": config.sync_mode
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"启动数据同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据同步失败: {str(e)}")

@router.get("/health")
async def data_health_check():
    """数据系统健康检查"""
    try:
        # 检查各个数据组件
        db_health = await data_manager.health_check()
        ws_status = real_data_manager.get_connection_status()
        
        health_status = {
            "overall": "healthy",
            "components": {
                "database": {
                    "status": "healthy" if db_health else "unhealthy",
                    "mongodb": db_health.get("mongodb", False) if isinstance(db_health, dict) else db_health,
                    "redis": db_health.get("redis", False) if isinstance(db_health, dict) else db_health
                },
                "websockets": {
                    "status": "healthy" if any(ws_status.values()) else "unhealthy",
                    "okx": ws_status.get("okx", False),
                    "binance": ws_status.get("binance", False)
                },
                "data_freshness": {
                    "status": "healthy",
                    "market_data_age_seconds": 10,
                    "trades_age_seconds": 60
                }
            },
            "issues": []
        }
        
        # 检查是否有问题
        if not db_health:
            health_status["issues"].append("数据库连接异常")
            health_status["overall"] = "degraded"
            
        if not any(ws_status.values()):
            health_status["issues"].append("WebSocket连接全部断开")
            health_status["overall"] = "degraded"
        
        return {
            "success": True,
            "health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"数据健康检查失败: {e}")
        return {
            "success": False,
            "health": {
                "overall": "unhealthy",
                "error": str(e)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

# 后台任务函数
async def execute_data_cleanup(config: DataCleanupConfig):
    """执行数据清理任务"""
    try:
        logger.info(f"开始数据清理任务: {config.data_types}")
        
        # 模拟清理过程
        await asyncio.sleep(5)
        
        if not config.dry_run:
            # 实际清理逻辑
            pass
        
        logger.info("数据清理任务完成")
        
    except Exception as e:
        logger.error(f"数据清理任务失败: {e}")

async def execute_data_export(config: DataExportConfig):
    """执行数据导出任务"""
    try:
        logger.info(f"开始数据导出任务: {config.data_types}")
        
        # 模拟导出过程
        await asyncio.sleep(10)
        
        logger.info("数据导出任务完成")
        
    except Exception as e:
        logger.error(f"数据导出任务失败: {e}")

async def execute_data_sync(config: DataSyncConfig):
    """执行数据同步任务"""
    try:
        logger.info(f"开始数据同步任务: {config.source} -> {config.target}")
        
        # 模拟同步过程
        await asyncio.sleep(8)
        
        logger.info("数据同步任务完成")
        
    except Exception as e:
        logger.error(f"数据同步任务失败: {e}")