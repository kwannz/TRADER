"""
回测系统API路由
提供完整的回测配置、运行和结果查询接口
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
import asyncio
import json

# 添加项目路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.trading_simulator import trading_simulator
from core.backtest.backtest_engine import BacktestEngine, BacktestConfig
from core.backtest.trading_environment import TradingEnvironment
from core.backtest.data_replay_system import DataReplaySystem, DataSource
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ============ Pydantic Models ============

class BacktestConfigRequest(BaseModel):
    """回测配置请求"""
    start_date: datetime = Field(..., description="回测开始日期")
    end_date: datetime = Field(..., description="回测结束日期")
    initial_cash: float = Field(100000.0, description="初始资金", gt=0)
    symbols: List[str] = Field(["BTC-USDT", "ETH-USDT"], description="交易品种列表")
    data_source: str = Field("simulated", description="数据源类型")
    commission_rate: float = Field(0.001, description="手续费率", ge=0, le=1)
    slippage_bps: float = Field(1.0, description="滑点(基点)", ge=0)
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("结束日期必须晚于开始日期")
        return v

class StrategyConfigRequest(BaseModel):
    """策略配置请求"""
    strategy_name: str = Field(..., description="策略名称")
    strategy_params: Dict[str, Any] = Field({}, description="策略参数")
    enabled: bool = Field(True, description="是否启用")

class BacktestStartRequest(BaseModel):
    """启动回测请求"""
    config: BacktestConfigRequest
    strategies: List[StrategyConfigRequest]
    name: Optional[str] = Field(None, description="回测名称")
    description: Optional[str] = Field(None, description="回测描述")

class BacktestStatusResponse(BaseModel):
    """回测状态响应"""
    backtest_id: Optional[str]
    status: str
    progress: float
    start_time: Optional[datetime]
    elapsed_time: Optional[float]
    estimated_remaining: Optional[float]
    current_date: Optional[datetime]
    processed_events: int
    total_events: Optional[int]
    error_message: Optional[str]

class BacktestResultResponse(BaseModel):
    """回测结果响应"""
    backtest_id: str
    config: Dict[str, Any]
    status: str
    performance_metrics: Optional[Dict[str, Any]]
    portfolio_value_history: Optional[List[Dict]]
    trade_history: Optional[List[Dict]]
    risk_metrics: Optional[Dict[str, Any]]
    strategy_performance: Optional[Dict[str, Any]]
    execution_time: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

# ============ API端点 ============

@router.get("/status", response_model=BacktestStatusResponse)
async def get_backtest_status():
    """获取回测系统当前状态"""
    try:
        status = trading_simulator.get_backtest_status()
        
        return BacktestStatusResponse(
            backtest_id=status.get('backtest_id'),
            status=status.get('status', 'idle'),
            progress=status.get('progress', 0.0),
            start_time=status.get('start_time'),
            elapsed_time=status.get('elapsed_time'),
            estimated_remaining=status.get('estimated_remaining'),
            current_date=status.get('current_date'),
            processed_events=status.get('processed_events', 0),
            total_events=status.get('total_events'),
            error_message=status.get('error_message')
        )
        
    except Exception as e:
        logger.error(f"获取回测状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/configure")
async def configure_backtest(config_request: BacktestConfigRequest):
    """配置回测参数"""
    try:
        # 转换为BacktestConfig
        backtest_config = BacktestConfig(
            start_date=config_request.start_date,
            end_date=config_request.end_date,
            initial_cash=config_request.initial_cash,
            symbols=config_request.symbols,
            data_source=DataSource(config_request.data_source),
            commission_rate=config_request.commission_rate,
            slippage_bps=config_request.slippage_bps
        )
        
        # 配置交易模拟器
        await trading_simulator.configure_backtest(backtest_config)
        
        logger.info(f"回测配置成功: {config_request.symbols}, {config_request.start_date} - {config_request.end_date}")
        
        return {
            "success": True,
            "message": "回测配置成功",
            "config": {
                "start_date": config_request.start_date,
                "end_date": config_request.end_date,
                "symbols": config_request.symbols,
                "initial_cash": config_request.initial_cash,
                "commission_rate": config_request.commission_rate,
                "slippage_bps": config_request.slippage_bps
            }
        }
        
    except Exception as e:
        logger.error(f"配置回测失败: {e}")
        raise HTTPException(status_code=400, detail=f"配置失败: {str(e)}")

@router.post("/strategies/add")
async def add_strategy(strategy_request: StrategyConfigRequest):
    """添加回测策略"""
    try:
        # 这里需要根据策略名称创建具体的策略实例
        # 为简化，暂时使用模拟策略
        strategy_id = await trading_simulator.add_strategy(
            strategy_name=strategy_request.strategy_name,
            strategy_params=strategy_request.strategy_params,
            enabled=strategy_request.enabled
        )
        
        logger.info(f"添加策略成功: {strategy_request.strategy_name} -> {strategy_id}")
        
        return {
            "success": True,
            "message": "策略添加成功",
            "strategy_id": strategy_id,
            "strategy_name": strategy_request.strategy_name
        }
        
    except Exception as e:
        logger.error(f"添加策略失败: {e}")
        raise HTTPException(status_code=400, detail=f"添加策略失败: {str(e)}")

@router.get("/strategies")
async def list_strategies():
    """获取已配置的策略列表"""
    try:
        strategies = []
        for strategy_id, strategy in trading_simulator.strategies.items():
            strategies.append({
                "strategy_id": strategy_id,
                "strategy_name": getattr(strategy, 'name', 'Unknown'),
                "enabled": getattr(strategy, 'enabled', True),
                "params": getattr(strategy, 'params', {})
            })
        
        return {
            "success": True,
            "strategies": strategies,
            "total_count": len(strategies)
        }
        
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略列表失败: {str(e)}")

@router.delete("/strategies/{strategy_id}")
async def remove_strategy(strategy_id: str):
    """移除策略"""
    try:
        removed = await trading_simulator.remove_strategy(strategy_id)
        
        if removed:
            logger.info(f"移除策略成功: {strategy_id}")
            return {
                "success": True,
                "message": f"策略 {strategy_id} 已移除"
            }
        else:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"移除策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"移除策略失败: {str(e)}")

@router.post("/start")
async def start_backtest(
    background_tasks: BackgroundTasks,
    start_request: BacktestStartRequest
):
    """启动回测"""
    try:
        # 检查是否已在运行
        current_status = trading_simulator.get_backtest_status()
        if current_status.get('status') == 'running':
            raise HTTPException(status_code=400, detail="回测已在运行中")
        
        # 配置回测
        backtest_config = BacktestConfig(
            start_date=start_request.config.start_date,
            end_date=start_request.config.end_date,
            initial_cash=start_request.config.initial_cash,
            symbols=start_request.config.symbols,
            data_source=DataSource(start_request.config.data_source),
            commission_rate=start_request.config.commission_rate,
            slippage_bps=start_request.config.slippage_bps
        )
        
        await trading_simulator.configure_backtest(backtest_config)
        
        # 添加策略
        for strategy_config in start_request.strategies:
            await trading_simulator.add_strategy(
                strategy_name=strategy_config.strategy_name,
                strategy_params=strategy_config.strategy_params,
                enabled=strategy_config.enabled
            )
        
        # 后台启动回测
        background_tasks.add_task(run_backtest_background)
        
        backtest_id = f"bt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"回测启动: {backtest_id}")
        
        return {
            "success": True,
            "message": "回测已启动",
            "backtest_id": backtest_id,
            "config": {
                "start_date": start_request.config.start_date,
                "end_date": start_request.config.end_date,
                "symbols": start_request.config.symbols,
                "strategies_count": len(start_request.strategies)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动回测失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动回测失败: {str(e)}")

async def run_backtest_background():
    """后台运行回测"""
    try:
        logger.info("开始后台回测执行")
        result = await trading_simulator.run_backtest()
        logger.info(f"回测完成: {result is not None}")
        
    except Exception as e:
        logger.error(f"后台回测执行失败: {e}")

@router.post("/stop")
async def stop_backtest():
    """停止回测"""
    try:
        stopped = await trading_simulator.stop_backtest()
        
        if stopped:
            logger.info("回测已停止")
            return {
                "success": True,
                "message": "回测已停止"
            }
        else:
            return {
                "success": False,
                "message": "没有正在运行的回测"
            }
            
    except Exception as e:
        logger.error(f"停止回测失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止回测失败: {str(e)}")

@router.get("/results/latest", response_model=BacktestResultResponse)
async def get_latest_results():
    """获取最新的回测结果"""
    try:
        results = await trading_simulator.get_backtest_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="没有找到回测结果")
        
        return BacktestResultResponse(
            backtest_id=results.get('backtest_id', 'unknown'),
            config=results.get('config', {}),
            status=results.get('status', 'unknown'),
            performance_metrics=results.get('performance_metrics'),
            portfolio_value_history=results.get('portfolio_value_history'),
            trade_history=results.get('trade_history'),
            risk_metrics=results.get('risk_metrics'),
            strategy_performance=results.get('strategy_performance'),
            execution_time=results.get('execution_time'),
            created_at=results.get('created_at', datetime.utcnow()),
            completed_at=results.get('completed_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")

@router.get("/results/summary")
async def get_results_summary():
    """获取回测结果摘要"""
    try:
        results = await trading_simulator.get_backtest_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="没有找到回测结果")
        
        # 提取关键指标
        performance = results.get('performance_metrics', {})
        
        summary = {
            "backtest_id": results.get('backtest_id'),
            "status": results.get('status'),
            "duration_days": (results.get('config', {}).get('end_date', datetime.utcnow()) - 
                            results.get('config', {}).get('start_date', datetime.utcnow())).days,
            "total_return": performance.get('total_return', 0.0),
            "annualized_return": performance.get('annualized_return', 0.0),
            "max_drawdown": performance.get('max_drawdown', 0.0),
            "sharpe_ratio": performance.get('sharpe_ratio', 0.0),
            "total_trades": len(results.get('trade_history', [])),
            "win_rate": performance.get('win_rate', 0.0),
            "final_portfolio_value": performance.get('final_value', 0.0)
        }
        
        return {
            "success": True,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取结果摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取摘要失败: {str(e)}")

@router.get("/results/trades")
async def get_trade_history(
    limit: int = Query(100, ge=1, le=1000, description="返回记录数量"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """获取交易历史"""
    try:
        results = await trading_simulator.get_backtest_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="没有找到回测结果")
        
        trade_history = results.get('trade_history', [])
        
        # 分页处理
        total_trades = len(trade_history)
        paginated_trades = trade_history[offset:offset + limit]
        
        return {
            "success": True,
            "trades": paginated_trades,
            "pagination": {
                "total": total_trades,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_trades
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取交易历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取交易历史失败: {str(e)}")

@router.get("/results/portfolio-value")
async def get_portfolio_value_history():
    """获取投资组合价值历史"""
    try:
        results = await trading_simulator.get_backtest_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="没有找到回测结果")
        
        value_history = results.get('portfolio_value_history', [])
        
        return {
            "success": True,
            "value_history": value_history,
            "data_points": len(value_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取投资组合历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取投资组合历史失败: {str(e)}")

@router.get("/results/risk-metrics")
async def get_risk_metrics():
    """获取风险指标"""
    try:
        results = await trading_simulator.get_backtest_results()
        
        if not results:
            raise HTTPException(status_code=404, detail="没有找到回测结果")
        
        risk_metrics = results.get('risk_metrics', {})
        
        return {
            "success": True,
            "risk_metrics": risk_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取风险指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取风险指标失败: {str(e)}")

@router.delete("/clear")
async def clear_backtest_data():
    """清除回测数据"""
    try:
        await trading_simulator.clear_backtest_data()
        
        logger.info("回测数据已清除")
        
        return {
            "success": True,
            "message": "回测数据已清除"
        }
        
    except Exception as e:
        logger.error(f"清除回测数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除数据失败: {str(e)}")

@router.get("/health")
async def backtest_health_check():
    """回测系统健康检查"""
    try:
        # 检查交易模拟器状态
        simulator_status = trading_simulator.is_running
        mode = trading_simulator.mode
        
        # 检查是否有配置的回测引擎
        has_backtest_engine = trading_simulator.backtest_engine is not None
        
        # 检查策略数量
        strategies_count = len(trading_simulator.strategies)
        
        health_status = {
            "status": "healthy",
            "simulator_running": simulator_status,
            "current_mode": mode,
            "backtest_engine_configured": has_backtest_engine,
            "strategies_count": strategies_count,
            "timestamp": datetime.utcnow()
        }
        
        return {
            "success": True,
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"回测系统健康检查失败: {e}")
        return {
            "success": False,
            "health": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
        }