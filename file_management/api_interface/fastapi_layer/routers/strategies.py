"""
策略管理API路由

提供策略的CRUD操作、执行控制、性能分析等功能
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from ..dependencies.auth import get_current_user
from ..dependencies.database import get_database_manager
from ..schemas.strategy import (
    StrategyCreate,
    StrategyUpdate,
    StrategyResponse,
    StrategyListResponse,
    StrategyExecutionRequest,
    StrategyPerformanceResponse,
    StrategyBacktestRequest,
    StrategyBacktestResponse
)
from ..schemas.common import APIResponse, PaginationParams
from ...python_layer.models.strategy import Strategy, StrategyType, StrategyStatus
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ============ 策略CRUD操作 ============

@router.get("/", response_model=StrategyListResponse)
async def get_strategies(
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    pagination: PaginationParams = Depends(),
    strategy_type: Optional[StrategyType] = Query(None, description="按策略类型筛选"),
    status: Optional[StrategyStatus] = Query(None, description="按状态筛选"),
    search: Optional[str] = Query(None, description="搜索策略名称")
):
    """
    获取策略列表
    
    支持分页、筛选和搜索功能
    """
    try:
        # 构建查询条件
        filters = {}
        if strategy_type:
            filters["strategy_type"] = strategy_type.value
        if status:
            filters["status"] = status.value
        if search:
            filters["$text"] = {"$search": search}
        
        # 查询策略
        strategies, total_count = await db_manager.get_strategies_paginated(
            filters=filters,
            skip=pagination.skip,
            limit=pagination.limit,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order
        )
        
        return StrategyListResponse(
            success=True,
            data=strategies,
            pagination={
                "total": total_count,
                "page": pagination.page,
                "size": pagination.size,
                "pages": (total_count + pagination.size - 1) // pagination.size
            },
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """获取单个策略详情"""
    try:
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 获取策略性能数据
        performance = await db_manager.get_strategy_performance(strategy_id)
        
        return StrategyResponse(
            success=True,
            data={
                **strategy,
                "performance": performance
            },
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=StrategyResponse, status_code=201)
async def create_strategy(
    strategy_data: StrategyCreate,
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    创建新策略
    
    支持多种策略类型，自动进行配置验证
    """
    try:
        # 验证策略配置
        await _validate_strategy_config(strategy_data.strategy_type, strategy_data.config)
        
        # 创建策略对象
        strategy = Strategy(
            id=str(uuid4()),
            name=strategy_data.name,
            description=strategy_data.description,
            strategy_type=strategy_data.strategy_type,
            symbol=strategy_data.symbol,
            config=strategy_data.config,
            risk_config=strategy_data.risk_config,
            created_by=current_user["user_id"],
            status=StrategyStatus.INACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # 保存到数据库
        await db_manager.create_strategy(strategy)
        
        # 后台任务：验证策略代码（如果是AI生成策略）
        if strategy.strategy_type == StrategyType.AI_GENERATED:
            background_tasks.add_task(_validate_ai_strategy, strategy.id, db_manager)
        
        logger.info(f"策略创建成功: {strategy.id} - {strategy.name}")
        
        return StrategyResponse(
            success=True,
            data=strategy.dict(),
            message="策略创建成功",
            timestamp=datetime.utcnow()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    strategy_data: StrategyUpdate = None,
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """更新策略配置"""
    try:
        # 检查策略是否存在
        existing_strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not existing_strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 检查策略是否正在运行
        if existing_strategy["status"] == StrategyStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="无法修改正在运行的策略")
        
        # 构建更新数据
        update_data = {
            "updated_at": datetime.utcnow()
        }
        
        if strategy_data.name is not None:
            update_data["name"] = strategy_data.name
        if strategy_data.description is not None:
            update_data["description"] = strategy_data.description
        if strategy_data.config is not None:
            # 验证新配置
            await _validate_strategy_config(
                StrategyType(existing_strategy["strategy_type"]), 
                strategy_data.config
            )
            update_data["config"] = strategy_data.config
        if strategy_data.risk_config is not None:
            update_data["risk_config"] = strategy_data.risk_config
        
        # 更新策略
        updated_strategy = await db_manager.update_strategy(strategy_id, update_data)
        
        logger.info(f"策略更新成功: {strategy_id}")
        
        return StrategyResponse(
            success=True,
            data=updated_strategy,
            message="策略更新成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{strategy_id}", response_model=APIResponse)
async def delete_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """删除策略"""
    try:
        # 检查策略是否存在
        existing_strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not existing_strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 检查策略是否正在运行
        if existing_strategy["status"] == StrategyStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="无法删除正在运行的策略，请先停止策略")
        
        # 删除策略（软删除）
        await db_manager.delete_strategy(strategy_id)
        
        logger.info(f"策略删除成功: {strategy_id}")
        
        return APIResponse(
            success=True,
            message="策略删除成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ 策略执行控制 ============

@router.post("/{strategy_id}/start", response_model=APIResponse)
async def start_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    execution_request: StrategyExecutionRequest = None,
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """启动策略执行"""
    try:
        # 检查策略是否存在
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 检查策略状态
        if strategy["status"] == StrategyStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="策略已在运行中")
        
        # 执行前置检查
        await _pre_execution_check(strategy, execution_request)
        
        # 更新策略状态
        await db_manager.update_strategy(strategy_id, {
            "status": StrategyStatus.RUNNING.value,
            "started_at": datetime.utcnow(),
            "execution_config": execution_request.dict() if execution_request else {}
        })
        
        # 后台任务：启动策略执行
        background_tasks.add_task(_execute_strategy, strategy_id, db_manager)
        
        logger.info(f"策略启动成功: {strategy_id}")
        
        return APIResponse(
            success=True,
            message="策略启动成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/stop", response_model=APIResponse)
async def stop_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """停止策略执行"""
    try:
        # 检查策略是否存在
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 检查策略状态
        if strategy["status"] != StrategyStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="策略未在运行")
        
        # 停止策略执行（通过信号量）
        await _stop_strategy_execution(strategy_id)
        
        # 更新策略状态
        await db_manager.update_strategy(strategy_id, {
            "status": StrategyStatus.STOPPED.value,
            "stopped_at": datetime.utcnow()
        })
        
        logger.info(f"策略停止成功: {strategy_id}")
        
        return APIResponse(
            success=True,
            message="策略停止成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/pause", response_model=APIResponse)
async def pause_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """暂停策略执行"""
    try:
        # 检查策略是否存在和运行中
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        if strategy["status"] != StrategyStatus.RUNNING.value:
            raise HTTPException(status_code=400, detail="策略未在运行")
        
        # 暂停策略
        await _pause_strategy_execution(strategy_id)
        
        # 更新策略状态
        await db_manager.update_strategy(strategy_id, {
            "status": StrategyStatus.PAUSED.value,
            "paused_at": datetime.utcnow()
        })
        
        logger.info(f"策略暂停成功: {strategy_id}")
        
        return APIResponse(
            success=True,
            message="策略暂停成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/resume", response_model=APIResponse)
async def resume_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """恢复策略执行"""
    try:
        # 检查策略状态
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        if strategy["status"] != StrategyStatus.PAUSED.value:
            raise HTTPException(status_code=400, detail="策略未暂停")
        
        # 恢复策略执行
        await db_manager.update_strategy(strategy_id, {
            "status": StrategyStatus.RUNNING.value,
            "resumed_at": datetime.utcnow()
        })
        
        # 后台任务：恢复策略执行
        background_tasks.add_task(_resume_strategy_execution, strategy_id, db_manager)
        
        logger.info(f"策略恢复成功: {strategy_id}")
        
        return APIResponse(
            success=True,
            message="策略恢复成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ 策略性能和分析 ============

@router.get("/{strategy_id}/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    strategy_id: str = Path(..., description="策略ID"),
    days: int = Query(30, ge=1, le=365, description="查询天数"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """获取策略性能分析"""
    try:
        # 检查策略是否存在
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 获取性能数据
        start_date = datetime.utcnow() - timedelta(days=days)
        performance_data = await db_manager.get_strategy_performance_detailed(
            strategy_id, start_date
        )
        
        return StrategyPerformanceResponse(
            success=True,
            data=performance_data,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略性能失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{strategy_id}/backtest", response_model=StrategyBacktestResponse)
async def backtest_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    backtest_request: StrategyBacktestRequest = None,
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    策略回测
    
    支持自定义时间范围和参数的策略回测
    """
    try:
        # 检查策略是否存在
        strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 创建回测任务
        backtest_id = str(uuid4())
        backtest_task = {
            "id": backtest_id,
            "strategy_id": strategy_id,
            "config": backtest_request.dict(),
            "status": "pending",
            "created_at": datetime.utcnow(),
            "created_by": current_user["user_id"]
        }
        
        # 保存回测任务
        await db_manager.create_backtest_task(backtest_task)
        
        # 后台执行回测
        background_tasks.add_task(
            _execute_backtest, 
            backtest_id, 
            strategy,
            backtest_request,
            db_manager
        )
        
        return StrategyBacktestResponse(
            success=True,
            data={
                "backtest_id": backtest_id,
                "status": "pending",
                "estimated_duration": _estimate_backtest_duration(backtest_request)
            },
            message="回测任务已提交",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"提交回测任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{strategy_id}/backtest/{backtest_id}", response_model=StrategyBacktestResponse)
async def get_backtest_result(
    strategy_id: str = Path(..., description="策略ID"),
    backtest_id: str = Path(..., description="回测ID"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """获取回测结果"""
    try:
        # 获取回测结果
        backtest_result = await db_manager.get_backtest_result(backtest_id)
        if not backtest_result:
            raise HTTPException(status_code=404, detail="回测任务不存在")
        
        if backtest_result["strategy_id"] != strategy_id:
            raise HTTPException(status_code=400, detail="策略ID不匹配")
        
        return StrategyBacktestResponse(
            success=True,
            data=backtest_result,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ 策略克隆和模板 ============

@router.post("/{strategy_id}/clone", response_model=StrategyResponse)
async def clone_strategy(
    strategy_id: str = Path(..., description="策略ID"),
    clone_name: str = Query(..., description="新策略名称"),
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager)
):
    """克隆策略"""
    try:
        # 获取原策略
        original_strategy = await db_manager.get_strategy_by_id(strategy_id)
        if not original_strategy:
            raise HTTPException(status_code=404, detail="策略不存在")
        
        # 创建克隆策略
        cloned_strategy = Strategy(
            id=str(uuid4()),
            name=clone_name,
            description=f"克隆自: {original_strategy['name']}",
            strategy_type=StrategyType(original_strategy["strategy_type"]),
            symbol=original_strategy["symbol"],
            config=original_strategy["config"].copy(),
            risk_config=original_strategy.get("risk_config", {}).copy(),
            created_by=current_user["user_id"],
            status=StrategyStatus.INACTIVE,
            cloned_from=strategy_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # 保存克隆策略
        await db_manager.create_strategy(cloned_strategy)
        
        logger.info(f"策略克隆成功: {strategy_id} -> {cloned_strategy.id}")
        
        return StrategyResponse(
            success=True,
            data=cloned_strategy.dict(),
            message="策略克隆成功",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"克隆策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates", response_model=StrategyListResponse)
async def get_strategy_templates(
    current_user = Depends(get_current_user),
    db_manager = Depends(get_database_manager),
    strategy_type: Optional[StrategyType] = Query(None, description="策略类型筛选")
):
    """获取策略模板列表"""
    try:
        # 获取预定义策略模板
        templates = await db_manager.get_strategy_templates(strategy_type)
        
        return StrategyListResponse(
            success=True,
            data=templates,
            pagination={
                "total": len(templates),
                "page": 1,
                "size": len(templates),
                "pages": 1
            },
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"获取策略模板失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ 私有辅助函数 ============

async def _validate_strategy_config(strategy_type: StrategyType, config: Dict[str, Any]) -> None:
    """验证策略配置"""
    required_fields = {
        StrategyType.GRID: ["grid_size", "upper_price", "lower_price", "investment"],
        StrategyType.DCA: ["investment_amount", "frequency_hours"],
        StrategyType.MACD: ["fast_period", "slow_period", "signal_period"],
        StrategyType.RSI: ["rsi_period", "oversold_threshold", "overbought_threshold"],
        StrategyType.MEAN_REVERSION: ["lookback_period", "threshold"],
        StrategyType.MOMENTUM: ["momentum_period", "rsi_period"],
    }
    
    if strategy_type in required_fields:
        for field in required_fields[strategy_type]:
            if field not in config:
                raise ValueError(f"策略配置缺少必需字段: {field}")

async def _pre_execution_check(strategy: Dict[str, Any], execution_request: Optional[StrategyExecutionRequest]) -> None:
    """执行前置检查"""
    # 检查市场状态
    # 检查资金充足性
    # 检查网络连接
    # 检查风控参数
    pass

async def _execute_strategy(strategy_id: str, db_manager) -> None:
    """后台执行策略（占位符）"""
    logger.info(f"开始执行策略: {strategy_id}")
    # 这里应该调用策略管理器执行策略
    pass

async def _stop_strategy_execution(strategy_id: str) -> None:
    """停止策略执行（占位符）"""
    logger.info(f"停止策略执行: {strategy_id}")
    # 这里应该发送停止信号到策略执行器
    pass

async def _pause_strategy_execution(strategy_id: str) -> None:
    """暂停策略执行（占位符）"""
    logger.info(f"暂停策略执行: {strategy_id}")

async def _resume_strategy_execution(strategy_id: str, db_manager) -> None:
    """恢复策略执行（占位符）"""
    logger.info(f"恢复策略执行: {strategy_id}")

async def _validate_ai_strategy(strategy_id: str, db_manager) -> None:
    """验证AI生成策略（后台任务）"""
    logger.info(f"验证AI策略: {strategy_id}")

async def _execute_backtest(
    backtest_id: str,
    strategy: Dict[str, Any],
    backtest_request: StrategyBacktestRequest,
    db_manager
) -> None:
    """执行回测（后台任务）"""
    logger.info(f"开始执行回测: {backtest_id}")
    
    try:
        # 更新回测状态为运行中
        await db_manager.update_backtest_task(backtest_id, {
            "status": "running",
            "started_at": datetime.utcnow()
        })
        
        # 这里应该实现实际的回测逻辑
        # 调用Rust引擎进行高性能回测
        
        # 模拟回测完成
        import asyncio
        await asyncio.sleep(10)  # 模拟回测耗时
        
        # 更新回测结果
        await db_manager.update_backtest_task(backtest_id, {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "result": {
                "total_return": 0.125,
                "sharpe_ratio": 1.45,
                "max_drawdown": 0.08,
                "win_rate": 0.62,
                "trade_count": 156
            }
        })
        
        logger.info(f"回测完成: {backtest_id}")
        
    except Exception as e:
        logger.error(f"回测执行失败: {backtest_id}, 错误: {e}")
        await db_manager.update_backtest_task(backtest_id, {
            "status": "failed",
            "failed_at": datetime.utcnow(),
            "error": str(e)
        })

def _estimate_backtest_duration(backtest_request: StrategyBacktestRequest) -> int:
    """估算回测所需时间（秒）"""
    days = (backtest_request.end_date - backtest_request.start_date).days
    return max(30, days * 2)  # 每天约2秒，最少30秒