"""
CTBench FastAPI Routes
CTBench时序生成模型的API接口层
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..integration.model_service import (
    get_ctbench_service, CTBenchModelService, 
    DataGenerationRequest, ModelTrainingRequest
)
from ..integration.risk_control.enhanced_risk_manager import EnhancedRiskManager

# Pydantic模型定义
class ModelConfigRequest(BaseModel):
    """模型配置请求"""
    model_type: str = Field(..., description="模型类型 (timevae, quantgan, etc.)")
    input_dim: int = Field(6, description="输入维度")
    output_dim: int = Field(6, description="输出维度") 
    sequence_length: int = Field(60, description="序列长度")
    hidden_dim: int = Field(128, description="隐藏层维度")
    learning_rate: float = Field(1e-3, description="学习率")
    batch_size: int = Field(64, description="批次大小")
    epochs: int = Field(100, description="训练轮数")
    
class DataGenerationRequestModel(BaseModel):
    """数据生成请求模型"""
    model_type: str = Field(..., description="生成模型类型")
    num_samples: int = Field(100, description="生成样本数")
    scenario_type: Optional[str] = Field(None, description="场景类型")
    priority: int = Field(1, description="优先级 1-3")

class TrainingDataRequest(BaseModel):
    """训练数据请求"""
    model_type: str = Field(..., description="模型类型")
    data: List[List[float]] = Field(..., description="训练数据")
    validation_split: float = Field(0.2, description="验证集比例")

class ScenarioGenerationRequest(BaseModel):
    """场景生成请求"""
    scenario_type: str = Field(..., description="场景类型")
    base_data: List[List[float]] = Field(..., description="基础数据")
    num_scenarios: int = Field(100, description="生成场景数")

class RiskAssessmentRequest(BaseModel):
    """风险评估请求"""
    portfolio_data: Dict[str, Any] = Field(..., description="组合数据")
    market_data: List[List[float]] = Field(..., description="市场数据")

# 创建路由器
router = APIRouter(prefix="/api/ctbench", tags=["CTBench"])

# 全局服务实例
ctbench_service: Optional[CTBenchModelService] = None
risk_manager: Optional[EnhancedRiskManager] = None

async def get_services():
    """获取服务实例"""
    global ctbench_service, risk_manager
    if ctbench_service is None:
        ctbench_service = await get_ctbench_service()
    if risk_manager is None:
        risk_manager = EnhancedRiskManager()
        await risk_manager.initialize()
    return ctbench_service, risk_manager

@router.on_event("startup")
async def startup_event():
    """启动事件"""
    await get_services()
    logging.info("CTBench API服务已启动")

# 模型管理接口
@router.get("/models/status")
async def get_model_status():
    """获取所有模型状态"""
    try:
        service, _ = await get_services()
        status = service.synthetic_manager.get_model_status()
        return JSONResponse({
            "success": True,
            "data": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/initialize/{model_type}")
async def initialize_model(model_type: str, config: Optional[ModelConfigRequest] = None):
    """初始化指定模型"""
    try:
        service, _ = await get_services()
        
        # 如果提供了配置，更新模型配置
        if config:
            service.synthetic_manager.model_configs[model_type] = config.dict()
            
        success = service.synthetic_manager.initialize_model(model_type)
        
        return JSONResponse({
            "success": success,
            "message": f"模型 {model_type} {'初始化成功' if success else '初始化失败'}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train")
async def train_model(request: TrainingDataRequest, background_tasks: BackgroundTasks):
    """训练模型"""
    try:
        service, _ = await get_services()
        
        # 准备训练数据
        data_array = np.array(request.data)
        
        # 分割训练和验证数据
        split_idx = int(len(data_array) * (1 - request.validation_split))
        train_data = data_array[:split_idx]
        val_data = data_array[split_idx:] if split_idx < len(data_array) else None
        
        # 创建训练请求
        training_request = ModelTrainingRequest(
            model_type=request.model_type,
            training_data=train_data,
            validation_data=val_data
        )
        
        # 提交训练请求
        request_id = await service.submit_training_request(training_request)
        
        return JSONResponse({
            "success": True,
            "request_id": request_id,
            "message": f"模型 {request.model_type} 训练请求已提交",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 数据生成接口
@router.post("/generate/synthetic")
async def generate_synthetic_data(request: DataGenerationRequestModel):
    """生成合成数据"""
    try:
        service, _ = await get_services()
        
        generation_request = DataGenerationRequest(
            model_type=request.model_type,
            num_samples=request.num_samples,
            scenario_type=request.scenario_type,
            priority=request.priority
        )
        
        request_id = await service.submit_generation_request(generation_request)
        
        return JSONResponse({
            "success": True,
            "request_id": request_id,
            "message": "数据生成请求已提交",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/scenarios")
async def generate_market_scenarios(request: ScenarioGenerationRequest):
    """生成市场场景"""
    try:
        service, _ = await get_services()
        
        base_data = np.array(request.base_data)
        result = service.synthetic_manager.generate_market_scenarios(
            request.scenario_type,
            base_data,
            request.num_scenarios
        )
        
        if result['success']:
            return JSONResponse({
                "success": True,
                "data": {
                    "scenario_type": result['scenario_type'],
                    "num_scenarios": result['num_scenarios'],
                    "data_shape": list(result['data'].shape),
                    "data": result['data'].tolist()
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '生成失败'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/stress-test")
async def generate_stress_test_scenarios(
    base_data: List[List[float]], 
    stress_types: Optional[List[str]] = None
):
    """生成压力测试场景"""
    try:
        service, _ = await get_services()
        
        base_array = np.array(base_data)
        
        # 重塑数据格式 (samples, sequence, features)
        if len(base_array.shape) == 2:
            base_array = base_array.reshape(1, base_array.shape[0], base_array.shape[1])
            
        result = await service.generate_stress_test_scenarios(base_array, stress_types)
        
        if result['success']:
            # 转换numpy数组为列表以便JSON序列化
            stress_scenarios_serializable = {}
            for scenario_type, scenarios in result['stress_scenarios'].items():
                stress_scenarios_serializable[scenario_type] = scenarios.tolist()
                
            return JSONResponse({
                "success": True,
                "data": {
                    "stress_scenarios": stress_scenarios_serializable,
                    "total_scenarios": result['total_scenarios']
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '生成失败'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 风险管理接口
@router.post("/risk/assessment")
async def comprehensive_risk_assessment(request: RiskAssessmentRequest):
    """综合风险评估"""
    try:
        _, risk_manager = await get_services()
        
        # 转换市场数据
        market_df = pd.DataFrame(request.market_data)
        
        # 执行风险评估
        assessment = await risk_manager.comprehensive_risk_assessment(
            request.portfolio_data, market_df
        )
        
        return JSONResponse({
            "success": True,
            "data": {
                "overall_risk": assessment.overall_risk.name,
                "portfolio_value": assessment.portfolio_value,
                "var_1d": assessment.var_1d,
                "var_5d": assessment.var_5d,
                "max_drawdown": assessment.max_drawdown,
                "sharpe_ratio": assessment.sharpe_ratio,
                "volatility": assessment.volatility,
                "black_swan_probability": assessment.black_swan_probability,
                "stress_test_results": assessment.stress_test_results,
                "recommendations": assessment.recommendations
            },
            "timestamp": assessment.timestamp.isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk/monitor")
async def real_time_risk_monitoring(request: RiskAssessmentRequest):
    """实时风险监控"""
    try:
        _, risk_manager = await get_services()
        
        market_df = pd.DataFrame(request.market_data)
        result = await risk_manager.real_time_risk_monitoring(
            request.portfolio_data, market_df
        )
        
        return JSONResponse({
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 数据增强接口
@router.post("/augment/market-data")
async def augment_market_data(
    market_data: List[List[float]], 
    augmentation_factor: int = 5
):
    """市场数据增强"""
    try:
        service, _ = await get_services()
        
        data_array = np.array(market_data)
        if len(data_array.shape) == 2:
            data_array = data_array.reshape(1, data_array.shape[0], data_array.shape[1])
            
        result = await service.get_real_time_market_data_augmentation(
            data_array, augmentation_factor
        )
        
        if result['success']:
            return JSONResponse({
                "success": True,
                "data": {
                    "original_data_shape": list(result['original_data_shape']),
                    "augmented_data": result['augmented_data'].tolist(),
                    "augmentation_factor": result['augmentation_factor'],
                    "scenarios_generated": result['scenarios_generated']
                },
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail=result.get('error', '增强失败'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 服务状态接口
@router.get("/service/status")
async def get_service_status():
    """获取服务状态"""
    try:
        service, _ = await get_services()
        stats = service.get_service_stats()
        
        return JSONResponse({
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/service/reset-stats")
async def reset_service_stats():
    """重置服务统计"""
    try:
        service, _ = await get_services()
        service.reset_stats()
        
        return JSONResponse({
            "success": True,
            "message": "服务统计已重置",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket实时数据流
@router.websocket("/ws/realtime-risk")
async def websocket_realtime_risk(websocket: WebSocket):
    """实时风险监控WebSocket"""
    await websocket.accept()
    _, risk_manager = await get_services()
    
    try:
        while True:
            # 接收客户端数据
            data = await websocket.receive_json()
            
            portfolio_data = data.get('portfolio_data', {})
            market_data_list = data.get('market_data', [])
            
            if market_data_list:
                market_df = pd.DataFrame(market_data_list)
                
                # 执行实时风险监控
                risk_result = await risk_manager.real_time_risk_monitoring(
                    portfolio_data, market_df
                )
                
                # 发送结果
                await websocket.send_json({
                    "type": "risk_update",
                    "data": risk_result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # 等待下一次更新
            await asyncio.sleep(0.25)  # 4Hz更新频率
            
    except WebSocketDisconnect:
        logging.info("WebSocket客户端断开连接")
    except Exception as e:
        logging.error(f"WebSocket错误: {e}")
        await websocket.close()

@router.websocket("/ws/data-generation")
async def websocket_data_generation(websocket: WebSocket):
    """实时数据生成WebSocket"""
    await websocket.accept()
    service, _ = await get_services()
    
    try:
        while True:
            # 接收生成请求
            request_data = await websocket.receive_json()
            
            model_type = request_data.get('model_type', 'timevae')
            num_samples = request_data.get('num_samples', 10)
            scenario_type = request_data.get('scenario_type')
            
            # 生成数据
            if scenario_type and 'base_data' in request_data:
                base_data = np.array(request_data['base_data'])
                result = service.synthetic_manager.generate_market_scenarios(
                    scenario_type, base_data, num_samples
                )
            else:
                result = service.synthetic_manager.generate_synthetic_data(
                    model_type, num_samples
                )
            
            # 发送结果
            if result['success']:
                await websocket.send_json({
                    "type": "generation_result",
                    "success": True,
                    "data": result['data'].tolist() if hasattr(result['data'], 'tolist') else result['data'],
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "generation_error",
                    "success": False,
                    "error": result.get('error', '未知错误'),
                    "timestamp": datetime.now().isoformat()
                })
                
            await asyncio.sleep(1)  # 防止过于频繁的生成
            
    except WebSocketDisconnect:
        logging.info("数据生成WebSocket客户端断开连接")
    except Exception as e:
        logging.error(f"数据生成WebSocket错误: {e}")
        await websocket.close()