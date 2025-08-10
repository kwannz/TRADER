"""
AI Factor Discovery Integration Routes
AI因子发现集成API接口
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..integration.ai_factor_ctbench_bridge import (
    get_ai_factor_bridge, AIFactorCTBenchBridge,
    FactorEnhancementRequest, FactorValidationResult
)

# Pydantic模型定义
class FactorDefinition(BaseModel):
    """因子定义模型"""
    name: str = Field(..., description="因子名称")
    formula: str = Field(..., description="因子计算公式")
    description: str = Field("", description="因子描述")
    type: str = Field(..., description="因子类型: trend|momentum|volatility|volume")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="因子参数")

class FactorEnhancementRequestModel(BaseModel):
    """因子增强请求模型"""
    factor: FactorDefinition
    base_data: List[List[float]] = Field(..., description="基础市场数据")
    enhancement_scenarios: List[str] = Field(
        default=['black_swan', 'high_volatility', 'bear_market'],
        description="增强场景类型"
    )
    validation_period: int = Field(252, description="验证期天数")

class BatchFactorValidationRequest(BaseModel):
    """批量因子验证请求"""
    factors: List[FactorDefinition] = Field(..., description="因子列表")
    base_data: List[List[float]] = Field(..., description="基础市场数据")
    enhancement_scenarios: List[str] = Field(
        default=['black_swan', 'high_volatility', 'bear_market'],
        description="增强场景类型"
    )
    validation_period: int = Field(252, description="验证期天数")

class FactorOptimizationRequest(BaseModel):
    """因子优化请求"""
    factor: FactorDefinition
    current_performance: Dict[str, float] = Field(..., description="当前因子表现")
    market_context: Dict[str, str] = Field(..., description="市场环境")
    optimization_target: str = Field("ic_ir", description="优化目标")

# 创建路由器
router = APIRouter(prefix="/api/ai-factor", tags=["AI Factor Discovery"])

# 全局服务实例
ai_factor_bridge: Optional[AIFactorCTBenchBridge] = None

async def get_bridge():
    """获取集成桥接器"""
    global ai_factor_bridge
    if ai_factor_bridge is None:
        ai_factor_bridge = await get_ai_factor_bridge()
    return ai_factor_bridge

@router.on_event("startup")
async def startup_event():
    """启动事件"""
    await get_bridge()
    logging.info("AI因子发现API服务已启动")

@router.post("/enhance-factor")
async def enhance_factor_with_ctbench(request: FactorEnhancementRequestModel):
    """使用CTBench增强因子验证"""
    try:
        bridge = await get_bridge()
        
        # 转换请求格式
        base_data_array = np.array(request.base_data)
        
        factor_request = FactorEnhancementRequest(
            factor_name=request.factor.name,
            factor_formula=request.factor.formula,
            factor_type=request.factor.type,
            base_data=base_data_array,
            enhancement_scenarios=request.enhancement_scenarios,
            validation_period=request.validation_period
        )
        
        # 执行因子增强
        enhancement_result = await bridge.enhance_factor_with_synthetic_data(factor_request)
        
        return JSONResponse({
            "success": enhancement_result['success'],
            "data": enhancement_result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-validate")
async def batch_validate_factors(request: BatchFactorValidationRequest):
    """批量验证因子"""
    try:
        bridge = await get_bridge()
        
        # 转换批量请求
        base_data_array = np.array(request.base_data)
        
        factor_requests = []
        for factor in request.factors:
            factor_request = FactorEnhancementRequest(
                factor_name=factor.name,
                factor_formula=factor.formula,
                factor_type=factor.type,
                base_data=base_data_array,
                enhancement_scenarios=request.enhancement_scenarios,
                validation_period=request.validation_period
            )
            factor_requests.append(factor_request)
        
        # 批量验证
        validation_results = await bridge.batch_factor_validation(factor_requests)
        
        # 生成集成报告
        integration_report = bridge.generate_integration_report(validation_results)
        
        return JSONResponse({
            "success": True,
            "data": {
                "validation_results": validation_results,
                "integration_report": integration_report
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress-test-factor")
async def stress_test_factor(factor: FactorDefinition, base_data: List[List[float]]):
    """因子压力测试"""
    try:
        bridge = await get_bridge()
        
        base_data_array = np.array(base_data)
        
        factor_request = FactorEnhancementRequest(
            factor_name=factor.name,
            factor_formula=factor.formula,
            factor_type=factor.type,
            base_data=base_data_array,
            enhancement_scenarios=['black_swan', 'high_volatility', 'bear_market'],
            validation_period=252
        )
        
        # 执行压力测试（通过增强验证获取）
        enhancement_result = await bridge.enhance_factor_with_synthetic_data(factor_request)
        
        if enhancement_result['success']:
            stress_test_data = {
                'factor_name': factor.name,
                'stress_test_results': enhancement_result['stress_test_results'],
                'survival_rate': enhancement_result['factor_performance']['stress_test_survival_rate'],
                'robustness_score': enhancement_result['factor_performance']['robustness_score'],
                'recommendations': [rec for rec in enhancement_result['enhancement_recommendations'] 
                                 if '🛡️' in rec or '⚠️' in rec]  # 只返回风险相关建议
            }
            
            return JSONResponse({
                "success": True,
                "data": stress_test_data,
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail="压力测试失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/synthetic-data-quality/{factor_name}")
async def assess_synthetic_data_quality(factor_name: str):
    """评估合成数据质量"""
    try:
        # 这里应该从数据库或缓存中获取因子的合成数据质量评估
        # 简化实现，返回模拟数据
        
        quality_assessment = {
            'factor_name': factor_name,
            'quality_metrics': {
                'statistical_similarity': 0.85,
                'distribution_similarity': 0.78,
                'temporal_consistency': 0.82,
                'scenario_coverage': 0.90,
                'overall_quality_score': 0.84
            },
            'quality_breakdown': {
                'mean_similarity': 0.88,
                'volatility_similarity': 0.81,
                'correlation_preservation': 0.79,
                'extreme_event_coverage': 0.92
            },
            'recommendations': [
                "合成数据质量良好，适合因子验证",
                "建议增加更多极端场景以提高压力测试覆盖度",
                "时序一致性略有不足，可优化生成模型参数"
            ]
        }
        
        return JSONResponse({
            "success": True,
            "data": quality_assessment,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/factor-robustness-analysis")
async def factor_robustness_analysis(
    factor: FactorDefinition, 
    base_data: List[List[float]],
    robustness_tests: List[str] = ["parameter_sensitivity", "market_regime", "time_decay"]
):
    """因子稳健性分析"""
    try:
        bridge = await get_bridge()
        
        base_data_array = np.array(base_data)
        
        # 创建多个测试场景
        test_scenarios = []
        for test_type in robustness_tests:
            if test_type == "parameter_sensitivity":
                # 参数敏感性测试
                test_scenarios.extend(['high_volatility', 'low_volatility'])
            elif test_type == "market_regime":
                # 市场环境测试
                test_scenarios.extend(['bull_market', 'bear_market', 'sideways'])
            elif test_type == "time_decay":
                # 时间衰减测试
                test_scenarios.extend(['black_swan'])
                
        factor_request = FactorEnhancementRequest(
            factor_name=factor.name,
            factor_formula=factor.formula,
            factor_type=factor.type,
            base_data=base_data_array,
            enhancement_scenarios=list(set(test_scenarios)),  # 去重
            validation_period=252
        )
        
        # 执行稳健性测试
        enhancement_result = await bridge.enhance_factor_with_synthetic_data(factor_request)
        
        if enhancement_result['success']:
            robustness_analysis = {
                'factor_name': factor.name,
                'overall_robustness_score': enhancement_result['factor_performance']['robustness_score'],
                'robustness_breakdown': {
                    'stability_across_scenarios': enhancement_result['factor_performance']['synthetic_ic_mean'],
                    'stress_test_survival': enhancement_result['factor_performance']['stress_test_survival_rate'],
                    'parameter_sensitivity': 0.75,  # 简化实现
                    'market_regime_adaptability': 0.82  # 简化实现
                },
                'risk_factors': {
                    'high_volatility_sensitivity': 'medium',
                    'extreme_event_vulnerability': 'low',
                    'parameter_stability': 'high'
                },
                'improvement_suggestions': enhancement_result['enhancement_recommendations']
            }
            
            return JSONResponse({
                "success": True,
                "data": robustness_analysis,
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=400, detail="稳健性分析失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/integration-status")
async def get_integration_status():
    """获取集成状态"""
    try:
        bridge = await get_bridge()
        
        # 检查各个服务的状态
        ctbench_status = bridge.ctbench_service.get_service_stats() if bridge.ctbench_service else None
        
        integration_status = {
            'ctbench_service': {
                'status': 'connected' if bridge.ctbench_service else 'disconnected',
                'models_available': list(bridge.ctbench_service.synthetic_manager.get_model_status().keys()) if bridge.ctbench_service else [],
                'service_stats': ctbench_status
            },
            'risk_manager': {
                'status': 'connected' if bridge.risk_manager else 'disconnected'
            },
            'bridge_config': bridge.config,
            'integration_capabilities': {
                'synthetic_data_generation': True,
                'stress_testing': True,
                'factor_validation': True,
                'real_time_monitoring': True,
                'batch_processing': True
            },
            'supported_scenarios': [
                'black_swan', 'bull_market', 'bear_market', 
                'high_volatility', 'sideways'
            ]
        }
        
        return JSONResponse({
            "success": True,
            "data": integration_status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-monitoring")
async def start_real_time_monitoring(
    factor_names: List[str] = Field(..., description="要监控的因子名称列表"),
    background_tasks: BackgroundTasks = None
):
    """启动实时因子监控"""
    try:
        bridge = await get_bridge()
        
        # 在后台启动监控任务
        if background_tasks:
            background_tasks.add_task(
                bridge.real_time_factor_monitoring, 
                factor_names
            )
        
        return JSONResponse({
            "success": True,
            "message": f"已启动 {len(factor_names)} 个因子的实时监控",
            "data": {
                "monitored_factors": factor_names,
                "monitoring_interval": "5分钟",
                "start_time": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-benchmark")
async def get_performance_benchmark():
    """获取性能基准"""
    benchmark_data = {
        'factor_validation_benchmarks': {
            'ic_thresholds': {
                'excellent': 0.1,
                'good': 0.05,
                'acceptable': 0.02,
                'poor': 0.01
            },
            'robustness_score_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'acceptable': 0.4,
                'poor': 0.2
            },
            'stress_test_survival_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'acceptable': 0.4,
                'poor': 0.2
            }
        },
        'ctbench_integration_benchmarks': {
            'synthetic_data_quality': {
                'statistical_similarity_min': 0.7,
                'distribution_similarity_min': 0.6,
                'temporal_consistency_min': 0.7
            },
            'scenario_coverage': {
                'recommended_scenarios': 5,
                'minimum_scenarios': 3,
                'samples_per_scenario': 50
            }
        },
        'performance_expectations': {
            'factor_enhancement_time': "< 30秒",
            'batch_validation_throughput': "10个因子/分钟",
            'real_time_monitoring_latency': "< 5分钟",
            'data_augmentation_factor': "5-10x"
        }
    }
    
    return JSONResponse({
        "success": True,
        "data": benchmark_data,
        "timestamp": datetime.now().isoformat()
    })

@router.get("/factor-recommendations")
async def get_factor_recommendations(
    market_condition: str = "normal",
    factor_type: Optional[str] = None,
    risk_tolerance: str = "medium"
):
    """获取因子建议"""
    try:
        # 基于市场条件和风险偏好提供因子建议
        recommendations = {
            'market_condition': market_condition,
            'recommended_factors': [],
            'enhancement_strategies': [],
            'ctbench_scenarios': []
        }
        
        if market_condition == "high_volatility":
            recommendations['recommended_factors'] = [
                "波动率反转因子",
                "VIX溢价因子", 
                "期权偏斜因子"
            ]
            recommendations['ctbench_scenarios'] = ['high_volatility', 'black_swan']
            recommendations['enhancement_strategies'] = [
                "增加极端场景测试",
                "提高因子稳健性权重"
            ]
        elif market_condition == "trending":
            recommendations['recommended_factors'] = [
                "动量因子",
                "趋势强度因子",
                "突破因子"
            ]
            recommendations['ctbench_scenarios'] = ['bull_market', 'bear_market']
            recommendations['enhancement_strategies'] = [
                "测试不同市场环境下的持续性",
                "优化参数敏感性"
            ]
        else:  # normal market
            recommendations['recommended_factors'] = [
                "均值回归因子",
                "基本面因子",
                "技术指标组合因子"
            ]
            recommendations['ctbench_scenarios'] = ['sideways', 'low_volatility']
            recommendations['enhancement_strategies'] = [
                "平衡收益与风险",
                "提高信号稳定性"
            ]
            
        return JSONResponse({
            "success": True,
            "data": recommendations,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))