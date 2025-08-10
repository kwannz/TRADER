# CTBench集成代码实现

## 1. FastAPI端点集成

```python
# backend/src/api/v1/ctbench.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import numpy as np
from datetime import datetime

from ...ctbench.data_generation.synthetic_data_manager import SyntheticDataManager
from ...ctbench.models.model_registry import ModelRegistry
from ...ctbench.evaluation.metrics import MetricsCalculator
from ...core.auth import get_current_user

router = APIRouter(prefix="/ctbench", tags=["CTBench"])

class DataGenerationRequest(BaseModel):
    """数据生成请求模型"""
    model_name: str  # 'timevae', 'quantgan', 'diffusion_ts', 'fourier_flow'
    n_samples: int = 1000
    seq_length: int = 500
    scenario_type: str = 'normal'  # 'normal', 'stress', 'black_swan'
    strategy_context: Optional[str] = None
    augmentation_ratio: float = 0.3
    
class ModelTrainingRequest(BaseModel):
    """模型训练请求模型"""
    model_name: str
    training_data_symbols: List[str]
    training_period_days: int = 365
    hyperparameters: Dict[str, Any] = {}
    use_gpu: bool = True
    distributed_training: bool = False

class StressTestRequest(BaseModel):
    """压力测试请求模型"""
    portfolio_config: Dict[str, float]  # 资产权重
    test_types: List[str] = ['black_swan', 'flash_crash', 'correlation_breakdown']
    n_scenarios: int = 100
    confidence_levels: List[float] = [0.95, 0.99]

# 全局服务实例
synthetic_data_manager = SyntheticDataManager()
model_registry = ModelRegistry()
metrics_calculator = MetricsCalculator()

@router.post("/models/train")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    训练TSG模型
    支持分布式训练和GPU加速
    """
    try:
        # 验证模型名称
        if request.model_name not in model_registry.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型类型: {request.model_name}"
            )
        
        # 获取训练数据
        training_data = await synthetic_data_manager.prepare_training_data(
            symbols=request.training_data_symbols,
            period_days=request.training_period_days
        )
        
        if len(training_data) < 1000:
            raise HTTPException(
                status_code=400,
                detail="训练数据不足，至少需要1000个样本"
            )
        
        # 创建训练任务
        training_task_id = f"train_{request.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 异步启动训练任务
        background_tasks.add_task(
            _execute_model_training,
            task_id=training_task_id,
            model_name=request.model_name,
            training_data=training_data,
            hyperparameters=request.hyperparameters,
            use_gpu=request.use_gpu,
            distributed=request.distributed_training,
            user_id=current_user.id
        )
        
        return {
            "status": "training_started",
            "task_id": training_task_id,
            "model_name": request.model_name,
            "estimated_duration_minutes": _estimate_training_time(
                request.model_name, 
                len(training_data),
                request.use_gpu
            ),
            "monitoring_endpoint": f"/ctbench/training/status/{training_task_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练启动失败: {str(e)}")

@router.get("/training/status/{task_id}")
async def get_training_status(
    task_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取模型训练状态
    实时返回训练进度和指标
    """
    try:
        training_status = await model_registry.get_training_status(task_id)
        
        if not training_status:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        
        # 检查用户权限
        if training_status['user_id'] != current_user.id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="无权限访问该训练任务")
        
        return {
            "task_id": task_id,
            "status": training_status['status'],  # 'running', 'completed', 'failed', 'paused'
            "progress": {
                "current_epoch": training_status.get('current_epoch', 0),
                "total_epochs": training_status.get('total_epochs', 0),
                "progress_percentage": training_status.get('progress_percentage', 0.0)
            },
            "metrics": {
                "training_loss": training_status.get('training_loss', []),
                "validation_loss": training_status.get('validation_loss', []),
                "quality_score": training_status.get('quality_score', 0.0)
            },
            "resource_usage": {
                "gpu_usage": training_status.get('gpu_usage', 0.0),
                "memory_usage": training_status.get('memory_usage', 0.0),
                "training_time_elapsed": training_status.get('training_time_elapsed', 0)
            },
            "checkpoints": training_status.get('saved_checkpoints', []),
            "logs": training_status.get('recent_logs', [])[-10:]  # 最近10条日志
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {str(e)}")

@router.post("/generate/synthetic-data")
async def generate_synthetic_data(
    request: DataGenerationRequest,
    current_user = Depends(get_current_user)
):
    """
    生成合成数据
    支持策略特定的数据增强
    """
    try:
        # 验证模型是否已训练
        model_status = await model_registry.get_model_status(request.model_name)
        if model_status['status'] != 'trained':
            raise HTTPException(
                status_code=400,
                detail=f"模型{request.model_name}尚未训练完成"
            )
        
        # 根据场景类型生成数据
        if request.strategy_context:
            # 策略特定的数据增强
            base_data = await synthetic_data_manager.get_strategy_base_data(
                request.strategy_context
            )
            
            generation_results = await synthetic_data_manager.generate_augmented_training_data(
                strategy_type=request.strategy_context,
                base_data=base_data,
                augmentation_ratio=request.augmentation_ratio
            )
        else:
            # 通用场景生成
            model = model_registry.get_model(request.model_name)
            generation_results = await model.generate_trading_scenarios(
                scenario_type=request.scenario_type,
                n_scenarios=request.n_samples
            )
        
        # 计算数据质量指标
        quality_metrics = await metrics_calculator.calculate_generation_quality(
            synthetic_data=generation_results['scenarios'],
            model_name=request.model_name
        )
        
        # 保存生成结果
        generation_id = await synthetic_data_manager.save_generation_results(
            user_id=current_user.id,
            model_name=request.model_name,
            request_params=request.dict(),
            generated_data=generation_results,
            quality_metrics=quality_metrics
        )
        
        return {
            "status": "success",
            "generation_id": generation_id,
            "data_summary": {
                "n_samples": len(generation_results.get('scenarios', [])),
                "seq_length": request.seq_length,
                "scenario_type": request.scenario_type,
                "file_size_mb": generation_results.get('file_size_mb', 0)
            },
            "quality_metrics": quality_metrics,
            "download_endpoint": f"/ctbench/download/{generation_id}",
            "usage_recommendations": generation_results.get('usage_recommendations', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据生成失败: {str(e)}")

@router.post("/stress-test/scenarios")
async def generate_stress_test_scenarios(
    request: StressTestRequest,
    current_user = Depends(get_current_user)
):
    """
    生成压力测试场景
    用于投资组合风险评估
    """
    try:
        # 验证投资组合配置
        portfolio_weights = request.portfolio_config
        if abs(sum(portfolio_weights.values()) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="投资组合权重之和必须等于1.0"
            )
        
        # 生成压力测试场景
        stress_test_results = await synthetic_data_manager.generate_stress_test_scenarios(
            current_portfolio=portfolio_weights,
            test_types=request.test_types
        )
        
        # 计算风险指标
        risk_analysis = {}
        for confidence_level in request.confidence_levels:
            risk_metrics = await metrics_calculator.calculate_portfolio_risk_metrics(
                portfolio=portfolio_weights,
                scenarios=stress_test_results['scenarios'],
                confidence_level=confidence_level
            )
            risk_analysis[f"confidence_{int(confidence_level*100)}"] = risk_metrics
        
        # 生成压力测试报告
        test_report = {
            "test_id": f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "portfolio_config": portfolio_weights,
            "test_scenarios": {
                test_type: {
                    "n_scenarios": len(scenarios['scenario_data']['scenarios']),
                    "worst_case_loss": scenarios['risk_metrics']['max_loss'],
                    "expected_loss": scenarios['risk_metrics']['expected_loss'],
                    "recovery_time_days": scenarios.get('recovery_time_days', 'unknown')
                }
                for test_type, scenarios in stress_test_results['scenarios'].items()
            },
            "risk_analysis": risk_analysis,
            "overall_recommendations": stress_test_results['recommendations'],
            "generated_at": datetime.now().isoformat()
        }
        
        # 保存压力测试结果
        await synthetic_data_manager.save_stress_test_report(
            user_id=current_user.id,
            report=test_report
        )
        
        # 检查是否有严重风险警报
        critical_alerts = []
        for test_type, scenarios in stress_test_results['scenarios'].items():
            max_loss = scenarios['risk_metrics']['max_loss']
            if max_loss > 0.2:  # 超过20%损失
                critical_alerts.append({
                    "test_type": test_type,
                    "max_potential_loss": max_loss,
                    "severity": "critical" if max_loss > 0.5 else "high"
                })
        
        return {
            "status": "completed",
            "test_report": test_report,
            "critical_alerts": critical_alerts,
            "visualization_data": {
                "risk_distribution_charts": stress_test_results.get('charts', {}),
                "scenario_timelines": stress_test_results.get('timelines', {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"压力测试失败: {str(e)}")

@router.get("/models/available")
async def list_available_models(current_user = Depends(get_current_user)):
    """
    获取可用的TSG模型列表
    """
    try:
        models = await model_registry.list_available_models()
        
        model_info = []
        for model_name, model_data in models.items():
            # 获取模型详细信息
            model_status = await model_registry.get_model_status(model_name)
            
            model_info.append({
                "name": model_name,
                "display_name": model_data['display_name'],
                "type": model_data['model_type'],  # 'VAE', 'GAN', 'Diffusion', 'Flow'
                "description": model_data['description'],
                "status": model_status['status'],
                "training_progress": model_status.get('training_progress', 0),
                "quality_score": model_status.get('quality_score', 0.0),
                "last_trained": model_status.get('last_trained'),
                "supported_scenarios": model_data['supported_scenarios'],
                "computational_requirements": {
                    "gpu_required": model_data['gpu_required'],
                    "min_memory_gb": model_data['min_memory_gb'],
                    "estimated_training_time_hours": model_data['estimated_training_time_hours']
                }
            })
        
        return {
            "available_models": model_info,
            "total_count": len(model_info),
            "trained_count": len([m for m in model_info if m['status'] == 'trained']),
            "training_count": len([m for m in model_info if m['status'] == 'training'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@router.get("/benchmark/performance")
async def get_benchmark_performance(
    model_names: Optional[List[str]] = None,
    current_user = Depends(get_current_user)
):
    """
    获取模型基准测试性能对比
    """
    try:
        if not model_names:
            model_names = list(model_registry.available_models.keys())
        
        benchmark_results = {}
        
        for model_name in model_names:
            model_status = await model_registry.get_model_status(model_name)
            
            if model_status['status'] == 'trained':
                # 获取基准测试结果
                performance_metrics = await metrics_calculator.get_model_benchmark_metrics(model_name)
                
                benchmark_results[model_name] = {
                    "generation_quality": performance_metrics['quality_metrics'],
                    "computational_performance": {
                        "training_time_minutes": performance_metrics['training_time_minutes'],
                        "inference_time_ms": performance_metrics['inference_time_ms'],
                        "memory_usage_gb": performance_metrics['peak_memory_usage_gb'],
                        "samples_per_second": performance_metrics['generation_throughput']
                    },
                    "trading_utility": performance_metrics['trading_utility_metrics'],
                    "overall_ranking": performance_metrics['overall_ranking']
                }
        
        # 生成排名对比
        ranking_comparison = _generate_model_ranking_comparison(benchmark_results)
        
        return {
            "benchmark_results": benchmark_results,
            "performance_ranking": ranking_comparison,
            "evaluation_criteria": {
                "data_quality_weight": 0.4,
                "computational_efficiency_weight": 0.3,
                "trading_utility_weight": 0.3
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取基准性能失败: {str(e)}")

# 后台任务函数
async def _execute_model_training(
    task_id: str,
    model_name: str,
    training_data: np.ndarray,
    hyperparameters: Dict[str, Any],
    use_gpu: bool,
    distributed: bool,
    user_id: str
):
    """
    执行模型训练的后台任务
    """
    try:
        # 初始化训练状态
        await model_registry.initialize_training_task(
            task_id=task_id,
            model_name=model_name,
            user_id=user_id,
            config={
                'use_gpu': use_gpu,
                'distributed': distributed,
                'hyperparameters': hyperparameters
            }
        )
        
        # 获取模型实例
        model = model_registry.create_model_instance(model_name, hyperparameters)
        
        # 准备训练配置
        training_config = {
            'epochs': hyperparameters.get('epochs', 100),
            'batch_size': hyperparameters.get('batch_size', 32),
            'learning_rate': hyperparameters.get('learning_rate', 0.001),
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'save_checkpoints': True,
            'checkpoint_frequency': 10
        }
        
        # 创建训练回调
        callbacks = [
            ModelCheckpointCallback(task_id),
            ProgressTrackingCallback(task_id),
            EarlyStoppingCallback(patience=training_config['early_stopping_patience']),
            MetricsLoggingCallback(task_id)
        ]
        
        # 执行训练
        training_history = await model.train(
            train_data=training_data,
            config=training_config,
            callbacks=callbacks
        )
        
        # 训练完成后的质量评估
        quality_assessment = await metrics_calculator.assess_trained_model_quality(
            model=model,
            validation_data=training_data[-1000:],  # 使用最后1000个样本作为验证
            model_name=model_name
        )
        
        # 更新模型注册表
        await model_registry.complete_training_task(
            task_id=task_id,
            model=model,
            training_history=training_history,
            quality_metrics=quality_assessment
        )
        
        print(f"✅ 模型 {model_name} 训练完成 (任务ID: {task_id})")
        
    except Exception as e:
        # 训练失败处理
        await model_registry.fail_training_task(
            task_id=task_id,
            error_message=str(e)
        )
        print(f"❌ 模型训练失败: {str(e)}")

def _estimate_training_time(model_name: str, data_size: int, use_gpu: bool) -> int:
    """估算训练时间（分钟）"""
    base_times = {
        'timevae': 30,
        'quantgan': 60,
        'diffusion_ts': 90,
        'fourier_flow': 45
    }
    
    base_time = base_times.get(model_name, 60)
    
    # 根据数据大小调整
    size_factor = max(1.0, data_size / 10000)
    
    # GPU加速
    gpu_factor = 0.3 if use_gpu else 1.0
    
    return int(base_time * size_factor * gpu_factor)

def _generate_model_ranking_comparison(benchmark_results: Dict) -> List[Dict]:
    """生成模型排名对比"""
    rankings = []
    
    for model_name, results in benchmark_results.items():
        overall_score = (
            results['generation_quality']['overall_quality'] * 0.4 +
            (1.0 - results['computational_performance']['training_time_minutes'] / 1000) * 0.3 +
            results['trading_utility']['sharpe_ratio'] * 0.3
        )
        
        rankings.append({
            'model_name': model_name,
            'overall_score': overall_score,
            'quality_rank': 0,  # 将在排序后填充
            'speed_rank': 0,
            'utility_rank': 0
        })
    
    # 排序并分配排名
    rankings.sort(key=lambda x: x['overall_score'], reverse=True)
    for i, ranking in enumerate(rankings):
        ranking['overall_rank'] = i + 1
    
    return rankings
```

## 2. 核心业务逻辑集成

```python
# backend/src/ctbench/data_generation/synthetic_data_manager.py
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import torch

from ..models.model_registry import ModelRegistry
from ..evaluation.metrics import MetricsCalculator
from ...database.mongodb import get_mongo_client
from ...database.redis import get_redis_client
from ...services.exchange_connectors.okx_connector import OKXConnector
from ...core.enhanced_risk_manager import EnhancedRiskManager

class SyntheticDataManager:
    """
    合成数据管理器
    将CTBench时间序列生成能力集成到交易系统
    """
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.metrics_calculator = MetricsCalculator()
        self.mongo_client = None
        self.redis_client = None
        self.okx_connector = None
        self.risk_manager = None
        
        # 数据质量控制参数
        self.quality_thresholds = {
            'min_overall_quality': 0.7,
            'min_price_similarity': 0.6,
            'min_volatility_clustering': 0.65,
            'min_correlation_preservation': 0.6
        }
        
        # 缓存配置
        self.cache_ttl = 3600  # 1小时
        self.max_cache_size_mb = 1000  # 1GB
        
    async def initialize(self):
        """初始化连接和服务"""
        self.mongo_client = await get_mongo_client()
        self.redis_client = await get_redis_client()
        self.okx_connector = OKXConnector()
        self.risk_manager = EnhancedRiskManager()
        await self.model_registry.initialize()
        
    async def generate_augmented_training_data(self,
                                             strategy_type: str,
                                             base_data: np.ndarray,
                                             augmentation_ratio: float = 0.3,
                                             quality_override: bool = False) -> Dict:
        """
        为策略训练生成增强数据
        
        参数:
            strategy_type: 策略类型 ('grid', 'dca', 'momentum', 'mean_reversion', 'arbitrage')
            base_data: 基础历史数据 (n_samples, n_assets, seq_length, n_features)
            augmentation_ratio: 增强比例，合成数据占总数据的比例
            quality_override: 是否忽略质量检查强制生成
            
        返回:
            Dict: {
                'status': 'success'|'low_quality'|'failed',
                'augmented_data': 增强后的数据集,
                'synthetic_data': 合成数据部分,
                'quality_metrics': 质量评估指标,
                'model_used': 使用的TSG模型,
                'usage_recommendations': 使用建议
            }
        """
        try:
            print(f"🔬 开始为 {strategy_type} 策略生成增强数据...")
            
            # 1. 数据预处理和验证
            validated_data = await self._validate_and_preprocess_data(base_data)
            if validated_data is None:
                return {'status': 'failed', 'error': '基础数据验证失败'}
            
            # 2. 根据策略类型选择最优TSG模型
            optimal_model = await self._select_optimal_model_for_strategy(
                strategy_type, validated_data
            )
            print(f"📊 选择模型: {optimal_model}")
            
            # 3. 检查模型状态
            model_status = await self.model_registry.get_model_status(optimal_model)
            if model_status['status'] != 'trained':
                # 尝试自动训练模型
                print(f"🏋️ 模型 {optimal_model} 尚未训练，开始自动训练...")
                await self._auto_train_model(optimal_model, validated_data)
            
            # 4. 生成合成数据
            n_synthetic_samples = int(len(validated_data) * augmentation_ratio)
            
            generation_config = self._get_strategy_specific_generation_config(
                strategy_type, validated_data
            )
            
            model = self.model_registry.get_model(optimal_model)
            synthetic_data = await model.generate(
                n_samples=n_synthetic_samples,
                conditioning_data=validated_data[-200:],  # 使用最近200个样本作为条件
                **generation_config
            )
            
            print(f"📡 生成了 {len(synthetic_data)} 个合成样本")
            
            # 5. 数据质量评估
            quality_metrics = await self.metrics_calculator.calculate_generation_quality(
                synthetic_data=synthetic_data,
                real_data=validated_data,
                strategy_context=strategy_type
            )
            
            print(f"📊 数据质量评分: {quality_metrics['overall_quality']:.3f}")
            
            # 6. 质量检查
            if not quality_override and not self._passes_quality_check(quality_metrics):
                return {
                    'status': 'low_quality',
                    'quality_metrics': quality_metrics,
                    'recommendations': self._get_quality_improvement_suggestions(quality_metrics),
                    'error': f"数据质量不达标 (得分: {quality_metrics['overall_quality']:.3f})"
                }
            
            # 7. 数据混合和后处理
            augmented_data = await self._blend_and_postprocess_data(
                real_data=validated_data,
                synthetic_data=synthetic_data,
                augmentation_ratio=augmentation_ratio,
                strategy_type=strategy_type
            )
            
            # 8. 缓存结果
            cache_key = f"augmented_data:{strategy_type}:{hash(str(base_data.tobytes()))}"
            await self._cache_generation_results(
                cache_key, {
                    'augmented_data': augmented_data,
                    'synthetic_data': synthetic_data,
                    'quality_metrics': quality_metrics,
                    'model_used': optimal_model,
                    'generation_timestamp': datetime.now().isoformat()
                }
            )
            
            # 9. 生成使用建议
            usage_recommendations = self._generate_usage_recommendations(
                quality_metrics, strategy_type, optimal_model
            )
            
            return {
                'status': 'success',
                'augmented_data': augmented_data,
                'synthetic_data': synthetic_data,
                'original_size': len(validated_data),
                'synthetic_size': len(synthetic_data),
                'quality_metrics': quality_metrics,
                'model_used': optimal_model,
                'generation_config': generation_config,
                'usage_recommendations': usage_recommendations,
                'cache_key': cache_key
            }
            
        except Exception as e:
            print(f"❌ 数据生成失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    async def generate_stress_test_scenarios(self,
                                           current_portfolio: Dict[str, float],
                                           test_types: List[str],
                                           n_scenarios: int = 100) -> Dict:
        """
        生成压力测试场景数据
        
        参数:
            current_portfolio: 当前投资组合 {symbol: weight}
            test_types: 测试类型 ['black_swan', 'flash_crash', 'correlation_breakdown', 'liquidity_crisis']
            n_scenarios: 每种类型生成的场景数量
            
        返回:
            Dict: 包含各类压力测试场景的数据和风险分析
        """
        try:
            print(f"🧪 生成 {len(test_types)} 种压力测试场景...")
            
            scenarios = {}
            portfolio_symbols = list(current_portfolio.keys())
            
            # 获取当前市场数据作为基准
            base_market_data = await self._get_current_market_data(portfolio_symbols)
            
            for test_type in test_types:
                print(f"  🎯 生成 {test_type} 场景...")
                
                # 根据测试类型选择最适合的模型
                optimal_model = self._select_model_for_stress_test(test_type)
                model = self.model_registry.get_model(optimal_model)
                
                # 生成特定类型的场景数据
                scenario_config = self._get_stress_test_config(test_type, current_portfolio)
                
                scenario_data = await model.generate_trading_scenarios(
                    scenario_type=test_type,
                    n_scenarios=n_scenarios,
                    base_data=base_market_data,
                    portfolio_context=current_portfolio,
                    **scenario_config
                )
                
                # 计算该场景下的投资组合风险
                portfolio_risk_metrics = await self._calculate_portfolio_risk_under_scenario(
                    portfolio=current_portfolio,
                    scenario_data=scenario_data['scenarios'],
                    scenario_type=test_type
                )
                
                # 影响分析
                impact_analysis = await self._analyze_scenario_impact(
                    current_portfolio, scenario_data['scenarios'], test_type
                )
                
                scenarios[test_type] = {
                    'scenario_data': scenario_data,
                    'risk_metrics': portfolio_risk_metrics,
                    'impact_analysis': impact_analysis,
                    'model_used': optimal_model,
                    'scenario_config': scenario_config
                }
            
            # 生成综合压力测试报告
            comprehensive_report = await self._generate_comprehensive_stress_report(
                scenarios, current_portfolio
            )
            
            # 生成风险管理建议
            risk_recommendations = await self._generate_risk_management_recommendations(
                scenarios, current_portfolio
            )
            
            # 保存压力测试结果
            test_report_id = await self._save_stress_test_results({
                'portfolio': current_portfolio,
                'scenarios': scenarios,
                'report': comprehensive_report,
                'recommendations': risk_recommendations,
                'test_timestamp': datetime.now().isoformat()
            })
            
            return {
                'status': 'success',
                'test_report_id': test_report_id,
                'scenarios': scenarios,
                'comprehensive_report': comprehensive_report,
                'risk_recommendations': risk_recommendations,
                'critical_findings': comprehensive_report['critical_findings'],
                'next_steps': risk_recommendations['immediate_actions']
            }
            
        except Exception as e:
            print(f"❌ 压力测试生成失败: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def enhance_factor_discovery(self,
                                     base_factors: pd.DataFrame,
                                     target_strategy: str,
                                     n_synthetic_factors: int = 50) -> Dict:
        """
        使用CTBench增强Alpha因子发现
        
        参数:
            base_factors: 基础因子数据
            target_strategy: 目标策略类型
            n_synthetic_factors: 要生成的合成因子数量
            
        返回:
            Dict: 增强后的因子集合和评估结果
        """
        try:
            print(f"🔬 为 {target_strategy} 策略增强因子发现...")
            
            # 1. 分析现有因子特性
            factor_characteristics = await self._analyze_factor_characteristics(base_factors)
            
            # 2. 选择适合因子生成的模型 (通常使用VAE)
            model = self.model_registry.get_model('timevae')
            
            # 3. 基于现有因子生成新的合成因子
            synthetic_factors = await self._generate_synthetic_factors(
                model=model,
                base_factors=base_factors,
                n_factors=n_synthetic_factors,
                strategy_context=target_strategy
            )
            
            # 4. 因子质量评估
            factor_quality = await self._evaluate_factor_quality(
                synthetic_factors, base_factors, target_strategy
            )
            
            # 5. 因子选择和优化
            optimized_factors = await self._optimize_factor_combination(
                base_factors, synthetic_factors, factor_quality
            )
            
            # 6. 生成因子使用建议
            factor_recommendations = self._generate_factor_recommendations(
                optimized_factors, factor_quality, target_strategy
            )
            
            return {
                'status': 'success',
                'enhanced_factors': optimized_factors,
                'synthetic_factors': synthetic_factors,
                'factor_quality_metrics': factor_quality,
                'recommendations': factor_recommendations,
                'improvement_over_base': factor_quality['improvement_metrics']
            }
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    # 私有方法实现
    async def _validate_and_preprocess_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        """验证和预处理输入数据"""
        try:
            # 基本形状检查
            if len(data.shape) != 4:  # (n_samples, n_assets, seq_length, n_features)
                print(f"❌ 数据形状错误: {data.shape}, 期望: (n_samples, n_assets, seq_length, n_features)")
                return None
            
            n_samples, n_assets, seq_length, n_features = data.shape
            
            # 检查数据大小限制
            if n_samples < 100:
                print(f"❌ 样本数量过少: {n_samples}, 最少需要100个样本")
                return None
            
            if seq_length < 24:  # 至少24小时数据
                print(f"❌ 时间序列长度过短: {seq_length}, 最少需要24个时间点")
                return None
            
            # 检查缺失值
            missing_ratio = np.isnan(data).sum() / data.size
            if missing_ratio > 0.05:  # 超过5%缺失
                print(f"⚠️ 数据缺失率较高: {missing_ratio:.2%}, 进行插值处理...")
                data = self._interpolate_missing_values(data)
            
            # 检查异常值
            data = self._detect_and_handle_outliers(data)
            
            # 数据标准化
            data = self._normalize_data(data)
            
            print(f"✅ 数据验证通过: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {str(e)}")
            return None
    
    async def _select_optimal_model_for_strategy(self, 
                                               strategy_type: str, 
                                               data: np.ndarray) -> str:
        """根据策略类型和数据特性选择最优模型"""
        
        # 分析数据特性
        volatility = np.std(np.diff(data[:, :, :, 0], axis=2))  # 价格波动率
        trend_strength = self._calculate_trend_strength(data)
        mean_reversion_strength = self._calculate_mean_reversion_strength(data)
        
        # 模型选择逻辑
        if strategy_type == 'grid':
            # 网格策略需要横盘整理的市场，VAE更适合
            return 'timevae'
        elif strategy_type == 'momentum':
            # 动量策略需要趋势延续，Flow模型更适合
            return 'fourier_flow'
        elif strategy_type == 'mean_reversion':
            # 均值回归策略，GAN模型更适合生成回归模式
            return 'quantgan'
        elif strategy_type == 'arbitrage':
            # 套利策略需要精确的价格关系，Diffusion模型更适合
            return 'diffusion_ts'
        elif volatility > 0.1:  # 高波动市场
            return 'quantgan'  # GAN更适合极端情况
        elif trend_strength > 0.7:  # 强趋势市场
            return 'fourier_flow'
        else:
            return 'timevae'  # 默认选择
    
    def _get_strategy_specific_generation_config(self, 
                                               strategy_type: str, 
                                               data: np.ndarray) -> Dict:
        """获取策略特定的生成配置"""
        
        base_config = {
            'temperature': 1.0,
            'top_p': 0.9,
            'diversity_penalty': 0.1
        }
        
        if strategy_type == 'grid':
            # 网格策略需要更稳定的价格波动
            base_config.update({
                'volatility_scaling': 0.8,
                'trend_suppression': 0.3,
                'range_bound_emphasis': 1.2
            })
        elif strategy_type == 'momentum':
            # 动量策略需要更明显的趋势
            base_config.update({
                'trend_amplification': 1.3,
                'momentum_persistence': 1.1,
                'reversal_suppression': 0.7
            })
        elif strategy_type == 'mean_reversion':
            # 均值回归需要更多的价格回归模式
            base_config.update({
                'mean_reversion_strength': 1.2,
                'oscillation_amplification': 1.1,
                'trend_suppression': 0.6
            })
        
        return base_config
    
    def _passes_quality_check(self, quality_metrics: Dict[str, float]) -> bool:
        """检查数据质量是否达标"""
        for metric, threshold in self.quality_thresholds.items():
            if quality_metrics.get(metric.replace('min_', ''), 0) < threshold:
                return False
        return True
    
    async def _blend_and_postprocess_data(self,
                                        real_data: np.ndarray,
                                        synthetic_data: np.ndarray,
                                        augmentation_ratio: float,
                                        strategy_type: str) -> np.ndarray:
        """混合真实数据和合成数据"""
        
        # 计算混合比例
        n_real = len(real_data)
        n_synthetic = int(n_real * augmentation_ratio / (1 - augmentation_ratio))
        n_synthetic = min(n_synthetic, len(synthetic_data))
        
        # 随机选择合成数据样本
        synthetic_indices = np.random.choice(
            len(synthetic_data), 
            size=n_synthetic, 
            replace=False
        )
        selected_synthetic = synthetic_data[synthetic_indices]
        
        # 数据混合
        blended_data = np.concatenate([real_data, selected_synthetic], axis=0)
        
        # 随机打乱顺序
        shuffle_indices = np.random.permutation(len(blended_data))
        blended_data = blended_data[shuffle_indices]
        
        # 策略特定的后处理
        if strategy_type == 'grid':
            # 确保价格在合理范围内
            blended_data = self._ensure_price_bounds(blended_data)
        elif strategy_type == 'momentum':
            # 增强趋势连续性
            blended_data = self._enhance_trend_continuity(blended_data)
        
        return blended_data
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\u5206\u6790CTBench\u4e0e\u73b0\u6709\u7cfb\u7edf\u7684\u96c6\u6210\u70b9", "status": "completed", "id": "14"}, {"content": "\u8bbe\u8ba1CTBench\u96c6\u6210\u67b6\u6784", "status": "completed", "id": "15"}, {"content": "\u66f4\u65b0\u5168\u6808\u67b6\u6784\u6587\u6863", "status": "completed", "id": "16"}, {"content": "\u751f\u6210CTBench\u96c6\u6210\u4ee3\u7801", "status": "completed", "id": "17"}]