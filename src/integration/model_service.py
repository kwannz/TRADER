"""
CTBench Model Service Integration
CTBench模型服务集成层，连接AI交易系统与时序生成模型
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

from ..ctbench.generators.synthetic_data_manager import SyntheticDataManager

@dataclass
class DataGenerationRequest:
    """数据生成请求"""
    model_type: str
    num_samples: int
    scenario_type: Optional[str] = None
    condition_data: Optional[np.ndarray] = None
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=高优先级, 2=中等, 3=低优先级

@dataclass
class ModelTrainingRequest:
    """模型训练请求"""
    model_type: str
    training_data: np.ndarray
    validation_data: Optional[np.ndarray] = None
    training_config: Optional[Dict[str, Any]] = None

class CTBenchModelService:
    """CTBench模型服务"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.synthetic_manager = SyntheticDataManager(config_path)
        self.request_queue = asyncio.Queue()
        self.training_queue = asyncio.Queue()
        self.is_running = False
        
        # 性能统计
        self.stats = {
            'requests_processed': 0,
            'models_trained': 0,
            'data_generated_samples': 0,
            'errors': 0,
            'last_reset': datetime.now()
        }
        
    async def start_service(self):
        """启动模型服务"""
        self.is_running = True
        self.logger.info("CTBench模型服务已启动")
        
        # 启动处理任务
        generation_task = asyncio.create_task(self._process_generation_queue())
        training_task = asyncio.create_task(self._process_training_queue())
        
        await asyncio.gather(generation_task, training_task)
        
    async def stop_service(self):
        """停止模型服务"""
        self.is_running = False
        self.logger.info("CTBench模型服务已停止")
        
    async def submit_generation_request(self, request: DataGenerationRequest) -> str:
        """提交数据生成请求"""
        request_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        await self.request_queue.put((request_id, request))
        self.logger.info(f"数据生成请求已提交: {request_id}")
        return request_id
        
    async def submit_training_request(self, request: ModelTrainingRequest) -> str:
        """提交模型训练请求"""
        request_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        await self.training_queue.put((request_id, request))
        self.logger.info(f"模型训练请求已提交: {request_id}")
        return request_id
        
    async def _process_generation_queue(self):
        """处理数据生成队列"""
        while self.is_running:
            try:
                if not self.request_queue.empty():
                    request_id, request = await self.request_queue.get()
                    await self._handle_generation_request(request_id, request)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"处理生成队列时出错: {e}")
                self.stats['errors'] += 1
                
    async def _process_training_queue(self):
        """处理模型训练队列"""
        while self.is_running:
            try:
                if not self.training_queue.empty():
                    request_id, request = await self.training_queue.get()
                    await self._handle_training_request(request_id, request)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"处理训练队列时出错: {e}")
                self.stats['errors'] += 1
                
    async def _handle_generation_request(self, request_id: str, 
                                       request: DataGenerationRequest):
        """处理数据生成请求"""
        try:
            self.logger.info(f"开始处理生成请求: {request_id}")
            
            if request.scenario_type:
                # 生成特定市场场景
                if request.condition_data is None:
                    raise ValueError("生成场景需要基础数据")
                    
                result = self.synthetic_manager.generate_market_scenarios(
                    request.scenario_type,
                    request.condition_data,
                    request.num_samples
                )
            else:
                # 生成一般合成数据
                result = self.synthetic_manager.generate_synthetic_data(
                    request.model_type,
                    request.num_samples,
                    request.condition_data
                )
                
            if result['success']:
                self.stats['requests_processed'] += 1
                self.stats['data_generated_samples'] += request.num_samples
                self.logger.info(f"生成请求完成: {request_id}")
                
                # 这里可以添加结果存储逻辑
                await self._store_generation_result(request_id, result)
            else:
                self.logger.error(f"生成请求失败: {request_id}, 错误: {result.get('error')}")
                self.stats['errors'] += 1
                
        except Exception as e:
            self.logger.error(f"处理生成请求 {request_id} 时出错: {e}")
            self.stats['errors'] += 1
            
    async def _handle_training_request(self, request_id: str,
                                     request: ModelTrainingRequest):
        """处理模型训练请求"""
        try:
            self.logger.info(f"开始处理训练请求: {request_id}")
            
            result = self.synthetic_manager.train_model(
                request.model_type,
                request.training_data,
                request.validation_data
            )
            
            if result['success']:
                self.stats['models_trained'] += 1
                self.logger.info(f"训练请求完成: {request_id}")
                
                # 保存训练结果
                await self._store_training_result(request_id, result)
            else:
                self.logger.error(f"训练请求失败: {request_id}, 错误: {result.get('error')}")
                self.stats['errors'] += 1
                
        except Exception as e:
            self.logger.error(f"处理训练请求 {request_id} 时出错: {e}")
            self.stats['errors'] += 1
            
    async def _store_generation_result(self, request_id: str, result: Dict[str, Any]):
        """存储生成结果"""
        # 这里可以实现结果存储逻辑（Redis、数据库等）
        # 现在只是记录日志
        self.logger.info(f"生成结果已存储: {request_id}, 样本数: {result.get('num_samples', 0)}")
        
    async def _store_training_result(self, request_id: str, result: Dict[str, Any]):
        """存储训练结果"""
        # 这里可以实现训练结果存储逻辑
        self.logger.info(f"训练结果已存储: {request_id}, 模型: {result.get('model_type')}")
        
    async def get_real_time_market_data_augmentation(self, 
                                                   market_data: np.ndarray,
                                                   augmentation_factor: int = 5) -> Dict[str, Any]:
        """实时市场数据增强"""
        try:
            # 使用最新的市场数据作为条件生成增强数据
            augmented_data = []
            
            # 生成不同场景的增强数据
            scenarios = ['black_swan', 'high_volatility', 'bull_market', 'bear_market']
            samples_per_scenario = augmentation_factor // len(scenarios)
            
            for scenario in scenarios:
                scenario_result = self.synthetic_manager.generate_market_scenarios(
                    scenario, market_data, samples_per_scenario
                )
                
                if scenario_result['success']:
                    augmented_data.extend(scenario_result['data'])
                    
            return {
                'success': True,
                'original_data_shape': market_data.shape,
                'augmented_data': np.array(augmented_data),
                'augmentation_factor': len(augmented_data),
                'scenarios_generated': scenarios
            }
            
        except Exception as e:
            self.logger.error(f"实时数据增强失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def generate_stress_test_scenarios(self, 
                                           base_data: np.ndarray,
                                           stress_types: List[str] = None) -> Dict[str, Any]:
        """生成压力测试场景"""
        if stress_types is None:
            stress_types = ['black_swan', 'high_volatility', 'bear_market']
            
        stress_scenarios = {}
        
        try:
            for stress_type in stress_types:
                result = self.synthetic_manager.generate_market_scenarios(
                    stress_type, base_data, 50  # 每种类型生成50个场景
                )
                
                if result['success']:
                    stress_scenarios[stress_type] = result['data']
                else:
                    self.logger.warning(f"生成压力测试场景 {stress_type} 失败")
                    
            return {
                'success': True,
                'stress_scenarios': stress_scenarios,
                'total_scenarios': sum(len(scenarios) for scenarios in stress_scenarios.values())
            }
            
        except Exception as e:
            self.logger.error(f"生成压力测试场景失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def evaluate_model_performance(self, model_type: str,
                                       test_data: np.ndarray) -> Dict[str, Any]:
        """评估模型性能"""
        try:
            # 生成测试数据
            generated_result = self.synthetic_manager.generate_synthetic_data(
                model_type, test_data.shape[0]
            )
            
            if not generated_result['success']:
                return generated_result
                
            generated_data = generated_result['data']
            
            # 计算各种评估指标
            metrics = self._calculate_metrics(test_data, generated_data)
            
            return {
                'success': True,
                'model_type': model_type,
                'metrics': metrics,
                'test_data_shape': test_data.shape,
                'generated_data_shape': generated_data.shape
            }
            
        except Exception as e:
            self.logger.error(f"评估模型性能失败: {e}")
            return {'success': False, 'error': str(e)}
            
    def _calculate_metrics(self, real_data: np.ndarray, 
                          synthetic_data: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        try:
            # 统计特征相似性
            real_mean = np.mean(real_data, axis=(0, 1))
            synthetic_mean = np.mean(synthetic_data, axis=(0, 1))
            metrics['mean_difference'] = np.mean(np.abs(real_mean - synthetic_mean))
            
            real_std = np.std(real_data, axis=(0, 1))
            synthetic_std = np.std(synthetic_data, axis=(0, 1))
            metrics['std_difference'] = np.mean(np.abs(real_std - synthetic_std))
            
            # 分布相似性 (简化版本)
            real_flat = real_data.reshape(-1, real_data.shape[-1])
            synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            
            correlation_sum = 0
            for i in range(real_data.shape[-1]):
                corr = np.corrcoef(real_flat[:, i], synthetic_flat[:, i])[0, 1]
                if not np.isnan(corr):
                    correlation_sum += corr
                    
            metrics['average_correlation'] = correlation_sum / real_data.shape[-1]
            
            # 波动率相似性
            real_returns = np.diff(real_data[:, :, 0], axis=1)  # 假设第0列是价格
            synthetic_returns = np.diff(synthetic_data[:, :, 0], axis=1)
            
            real_volatility = np.std(real_returns)
            synthetic_volatility = np.std(synthetic_returns)
            metrics['volatility_ratio'] = synthetic_volatility / real_volatility if real_volatility != 0 else 0
            
        except Exception as e:
            self.logger.warning(f"计算部分指标时出错: {e}")
            
        return metrics
        
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        uptime = datetime.now() - self.stats['last_reset']
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'queue_sizes': {
                'generation_queue': self.request_queue.qsize(),
                'training_queue': self.training_queue.qsize()
            },
            'model_status': self.synthetic_manager.get_model_status(),
            'is_running': self.is_running
        }
        
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'requests_processed': 0,
            'models_trained': 0,
            'data_generated_samples': 0,
            'errors': 0,
            'last_reset': datetime.now()
        }
        self.logger.info("服务统计信息已重置")

# 单例模式的服务实例
ctbench_service = None

async def get_ctbench_service(config_path: Optional[str] = None) -> CTBenchModelService:
    """获取CTBench服务单例"""
    global ctbench_service
    if ctbench_service is None:
        ctbench_service = CTBenchModelService(config_path)
    return ctbench_service