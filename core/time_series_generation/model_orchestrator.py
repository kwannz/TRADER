"""
CTBench模型编排系统
统一管理和协调多个时间序列生成模型，提供模型集成、负载均衡和自动选择功能
"""

import asyncio
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
from abc import ABC, abstractmethod

from .ctbench_models import BaseGenerativeModel, CTBenchModelFactory, CTBenchConfig, GenerativeModelType
from .ctbench_evaluator import CTBenchEvaluator, EvaluationMetrics

class OrchestratorMode(Enum):
    """编排模式"""
    SINGLE_BEST = "single_best"  # 使用单个最佳模型
    ENSEMBLE = "ensemble"  # 集成多个模型
    ADAPTIVE = "adaptive"  # 自适应选择
    LOAD_BALANCED = "load_balanced"  # 负载均衡

class ModelStatus(Enum):
    """模型状态"""
    IDLE = "idle"
    TRAINING = "training" 
    GENERATING = "generating"
    EVALUATING = "evaluating"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_type: GenerativeModelType
    config: CTBenchConfig
    
    # 状态信息
    status: ModelStatus = ModelStatus.IDLE
    last_updated: datetime = None
    
    # 性能指标
    quality_score: float = 0.0
    generation_speed: float = 0.0  # 每秒生成样本数
    training_time: float = 0.0  # 训练时间(小时)
    memory_usage: float = 0.0  # 内存使用(MB)
    
    # 使用统计
    total_generations: int = 0
    total_samples_generated: int = 0
    success_rate: float = 1.0
    
    # 专长领域
    best_market_regimes: List[str] = field(default_factory=list)
    optimal_sequence_lengths: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()

@dataclass
class GenerationRequest:
    """生成请求"""
    request_id: str
    num_samples: int
    sequence_length: Optional[int] = None
    conditions: Optional[Dict[str, Any]] = None
    quality_requirements: Optional[Dict[str, float]] = None
    priority: int = 1  # 1-10，数字越大优先级越高
    timeout: float = 300.0  # 超时时间(秒)
    
    # 请求元数据
    created_at: datetime = field(default_factory=datetime.utcnow)
    requested_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None

@dataclass 
class GenerationResult:
    """生成结果"""
    request_id: str
    samples: np.ndarray
    model_id: str
    generation_time: float
    quality_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class ModelPool:
    """模型池"""
    
    def __init__(self, max_concurrent_models: int = 4):
        self.models: Dict[str, BaseGenerativeModel] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.max_concurrent_models = max_concurrent_models
        
        # 线程安全
        self.lock = threading.RLock()
        self.generation_locks: Dict[str, threading.Lock] = {}
        
        # 性能监控
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger = logging.getLogger("ModelPool")
    
    def add_model(self, model_id: str, model: BaseGenerativeModel, 
                  metadata: ModelMetadata):
        """添加模型到池中"""
        with self.lock:
            self.models[model_id] = model
            self.metadata[model_id] = metadata
            self.generation_locks[model_id] = threading.Lock()
            self.performance_history[model_id] = []
        
        self.logger.info(f"添加模型到池中: {model_id} ({metadata.model_type.value})")
    
    def remove_model(self, model_id: str) -> bool:
        """从池中移除模型"""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                del self.metadata[model_id]
                del self.generation_locks[model_id]
                del self.performance_history[model_id]
                self.logger.info(f"从池中移除模型: {model_id}")
                return True
            return False
    
    def get_model(self, model_id: str) -> Optional[BaseGenerativeModel]:
        """获取模型"""
        return self.models.get(model_id)
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self.metadata.get(model_id)
    
    def list_available_models(self, status_filter: Optional[List[ModelStatus]] = None) -> List[str]:
        """列出可用模型"""
        if status_filter is None:
            status_filter = [ModelStatus.IDLE]
        
        available = []
        with self.lock:
            for model_id, metadata in self.metadata.items():
                if metadata.status in status_filter:
                    available.append(model_id)
        
        return available
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """更新模型状态"""
        with self.lock:
            if model_id in self.metadata:
                self.metadata[model_id].status = status
                self.metadata[model_id].last_updated = datetime.utcnow()
    
    def record_performance(self, model_id: str, performance_data: Dict[str, Any]):
        """记录性能数据"""
        with self.lock:
            if model_id in self.performance_history:
                performance_data['timestamp'] = datetime.utcnow()
                self.performance_history[model_id].append(performance_data)
                
                # 保持历史记录不过长
                if len(self.performance_history[model_id]) > 1000:
                    self.performance_history[model_id] = self.performance_history[model_id][-500:]
    
    def get_best_models(self, criteria: str = "quality_score", 
                       top_k: int = 3) -> List[str]:
        """获取最佳模型"""
        with self.lock:
            model_scores = []
            for model_id, metadata in self.metadata.items():
                if metadata.status not in [ModelStatus.ERROR, ModelStatus.DISABLED]:
                    score = getattr(metadata, criteria, 0)
                    model_scores.append((model_id, score))
            
            # 按得分排序
            model_scores.sort(key=lambda x: x[1], reverse=True)
            return [model_id for model_id, _ in model_scores[:top_k]]

class ModelSelector:
    """模型选择器"""
    
    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool
        self.logger = logging.getLogger("ModelSelector")
        
        # 选择策略
        self.selection_strategies = {
            OrchestratorMode.SINGLE_BEST: self._select_single_best,
            OrchestratorMode.ENSEMBLE: self._select_for_ensemble,
            OrchestratorMode.ADAPTIVE: self._select_adaptive,
            OrchestratorMode.LOAD_BALANCED: self._select_load_balanced
        }
    
    def select_models(self, request: GenerationRequest, 
                     mode: OrchestratorMode) -> List[str]:
        """选择模型"""
        strategy = self.selection_strategies.get(mode)
        if not strategy:
            raise ValueError(f"不支持的选择模式: {mode}")
        
        return strategy(request)
    
    def _select_single_best(self, request: GenerationRequest) -> List[str]:
        """选择单个最佳模型"""
        available_models = self.model_pool.list_available_models()
        
        # 如果指定了模型，优先使用
        if request.requested_models:
            for model_id in request.requested_models:
                if model_id in available_models:
                    return [model_id]
        
        # 否则选择质量得分最高的模型
        best_models = self.model_pool.get_best_models("quality_score", 1)
        return best_models if best_models else available_models[:1]
    
    def _select_for_ensemble(self, request: GenerationRequest) -> List[str]:
        """选择集成模型"""
        available_models = self.model_pool.list_available_models()
        
        # 选择不同类型的高质量模型
        selected = []
        seen_types = set()
        
        best_models = self.model_pool.get_best_models("quality_score", len(available_models))
        
        for model_id in best_models:
            metadata = self.model_pool.get_metadata(model_id)
            if metadata and metadata.model_type not in seen_types:
                selected.append(model_id)
                seen_types.add(metadata.model_type)
                
                if len(selected) >= 3:  # 最多选择3个模型进行集成
                    break
        
        return selected if selected else available_models[:3]
    
    def _select_adaptive(self, request: GenerationRequest) -> List[str]:
        """自适应选择"""
        available_models = self.model_pool.list_available_models()
        
        # 根据请求条件选择最适合的模型
        scores = {}
        
        for model_id in available_models:
            metadata = self.model_pool.get_metadata(model_id)
            if not metadata:
                continue
            
            score = metadata.quality_score
            
            # 考虑序列长度适配性
            if (request.sequence_length and 
                metadata.optimal_sequence_lengths and
                request.sequence_length in metadata.optimal_sequence_lengths):
                score += 0.2
            
            # 考虑市场状态适配性
            if (request.conditions and 
                "market_regime" in request.conditions and
                metadata.best_market_regimes):
                regime = request.conditions["market_regime"]
                if regime in metadata.best_market_regimes:
                    score += 0.3
            
            # 考虑成功率
            score *= metadata.success_rate
            
            scores[model_id] = score
        
        # 选择得分最高的模型
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [sorted_models[0][0]] if sorted_models else available_models[:1]
    
    def _select_load_balanced(self, request: GenerationRequest) -> List[str]:
        """负载均衡选择"""
        available_models = self.model_pool.list_available_models()
        
        # 选择最近使用最少的高质量模型
        usage_scores = {}
        
        for model_id in available_models:
            metadata = self.model_pool.get_metadata(model_id)
            if not metadata:
                continue
            
            # 质量权重
            quality_weight = metadata.quality_score * 0.7
            
            # 使用频率权重 (使用越少权重越高)
            time_since_last_use = (datetime.utcnow() - metadata.last_updated).total_seconds()
            usage_weight = min(1.0, time_since_last_use / 3600) * 0.3  # 1小时内的使用频率
            
            usage_scores[model_id] = quality_weight + usage_weight
        
        # 选择得分最高的模型
        sorted_models = sorted(usage_scores.items(), key=lambda x: x[1], reverse=True)
        return [sorted_models[0][0]] if sorted_models else available_models[:1]

class GenerationWorker:
    """生成工作器"""
    
    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool
        self.logger = logging.getLogger("GenerationWorker")
    
    async def generate_samples(self, model_id: str, request: GenerationRequest) -> GenerationResult:
        """生成样本"""
        start_time = time.time()
        
        try:
            # 获取模型
            model = self.model_pool.get_model(model_id)
            if not model:
                raise ValueError(f"模型不存在: {model_id}")
            
            # 更新状态
            self.model_pool.update_model_status(model_id, ModelStatus.GENERATING)
            
            # 获取生成锁
            generation_lock = self.model_pool.generation_locks.get(model_id)
            if not generation_lock:
                raise RuntimeError(f"模型 {model_id} 没有生成锁")
            
            with generation_lock:
                # 准备条件
                condition = None
                if request.conditions:
                    # 转换条件为模型可接受的格式
                    condition = self._prepare_condition(request.conditions, request)
                
                # 生成样本
                samples = model.generate_samples(
                    num_samples=request.num_samples,
                    condition=condition
                )
                
                generation_time = time.time() - start_time
                
                # 更新统计
                metadata = self.model_pool.get_metadata(model_id)
                if metadata:
                    metadata.total_generations += 1
                    metadata.total_samples_generated += request.num_samples
                    metadata.generation_speed = request.num_samples / generation_time
                
                # 记录性能
                self.model_pool.record_performance(model_id, {
                    "generation_time": generation_time,
                    "samples_generated": request.num_samples,
                    "success": True
                })
                
                # 更新状态
                self.model_pool.update_model_status(model_id, ModelStatus.IDLE)
                
                return GenerationResult(
                    request_id=request.request_id,
                    samples=samples,
                    model_id=model_id,
                    generation_time=generation_time,
                    metadata={"conditions": request.conditions}
                )
                
        except Exception as e:
            generation_time = time.time() - start_time
            
            # 更新失败统计
            metadata = self.model_pool.get_metadata(model_id)
            if metadata:
                total_attempts = metadata.total_generations + 1
                successful_attempts = metadata.total_generations * metadata.success_rate
                metadata.success_rate = successful_attempts / total_attempts
                metadata.total_generations = total_attempts
            
            # 记录错误
            self.model_pool.record_performance(model_id, {
                "generation_time": generation_time,
                "samples_generated": 0,
                "success": False,
                "error": str(e)
            })
            
            # 更新状态
            self.model_pool.update_model_status(model_id, ModelStatus.ERROR)
            
            self.logger.error(f"生成失败 {model_id}: {e}")
            raise
    
    def _prepare_condition(self, conditions: Dict[str, Any], 
                          request: GenerationRequest) -> Optional[torch.Tensor]:
        """准备条件张量"""
        # 这里可以根据具体需求实现条件准备逻辑
        # 例如将市场状态、技术指标等转换为模型可接受的格式
        return None

class EnsembleGenerator:
    """集成生成器"""
    
    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool
        self.logger = logging.getLogger("EnsembleGenerator")
        
        # 集成策略
        self.ensemble_strategies = {
            "average": self._average_ensemble,
            "weighted_average": self._weighted_average_ensemble,
            "best_quality": self._best_quality_ensemble,
            "diversity_selection": self._diversity_selection_ensemble
        }
    
    async def generate_ensemble_samples(self, model_ids: List[str], 
                                       request: GenerationRequest,
                                       strategy: str = "weighted_average") -> GenerationResult:
        """集成生成样本"""
        ensemble_func = self.ensemble_strategies.get(strategy, self._weighted_average_ensemble)
        
        # 并行生成
        generation_tasks = []
        generation_worker = GenerationWorker(self.model_pool)
        
        for model_id in model_ids:
            task = generation_worker.generate_samples(model_id, request)
            generation_tasks.append(task)
        
        # 等待所有生成完成
        results = []
        for task in asyncio.as_completed(generation_tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                self.logger.warning(f"模型生成失败: {e}")
        
        if not results:
            raise RuntimeError("所有模型生成都失败了")
        
        # 集成结果
        ensemble_samples = ensemble_func(results)
        
        return GenerationResult(
            request_id=request.request_id,
            samples=ensemble_samples,
            model_id=f"ensemble_{'_'.join([r.model_id for r in results])}",
            generation_time=max(r.generation_time for r in results),
            metadata={
                "ensemble_strategy": strategy,
                "contributing_models": [r.model_id for r in results],
                "model_count": len(results)
            }
        )
    
    def _average_ensemble(self, results: List[GenerationResult]) -> np.ndarray:
        """平均集成"""
        return np.mean([r.samples for r in results], axis=0)
    
    def _weighted_average_ensemble(self, results: List[GenerationResult]) -> np.ndarray:
        """加权平均集成"""
        weights = []
        samples_list = []
        
        for result in results:
            metadata = self.model_pool.get_metadata(result.model_id)
            weight = metadata.quality_score if metadata else 1.0
            weights.append(weight)
            samples_list.append(result.samples)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 加权平均
        weighted_samples = np.zeros_like(samples_list[0])
        for samples, weight in zip(samples_list, weights):
            weighted_samples += samples * weight
        
        return weighted_samples
    
    def _best_quality_ensemble(self, results: List[GenerationResult]) -> np.ndarray:
        """选择最佳质量模型的结果"""
        best_result = results[0]
        best_quality = 0
        
        for result in results:
            metadata = self.model_pool.get_metadata(result.model_id)
            quality = metadata.quality_score if metadata else 0
            if quality > best_quality:
                best_quality = quality
                best_result = result
        
        return best_result.samples
    
    def _diversity_selection_ensemble(self, results: List[GenerationResult]) -> np.ndarray:
        """多样性选择集成"""
        # 从每个模型选择不同的样本
        all_samples = []
        samples_per_model = len(results[0].samples) // len(results)
        
        for i, result in enumerate(results):
            start_idx = i * samples_per_model
            end_idx = start_idx + samples_per_model
            if i == len(results) - 1:  # 最后一个模型取剩余所有样本
                end_idx = len(result.samples)
            
            all_samples.append(result.samples[start_idx:end_idx])
        
        return np.concatenate(all_samples, axis=0)

class ModelOrchestrator:
    """模型编排器"""
    
    def __init__(self, mode: OrchestratorMode = OrchestratorMode.ADAPTIVE,
                 max_concurrent_models: int = 4,
                 enable_auto_evaluation: bool = True):
        
        self.mode = mode
        self.enable_auto_evaluation = enable_auto_evaluation
        
        # 核心组件
        self.model_pool = ModelPool(max_concurrent_models)
        self.model_selector = ModelSelector(self.model_pool)
        self.ensemble_generator = EnsembleGenerator(self.model_pool)
        self.evaluator = CTBenchEvaluator() if enable_auto_evaluation else None
        
        # 请求队列
        self.request_queue = asyncio.Queue()
        self.active_requests: Dict[str, GenerationRequest] = {}
        
        # 工作状态
        self.is_running = False
        self.worker_tasks = []
        
        # 统计信息
        self.total_requests_processed = 0
        self.total_samples_generated = 0
        self.average_response_time = 0.0
        
        self.logger = logging.getLogger("ModelOrchestrator")
    
    async def start(self, num_workers: int = 2):
        """启动编排器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动工作器
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker_{i}"))
            self.worker_tasks.append(task)
        
        # 如果启用了自动评估，启动评估任务
        if self.enable_auto_evaluation:
            eval_task = asyncio.create_task(self._auto_evaluation_loop())
            self.worker_tasks.append(eval_task)
        
        self.logger.info(f"模型编排器已启动，{num_workers}个工作器，模式: {self.mode.value}")
    
    async def stop(self):
        """停止编排器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 取消所有工作器任务
        for task in self.worker_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        self.logger.info("模型编排器已停止")
    
    def register_model(self, model_id: str, config: CTBenchConfig,
                      checkpoint_path: Optional[str] = None) -> bool:
        """注册模型"""
        try:
            # 创建模型
            model = CTBenchModelFactory.create_model(config)
            
            # 加载检查点
            if checkpoint_path and os.path.exists(checkpoint_path):
                success = model.load_checkpoint(checkpoint_path)
                if not success:
                    self.logger.warning(f"加载检查点失败: {checkpoint_path}")
            
            # 创建元数据
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=config.model_type,
                config=config
            )
            
            # 添加到模型池
            self.model_pool.add_model(model_id, model, metadata)
            
            self.logger.info(f"模型注册成功: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型注册失败 {model_id}: {e}")
            return False
    
    def unregister_model(self, model_id: str) -> bool:
        """注销模型"""
        return self.model_pool.remove_model(model_id)
    
    async def generate_samples_async(self, request: GenerationRequest) -> GenerationResult:
        """异步生成样本"""
        # 添加到请求队列
        await self.request_queue.put(request)
        self.active_requests[request.request_id] = request
        
        # 等待结果 (这里简化处理，实际应该使用事件机制)
        timeout = request.timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request.request_id not in self.active_requests:
                # 请求已完成，从某个结果缓存中获取
                break
            await asyncio.sleep(0.1)
        
        # 这里应该返回实际结果，简化处理
        raise NotImplementedError("需要实现结果返回机制")
    
    def generate_samples_sync(self, num_samples: int, 
                             sequence_length: Optional[int] = None,
                             conditions: Optional[Dict[str, Any]] = None,
                             quality_requirements: Optional[Dict[str, float]] = None) -> np.ndarray:
        """同步生成样本"""
        request = GenerationRequest(
            request_id=f"sync_{int(time.time() * 1000)}",
            num_samples=num_samples,
            sequence_length=sequence_length,
            conditions=conditions,
            quality_requirements=quality_requirements
        )
        
        # 选择模型
        selected_models = self.model_selector.select_models(request, self.mode)
        
        if not selected_models:
            raise RuntimeError("没有可用的模型")
        
        # 生成样本
        if self.mode == OrchestratorMode.ENSEMBLE and len(selected_models) > 1:
            # 集成生成 (简化同步版本)
            results = []
            generation_worker = GenerationWorker(self.model_pool)
            
            for model_id in selected_models:
                try:
                    # 在同步环境中运行异步函数
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        generation_worker.generate_samples(model_id, request)
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"模型 {model_id} 生成失败: {e}")
                finally:
                    loop.close()
            
            if not results:
                raise RuntimeError("所有模型生成都失败了")
            
            # 使用加权平均集成
            ensemble_samples = self.ensemble_generator._weighted_average_ensemble(results)
            return ensemble_samples
        
        else:
            # 单模型生成
            model_id = selected_models[0]
            generation_worker = GenerationWorker(self.model_pool)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    generation_worker.generate_samples(model_id, request)
                )
                return result.samples
            finally:
                loop.close()
    
    async def _worker_loop(self, worker_name: str):
        """工作器循环"""
        self.logger.info(f"工作器 {worker_name} 已启动")
        
        while self.is_running:
            try:
                # 获取请求
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                # 处理请求
                await self._process_request(request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"工作器 {worker_name} 处理请求失败: {e}")
    
    async def _process_request(self, request: GenerationRequest):
        """处理请求"""
        try:
            start_time = time.time()
            
            # 选择模型
            selected_models = self.model_selector.select_models(request, self.mode)
            
            if not selected_models:
                self.logger.error(f"请求 {request.request_id}: 没有可用模型")
                return
            
            # 生成样本
            if self.mode == OrchestratorMode.ENSEMBLE and len(selected_models) > 1:
                result = await self.ensemble_generator.generate_ensemble_samples(
                    selected_models, request
                )
            else:
                model_id = selected_models[0]
                generation_worker = GenerationWorker(self.model_pool)
                result = await generation_worker.generate_samples(model_id, request)
            
            # 处理结果
            processing_time = time.time() - start_time
            
            # 更新统计
            self.total_requests_processed += 1
            self.total_samples_generated += request.num_samples
            self.average_response_time = (
                (self.average_response_time * (self.total_requests_processed - 1) + processing_time) /
                self.total_requests_processed
            )
            
            # 移除活跃请求
            self.active_requests.pop(request.request_id, None)
            
            self.logger.info(f"请求完成 {request.request_id}: {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"处理请求失败 {request.request_id}: {e}")
            self.active_requests.pop(request.request_id, None)
    
    async def _auto_evaluation_loop(self):
        """自动评估循环"""
        self.logger.info("自动评估任务已启动")
        
        while self.is_running:
            try:
                # 每小时进行一次评估
                await asyncio.sleep(3600)
                
                # 评估所有模型
                await self._evaluate_all_models()
                
            except Exception as e:
                self.logger.error(f"自动评估失败: {e}")
    
    async def _evaluate_all_models(self):
        """评估所有模型"""
        # 这里应该实现模型评估逻辑
        # 生成测试样本，与真实数据比较，更新质量得分
        pass
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器统计信息"""
        return {
            "mode": self.mode.value,
            "is_running": self.is_running,
            "total_requests_processed": self.total_requests_processed,
            "total_samples_generated": self.total_samples_generated,
            "average_response_time": self.average_response_time,
            "active_requests": len(self.active_requests),
            "registered_models": len(self.model_pool.models),
            "available_models": len(self.model_pool.list_available_models()),
            "model_stats": {
                model_id: {
                    "status": metadata.status.value,
                    "quality_score": metadata.quality_score,
                    "total_generations": metadata.total_generations,
                    "success_rate": metadata.success_rate
                }
                for model_id, metadata in self.model_pool.metadata.items()
            }
        }
    
    def switch_mode(self, new_mode: OrchestratorMode):
        """切换编排模式"""
        old_mode = self.mode
        self.mode = new_mode
        self.logger.info(f"编排模式已切换: {old_mode.value} -> {new_mode.value}")

# 全局编排器实例
default_orchestrator = ModelOrchestrator()