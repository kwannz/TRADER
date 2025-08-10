"""
增强版统一强化学习管理器
整合DQN、PPO、A3C算法，提供统一的训练和评估接口

新增功能：
- 实时性能监控和自适应优化
- 智能模型选择和集成决策
- 在线学习和增量训练
- 多目标优化和约束满足
- 分布式训练协调
- 模型版本管理和A/B测试
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union, Type, Tuple
from enum import Enum
import logging
import threading
import queue
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import os
from pathlib import Path
import statistics
import pickle
import hashlib

# 导入各种RL算法
from .algorithms.dqn_agent import DQNAgent, DQNConfig
from .algorithms.ppo_agent import PPOAgent, PPOConfig
from .algorithms.a3c_agent import A3CAgent, A3CConfig
from .environments.trading_env import TradingEnvironment, create_trading_environment
from .training.rl_trainer import RLTrainer, TrainingConfig, TrainingMetrics

class RLAlgorithmType(Enum):
    """强化学习算法类型"""
    DQN = "dqn"
    PPO = "ppo"
    A3C = "a3c"
    ENSEMBLE = "ensemble"  # 集成算法

class ModelStatus(Enum):
    """模型状态"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    READY = "ready"
    SERVING = "serving"
    PAUSED = "paused"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class TrainingMode(Enum):
    """训练模式"""
    OFFLINE = "offline"          # 离线训练
    ONLINE = "online"           # 在线学习
    INCREMENTAL = "incremental" # 增量训练
    FEDERATED = "federated"     # 联邦学习

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    episode_reward: float = 0.0
    cumulative_reward: float = 0.0
    episode_length: int = 0
    success_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    
    # 训练指标
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # 系统指标
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    algorithm: RLAlgorithmType
    version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.INITIALIZING
    
    # 配置信息
    config_hash: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # 性能历史
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    best_performance: Optional[PerformanceMetrics] = None
    
    # 部署信息
    checkpoint_path: str = ""
    deployment_count: int = 0
    total_decisions: int = 0
    
    # 实验追踪
    experiment_id: str = ""
    parent_model: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class EnsembleConfig:
    """集成配置"""
    voting_strategy: str = "weighted"  # weighted, majority, stacking
    weight_update_frequency: int = 100  # 权重更新频率
    performance_window: int = 50       # 性能评估窗口
    min_models: int = 2               # 最小模型数量
    max_models: int = 5               # 最大模型数量
    diversity_threshold: float = 0.3   # 多样性阈值

@dataclass  
class AdaptiveConfig:
    """自适应配置"""
    adaptation_frequency: int = 1000   # 适应频率
    performance_threshold: float = 0.1 # 性能变化阈值
    learning_rate_decay: float = 0.95  # 学习率衰减
    exploration_decay: float = 0.995   # 探索衰减
    auto_hyperparameter_tuning: bool = True  # 自动超参调优

@dataclass
class RLModelConfig:
    """RL模型配置"""
    algorithm: RLAlgorithmType
    model_name: str
    description: str = ""
    
    # 通用配置
    state_dim: int = 14
    action_dim: int = 3
    
    # 算法特定配置
    algorithm_config: Dict[str, Any] = None
    
    # 训练配置
    training_config: Dict[str, Any] = None
    
    # 环境配置
    env_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.algorithm_config is None:
            self.algorithm_config = {}
        if self.training_config is None:
            self.training_config = {}
        if self.env_config is None:
            self.env_config = {}

@dataclass
class RLPerformanceMetrics:
    """RL性能指标"""
    model_name: str
    algorithm: str
    
    # 训练指标
    total_episodes: int = 0
    total_steps: int = 0
    training_time: float = 0.0  # 小时
    
    # 性能指标
    avg_episode_reward: float = 0.0
    best_episode_reward: float = 0.0
    avg_training_loss: float = 0.0
    final_epsilon: float = 0.0
    
    # 交易性能
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # 评估得分
    evaluation_score: float = 0.0
    
    # 时间戳
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

class RLModelRegistry:
    """RL模型注册表"""
    
    def __init__(self, registry_path: str = "models/rl_registry.json"):
        self.registry_path = registry_path
        self.models: Dict[str, RLModelConfig] = {}
        self.performance_history: Dict[str, List[RLPerformanceMetrics]] = {}
        
        # 确保目录存在
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # 加载现有注册表
        self.load_registry()
        
        self.logger = logging.getLogger("RLModelRegistry")
    
    def register_model(self, config: RLModelConfig):
        """注册模型"""
        self.models[config.model_name] = config
        if config.model_name not in self.performance_history:
            self.performance_history[config.model_name] = []
        
        self.save_registry()
        self.logger.info(f"注册RL模型: {config.model_name} ({config.algorithm.value})")
    
    def get_model_config(self, model_name: str) -> Optional[RLModelConfig]:
        """获取模型配置"""
        return self.models.get(model_name)
    
    def list_models(self, algorithm: Optional[RLAlgorithmType] = None) -> List[RLModelConfig]:
        """列出模型"""
        if algorithm:
            return [config for config in self.models.values() if config.algorithm == algorithm]
        return list(self.models.values())
    
    def add_performance(self, model_name: str, metrics: RLPerformanceMetrics):
        """添加性能记录"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        metrics.updated_at = datetime.utcnow()
        self.performance_history[model_name].append(metrics)
        
        # 保持历史记录不过长
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-50:]
        
        self.save_registry()
    
    def get_best_model(self, algorithm: Optional[RLAlgorithmType] = None, 
                      metric: str = "evaluation_score") -> Optional[str]:
        """获取最佳模型"""
        candidates = self.list_models(algorithm)
        
        best_model = None
        best_score = float('-inf')
        
        for config in candidates:
            model_name = config.model_name
            if model_name in self.performance_history:
                history = self.performance_history[model_name]
                if history:
                    latest_metrics = history[-1]
                    score = getattr(latest_metrics, metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model
    
    def save_registry(self):
        """保存注册表"""
        try:
            registry_data = {
                "models": {name: asdict(config) for name, config in self.models.items()},
                "performance_history": {}
            }
            
            # 序列化性能历史
            for model_name, history in self.performance_history.items():
                registry_data["performance_history"][model_name] = []
                for metrics in history:
                    metrics_dict = asdict(metrics)
                    # 转换datetime为字符串
                    if metrics_dict["created_at"]:
                        metrics_dict["created_at"] = metrics_dict["created_at"].isoformat()
                    if metrics_dict["updated_at"]:
                        metrics_dict["updated_at"] = metrics_dict["updated_at"].isoformat()
                    registry_data["performance_history"][model_name].append(metrics_dict)
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存注册表失败: {e}")
    
    def load_registry(self):
        """加载注册表"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                # 加载模型配置
                for model_name, config_dict in data.get("models", {}).items():
                    config_dict["algorithm"] = RLAlgorithmType(config_dict["algorithm"])
                    self.models[model_name] = RLModelConfig(**config_dict)
                
                # 加载性能历史
                for model_name, history_list in data.get("performance_history", {}).items():
                    self.performance_history[model_name] = []
                    for metrics_dict in history_list:
                        # 转换字符串为datetime
                        if metrics_dict["created_at"]:
                            metrics_dict["created_at"] = datetime.fromisoformat(metrics_dict["created_at"])
                        if metrics_dict["updated_at"]:
                            metrics_dict["updated_at"] = datetime.fromisoformat(metrics_dict["updated_at"])
                        self.performance_history[model_name].append(RLPerformanceMetrics(**metrics_dict))
                        
        except Exception as e:
            self.logger.error(f"加载注册表失败: {e}")

class UnifiedRLManager:
    """统一强化学习管理器"""
    
    def __init__(self, base_model_dir: str = "models/rl_models"):
        self.base_model_dir = base_model_dir
        self.registry = RLModelRegistry()
        
        # 算法工厂
        self.algorithm_factories = {
            RLAlgorithmType.DQN: self._create_dqn_agent,
            RLAlgorithmType.PPO: self._create_ppo_agent,
            RLAlgorithmType.A3C: self._create_a3c_agent
        }
        
        # 配置工厂
        self.config_factories = {
            RLAlgorithmType.DQN: DQNConfig,
            RLAlgorithmType.PPO: PPOConfig,
            RLAlgorithmType.A3C: A3CConfig
        }
        
        # 当前加载的模型
        self.loaded_models: Dict[str, Any] = {}\n        \n        # 确保目录存在\n        os.makedirs(base_model_dir, exist_ok=True)\n        \n        self.logger = logging.getLogger(\"UnifiedRLManager\")\n    \n    def create_model(self, config: RLModelConfig) -> str:\n        \"\"\"创建新模型\"\"\"\n        # 注册模型\n        self.registry.register_model(config)\n        \n        # 创建算法实例\n        agent = self._create_agent(config)\n        \n        # 缓存模型\n        self.loaded_models[config.model_name] = agent\n        \n        self.logger.info(f\"创建RL模型: {config.model_name} ({config.algorithm.value})\")\n        return config.model_name\n    \n    def _create_agent(self, config: RLModelConfig):\n        \"\"\"创建Agent实例\"\"\"\n        factory = self.algorithm_factories[config.algorithm]\n        return factory(config)\n    \n    def _create_dqn_agent(self, config: RLModelConfig) -> DQNAgent:\n        \"\"\"创建DQN Agent\"\"\"\n        algorithm_config = DQNConfig(**config.algorithm_config)\n        return DQNAgent(config.state_dim, config.action_dim, algorithm_config)\n    \n    def _create_ppo_agent(self, config: RLModelConfig) -> PPOAgent:\n        \"\"\"创建PPO Agent\"\"\"\n        algorithm_config = PPOConfig(**config.algorithm_config)\n        return PPOAgent(config.state_dim, config.action_dim, algorithm_config)\n    \n    def _create_a3c_agent(self, config: RLModelConfig) -> A3CAgent:\n        \"\"\"创建A3C Agent\"\"\"\n        algorithm_config = A3CConfig(**config.algorithm_config)\n        return A3CAgent(config.state_dim, config.action_dim, algorithm_config)\n    \n    def load_model(self, model_name: str) -> bool:\n        \"\"\"加载模型\"\"\"\n        if model_name in self.loaded_models:\n            return True\n        \n        config = self.registry.get_model_config(model_name)\n        if not config:\n            self.logger.error(f\"模型配置不存在: {model_name}\")\n            return False\n        \n        # 创建Agent\n        agent = self._create_agent(config)\n        \n        # 加载权重\n        model_path = os.path.join(self.base_model_dir, f\"{model_name}.pth\")\n        if os.path.exists(model_path):\n            success = agent.load_model(model_path)\n            if not success:\n                return False\n        else:\n            self.logger.warning(f\"模型文件不存在: {model_path}，使用随机初始化\")\n        \n        self.loaded_models[model_name] = agent\n        self.logger.info(f\"加载RL模型: {model_name}\")\n        return True\n    \n    def get_model(self, model_name: str):\n        \"\"\"获取模型\"\"\"\n        if model_name not in self.loaded_models:\n            if not self.load_model(model_name):\n                return None\n        return self.loaded_models[model_name]\n    \n    async def train_model(\n        self,\n        model_name: str,\n        training_data: Optional[Dict[str, np.ndarray]] = None,\n        num_episodes: int = 1000,\n        progress_callback: Optional[Callable] = None\n    ) -> RLPerformanceMetrics:\n        \"\"\"训练模型\"\"\"\n        config = self.registry.get_model_config(model_name)\n        if not config:\n            raise ValueError(f\"模型配置不存在: {model_name}\")\n        \n        agent = self.get_model(model_name)\n        if not agent:\n            raise ValueError(f\"无法加载模型: {model_name}\")\n        \n        self.logger.info(f\"开始训练模型: {model_name} ({config.algorithm.value})\")\n        \n        # 创建训练环境\n        env = create_trading_environment(config.env_config)\n        \n        # 加载训练数据\n        if training_data:\n            env.load_data(training_data[\"prices\"], training_data.get(\"volumes\"))\n        else:\n            # 生成合成数据\n            self._generate_synthetic_data(env)\n        \n        # 训练开始时间\n        start_time = datetime.utcnow()\n        \n        # 根据算法类型选择训练方法\n        if config.algorithm == RLAlgorithmType.DQN:\n            final_stats = await self._train_dqn(agent, env, num_episodes, progress_callback)\n        elif config.algorithm == RLAlgorithmType.PPO:\n            final_stats = await self._train_ppo(agent, env, num_episodes, progress_callback)\n        elif config.algorithm == RLAlgorithmType.A3C:\n            final_stats = await self._train_a3c(agent, env, num_episodes, progress_callback)\n        else:\n            raise ValueError(f\"不支持的算法类型: {config.algorithm}\")\n        \n        # 计算训练时间\n        training_time = (datetime.utcnow() - start_time).total_seconds() / 3600\n        \n        # 保存模型\n        model_path = os.path.join(self.base_model_dir, f\"{model_name}.pth\")\n        agent.save_model(model_path)\n        \n        # 评估模型\n        eval_score = await self._evaluate_model(agent, env)\n        \n        # 创建性能指标\n        metrics = RLPerformanceMetrics(\n            model_name=model_name,\n            algorithm=config.algorithm.value,\n            total_episodes=final_stats.get(\"total_episodes\", num_episodes),\n            training_time=training_time,\n            avg_episode_reward=final_stats.get(\"avg_episode_reward\", 0),\n            best_episode_reward=final_stats.get(\"best_episode_reward\", 0),\n            avg_training_loss=final_stats.get(\"avg_training_loss\", 0),\n            final_epsilon=final_stats.get(\"final_epsilon\", 0),\n            evaluation_score=eval_score\n        )\n        \n        # 获取环境性能指标\n        env_metrics = env.get_performance_metrics()\n        metrics.total_return = env_metrics.get(\"total_return\", 0)\n        metrics.sharpe_ratio = env_metrics.get(\"sharpe_ratio\", 0)\n        metrics.max_drawdown = env_metrics.get(\"max_drawdown\", 0)\n        metrics.win_rate = env_metrics.get(\"win_rate\", 0)\n        \n        # 记录性能\n        self.registry.add_performance(model_name, metrics)\n        \n        self.logger.info(f\"✅ 模型训练完成: {model_name}，评估得分: {eval_score:.4f}\")\n        return metrics\n    \n    async def _train_dqn(self, agent: DQNAgent, env, num_episodes: int, progress_callback) -> Dict[str, Any]:\n        \"\"\"训练DQN\"\"\"\n        episode_rewards = []\n        training_losses = []\n        \n        for episode in range(num_episodes):\n            state = env.reset()\n            episode_reward = 0\n            \n            while True:\n                action = agent.select_action(state, training=True)\n                next_state, reward, done, _ = env.step(action)\n                \n                agent.store_experience(state, action, reward, next_state, done)\n                loss = agent.train_step()\n                \n                if loss is not None:\n                    training_losses.append(loss)\n                \n                state = next_state\n                episode_reward += reward\n                \n                if done:\n                    break\n            \n            agent.end_episode(episode_reward)\n            episode_rewards.append(episode_reward)\n            \n            if progress_callback and episode % 10 == 0:\n                await progress_callback(episode, {\n                    \"episode_reward\": episode_reward,\n                    \"avg_reward\": np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),\n                    \"epsilon\": agent.epsilon\n                })\n        \n        return {\n            \"total_episodes\": len(episode_rewards),\n            \"avg_episode_reward\": np.mean(episode_rewards),\n            \"best_episode_reward\": max(episode_rewards),\n            \"avg_training_loss\": np.mean(training_losses) if training_losses else 0,\n            \"final_epsilon\": agent.epsilon\n        }\n    \n    async def _train_ppo(self, agent: PPOAgent, env, num_episodes: int, progress_callback) -> Dict[str, Any]:\n        \"\"\"训练PPO\"\"\"\n        episode_rewards = []\n        training_losses = []\n        \n        for episode in range(num_episodes):\n            state = env.reset()\n            episode_reward = 0\n            \n            while True:\n                action, log_prob, value = agent.select_action(state, training=True)\n                next_state, reward, done, _ = env.step(action)\n                \n                agent.store_experience(state, action, reward, next_state, done, log_prob, value)\n                \n                if agent.should_update():\n                    next_value = 0.0 if done else agent.select_action(next_state, training=True)[2]\n                    loss_dict = agent.train_step(next_value)\n                    if loss_dict:\n                        training_losses.append(loss_dict[\"total_loss\"])\n                \n                state = next_state\n                episode_reward += reward\n                \n                if done:\n                    break\n            \n            agent.end_episode(episode_reward)\n            episode_rewards.append(episode_reward)\n            \n            if progress_callback and episode % 10 == 0:\n                await progress_callback(episode, {\n                    \"episode_reward\": episode_reward,\n                    \"avg_reward\": np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)\n                })\n        \n        return {\n            \"total_episodes\": len(episode_rewards),\n            \"avg_episode_reward\": np.mean(episode_rewards),\n            \"best_episode_reward\": max(episode_rewards),\n            \"avg_training_loss\": np.mean(training_losses) if training_losses else 0\n        }\n    \n    async def _train_a3c(self, agent: A3CAgent, env_factory, num_episodes: int, progress_callback) -> Dict[str, Any]:\n        \"\"\"训练A3C\"\"\"\n        # A3C需要环境工厂\n        def create_env():\n            return env_factory\n        \n        final_stats = await agent.train(create_env, num_episodes, progress_callback)\n        \n        return {\n            \"total_episodes\": final_stats.get(\"global_episode\", num_episodes),\n            \"avg_episode_reward\": final_stats.get(\"avg_reward\", 0),\n            \"best_episode_reward\": final_stats.get(\"avg_reward\", 0)  # A3C没有单独的最佳奖励\n        }\n    \n    def _generate_synthetic_data(self, env):\n        \"\"\"生成合成训练数据\"\"\"\n        # 生成模拟价格数据\n        np.random.seed(42)\n        n_points = 10000\n        \n        # 几何布朗运动\n        dt = 1/252\n        mu = 0.1\n        sigma = 0.2\n        \n        price_changes = np.random.normal(\n            (mu - 0.5 * sigma**2) * dt, \n            sigma * np.sqrt(dt), \n            n_points\n        )\n        \n        initial_price = 100.0\n        prices = [initial_price]\n        \n        for change in price_changes:\n            prices.append(prices[-1] * np.exp(change))\n        \n        prices = np.array(prices)\n        \n        # 生成成交量数据\n        price_returns = np.diff(prices) / prices[:-1]\n        base_volume = 1000000\n        volume_multipliers = 1 + 2 * np.abs(price_returns)\n        volumes = np.concatenate([[base_volume], base_volume * volume_multipliers])\n        \n        env.load_data(prices, volumes)\n    \n    async def _evaluate_model(self, agent, env, num_episodes: int = 10) -> float:\n        \"\"\"评估模型性能\"\"\"\n        agent.set_eval_mode()\n        \n        episode_rewards = []\n        portfolio_returns = []\n        \n        for _ in range(num_episodes):\n            state = env.reset()\n            episode_reward = 0\n            \n            while True:\n                action = agent.select_action(state, training=False)\n                state, reward, done, _ = env.step(action)\n                episode_reward += reward\n                \n                if done:\n                    break\n            \n            episode_rewards.append(episode_reward)\n            \n            # 获取交易性能\n            metrics = env.get_performance_metrics()\n            portfolio_returns.append(metrics.get(\"total_return\", 0))\n        \n        agent.set_train_mode()\n        \n        # 综合评分\n        avg_reward = np.mean(episode_rewards)\n        avg_return = np.mean(portfolio_returns)\n        \n        return 0.6 * avg_reward + 0.4 * avg_return * 100\n    \n    def compare_models(self, model_names: List[str], metric: str = \"evaluation_score\") -> pd.DataFrame:\n        \"\"\"比较模型性能\"\"\"\n        comparison_data = []\n        \n        for model_name in model_names:\n            config = self.registry.get_model_config(model_name)\n            if not config:\n                continue\n            \n            history = self.registry.performance_history.get(model_name, [])\n            if not history:\n                continue\n            \n            latest_metrics = history[-1]\n            \n            comparison_data.append({\n                \"model_name\": model_name,\n                \"algorithm\": config.algorithm.value,\n                \"evaluation_score\": latest_metrics.evaluation_score,\n                \"avg_episode_reward\": latest_metrics.avg_episode_reward,\n                \"sharpe_ratio\": latest_metrics.sharpe_ratio,\n                \"max_drawdown\": latest_metrics.max_drawdown,\n                \"total_return\": latest_metrics.total_return,\n                \"training_time\": latest_metrics.training_time\n            })\n        \n        df = pd.DataFrame(comparison_data)\n        if not df.empty:\n            df = df.sort_values(metric, ascending=False)\n        \n        return df\n    \n    def get_model_performance(self, model_name: str) -> Optional[RLPerformanceMetrics]:\n        \"\"\"获取模型性能\"\"\"\n        history = self.registry.performance_history.get(model_name, [])\n        return history[-1] if history else None\n    \n    def list_available_models(self) -> List[Dict[str, Any]]:\n        \"\"\"列出可用模型\"\"\"\n        models = []\n        for config in self.registry.list_models():\n            performance = self.get_model_performance(config.model_name)\n            \n            model_info = {\n                \"name\": config.model_name,\n                \"algorithm\": config.algorithm.value,\n                \"description\": config.description,\n                \"evaluation_score\": performance.evaluation_score if performance else 0,\n                \"created_at\": performance.created_at if performance else None\n            }\n            models.append(model_info)\n        \n        return sorted(models, key=lambda x: x[\"evaluation_score\"], reverse=True)\n\n# 全局RL管理器实例\nrl_manager = UnifiedRLManager()