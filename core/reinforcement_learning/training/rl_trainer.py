"""
强化学习训练器
提供统一的强化学习模型训练接口和管理
"""

import asyncio
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from ..environments.trading_env import TradingEnvironment, create_trading_environment
from ..algorithms.dqn_agent import DQNAgent, DQNConfig

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # 环境配置
    env_config: Dict[str, Any] = None
    
    # 模型配置
    model_config: Dict[str, Any] = None
    
    # 数据配置
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    
    # 保存路径
    model_save_dir: str = "models/rl_models"
    log_save_dir: str = "logs/rl_training"
    
    # 早停参数
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
    
    # 学习率调度
    lr_scheduler: bool = True
    lr_step_size: int = 200
    lr_gamma: float = 0.9
    
    def __post_init__(self):
        if self.env_config is None:
            self.env_config = {}
        if self.model_config is None:
            self.model_config = {}

@dataclass
class TrainingMetrics:
    """训练指标"""
    episode: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    training_loss: float = 0.0
    epsilon: float = 1.0
    eval_score: float = 0.0
    portfolio_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 50, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.patience_counter = 0
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("RLTrainer")
        
        # 创建保存目录
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        os.makedirs(self.config.log_save_dir, exist_ok=True)
        
        # 训练历史
        self.training_history: List[TrainingMetrics] = []
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta
        )
        
        # 模型和环境
        self.agent: Optional[DQNAgent] = None
        self.env: Optional[TradingEnvironment] = None
        self.eval_env: Optional[TradingEnvironment] = None
        
        # 训练状态
        self.is_training = False
        self.should_stop = False
        
    async def setup_training(self):
        """设置训练环境"""
        try:
            self.logger.info("设置训练环境...")
            
            # 创建训练环境
            self.env = create_trading_environment(self.config.env_config)
            self.eval_env = create_trading_environment(self.config.env_config)
            
            # 加载训练数据
            await self._load_training_data()
            
            # 创建Agent
            self._create_agent()
            
            self.logger.info("✅ 训练环境设置完成")
            
        except Exception as e:
            self.logger.error(f"❌ 设置训练环境失败: {e}")
            raise
    
    async def _load_training_data(self):
        """加载训练数据"""
        if self.config.train_data_path and os.path.exists(self.config.train_data_path):
            # 从文件加载
            data = pd.read_csv(self.config.train_data_path)
            prices = data['close'].values
            volumes = data.get('volume', pd.Series([1000000] * len(data))).values
            
            self.env.load_data(prices, volumes)
            self.eval_env.load_data(prices, volumes)
            
            self.logger.info(f"从文件加载数据: {len(prices)} 个数据点")
        else:
            # 生成模拟数据
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """生成合成训练数据"""
        self.logger.info("生成合成训练数据...")
        
        # 生成模拟价格数据
        np.random.seed(42)
        n_points = 10000
        
        # 几何布朗运动
        dt = 1/252  # 日频率
        mu = 0.1  # 年化收益率
        sigma = 0.2  # 年化波动率
        
        price_changes = np.random.normal(
            (mu - 0.5 * sigma**2) * dt, 
            sigma * np.sqrt(dt), 
            n_points
        )
        
        # 生成价格序列
        initial_price = 100.0
        prices = [initial_price]
        
        for change in price_changes:
            prices.append(prices[-1] * np.exp(change))
        
        prices = np.array(prices)
        
        # 生成成交量数据（与价格变化相关）
        price_returns = np.diff(prices) / prices[:-1]
        base_volume = 1000000
        volume_multipliers = 1 + 2 * np.abs(price_returns)  # 价格变化大时成交量大
        volumes = np.concatenate([[base_volume], base_volume * volume_multipliers])
        
        # 加载到环境
        self.env.load_data(prices, volumes)
        self.eval_env.load_data(prices, volumes)
        
        self.logger.info(f"生成合成数据: {len(prices)} 个数据点")
    
    def _create_agent(self):
        """创建强化学习Agent"""
        # 获取状态和动作空间维度
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # 创建DQN配置
        dqn_config = DQNConfig(**self.config.model_config)
        
        # 创建Agent
        self.agent = DQNAgent(state_dim, action_dim, dqn_config)
        
        self.logger.info(f"创建DQN Agent: 状态维度={state_dim}, 动作维度={action_dim}")
    
    async def train(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """开始训练"""
        try:
            self.logger.info("🚀 开始强化学习训练...")
            
            if not self.agent or not self.env:
                await self.setup_training()
            
            self.is_training = True
            self.should_stop = False
            
            # 学习率调度器
            if self.config.lr_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.agent.optimizer,
                    step_size=self.config.lr_step_size,
                    gamma=self.config.lr_gamma
                )
            
            best_eval_score = float('-inf')
            
            for episode in range(self.config.num_episodes):
                if self.should_stop:
                    self.logger.info("收到停止信号，终止训练")
                    break
                
                # 训练一个episode
                metrics = await self._train_episode(episode)
                self.training_history.append(metrics)
                
                # 评估
                if episode % self.config.eval_frequency == 0:
                    eval_score = await self._evaluate_agent()
                    metrics.eval_score = eval_score
                    
                    # 保存最佳模型
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        best_model_path = os.path.join(self.config.model_save_dir, "best_model.pth")
                        self.agent.save_model(best_model_path)
                    
                    # 早停检查
                    if self.early_stopping(eval_score):
                        self.logger.info(f"早停触发，在episode {episode}停止训练")
                        break
                    
                    self.logger.info(
                        f"Episode {episode}: Reward={metrics.episode_reward:.2f}, "
                        f"Eval={eval_score:.2f}, Epsilon={metrics.epsilon:.3f}"
                    )
                
                # 保存检查点
                if episode % self.config.save_frequency == 0 and episode > 0:
                    checkpoint_path = os.path.join(
                        self.config.model_save_dir, 
                        f"checkpoint_ep{episode}.pth"
                    )
                    self.agent.save_model(checkpoint_path)
                
                # 学习率调度
                if self.config.lr_scheduler:
                    scheduler.step()
                
                # 进度回调
                if progress_callback:
                    await progress_callback(episode, metrics)
            
            # 保存最终模型
            final_model_path = os.path.join(self.config.model_save_dir, "final_model.pth")
            self.agent.save_model(final_model_path)
            
            # 保存训练历史
            await self._save_training_history()
            
            self.is_training = False
            
            # 计算最终统计
            final_stats = self._calculate_training_stats()
            
            self.logger.info("✅ 训练完成")
            return final_stats
            
        except Exception as e:
            self.logger.error(f"❌ 训练失败: {e}")
            self.is_training = False
            raise
    
    async def _train_episode(self, episode: int) -> TrainingMetrics:
        """训练一个episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        total_loss = 0
        loss_count = 0
        
        for step in range(self.config.max_steps_per_episode):
            # 选择动作
            action = self.agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # 训练
            loss = self.agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 结束episode
        self.agent.end_episode(episode_reward)
        
        # 获取环境性能指标
        env_metrics = self.env.get_performance_metrics()
        
        # 创建训练指标
        metrics = TrainingMetrics(
            episode=episode,
            episode_reward=episode_reward,
            episode_length=episode_length,
            training_loss=total_loss / max(loss_count, 1),
            epsilon=self.agent.epsilon,
            portfolio_return=env_metrics.get('total_return', 0),
            sharpe_ratio=env_metrics.get('sharpe_ratio', 0),
            max_drawdown=env_metrics.get('max_drawdown', 0)
        )
        
        return metrics
    
    async def _evaluate_agent(self, num_episodes: int = 5) -> float:
        """评估Agent性能"""
        if not self.agent or not self.eval_env:
            return 0.0
        
        self.agent.set_eval_mode()
        
        total_rewards = []
        portfolio_returns = []
        
        for _ in range(num_episodes):
            state = self.eval_env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                state, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            
            # 获取投资组合表现
            metrics = self.eval_env.get_performance_metrics()
            portfolio_returns.append(metrics.get('total_return', 0))
        
        self.agent.set_train_mode()
        
        # 综合评分：平均奖励和投资组合回报的加权平均
        avg_reward = np.mean(total_rewards)
        avg_return = np.mean(portfolio_returns)
        
        # 综合评分 = 50% 奖励 + 50% 投资组合回报
        composite_score = 0.5 * avg_reward + 0.5 * avg_return * 100  # 放大回报的权重
        
        return composite_score
    
    def _calculate_training_stats(self) -> Dict[str, Any]:
        """计算训练统计"""
        if not self.training_history:
            return {}
        
        rewards = [m.episode_reward for m in self.training_history]
        losses = [m.training_loss for m in self.training_history if m.training_loss > 0]
        returns = [m.portfolio_return for m in self.training_history]
        sharpe_ratios = [m.sharpe_ratio for m in self.training_history if m.sharpe_ratio != 0]
        
        stats = {
            "total_episodes": len(self.training_history),
            "avg_episode_reward": np.mean(rewards),
            "avg_training_loss": np.mean(losses) if losses else 0,
            "avg_portfolio_return": np.mean(returns),
            "final_epsilon": self.training_history[-1].epsilon,
            "best_episode_reward": max(rewards),
            "best_portfolio_return": max(returns),
            "avg_sharpe_ratio": np.mean(sharpe_ratios) if sharpe_ratios else 0,
            "training_duration": (
                self.training_history[-1].timestamp - self.training_history[0].timestamp
            ).total_seconds() / 3600  # 小时
        }
        
        # 最近100个episode的统计
        if len(rewards) >= 100:
            recent_rewards = rewards[-100:]
            recent_returns = returns[-100:]
            
            stats.update({
                "recent_avg_reward": np.mean(recent_rewards),
                "recent_avg_return": np.mean(recent_returns),
                "recent_best_reward": max(recent_rewards),
                "recent_best_return": max(recent_returns)
            })
        
        return stats
    
    async def _save_training_history(self):
        """保存训练历史"""
        try:
            # 保存为JSON
            history_data = [asdict(metrics) for metrics in self.training_history]
            
            # 转换datetime为字符串
            for entry in history_data:
                entry['timestamp'] = entry['timestamp'].isoformat()
            
            history_path = os.path.join(self.config.log_save_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # 保存为CSV
            df = pd.DataFrame(history_data)
            csv_path = os.path.join(self.config.log_save_dir, "training_history.csv")
            df.to_csv(csv_path, index=False)
            
            # 生成训练图表
            await self._generate_training_plots()
            
            self.logger.info(f"训练历史已保存到 {history_path}")
            
        except Exception as e:
            self.logger.error(f"保存训练历史失败: {e}")
    
    async def _generate_training_plots(self):
        """生成训练图表"""
        try:
            if len(self.training_history) < 2:
                return
            
            # 准备数据
            episodes = [m.episode for m in self.training_history]
            rewards = [m.episode_reward for m in self.training_history]
            losses = [m.training_loss for m in self.training_history if m.training_loss > 0]
            returns = [m.portfolio_return for m in self.training_history]
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RL Training Progress', fontsize=16)
            
            # 奖励曲线
            axes[0, 0].plot(episodes, rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # 损失曲线
            if losses:
                loss_episodes = [m.episode for m in self.training_history if m.training_loss > 0]
                axes[0, 1].plot(loss_episodes, losses)
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True)
            
            # 投资组合回报
            axes[1, 0].plot(episodes, returns)
            axes[1, 0].set_title('Portfolio Returns')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].grid(True)
            
            # 移动平均奖励
            window_size = min(50, len(rewards) // 4)
            if window_size > 1:
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                axes[1, 1].plot(episodes, rewards, alpha=0.3, label='Raw Rewards')
                axes[1, 1].plot(episodes, moving_avg, label=f'{window_size}-Episode MA')
                axes[1, 1].set_title('Reward Moving Average')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join(self.config.log_save_dir, "training_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"训练图表已保存到 {plot_path}")
            
        except Exception as e:
            self.logger.error(f"生成训练图表失败: {e}")
    
    async def stop_training(self):
        """停止训练"""
        self.should_stop = True
        self.logger.info("发送停止训练信号")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        if not self.training_history:
            return {"progress": 0, "status": "not_started"}
        
        current_episode = len(self.training_history)
        progress_pct = (current_episode / self.config.num_episodes) * 100
        
        latest_metrics = self.training_history[-1]
        
        return {
            "progress": min(100, progress_pct),
            "current_episode": current_episode,
            "total_episodes": self.config.num_episodes,
            "status": "training" if self.is_training else "completed",
            "latest_reward": latest_metrics.episode_reward,
            "latest_return": latest_metrics.portfolio_return,
            "epsilon": latest_metrics.epsilon,
            "best_reward": max(m.episode_reward for m in self.training_history),
            "avg_recent_reward": np.mean([m.episode_reward for m in self.training_history[-10:]]) if len(self.training_history) >= 10 else latest_metrics.episode_reward
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            if not self.agent:
                self.logger.error("Agent未初始化，无法加载检查点")
                return False
            
            success = self.agent.load_model(checkpoint_path)
            if success:
                self.logger.info(f"检查点已加载: {checkpoint_path}")
            return success
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False

# 训练任务管理
class TrainingTaskManager:
    """训练任务管理器"""
    
    def __init__(self):
        self.active_trainers: Dict[str, RLTrainer] = {}
        self.logger = logging.getLogger("TrainingTaskManager")
    
    async def start_training_task(
        self, 
        task_id: str, 
        config: TrainingConfig,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """启动训练任务"""
        if task_id in self.active_trainers:
            self.logger.warning(f"训练任务 {task_id} 已存在")
            return task_id
        
        trainer = RLTrainer(config)
        self.active_trainers[task_id] = trainer
        
        # 异步执行训练
        asyncio.create_task(self._run_training_task(task_id, trainer, progress_callback))
        
        self.logger.info(f"启动训练任务: {task_id}")
        return task_id
    
    async def _run_training_task(
        self, 
        task_id: str, 
        trainer: RLTrainer, 
        progress_callback: Optional[Callable] = None
    ):
        """运行训练任务"""
        try:
            await trainer.train(progress_callback)
            self.logger.info(f"训练任务完成: {task_id}")
        except Exception as e:
            self.logger.error(f"训练任务失败 {task_id}: {e}")
        finally:
            # 任务完成后清理
            self.active_trainers.pop(task_id, None)
    
    async def stop_training_task(self, task_id: str) -> bool:
        """停止训练任务"""
        if task_id not in self.active_trainers:
            return False
        
        trainer = self.active_trainers[task_id]
        await trainer.stop_training()
        
        self.logger.info(f"停止训练任务: {task_id}")
        return True
    
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务进度"""
        if task_id not in self.active_trainers:
            return None
        
        trainer = self.active_trainers[task_id]
        return trainer.get_training_progress()
    
    def list_active_tasks(self) -> List[str]:
        """列出活跃任务"""
        return list(self.active_trainers.keys())

# 全局训练任务管理器
training_task_manager = TrainingTaskManager()