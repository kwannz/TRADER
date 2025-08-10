"""
Asynchronous Advantage Actor-Critic (A3C) 强化学习Agent
实现A3C算法用于交易决策学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import threading
import queue
from collections import namedtuple, deque
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass
import json
import os
import time
import asyncio

# A3C经验元组
A3CExperience = namedtuple('A3CExperience', ['state', 'action', 'reward', 'next_state', 'done', 'value'])

@dataclass
class A3CConfig:
    """A3C配置"""
    # 网络结构
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "relu"  # relu, tanh, leaky_relu
    
    # A3C特定参数
    learning_rate: float = 1e-3
    gamma: float = 0.99  # 折扣因子
    entropy_coef: float = 0.01  # 熵损失系数
    value_loss_coef: float = 0.5  # 价值损失系数
    
    # 训练参数
    n_step: int = 20  # n-step TD
    max_grad_norm: float = 40.0  # 梯度裁剪
    
    # 异步参数
    num_workers: int = 4  # 工作进程数量
    update_frequency: int = 20  # 更新频率
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1000000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class A3CNetwork(nn.Module):
    """A3C Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, config: A3CConfig):
        super(A3CNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 激活函数
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        elif config.activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # 构建共享层
        self._build_shared_layers()
        
        # Actor头部 (策略网络)
        self._build_actor_head()
        
        # Critic头部 (价值网络)
        self._build_critic_head()
        
        # 初始化权重
        self._init_weights()
    
    def _build_shared_layers(self):
        """构建共享层"""
        layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in self.config.hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        self.shared_output_dim = prev_dim
    
    def _build_actor_head(self):
        """构建Actor头部"""
        final_hidden_dim = self.config.hidden_dims[-1]
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.shared_output_dim, final_hidden_dim),
            self.activation,
            nn.Linear(final_hidden_dim, self.action_dim)
        )
    
    def _build_critic_head(self):
        """构建Critic头部"""
        final_hidden_dim = self.config.hidden_dims[-1]
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.shared_output_dim, final_hidden_dim),
            self.activation,
            nn.Linear(final_hidden_dim, 1)
        )
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        shared_features = self.shared_layers(state)
        
        # Actor输出 (动作概率logits)
        action_logits = self.actor_head(shared_features)
        
        # Critic输出 (状态价值)
        state_value = self.critic_head(shared_features).squeeze(-1)
        
        return action_logits, state_value
    
    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """执行动作"""
        action_logits, value = self.forward(state)
        
        # 创建动作分布
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        # 采样动作
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估状态和动作"""
        action_logits, value = self.forward(state)
        
        # 创建动作分布
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        # 计算log概率和熵
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return log_prob, value, entropy

class GlobalNetwork:
    """全局网络 (参数服务器)"""
    
    def __init__(self, state_dim: int, action_dim: int, config: A3CConfig):
        self.network = A3CNetwork(state_dim, action_dim, config)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.lock = threading.Lock()
        
        # 全局统计
        self.global_step = 0
        self.global_episode = 0
        self.total_rewards = []
        
    def update(self, gradients: List[torch.Tensor]):
        """更新全局网络"""
        with self.lock:
            # 应用梯度
            for param, grad in zip(self.network.parameters(), gradients):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad
            
            # 执行优化步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局参数"""
        with self.lock:
            return {name: param.clone() for name, param in self.network.state_dict().items()}
    
    def add_reward(self, reward: float):
        """添加episode奖励"""
        with self.lock:
            self.total_rewards.append(reward)
            if len(self.total_rewards) > 1000:
                self.total_rewards = self.total_rewards[-500:]
            self.global_episode += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                "global_step": self.global_step,
                "global_episode": self.global_episode,
                "avg_reward": np.mean(self.total_rewards) if self.total_rewards else 0,
                "recent_avg_reward": np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards) if self.total_rewards else 0
            }

class A3CWorker:
    """A3C工作进程"""
    
    def __init__(self, worker_id: int, global_network: GlobalNetwork, 
                 env_factory: Callable, config: A3CConfig):
        self.worker_id = worker_id
        self.global_network = global_network
        self.env_factory = env_factory
        self.config = config
        
        # 本地网络
        self.local_network = A3CNetwork(
            global_network.network.state_dim,
            global_network.network.action_dim,
            config
        )
        
        # 经验缓冲
        self.experience_buffer = []
        
        # 统计
        self.local_step = 0
        self.local_episode = 0
        self.episode_rewards = []
        
        self.logger = logging.getLogger(f"A3CWorker-{worker_id}")
        
    def run(self, max_episodes: int = 1000):
        """运行工作进程"""
        env = self.env_factory()
        
        for episode in range(max_episodes):
            self._run_episode(env)
            
            if episode % 100 == 0:
                stats = self.global_network.get_stats()
                self.logger.info(f"Worker {self.worker_id}: Episode {episode}, "
                               f"Global Step: {stats['global_step']}, "
                               f"Avg Reward: {stats['avg_reward']:.2f}")
    
    def _run_episode(self, env):
        """运行一个episode"""
        # 同步本地网络与全局网络
        self._sync_with_global()
        
        state = env.reset()
        episode_reward = 0
        self.experience_buffer.clear()
        
        while True:
            # 选择动作
            action, log_prob, value = self._select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            experience = A3CExperience(state, action, reward, next_state, done, value)
            self.experience_buffer.append(experience)
            
            episode_reward += reward
            self.local_step += 1
            
            # 检查是否需要更新
            if len(self.experience_buffer) >= self.config.n_step or done:
                self._update_global_network(next_state, done)
                self.experience_buffer.clear()
            
            state = next_state
            
            if done:
                break
        
        # 记录episode奖励
        self.episode_rewards.append(episode_reward)
        self.global_network.add_reward(episode_reward)
        self.local_episode += 1
    
    def _select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.local_network.act(state_tensor)
            return action, log_prob, value
    
    def _sync_with_global(self):
        """与全局网络同步"""
        global_params = self.global_network.get_parameters()
        self.local_network.load_state_dict(global_params)
    
    def _update_global_network(self, next_state: np.ndarray, done: bool):
        """更新全局网络"""
        if not self.experience_buffer:
            return
        
        # 计算回报
        returns = self._compute_returns(next_state, done)
        
        # 计算梯度
        gradients = self._compute_gradients(returns)
        
        # 更新全局网络
        self.global_network.update(gradients)
    
    def _compute_returns(self, next_state: np.ndarray, done: bool) -> List[float]:
        """计算n-step回报"""
        returns = []
        
        # 计算bootstrap价值
        if done:
            R = 0.0
        else:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = self.local_network.forward(next_state_tensor)
                R = next_value.item()
        
        # 反向计算回报
        for experience in reversed(self.experience_buffer):
            R = experience.reward + self.config.gamma * R
            returns.append(R)
        
        returns.reverse()
        return returns
    
    def _compute_gradients(self, returns: List[float]) -> List[torch.Tensor]:
        """计算梯度"""
        # 准备数据
        states = torch.FloatTensor([exp.state for exp in self.experience_buffer])
        actions = torch.LongTensor([exp.action for exp in self.experience_buffer])
        returns_tensor = torch.FloatTensor(returns)
        values = torch.FloatTensor([exp.value for exp in self.experience_buffer])
        
        # 计算优势
        advantages = returns_tensor - values
        
        # 前向传播
        log_probs, current_values, entropy = self.local_network.evaluate(states, actions)
        
        # 计算损失
        # Actor损失 (策略梯度)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic损失 (价值函数)
        critic_loss = F.mse_loss(current_values, returns_tensor)
        
        # 熵损失 (鼓励探索)
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (actor_loss + 
                     self.config.value_loss_coef * critic_loss + 
                     self.config.entropy_coef * entropy_loss)
        
        # 计算梯度
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.config.max_grad_norm)
        
        # 获取梯度
        gradients = [param.grad.clone() for param in self.local_network.parameters()]
        
        # 清零本地梯度
        self.local_network.zero_grad()
        
        return gradients

class A3CAgent:
    """A3C主Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: A3CConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or A3CConfig()
        
        # 全局网络
        self.global_network = GlobalNetwork(state_dim, action_dim, self.config)
        
        # 工作进程池
        self.workers = []
        self.worker_threads = []
        
        # 训练状态
        self.is_training = False
        
        self.logger = logging.getLogger("A3CAgent")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作 (用于评估)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, _ = self.global_network.network.forward(state_tensor)
            
            if training:
                # 使用策略分布采样
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
            else:
                # 使用贪婪策略
                action = action_logits.argmax()
            
            return action.item()
    
    async def train(self, env_factory: Callable, num_episodes: int = 1000,
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """训练A3C"""
        try:
            self.logger.info("🚀 开始A3C训练...")
            self.is_training = True
            
            # 创建工作进程
            self._create_workers(env_factory, num_episodes)
            
            # 启动工作线程
            for worker in self.workers:
                thread = threading.Thread(target=worker.run, args=(num_episodes,))
                thread.start()
                self.worker_threads.append(thread)
            
            # 监控训练进程
            await self._monitor_training(progress_callback)
            
            # 等待所有工作线程完成
            for thread in self.worker_threads:
                thread.join()
            
            self.is_training = False
            
            # 计算最终统计
            final_stats = self.global_network.get_stats()
            
            self.logger.info("✅ A3C训练完成")
            return final_stats
            
        except Exception as e:
            self.logger.error(f"❌ A3C训练失败: {e}")
            self.is_training = False
            raise
    
    def _create_workers(self, env_factory: Callable, num_episodes: int):
        """创建工作进程"""
        self.workers = []
        for i in range(self.config.num_workers):
            worker = A3CWorker(i, self.global_network, env_factory, self.config)
            self.workers.append(worker)
        
        self.logger.info(f"创建了 {len(self.workers)} 个A3C工作进程")
    
    async def _monitor_training(self, progress_callback: Optional[Callable]):
        """监控训练进程"""
        last_episode = 0
        
        while self.is_training and any(thread.is_alive() for thread in self.worker_threads):
            await asyncio.sleep(1)  # 每秒检查一次
            
            stats = self.global_network.get_stats()
            current_episode = stats["global_episode"]
            
            # 调用进度回调
            if progress_callback and current_episode > last_episode:
                await progress_callback(current_episode, {
                    "episode_reward": stats["avg_reward"],
                    "recent_avg_reward": stats["recent_avg_reward"],
                    "global_step": stats["global_step"]
                })
            
            last_episode = current_episode
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return self.global_network.get_stats()
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            "network_state_dict": self.global_network.network.state_dict(),
            "optimizer_state_dict": self.global_network.optimizer.state_dict(),
            "config": self.config.__dict__,
            "training_stats": self.get_training_stats()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f"A3C模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            self.logger.error(f"模型文件不存在: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location="cpu")
            
            self.global_network.network.load_state_dict(checkpoint["network_state_dict"])
            self.global_network.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.logger.info(f"A3C模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载A3C模型失败: {e}")
            return False
    
    def set_eval_mode(self):
        """设置评估模式"""
        self.global_network.network.eval()
    
    def set_train_mode(self):
        """设置训练模式"""
        self.global_network.network.train()
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, _ = self.global_network.network.forward(state_tensor)
            probabilities = F.softmax(action_logits, dim=-1)
            return probabilities.numpy()[0]
    
    def get_state_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, value = self.global_network.network.forward(state_tensor)
            return value.item()

# 训练工具函数
def train_a3c_agent(
    agent: A3CAgent,
    env_factory: Callable,
    num_episodes: int = 1000,
    eval_frequency: int = 100,
    save_frequency: int = 500,
    model_save_path: str = "models/a3c_trading_agent.pth"
) -> Dict[str, List]:
    """训练A3C Agent"""
    
    logger = logging.getLogger("A3CTrainer")
    training_history = {
        "episode_rewards": [],
        "global_steps": [],
        "evaluation_scores": []
    }
    
    async def progress_callback(episode: int, metrics: Dict[str, Any]):
        training_history["episode_rewards"].append(metrics["episode_reward"])
        training_history["global_steps"].append(metrics["global_step"])
        
        # 评估
        if episode % eval_frequency == 0:
            eval_score = evaluate_a3c_agent(agent, env_factory(), num_episodes=5)
            training_history["evaluation_scores"].append(eval_score)
            
            logger.info(f"Episode {episode}: Avg Reward={metrics['episode_reward']:.2f}, "
                       f"Eval Score={eval_score:.2f}, Global Step={metrics['global_step']}")
        
        # 保存模型
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(model_save_path.replace(".pth", f"_ep{episode}.pth"))
    
    # 运行训练
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        final_stats = loop.run_until_complete(
            agent.train(env_factory, num_episodes, progress_callback)
        )
    finally:
        loop.close()
    
    # 保存最终模型
    agent.save_model(model_save_path)
    
    return training_history

def evaluate_a3c_agent(agent: A3CAgent, env, num_episodes: int = 10) -> float:
    """评估A3C Agent性能"""
    agent.set_eval_mode()
    
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    agent.set_train_mode()
    
    return np.mean(total_rewards)