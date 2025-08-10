"""
Proximal Policy Optimization (PPO) 强化学习Agent
实现PPO算法用于交易决策学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import namedtuple
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
import os

# PPO经验存储
PPOExperience = namedtuple('PPOExperience', 
                         ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

@dataclass
class PPOConfig:
    """PPO配置"""
    # 网络结构
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "relu"  # relu, tanh, leaky_relu
    
    # PPO特定参数
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2  # PPO裁剪参数
    value_loss_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵损失系数
    
    # 训练参数
    batch_size: int = 64
    n_epochs: int = 10  # 每次更新的epoch数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE lambda参数
    
    # 经验收集
    rollout_length: int = 2048  # 每次收集的经验长度
    
    # 其他参数
    max_grad_norm: float = 0.5  # 梯度裁剪
    target_kl: float = 0.01  # 目标KL散度
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class PPOActorCritic(nn.Module):
    """PPO Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        super(PPOActorCritic, self).__init__()
        
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
        
        # 构建共享特征层
        self._build_shared_layers()
        
        # Actor网络 (策略网络)
        self._build_actor_head()
        
        # Critic网络 (价值网络)
        self._build_critic_head()
    
    def _build_shared_layers(self):
        """构建共享特征提取层"""
        layers = []
        prev_dim = self.state_dim
        
        # 共享的隐藏层
        for i, hidden_dim in enumerate(self.config.hidden_dims[:-1]):
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
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(final_hidden_dim, self.action_dim)
        )
    
    def _build_critic_head(self):
        """构建Critic头部"""
        final_hidden_dim = self.config.hidden_dims[-1]
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.shared_output_dim, final_hidden_dim),
            self.activation,
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(final_hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        shared_features = self.shared_layers(state)
        
        # Actor输出 (动作概率logits)
        action_logits = self.actor_head(shared_features)
        
        # Critic输出 (状态价值)
        state_value = self.critic_head(shared_features).squeeze(-1)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """获取动作和价值"""
        action_logits, value = self.forward(state)
        
        # 创建动作分布
        action_dist = Categorical(logits=action_logits)
        
        # 采样动作
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        action_logits, values = self.forward(states)
        
        # 创建动作分布
        action_dist = Categorical(logits=action_logits)
        
        # 计算log概率和熵
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values, entropy

class PPORolloutBuffer:
    """PPO经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        
        # 存储数组
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        
        # GAE相关
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """存储经验"""
        idx = self.ptr
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def compute_gae(self, next_value: float, gamma: float, gae_lambda: float):
        """计算GAE (Generalized Advantage Estimation)"""
        # 添加下一个状态的价值
        values = np.append(self.values[:self.size], next_value)
        
        # 计算TD残差
        deltas = self.rewards[:self.size] + gamma * values[1:] * (1 - self.dones[:self.size]) - values[:-1]
        
        # 计算GAE
        advantages = np.zeros(self.size, dtype=np.float32)
        advantage = 0
        
        for t in reversed(range(self.size)):
            advantage = deltas[t] + gamma * gae_lambda * (1 - self.dones[t]) * advantage
            advantages[t] = advantage
        
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = advantages + values[:-1]
        
        # 归一化优势
        self.advantages[:self.size] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """获取批次数据"""
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

class PPOAgent:
    """PPO强化学习Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.actor_critic = PPOActorCritic(state_dim, action_dim, self.config).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.learning_rate)
        
        # 经验缓冲区
        self.rollout_buffer = PPORolloutBuffer(self.config.rollout_length, state_dim)
        
        # 训练状态
        self.steps_done = 0
        self.episode_count = 0
        self.rollout_step = 0
        
        # 性能指标
        self.training_losses = []
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        
        self.logger = logging.getLogger("PPOAgent")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, Optional[float], Optional[float]]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob, value = self.actor_critic.get_action_and_value(state_tensor)
                return action, log_prob.item(), value.item()
            else:
                action_logits, value = self.actor_critic(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action = action_probs.argmax().item()
                return action, None, value.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """存储经验"""
        self.rollout_buffer.store(state, action, reward, next_state, done, log_prob, value)
        self.rollout_step += 1
    
    def should_update(self) -> bool:
        """检查是否应该更新"""
        return self.rollout_step >= self.config.rollout_length
    
    def train_step(self, next_value: float = 0.0) -> Optional[Dict[str, float]]:
        """训练一步"""
        if not self.should_update():
            return None
        
        # 计算GAE
        self.rollout_buffer.compute_gae(next_value, self.config.gamma, self.config.gae_lambda)
        
        # 训练多个epoch
        total_losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        
        for epoch in range(self.config.n_epochs):
            # 获取批次数据
            batch = self.rollout_buffer.get_batch(self.config.batch_size)
            
            # 计算损失
            loss_dict = self._compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # 记录指标
            total_losses.append(loss_dict['total_loss'].item())
            policy_losses.append(loss_dict['policy_loss'].item())
            value_losses.append(loss_dict['value_loss'].item())
            entropy_losses.append(loss_dict['entropy_loss'].item())
            kl_divs.append(loss_dict['kl_div'].item())
            
            # 早停检查 (KL散度过大)
            if loss_dict['kl_div'].item() > self.config.target_kl * 1.5:
                self.logger.warning(f"Early stopping at epoch {epoch} due to high KL divergence: {loss_dict['kl_div'].item():.4f}")
                break
        
        # 清空缓冲区
        self.rollout_buffer.clear()
        self.rollout_step = 0
        
        # 记录训练指标
        avg_total_loss = np.mean(total_losses)
        self.training_losses.append(avg_total_loss)
        self.policy_losses.append(np.mean(policy_losses))
        self.value_losses.append(np.mean(value_losses))
        self.entropy_losses.append(np.mean(entropy_losses))
        
        # 保持历史记录不过长
        if len(self.training_losses) > 1000:
            self.training_losses = self.training_losses[-500:]
            self.policy_losses = self.policy_losses[-500:]
            self.value_losses = self.value_losses[-500:]
            self.entropy_losses = self.entropy_losses[-500:]
        
        self.steps_done += 1
        
        return {
            'total_loss': avg_total_loss,
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_div': np.mean(kl_divs),
            'epochs_trained': len(total_losses)
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算PPO损失"""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        
        # 前向传播
        new_log_probs, values, entropy = self.actor_critic.evaluate_actions(states, actions)
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 熵损失
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss)
        
        # KL散度 (用于监控)
        kl_div = (old_log_probs - new_log_probs).mean()
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'kl_div': kl_div
        }
    
    def end_episode(self, episode_reward: float):
        """结束一个episode"""
        self.episode_rewards.append(episode_reward)
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-500:]
        
        self.episode_count += 1
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        stats = {
            "episode_count": self.episode_count,
            "steps_done": self.steps_done,
            "rollout_step": self.rollout_step,
            "avg_total_loss": np.mean(self.training_losses) if self.training_losses else 0,
            "avg_policy_loss": np.mean(self.policy_losses) if self.policy_losses else 0,
            "avg_value_loss": np.mean(self.value_losses) if self.value_losses else 0,
            "avg_entropy_loss": np.mean(self.entropy_losses) if self.entropy_losses else 0,
            "avg_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0
        }
        
        if len(self.episode_rewards) >= 100:
            stats["avg_reward_100"] = np.mean(self.episode_rewards[-100:])
        
        return stats
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "training_stats": self.get_training_stats(),
            "steps_done": self.steps_done,
            "episode_count": self.episode_count
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f"PPO模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            self.logger.error(f"模型文件不存在: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.steps_done = checkpoint.get("steps_done", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            
            self.logger.info(f"PPO模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载PPO模型失败: {e}")
            return False
    
    def set_eval_mode(self):
        """设置评估模式"""
        self.actor_critic.eval()
    
    def set_train_mode(self):
        """设置训练模式"""
        self.actor_critic.train()
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, _ = self.actor_critic(state_tensor)
            probabilities = F.softmax(action_logits, dim=-1)
            return probabilities.cpu().numpy()[0]
    
    def get_state_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.actor_critic(state_tensor)
            return value.item()

# 训练工具函数
def train_ppo_agent(
    agent: PPOAgent,
    env,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    eval_frequency: int = 100,
    save_frequency: int = 500,
    model_save_path: str = "models/ppo_trading_agent.pth"
) -> Dict[str, List]:
    """训练PPO Agent"""
    
    logger = logging.getLogger("PPOTrainer")
    training_history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "training_losses": [],
        "evaluation_scores": [],
        "policy_losses": [],
        "value_losses": []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action, log_prob, value = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done, log_prob, value)
            
            # 检查是否需要更新
            if agent.should_update():
                # 获取下一个状态的价值 (用于GAE计算)
                if not done:
                    _, _, next_value = agent.select_action(next_state, training=True)
                else:
                    next_value = 0.0
                
                # 训练
                loss_dict = agent.train_step(next_value)
                if loss_dict:
                    training_history["training_losses"].append(loss_dict["total_loss"])
                    training_history["policy_losses"].append(loss_dict["policy_loss"])
                    training_history["value_losses"].append(loss_dict["value_loss"])
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 结束episode
        agent.end_episode(episode_reward)
        training_history["episode_rewards"].append(episode_reward)
        training_history["episode_lengths"].append(episode_length)
        
        # 评估
        if episode % eval_frequency == 0:
            eval_score = evaluate_ppo_agent(agent, env, num_episodes=5)
            training_history["evaluation_scores"].append(eval_score)
            
            logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Eval Score={eval_score:.2f}")
        
        # 保存模型
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(model_save_path.replace(".pth", f"_ep{episode}.pth"))
    
    # 保存最终模型
    agent.save_model(model_save_path)
    
    return training_history

def evaluate_ppo_agent(agent: PPOAgent, env, num_episodes: int = 10) -> float:
    """评估PPO Agent性能"""
    agent.set_eval_mode()
    
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    agent.set_train_mode()
    
    return np.mean(total_rewards)