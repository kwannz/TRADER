"""
深度Q网络 (DQN) 强化学习Agent
实现DQN算法用于交易决策学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import json
import os

# 经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class DQNConfig:
    """DQN配置"""
    # 网络结构
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    activation: str = "relu"  # relu, tanh, leaky_relu
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.001  # 软更新参数
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 经验回放
    memory_size: int = 100000
    min_memory_size: int = 1000
    
    # 更新频率
    target_update_frequency: int = 100  # 目标网络更新频率
    train_frequency: int = 4  # 训练频率
    
    # 其他参数
    double_dqn: bool = True  # 是否使用Double DQN
    dueling_dqn: bool = True  # 是否使用Dueling DQN
    prioritized_replay: bool = False  # 优先经验回放
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]

class DQNNetwork(nn.Module):
    """DQN网络结构"""
    
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        super(DQNNetwork, self).__init__()
        
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
        
        if config.dueling_dqn:
            self._build_dueling_network()
        else:
            self._build_standard_network()
    
    def _build_standard_network(self):
        """构建标准DQN网络"""
        layers = []
        
        # 输入层
        prev_dim = self.state_dim
        
        # 隐藏层
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _build_dueling_network(self):
        """构建Dueling DQN网络"""
        # 特征提取层
        feature_layers = []
        prev_dim = self.state_dim
        
        for i, hidden_dim in enumerate(self.config.hidden_dims[:-1]):
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # 价值流和优势流
        final_hidden_dim = self.config.hidden_dims[-1]
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, final_hidden_dim),
            self.activation,
            nn.Linear(final_hidden_dim, 1)
        )
        
        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, final_hidden_dim),
            self.activation,
            nn.Linear(final_hidden_dim, self.action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.config.dueling_dqn:
            return self._forward_dueling(state)
        else:
            return self.network(state)
    
    def _forward_dueling(self, state: torch.Tensor) -> torch.Tensor:
        """Dueling网络前向传播"""
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values

class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, experience: Experience, priority: float = None):
        """添加经验"""
        if priority is None:
            priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """优先采样"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """DQN强化学习Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or DQNConfig()
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = DQNNetwork(state_dim, action_dim, self.config).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, self.config).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # 经验回放
        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.config.memory_size)
        else:
            self.memory = ExperienceReplayBuffer(self.config.memory_size)
        
        # 训练状态
        self.epsilon = self.config.epsilon_start
        self.steps_done = 0
        self.episode_count = 0
        
        # 性能指标
        self.training_losses = []
        self.episode_rewards = []
        self.q_values_history = []
        
        # 同步目标网络
        self._sync_target_network()
        
        self.logger = logging.getLogger("DQNAgent")
        
    def _sync_target_network(self):
        """同步目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.action_dim)
        
        # 贪婪策略
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # 记录Q值用于分析
            if training:
                self.q_values_history.append(q_values.mean().item())
                if len(self.q_values_history) > 1000:
                    self.q_values_history = self.q_values_history[-500:]
            
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done)
        
        if self.config.prioritized_replay:
            # 计算TD误差作为优先级
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                current_q = self.q_network(state_tensor)[0, action]
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward + (self.config.gamma * next_q * (not done))
                
                td_error = abs(current_q - target_q).item()
                self.memory.push(experience, td_error)
        else:
            self.memory.push(experience)
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.memory) < self.config.min_memory_size:
            return None
        
        if self.steps_done % self.config.train_frequency != 0:
            self.steps_done += 1
            return None
        
        # 采样经验
        if self.config.prioritized_replay:
            beta = min(1.0, 0.4 + 0.6 * (self.steps_done / 100000))  # 逐渐增加beta
            experiences, indices, weights = self.memory.sample(self.config.batch_size, beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
        
        # 准备批次数据
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
        # 计算损失
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # 更新优先级
        if self.config.prioritized_replay:
            new_priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % self.config.target_update_frequency == 0:
            self._sync_target_network()
        else:
            self._soft_update_target_network()
        
        # 更新epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        # 记录训练指标
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        if len(self.training_losses) > 1000:
            self.training_losses = self.training_losses[-500:]
        
        self.steps_done += 1
        
        return loss_value
    
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
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "avg_loss": np.mean(self.training_losses) if self.training_losses else 0,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_q_value": np.mean(self.q_values_history) if self.q_values_history else 0
        }
        
        if len(self.episode_rewards) >= 100:
            stats["avg_reward_100"] = np.mean(self.episode_rewards[-100:])
        
        return stats
    
    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "training_stats": self.get_training_stats(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episode_count": self.episode_count
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        self.logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            self.logger.error(f"模型文件不存在: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.epsilon = checkpoint.get("epsilon", self.config.epsilon_start)
            self.steps_done = checkpoint.get("steps_done", 0)
            self.episode_count = checkpoint.get("episode_count", 0)
            
            self.logger.info(f"模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False
    
    def set_eval_mode(self):
        """设置评估模式"""
        self.q_network.eval()
        self.target_network.eval()
    
    def set_train_mode(self):
        """设置训练模式"""
        self.q_network.train()
        self.target_network.train()
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # 使用softmax转换为概率
            probabilities = F.softmax(q_values / 0.1, dim=1)  # 温度参数0.1
            return probabilities.cpu().numpy()[0]
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取Q值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]

# 训练工具函数
def train_dqn_agent(
    agent: DQNAgent,
    env,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    eval_frequency: int = 100,
    save_frequency: int = 500,
    model_save_path: str = "models/dqn_trading_agent.pth"
) -> Dict[str, List]:
    """训练DQN Agent"""
    
    logger = logging.getLogger("DQNTrainer")
    training_history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "training_losses": [],
        "evaluation_scores": []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)
            
            # 训练
            loss = agent.train_step()
            if loss is not None:
                training_history["training_losses"].append(loss)
            
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
            eval_score = evaluate_agent(agent, env, num_episodes=5)
            training_history["evaluation_scores"].append(eval_score)
            
            logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                       f"Eval Score={eval_score:.2f}, Epsilon={agent.epsilon:.3f}")
        
        # 保存模型
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(model_save_path.replace(".pth", f"_ep{episode}.pth"))
    
    # 保存最终模型
    agent.save_model(model_save_path)
    
    return training_history

def evaluate_agent(agent: DQNAgent, env, num_episodes: int = 10) -> float:
    """评估Agent性能"""
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