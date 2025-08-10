"""
CTBench时间序列生成模型集群
实现多种先进的时间序列生成模型，用于创建高质量的合成金融数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

class GenerativeModelType(Enum):
    """生成模型类型"""
    TRANSFORMER_GAN = "transformer_gan"
    LSTM_VAE = "lstm_vae"
    DIFFUSION = "diffusion"
    FLOW_BASED = "flow_based"
    WGAN_GP = "wgan_gp"

@dataclass
class CTBenchConfig:
    """CTBench配置"""
    model_type: GenerativeModelType
    sequence_length: int = 100
    feature_dim: int = 5  # OHLCV
    latent_dim: int = 128
    
    # 模型特定参数
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 1000
    
    # 生成参数
    condition_on_market_regime: bool = True
    diversity_weight: float = 0.1
    
    # 其他参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "models/ctbench_checkpoints"
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

class BaseGenerativeModel(ABC, nn.Module):
    """基础生成模型"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 训练历史
        self.training_history = []
        self.generation_cache = {}
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        pass
    
    @abstractmethod
    def generate_samples(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成样本"""
        pass
    
    @abstractmethod
    def compute_loss(self, batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """计算损失"""
        pass
    
    def save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'loss': loss,
            'training_history': self.training_history
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"{self.__class__.__name__}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            self.logger.info(f"加载检查点: {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False

class TransformerGenerator(nn.Module):
    """基于Transformer的生成器"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__()
        self.config = config
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, config.sequence_length, config.hidden_dim))
        
        # 输入投影
        self.input_projection = nn.Linear(config.feature_dim + config.latent_dim, config.hidden_dim)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(config.hidden_dim, config.feature_dim)
        
    def forward(self, noise: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            noise: (batch_size, sequence_length, latent_dim)
            condition: (batch_size, sequence_length, feature_dim) 可选的条件输入
        """
        batch_size, seq_len = noise.shape[:2]
        
        if condition is not None:
            # 连接噪声和条件
            x = torch.cat([noise, condition], dim=-1)
        else:
            # 只使用噪声，填充零条件
            zero_condition = torch.zeros(batch_size, seq_len, self.config.feature_dim, device=noise.device)
            x = torch.cat([noise, zero_condition], dim=-1)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output

class TransformerDiscriminator(nn.Module):
    """基于Transformer的判别器"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, config.sequence_length, config.hidden_dim))
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output

class TransformerGAN(BaseGenerativeModel):
    """基于Transformer的GAN模型"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__(config)
        
        # 生成器和判别器
        self.generator = TransformerGenerator(config)
        self.discriminator = TransformerDiscriminator(config)
        
        # 优化器
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        
        # 损失函数
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # 移动到设备
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播 (仅生成器)"""
        return self.generator(x, **kwargs)
    
    def generate_samples(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成样本"""
        self.eval()
        with torch.no_grad():
            # 生成噪声
            noise = torch.randn(num_samples, self.config.sequence_length, self.config.latent_dim, device=self.device)
            
            # 生成样本
            samples = self.generator(noise, condition)
            
            return samples.cpu().numpy()
    
    def compute_loss(self, batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """计算损失"""
        real_data = batch.to(self.device)
        batch_size = real_data.size(0)
        
        # 生成假数据
        noise = torch.randn(batch_size, self.config.sequence_length, self.config.latent_dim, device=self.device)
        fake_data = self.generator(noise)
        
        # 判别器损失
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())
        
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        disc_loss_real = self.adversarial_loss(real_pred, real_labels)
        disc_loss_fake = self.adversarial_loss(fake_pred, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        
        # 生成器损失
        fake_pred_gen = self.discriminator(fake_data)
        gen_loss = self.adversarial_loss(fake_pred_gen, real_labels)
        
        return {
            "discriminator_loss": disc_loss,
            "generator_loss": gen_loss,
            "total_loss": disc_loss + gen_loss
        }
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """训练一步"""
        losses = self.compute_loss(batch)
        
        # 训练判别器
        self.disc_optimizer.zero_grad()
        losses["discriminator_loss"].backward(retain_graph=True)
        self.disc_optimizer.step()
        
        # 训练生成器
        self.gen_optimizer.zero_grad()
        losses["generator_loss"].backward()
        self.gen_optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            config.feature_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_layers > 1 else 0
        )
        
        # 均值和方差投影
        self.mu_layer = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_layer = nn.Linear(config.hidden_dim, config.latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码"""
        # LSTM编码
        output, (hidden, _) = self.lstm(x)
        
        # 使用最后时间步的隐藏状态
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # 计算均值和对数方差
        mu = self.mu_layer(last_hidden)
        logvar = self.logvar_layer(last_hidden)
        
        return mu, logvar

class LSTMDecoder(nn.Module):
    """LSTM解码器"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__()
        self.config = config
        
        # 从潜在空间到初始隐藏状态
        self.latent_to_hidden = nn.Linear(config.latent_dim, config.hidden_dim * config.num_layers)
        self.latent_to_cell = nn.Linear(config.latent_dim, config.hidden_dim * config.num_layers)
        
        self.lstm = nn.LSTM(
            config.latent_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(config.hidden_dim, config.feature_dim)
        
    def forward(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """解码"""
        batch_size = z.size(0)
        
        # 初始化隐藏状态
        hidden = self.latent_to_hidden(z).view(self.config.num_layers, batch_size, self.config.hidden_dim)
        cell = self.latent_to_cell(z).view(self.config.num_layers, batch_size, self.config.hidden_dim)
        
        # 重复潜在向量作为输入
        z_repeated = z.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # LSTM解码
        output, _ = self.lstm(z_repeated, (hidden, cell))
        
        # 输出投影
        decoded = self.output_layer(output)
        
        return decoded

class LSTMVAE(BaseGenerativeModel):
    """LSTM变分自编码器"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__(config)
        
        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        self.to(self.device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        reconstructed = self.decoder(z, x.size(1))
        
        return reconstructed, mu, logvar
    
    def generate_samples(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成样本"""
        self.eval()
        with torch.no_grad():
            # 从先验分布采样
            z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
            
            # 解码生成样本
            samples = self.decoder(z, self.config.sequence_length)
            
            return samples.cpu().numpy()
    
    def compute_loss(self, batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """计算VAE损失"""
        x = batch.to(self.device)
        
        # 前向传播
        reconstructed, mu, logvar = self.forward(x)
        
        # 重构损失
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 总损失
        total_loss = recon_loss + 0.1 * kl_loss  # β=0.1
        
        return {
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss
        }
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """训练一步"""
        losses = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

class DiffusionModel(BaseGenerativeModel):
    """扩散模型"""
    
    def __init__(self, config: CTBenchConfig):
        super().__init__(config)
        
        # 扩散参数
        self.num_timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # 计算扩散参数
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 噪声预测网络 (简化的UNet结构)
        self.noise_predictor = self._build_noise_predictor()
        
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # 移动参数到设备
        self.to(self.device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
    
    def _build_noise_predictor(self) -> nn.Module:
        """构建噪声预测网络"""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim + 1, self.config.hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.feature_dim)
        )
    
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散过程"""
        noise = torch.randn_like(x0)
        
        # 计算噪声图像
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # 重塑时间步张量以匹配批次维度
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播 (仅用于训练)"""
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (x.size(0),), device=self.device)
        
        # 前向扩散
        noisy_x, noise = self.forward_diffusion(x, t)
        
        # 预测噪声
        # 添加时间步信息
        t_embed = t.float().unsqueeze(-1).unsqueeze(-1).repeat(1, x.size(1), 1)
        input_with_t = torch.cat([noisy_x, t_embed], dim=-1)
        
        predicted_noise = self.noise_predictor(input_with_t)
        
        return predicted_noise, noise
    
    def generate_samples(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成样本 (反向扩散)"""
        self.eval()
        with torch.no_grad():
            # 从纯噪声开始
            x = torch.randn(num_samples, self.config.sequence_length, self.config.feature_dim, device=self.device)
            
            # 反向扩散过程
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, device=self.device)
                
                # 预测噪声
                t_embed = t_tensor.float().unsqueeze(-1).unsqueeze(-1).repeat(1, x.size(1), 1)
                input_with_t = torch.cat([x, t_embed], dim=-1)
                predicted_noise = self.noise_predictor(input_with_t)
                
                # 计算去噪后的x
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                if t > 0:
                    alpha_cumprod_t_prev = self.alphas_cumprod[t - 1]
                    noise = torch.randn_like(x)
                else:
                    alpha_cumprod_t_prev = torch.tensor(1.0)
                    noise = torch.zeros_like(x)
                
                # 更新x
                x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
                
                if t > 0:
                    x = x + torch.sqrt(beta_t) * noise
            
            return x.cpu().numpy()
    
    def compute_loss(self, batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """计算扩散损失"""
        x = batch.to(self.device)
        
        predicted_noise, true_noise = self.forward(x)
        
        # 简单的MSE损失
        loss = F.mse_loss(predicted_noise, true_noise)
        
        return {"total_loss": loss}
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """训练一步"""
        losses = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}

class CTBenchModelFactory:
    """CTBench模型工厂"""
    
    _model_classes = {
        GenerativeModelType.TRANSFORMER_GAN: TransformerGAN,
        GenerativeModelType.LSTM_VAE: LSTMVAE,
        GenerativeModelType.DIFFUSION: DiffusionModel,
        # 其他模型类型可以继续添加
    }
    
    @classmethod
    def create_model(cls, config: CTBenchConfig) -> BaseGenerativeModel:
        """创建模型"""
        model_class = cls._model_classes.get(config.model_type)
        if not model_class:
            raise ValueError(f"不支持的模型类型: {config.model_type}")
        
        return model_class(config)
    
    @classmethod
    def list_available_models(cls) -> List[GenerativeModelType]:
        """列出可用模型"""
        return list(cls._model_classes.keys())

class CTBenchTrainer:
    """CTBench训练器"""
    
    def __init__(self, model: BaseGenerativeModel, config: CTBenchConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger("CTBenchTrainer")
        
        # 训练历史
        self.training_history = []
    
    def prepare_data(self, data: np.ndarray) -> DataLoader:
        """准备训练数据"""
        # 创建滑动窗口序列
        sequences = []
        for i in range(len(data) - self.config.sequence_length + 1):
            sequences.append(data[i:i + self.config.sequence_length])
        
        sequences = np.array(sequences)
        
        # 转换为PyTorch数据集
        dataset = TensorDataset(torch.FloatTensor(sequences))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        return dataloader
    
    def train(self, dataloader: DataLoader, validate_every: int = 100) -> Dict[str, List[float]]:
        """训练模型"""
        self.logger.info(f"开始训练 {self.model.__class__.__name__}...")
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            self.model.train()
            for batch_idx, (batch,) in enumerate(dataloader):
                # 训练一步
                losses = self.model.train_step(batch)
                epoch_losses.append(losses)
                
                # 记录训练进度
                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{self.config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Loss: {losses.get('total_loss', 0):.6f}"
                    )
            
            # 计算平均损失
            avg_losses = {}
            if epoch_losses:
                loss_keys = epoch_losses[0].keys()
                for key in loss_keys:
                    avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            self.training_history.append(avg_losses)
            
            # 验证和保存
            if epoch % validate_every == 0 or epoch == self.config.num_epochs - 1:
                # 生成验证样本
                self._validate_generation()
                
                # 保存检查点
                self.model.save_checkpoint(epoch, avg_losses.get('total_loss', 0))
        
        self.logger.info("训练完成!")
        return self.training_history
    
    def _validate_generation(self):
        """验证生成质量"""
        try:
            # 生成少量样本进行验证
            samples = self.model.generate_samples(num_samples=10)
            
            # 基本统计检验
            sample_mean = np.mean(samples, axis=(0, 1))
            sample_std = np.std(samples, axis=(0, 1))
            
            self.logger.info(f"生成样本统计 - 均值: {sample_mean}, 标准差: {sample_std}")
            
        except Exception as e:
            self.logger.warning(f"验证生成失败: {e}")

class MarketRegimeClassifier(nn.Module):
    """市场状态分类器"""
    
    def __init__(self, feature_dim: int, num_regimes: int = 4):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_regimes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x.mean(dim=1))  # 时间维度平均池化

class ConditionalCTBench:
    """条件CTBench生成器"""
    
    def __init__(self, base_model: BaseGenerativeModel):
        self.base_model = base_model
        self.regime_classifier = MarketRegimeClassifier(base_model.config.feature_dim)
        self.logger = logging.getLogger("ConditionalCTBench")
    
    def train_regime_classifier(self, data: np.ndarray, labels: np.ndarray):
        """训练市场状态分类器"""
        # 实现分类器训练逻辑
        pass
    
    def generate_conditional_samples(self, num_samples: int, 
                                   target_regime: int) -> np.ndarray:
        """生成特定市场状态的样本"""
        # 实现条件生成逻辑
        pass