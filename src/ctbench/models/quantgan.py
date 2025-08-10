"""
Quant-GAN Implementation for CTBench
量化生成对抗网络 - 专为金融时序数据设计的GAN模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from .base_model import BaseTimeSeriesGenerator, ModelConfig

class QuantGANConfig(ModelConfig):
    """QuantGAN配置"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim: int = kwargs.get('latent_dim', 100)
        self.generator_hidden_dims: list = kwargs.get('generator_hidden_dims', [128, 256, 512])
        self.discriminator_hidden_dims: list = kwargs.get('discriminator_hidden_dims', [512, 256, 128])
        self.d_learning_rate: float = kwargs.get('d_learning_rate', 2e-4)
        self.g_learning_rate: float = kwargs.get('g_learning_rate', 2e-4)
        self.beta1: float = kwargs.get('beta1', 0.5)
        self.beta2: float = kwargs.get('beta2', 0.999)
        self.n_critic: int = kwargs.get('n_critic', 5)  # 判别器训练次数
        self.gp_lambda: float = kwargs.get('gp_lambda', 10.0)  # 梯度惩罚系数
        self.use_spectral_norm: bool = kwargs.get('use_spectral_norm', True)

class SpectralNorm(nn.Module):
    """谱归一化层"""
    def __init__(self, module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        
        if not hasattr(module, name):
            raise ValueError(f"Module {module} has no attribute {name}")
            
        weight = getattr(module, name)
        with torch.no_grad():
            weight_mat = weight.view(weight.size(dim), -1)
            u = F.normalize(torch.randn(weight_mat.size(0)), dim=0, eps=eps)
            v = F.normalize(torch.randn(weight_mat.size(1)), dim=0, eps=eps)
            
        self.register_buffer(f"{name}_u", u)
        self.register_buffer(f"{name}_v", v)
        self.register_buffer(f"{name}_orig", weight.data)
        
    def _get_sigma(self):
        weight_mat = getattr(self.module, self.name).view(
            getattr(self.module, self.name).size(self.dim), -1)
        u = getattr(self, f"{self.name}_u")
        v = getattr(self, f"{self.name}_v")
        
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
            
        sigma = torch.matmul(u, torch.matmul(weight_mat, v))
        return sigma
        
    def forward(self, *args, **kwargs):
        sigma = self._get_sigma()
        weight = getattr(self.module, self.name)
        setattr(self.module, self.name, weight / sigma)
        return self.module(*args, **kwargs)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_dim, out_dim, use_spectral_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if use_spectral_norm:
            self.conv1 = SpectralNorm(nn.Conv1d(in_dim, out_dim, 3, padding=1))
            self.conv2 = SpectralNorm(nn.Conv1d(out_dim, out_dim, 3, padding=1))
            if in_dim != out_dim:
                self.shortcut = SpectralNorm(nn.Conv1d(in_dim, out_dim, 1))
            else:
                self.shortcut = nn.Identity()
        else:
            self.conv1 = nn.Conv1d(in_dim, out_dim, 3, padding=1)
            self.conv2 = nn.Conv1d(out_dim, out_dim, 3, padding=1)
            if in_dim != out_dim:
                self.shortcut = nn.Conv1d(in_dim, out_dim, 1)
            else:
                self.shortcut = nn.Identity()
                
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.leaky_relu(out, 0.2)
        
        return out

class QuantGenerator(nn.Module):
    """量化生成器"""
    
    def __init__(self, config: QuantGANConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.seq_len = config.sequence_length
        self.output_dim = config.output_dim
        
        # 输入投影层
        self.input_proj = nn.Linear(config.latent_dim, 
                                   config.generator_hidden_dims[0] * (config.sequence_length // 4))
        
        # 上采样层
        layers = []
        in_channels = config.generator_hidden_dims[0]
        
        for hidden_dim in config.generator_hidden_dims[1:]:
            layers.append(ResidualBlock(in_channels, hidden_dim, config.use_spectral_norm))
            layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            in_channels = hidden_dim
            
        self.conv_layers = nn.Sequential(*layers)
        
        # LSTM层用于时序建模
        self.lstm = nn.LSTM(in_channels, config.hidden_dim, 2, 
                           batch_first=True, dropout=0.1)
        
        # 输出层
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_activation = nn.Tanh()
        
        # 金融约束层
        self.price_constraint = PriceConstraintLayer()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        生成器前向传播
        Args:
            z: (batch_size, latent_dim) 噪声向量
        Returns:
            x: (batch_size, seq_len, output_dim) 生成的时序数据
        """
        batch_size = z.size(0)
        
        # 输入投影
        x = self.input_proj(z)  # (batch_size, hidden_dim * seq_len//4)
        x = x.view(batch_size, self.config.generator_hidden_dims[0], self.seq_len // 4)
        
        # 卷积上采样
        x = self.conv_layers(x)
        
        # 调整到目标序列长度
        if x.size(2) != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
            
        # 转换为LSTM输入格式
        x = x.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # LSTM处理
        x, _ = self.lstm(x)
        
        # 输出投影
        x = self.output_proj(x)
        x = self.output_activation(x)
        
        # 应用金融约束
        x = self.price_constraint(x)
        
        return x

class QuantDiscriminator(nn.Module):
    """量化判别器"""
    
    def __init__(self, config: QuantGANConfig):
        super().__init__()
        self.config = config
        
        # LSTM特征提取器
        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, 2,
                           batch_first=True, dropout=0.1)
        
        # 卷积特征提取器
        conv_layers = []
        in_channels = 1
        
        for hidden_dim in config.discriminator_hidden_dims:
            if config.use_spectral_norm:
                conv_layers.append(SpectralNorm(nn.Conv1d(in_channels, hidden_dim, 4, 2, 1)))
            else:
                conv_layers.append(nn.Conv1d(in_channels, hidden_dim, 4, 2, 1))
            conv_layers.append(nn.LeakyReLU(0.2))
            conv_layers.append(nn.Dropout(0.1))
            in_channels = hidden_dim
            
        self.conv_layers = nn.Sequential(*layers)
        
        # 特征融合
        self.feature_fusion = nn.Linear(
            config.hidden_dim + config.discriminator_hidden_dims[-1], 
            config.hidden_dim
        )
        
        # 输出层
        self.output = nn.Linear(config.hidden_dim, 1)
        
        # 辅助分类器（用于条件生成）
        self.aux_classifier = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        判别器前向传播
        Args:
            x: (batch_size, seq_len, input_dim) 输入时序数据
        Returns:
            validity: (batch_size, 1) 真实性分数
            aux_out: (batch_size, output_dim) 辅助输出
        """
        batch_size = x.size(0)
        
        # LSTM特征提取
        lstm_out, (hidden, _) = self.lstm(x)
        lstm_features = hidden[-1]  # 取最后一层隐状态
        
        # 卷积特征提取（使用价格数据）
        price_data = x[:, :, 0:1].transpose(1, 2)  # (batch_size, 1, seq_len)
        conv_features = self.conv_layers(price_data)
        conv_features = torch.mean(conv_features, dim=2)  # 全局平均池化
        
        # 特征融合
        combined_features = torch.cat([lstm_features, conv_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = F.leaky_relu(fused_features, 0.2)
        
        # 输出
        validity = self.output(fused_features)
        aux_out = self.aux_classifier(fused_features)
        
        return validity, aux_out

class PriceConstraintLayer(nn.Module):
    """价格约束层 - 确保生成的金融数据符合现实约束"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用金融约束
        Args:
            x: (batch_size, seq_len, features) 生成的数据
        Returns:
            constrained_x: 应用约束后的数据
        """
        # 假设前4个特征是OHLC价格
        if x.size(-1) >= 4:
            open_price = x[:, :, 0:1]
            high_price = x[:, :, 1:2]
            low_price = x[:, :, 2:3]
            close_price = x[:, :, 3:4]
            
            # 确保 high >= max(open, close) 且 low <= min(open, close)
            max_oc = torch.max(open_price, close_price)
            min_oc = torch.min(open_price, close_price)
            
            high_price = torch.max(high_price, max_oc)
            low_price = torch.min(low_price, min_oc)
            
            # 重构价格数据
            x = torch.cat([open_price, high_price, low_price, close_price] + 
                         [x[:, :, i:i+1] for i in range(4, x.size(-1))], dim=-1)
            
        return x

class QuantGAN(BaseTimeSeriesGenerator):
    """量化生成对抗网络"""
    
    def __init__(self, config: QuantGANConfig):
        super().__init__(config)
        self.config = config
        
        # 构建生成器和判别器
        self.generator = QuantGenerator(config)
        self.discriminator = QuantDiscriminator(config)
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.g_learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # 损失历史
        self.training_history = {'g_loss': [], 'd_loss': []}
        
        self.to_device()
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向传播（生成数据）"""
        return self.generator(z)
        
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成新样本
        Args:
            num_samples: 生成样本数量
            condition: 条件信息（暂未使用）
        Returns:
            generated_samples: (num_samples, seq_len, output_dim)
        """
        self.generator.eval()
        with torch.no_grad():
            # 从先验分布采样噪声
            z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
            
            # 生成样本
            generated_samples = self.generator(z)
            
        return generated_samples
        
    def compute_gradient_penalty(self, real_data: torch.Tensor, 
                               fake_data: torch.Tensor) -> torch.Tensor:
        """计算梯度惩罚（WGAN-GP）"""
        batch_size = real_data.size(0)
        
        # 随机插值
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 判别器输出
        d_interpolated, _ = self.discriminator(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
        
    def train_discriminator(self, real_data: torch.Tensor) -> float:
        """训练判别器"""
        self.d_optimizer.zero_grad()
        
        batch_size = real_data.size(0)
        
        # 真实数据
        real_validity, real_aux = self.discriminator(real_data)
        
        # 假数据
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_data = self.generator(z).detach()
        fake_validity, fake_aux = self.discriminator(fake_data)
        
        # WGAN-GP损失
        d_loss = torch.mean(fake_validity) - torch.mean(real_validity)
        
        # 梯度惩罚
        gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
        d_loss += self.config.gp_lambda * gradient_penalty
        
        # 辅助损失（特征匹配）
        aux_loss = F.mse_loss(real_aux, torch.mean(real_data, dim=1))
        d_loss += 0.1 * aux_loss
        
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
        
    def train_generator(self, batch_size: int) -> float:
        """训练生成器"""
        self.g_optimizer.zero_grad()
        
        # 生成假数据
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_data = self.generator(z)
        fake_validity, fake_aux = self.discriminator(fake_data)
        
        # 生成器损失
        g_loss = -torch.mean(fake_validity)
        
        # 特征匹配损失
        feature_matching_loss = F.mse_loss(fake_aux, torch.mean(fake_data, dim=1))
        g_loss += 0.1 * feature_matching_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
        
    def fit(self, data: torch.Tensor, val_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """训练QuantGAN模型"""
        self.generator.train()
        self.discriminator.train()
        
        # 数据加载器
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        for epoch in range(self.config.epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            num_batches = 0
            
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                
                # 训练判别器
                d_loss = self.train_discriminator(batch_data)
                epoch_d_loss += d_loss
                
                # 每n_critic步训练一次生成器
                if num_batches % self.config.n_critic == 0:
                    g_loss = self.train_generator(batch_data.size(0))
                    epoch_g_loss += g_loss
                    
                num_batches += 1
                
            # 记录平均损失
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / (num_batches // self.config.n_critic + 1)
            
            self.training_history['d_loss'].append(avg_d_loss)
            self.training_history['g_loss'].append(avg_g_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
                
        return {
            'final_d_loss': self.training_history['d_loss'][-1],
            'final_g_loss': self.training_history['g_loss'][-1]
        }