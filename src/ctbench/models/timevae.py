"""
TimeVAE Implementation for CTBench
基于变分自编码器的时序生成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from .base_model import BaseTimeSeriesGenerator, ModelConfig

class TimeVAEConfig(ModelConfig):
    """TimeVAE配置"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim: int = kwargs.get('latent_dim', 32)
        self.beta: float = kwargs.get('beta', 1.0)  # KL散度权重
        self.reconstruction_loss: str = kwargs.get('reconstruction_loss', 'mse')

class Encoder(nn.Module):
    """编码器网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )
        
        # 均值和方差网络
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入序列
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        # LSTM编码
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]  # 取最后一层的隐状态
        
        # 计算均值和对数方差
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

class Decoder(nn.Module):
    """解码器网络"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 潜在向量到隐状态
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            latent_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.1 if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在向量
        Args:
            z: (batch_size, latent_dim)
        Returns:
            x_recon: (batch_size, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # 扩展潜在向量到序列长度
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # LSTM解码
        lstm_out, _ = self.lstm(z_expanded)
        
        # 输出层
        x_recon = self.fc_out(lstm_out)
        
        return x_recon

class TimeVAE(BaseTimeSeriesGenerator):
    """TimeVAE时序生成模型"""
    
    def __init__(self, config: TimeVAEConfig):
        super().__init__(config)
        self.config = config
        
        # 构建编码器和解码器
        self.encoder = Encoder(
            config.input_dim, config.hidden_dim, 
            config.latent_dim, config.num_layers
        )
        
        self.decoder = Decoder(
            config.latent_dim, config.hidden_dim,
            config.output_dim, config.sequence_length, config.num_layers
        )
        
        self.to_device()
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            x_recon: 重建序列
            mu: 潜在空间均值
            logvar: 潜在空间对数方差
        """
        # 编码
        mu, logvar = self.encoder(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
        
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成新样本
        Args:
            num_samples: 生成样本数量
            condition: 条件信息（暂未使用）
        Returns:
            generated_samples: (num_samples, seq_len, output_dim)
        """
        self.eval()
        with torch.no_grad():
            # 从先验分布采样
            z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
            
            # 解码生成样本
            generated_samples = self.decoder(z)
            
        return generated_samples
        
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                    mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算VAE损失"""
        # 重建损失
        if self.config.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        else:
            recon_loss = F.l1_loss(x_recon, x, reduction='sum')
            
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失
        total_loss = recon_loss + self.config.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        
    def fit(self, data: torch.Tensor, val_data: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """训练TimeVAE模型"""
        self.train()
        
        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        # 数据加载器
        dataset = torch.utils.data.TensorDataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        training_history = []
        
        for epoch in range(self.config.epochs):
            epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
            num_batches = 0
            
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                x_recon, mu, logvar = self(batch_data)
                
                # 计算损失
                losses = self.compute_loss(batch_data, x_recon, mu, logvar)
                
                # 反向传播
                losses['total_loss'].backward()
                optimizer.step()
                
                # 记录损失
                epoch_losses['total'] += losses['total_loss'].item()
                epoch_losses['recon'] += losses['recon_loss'].item()
                epoch_losses['kl'] += losses['kl_loss'].item()
                num_batches += 1
                
            # 平均损失
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                
            training_history.append(epoch_losses)
            
            # 验证
            if val_data is not None and epoch % 10 == 0:
                val_loss = self._validate(val_data)
                print(f"Epoch {epoch}, Train Loss: {epoch_losses['total']:.4f}, Val Loss: {val_loss:.4f}")
            elif epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {epoch_losses['total']:.4f}")
                
        return training_history[-1] if training_history else {}
        
    def _validate(self, val_data: torch.Tensor) -> float:
        """验证模型"""
        self.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            val_dataset = torch.utils.data.TensorDataset(val_data)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            for batch_data, in val_dataloader:
                batch_data = batch_data.to(self.device)
                x_recon, mu, logvar = self(batch_data)
                losses = self.compute_loss(batch_data, x_recon, mu, logvar)
                
                total_loss += losses['total_loss'].item()
                num_samples += batch_data.size(0)
                
        return total_loss / num_samples if num_samples > 0 else 0.0