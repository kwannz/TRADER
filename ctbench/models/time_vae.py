"""
TimeVAE模型实现
时序感知的变分自编码器，用于金融时间序列生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from .base_model import BaseTSGModel, SimpleTrainingCallback

class TimeVAEEncoder(nn.Module):
    """TimeVAE编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, sequence_length: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 均值和方差网络
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 自注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 使用最后一个时间步的输出
        final_hidden = attn_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 计算均值和对数方差
        mu = self.fc_mu(final_hidden)
        logvar = self.fc_logvar(final_hidden)
        
        return mu, logvar

class TimeVAEDecoder(nn.Module):
    """TimeVAE解码器"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, sequence_length: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        
        # 潜在变量到隐状态的映射
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=latent_dim + output_dim,  # 潜在变量 + 上一步输出
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # 时间嵌入
        self.time_embedding = nn.Embedding(sequence_length, latent_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = z.size(0)
        device = z.device
        
        # 初始化隐状态
        h_0 = self.fc_hidden(z).unsqueeze(0).repeat(2, 1, 1)  # [2, batch_size, hidden_dim]
        c_0 = self.fc_cell(z).unsqueeze(0).repeat(2, 1, 1)
        
        outputs = []
        input_t = torch.zeros(batch_size, self.output_dim, device=device)  # 初始输入
        
        for t in range(self.sequence_length):
            # 时间嵌入
            time_emb = self.time_embedding(torch.tensor([t], device=device)).repeat(batch_size, 1)
            
            # 组合输入：潜在变量 + 时间嵌入 + 上一步输出
            lstm_input = torch.cat([z, input_t], dim=1).unsqueeze(1)
            
            # LSTM前向传播
            lstm_out, (h_0, c_0) = self.lstm(lstm_input, (h_0, c_0))
            
            # 输出
            output_t = self.fc_out(lstm_out.squeeze(1))
            outputs.append(output_t.unsqueeze(1))
            
            # 更新下一步的输入
            input_t = output_t
        
        # 组合所有时间步的输出
        return torch.cat(outputs, dim=1)  # [batch_size, sequence_length, output_dim]

class TimeVAE(BaseTSGModel):
    """TimeVAE主模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.hidden_dim = config.get('hidden_dim', 128)
        self.latent_dim = config.get('latent_dim', 64)
        self.beta = config.get('beta', 1.0)  # KL散度权重
        
        # 构建模型
        self.model = self.build_model()
        
    def build_model(self) -> nn.Module:
        """构建模型架构"""
        class TimeVAEModel(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                
            def forward(self, x):
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                reconstructed = self.decoder(z)
                return reconstructed, mu, logvar, z
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
        
        encoder = TimeVAEEncoder(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            sequence_length=self.sequence_length
        )
        
        decoder = TimeVAEDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            sequence_length=self.sequence_length
        )
        
        model = TimeVAEModel(encoder, decoder)
        return model.to(self.device)
    
    def train(self, train_data: torch.Tensor) -> Dict[str, List]:
        """训练模型"""
        try:
            logger.info("开始训练TimeVAE模型...")
            
            # 准备数据
            if len(train_data.shape) == 2:
                # 如果是DataFrame转换过来的，需要重新组织
                train_data = train_data.view(-1, self.sequence_length, self.feature_dim)
            
            # 数据加载器
            dataset = torch.utils.data.TensorDataset(train_data)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
            
            # 优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            
            # 训练回调
            callback = SimpleTrainingCallback(log_interval=10)
            callback.on_training_begin()
            
            # 训练历史
            history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
            
            self.model.train()
            
            for epoch in range(self.epochs):
                callback.on_epoch_begin(epoch)
                
                epoch_loss = 0.0
                epoch_recon_loss = 0.0
                epoch_kl_loss = 0.0
                
                for batch_idx, (batch_data,) in enumerate(dataloader):
                    batch_data = batch_data.to(self.device)
                    
                    callback.on_batch_begin(batch_idx)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    reconstructed, mu, logvar, z = self.model(batch_data)
                    
                    # 计算损失
                    recon_loss = F.mse_loss(reconstructed, batch_data, reduction='mean')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_data.size(0)
                    
                    total_loss = recon_loss + self.beta * kl_loss
                    
                    # 反向传播
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # 记录损失
                    epoch_loss += total_loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    
                    callback.on_batch_end(batch_idx, {
                        'batch_loss': total_loss.item(),
                        'recon_loss': recon_loss.item(),
                        'kl_loss': kl_loss.item()
                    })
                
                # Epoch统计
                avg_loss = epoch_loss / len(dataloader)
                avg_recon_loss = epoch_recon_loss / len(dataloader)
                avg_kl_loss = epoch_kl_loss / len(dataloader)
                
                history['loss'].append(avg_loss)
                history['recon_loss'].append(avg_recon_loss)
                history['kl_loss'].append(avg_kl_loss)
                
                # 学习率调整
                scheduler.step(avg_loss)
                
                callback.on_epoch_end(epoch, {
                    'loss': avg_loss,
                    'recon_loss': avg_recon_loss,
                    'kl_loss': avg_kl_loss
                })
                
                # 早停检查
                if epoch > 20 and len(history['loss']) > 10:
                    recent_loss = np.mean(history['loss'][-10:])
                    earlier_loss = np.mean(history['loss'][-20:-10])
                    if recent_loss >= earlier_loss:
                        logger.info(f"早停在epoch {epoch}")
                        break
            
            self.is_trained = True
            callback.on_training_end({'final_loss': history['loss'][-1]})
            
            logger.info(f"TimeVAE训练完成! 最终损失: {history['loss'][-1]:.6f}")
            return history
            
        except Exception as e:
            logger.error(f"TimeVAE训练失败: {e}")
            raise
    
    def generate(self, n_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成合成时间序列数据"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                # 从潜在空间采样
                if context is not None:
                    # 使用上下文信息
                    mu, logvar = self.model.encoder(context)
                    z = self.model.reparameterize(mu, logvar)
                    # 生成多个变体
                    z = z.repeat(n_samples, 1)
                    noise = torch.randn_like(z) * 0.1  # 添加噪声
                    z = z + noise
                else:
                    # 从先验分布采样
                    z = torch.randn(n_samples, self.latent_dim, device=self.device)
                
                # 解码生成数据
                generated_data = self.model.decoder(z)
                
            logger.info(f"TimeVAE生成了 {n_samples} 个样本")
            return generated_data
            
        except Exception as e:
            logger.error(f"TimeVAE生成失败: {e}")
            raise
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        """重构输入数据"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                if len(data.shape) == 2:
                    data = data.view(-1, self.sequence_length, self.feature_dim)
                
                reconstructed, _, _, _ = self.model(data.to(self.device))
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"TimeVAE重构失败: {e}")
            raise
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """编码数据到潜在空间"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                if len(data.shape) == 2:
                    data = data.view(-1, self.sequence_length, self.feature_dim)
                
                mu, logvar = self.model.encoder(data.to(self.device))
                z = self.model.reparameterize(mu, logvar)
            
            return z
            
        except Exception as e:
            logger.error(f"TimeVAE编码失败: {e}")
            raise
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在变量解码"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                reconstructed = self.model.decoder(z.to(self.device))
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"TimeVAE解码失败: {e}")
            raise
    
    def interpolate(self, data1: torch.Tensor, data2: torch.Tensor, 
                   steps: int = 10) -> torch.Tensor:
        """在两个数据点之间插值"""
        try:
            # 编码到潜在空间
            z1 = self.encode(data1)
            z2 = self.encode(data2)
            
            # 在潜在空间插值
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_z = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                interpolated_z.append(z_interp)
            
            interpolated_z = torch.cat(interpolated_z, dim=0)
            
            # 解码回数据空间
            interpolated_data = self.decode(interpolated_z)
            
            return interpolated_data
            
        except Exception as e:
            logger.error(f"TimeVAE插值失败: {e}")
            raise
    
    def get_latent_statistics(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取潜在变量统计信息"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                if len(data.shape) == 2:
                    data = data.view(-1, self.sequence_length, self.feature_dim)
                
                mu, logvar = self.model.encoder(data.to(self.device))
                std = torch.exp(0.5 * logvar)
            
            return {
                'mean': mu.mean(dim=0),
                'std': std.mean(dim=0),
                'mu': mu,
                'logvar': logvar,
                'std_individual': std
            }
            
        except Exception as e:
            logger.error(f"获取潜在变量统计失败: {e}")
            return {}

# 创建TimeVAE实例的便捷函数
def create_time_vae(config: Dict[str, Any]) -> TimeVAE:
    """创建TimeVAE实例"""
    default_config = {
        'sequence_length': 100,
        'feature_dim': 5,
        'hidden_dim': 128,
        'latent_dim': 64,
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'beta': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    default_config.update(config)
    return TimeVAE(default_config)