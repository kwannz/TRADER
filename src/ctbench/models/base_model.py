"""
CTBench Base Model Implementation
时序生成模型基类
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置基类"""
    model_name: str
    input_dim: int
    output_dim: int
    sequence_length: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BaseTimeSeriesGenerator(nn.Module, ABC):
    """时序生成模型基类"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = config.model_name
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pass
        
    @abstractmethod
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成时序数据"""
        pass
        
    @abstractmethod
    def fit(self, data: torch.Tensor) -> Dict[str, float]:
        """训练模型"""
        pass
        
    def save_model(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name
        }, path)
        
    def load_model(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
    def to_device(self):
        """移动到指定设备"""
        return self.to(self.device)
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'config': self.config.__dict__,
            'device': str(self.device)
        }
        
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: BaseTimeSeriesGenerator):
        self.model = model
        self.training_history = []
        
    def train_epoch(self, dataloader, optimizer, criterion) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            batch_data = batch_data.to(self.model.device)
            optimizer.zero_grad()
            
            output = self.model(batch_data)
            loss = criterion(output, batch_data)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def validate(self, dataloader, criterion) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = batch_data.to(self.model.device)
                output = self.model(batch_data)
                loss = criterion(output, batch_data)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def get_training_history(self) -> list:
        """获取训练历史"""
        return self.training_history