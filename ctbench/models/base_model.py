"""
CTBench基础模型接口
定义时间序列生成模型的统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from loguru import logger

class BaseTSGModel(ABC):
    """时间序列生成模型基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        self.device = torch.device(config.get('device', 'cpu'))
        self.sequence_length = config.get('sequence_length', 100)
        self.feature_dim = config.get('feature_dim', 5)  # OHLCV
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        
    @abstractmethod
    def build_model(self) -> nn.Module:
        """构建模型架构"""
        pass
    
    @abstractmethod
    def train(self, train_data: torch.Tensor) -> Dict[str, List]:
        """训练模型"""
        pass
    
    @abstractmethod
    def generate(self, n_samples: int, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成合成时间序列数据"""
        pass
    
    @abstractmethod
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        """重构输入数据"""
        pass
    
    def prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        """准备训练数据"""
        try:
            # 确保数据包含OHLCV列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"数据缺少必需列: {required_columns}")
            
            # 提取数值数据
            values = data[required_columns].values.astype(np.float32)
            
            # 数据标准化
            values = self._normalize_data(values)
            
            # 创建序列
            sequences = []
            for i in range(len(values) - self.sequence_length + 1):
                sequence = values[i:i + self.sequence_length]
                sequences.append(sequence)
            
            return torch.tensor(np.array(sequences), dtype=torch.float32, device=self.device)
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            raise
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据标准化"""
        try:
            # 使用对数差分来处理价格数据
            normalized = np.zeros_like(data)
            
            # 价格列 (OHLC) 使用对数差分
            for i in range(4):  # OHLC
                price_data = data[:, i]
                log_prices = np.log(price_data + 1e-8)  # 避免log(0)
                log_returns = np.diff(log_prices, prepend=log_prices[0])
                normalized[:, i] = log_returns
            
            # 成交量使用标准化
            volume_data = data[:, 4]
            volume_mean = np.mean(volume_data)
            volume_std = np.std(volume_data)
            if volume_std > 0:
                normalized[:, 4] = (volume_data - volume_mean) / volume_std
            else:
                normalized[:, 4] = volume_data
            
            return normalized
            
        except Exception as e:
            logger.error(f"数据标准化失败: {e}")
            return data
    
    def _denormalize_data(self, normalized_data: np.ndarray, 
                         original_data: np.ndarray) -> np.ndarray:
        """数据反标准化"""
        try:
            denormalized = np.zeros_like(normalized_data)
            
            # 价格列反标准化 (从对数差分恢复)
            for i in range(4):  # OHLC
                log_returns = normalized_data[:, i]
                # 使用原始数据的第一个价格作为起点
                initial_price = original_data[0, i]
                
                log_prices = np.cumsum(log_returns) + np.log(initial_price + 1e-8)
                prices = np.exp(log_prices)
                denormalized[:, i] = prices
            
            # 成交量反标准化
            volume_data = normalized_data[:, 4]
            original_volume = original_data[:, 4]
            volume_mean = np.mean(original_volume)
            volume_std = np.std(original_volume)
            
            if volume_std > 0:
                denormalized[:, 4] = volume_data * volume_std + volume_mean
            else:
                denormalized[:, 4] = volume_data
            
            return denormalized
            
        except Exception as e:
            logger.error(f"数据反标准化失败: {e}")
            return normalized_data
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            if self.model is None:
                raise ValueError("模型未初始化")
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'is_trained': self.is_trained
            }, filepath)
            
            logger.info(f"模型已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.config.update(checkpoint.get('config', {}))
            self.is_trained = checkpoint.get('is_trained', False)
            
            # 重新构建模型
            self.model = self.build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            
            logger.info(f"模型已加载: {filepath}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def evaluate_generation_quality(self, generated_data: torch.Tensor, 
                                   real_data: torch.Tensor) -> Dict[str, float]:
        """评估生成数据质量"""
        try:
            metrics = {}
            
            # 转换为numpy
            gen_np = generated_data.detach().cpu().numpy()
            real_np = real_data.detach().cpu().numpy()
            
            # 1. 均值比较
            metrics['mean_mse'] = np.mean((np.mean(gen_np, axis=0) - np.mean(real_np, axis=0))**2)
            
            # 2. 标准差比较
            metrics['std_mse'] = np.mean((np.std(gen_np, axis=0) - np.std(real_np, axis=0))**2)
            
            # 3. 相关性保持
            if gen_np.shape[-1] > 1:
                gen_corr = np.corrcoef(gen_np.reshape(-1, gen_np.shape[-1]).T)
                real_corr = np.corrcoef(real_np.reshape(-1, real_np.shape[-1]).T)
                metrics['corr_mse'] = np.mean((gen_corr - real_corr)**2)
            
            # 4. Wasserstein距离 (简化版)
            metrics['wasserstein_dist'] = self._calculate_wasserstein_distance(gen_np, real_np)
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估生成质量失败: {e}")
            return {}
    
    def _calculate_wasserstein_distance(self, gen_data: np.ndarray, 
                                      real_data: np.ndarray) -> float:
        """计算Wasserstein距离 (简化实现)"""
        try:
            # 简化版本：计算每个特征的1维Wasserstein距离的平均
            distances = []
            
            for i in range(gen_data.shape[-1]):
                gen_feature = gen_data[:, :, i].flatten()
                real_feature = real_data[:, :, i].flatten()
                
                # 排序计算Wasserstein-1距离
                gen_sorted = np.sort(gen_feature)
                real_sorted = np.sort(real_feature)
                
                # 确保长度一致
                min_len = min(len(gen_sorted), len(real_sorted))
                gen_sorted = gen_sorted[:min_len]
                real_sorted = real_sorted[:min_len]
                
                distance = np.mean(np.abs(gen_sorted - real_sorted))
                distances.append(distance)
            
            return np.mean(distances)
            
        except Exception as e:
            logger.warning(f"计算Wasserstein距离失败: {e}")
            return float('inf')
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "device": str(self.device),
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }
        
        if self.model is not None:
            # 计算参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        
        return info

class ModelTrainingCallback:
    """模型训练回调接口"""
    
    def on_epoch_begin(self, epoch: int):
        """epoch开始回调"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """epoch结束回调"""
        pass
    
    def on_batch_begin(self, batch: int):
        """batch开始回调"""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, float]):
        """batch结束回调"""
        pass
    
    def on_training_begin(self):
        """训练开始回调"""
        pass
    
    def on_training_end(self, logs: Dict[str, Any]):
        """训练结束回调"""
        pass

class SimpleTrainingCallback(ModelTrainingCallback):
    """简单的训练回调实现"""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        if epoch % self.log_interval == 0:
            loss = logs.get('loss', 0)
            logger.info(f"Epoch {epoch}: loss = {loss:.6f}")
    
    def on_training_begin(self):
        logger.info("开始模型训练...")
    
    def on_training_end(self, logs: Dict[str, Any]):
        logger.info(f"训练完成! 最终loss: {logs.get('final_loss', 'N/A')}")

# 工具函数
def create_model(model_type: str, config: Dict[str, Any]) -> BaseTSGModel:
    """根据类型创建模型实例"""
    if model_type == "TimeVAE":
        from .time_vae import TimeVAE
        return TimeVAE(config)
    elif model_type == "QuantGAN":
        from .quant_gan import QuantGAN
        return QuantGAN(config)
    elif model_type == "DiffusionTS":
        from .diffusion_ts import DiffusionTS
        return DiffusionTS(config)
    else:
        raise ValueError(f"未知模型类型: {model_type}")