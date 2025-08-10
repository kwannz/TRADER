"""
Synthetic Data Manager for CTBench Integration
合成数据管理器，集成多种时序生成模型
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json

from ..models.timevae import TimeVAE, TimeVAEConfig
from ..models.base_model import BaseTimeSeriesGenerator

class SyntheticDataManager:
    """合成数据管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.models: Dict[str, BaseTimeSeriesGenerator] = {}
        self.model_configs: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # 默认配置
        self.default_config = {
            'timevae': {
                'model_name': 'TimeVAE',
                'input_dim': 6,  # OHLCV + Volume
                'output_dim': 6,
                'sequence_length': 60,  # 60个时间步
                'hidden_dim': 128,
                'latent_dim': 32,
                'num_layers': 2,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'epochs': 100,
                'beta': 1.0
            }
        }
        
        # 加载配置
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self.model_configs = self.default_config.copy()
            
    def load_config(self, config_path: str) -> None:
        """加载模型配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.model_configs = json.load(f)
            self.logger.info(f"配置已从 {config_path} 加载")
        except Exception as e:
            self.logger.warning(f"无法加载配置文件 {config_path}: {e}")
            self.model_configs = self.default_config.copy()
            
    def save_config(self, config_path: str) -> None:
        """保存模型配置"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_configs, f, indent=2, ensure_ascii=False)
            self.logger.info(f"配置已保存到 {config_path}")
        except Exception as e:
            self.logger.error(f"无法保存配置文件 {config_path}: {e}")
            
    def initialize_model(self, model_type: str) -> bool:
        """初始化指定类型的模型"""
        try:
            if model_type not in self.model_configs:
                self.logger.error(f"未找到模型类型 {model_type} 的配置")
                return False
                
            config_dict = self.model_configs[model_type]
            
            if model_type == 'timevae':
                config = TimeVAEConfig(**config_dict)
                model = TimeVAE(config)
            else:
                self.logger.error(f"不支持的模型类型: {model_type}")
                return False
                
            self.models[model_type] = model
            self.logger.info(f"模型 {model_type} 初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化模型 {model_type} 失败: {e}")
            return False
            
    def train_model(self, model_type: str, data: np.ndarray, 
                   val_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """训练指定模型"""
        if model_type not in self.models:
            if not self.initialize_model(model_type):
                return {'success': False, 'error': f'无法初始化模型 {model_type}'}
                
        model = self.models[model_type]
        
        try:
            # 数据预处理
            train_tensor = torch.FloatTensor(data)
            val_tensor = torch.FloatTensor(val_data) if val_data is not None else None
            
            # 训练模型
            self.logger.info(f"开始训练模型 {model_type}")
            training_history = model.fit(train_tensor, val_tensor)
            
            return {
                'success': True,
                'model_type': model_type,
                'training_history': training_history,
                'model_info': model.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"训练模型 {model_type} 失败: {e}")
            return {'success': False, 'error': str(e)}
            
    def generate_synthetic_data(self, model_type: str, num_samples: int,
                              condition: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """生成合成数据"""
        if model_type not in self.models:
            return {'success': False, 'error': f'模型 {model_type} 未初始化'}
            
        model = self.models[model_type]
        
        try:
            # 生成数据
            condition_tensor = torch.FloatTensor(condition) if condition is not None else None
            synthetic_data = model.generate(num_samples, condition_tensor)
            
            # 转换为numpy数组
            synthetic_array = synthetic_data.cpu().detach().numpy()
            
            return {
                'success': True,
                'model_type': model_type,
                'data': synthetic_array,
                'shape': synthetic_array.shape,
                'num_samples': num_samples
            }
            
        except Exception as e:
            self.logger.error(f"生成合成数据失败 {model_type}: {e}")
            return {'success': False, 'error': str(e)}
            
    def generate_market_scenarios(self, scenario_type: str, 
                                 base_data: np.ndarray,
                                 num_scenarios: int = 100) -> Dict[str, Any]:
        """生成特定市场场景"""
        scenarios = {
            'black_swan': self._generate_black_swan_scenarios,
            'bull_market': self._generate_bull_market_scenarios,
            'bear_market': self._generate_bear_market_scenarios,
            'high_volatility': self._generate_high_volatility_scenarios,
            'sideways': self._generate_sideways_scenarios
        }
        
        if scenario_type not in scenarios:
            return {'success': False, 'error': f'不支持的场景类型: {scenario_type}'}
            
        try:
            scenario_data = scenarios[scenario_type](base_data, num_scenarios)
            return {
                'success': True,
                'scenario_type': scenario_type,
                'data': scenario_data,
                'num_scenarios': num_scenarios
            }
        except Exception as e:
            self.logger.error(f"生成场景 {scenario_type} 失败: {e}")
            return {'success': False, 'error': str(e)}
            
    def _generate_black_swan_scenarios(self, base_data: np.ndarray, 
                                     num_scenarios: int) -> np.ndarray:
        """生成黑天鹅事件场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = base_data.copy()
            
            # 随机选择黑天鹅事件发生时间点
            event_start = np.random.randint(10, len(scenario) - 20)
            event_duration = np.random.randint(1, 10)
            
            # 极端价格变动（-50% 到 +200%）
            extreme_change = np.random.choice([-0.5, -0.3, 2.0, 1.5, 3.0])
            
            # 应用极端变动到价格数据 (假设前4列是OHLC)
            for i in range(event_start, min(event_start + event_duration, len(scenario))):
                scenario[i, :4] *= (1 + extreme_change)
                # 极端交易量增加
                scenario[i, 4] *= np.random.uniform(5, 20)
                
            scenarios.append(scenario)
            
        return np.array(scenarios)
        
    def _generate_bull_market_scenarios(self, base_data: np.ndarray,
                                       num_scenarios: int) -> np.ndarray:
        """生成牛市场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = base_data.copy()
            
            # 持续上涨趋势
            trend_strength = np.random.uniform(0.001, 0.005)  # 每日0.1%-0.5%上涨
            noise_level = np.random.uniform(0.8, 1.2)
            
            cumulative_return = 1.0
            for i in range(len(scenario)):
                daily_return = trend_strength * noise_level * np.random.normal(1, 0.1)
                cumulative_return *= (1 + daily_return)
                scenario[i, :4] *= cumulative_return
                
            scenarios.append(scenario)
            
        return np.array(scenarios)
        
    def _generate_bear_market_scenarios(self, base_data: np.ndarray,
                                       num_scenarios: int) -> np.ndarray:
        """生成熊市场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = base_data.copy()
            
            # 持续下跌趋势
            trend_strength = np.random.uniform(-0.005, -0.001)  # 每日-0.5%到-0.1%下跌
            noise_level = np.random.uniform(0.8, 1.2)
            
            cumulative_return = 1.0
            for i in range(len(scenario)):
                daily_return = trend_strength * noise_level * np.random.normal(1, 0.1)
                cumulative_return *= (1 + daily_return)
                scenario[i, :4] *= cumulative_return
                
            scenarios.append(scenario)
            
        return np.array(scenarios)
        
    def _generate_high_volatility_scenarios(self, base_data: np.ndarray,
                                          num_scenarios: int) -> np.ndarray:
        """生成高波动率场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = base_data.copy()
            
            # 高波动率
            volatility_multiplier = np.random.uniform(2.0, 5.0)
            
            for i in range(1, len(scenario)):
                # 增加价格波动
                price_change = np.random.normal(0, 0.02 * volatility_multiplier)
                scenario[i, :4] = scenario[i-1, :4] * (1 + price_change)
                # 增加交易量
                scenario[i, 4] *= np.random.uniform(1.5, 3.0)
                
            scenarios.append(scenario)
            
        return np.array(scenarios)
        
    def _generate_sideways_scenarios(self, base_data: np.ndarray,
                                   num_scenarios: int) -> np.ndarray:
        """生成横盘整理场景"""
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = base_data.copy()
            
            # 横盘整理，价格在小范围内波动
            base_price = np.mean(scenario[:, 0])  # 使用开盘价均值作为基准
            range_pct = np.random.uniform(0.02, 0.08)  # 2%-8%的波动范围
            
            for i in range(len(scenario)):
                # 在基准价格附近小幅波动
                price_factor = 1 + np.random.uniform(-range_pct, range_pct)
                scenario[i, :4] = base_price * price_factor
                
            scenarios.append(scenario)
            
        return np.array(scenarios)
        
    def save_model(self, model_type: str, path: str) -> bool:
        """保存模型"""
        if model_type not in self.models:
            self.logger.error(f"模型 {model_type} 未初始化")
            return False
            
        try:
            self.models[model_type].save_model(path)
            self.logger.info(f"模型 {model_type} 已保存到 {path}")
            return True
        except Exception as e:
            self.logger.error(f"保存模型 {model_type} 失败: {e}")
            return False
            
    def load_model(self, model_type: str, path: str) -> bool:
        """加载模型"""
        try:
            if model_type not in self.models:
                if not self.initialize_model(model_type):
                    return False
                    
            self.models[model_type].load_model(path)
            self.logger.info(f"模型 {model_type} 已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载模型 {model_type} 失败: {e}")
            return False
            
    def get_model_status(self) -> Dict[str, Any]:
        """获取所有模型状态"""
        status = {}
        for model_type, model in self.models.items():
            status[model_type] = {
                'initialized': True,
                'info': model.get_model_info()
            }
            
        # 添加未初始化的模型
        for model_type in self.model_configs:
            if model_type not in status:
                status[model_type] = {'initialized': False}
                
        return status
        
    async def batch_generate(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量生成合成数据"""
        tasks = []
        for request in requests:
            task = asyncio.create_task(
                self._async_generate(request)
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
        
    async def _async_generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """异步生成数据"""
        model_type = request.get('model_type')
        num_samples = request.get('num_samples', 100)
        condition = request.get('condition')
        
        return self.generate_synthetic_data(model_type, num_samples, condition)