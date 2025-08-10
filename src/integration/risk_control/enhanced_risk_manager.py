"""
Enhanced Risk Manager with CTBench Integration
增强风险管理器，集成CTBench合成数据生成能力
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from enum import Enum

from ..model_service import get_ctbench_service, DataGenerationRequest

class RiskLevel(Enum):
    """风险等级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5

@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_risk: RiskLevel
    portfolio_value: float
    var_1d: float  # 1日风险价值
    var_5d: float  # 5日风险价值
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    black_swan_probability: float
    stress_test_results: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

@dataclass
class PositionRisk:
    """仓位风险"""
    symbol: str
    position_size: float
    market_value: float
    risk_contribution: float
    beta: float
    correlation_risk: float
    liquidity_risk: float

class EnhancedRiskManager:
    """增强风险管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        self.ctbench_service = None
        self.risk_history = []
        
        # 风险阈值
        self.risk_thresholds = {
            RiskLevel.LOW: 0.02,      # 2% VaR
            RiskLevel.MEDIUM: 0.05,   # 5% VaR
            RiskLevel.HIGH: 0.10,     # 10% VaR
            RiskLevel.CRITICAL: 0.20, # 20% VaR
            RiskLevel.EXTREME: 0.50   # 50% VaR
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'var_confidence_level': 0.05,  # 95% VaR
            'stress_test_scenarios': 1000,
            'max_position_weight': 0.1,
            'correlation_threshold': 0.8,
            'liquidity_threshold': 0.05,
            'black_swan_threshold': 0.01,
            'rebalance_threshold': 0.05
        }
        
    async def initialize(self):
        """初始化风险管理器"""
        self.ctbench_service = await get_ctbench_service()
        self.logger.info("增强风险管理器已初始化")
        
    async def comprehensive_risk_assessment(self, 
                                          portfolio_data: Dict[str, Any],
                                          market_data: pd.DataFrame) -> RiskAssessment:
        """综合风险评估"""
        try:
            # 基础风险指标计算
            basic_metrics = self._calculate_basic_metrics(portfolio_data, market_data)
            
            # CTBench增强分析
            enhanced_metrics = await self._ctbench_enhanced_analysis(market_data)
            
            # 压力测试
            stress_results = await self._comprehensive_stress_testing(portfolio_data, market_data)
            
            # 黑天鹅事件概率估计
            black_swan_prob = await self._estimate_black_swan_probability(market_data)
            
            # 综合风险等级评估
            overall_risk = self._assess_overall_risk_level(
                basic_metrics, enhanced_metrics, stress_results, black_swan_prob
            )
            
            # 生成风险建议
            recommendations = self._generate_risk_recommendations(
                overall_risk, basic_metrics, enhanced_metrics, stress_results
            )
            
            risk_assessment = RiskAssessment(
                overall_risk=overall_risk,
                portfolio_value=portfolio_data.get('total_value', 0),
                var_1d=basic_metrics['var_1d'],
                var_5d=basic_metrics['var_5d'],
                max_drawdown=basic_metrics['max_drawdown'],
                sharpe_ratio=basic_metrics['sharpe_ratio'],
                volatility=basic_metrics['volatility'],
                black_swan_probability=black_swan_prob,
                stress_test_results=stress_results,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            # 记录风险评估历史
            self.risk_history.append(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"综合风险评估失败: {e}")
            raise
            
    def _calculate_basic_metrics(self, portfolio_data: Dict[str, Any],
                               market_data: pd.DataFrame) -> Dict[str, float]:
        """计算基础风险指标"""
        metrics = {}
        
        try:
            # 假设market_data包含价格数据
            prices = market_data.iloc[:, 0].values  # 假设第一列是价格
            returns = np.diff(np.log(prices))
            
            # VaR计算
            var_percentile = self.config['var_confidence_level']
            metrics['var_1d'] = np.percentile(returns, var_percentile * 100)
            metrics['var_5d'] = metrics['var_1d'] * np.sqrt(5)  # 简化的5日VaR
            
            # 最大回撤
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdowns)
            
            # 波动率
            metrics['volatility'] = np.std(returns) * np.sqrt(252)  # 年化波动率
            
            # 夏普比率（简化版本，假设无风险利率为0）
            mean_return = np.mean(returns) * 252
            metrics['sharpe_ratio'] = mean_return / metrics['volatility'] if metrics['volatility'] > 0 else 0
            
        except Exception as e:
            self.logger.error(f"计算基础指标时出错: {e}")
            # 返回默认值
            metrics = {
                'var_1d': 0.0, 'var_5d': 0.0, 'max_drawdown': 0.0,
                'volatility': 0.0, 'sharpe_ratio': 0.0
            }
            
        return metrics
        
    async def _ctbench_enhanced_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """CTBench增强分析"""
        if self.ctbench_service is None:
            return {}
            
        try:
            # 将市场数据转换为numpy数组（假设是OHLCV格式）
            data_array = market_data.values
            if data_array.shape[1] < 6:
                # 如果列数不足，用价格数据填充
                price_col = data_array[:, [0]]
                data_array = np.hstack([
                    price_col, price_col, price_col, price_col, 
                    price_col, price_col
                ])[:, :6]
                
            # 重塑为CTBench期望的格式 (batch, sequence, features)
            sequence_length = min(60, data_array.shape[0])
            data_reshaped = data_array[-sequence_length:].reshape(1, sequence_length, -1)
            
            # 使用CTBench生成增强数据进行分析
            augmentation_result = await self.ctbench_service.get_real_time_market_data_augmentation(
                data_reshaped, augmentation_factor=100
            )
            
            if augmentation_result['success']:
                augmented_data = augmentation_result['augmented_data']
                
                # 分析增强数据的统计特性
                enhanced_metrics = {
                    'scenario_volatility_range': self._analyze_scenario_volatility(augmented_data),
                    'extreme_event_frequency': self._count_extreme_events(augmented_data),
                    'tail_risk_metrics': self._calculate_tail_risks(augmented_data),
                    'scenario_correlation': self._analyze_scenario_correlations(augmented_data)
                }
                
                return enhanced_metrics
            else:
                self.logger.warning("CTBench数据增强失败")
                return {}
                
        except Exception as e:
            self.logger.error(f"CTBench增强分析失败: {e}")
            return {}
            
    def _analyze_scenario_volatility(self, augmented_data: np.ndarray) -> Dict[str, float]:
        """分析场景波动率分布"""
        volatilities = []
        
        for scenario in augmented_data:
            returns = np.diff(np.log(scenario[:, 0] + 1e-8))  # 避免log(0)
            vol = np.std(returns)
            volatilities.append(vol)
            
        return {
            'min_volatility': float(np.min(volatilities)),
            'max_volatility': float(np.max(volatilities)),
            'median_volatility': float(np.median(volatilities)),
            'volatility_95th': float(np.percentile(volatilities, 95))
        }
        
    def _count_extreme_events(self, augmented_data: np.ndarray) -> float:
        """统计极端事件频率"""
        extreme_threshold = 0.05  # 5%的价格变动视为极端事件
        total_events = 0
        total_observations = 0
        
        for scenario in augmented_data:
            returns = np.diff(scenario[:, 0]) / (scenario[:-1, 0] + 1e-8)
            extreme_events = np.sum(np.abs(returns) > extreme_threshold)
            total_events += extreme_events
            total_observations += len(returns)
            
        return total_events / total_observations if total_observations > 0 else 0
        
    def _calculate_tail_risks(self, augmented_data: np.ndarray) -> Dict[str, float]:
        """计算尾部风险指标"""
        all_returns = []
        
        for scenario in augmented_data:
            returns = np.diff(scenario[:, 0]) / (scenario[:-1, 0] + 1e-8)
            all_returns.extend(returns)
            
        all_returns = np.array(all_returns)
        
        return {
            'var_1pct': float(np.percentile(all_returns, 1)),
            'var_5pct': float(np.percentile(all_returns, 5)),
            'cvar_5pct': float(np.mean(all_returns[all_returns <= np.percentile(all_returns, 5)])),
            'skewness': float(self._calculate_skewness(all_returns)),
            'kurtosis': float(self._calculate_kurtosis(all_returns))
        }
        
    def _analyze_scenario_correlations(self, augmented_data: np.ndarray) -> Dict[str, float]:
        """分析场景间相关性"""
        if augmented_data.shape[2] < 2:
            return {'avg_correlation': 0.0}
            
        correlations = []
        for scenario in augmented_data:
            if scenario.shape[1] >= 2:
                corr_matrix = np.corrcoef(scenario[:, 0], scenario[:, 1])
                if not np.isnan(corr_matrix[0, 1]):
                    correlations.append(corr_matrix[0, 1])
                    
        return {
            'avg_correlation': float(np.mean(correlations)) if correlations else 0.0,
            'max_correlation': float(np.max(correlations)) if correlations else 0.0,
            'min_correlation': float(np.min(correlations)) if correlations else 0.0
        }
        
    async def _comprehensive_stress_testing(self, portfolio_data: Dict[str, Any],
                                          market_data: pd.DataFrame) -> Dict[str, float]:
        """综合压力测试"""
        if self.ctbench_service is None:
            return {}
            
        try:
            # 准备基础数据
            data_array = market_data.values
            if data_array.shape[1] < 6:
                price_col = data_array[:, [0]]
                data_array = np.hstack([price_col] * 6)[:, :6]
                
            sequence_length = min(60, data_array.shape[0])
            data_reshaped = data_array[-sequence_length:].reshape(1, sequence_length, -1)
            
            # 生成压力测试场景
            stress_result = await self.ctbench_service.generate_stress_test_scenarios(
                data_reshaped, ['black_swan', 'high_volatility', 'bear_market']
            )
            
            if not stress_result['success']:
                return {}
                
            stress_scenarios = stress_result['stress_scenarios']
            portfolio_value = portfolio_data.get('total_value', 1000000)
            
            # 评估每种压力场景下的损失
            stress_results = {}
            
            for scenario_type, scenarios in stress_scenarios.items():
                losses = []
                for scenario in scenarios:
                    # 计算该场景下的组合损失（简化计算）
                    scenario_returns = np.diff(scenario[:, 0]) / (scenario[:-1, 0] + 1e-8)
                    scenario_loss = -np.sum(scenario_returns) * portfolio_value
                    losses.append(scenario_loss)
                    
                stress_results[f'{scenario_type}_worst_loss'] = float(np.max(losses))
                stress_results[f'{scenario_type}_avg_loss'] = float(np.mean(losses))
                stress_results[f'{scenario_type}_var_95'] = float(np.percentile(losses, 95))
                
            return stress_results
            
        except Exception as e:
            self.logger.error(f"综合压力测试失败: {e}")
            return {}
            
    async def _estimate_black_swan_probability(self, market_data: pd.DataFrame) -> float:
        """估计黑天鹅事件概率"""
        try:
            prices = market_data.iloc[:, 0].values
            returns = np.diff(np.log(prices))
            
            # 使用极值理论估计尾部概率
            threshold = np.percentile(np.abs(returns), 95)  # 95%分位数作为阈值
            extreme_returns = returns[np.abs(returns) > threshold]
            
            if len(extreme_returns) == 0:
                return 0.0
                
            # 简化的黑天鹅概率估计（基于历史极值频率）
            extreme_frequency = len(extreme_returns) / len(returns)
            
            # 调整系数（考虑尾部厚度）
            kurtosis = self._calculate_kurtosis(returns)
            tail_adjustment = max(1.0, kurtosis / 3.0)  # 正态分布峰度为3
            
            black_swan_prob = extreme_frequency * tail_adjustment
            
            return min(black_swan_prob, 0.1)  # 最大概率限制为10%
            
        except Exception as e:
            self.logger.error(f"估计黑天鹅概率失败: {e}")
            return 0.0
            
    def _assess_overall_risk_level(self, basic_metrics: Dict[str, float],
                                 enhanced_metrics: Dict[str, Any],
                                 stress_results: Dict[str, float],
                                 black_swan_prob: float) -> RiskLevel:
        """评估总体风险等级"""
        risk_scores = []
        
        # 基于VaR的风险评分
        var_1d = abs(basic_metrics.get('var_1d', 0))
        for level, threshold in self.risk_thresholds.items():
            if var_1d >= threshold:
                risk_scores.append(level.value)
                
        # 基于最大回撤的风险评分
        max_dd = abs(basic_metrics.get('max_drawdown', 0))
        if max_dd > 0.3:
            risk_scores.append(RiskLevel.EXTREME.value)
        elif max_dd > 0.2:
            risk_scores.append(RiskLevel.CRITICAL.value)
        elif max_dd > 0.1:
            risk_scores.append(RiskLevel.HIGH.value)
            
        # 基于黑天鹅概率的风险评分
        if black_swan_prob > 0.05:
            risk_scores.append(RiskLevel.EXTREME.value)
        elif black_swan_prob > 0.02:
            risk_scores.append(RiskLevel.CRITICAL.value)
            
        # 基于压力测试结果的风险评分
        if stress_results:
            max_stress_loss = max([v for k, v in stress_results.items() if 'worst_loss' in k], default=0)
            portfolio_value = 1000000  # 假设组合价值
            stress_loss_ratio = max_stress_loss / portfolio_value
            
            if stress_loss_ratio > 0.5:
                risk_scores.append(RiskLevel.EXTREME.value)
            elif stress_loss_ratio > 0.3:
                risk_scores.append(RiskLevel.CRITICAL.value)
                
        # 取最高风险等级
        if risk_scores:
            max_risk_score = max(risk_scores)
            return RiskLevel(max_risk_score)
        else:
            return RiskLevel.LOW
            
    def _generate_risk_recommendations(self, overall_risk: RiskLevel,
                                     basic_metrics: Dict[str, float],
                                     enhanced_metrics: Dict[str, Any],
                                     stress_results: Dict[str, float]) -> List[str]:
        """生成风险管理建议"""
        recommendations = []
        
        if overall_risk == RiskLevel.EXTREME:
            recommendations.extend([
                "⚠️ 极端风险警告：建议立即减仓至最低水平",
                "🔴 启动紧急风险管理程序",
                "📊 暂停自动交易，启用人工审核",
                "💰 考虑对冲操作以降低系统性风险"
            ])
        elif overall_risk == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 高风险警告：建议大幅减少仓位",
                "📉 增加现金比例至50%以上",
                "🛡️ 启用严格的止损机制",
                "📈 考虑反向ETF对冲"
            ])
        elif overall_risk == RiskLevel.HIGH:
            recommendations.extend([
                "⚡ 风险偏高：建议适度减仓",
                "🎯 将单一仓位限制在5%以下",
                "📊 增加风险监控频率",
                "💼 考虑分散投资策略"
            ])
        elif overall_risk == RiskLevel.MEDIUM:
            recommendations.extend([
                "⚖️ 风险适中：保持当前风险管理策略",
                "🔄 定期重新平衡投资组合",
                "📋 监控关键风险指标变化"
            ])
        else:
            recommendations.extend([
                "✅ 风险较低：可适度增加仓位",
                "📈 考虑增加成长性资产配置"
            ])
            
        # 基于具体指标的建议
        if basic_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("📊 夏普比率偏低，建议优化收益风险比")
            
        if abs(basic_metrics.get('max_drawdown', 0)) > 0.15:
            recommendations.append("📉 最大回撤过大，建议加强止损管理")
            
        if enhanced_metrics.get('extreme_event_frequency', 0) > 0.1:
            recommendations.append("⚡ 极端事件频率较高，建议增加防御性配置")
            
        return recommendations
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skewness
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        n = len(data)
        kurtosis = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurtosis + 3  # 返回超额峰度 + 3
        
    async def real_time_risk_monitoring(self, portfolio_data: Dict[str, Any],
                                      market_data: pd.DataFrame) -> Dict[str, Any]:
        """实时风险监控"""
        try:
            # 快速风险评估
            basic_metrics = self._calculate_basic_metrics(portfolio_data, market_data)
            
            # 检查风险阈值触发
            alerts = []
            current_var = abs(basic_metrics.get('var_1d', 0))
            
            for level, threshold in self.risk_thresholds.items():
                if current_var >= threshold:
                    alerts.append({
                        'level': level.name,
                        'message': f'VaR超过{level.name}级别阈值: {current_var:.4f} >= {threshold:.4f}',
                        'timestamp': datetime.now()
                    })
                    break
                    
            # 检查最大回撤
            max_dd = abs(basic_metrics.get('max_drawdown', 0))
            if max_dd > 0.1:
                alerts.append({
                    'level': 'HIGH',
                    'message': f'最大回撤过大: {max_dd:.4f}',
                    'timestamp': datetime.now()
                })
                
            return {
                'timestamp': datetime.now(),
                'basic_metrics': basic_metrics,
                'alerts': alerts,
                'risk_status': 'NORMAL' if not alerts else 'ALERT'
            }
            
        except Exception as e:
            self.logger.error(f"实时风险监控失败: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}