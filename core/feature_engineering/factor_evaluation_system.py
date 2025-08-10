"""
因子评估和选择系统
智能评估1000+量子因子的有效性，实现自适应因子选择和组合优化
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, ICA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from datetime import datetime, timedelta
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import itertools
from collections import defaultdict
import threading

from .quantum_factor_engine import QuantumFactorEngine, QuantumFactor, FactorCategory

warnings.filterwarnings('ignore')

class FactorEvaluationMethod(Enum):
    """因子评估方法"""
    IC_ANALYSIS = "ic_analysis"
    RANK_IC = "rank_ic"
    MUTUAL_INFORMATION = "mutual_information"
    REGRESSION_IMPORTANCE = "regression_importance"
    CAUSAL_INFERENCE = "causal_inference"
    ENSEMBLE_IMPORTANCE = "ensemble_importance"
    INFORMATION_RATIO = "information_ratio"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

class FactorSelectionMethod(Enum):
    """因子选择方法"""
    FORWARD_SELECTION = "forward_selection"
    BACKWARD_ELIMINATION = "backward_elimination"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"

@dataclass
class FactorEvaluationResult:
    """因子评估结果"""
    factor_id: str
    
    # IC分析
    mean_ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0  # 信息比率
    rank_ic: float = 0.0
    
    # 统计显著性
    ic_t_stat: float = 0.0
    ic_p_value: float = 1.0
    
    # 稳定性指标
    ic_stability: float = 0.0
    period_consistency: float = 0.0
    
    # 其他评估指标
    mutual_information: float = 0.0
    regression_importance: float = 0.0
    ensemble_importance: float = 0.0
    
    # 风险调整指标
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # 容量和衰减
    turnover_rate: float = 0.0
    capacity_estimate: float = 0.0
    decay_rate: float = 0.0
    
    # 元数据
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)
    periods_evaluated: int = 0
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """计算综合得分"""
        if weights is None:
            weights = {
                'ic_ir': 0.3,
                'ic_stability': 0.2,
                'sharpe_ratio': 0.2,
                'mutual_information': 0.1,
                'ensemble_importance': 0.1,
                'max_drawdown': -0.1  # 负权重，回撤越小越好
            }
        
        score = 0
        for metric, weight in weights.items():
            if hasattr(self, metric):
                value = getattr(self, metric)
                if not np.isnan(value) and not np.isinf(value):
                    score += weight * value
        
        return score

@dataclass
class FactorPortfolio:
    """因子投资组合"""
    factor_ids: List[str]
    weights: np.ndarray
    
    # 性能指标
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # 多样性指标
    correlation_matrix: Optional[np.ndarray] = None
    effective_factors: float = 0.0
    concentration_index: float = 0.0
    
    # 构建信息
    selection_method: Optional[FactorSelectionMethod] = None
    optimization_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_portfolio_info(self) -> Dict[str, Any]:
        """获取投资组合信息"""
        return {
            'num_factors': len(self.factor_ids),
            'weights': self.weights.tolist(),
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'effective_factors': self.effective_factors,
            'concentration_index': self.concentration_index
        }

class ICAnalyzer:
    """信息系数分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger("ICAnalyzer")
    
    def calculate_ic_series(self, factor_values: np.ndarray, returns: np.ndarray,
                           method: str = 'pearson') -> np.ndarray:
        """计算IC时间序列"""
        if len(factor_values) != len(returns):
            raise ValueError("因子值和收益率长度不匹配")
        
        ic_series = []
        
        for i in range(len(factor_values)):
            factor_cross_section = factor_values[i]
            return_cross_section = returns[i]
            
            # 移除NaN值
            valid_mask = ~(np.isnan(factor_cross_section) | np.isnan(return_cross_section))
            if np.sum(valid_mask) < 10:  # 需要足够的样本
                ic_series.append(0)
                continue
            
            factor_clean = factor_cross_section[valid_mask]
            return_clean = return_cross_section[valid_mask]
            
            # 计算相关系数
            if method == 'pearson':
                if np.std(factor_clean) > 0 and np.std(return_clean) > 0:
                    ic = np.corrcoef(factor_clean, return_clean)[0, 1]
                else:
                    ic = 0
            elif method == 'spearman':
                if len(factor_clean) > 1:
                    ic = stats.spearmanr(factor_clean, return_clean)[0]
                else:
                    ic = 0
            else:
                raise ValueError(f"不支持的IC计算方法: {method}")
            
            ic_series.append(ic if not np.isnan(ic) else 0)
        
        return np.array(ic_series)
    
    def analyze_ic_characteristics(self, ic_series: np.ndarray) -> Dict[str, float]:
        """分析IC特征"""
        ic_clean = ic_series[~np.isnan(ic_series)]
        
        if len(ic_clean) == 0:
            return {
                'mean_ic': 0, 'ic_std': 0, 'ic_ir': 0, 'ic_skew': 0,
                'ic_kurtosis': 0, 'positive_ic_ratio': 0, 'ic_t_stat': 0,
                'ic_p_value': 1.0
            }
        
        mean_ic = np.mean(ic_clean)
        ic_std = np.std(ic_clean)
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        
        # 统计特征
        ic_skew = stats.skew(ic_clean)
        ic_kurtosis = stats.kurtosis(ic_clean)
        positive_ic_ratio = np.sum(ic_clean > 0) / len(ic_clean)
        
        # 显著性检验
        if len(ic_clean) > 1:
            ic_t_stat, ic_p_value = stats.ttest_1samp(ic_clean, 0)
        else:
            ic_t_stat, ic_p_value = 0, 1.0
        
        return {
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_skew': ic_skew,
            'ic_kurtosis': ic_kurtosis,
            'positive_ic_ratio': positive_ic_ratio,
            'ic_t_stat': ic_t_stat,
            'ic_p_value': ic_p_value
        }
    
    def calculate_ic_stability(self, ic_series: np.ndarray, window_size: int = 60) -> float:
        """计算IC稳定性"""
        if len(ic_series) < window_size * 2:
            return 0
        
        # 滚动窗口计算IC均值
        rolling_ic_means = []
        for i in range(len(ic_series) - window_size + 1):
            window_ic = ic_series[i:i + window_size]
            rolling_ic_means.append(np.mean(window_ic))
        
        rolling_ic_means = np.array(rolling_ic_means)
        
        # 稳定性 = 1 - (滚动IC的标准差 / 绝对均值)
        if len(rolling_ic_means) > 1 and np.mean(np.abs(rolling_ic_means)) > 0:
            stability = 1 - (np.std(rolling_ic_means) / np.mean(np.abs(rolling_ic_means)))
            return max(0, stability)
        else:
            return 0

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, quantum_engine: QuantumFactorEngine):
        self.quantum_engine = quantum_engine
        self.ic_analyzer = ICAnalyzer()
        
        # 评估结果缓存
        self.evaluation_cache: Dict[str, FactorEvaluationResult] = {}
        self.cache_lock = threading.RLock()
        
        self.logger = logging.getLogger("FactorEvaluator")
    
    def evaluate_single_factor(self, factor_id: str, price_data: np.ndarray,
                              forward_returns: np.ndarray,
                              method: FactorEvaluationMethod = FactorEvaluationMethod.IC_ANALYSIS) -> FactorEvaluationResult:
        """评估单个因子"""
        
        try:
            # 计算因子值
            factor_values = self.quantum_engine.compute_factor(factor_id, price_data)
            
            if len(factor_values) != len(forward_returns):
                # 对齐数据长度
                min_length = min(len(factor_values), len(forward_returns))
                factor_values = factor_values[:min_length]
                forward_returns = forward_returns[:min_length]
            
            result = FactorEvaluationResult(factor_id=factor_id)
            
            if method == FactorEvaluationMethod.IC_ANALYSIS:
                self._evaluate_ic_analysis(result, factor_values, forward_returns)
            elif method == FactorEvaluationMethod.MUTUAL_INFORMATION:
                self._evaluate_mutual_information(result, factor_values, forward_returns)
            elif method == FactorEvaluationMethod.REGRESSION_IMPORTANCE:
                self._evaluate_regression_importance(result, factor_values, forward_returns)
            elif method == FactorEvaluationMethod.ENSEMBLE_IMPORTANCE:
                self._evaluate_ensemble_importance(result, factor_values, forward_returns)
            
            # 通用风险调整指标
            self._evaluate_risk_adjusted_metrics(result, factor_values, forward_returns)
            
            result.periods_evaluated = len(factor_values)
            
            # 缓存结果
            with self.cache_lock:
                self.evaluation_cache[factor_id] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"评估因子失败 {factor_id}: {e}")
            return FactorEvaluationResult(factor_id=factor_id)
    
    def _evaluate_ic_analysis(self, result: FactorEvaluationResult, 
                             factor_values: np.ndarray, forward_returns: np.ndarray):
        """IC分析评估"""
        # 横截面IC（假设factor_values和forward_returns都是时间序列）
        # 这里简化处理，假设每个时间点就是一个横截面
        
        # 计算滞后相关性作为IC代理
        ic_lags = []
        for lag in range(1, min(6, len(factor_values)//10)):  # 测试1-5期滞后
            if lag < len(factor_values):
                factor_lagged = factor_values[:-lag]
                returns_lead = forward_returns[lag:]
                
                if len(factor_lagged) > 10 and np.std(factor_lagged) > 0 and np.std(returns_lead) > 0:
                    ic = np.corrcoef(factor_lagged, returns_lead)[0, 1]
                    if not np.isnan(ic):
                        ic_lags.append(ic)
        
        if ic_lags:
            result.mean_ic = np.mean(ic_lags)
            result.ic_std = np.std(ic_lags)
            result.ic_ir = result.mean_ic / result.ic_std if result.ic_std > 0 else 0
            
            # t检验
            if len(ic_lags) > 1:
                result.ic_t_stat, result.ic_p_value = stats.ttest_1samp(ic_lags, 0)
            
            # 稳定性
            result.ic_stability = 1 - (result.ic_std / abs(result.mean_ic) if result.mean_ic != 0 else 0)
        
        # Rank IC
        if len(factor_values) > 10:
            try:
                result.rank_ic = stats.spearmanr(factor_values[:-1], forward_returns[1:])[0]
                if np.isnan(result.rank_ic):
                    result.rank_ic = 0
            except:
                result.rank_ic = 0
    
    def _evaluate_mutual_information(self, result: FactorEvaluationResult,
                                   factor_values: np.ndarray, forward_returns: np.ndarray):
        """互信息评估"""
        try:
            # 离散化数据
            factor_discrete = pd.qcut(factor_values, q=10, duplicates='drop', labels=False)
            returns_discrete = pd.qcut(forward_returns, q=10, duplicates='drop', labels=False)
            
            # 移除NaN
            valid_mask = ~(pd.isna(factor_discrete) | pd.isna(returns_discrete))
            if np.sum(valid_mask) > 20:
                mi = mutual_info_score(factor_discrete[valid_mask], returns_discrete[valid_mask])
                result.mutual_information = mi
            
        except Exception as e:
            self.logger.debug(f"计算互信息失败: {e}")
            result.mutual_information = 0
    
    def _evaluate_regression_importance(self, result: FactorEvaluationResult,
                                      factor_values: np.ndarray, forward_returns: np.ndarray):
        """回归重要性评估"""
        try:
            # 线性回归
            factor_values_2d = factor_values.reshape(-1, 1)
            reg = LinearRegression().fit(factor_values_2d, forward_returns)
            
            # R²作为重要性指标
            r2_score = reg.score(factor_values_2d, forward_returns)
            result.regression_importance = max(0, r2_score)
            
        except Exception as e:
            self.logger.debug(f"计算回归重要性失败: {e}")
            result.regression_importance = 0
    
    def _evaluate_ensemble_importance(self, result: FactorEvaluationResult,
                                    factor_values: np.ndarray, forward_returns: np.ndarray):
        """集成重要性评估"""
        try:
            # 随机森林特征重要性
            factor_values_2d = factor_values.reshape(-1, 1)
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(factor_values_2d, forward_returns)
            
            result.ensemble_importance = rf.feature_importances_[0]
            
        except Exception as e:
            self.logger.debug(f"计算集成重要性失败: {e}")
            result.ensemble_importance = 0
    
    def _evaluate_risk_adjusted_metrics(self, result: FactorEvaluationResult,
                                      factor_values: np.ndarray, forward_returns: np.ndarray):
        """风险调整指标评估"""
        try:
            # 构建简单的因子策略回报
            # 假设因子值高的时候做多，低的时候做空
            factor_signals = np.sign(factor_values - np.median(factor_values))
            strategy_returns = factor_signals[:-1] * forward_returns[1:]  # 滞后一期
            
            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                # Sharpe比率 (年化)
                mean_return = np.mean(strategy_returns)
                std_return = np.std(strategy_returns)
                result.sharpe_ratio = mean_return / std_return * np.sqrt(252)
                
                # 最大回撤
                cumulative_returns = np.cumprod(1 + strategy_returns)
                rolling_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                result.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # Calmar比率
                if result.max_drawdown < 0:
                    result.calmar_ratio = mean_return * 252 / abs(result.max_drawdown)
            
        except Exception as e:
            self.logger.debug(f"计算风险调整指标失败: {e}")
    
    def evaluate_factors_batch(self, factor_ids: List[str], price_data: np.ndarray,
                              forward_returns: np.ndarray,
                              n_workers: int = None) -> Dict[str, FactorEvaluationResult]:
        """批量评估因子"""
        if n_workers is None:
            n_workers = min(8, mp.cpu_count())
        
        results = {}
        
        if n_workers <= 1:
            for factor_id in factor_ids:
                results[factor_id] = self.evaluate_single_factor(factor_id, price_data, forward_returns)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_factor = {
                    executor.submit(self.evaluate_single_factor, factor_id, price_data, forward_returns): factor_id
                    for factor_id in factor_ids
                }
                
                for future in future_to_factor:
                    factor_id = future_to_factor[future]
                    try:
                        results[factor_id] = future.result()
                    except Exception as e:
                        self.logger.error(f"批量评估因子失败 {factor_id}: {e}")
                        results[factor_id] = FactorEvaluationResult(factor_id=factor_id)
        
        return results
    
    def rank_factors(self, evaluation_results: Dict[str, FactorEvaluationResult],
                    ranking_method: str = 'composite_score',
                    top_k: int = None) -> List[Tuple[str, float]]:
        """因子排序"""
        scores = []
        
        for factor_id, result in evaluation_results.items():
            if ranking_method == 'composite_score':
                score = result.get_composite_score()
            elif ranking_method == 'ic_ir':
                score = result.ic_ir
            elif ranking_method == 'sharpe_ratio':
                score = result.sharpe_ratio
            elif ranking_method == 'mutual_information':
                score = result.mutual_information
            else:
                score = result.get_composite_score()
            
            scores.append((factor_id, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scores = scores[:top_k]
        
        return scores

class FactorSelector:
    """因子选择器"""
    
    def __init__(self, factor_evaluator: FactorEvaluator):
        self.factor_evaluator = factor_evaluator
        self.logger = logging.getLogger("FactorSelector")
    
    def select_factors(self, factor_ids: List[str], price_data: np.ndarray,
                      forward_returns: np.ndarray, max_factors: int = 20,
                      method: FactorSelectionMethod = FactorSelectionMethod.FORWARD_SELECTION) -> FactorPortfolio:
        """选择因子组合"""
        
        # 首先评估所有因子
        self.logger.info(f"评估 {len(factor_ids)} 个候选因子...")
        evaluation_results = self.factor_evaluator.evaluate_factors_batch(
            factor_ids, price_data, forward_returns
        )
        
        # 预筛选：移除明显无效的因子
        valid_factors = []
        for factor_id, result in evaluation_results.items():
            if (abs(result.mean_ic) > 0.01 or 
                abs(result.sharpe_ratio) > 0.1 or 
                result.mutual_information > 0.001):
                valid_factors.append(factor_id)
        
        self.logger.info(f"预筛选后剩余 {len(valid_factors)} 个有效因子")
        
        if len(valid_factors) == 0:
            return FactorPortfolio(factor_ids=[], weights=np.array([]))
        
        # 应用选择方法
        if method == FactorSelectionMethod.FORWARD_SELECTION:
            selected_factors = self._forward_selection(valid_factors, price_data, forward_returns, max_factors)
        elif method == FactorSelectionMethod.GENETIC_ALGORITHM:
            selected_factors = self._genetic_algorithm_selection(valid_factors, price_data, forward_returns, max_factors)
        else:
            # 默认使用前向选择
            selected_factors = self._forward_selection(valid_factors, price_data, forward_returns, max_factors)
        
        if not selected_factors:
            return FactorPortfolio(factor_ids=[], weights=np.array([]))
        
        # 优化权重
        weights = self._optimize_weights(selected_factors, price_data, forward_returns)
        
        # 创建投资组合
        portfolio = FactorPortfolio(
            factor_ids=selected_factors,
            weights=weights,
            selection_method=method
        )
        
        # 计算投资组合性能
        self._evaluate_portfolio_performance(portfolio, price_data, forward_returns)
        
        return portfolio
    
    def _forward_selection(self, factor_ids: List[str], price_data: np.ndarray,
                          forward_returns: np.ndarray, max_factors: int) -> List[str]:
        """前向选择算法"""
        selected_factors = []
        remaining_factors = factor_ids.copy()
        
        for i in range(max_factors):
            if not remaining_factors:
                break
            
            best_factor = None
            best_score = float('-inf')
            
            for candidate in remaining_factors:
                # 临时添加候选因子
                test_factors = selected_factors + [candidate]
                
                # 评估组合性能
                score = self._evaluate_factor_combination(test_factors, price_data, forward_returns)
                
                if score > best_score:
                    best_score = score
                    best_factor = candidate
            
            if best_factor and best_score > 0:
                selected_factors.append(best_factor)
                remaining_factors.remove(best_factor)
                self.logger.info(f"选择因子 {i+1}: {best_factor}, 得分: {best_score:.4f}")
            else:
                break
        
        return selected_factors
    
    def _genetic_algorithm_selection(self, factor_ids: List[str], price_data: np.ndarray,
                                   forward_returns: np.ndarray, max_factors: int) -> List[str]:
        """遗传算法选择"""
        
        def objective(x):
            # x是二进制向量，表示是否选择对应因子
            selected_indices = np.where(x > 0.5)[0]
            
            if len(selected_indices) == 0 or len(selected_indices) > max_factors:
                return -999999  # 惩罚无效解
            
            selected_factors = [factor_ids[i] for i in selected_indices]
            score = self._evaluate_factor_combination(selected_factors, price_data, forward_returns)
            
            # 加入正则化项，惩罚过多因子
            regularization = -0.01 * len(selected_indices)
            
            return score + regularization
        
        # 遗传算法优化
        bounds = [(0, 1) for _ in range(len(factor_ids))]
        
        try:
            result = differential_evolution(
                lambda x: -objective(x),  # 最小化负值
                bounds,
                maxiter=50,
                popsize=15,
                seed=42
            )
            
            # 解码结果
            selected_indices = np.where(result.x > 0.5)[0]
            selected_factors = [factor_ids[i] for i in selected_indices[:max_factors]]
            
            return selected_factors
            
        except Exception as e:
            self.logger.error(f"遗传算法选择失败: {e}")
            # 回退到简单选择
            return factor_ids[:max_factors]
    
    def _evaluate_factor_combination(self, factor_ids: List[str], price_data: np.ndarray,
                                   forward_returns: np.ndarray) -> float:
        """评估因子组合"""
        if not factor_ids:
            return 0
        
        try:
            # 计算所有因子值
            factor_matrix = np.column_stack([
                self.factor_evaluator.quantum_engine.compute_factor(factor_id, price_data)
                for factor_id in factor_ids
            ])
            
            # 标准化
            scaler = StandardScaler()
            factor_matrix_scaled = scaler.fit_transform(factor_matrix)
            
            # 简单等权重组合
            combined_factor = np.mean(factor_matrix_scaled, axis=1)
            
            # 计算IC
            if len(combined_factor) > len(forward_returns):
                combined_factor = combined_factor[:len(forward_returns)]
            elif len(forward_returns) > len(combined_factor):
                forward_returns = forward_returns[:len(combined_factor)]
            
            # 滞后一期计算相关性
            if len(combined_factor) > 1:
                ic = np.corrcoef(combined_factor[:-1], forward_returns[1:])[0, 1]
                if np.isnan(ic):
                    ic = 0
            else:
                ic = 0
            
            # 计算信息比率
            ic_series = []
            window_size = min(60, len(combined_factor) // 4)
            
            if window_size > 10:
                for i in range(window_size, len(combined_factor)):
                    window_factor = combined_factor[i-window_size:i]
                    window_return = forward_returns[i-window_size:i]
                    
                    if np.std(window_factor) > 0 and np.std(window_return) > 0:
                        window_ic = np.corrcoef(window_factor, window_return)[0, 1]
                        if not np.isnan(window_ic):
                            ic_series.append(window_ic)
                
                if ic_series:
                    ic_mean = np.mean(ic_series)
                    ic_std = np.std(ic_series)
                    ir = ic_mean / ic_std if ic_std > 0 else 0
                else:
                    ir = 0
            else:
                ir = 0
            
            # 综合得分：IC + 信息比率
            score = abs(ic) * 0.6 + abs(ir) * 0.4
            
            return score
            
        except Exception as e:
            self.logger.debug(f"评估因子组合失败: {e}")
            return 0
    
    def _optimize_weights(self, factor_ids: List[str], price_data: np.ndarray,
                         forward_returns: np.ndarray) -> np.ndarray:
        """优化因子权重"""
        if not factor_ids:
            return np.array([])
        
        try:
            # 计算因子矩阵
            factor_matrix = np.column_stack([
                self.factor_evaluator.quantum_engine.compute_factor(factor_id, price_data)
                for factor_id in factor_ids
            ])
            
            # 标准化
            scaler = StandardScaler()
            factor_matrix_scaled = scaler.fit_transform(factor_matrix)
            
            # 对齐长度
            min_length = min(len(factor_matrix_scaled), len(forward_returns))
            factor_matrix_scaled = factor_matrix_scaled[:min_length]
            forward_returns_aligned = forward_returns[:min_length]
            
            # 使用岭回归优化权重
            ridge = Ridge(alpha=0.1)
            ridge.fit(factor_matrix_scaled[:-1], forward_returns_aligned[1:])  # 滞后一期
            
            weights = ridge.coef_
            
            # 归一化权重
            weights = weights / np.sum(np.abs(weights)) if np.sum(np.abs(weights)) > 0 else np.ones(len(weights)) / len(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"优化权重失败: {e}")
            # 返回等权重
            return np.ones(len(factor_ids)) / len(factor_ids)
    
    def _evaluate_portfolio_performance(self, portfolio: FactorPortfolio, 
                                      price_data: np.ndarray, forward_returns: np.ndarray):
        """评估投资组合性能"""
        try:
            if not portfolio.factor_ids:
                return
            
            # 计算组合因子值
            factor_matrix = np.column_stack([
                self.factor_evaluator.quantum_engine.compute_factor(factor_id, price_data)
                for factor_id in portfolio.factor_ids
            ])
            
            # 标准化
            scaler = StandardScaler()
            factor_matrix_scaled = scaler.fit_transform(factor_matrix)
            
            # 加权组合
            combined_factor = np.dot(factor_matrix_scaled, portfolio.weights)
            
            # 构建策略回报
            factor_signals = np.sign(combined_factor - np.median(combined_factor))
            strategy_returns = factor_signals[:-1] * forward_returns[1:]  # 滞后一期
            
            if len(strategy_returns) > 0:
                # 总回报
                portfolio.total_return = np.prod(1 + strategy_returns) - 1
                
                # Sharpe比率
                if np.std(strategy_returns) > 0:
                    portfolio.sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                
                # 最大回撤
                cumulative_returns = np.cumprod(1 + strategy_returns)
                rolling_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                portfolio.max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # 胜率
                portfolio.win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
            
            # 多样性指标
            if len(factor_matrix_scaled) > 1:
                portfolio.correlation_matrix = np.corrcoef(factor_matrix_scaled.T)
                
                # 有效因子数 (基于相关性)
                eigenvals = np.linalg.eigvals(portfolio.correlation_matrix)
                eigenvals = eigenvals[eigenvals > 0]
                if len(eigenvals) > 0:
                    portfolio.effective_factors = np.exp(-np.sum(eigenvals * np.log(eigenvals + 1e-10)))
                
                # 集中度指数 (权重的Herfindahl指数)
                portfolio.concentration_index = np.sum(portfolio.weights**2)
            
        except Exception as e:
            self.logger.error(f"评估投资组合性能失败: {e}")

class FactorEvaluationSystem:
    """因子评估选择系统"""
    
    def __init__(self, quantum_engine: QuantumFactorEngine):
        self.quantum_engine = quantum_engine
        self.factor_evaluator = FactorEvaluator(quantum_engine)
        self.factor_selector = FactorSelector(self.factor_evaluator)
        
        # 历史记录
        self.evaluation_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[FactorPortfolio] = []
        
        self.logger = logging.getLogger("FactorEvaluationSystem")
    
    def run_full_evaluation(self, price_data: np.ndarray, forward_returns: np.ndarray,
                           factor_categories: List[FactorCategory] = None,
                           max_factors_per_category: int = 50,
                           final_portfolio_size: int = 20) -> Dict[str, Any]:
        """运行完整的因子评估和选择流程"""
        
        self.logger.info("🚀 开始运行完整的1000+因子评估和选择流程...")
        
        start_time = datetime.utcnow()
        
        # 1. 获取候选因子
        if factor_categories is None:
            factor_categories = list(FactorCategory)
        
        all_candidate_factors = []
        for category in factor_categories:
            category_factors = self.quantum_engine.get_factors_by_category(category)
            # 限制每个类别的因子数量
            if len(category_factors) > max_factors_per_category:
                # 随机采样或者按某种策略选择
                category_factors = category_factors[:max_factors_per_category]
            all_candidate_factors.extend(category_factors)
        
        self.logger.info(f"📊 候选因子总数: {len(all_candidate_factors)}")
        
        # 2. 批量评估所有因子
        self.logger.info("⚡ 开始批量评估因子有效性...")
        evaluation_results = self.factor_evaluator.evaluate_factors_batch(
            all_candidate_factors, price_data, forward_returns
        )
        
        # 3. 因子排序
        factor_rankings = self.factor_evaluator.rank_factors(
            evaluation_results, ranking_method='composite_score', top_k=200  # 取前200个
        )
        
        self.logger.info(f"🏆 Top 10 因子排名:")
        for i, (factor_id, score) in enumerate(factor_rankings[:10]):
            self.logger.info(f"  {i+1}. {factor_id}: {score:.4f}")
        
        # 4. 因子选择和组合优化
        self.logger.info("🎯 开始因子选择和组合优化...")
        
        top_factors = [factor_id for factor_id, _ in factor_rankings[:100]]  # 从前100中选择
        
        # 尝试多种选择方法
        portfolios = {}
        
        for method in [FactorSelectionMethod.FORWARD_SELECTION, 
                      FactorSelectionMethod.GENETIC_ALGORITHM]:
            try:
                self.logger.info(f"使用 {method.value} 方法构建投资组合...")
                portfolio = self.factor_selector.select_factors(
                    top_factors, price_data, forward_returns, 
                    max_factors=final_portfolio_size, method=method
                )
                portfolios[method.value] = portfolio
            except Exception as e:
                self.logger.error(f"使用 {method.value} 方法失败: {e}")
        
        # 5. 选择最佳投资组合
        best_portfolio = None
        best_score = float('-inf')
        
        for method_name, portfolio in portfolios.items():
            if portfolio.factor_ids:
                score = portfolio.sharpe_ratio * 0.5 + abs(portfolio.total_return) * 0.3 - abs(portfolio.max_drawdown) * 0.2
                self.logger.info(f"{method_name} 组合得分: {score:.4f} (因子数: {len(portfolio.factor_ids)})")
                
                if score > best_score:
                    best_score = score
                    best_portfolio = portfolio
        
        # 6. 生成评估报告
        evaluation_time = (datetime.utcnow() - start_time).total_seconds()
        
        report = {
            'evaluation_timestamp': start_time,
            'evaluation_time_seconds': evaluation_time,
            'total_factors_evaluated': len(all_candidate_factors),
            'valid_factors_found': len([r for r in evaluation_results.values() if r.get_composite_score() > 0]),
            'top_factors': factor_rankings[:20],
            'best_portfolio': best_portfolio.get_portfolio_info() if best_portfolio else None,
            'best_portfolio_factors': best_portfolio.factor_ids if best_portfolio else [],
            'portfolio_comparison': {
                method: portfolio.get_portfolio_info() 
                for method, portfolio in portfolios.items() if portfolio.factor_ids
            },
            'category_distribution': self._analyze_category_distribution(evaluation_results),
            'performance_summary': self._generate_performance_summary(evaluation_results, best_portfolio)
        }
        
        # 7. 记录历史
        self.evaluation_history.append(report)
        if best_portfolio:
            self.portfolio_history.append(best_portfolio)
        
        self.logger.info("✅ 完整评估流程已完成!")
        self.logger.info(f"⏱️  总耗时: {evaluation_time:.2f} 秒")
        self.logger.info(f"🎯 最佳投资组合包含 {len(best_portfolio.factor_ids) if best_portfolio else 0} 个因子")
        self.logger.info(f"📈 最佳组合Sharpe比率: {best_portfolio.sharpe_ratio:.4f}" if best_portfolio else "未找到有效组合")
        
        return report
    
    def _analyze_category_distribution(self, evaluation_results: Dict[str, FactorEvaluationResult]) -> Dict[str, Any]:
        """分析因子类别分布"""
        category_stats = defaultdict(list)
        
        for factor_id, result in evaluation_results.items():
            factor_info = self.quantum_engine.get_factor_info(factor_id)
            if factor_info:
                category_stats[factor_info.category.value].append(result.get_composite_score())
        
        category_summary = {}
        for category, scores in category_stats.items():
            category_summary[category] = {
                'count': len(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'max_score': np.max(scores),
                'top_10_ratio': np.sum(np.array(scores) > np.percentile(list(evaluation_results.values()), 90)) / len(scores)
            }
        
        return category_summary
    
    def _generate_performance_summary(self, evaluation_results: Dict[str, FactorEvaluationResult],
                                    best_portfolio: FactorPortfolio) -> Dict[str, Any]:
        """生成性能摘要"""
        all_scores = [result.get_composite_score() for result in evaluation_results.values()]
        all_ics = [result.mean_ic for result in evaluation_results.values()]
        all_sharpes = [result.sharpe_ratio for result in evaluation_results.values()]
        
        summary = {
            'factor_score_distribution': {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'percentiles': {
                    '50': np.percentile(all_scores, 50),
                    '75': np.percentile(all_scores, 75),
                    '90': np.percentile(all_scores, 90),
                    '95': np.percentile(all_scores, 95)
                }
            },
            'ic_distribution': {
                'mean': np.mean([ic for ic in all_ics if not np.isnan(ic)]),
                'std': np.std([ic for ic in all_ics if not np.isnan(ic)]),
                'positive_ratio': np.sum(np.array(all_ics) > 0) / len(all_ics)
            },
            'portfolio_performance': {
                'sharpe_ratio': best_portfolio.sharpe_ratio if best_portfolio else 0,
                'total_return': best_portfolio.total_return if best_portfolio else 0,
                'max_drawdown': best_portfolio.max_drawdown if best_portfolio else 0,
                'effective_factors': best_portfolio.effective_factors if best_portfolio else 0,
                'concentration_index': best_portfolio.concentration_index if best_portfolio else 0
            }
        }
        
        return summary
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """获取评估历史"""
        return self.evaluation_history
    
    def get_best_portfolio(self) -> Optional[FactorPortfolio]:
        """获取最佳投资组合"""
        if self.portfolio_history:
            return max(self.portfolio_history, key=lambda p: p.sharpe_ratio)
        return None
    
    def export_evaluation_report(self, output_path: str) -> bool:
        """导出评估报告"""
        try:
            report_data = {
                'evaluation_history': self.evaluation_history,
                'portfolio_history': [p.get_portfolio_info() for p in self.portfolio_history],
                'system_info': {
                    'total_factors': len(self.quantum_engine.factor_registry),
                    'factor_statistics': self.quantum_engine.get_factor_statistics(),
                    'export_timestamp': datetime.utcnow().isoformat()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"评估报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出评估报告失败: {e}")
            return False

# 全局因子评估系统实例
def create_factor_evaluation_system():
    """创建因子评估系统实例"""
    from .quantum_factor_engine import quantum_factor_engine
    return FactorEvaluationSystem(quantum_factor_engine)

# 延迟初始化，避免循环导入
_global_evaluation_system = None

def get_global_evaluation_system():
    """获取全局评估系统实例"""
    global _global_evaluation_system
    if _global_evaluation_system is None:
        _global_evaluation_system = create_factor_evaluation_system()
    return _global_evaluation_system