"""
CTBench生成模型评估系统
用于评估时间序列生成模型的质量和真实性
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import logging
import warnings
from datetime import datetime, timedelta
import json
import os

warnings.filterwarnings('ignore')

@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 统计相似性
    statistical_similarity: Dict[str, float]
    
    # 分布相似性
    distribution_similarity: Dict[str, float]
    
    # 时间序列特征
    temporal_features: Dict[str, float]
    
    # 金融特征
    financial_features: Dict[str, float]
    
    # 判别器得分
    discriminative_score: float
    
    # 多样性指标
    diversity_metrics: Dict[str, float]
    
    # 综合质量得分
    overall_quality_score: float
    
    # 评估时间戳
    evaluation_timestamp: datetime

class StatisticalEvaluator:
    """统计评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger("StatisticalEvaluator")
    
    def evaluate_statistical_similarity(self, real_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> Dict[str, float]:
        """评估统计相似性"""
        metrics = {}
        
        try:
            # 基本统计量比较
            metrics.update(self._compare_basic_statistics(real_data, synthetic_data))
            
            # 分布检验
            metrics.update(self._distribution_tests(real_data, synthetic_data))
            
            # 相关性比较
            metrics.update(self._correlation_comparison(real_data, synthetic_data))
            
        except Exception as e:
            self.logger.error(f"统计相似性评估失败: {e}")
            
        return metrics
    
    def _compare_basic_statistics(self, real_data: np.ndarray, 
                                 synthetic_data: np.ndarray) -> Dict[str, float]:
        """比较基本统计量"""
        metrics = {}
        
        # 平均值差异
        real_mean = np.mean(real_data, axis=(0, 1))
        synthetic_mean = np.mean(synthetic_data, axis=(0, 1))
        mean_diff = np.mean(np.abs(real_mean - synthetic_mean))
        metrics["mean_difference"] = mean_diff
        
        # 标准差差异
        real_std = np.std(real_data, axis=(0, 1))
        synthetic_std = np.std(synthetic_data, axis=(0, 1))
        std_diff = np.mean(np.abs(real_std - synthetic_std))
        metrics["std_difference"] = std_diff
        
        # 偏度差异
        real_skew = stats.skew(real_data.reshape(-1, real_data.shape[-1]))
        synthetic_skew = stats.skew(synthetic_data.reshape(-1, synthetic_data.shape[-1]))
        skew_diff = np.mean(np.abs(real_skew - synthetic_skew))
        metrics["skewness_difference"] = skew_diff
        
        # 峰度差异
        real_kurtosis = stats.kurtosis(real_data.reshape(-1, real_data.shape[-1]))
        synthetic_kurtosis = stats.kurtosis(synthetic_data.reshape(-1, synthetic_data.shape[-1]))
        kurtosis_diff = np.mean(np.abs(real_kurtosis - synthetic_kurtosis))
        metrics["kurtosis_difference"] = kurtosis_diff
        
        return metrics
    
    def _distribution_tests(self, real_data: np.ndarray, 
                           synthetic_data: np.ndarray) -> Dict[str, float]:
        """分布检验"""
        metrics = {}
        
        # KS检验
        ks_scores = []
        for feature_idx in range(real_data.shape[-1]):
            real_feature = real_data[:, :, feature_idx].flatten()
            synthetic_feature = synthetic_data[:, :, feature_idx].flatten()
            
            ks_statistic, ks_pvalue = stats.ks_2samp(real_feature, synthetic_feature)
            ks_scores.append(ks_statistic)
        
        metrics["ks_test_score"] = np.mean(ks_scores)
        
        # Wasserstein距离
        wasserstein_distances = []
        for feature_idx in range(real_data.shape[-1]):
            real_feature = real_data[:, :, feature_idx].flatten()
            synthetic_feature = synthetic_data[:, :, feature_idx].flatten()
            
            wasserstein_dist = stats.wasserstein_distance(real_feature, synthetic_feature)
            wasserstein_distances.append(wasserstein_dist)
        
        metrics["wasserstein_distance"] = np.mean(wasserstein_distances)
        
        return metrics
    
    def _correlation_comparison(self, real_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> Dict[str, float]:
        """相关性比较"""
        # 重塑数据为 (samples, features)
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
        
        # 计算相关矩阵
        real_corr = np.corrcoef(real_flat.T)
        synthetic_corr = np.corrcoef(synthetic_flat.T)
        
        # 相关矩阵差异
        corr_diff = np.mean(np.abs(real_corr - synthetic_corr))
        
        return {"correlation_difference": corr_diff}

class TemporalEvaluator:
    """时间序列特征评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger("TemporalEvaluator")
    
    def evaluate_temporal_features(self, real_data: np.ndarray, 
                                  synthetic_data: np.ndarray) -> Dict[str, float]:
        """评估时间序列特征"""
        metrics = {}
        
        try:
            # 自相关性分析
            metrics.update(self._autocorrelation_analysis(real_data, synthetic_data))
            
            # 趋势分析
            metrics.update(self._trend_analysis(real_data, synthetic_data))
            
            # 波动性分析
            metrics.update(self._volatility_analysis(real_data, synthetic_data))
            
            # 长期记忆性分析
            metrics.update(self._long_memory_analysis(real_data, synthetic_data))
            
        except Exception as e:
            self.logger.error(f"时间序列特征评估失败: {e}")
            
        return metrics
    
    def _autocorrelation_analysis(self, real_data: np.ndarray, 
                                 synthetic_data: np.ndarray) -> Dict[str, float]:
        """自相关性分析"""
        def compute_autocorr(data, max_lag=20):
            """计算自相关函数"""
            autocorrs = []
            for sample in data:
                for feature_idx in range(sample.shape[1]):
                    series = sample[:, feature_idx]
                    corrs = [np.corrcoef(series[:-lag], series[lag:])[0, 1] 
                            for lag in range(1, min(max_lag + 1, len(series)))]
                    autocorrs.extend(corrs)
            return np.array(autocorrs)
        
        real_autocorr = compute_autocorr(real_data)
        synthetic_autocorr = compute_autocorr(synthetic_data)
        
        # 移除NaN值
        real_autocorr = real_autocorr[~np.isnan(real_autocorr)]
        synthetic_autocorr = synthetic_autocorr[~np.isnan(synthetic_autocorr)]
        
        if len(real_autocorr) > 0 and len(synthetic_autocorr) > 0:
            autocorr_diff = np.mean(np.abs(real_autocorr.mean() - synthetic_autocorr.mean()))
        else:
            autocorr_diff = 0
        
        return {"autocorrelation_difference": autocorr_diff}
    
    def _trend_analysis(self, real_data: np.ndarray, 
                       synthetic_data: np.ndarray) -> Dict[str, float]:
        """趋势分析"""
        def compute_trend_strength(data):
            """计算趋势强度"""
            trends = []
            for sample in data:
                for feature_idx in range(sample.shape[1]):
                    series = sample[:, feature_idx]
                    # 线性回归斜率作为趋势强度
                    x = np.arange(len(series))
                    slope, _ = np.polyfit(x, series, 1)
                    trends.append(abs(slope))
            return np.mean(trends)
        
        real_trend = compute_trend_strength(real_data)
        synthetic_trend = compute_trend_strength(synthetic_data)
        
        trend_diff = abs(real_trend - synthetic_trend)
        
        return {"trend_difference": trend_diff}
    
    def _volatility_analysis(self, real_data: np.ndarray, 
                            synthetic_data: np.ndarray) -> Dict[str, float]:
        """波动性分析"""
        def compute_volatility_clustering(data):
            """计算波动性聚集性"""
            volatilities = []
            for sample in data:
                for feature_idx in range(sample.shape[1]):
                    series = sample[:, feature_idx]
                    returns = np.diff(series) / series[:-1]
                    volatility = np.std(returns)
                    volatilities.append(volatility)
            return np.mean(volatilities)
        
        real_vol = compute_volatility_clustering(real_data)
        synthetic_vol = compute_volatility_clustering(synthetic_data)
        
        vol_diff = abs(real_vol - synthetic_vol)
        
        return {"volatility_difference": vol_diff}
    
    def _long_memory_analysis(self, real_data: np.ndarray, 
                             synthetic_data: np.ndarray) -> Dict[str, float]:
        """长期记忆性分析 (Hurst指数)"""
        def hurst_exponent(ts):
            """计算Hurst指数"""
            try:
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            except:
                return 0.5  # 默认值
        
        real_hurst = []
        synthetic_hurst = []
        
        for sample in real_data:
            for feature_idx in range(sample.shape[1]):
                series = sample[:, feature_idx]
                hurst = hurst_exponent(series)
                if not np.isnan(hurst):
                    real_hurst.append(hurst)
        
        for sample in synthetic_data:
            for feature_idx in range(sample.shape[1]):
                series = sample[:, feature_idx]
                hurst = hurst_exponent(series)
                if not np.isnan(hurst):
                    synthetic_hurst.append(hurst)
        
        if len(real_hurst) > 0 and len(synthetic_hurst) > 0:
            hurst_diff = abs(np.mean(real_hurst) - np.mean(synthetic_hurst))
        else:
            hurst_diff = 0
        
        return {"hurst_difference": hurst_diff}

class FinancialEvaluator:
    """金融特征评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger("FinancialEvaluator")
    
    def evaluate_financial_features(self, real_data: np.ndarray, 
                                   synthetic_data: np.ndarray) -> Dict[str, float]:
        """评估金融特征"""
        metrics = {}
        
        try:
            # 假设数据格式为 (batch, time, features) 其中 features = [Open, High, Low, Close, Volume]
            
            # 收益率分析
            metrics.update(self._return_analysis(real_data, synthetic_data))
            
            # 波动性分析
            metrics.update(self._financial_volatility_analysis(real_data, synthetic_data))
            
            # 技术指标分析
            metrics.update(self._technical_indicators_analysis(real_data, synthetic_data))
            
            # 极值事件分析
            metrics.update(self._extreme_events_analysis(real_data, synthetic_data))
            
        except Exception as e:
            self.logger.error(f"金融特征评估失败: {e}")
            
        return metrics
    
    def _return_analysis(self, real_data: np.ndarray, 
                        synthetic_data: np.ndarray) -> Dict[str, float]:
        """收益率分析"""
        def compute_returns(data, price_idx=3):  # 假设Close价格在索引3
            """计算收益率"""
            returns = []
            for sample in data:
                prices = sample[:, price_idx]
                sample_returns = np.diff(prices) / prices[:-1]
                returns.extend(sample_returns)
            return np.array(returns)
        
        real_returns = compute_returns(real_data)
        synthetic_returns = compute_returns(synthetic_data)
        
        # 移除异常值
        real_returns = real_returns[np.isfinite(real_returns)]
        synthetic_returns = synthetic_returns[np.isfinite(synthetic_returns)]
        
        if len(real_returns) > 0 and len(synthetic_returns) > 0:
            # 收益率分布相似性
            return_mean_diff = abs(np.mean(real_returns) - np.mean(synthetic_returns))
            return_std_diff = abs(np.std(real_returns) - np.std(synthetic_returns))
            
            # 收益率偏度和峰度
            real_skew = stats.skew(real_returns)
            synthetic_skew = stats.skew(synthetic_returns)
            skew_diff = abs(real_skew - synthetic_skew)
            
            real_kurtosis = stats.kurtosis(real_returns)
            synthetic_kurtosis = stats.kurtosis(synthetic_returns)
            kurtosis_diff = abs(real_kurtosis - synthetic_kurtosis)
        else:
            return_mean_diff = return_std_diff = skew_diff = kurtosis_diff = 0
        
        return {
            "return_mean_difference": return_mean_diff,
            "return_std_difference": return_std_diff,
            "return_skew_difference": skew_diff,
            "return_kurtosis_difference": kurtosis_diff
        }
    
    def _financial_volatility_analysis(self, real_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> Dict[str, float]:
        """金融波动性分析"""
        def compute_volatility_features(data, price_idx=3):
            """计算波动性特征"""
            volatilities = []
            for sample in data:
                prices = sample[:, price_idx]
                returns = np.diff(prices) / prices[:-1]
                returns = returns[np.isfinite(returns)]
                if len(returns) > 1:
                    vol = np.std(returns) * np.sqrt(252)  # 年化波动率
                    volatilities.append(vol)
            return volatilities
        
        real_vol = compute_volatility_features(real_data)
        synthetic_vol = compute_volatility_features(synthetic_data)
        
        if len(real_vol) > 0 and len(synthetic_vol) > 0:
            vol_diff = abs(np.mean(real_vol) - np.mean(synthetic_vol))
        else:
            vol_diff = 0
        
        return {"volatility_difference": vol_diff}
    
    def _technical_indicators_analysis(self, real_data: np.ndarray, 
                                      synthetic_data: np.ndarray) -> Dict[str, float]:
        """技术指标分析"""
        def compute_rsi(prices, window=14):
            """计算RSI"""
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            if len(gains) < window:
                return 50  # 默认中性值
            
            avg_gain = np.mean(gains[-window:])
            avg_loss = np.mean(losses[-window:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def compute_technical_features(data, price_idx=3):
            """计算技术指标特征"""
            rsi_values = []
            for sample in data:
                prices = sample[:, price_idx]
                rsi = compute_rsi(prices)
                if not np.isnan(rsi):
                    rsi_values.append(rsi)
            return rsi_values
        
        real_rsi = compute_technical_features(real_data)
        synthetic_rsi = compute_technical_features(synthetic_data)
        
        if len(real_rsi) > 0 and len(synthetic_rsi) > 0:
            rsi_diff = abs(np.mean(real_rsi) - np.mean(synthetic_rsi))
        else:
            rsi_diff = 0
        
        return {"rsi_difference": rsi_diff}
    
    def _extreme_events_analysis(self, real_data: np.ndarray, 
                                synthetic_data: np.ndarray) -> Dict[str, float]:
        """极值事件分析"""
        def compute_extreme_events(data, price_idx=3, threshold_percentile=95):
            """计算极值事件频率"""
            all_returns = []
            for sample in data:
                prices = sample[:, price_idx]
                returns = np.diff(prices) / prices[:-1]
                returns = returns[np.isfinite(returns)]
                all_returns.extend(returns)
            
            all_returns = np.array(all_returns)
            if len(all_returns) == 0:
                return 0
            
            threshold = np.percentile(np.abs(all_returns), threshold_percentile)
            extreme_events = np.sum(np.abs(all_returns) > threshold)
            return extreme_events / len(all_returns)
        
        real_extreme_freq = compute_extreme_events(real_data)
        synthetic_extreme_freq = compute_extreme_events(synthetic_data)
        
        extreme_diff = abs(real_extreme_freq - synthetic_extreme_freq)
        
        return {"extreme_events_difference": extreme_diff}

class DiscriminativeEvaluator:
    """判别性评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger("DiscriminativeEvaluator")
        self.classifier = None
    
    def evaluate_discriminative_score(self, real_data: np.ndarray, 
                                     synthetic_data: np.ndarray) -> float:
        """评估判别得分"""
        try:
            # 准备数据
            X, y = self._prepare_discriminative_data(real_data, synthetic_data)
            
            # 训练判别器
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X, y)
            
            # 计算准确率 (越接近0.5越好，说明难以区分)
            accuracy = self.classifier.score(X, y)
            
            # 转换为质量得分 (1 - |accuracy - 0.5| * 2)
            discriminative_score = 1 - abs(accuracy - 0.5) * 2
            
            return discriminative_score
            
        except Exception as e:
            self.logger.error(f"判别性评估失败: {e}")
            return 0.0
    
    def _prepare_discriminative_data(self, real_data: np.ndarray, 
                                    synthetic_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """准备判别性评估数据"""
        # 提取特征 (简化版本，使用统计特征)
        real_features = self._extract_features(real_data)
        synthetic_features = self._extract_features(synthetic_data)
        
        # 合并数据
        X = np.vstack([real_features, synthetic_features])
        y = np.hstack([np.ones(len(real_features)), np.zeros(len(synthetic_features))])
        
        return X, y
    
    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """提取特征"""
        features = []
        
        for sample in data:
            sample_features = []
            
            # 基本统计特征
            sample_features.extend([
                np.mean(sample),
                np.std(sample),
                stats.skew(sample.flatten()),
                stats.kurtosis(sample.flatten())
            ])
            
            # 时间序列特征
            for feature_idx in range(sample.shape[1]):
                series = sample[:, feature_idx]
                
                # 自相关
                if len(series) > 1:
                    autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0
                else:
                    autocorr = 0
                sample_features.append(autocorr)
                
                # 趋势
                if len(series) > 1:
                    x = np.arange(len(series))
                    slope, _ = np.polyfit(x, series, 1)
                else:
                    slope = 0
                sample_features.append(slope)
            
            features.append(sample_features)
        
        return np.array(features)

class DiversityEvaluator:
    """多样性评估器"""
    
    def __init__(self):
        self.logger = logging.getLogger("DiversityEvaluator")
    
    def evaluate_diversity(self, synthetic_data: np.ndarray) -> Dict[str, float]:
        """评估生成样本多样性"""
        metrics = {}
        
        try:
            # 样本间距离多样性
            metrics.update(self._pairwise_diversity(synthetic_data))
            
            # 特征空间覆盖度
            metrics.update(self._feature_space_coverage(synthetic_data))
            
            # 模式多样性
            metrics.update(self._pattern_diversity(synthetic_data))
            
        except Exception as e:
            self.logger.error(f"多样性评估失败: {e}")
            
        return metrics
    
    def _pairwise_diversity(self, data: np.ndarray) -> Dict[str, float]:
        """样本间多样性"""
        # 随机选择样本进行比较 (避免计算量过大)
        sample_indices = np.random.choice(len(data), min(100, len(data)), replace=False)
        selected_samples = data[sample_indices]
        
        # 计算样本间距离
        distances = []
        for i in range(len(selected_samples)):
            for j in range(i + 1, len(selected_samples)):
                dist = np.linalg.norm(selected_samples[i] - selected_samples[j])
                distances.append(dist)
        
        mean_distance = np.mean(distances) if distances else 0
        std_distance = np.std(distances) if distances else 0
        
        return {
            "mean_pairwise_distance": mean_distance,
            "std_pairwise_distance": std_distance
        }
    
    def _feature_space_coverage(self, data: np.ndarray) -> Dict[str, float]:
        """特征空间覆盖度"""
        # 使用PCA降维分析覆盖度
        data_flat = data.reshape(len(data), -1)
        
        try:
            pca = PCA(n_components=min(10, data_flat.shape[1]))
            pca_data = pca.fit_transform(data_flat)
            
            # 计算各主成分的方差
            explained_variance_ratio = pca.explained_variance_ratio_
            coverage_score = np.sum(explained_variance_ratio)
            
        except:
            coverage_score = 0
        
        return {"feature_space_coverage": coverage_score}
    
    def _pattern_diversity(self, data: np.ndarray) -> Dict[str, float]:
        """模式多样性"""
        # 使用聚类分析模式多样性
        from sklearn.cluster import KMeans
        
        data_flat = data.reshape(len(data), -1)
        
        try:
            # 尝试不同的聚类数
            n_clusters = min(10, len(data) // 2)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data_flat)
                
                # 计算聚类的均匀性
                unique_labels, counts = np.unique(labels, return_counts=True)
                pattern_diversity = len(unique_labels) / n_clusters
            else:
                pattern_diversity = 0
                
        except:
            pattern_diversity = 0
        
        return {"pattern_diversity": pattern_diversity}

class CTBenchEvaluator:
    """CTBench综合评估器"""
    
    def __init__(self):
        self.statistical_evaluator = StatisticalEvaluator()
        self.temporal_evaluator = TemporalEvaluator()
        self.financial_evaluator = FinancialEvaluator()
        self.discriminative_evaluator = DiscriminativeEvaluator()
        self.diversity_evaluator = DiversityEvaluator()
        
        self.logger = logging.getLogger("CTBenchEvaluator")
    
    def comprehensive_evaluation(self, real_data: np.ndarray, 
                               synthetic_data: np.ndarray) -> EvaluationMetrics:
        """综合评估"""
        self.logger.info("开始综合评估...")
        
        # 各项评估
        statistical_similarity = self.statistical_evaluator.evaluate_statistical_similarity(
            real_data, synthetic_data
        )
        
        temporal_features = self.temporal_evaluator.evaluate_temporal_features(
            real_data, synthetic_data
        )
        
        financial_features = self.financial_evaluator.evaluate_financial_features(
            real_data, synthetic_data
        )
        
        discriminative_score = self.discriminative_evaluator.evaluate_discriminative_score(
            real_data, synthetic_data
        )
        
        diversity_metrics = self.diversity_evaluator.evaluate_diversity(synthetic_data)
        
        # 计算综合质量得分
        overall_score = self._calculate_overall_score(
            statistical_similarity, temporal_features, financial_features,
            discriminative_score, diversity_metrics
        )
        
        # 创建评估结果
        metrics = EvaluationMetrics(
            statistical_similarity=statistical_similarity,
            distribution_similarity=statistical_similarity,  # 暂时复用
            temporal_features=temporal_features,
            financial_features=financial_features,
            discriminative_score=discriminative_score,
            diversity_metrics=diversity_metrics,
            overall_quality_score=overall_score,
            evaluation_timestamp=datetime.utcnow()
        )
        
        self.logger.info(f"评估完成，综合得分: {overall_score:.4f}")
        return metrics
    
    def _calculate_overall_score(self, statistical_similarity: Dict[str, float],
                               temporal_features: Dict[str, float],
                               financial_features: Dict[str, float],
                               discriminative_score: float,
                               diversity_metrics: Dict[str, float]) -> float:
        """计算综合得分"""
        
        # 权重设置
        weights = {
            "statistical": 0.25,
            "temporal": 0.25,
            "financial": 0.25,
            "discriminative": 0.15,
            "diversity": 0.10
        }
        
        # 统计相似性得分 (差异越小越好)
        stat_scores = []
        for key, value in statistical_similarity.items():
            if "difference" in key:
                # 转换为0-1得分
                score = max(0, 1 - min(1, value))
                stat_scores.append(score)
        statistical_score = np.mean(stat_scores) if stat_scores else 0
        
        # 时间序列特征得分
        temporal_scores = []
        for key, value in temporal_features.items():
            if "difference" in key:
                score = max(0, 1 - min(1, value))
                temporal_scores.append(score)
        temporal_score = np.mean(temporal_scores) if temporal_scores else 0
        
        # 金融特征得分
        financial_scores = []
        for key, value in financial_features.items():
            if "difference" in key:
                score = max(0, 1 - min(1, value * 10))  # 放大差异的影响
                financial_scores.append(score)
        financial_score = np.mean(financial_scores) if financial_scores else 0
        
        # 多样性得分
        diversity_score = diversity_metrics.get("pattern_diversity", 0)
        
        # 计算加权总分
        total_score = (weights["statistical"] * statistical_score +
                      weights["temporal"] * temporal_score +
                      weights["financial"] * financial_score +
                      weights["discriminative"] * discriminative_score +
                      weights["diversity"] * diversity_score)
        
        return total_score
    
    def generate_evaluation_report(self, metrics: EvaluationMetrics, 
                                 output_path: str = None) -> str:
        """生成评估报告"""
        report = f"""
# CTBench模型评估报告

评估时间: {metrics.evaluation_timestamp}

## 综合质量得分: {metrics.overall_quality_score:.4f}

## 统计相似性评估
{self._format_metrics_dict(metrics.statistical_similarity)}

## 时间序列特征评估
{self._format_metrics_dict(metrics.temporal_features)}

## 金融特征评估
{self._format_metrics_dict(metrics.financial_features)}

## 判别性得分: {metrics.discriminative_score:.4f}

## 多样性指标
{self._format_metrics_dict(metrics.diversity_metrics)}

## 评估总结
- 统计相似性: {'优秀' if self._get_avg_score(metrics.statistical_similarity) > 0.8 else '良好' if self._get_avg_score(metrics.statistical_similarity) > 0.6 else '需改进'}
- 时间序列特征: {'优秀' if self._get_avg_score(metrics.temporal_features) > 0.8 else '良好' if self._get_avg_score(metrics.temporal_features) > 0.6 else '需改进'}
- 金融特征: {'优秀' if self._get_avg_score(metrics.financial_features) > 0.8 else '良好' if self._get_avg_score(metrics.financial_features) > 0.6 else '需改进'}
- 判别难度: {'优秀' if metrics.discriminative_score > 0.8 else '良好' if metrics.discriminative_score > 0.6 else '需改进'}
- 样本多样性: {'优秀' if self._get_avg_score(metrics.diversity_metrics) > 0.8 else '良好' if self._get_avg_score(metrics.diversity_metrics) > 0.6 else '需改进'}
        """
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"评估报告已保存到: {output_path}")
        
        return report
    
    def _format_metrics_dict(self, metrics_dict: Dict[str, float]) -> str:
        """格式化指标字典"""
        formatted = ""
        for key, value in metrics_dict.items():
            formatted += f"- {key}: {value:.6f}\n"
        return formatted
    
    def _get_avg_score(self, metrics_dict: Dict[str, float]) -> float:
        """获取平均得分"""
        if not metrics_dict:
            return 0
        
        scores = []
        for key, value in metrics_dict.items():
            if "difference" in key:
                # 差异指标转换为得分
                score = max(0, 1 - min(1, value))
                scores.append(score)
            else:
                # 直接得分
                scores.append(value)
        
        return np.mean(scores) if scores else 0