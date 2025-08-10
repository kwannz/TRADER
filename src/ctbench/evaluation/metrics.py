"""
CTBench Evaluation Metrics
时序生成模型的评估指标体系
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
import logging

@dataclass
class EvaluationResult:
    """评估结果"""
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    detailed_results: Dict[str, Any]
    timestamp: str
    
class TimeSeriesMetrics:
    """时序数据评估指标"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_all_metrics(self, real_data: np.ndarray, 
                            synthetic_data: np.ndarray,
                            model_name: str = "Unknown") -> EvaluationResult:
        """计算所有评估指标"""
        try:
            metrics = {}
            detailed = {}
            
            # 1. 基础统计相似性
            basic_stats = self.calculate_basic_statistics_similarity(real_data, synthetic_data)
            metrics.update(basic_stats['metrics'])
            detailed['basic_statistics'] = basic_stats['details']
            
            # 2. 分布相似性
            dist_stats = self.calculate_distribution_similarity(real_data, synthetic_data)
            metrics.update(dist_stats['metrics'])
            detailed['distribution'] = dist_stats['details']
            
            # 3. 时序特性
            temporal_stats = self.calculate_temporal_characteristics(real_data, synthetic_data)
            metrics.update(temporal_stats['metrics'])
            detailed['temporal'] = temporal_stats['details']
            
            # 4. 财务指标相似性
            financial_stats = self.calculate_financial_metrics_similarity(real_data, synthetic_data)
            metrics.update(financial_stats['metrics'])
            detailed['financial'] = financial_stats['details']
            
            # 5. 多元性分析
            multivariate_stats = self.calculate_multivariate_similarity(real_data, synthetic_data)
            metrics.update(multivariate_stats['metrics'])
            detailed['multivariate'] = multivariate_stats['details']
            
            # 6. 综合评分
            overall_score = self.calculate_overall_score(metrics)
            metrics['overall_score'] = overall_score
            
            return EvaluationResult(
                model_name=model_name,
                dataset_name="evaluation_dataset",
                metrics=metrics,
                detailed_results=detailed,
                timestamp=pd.Timestamp.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"评估指标计算失败: {e}")
            raise
            
    def calculate_basic_statistics_similarity(self, real_data: np.ndarray, 
                                            synthetic_data: np.ndarray) -> Dict[str, Any]:
        """计算基础统计指标相似性"""
        metrics = {}
        details = {}
        
        try:
            # 展平数据以便计算统计量
            real_flat = real_data.reshape(-1, real_data.shape[-1])
            synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            
            # 均值相似性
            real_means = np.mean(real_flat, axis=0)
            synthetic_means = np.mean(synthetic_flat, axis=0)
            mean_diff = np.mean(np.abs(real_means - synthetic_means) / (np.abs(real_means) + 1e-8))
            metrics['mean_similarity'] = 1 - mean_diff
            
            # 标准差相似性
            real_stds = np.std(real_flat, axis=0)
            synthetic_stds = np.std(synthetic_flat, axis=0)
            std_diff = np.mean(np.abs(real_stds - synthetic_stds) / (real_stds + 1e-8))
            metrics['std_similarity'] = 1 - std_diff
            
            # 偏度相似性
            real_skew = stats.skew(real_flat, axis=0)
            synthetic_skew = stats.skew(synthetic_flat, axis=0)
            skew_diff = np.mean(np.abs(real_skew - synthetic_skew))
            metrics['skewness_similarity'] = np.exp(-skew_diff)
            
            # 峰度相似性
            real_kurtosis = stats.kurtosis(real_flat, axis=0)
            synthetic_kurtosis = stats.kurtosis(synthetic_flat, axis=0)
            kurtosis_diff = np.mean(np.abs(real_kurtosis - synthetic_kurtosis))
            metrics['kurtosis_similarity'] = np.exp(-kurtosis_diff)
            
            # 详细信息
            details['real_means'] = real_means.tolist()
            details['synthetic_means'] = synthetic_means.tolist()
            details['real_stds'] = real_stds.tolist()
            details['synthetic_stds'] = synthetic_stds.tolist()
            details['real_skew'] = real_skew.tolist()
            details['synthetic_skew'] = synthetic_skew.tolist()
            details['real_kurtosis'] = real_kurtosis.tolist()
            details['synthetic_kurtosis'] = synthetic_kurtosis.tolist()
            
        except Exception as e:
            self.logger.warning(f"基础统计相似性计算失败: {e}")
            
        return {'metrics': metrics, 'details': details}
        
    def calculate_distribution_similarity(self, real_data: np.ndarray, 
                                        synthetic_data: np.ndarray) -> Dict[str, Any]:
        """计算分布相似性"""
        metrics = {}
        details = {}
        
        try:
            real_flat = real_data.reshape(-1, real_data.shape[-1])
            synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            
            wasserstein_distances = []
            ks_statistics = []
            
            for i in range(real_data.shape[-1]):
                real_feature = real_flat[:, i]
                synthetic_feature = synthetic_flat[:, i]
                
                # Wasserstein距离
                wd = wasserstein_distance(real_feature, synthetic_feature)
                wasserstein_distances.append(wd)
                
                # KS检验
                ks_stat, ks_pvalue = stats.ks_2samp(real_feature, synthetic_feature)
                ks_statistics.append(ks_stat)
                
            # 平均Wasserstein距离 (归一化)
            avg_wasserstein = np.mean(wasserstein_distances)
            real_range = np.mean([np.ptp(real_flat[:, i]) for i in range(real_flat.shape[1])])
            metrics['wasserstein_similarity'] = np.exp(-avg_wasserstein / (real_range + 1e-8))
            
            # 平均KS统计量
            avg_ks = np.mean(ks_statistics)
            metrics['ks_similarity'] = 1 - avg_ks
            
            details['wasserstein_distances'] = wasserstein_distances
            details['ks_statistics'] = ks_statistics
            
        except Exception as e:
            self.logger.warning(f"分布相似性计算失败: {e}")
            
        return {'metrics': metrics, 'details': details}
        
    def calculate_temporal_characteristics(self, real_data: np.ndarray, 
                                         synthetic_data: np.ndarray) -> Dict[str, Any]:
        """计算时序特性"""
        metrics = {}
        details = {}
        
        try:
            # 自相关函数相似性
            real_autocorr = self._calculate_autocorrelation(real_data)
            synthetic_autocorr = self._calculate_autocorrelation(synthetic_data)
            
            autocorr_similarity = 1 - np.mean(np.abs(real_autocorr - synthetic_autocorr))
            metrics['autocorrelation_similarity'] = max(0, autocorr_similarity)
            
            # 趋势相似性
            real_trends = self._calculate_trends(real_data)
            synthetic_trends = self._calculate_trends(synthetic_data)
            
            trend_correlation = np.corrcoef(real_trends, synthetic_trends)[0, 1]
            metrics['trend_similarity'] = trend_correlation if not np.isnan(trend_correlation) else 0
            
            # 波动率聚类性
            real_volatility_clustering = self._calculate_volatility_clustering(real_data)
            synthetic_volatility_clustering = self._calculate_volatility_clustering(synthetic_data)
            
            vol_clustering_diff = abs(real_volatility_clustering - synthetic_volatility_clustering)
            metrics['volatility_clustering_similarity'] = np.exp(-vol_clustering_diff)
            
            details['real_autocorr'] = real_autocorr.tolist()
            details['synthetic_autocorr'] = synthetic_autocorr.tolist()
            details['real_volatility_clustering'] = real_volatility_clustering
            details['synthetic_volatility_clustering'] = synthetic_volatility_clustering
            
        except Exception as e:
            self.logger.warning(f"时序特性计算失败: {e}")
            
        return {'metrics': metrics, 'details': details}
        
    def calculate_financial_metrics_similarity(self, real_data: np.ndarray, 
                                             synthetic_data: np.ndarray) -> Dict[str, Any]:
        """计算财务指标相似性"""
        metrics = {}
        details = {}
        
        try:
            # 假设数据格式为 (samples, time_steps, features)
            # 其中第0列为价格数据
            
            real_returns = self._calculate_returns(real_data[:, :, 0])
            synthetic_returns = self._calculate_returns(synthetic_data[:, :, 0])
            
            # 夏普比率相似性
            real_sharpe = self._calculate_sharpe_ratio(real_returns)
            synthetic_sharpe = self._calculate_sharpe_ratio(synthetic_returns)
            
            sharpe_diff = abs(real_sharpe - synthetic_sharpe) / (abs(real_sharpe) + 1e-8)
            metrics['sharpe_ratio_similarity'] = np.exp(-sharpe_diff)
            
            # 最大回撤相似性
            real_max_dd = self._calculate_max_drawdown(real_returns)
            synthetic_max_dd = self._calculate_max_drawdown(synthetic_returns)
            
            dd_diff = abs(real_max_dd - synthetic_max_dd) / (abs(real_max_dd) + 1e-8)
            metrics['max_drawdown_similarity'] = np.exp(-dd_diff)
            
            # VaR相似性
            real_var = np.percentile(real_returns.flatten(), 5)
            synthetic_var = np.percentile(synthetic_returns.flatten(), 5)
            
            var_diff = abs(real_var - synthetic_var) / (abs(real_var) + 1e-8)
            metrics['var_similarity'] = np.exp(-var_diff)
            
            # 尾部风险相似性
            real_tail = np.mean(real_returns.flatten()[real_returns.flatten() <= real_var])
            synthetic_tail = np.mean(synthetic_returns.flatten()[synthetic_returns.flatten() <= synthetic_var])
            
            tail_diff = abs(real_tail - synthetic_tail) / (abs(real_tail) + 1e-8)
            metrics['tail_risk_similarity'] = np.exp(-tail_diff)
            
            details.update({
                'real_sharpe': real_sharpe,
                'synthetic_sharpe': synthetic_sharpe,
                'real_max_drawdown': real_max_dd,
                'synthetic_max_drawdown': synthetic_max_dd,
                'real_var': real_var,
                'synthetic_var': synthetic_var,
                'real_tail_risk': real_tail,
                'synthetic_tail_risk': synthetic_tail
            })
            
        except Exception as e:
            self.logger.warning(f"财务指标计算失败: {e}")
            
        return {'metrics': metrics, 'details': details}
        
    def calculate_multivariate_similarity(self, real_data: np.ndarray, 
                                        synthetic_data: np.ndarray) -> Dict[str, Any]:
        """计算多元性分析"""
        metrics = {}
        details = {}
        
        try:
            if real_data.shape[-1] > 1:
                # 相关矩阵相似性
                real_corr = self._calculate_correlation_matrix(real_data)
                synthetic_corr = self._calculate_correlation_matrix(synthetic_data)
                
                corr_frobenius_diff = np.linalg.norm(real_corr - synthetic_corr, 'fro')
                max_corr_norm = max(np.linalg.norm(real_corr, 'fro'), np.linalg.norm(synthetic_corr, 'fro'))
                metrics['correlation_matrix_similarity'] = 1 - corr_frobenius_diff / (max_corr_norm + 1e-8)
                
                # 协方差矩阵相似性
                real_cov = self._calculate_covariance_matrix(real_data)
                synthetic_cov = self._calculate_covariance_matrix(synthetic_data)
                
                cov_frobenius_diff = np.linalg.norm(real_cov - synthetic_cov, 'fro')
                max_cov_norm = max(np.linalg.norm(real_cov, 'fro'), np.linalg.norm(synthetic_cov, 'fro'))
                metrics['covariance_matrix_similarity'] = 1 - cov_frobenius_diff / (max_cov_norm + 1e-8)
                
                details['real_correlation_matrix'] = real_corr.tolist()
                details['synthetic_correlation_matrix'] = synthetic_corr.tolist()
                
            else:
                metrics['correlation_matrix_similarity'] = 1.0
                metrics['covariance_matrix_similarity'] = 1.0
                
        except Exception as e:
            self.logger.warning(f"多元性分析计算失败: {e}")
            
        return {'metrics': metrics, 'details': details}
        
    def calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        # 权重设定
        weights = {
            'mean_similarity': 0.15,
            'std_similarity': 0.15,
            'wasserstein_similarity': 0.20,
            'autocorrelation_similarity': 0.15,
            'sharpe_ratio_similarity': 0.10,
            'max_drawdown_similarity': 0.10,
            'correlation_matrix_similarity': 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def _calculate_autocorrelation(self, data: np.ndarray, max_lags: int = 10) -> np.ndarray:
        """计算自相关函数"""
        # 取第一个特征的自相关
        flat_data = data.reshape(-1, data.shape[-1])
        feature_data = flat_data[:, 0] if flat_data.shape[1] > 0 else flat_data.flatten()
        
        autocorr = []
        for lag in range(1, max_lags + 1):
            if len(feature_data) > lag:
                corr = np.corrcoef(feature_data[:-lag], feature_data[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
            else:
                autocorr.append(0)
                
        return np.array(autocorr)
        
    def _calculate_trends(self, data: np.ndarray) -> np.ndarray:
        """计算趋势"""
        # 计算每个样本的线性趋势
        trends = []
        
        for sample in data:
            if len(sample.shape) == 2:
                price_series = sample[:, 0]
            else:
                price_series = sample
                
            x = np.arange(len(price_series))
            slope, _ = np.polyfit(x, price_series, 1)
            trends.append(slope)
            
        return np.array(trends)
        
    def _calculate_volatility_clustering(self, data: np.ndarray) -> float:
        """计算波动率聚类性（ARCH效应）"""
        try:
            # 计算收益率
            returns = self._calculate_returns(data[:, :, 0] if len(data.shape) == 3 else data)
            returns_flat = returns.flatten()
            
            # 计算平方收益率的自相关
            squared_returns = returns_flat ** 2
            if len(squared_returns) > 1:
                autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                return autocorr if not np.isnan(autocorr) else 0
            else:
                return 0
                
        except Exception:
            return 0
            
    def _calculate_returns(self, price_data: np.ndarray) -> np.ndarray:
        """计算收益率"""
        if len(price_data.shape) == 1:
            return np.diff(price_data) / (price_data[:-1] + 1e-8)
        else:
            returns = []
            for series in price_data:
                ret = np.diff(series) / (series[:-1] + 1e-8)
                returns.append(ret)
            return np.array(returns)
            
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        flat_returns = returns.flatten()
        excess_return = np.mean(flat_returns) - risk_free_rate
        volatility = np.std(flat_returns)
        
        return excess_return / volatility if volatility > 0 else 0
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        flat_returns = returns.flatten()
        cumulative = np.cumprod(1 + flat_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        return np.min(drawdowns)
        
    def _calculate_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """计算相关矩阵"""
        flat_data = data.reshape(-1, data.shape[-1])
        return np.corrcoef(flat_data.T)
        
    def _calculate_covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """计算协方差矩阵"""
        flat_data = data.reshape(-1, data.shape[-1])
        return np.cov(flat_data.T)

class BenchmarkEvaluator:
    """基准评估器"""
    
    def __init__(self):
        self.metrics_calculator = TimeSeriesMetrics()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, model, test_data: np.ndarray, 
                      num_samples: int = None) -> EvaluationResult:
        """评估单个模型"""
        try:
            if num_samples is None:
                num_samples = test_data.shape[0]
                
            # 生成合成数据
            synthetic_data = model.generate(num_samples)
            
            if hasattr(synthetic_data, 'cpu'):
                synthetic_data = synthetic_data.cpu().detach().numpy()
                
            # 计算评估指标
            result = self.metrics_calculator.calculate_all_metrics(
                test_data, synthetic_data, 
                model_name=getattr(model, 'model_name', 'Unknown')
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            raise
            
    def compare_models(self, models: List[Any], test_data: np.ndarray,
                      num_samples: int = None) -> List[EvaluationResult]:
        """比较多个模型"""
        results = []
        
        for model in models:
            try:
                result = self.evaluate_model(model, test_data, num_samples)
                results.append(result)
                self.logger.info(f"模型 {result.model_name} 评估完成，综合得分: {result.metrics['overall_score']:.4f}")
            except Exception as e:
                self.logger.error(f"模型评估失败: {e}")
                
        # 按综合得分排序
        results.sort(key=lambda x: x.metrics['overall_score'], reverse=True)
        
        return results
        
    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """生成评估报告"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'num_models': len(results),
            'models_evaluated': []
        }
        
        for result in results:
            model_report = {
                'model_name': result.model_name,
                'overall_score': result.metrics['overall_score'],
                'key_metrics': {
                    'mean_similarity': result.metrics.get('mean_similarity', 0),
                    'distribution_similarity': result.metrics.get('wasserstein_similarity', 0),
                    'temporal_similarity': result.metrics.get('autocorrelation_similarity', 0),
                    'financial_similarity': result.metrics.get('sharpe_ratio_similarity', 0)
                },
                'ranking': results.index(result) + 1
            }
            report['models_evaluated'].append(model_report)
            
        # 找出最佳模型
        if results:
            best_model = results[0]
            report['best_model'] = {
                'name': best_model.model_name,
                'score': best_model.metrics['overall_score'],
                'strengths': self._identify_model_strengths(best_model.metrics),
                'weaknesses': self._identify_model_weaknesses(best_model.metrics)
            }
            
        return report
        
    def _identify_model_strengths(self, metrics: Dict[str, float]) -> List[str]:
        """识别模型优势"""
        strengths = []
        threshold = 0.8
        
        if metrics.get('mean_similarity', 0) > threshold:
            strengths.append("优秀的均值拟合能力")
        if metrics.get('wasserstein_similarity', 0) > threshold:
            strengths.append("出色的分布相似性")
        if metrics.get('autocorrelation_similarity', 0) > threshold:
            strengths.append("良好的时序依赖性建模")
        if metrics.get('sharpe_ratio_similarity', 0) > threshold:
            strengths.append("准确的风险收益特征")
            
        return strengths
        
    def _identify_model_weaknesses(self, metrics: Dict[str, float]) -> List[str]:
        """识别模型弱点"""
        weaknesses = []
        threshold = 0.6
        
        if metrics.get('mean_similarity', 0) < threshold:
            weaknesses.append("均值拟合有待改进")
        if metrics.get('wasserstein_similarity', 0) < threshold:
            weaknesses.append("分布拟合需要优化")
        if metrics.get('autocorrelation_similarity', 0) < threshold:
            weaknesses.append("时序依赖性建模不足")
        if metrics.get('sharpe_ratio_similarity', 0) < threshold:
            weaknesses.append("风险收益特征拟合有偏差")
            
        return weaknesses