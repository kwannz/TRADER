"""
Unified Factor Validator
统一因子验证器 - 融合传统IC分析与CTBench压力测试
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from scipy import stats
import warnings

# 抑制警告
warnings.filterwarnings('ignore')


@dataclass
class ICAnalysisResult:
    """IC分析结果数据类"""
    factor_name: str
    period: int
    ic_mean: float
    ic_std: float
    ic_ir: float  # Information Ratio
    ic_positive_ratio: float
    ic_skewness: float
    ic_kurtosis: float
    monthly_ic: Dict[str, float]
    rolling_ic: pd.Series


@dataclass
class FactorPerformanceMetrics:
    """因子性能指标数据类"""
    factor_name: str
    sample_period: Tuple[str, str]
    total_observations: int
    missing_ratio: float
    
    # 基础统计
    mean: float
    std: float
    skewness: float
    kurtosis: float
    
    # IC分析结果
    ic_results: Dict[int, ICAnalysisResult]
    
    # 分层回测结果 
    layered_returns: Dict[int, float]  # 分层收益
    long_short_return: float  # 多空收益
    
    # 压力测试结果
    stress_test_results: Dict[str, Any]


class ICAnalyzer:
    """
    IC分析器 - 传统因子IC分析
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ICAnalyzer")
    
    def calculate_ic(self, factor_values: pd.Series, future_returns: pd.Series,
                    method: str = 'pearson') -> pd.Series:
        """
        计算IC值
        
        Args:
            factor_values: 因子值 (MultiIndex: date, symbol)
            future_returns: 未来收益率 (MultiIndex: date, symbol)
            method: 相关性计算方法 ('pearson', 'spearman')
            
        Returns:
            时间序列IC值
        """
        try:
            # 对齐数据
            factor_aligned, returns_aligned = factor_values.align(future_returns, join='inner')
            
            if len(factor_aligned) == 0:
                self.logger.warning("No aligned data for IC calculation")
                return pd.Series(dtype=float)
            
            # 按日期分组计算IC
            def calculate_daily_ic(group_data):
                date_factor = factor_aligned.loc[group_data.index]
                date_returns = returns_aligned.loc[group_data.index]
                
                # 去除缺失值
                valid_mask = (~date_factor.isna()) & (~date_returns.isna())
                if valid_mask.sum() < 5:  # 至少需要5个有效观测
                    return np.nan
                
                valid_factor = date_factor[valid_mask]
                valid_returns = date_returns[valid_mask]
                
                if method == 'spearman':
                    ic, _ = stats.spearmanr(valid_factor, valid_returns)
                else:
                    ic, _ = stats.pearsonr(valid_factor, valid_returns)
                
                return ic
            
            # 计算每日IC
            dates = factor_aligned.index.get_level_values('date').unique()
            ic_series = []
            
            for date in dates:
                try:
                    date_mask = factor_aligned.index.get_level_values('date') == date
                    date_factor = factor_aligned[date_mask]
                    date_returns = returns_aligned[date_mask]
                    
                    # 去除缺失值
                    valid_mask = (~date_factor.isna()) & (~date_returns.isna())
                    
                    if valid_mask.sum() < 5:
                        ic_series.append((date, np.nan))
                        continue
                    
                    valid_factor = date_factor[valid_mask].values
                    valid_returns = date_returns[valid_mask].values
                    
                    if method == 'spearman':
                        ic, _ = stats.spearmanr(valid_factor, valid_returns)
                    else:
                        ic, _ = stats.pearsonr(valid_factor, valid_returns)
                    
                    ic_series.append((date, ic))
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating IC for date {date}: {str(e)}")
                    ic_series.append((date, np.nan))
            
            # 转换为Series
            if ic_series:
                dates, ic_values = zip(*ic_series)
                return pd.Series(ic_values, index=pd.to_datetime(dates), name='IC')
            else:
                return pd.Series(dtype=float)
            
        except Exception as e:
            self.logger.error(f"Error in IC calculation: {str(e)}")
            return pd.Series(dtype=float)
    
    def analyze_ic_performance(self, ic_series: pd.Series, factor_name: str, 
                              period: int) -> ICAnalysisResult:
        """
        分析IC性能
        
        Args:
            ic_series: IC时间序列
            factor_name: 因子名称
            period: 持有期
            
        Returns:
            IC分析结果
        """
        try:
            # 去除缺失值
            valid_ic = ic_series.dropna()
            
            if len(valid_ic) == 0:
                self.logger.warning(f"No valid IC values for factor {factor_name}")
                return ICAnalysisResult(
                    factor_name=factor_name,
                    period=period,
                    ic_mean=np.nan,
                    ic_std=np.nan,
                    ic_ir=np.nan,
                    ic_positive_ratio=0.0,
                    ic_skewness=np.nan,
                    ic_kurtosis=np.nan,
                    monthly_ic={},
                    rolling_ic=pd.Series(dtype=float)
                )
            
            # 基础统计
            ic_mean = valid_ic.mean()
            ic_std = valid_ic.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
            ic_positive_ratio = (valid_ic > 0).sum() / len(valid_ic)
            ic_skewness = valid_ic.skew()
            ic_kurtosis = valid_ic.kurtosis()
            
            # 按月统计IC
            monthly_ic = {}
            if len(valid_ic) > 0:
                monthly_grouped = valid_ic.groupby(pd.Grouper(freq='M'))
                for month, group in monthly_grouped:
                    if len(group) > 0:
                        monthly_ic[month.strftime('%Y-%m')] = group.mean()
            
            # 滚动IC（20日窗口）
            rolling_ic = valid_ic.rolling(window=20, min_periods=10).mean()
            
            return ICAnalysisResult(
                factor_name=factor_name,
                period=period,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_ir=ic_ir,
                ic_positive_ratio=ic_positive_ratio,
                ic_skewness=ic_skewness,
                ic_kurtosis=ic_kurtosis,
                monthly_ic=monthly_ic,
                rolling_ic=rolling_ic
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing IC performance: {str(e)}")
            return ICAnalysisResult(
                factor_name=factor_name,
                period=period,
                ic_mean=np.nan,
                ic_std=np.nan,
                ic_ir=np.nan,
                ic_positive_ratio=0.0,
                ic_skewness=np.nan,
                ic_kurtosis=np.nan,
                monthly_ic={},
                rolling_ic=pd.Series(dtype=float)
            )


class LayeredBacktester:
    """
    分层回测器 - 因子分层回测分析
    """
    
    def __init__(self, n_layers: int = 5):
        self.n_layers = n_layers
        self.logger = logging.getLogger("LayeredBacktester")
    
    def run_layered_backtest(self, factor_values: pd.Series, future_returns: pd.Series,
                           holding_period: int = 1) -> Dict[int, float]:
        """
        执行分层回测
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益率
            holding_period: 持有期
            
        Returns:
            各层平均收益率
        """
        try:
            # 对齐数据
            factor_aligned, returns_aligned = factor_values.align(future_returns, join='inner')
            
            if len(factor_aligned) == 0:
                self.logger.warning("No aligned data for layered backtest")
                return {}
            
            layer_returns = {i: [] for i in range(1, self.n_layers + 1)}
            
            # 按日期分组
            dates = factor_aligned.index.get_level_values('date').unique()
            
            for date in dates:
                try:
                    date_mask = factor_aligned.index.get_level_values('date') == date
                    date_factor = factor_aligned[date_mask]
                    date_returns = returns_aligned[date_mask]
                    
                    # 去除缺失值
                    valid_mask = (~date_factor.isna()) & (~date_returns.isna())
                    valid_factor = date_factor[valid_mask]
                    valid_returns = date_returns[valid_mask]
                    
                    if len(valid_factor) < self.n_layers * 2:  # 每层至少2个股票
                        continue
                    
                    # 按因子值分层
                    quantiles = np.linspace(0, 1, self.n_layers + 1)
                    layers = pd.qcut(valid_factor, q=quantiles, labels=False, duplicates='drop') + 1
                    
                    # 计算各层收益率
                    for layer in range(1, self.n_layers + 1):
                        layer_mask = layers == layer
                        if layer_mask.sum() > 0:
                            layer_return = valid_returns[layer_mask].mean()
                            layer_returns[layer].append(layer_return)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing date {date} in layered backtest: {str(e)}")
                    continue
            
            # 计算各层平均收益率
            avg_layer_returns = {}
            for layer in range(1, self.n_layers + 1):
                if len(layer_returns[layer]) > 0:
                    avg_layer_returns[layer] = np.mean(layer_returns[layer])
                else:
                    avg_layer_returns[layer] = np.nan
            
            return avg_layer_returns
            
        except Exception as e:
            self.logger.error(f"Error in layered backtest: {str(e)}")
            return {}


class StressTestEngine:
    """
    压力测试引擎 - 集成CTBench压力测试能力
    """
    
    def __init__(self):
        self.logger = logging.getLogger("StressTestEngine")
    
    def run_stress_tests(self, factor_values: pd.Series, market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        执行综合压力测试
        
        Args:
            factor_values: 因子值
            market_data: 市场数据
            
        Returns:
            压力测试结果
        """
        results = {}
        
        try:
            # 1. 极端市场条件测试
            results['extreme_market'] = self._test_extreme_market_conditions(factor_values, market_data)
            
            # 2. 波动率冲击测试
            results['volatility_shock'] = self._test_volatility_shock(factor_values, market_data)
            
            # 3. 流动性干扰测试
            results['liquidity_stress'] = self._test_liquidity_stress(factor_values, market_data)
            
            # 4. 因子衰减测试
            results['factor_decay'] = self._test_factor_decay(factor_values)
            
            # 5. 数据完整性测试
            results['data_integrity'] = self._test_data_integrity(factor_values)
            
            self.logger.info("Completed all stress tests")
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _test_extreme_market_conditions(self, factor_values: pd.Series, 
                                      market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """测试极端市场条件下的表现"""
        try:
            results = {}
            
            if 'close' not in market_data:
                return {"error": "No close price data available"}
            
            close_prices = market_data['close']
            
            # 计算市场收益率
            market_returns = close_prices.groupby(level='symbol').pct_change()
            daily_market_returns = market_returns.groupby(level='date').mean()
            
            if len(daily_market_returns) < 10:
                return {"error": "Insufficient market data"}
            
            # 识别极端市场日（收益率超过2个标准差）
            return_threshold = daily_market_returns.std() * 2
            extreme_up_days = daily_market_returns[daily_market_returns > return_threshold].index
            extreme_down_days = daily_market_returns[daily_market_returns < -return_threshold].index
            
            results['extreme_up_days'] = len(extreme_up_days)
            results['extreme_down_days'] = len(extreme_down_days)
            
            # 计算因子在极端市场日的表现
            if len(extreme_up_days) > 0:
                extreme_up_factor = factor_values[factor_values.index.get_level_values('date').isin(extreme_up_days)]
                results['factor_stability_bull'] = {
                    'std': extreme_up_factor.std(),
                    'mean': extreme_up_factor.mean(),
                    'count': len(extreme_up_factor)
                }
            
            if len(extreme_down_days) > 0:
                extreme_down_factor = factor_values[factor_values.index.get_level_values('date').isin(extreme_down_days)]
                results['factor_stability_bear'] = {
                    'std': extreme_down_factor.std(),
                    'mean': extreme_down_factor.mean(),
                    'count': len(extreme_down_factor)
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Extreme market test failed: {str(e)}"}
    
    def _test_volatility_shock(self, factor_values: pd.Series, 
                             market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """测试波动率冲击"""
        try:
            results = {}
            
            if 'close' not in market_data:
                return {"error": "No close price data for volatility test"}
            
            close_prices = market_data['close']
            
            # 计算滚动波动率
            returns = close_prices.groupby(level='symbol').pct_change()
            rolling_vol = returns.groupby(level='symbol').rolling(window=20).std()
            
            # 识别高波动期
            vol_threshold = rolling_vol.quantile(0.9)  # 90%分位数
            high_vol_periods = rolling_vol[rolling_vol > vol_threshold].index
            
            if len(high_vol_periods) > 0:
                high_vol_factor = factor_values[factor_values.index.isin(high_vol_periods)]
                results['high_volatility_periods'] = {
                    'count': len(high_vol_periods),
                    'factor_mean': high_vol_factor.mean(),
                    'factor_std': high_vol_factor.std()
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Volatility shock test failed: {str(e)}"}
    
    def _test_liquidity_stress(self, factor_values: pd.Series, 
                             market_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """测试流动性压力"""
        try:
            results = {}
            
            if 'volume' not in market_data:
                return {"error": "No volume data for liquidity test"}
            
            volume = market_data['volume']
            
            # 识别低流动性期
            low_volume_threshold = volume.quantile(0.1)  # 10%分位数
            low_liquidity_mask = volume < low_volume_threshold
            
            low_liquidity_factor = factor_values[low_liquidity_mask]
            
            if len(low_liquidity_factor) > 0:
                results['low_liquidity_impact'] = {
                    'affected_observations': len(low_liquidity_factor),
                    'factor_mean': low_liquidity_factor.mean(),
                    'factor_std': low_liquidity_factor.std(),
                    'coverage_ratio': len(low_liquidity_factor) / len(factor_values)
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Liquidity stress test failed: {str(e)}"}
    
    def _test_factor_decay(self, factor_values: pd.Series) -> Dict[str, Any]:
        """测试因子时间衰减"""
        try:
            results = {}
            
            # 按时间分段计算因子统计特征
            dates = factor_values.index.get_level_values('date')
            date_range = dates.max() - dates.min()
            
            if date_range.days < 60:
                return {"error": "Insufficient time range for decay test"}
            
            # 分为前半段和后半段
            mid_date = dates.min() + date_range / 2
            
            early_factor = factor_values[dates <= mid_date]
            late_factor = factor_values[dates > mid_date]
            
            if len(early_factor) > 0 and len(late_factor) > 0:
                results['time_decay_analysis'] = {
                    'early_period': {
                        'mean': early_factor.mean(),
                        'std': early_factor.std(),
                        'count': len(early_factor)
                    },
                    'late_period': {
                        'mean': late_factor.mean(),
                        'std': late_factor.std(),
                        'count': len(late_factor)
                    },
                    'stability_score': 1 - abs(early_factor.mean() - late_factor.mean()) / (early_factor.std() + 1e-6)
                }
            
            return results
            
        except Exception as e:
            return {"error": f"Factor decay test failed: {str(e)}"}
    
    def _test_data_integrity(self, factor_values: pd.Series) -> Dict[str, Any]:
        """测试数据完整性"""
        try:
            results = {}
            
            # 缺失值分析
            total_values = len(factor_values)
            missing_values = factor_values.isna().sum()
            missing_ratio = missing_values / total_values
            
            # 异常值检测 (使用IQR方法)
            Q1 = factor_values.quantile(0.25)
            Q3 = factor_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = factor_values[(factor_values < lower_bound) | (factor_values > upper_bound)]
            outlier_ratio = len(outliers) / total_values
            
            results['data_quality'] = {
                'total_observations': total_values,
                'missing_count': int(missing_values),
                'missing_ratio': missing_ratio,
                'outlier_count': len(outliers),
                'outlier_ratio': outlier_ratio,
                'data_range': {
                    'min': factor_values.min(),
                    'max': factor_values.max(),
                    'q1': Q1,
                    'q3': Q3
                }
            }
            
            # 数据完整性评分
            integrity_score = 1.0
            if missing_ratio > 0.1:
                integrity_score -= 0.3
            if outlier_ratio > 0.05:
                integrity_score -= 0.2
            
            results['integrity_score'] = max(0, integrity_score)
            
            return results
            
        except Exception as e:
            return {"error": f"Data integrity test failed: {str(e)}"}


class UnifiedFactorValidator:
    """
    统一因子验证器 - 整合IC分析、分层回测和压力测试
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedFactorValidator")
        
        # 初始化组件
        self.ic_analyzer = ICAnalyzer()
        self.layered_backtester = LayeredBacktester(n_layers=self.config.get('n_layers', 5))
        self.stress_tester = StressTestEngine()
        
        self.logger.info("Unified Factor Validator initialized")
    
    async def comprehensive_validation(self, factor_values: pd.Series, 
                                     market_data: Dict[str, pd.Series],
                                     validation_periods: List[int] = [1, 5, 10]) -> FactorPerformanceMetrics:
        """
        综合因子验证
        
        Args:
            factor_values: 因子值 (MultiIndex: date, symbol)
            market_data: 市场数据
            validation_periods: 验证持有期列表
            
        Returns:
            因子性能指标
        """
        factor_name = getattr(factor_values, 'name', 'unnamed_factor')
        self.logger.info(f"Starting comprehensive validation for factor: {factor_name}")
        
        try:
            # 1. 基础统计分析
            basic_stats = self._calculate_basic_statistics(factor_values)
            
            # 2. IC分析
            ic_results = {}
            if 'close' in market_data:
                close_prices = market_data['close']
                
                for period in validation_periods:
                    # 计算未来收益率
                    future_returns = self._calculate_future_returns(close_prices, period)
                    
                    # 计算IC
                    ic_series = self.ic_analyzer.calculate_ic(factor_values, future_returns)
                    
                    # IC性能分析
                    ic_result = self.ic_analyzer.analyze_ic_performance(ic_series, factor_name, period)
                    ic_results[period] = ic_result
            
            # 3. 分层回测
            layered_returns = {}
            long_short_return = 0.0
            
            if 'close' in market_data and len(validation_periods) > 0:
                future_returns = self._calculate_future_returns(market_data['close'], validation_periods[0])
                layered_returns = self.layered_backtester.run_layered_backtest(factor_values, future_returns)
                
                # 计算多空收益
                if len(layered_returns) >= 2:
                    top_layer = max(layered_returns.keys())
                    bottom_layer = min(layered_returns.keys())
                    if not np.isnan(layered_returns[top_layer]) and not np.isnan(layered_returns[bottom_layer]):
                        long_short_return = layered_returns[top_layer] - layered_returns[bottom_layer]
            
            # 4. 压力测试
            stress_test_results = self.stress_tester.run_stress_tests(factor_values, market_data)
            
            # 5. 汇总结果
            sample_period = self._get_sample_period(factor_values)
            
            metrics = FactorPerformanceMetrics(
                factor_name=factor_name,
                sample_period=sample_period,
                total_observations=basic_stats['total_observations'],
                missing_ratio=basic_stats['missing_ratio'],
                mean=basic_stats['mean'],
                std=basic_stats['std'],
                skewness=basic_stats['skewness'],
                kurtosis=basic_stats['kurtosis'],
                ic_results=ic_results,
                layered_returns=layered_returns,
                long_short_return=long_short_return,
                stress_test_results=stress_test_results
            )
            
            self.logger.info(f"Comprehensive validation completed for factor: {factor_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {str(e)}")
            raise
    
    def _calculate_basic_statistics(self, factor_values: pd.Series) -> Dict[str, Any]:
        """计算基础统计信息"""
        try:
            total_observations = len(factor_values)
            missing_count = factor_values.isna().sum()
            missing_ratio = missing_count / total_observations if total_observations > 0 else 1.0
            
            valid_values = factor_values.dropna()
            
            if len(valid_values) > 0:
                return {
                    'total_observations': total_observations,
                    'missing_ratio': missing_ratio,
                    'mean': valid_values.mean(),
                    'std': valid_values.std(),
                    'skewness': valid_values.skew(),
                    'kurtosis': valid_values.kurtosis()
                }
            else:
                return {
                    'total_observations': total_observations,
                    'missing_ratio': missing_ratio,
                    'mean': np.nan,
                    'std': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating basic statistics: {str(e)}")
            return {
                'total_observations': 0,
                'missing_ratio': 1.0,
                'mean': np.nan,
                'std': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan
            }
    
    def _calculate_future_returns(self, close_prices: pd.Series, period: int) -> pd.Series:
        """计算未来收益率"""
        try:
            # 按股票分组计算未来收益率
            future_returns = close_prices.groupby(level='symbol').apply(
                lambda x: (x.shift(-period) / x) - 1
            )
            future_returns.name = f'future_return_{period}d'
            return future_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating future returns: {str(e)}")
            return pd.Series(dtype=float)
    
    def _get_sample_period(self, factor_values: pd.Series) -> Tuple[str, str]:
        """获取样本期间"""
        try:
            dates = factor_values.index.get_level_values('date')
            start_date = dates.min().strftime('%Y-%m-%d')
            end_date = dates.max().strftime('%Y-%m-%d')
            return (start_date, end_date)
        except:
            return ('unknown', 'unknown')
    
    def generate_validation_report(self, metrics: FactorPerformanceMetrics) -> Dict[str, Any]:
        """生成验证报告"""
        try:
            report = {
                "factor_info": {
                    "name": metrics.factor_name,
                    "sample_period": metrics.sample_period,
                    "total_observations": metrics.total_observations,
                    "missing_ratio": f"{metrics.missing_ratio:.2%}"
                },
                "basic_statistics": {
                    "mean": round(metrics.mean, 4) if not np.isnan(metrics.mean) else None,
                    "std": round(metrics.std, 4) if not np.isnan(metrics.std) else None,
                    "skewness": round(metrics.skewness, 4) if not np.isnan(metrics.skewness) else None,
                    "kurtosis": round(metrics.kurtosis, 4) if not np.isnan(metrics.kurtosis) else None
                },
                "ic_analysis": {},
                "layered_performance": {},
                "stress_test_summary": {},
                "overall_score": 0.0,
                "recommendations": []
            }
            
            # IC分析摘要
            for period, ic_result in metrics.ic_results.items():
                report["ic_analysis"][f"period_{period}d"] = {
                    "ic_mean": round(ic_result.ic_mean, 4) if not np.isnan(ic_result.ic_mean) else None,
                    "ic_ir": round(ic_result.ic_ir, 4) if not np.isnan(ic_result.ic_ir) else None,
                    "ic_positive_ratio": f"{ic_result.ic_positive_ratio:.2%}",
                    "performance": self._evaluate_ic_performance(ic_result)
                }
            
            # 分层回测摘要
            if metrics.layered_returns:
                report["layered_performance"] = {
                    "layer_returns": {f"layer_{k}": round(v, 4) if not np.isnan(v) else None 
                                    for k, v in metrics.layered_returns.items()},
                    "long_short_return": round(metrics.long_short_return, 4) if not np.isnan(metrics.long_short_return) else None,
                    "monotonicity": self._check_monotonicity(metrics.layered_returns)
                }
            
            # 压力测试摘要
            if metrics.stress_test_results:
                report["stress_test_summary"] = {
                    "data_integrity_score": metrics.stress_test_results.get('data_integrity', {}).get('integrity_score', 0),
                    "extreme_market_stability": self._summarize_stress_results(metrics.stress_test_results),
                    "overall_robustness": self._evaluate_robustness(metrics.stress_test_results)
                }
            
            # 综合评分
            report["overall_score"] = self._calculate_overall_score(metrics)
            
            # 改进建议
            report["recommendations"] = self._generate_recommendations(metrics)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {str(e)}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    def _evaluate_ic_performance(self, ic_result: ICAnalysisResult) -> str:
        """评估IC性能"""
        if np.isnan(ic_result.ic_ir):
            return "insufficient_data"
        
        if ic_result.ic_ir > 0.5:
            return "excellent"
        elif ic_result.ic_ir > 0.3:
            return "good"
        elif ic_result.ic_ir > 0.1:
            return "fair"
        else:
            return "poor"
    
    def _check_monotonicity(self, layered_returns: Dict[int, float]) -> str:
        """检查分层收益单调性"""
        if len(layered_returns) < 2:
            return "insufficient_layers"
        
        sorted_layers = sorted(layered_returns.keys())
        returns_sequence = [layered_returns[layer] for layer in sorted_layers if not np.isnan(layered_returns[layer])]
        
        if len(returns_sequence) < 2:
            return "insufficient_data"
        
        increasing = all(returns_sequence[i] <= returns_sequence[i+1] for i in range(len(returns_sequence)-1))
        decreasing = all(returns_sequence[i] >= returns_sequence[i+1] for i in range(len(returns_sequence)-1))
        
        if increasing:
            return "monotonic_increasing"
        elif decreasing:
            return "monotonic_decreasing"
        else:
            return "non_monotonic"
    
    def _summarize_stress_results(self, stress_results: Dict[str, Any]) -> str:
        """汇总压力测试结果"""
        if 'error' in stress_results:
            return "test_failed"
        
        stability_score = 0
        total_tests = 0
        
        # 评估各项压力测试
        for test_name, result in stress_results.items():
            if isinstance(result, dict) and 'error' not in result:
                total_tests += 1
                # 简单评分逻辑
                stability_score += 1
        
        if total_tests == 0:
            return "no_tests"
        
        stability_ratio = stability_score / total_tests
        if stability_ratio > 0.8:
            return "highly_stable"
        elif stability_ratio > 0.6:
            return "stable"
        elif stability_ratio > 0.4:
            return "moderately_stable"
        else:
            return "unstable"
    
    def _evaluate_robustness(self, stress_results: Dict[str, Any]) -> float:
        """评估整体鲁棒性"""
        if 'data_integrity' in stress_results:
            integrity_data = stress_results['data_integrity']
            if isinstance(integrity_data, dict) and 'integrity_score' in integrity_data:
                return integrity_data['integrity_score']
        
        return 0.5  # 默认中等鲁棒性
    
    def _calculate_overall_score(self, metrics: FactorPerformanceMetrics) -> float:
        """计算综合评分"""
        try:
            score = 0.0
            weights = {
                'ic_performance': 0.4,
                'monotonicity': 0.2,
                'data_quality': 0.2,
                'robustness': 0.2
            }
            
            # IC性能评分
            if metrics.ic_results:
                avg_ic_ir = np.mean([ic.ic_ir for ic in metrics.ic_results.values() if not np.isnan(ic.ic_ir)])
                if not np.isnan(avg_ic_ir):
                    ic_score = min(1.0, max(0.0, avg_ic_ir / 0.5))  # 标准化到[0,1]
                    score += weights['ic_performance'] * ic_score
            
            # 单调性评分
            if metrics.layered_returns:
                mono_result = self._check_monotonicity(metrics.layered_returns)
                mono_score = {
                    'monotonic_increasing': 1.0,
                    'monotonic_decreasing': 1.0,
                    'non_monotonic': 0.5,
                    'insufficient_data': 0.0
                }.get(mono_result, 0.0)
                score += weights['monotonicity'] * mono_score
            
            # 数据质量评分
            data_quality_score = 1 - metrics.missing_ratio
            score += weights['data_quality'] * data_quality_score
            
            # 鲁棒性评分
            robustness_score = self._evaluate_robustness(metrics.stress_test_results)
            score += weights['robustness'] * robustness_score
            
            return round(score, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {str(e)}")
            return 0.0
    
    def _generate_recommendations(self, metrics: FactorPerformanceMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        try:
            # 数据质量建议
            if metrics.missing_ratio > 0.1:
                recommendations.append("高缺失值比例，建议检查数据源质量或调整数据清洗策略")
            
            # IC表现建议
            if metrics.ic_results:
                avg_ic_ir = np.mean([ic.ic_ir for ic in metrics.ic_results.values() if not np.isnan(ic.ic_ir)])
                if not np.isnan(avg_ic_ir) and avg_ic_ir < 0.1:
                    recommendations.append("IC信息比率偏低，建议重新审视因子构造逻辑")
            
            # 分层表现建议
            if metrics.layered_returns:
                mono_result = self._check_monotonicity(metrics.layered_returns)
                if mono_result == 'non_monotonic':
                    recommendations.append("分层收益缺乏单调性，建议检查因子有效性")
            
            # 鲁棒性建议
            robustness_score = self._evaluate_robustness(metrics.stress_test_results)
            if robustness_score < 0.5:
                recommendations.append("因子鲁棒性较弱，建议增强极端市场条件下的稳定性")
            
            if not recommendations:
                recommendations.append("因子整体表现良好，可考虑进一步优化或组合使用")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("建议联系技术支持获取详细分析")
        
        return recommendations


# 创建全局统一因子验证器实例
unified_factor_validator = UnifiedFactorValidator()