"""
Alpha101因子计算器
实现30个核心Alpha因子，用于量化交易策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class Alpha101Calculator:
    """Alpha101因子计算器"""
    
    def __init__(self):
        self.factors = {}
        
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有可用因子"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"数据缺少必需列: {required_columns}")
            
            result_df = data.copy()
            
            # 计算各个Alpha因子
            factor_methods = [
                self.alpha_001, self.alpha_002, self.alpha_003, self.alpha_004, self.alpha_005,
                self.alpha_006, self.alpha_007, self.alpha_008, self.alpha_009, self.alpha_010,
                self.alpha_011, self.alpha_012, self.alpha_013, self.alpha_014, self.alpha_015,
                self.alpha_016, self.alpha_017, self.alpha_018, self.alpha_019, self.alpha_020,
                self.alpha_021, self.alpha_022, self.alpha_023, self.alpha_024, self.alpha_025,
                self.alpha_026, self.alpha_027, self.alpha_028, self.alpha_029, self.alpha_030
            ]
            
            for method in factor_methods:
                try:
                    factor_name = method.__name__
                    factor_values = method(data)
                    if factor_values is not None:
                        result_df[factor_name] = factor_values
                        logger.debug(f"计算因子成功: {factor_name}")
                except Exception as e:
                    logger.warning(f"计算因子失败 {method.__name__}: {e}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"计算Alpha101因子失败: {e}")
            return data.copy()
    
    # ============ 核心Alpha因子实现 ============
    
    def alpha_001(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#001: 价量背离因子
        (-1 * CORR(RANK(DELTA(LOG(VOLUME))), RANK(((CLOSE - OPEN) / OPEN)), 6))
        """
        try:
            volume_change = np.log(data['volume']).diff()
            price_change = (data['close'] - data['open']) / data['open']
            
            volume_rank = volume_change.rolling(window=20).rank()
            price_rank = price_change.rolling(window=20).rank()
            
            correlation = volume_rank.rolling(window=6).corr(price_rank)
            return -1 * correlation
            
        except Exception as e:
            logger.warning(f"Alpha001计算失败: {e}")
            return None
    
    def alpha_002(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#002: 开盘价动量因子
        (-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
        """
        try:
            k_percent = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            return -1 * k_percent.diff(1)
            
        except Exception as e:
            logger.warning(f"Alpha002计算失败: {e}")
            return None
    
    def alpha_003(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#003: 收盘价序列相关性
        """
        try:
            close = data['close']
            condition1 = close == close.shift(1)
            condition2 = close > close.shift(1)
            
            result = pd.Series(index=data.index, dtype=float)
            result[condition1] = 0
            result[condition2] = -1 * (close - close.shift(1))
            result[~(condition1 | condition2)] = close - close.shift(1)
            
            return result.rolling(window=6).sum()
            
        except Exception as e:
            logger.warning(f"Alpha003计算失败: {e}")
            return None
    
    def alpha_004(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#004: 趋势强度因子
        """
        try:
            condition = (data['close'].rolling(window=8).sum() + data['close'].rolling(window=20).sum()) < data['close'].rolling(window=2).sum()
            
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1
            result[~condition] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha004计算失败: {e}")
            return None
    
    def alpha_005(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#005: 价量组合因子
        """
        try:
            volume_ma = data['volume'].rolling(window=10).mean()
            volume_ratio = data['volume'] / volume_ma
            
            price_change = (data['close'] / data['open'] - 1)
            
            return -1 * price_change * volume_ratio.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha005计算失败: {e}")
            return None
    
    def alpha_006(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#006: 开盘价与收盘价相关性
        """
        try:
            open_rank = data['open'].rolling(window=10).rank()
            close_rank = data['close'].rolling(window=10).rank()
            
            return -1 * open_rank.rolling(window=5).corr(close_rank.rolling(window=5))
            
        except Exception as e:
            logger.warning(f"Alpha006计算失败: {e}")
            return None
    
    def alpha_007(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#007: 价格波动率因子
        """
        try:
            adv20 = data['volume'].rolling(window=20).mean()
            close_delta = data['close'].diff()
            ts_rank = close_delta.rolling(window=7).rank()
            
            condition = adv20 < data['volume']
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1 * ts_rank[condition]
            result[~condition] = close_delta[~condition]
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha007计算失败: {e}")
            return None
    
    def alpha_008(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#008: 价格与成交量的时间序列关系
        """
        try:
            open_delta = data['open'].diff()
            close_delta = data['close'].diff()
            volume_delta = data['volume'].diff()
            
            sum_open = open_delta.rolling(window=5).sum()
            sum_close = close_delta.rolling(window=5).sum()
            
            return -1 * (sum_open + sum_close) * volume_delta.rolling(window=10).rank()
            
        except Exception as e:
            logger.warning(f"Alpha008计算失败: {e}")
            return None
    
    def alpha_009(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#009: 高低价差动量
        """
        try:
            hl_avg = (data['high'] + data['low']) / 2
            hl_avg_delta = hl_avg.diff()
            
            condition = hl_avg_delta.rolling(window=7).min() > 0
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = hl_avg_delta[condition]
            
            condition2 = hl_avg_delta.rolling(window=7).max() < 0
            result[condition2] = -1 * hl_avg_delta[condition2]
            
            mask = ~(condition | condition2)
            result[mask] = -1 * hl_avg_delta[mask]
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha009计算失败: {e}")
            return None
    
    def alpha_010(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#010: 收盘价变化率因子
        """
        try:
            close_change = data['close'].pct_change()
            
            condition = close_change.rolling(window=20).min() > 0
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = close_change[condition]
            
            condition2 = close_change.rolling(window=20).max() < 0
            result[condition2] = -1 * close_change[condition2]
            
            mask = ~(condition | condition2)
            result[mask] = -1 * close_change[mask]
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha010计算失败: {e}")
            return None
    
    def alpha_011(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#011: 成交量价格确认因子
        """
        try:
            vwap = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).rolling(window=6).sum() / data['volume'].rolling(window=6).sum()
            close_vwap_diff = data['close'] - vwap
            volume_rank = data['volume'].rolling(window=6).rank()
            
            return close_vwap_diff * volume_rank
            
        except Exception as e:
            logger.warning(f"Alpha011计算失败: {e}")
            return None
    
    def alpha_012(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#012: 开盘价收盘价偏离度
        """
        try:
            open_close_ratio = data['open'] / data['close']
            volume_rank = data['volume'].rolling(window=10).rank()
            
            return (open_close_ratio - 1) * volume_rank
            
        except Exception as e:
            logger.warning(f"Alpha012计算失败: {e}")
            return None
    
    def alpha_013(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#013: 高低价相对强度
        """
        try:
            high_low_ratio = data['high'] / data['low']
            volume_ma = data['volume'].rolling(window=5).mean()
            
            return -1 * (high_low_ratio - 1) * volume_ma
            
        except Exception as e:
            logger.warning(f"Alpha013计算失败: {e}")
            return None
    
    def alpha_014(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#014: 开盘价与前收盘价关系
        """
        try:
            open_prev_close = data['open'] / data['close'].shift(1)
            volume_corr = data['volume'].rolling(window=5).corr(data['close'].rolling(window=5))
            
            return -1 * (open_prev_close - 1) * volume_corr
            
        except Exception as e:
            logger.warning(f"Alpha014计算失败: {e}")
            return None
    
    def alpha_015(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#015: 收盘价与成交量线性关系
        """
        try:
            high_volume_corr = data['high'].rolling(window=3).corr(data['volume'].rolling(window=3))
            close_rank = data['close'].rolling(window=5).rank()
            
            return -1 * high_volume_corr * close_rank
            
        except Exception as e:
            logger.warning(f"Alpha015计算失败: {e}")
            return None
    
    def alpha_016(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#016: 高价与成交量关系
        """
        try:
            high_rank = data['high'].rolling(window=5).rank()
            volume_rank = data['volume'].rolling(window=5).rank()
            
            return -1 * high_rank.rolling(window=5).corr(volume_rank.rolling(window=5))
            
        except Exception as e:
            logger.warning(f"Alpha016计算失败: {e}")
            return None
    
    def alpha_017(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#017: 收盘价变动趋势
        """
        try:
            close_rank = data['close'].rolling(window=10).rank()
            volume_rank = data['volume'].rolling(window=5).rank()
            
            return -1 * close_rank * volume_rank
            
        except Exception as e:
            logger.warning(f"Alpha017计算失败: {e}")
            return None
    
    def alpha_018(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#018: 收盘价与标准差关系
        """
        try:
            close_std = data['close'].rolling(window=5).std()
            close_mean = data['close'].rolling(window=5).mean()
            
            return -1 * (close_std / close_mean).rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha018计算失败: {e}")
            return None
    
    def alpha_019(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#019: 收盘价延迟相关性
        """
        try:
            close_delay = data['close'].shift(5)
            volume_delay = data['volume'].shift(5)
            
            close_corr = data['close'].rolling(window=6).corr(close_delay.rolling(window=6))
            volume_sum = volume_delay.rolling(window=230).sum()
            
            return (-1 * close_corr) * volume_sum.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha019计算失败: {e}")
            return None
    
    def alpha_020(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#020: 开盘价滞后因子
        """
        try:
            open_lag = data['open'].shift(1)
            high_lag = data['high'].shift(1)
            low_lag = data['low'].shift(1)
            
            factor = ((open_lag + high_lag + low_lag) / 3) / data['close']
            
            return -1 * factor.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha020计算失败: {e}")
            return None
    
    def alpha_021(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#021: 线性回归斜率
        """
        try:
            close = data['close']
            x = np.arange(len(close))
            
            def linear_slope(y):
                if len(y) < 2:
                    return np.nan
                return np.polyfit(x[:len(y)], y, 1)[0]
            
            slopes = close.rolling(window=8).apply(linear_slope)
            volume_mean = data['volume'].rolling(window=20).mean()
            
            condition = slopes < volume_mean / 20000  # 简化条件
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1
            result[~condition] = (close / close.shift(1) - 1)
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha021计算失败: {e}")
            return None
    
    def alpha_022(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#022: 高低价差与成交量的关系
        """
        try:
            hl_corr = data['high'].rolling(window=5).corr(data['volume'].rolling(window=5))
            close_delta = data['close'].diff()
            
            return -1 * hl_corr * close_delta.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha022计算失败: {e}")
            return None
    
    def alpha_023(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#023: 高价突破因子
        """
        try:
            high_ma = data['high'].rolling(window=20).mean()
            condition = data['high'] > high_ma.shift(1)
            
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1 * data['high'][condition].diff()
            result[~condition] = 0
            
            return result.rolling(window=2).sum()
            
        except Exception as e:
            logger.warning(f"Alpha023计算失败: {e}")
            return None
    
    def alpha_024(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#024: 收盘价趋势确认
        """
        try:
            close_delta = data['close'].diff()
            condition = close_delta.rolling(window=5).sum() < 0
            
            close_ma = data['close'].rolling(window=100).mean()
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1 * (data['close'][condition] - close_ma[condition])
            result[~condition] = close_delta[~condition]
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha024计算失败: {e}")
            return None
    
    def alpha_025(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#025: 成交量加权价格趋势
        """
        try:
            volume_rank = data['volume'].rolling(window=20).rank()
            close_rank = data['close'].rolling(window=5).rank()
            high_rank = data['high'].rolling(window=5).rank()
            
            return -1 * volume_rank * close_rank * high_rank
            
        except Exception as e:
            logger.warning(f"Alpha025计算失败: {e}")
            return None
    
    def alpha_026(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#026: 高价与成交量时序关系
        """
        try:
            high_ma = data['high'].rolling(window=7).mean()
            volume_ts_sum = data['volume'].rolling(window=20).sum()
            
            return -1 * high_ma.rolling(window=7).sum() * volume_ts_sum.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha026计算失败: {e}")
            return None
    
    def alpha_027(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#027: 价格波动率与成交量关系
        """
        try:
            volume_mean = data['volume'].rolling(window=3).mean()
            close_corr = data['close'].rolling(window=3).corr(volume_mean)
            
            condition = close_corr > 0.5
            result = pd.Series(index=data.index, dtype=float)
            result[condition] = -1
            result[~condition] = 1
            
            return result
            
        except Exception as e:
            logger.warning(f"Alpha027计算失败: {e}")
            return None
    
    def alpha_028(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#028: 价格与成交量斜率关系
        """
        try:
            adv20 = data['volume'].rolling(window=20).mean()
            low_corr = data['low'].rolling(window=5).corr(adv20.rolling(window=5))
            high_corr = data['high'].rolling(window=5).corr(adv20.rolling(window=5))
            
            return -1 * (low_corr + high_corr) * data['volume'].rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha028计算失败: {e}")
            return None
    
    def alpha_029(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#029: 收盘价与延迟收盘价关系
        """
        try:
            close_delay = data['close'].shift(6)
            volume_delay = data['volume'].shift(6)
            
            close_ratio = data['close'] / close_delay
            volume_min = volume_delay.rolling(window=200).min()
            
            return close_ratio * volume_min.rolling(window=5).rank()
            
        except Exception as e:
            logger.warning(f"Alpha029计算失败: {e}")
            return None
    
    def alpha_030(self, data: pd.DataFrame) -> pd.Series:
        """
        Alpha#030: 成交量标准化因子
        """
        try:
            volume_mean = data['volume'].rolling(window=20).mean()
            volume_std = data['volume'].rolling(window=20).std()
            
            volume_norm = (data['volume'] - volume_mean) / volume_std
            close_delta = data['close'].diff()
            
            return volume_norm * close_delta.rolling(window=3).rank()
            
        except Exception as e:
            logger.warning(f"Alpha030计算失败: {e}")
            return None
    
    # ============ 辅助函数 ============
    
    def rank(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算滚动排名"""
        return df.rolling(window=window).rank()
    
    def correlation(self, x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
        """计算滚动相关性"""
        return x.rolling(window=window).corr(y)
    
    def delta(self, df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """计算差分"""
        return df.diff(period)
    
    def ts_sum(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算时间序列求和"""
        return df.rolling(window=window).sum()
    
    def ts_mean(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算时间序列均值"""
        return df.rolling(window=window).mean()
    
    def ts_min(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算时间序列最小值"""
        return df.rolling(window=window).min()
    
    def ts_max(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算时间序列最大值"""
        return df.rolling(window=window).max()
    
    def ts_std(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """计算时间序列标准差"""
        return df.rolling(window=window).std()
    
    def calculate_factor_ic(self, factor_values: pd.Series, returns: pd.Series, 
                           periods: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """计算因子IC值"""
        try:
            ic_results = {}
            
            for period in periods:
                future_returns = returns.shift(-period)
                ic = factor_values.corr(future_returns)
                ic_results[f'IC_{period}d'] = ic if not np.isnan(ic) else 0.0
            
            return ic_results
            
        except Exception as e:
            logger.error(f"计算IC值失败: {e}")
            return {f'IC_{p}d': 0.0 for p in periods}
    
    def calculate_factor_statistics(self, factor_values: pd.Series, 
                                   returns: pd.Series) -> Dict[str, Any]:
        """计算因子统计信息"""
        try:
            if factor_values.isna().all():
                return {"error": "因子值全为空"}
            
            # 基础统计
            stats = {
                "count": factor_values.count(),
                "mean": factor_values.mean(),
                "std": factor_values.std(),
                "min": factor_values.min(),
                "max": factor_values.max(),
                "skew": factor_values.skew(),
                "kurtosis": factor_values.kurtosis()
            }
            
            # IC统计
            ic_stats = self.calculate_factor_ic(factor_values, returns)
            stats.update(ic_stats)
            
            # 计算ICIR (IC信息比率)
            ic_1d = ic_stats.get('IC_1d', 0)
            if abs(ic_1d) > 0.001:  # 避免除零
                stats['ICIR'] = ic_1d / factor_values.std() if factor_values.std() > 0 else 0
            else:
                stats['ICIR'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"计算因子统计失败: {e}")
            return {"error": str(e)}

# 全局Alpha101计算器实例
alpha101_calculator = Alpha101Calculator()