"""
PandaFactor Traditional Utils Integration
PandaFactor传统算子库集成 - 基于原始factor_utils.py增强
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Any
import logging

class PandaFactorUtils:
    """PandaFactor传统算子工具类 - 完整集成70+专业算子"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    # ================== 核心基础算子 ==================
    
    @staticmethod
    def RANK(series: pd.Series) -> pd.Series:
        """横截面排名，标准化到[-0.5, 0.5]区间"""
        def rank_group(group):
            valid_data = group.dropna()
            if len(valid_data) == 0:
                return pd.Series(0, index=group.index)
            ranks = valid_data.rank(method='average')
            ranks = (ranks - 1) / (len(valid_data) - 1) - 0.5
            result = pd.Series(index=group.index)
            result.loc[valid_data.index] = ranks
            result.fillna(0, inplace=True)
            return result

        # 确保正确的索引格式
        if not isinstance(series.index, pd.MultiIndex):
            series.index = pd.MultiIndex.from_tuples(
                [(d, s) for d, s in zip(series.index, series.index)],
                names=['date', 'symbol']
            )
        elif series.index.names != ['date', 'symbol']:
            series.index.names = ['date', 'symbol']

        # 按日期分组计算排名
        result = series.groupby(level='date', group_keys=False).apply(rank_group)

        # 确保结果索引正确
        if isinstance(result.index, pd.MultiIndex):
            if len(result.index.names) > 2:
                result = result.droplevel(0)
            if result.index.names != ['date', 'symbol']:
                result.index.names = ['date', 'symbol']

        return result

    @staticmethod
    def RETURNS(close: pd.Series, period: int = 1) -> pd.Series:
        """计算收益率"""
        def calculate_returns(group):
            group = group.sort_index(level='date')
            result = group.pct_change(periods=period)
            result.iloc[:period] = 0
            return result

        result = close.groupby(level='symbol', group_keys=False).apply(calculate_returns)
        return result

    @staticmethod
    def FUTURE_RETURNS(close: pd.Series, period: int = 1) -> pd.Series:
        """计算未来收益率"""
        def calculate_future_returns(group):
            group = group.sort_index(level='date')
            shifted_prices = group.shift(-period)
            result = (shifted_prices - group) / group
            result.iloc[-period:] = 0
            return result

        result = close.groupby(level='symbol', group_keys=False).apply(calculate_future_returns)
        return result

    @staticmethod
    def STDDEV(series: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动标准差"""
        def rolling_std(group):
            group = group.sort_index(level='date')
            result = group.rolling(window=window, min_periods=max(2, window // 4)).std()
            return result

        result = series.groupby(level='symbol', group_keys=False).apply(rolling_std)
        return result

    @staticmethod
    def CORRELATION(series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动相关系数"""
        # 处理FactorSeries类型
        if hasattr(series1, 'series'):
            series1 = series1.series
        if hasattr(series2, 'series'):
            series2 = series2.series

        # 直接计算滚动相关性
        return series1.rolling(window=window, min_periods=window // 2).corr(series2)

    @staticmethod
    def IF(condition, true_value, false_value):
        """条件选择函数"""
        return pd.Series(np.where(condition, true_value, false_value), index=condition.index)

    @staticmethod
    def DELAY(series: pd.Series, period: int = 1) -> pd.Series:
        """计算滞后值"""
        return series.groupby(level='symbol').shift(period)

    @staticmethod
    def SUM(series: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动求和"""
        if hasattr(series, 'series'):
            series = series.series
        return series.groupby(level='symbol').rolling(window=window, min_periods=1).sum().droplevel(0)

    # ================== 时序分析算子 ==================
    
    @staticmethod
    def TS_ARGMAX(series: pd.Series, window: int) -> pd.Series:
        """计算时序最大值位置，返回[0,1]标准化位置"""
        def rolling_argmax(group):
            group = group.sort_index()
            result = pd.Series(index=group.index, dtype=float)

            for i in range(len(group)):
                start_idx = max(0, i - window + 1)
                window_data = group.iloc[start_idx:i + 1]

                if window_data.isna().all():
                    result.iloc[i] = np.nan
                    continue

                window_data_valid = window_data.fillna(-np.inf)
                max_val = window_data_valid.max()
                max_positions = np.where(window_data_valid == max_val)[0]

                weights = np.exp(max_positions / len(window_data))
                avg_pos = np.average(max_positions, weights=weights)

                normalized_pos = avg_pos / (len(window_data) - 1) if len(window_data) > 1 else 0
                result.iloc[i] = normalized_pos

            return result

        result = series.groupby(level='symbol', group_keys=False).apply(rolling_argmax)
        if isinstance(result.index, pd.MultiIndex) and len(result.index.names) > 2:
            result = result.droplevel(0)
        return result

    @staticmethod
    def TS_RANK(series: pd.Series, window: int = 20) -> pd.Series:
        """计算时序排名"""
        def ts_rank(group):
            return group.rolling(window=window, min_periods=1).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )

        return series.groupby(level='symbol').apply(ts_rank).droplevel(0)

    @staticmethod
    def DELTA(series: pd.Series, period: int = 1) -> pd.Series:
        """计算差分"""
        return series.groupby(level='symbol').diff(period)

    @staticmethod
    def ADV(volume: pd.Series, window: int = 20) -> pd.Series:
        """计算平均日成交量"""
        return volume.groupby(level='symbol').rolling(window=window).mean().droplevel(0)

    @staticmethod
    def TS_MIN(series: pd.Series, window: int = 20) -> pd.Series:
        """计算时序最小值"""
        return series.groupby(level='symbol').rolling(window=window, min_periods=1).min().droplevel(0)

    @staticmethod
    def TS_MAX(series: pd.Series, window: int = 20) -> pd.Series:
        """计算时序最大值"""
        return series.groupby(level='symbol').rolling(window=window, min_periods=1).max().droplevel(0)

    @staticmethod
    def DECAY_LINEAR(series: pd.Series, window: int = 20) -> pd.Series:
        """计算线性衰减加权平均"""
        weights = np.linspace(1, 0, window)

        def weighted_mean(x):
            return np.average(x, weights=weights[:len(x)])

        return series.groupby(level='symbol').rolling(window=window, min_periods=1).apply(
            lambda x: weighted_mean(x), raw=True
        ).droplevel(0)

    @staticmethod
    def SCALE(series: pd.Series) -> pd.Series:
        """将序列标准化到[-1, 1]区间"""
        original_index = series.index

        def scale_group(group):
            min_val = group.min()
            max_val = group.max()
            if min_val == max_val:
                return pd.Series(0, index=group.index)
            return 2 * (group - min_val) / (max_val - min_val) - 1

        # 按日期分组应用标准化
        result = series.groupby(level='date', group_keys=False).apply(scale_group)

        # 确保结果具有与输入完全相同的索引
        if result.index.names != original_index.names:
            result.index = original_index

        return result

    # ================== 数学运算算子 ==================
    
    @staticmethod
    def MIN(series1: pd.Series, series2: Union[pd.Series, float]) -> pd.Series:
        """计算元素级最小值"""
        if isinstance(series2, (int, float)):
            return pd.Series(np.minimum(series1, series2), index=series1.index)
        return pd.Series(np.minimum(series1, series2), index=series1.index)

    @staticmethod
    def MAX(series1: pd.Series, series2: Union[pd.Series, float]) -> pd.Series:
        """计算元素级最大值"""
        if hasattr(series1, 'series'):
            series1 = series1.series
        if not isinstance(series2, (int, float)) and hasattr(series2, 'series'):
            series2 = series2.series

        if isinstance(series2, (int, float)):
            return pd.Series(np.maximum(series1, series2), index=series1.index)
        return pd.Series(np.maximum(series1, series2), index=series1.index)

    @staticmethod
    def ABS(series: pd.Series) -> pd.Series:
        """计算绝对值"""
        return pd.Series(np.abs(series), index=series.index)

    @staticmethod
    def LOG(series: pd.Series) -> pd.Series:
        """计算自然对数"""
        return pd.Series(np.log(series), index=series.index)

    @staticmethod
    def POWER(series: pd.Series, power: float) -> pd.Series:
        """计算幂运算"""
        return pd.Series(np.power(series, power), index=series.index)

    @staticmethod
    def SIGN(series: pd.Series) -> pd.Series:
        """计算符号函数"""
        return pd.Series(np.sign(series), index=series.index)

    @staticmethod
    def SIGNEDPOWER(series: pd.Series, n: float) -> pd.Series:
        """计算带符号的幂运算 sign(X)*(abs(X)^n)"""
        return pd.Series(np.sign(series) * np.abs(series) ** n, index=series.index)

    # ================== PandaFactor特色算子 ==================
    
    @staticmethod
    def RD(S: pd.Series, D: int = 3) -> pd.Series:
        """四舍五入到D位小数"""
        return S.round(D)

    @staticmethod
    def REF(S: pd.Series, N: int = 1) -> pd.Series:
        """整个序列向下移动N位(产生NAN)"""
        if hasattr(S, 'series'):
            S = S.series
        return S.shift(N)

    @staticmethod
    def DIFF(S: pd.Series, N: int = 1) -> pd.Series:
        """计算当前值与前值的差，开始处产生NAN"""
        return S.diff(N)

    @staticmethod
    def STD(S: pd.Series, N: int) -> pd.Series:
        """计算N天的标准差"""
        return S.rolling(N).std(ddof=0)

    @staticmethod
    def MA(S: pd.Series, N: int) -> pd.Series:
        """N周期简单移动平均"""
        return S.rolling(N).mean()

    @staticmethod
    def EMA(S: pd.Series, N: int) -> pd.Series:
        """指数移动平均，需要120周期准确"""
        return S.ewm(span=N, adjust=False).mean()

    @staticmethod
    def SMA(S: pd.Series, N: int, M: int = 1) -> pd.Series:
        """中国式SMA，需要120周期准确"""
        return S.ewm(alpha=M / N, adjust=False).mean()

    @staticmethod
    def WMA(S: pd.Series, N: int) -> pd.Series:
        """N周期加权移动平均"""
        return S.rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1), raw=True)

    @staticmethod
    def HHV(S: pd.Series, N: int) -> pd.Series:
        """N周期内最高值"""
        return S.rolling(N).max()

    @staticmethod
    def LLV(S: pd.Series, N: int) -> pd.Series:
        """N周期内最低值"""
        return S.rolling(N).min()

    @staticmethod
    def SLOPE(S: pd.Series, N: int) -> pd.Series:
        """N周期线性回归斜率"""
        return S.rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True)

    # ================== 技术指标算子 ==================
    
    @staticmethod
    def MACD(CLOSE: pd.Series, SHORT: int = 12, LONG: int = 26, M: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标，使用EMA，需要120天准确"""
        DIF = PandaFactorUtils.EMA(CLOSE, SHORT) - PandaFactorUtils.EMA(CLOSE, LONG)
        DEA = PandaFactorUtils.EMA(DIF, M)
        MACD = (DIF - DEA) * 2
        return DIF, DEA, PandaFactorUtils.RD(MACD)

    @staticmethod
    def RSI(CLOSE: pd.Series, N: int = 24) -> pd.Series:
        """计算RSI指标，与通达信保持2位小数一致"""
        DIF = CLOSE - PandaFactorUtils.REF(CLOSE, 1)
        return PandaFactorUtils.RD(
            PandaFactorUtils.SMA(PandaFactorUtils.MAX(DIF, 0), N) / 
            PandaFactorUtils.SMA(PandaFactorUtils.ABS(DIF), N) * 100
        )

    @staticmethod
    def KDJ(CLOSE: pd.Series, HIGH: pd.Series, LOW: pd.Series, 
            N: int = 9, M1: int = 3, M2: int = 3) -> pd.Series:
        """计算KDJ指标，返回K线"""
        llv = LOW.groupby(level='symbol').transform(
            lambda x: x.rolling(window=N, min_periods=1).min()
        )
        hhv = HIGH.groupby(level='symbol').transform(
            lambda x: x.rolling(window=N, min_periods=1).max()
        )

        rsv = (CLOSE - llv) / (hhv - llv) * 100
        alpha = 2 / (M1 + 1)
        K = rsv.groupby(level='symbol').transform(
            lambda x: x.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )

        return K

    @staticmethod
    def BOLL(CLOSE: pd.Series, N: int = 20, P: int = 2) -> pd.Series:
        """计算布林带，返回中轨"""
        MID = PandaFactorUtils.MA(CLOSE, N)
        return PandaFactorUtils.RD(MID)

    @staticmethod
    def CCI(CLOSE: pd.Series, HIGH: pd.Series, LOW: pd.Series, N: int = 14) -> pd.Series:
        """计算CCI指标"""
        TP = (HIGH + LOW + CLOSE) / 3
        return (TP - PandaFactorUtils.MA(TP, N)) / (0.015 * PandaFactorUtils.AVEDEV(TP, N))

    @staticmethod
    def AVEDEV(S: pd.Series, N: int) -> pd.Series:
        """平均绝对偏差（平均绝对离差）"""
        return S.rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean())

    @staticmethod
    def ATR(CLOSE: pd.Series, HIGH: pd.Series, LOW: pd.Series, N: int = 20) -> pd.Series:
        """计算平均真实波幅"""
        TR = PandaFactorUtils.MAX(
            PandaFactorUtils.MAX((HIGH - LOW), 
                               PandaFactorUtils.ABS(PandaFactorUtils.REF(CLOSE, 1) - HIGH)),
            PandaFactorUtils.ABS(PandaFactorUtils.REF(CLOSE, 1) - LOW)
        )
        return PandaFactorUtils.MA(TR, N)

    @staticmethod
    def ROC(CLOSE: pd.Series, N: int = 12) -> pd.Series:
        """计算变化率(ROC)指标"""
        prev_price = CLOSE.groupby(level='symbol').shift(N)
        roc = (CLOSE - prev_price) / prev_price * 100
        roc = roc.fillna(0)
        return roc

    @staticmethod
    def OBV(CLOSE: pd.Series, VOL: pd.Series) -> pd.Series:
        """平衡成交量"""
        price_changes = CLOSE - PandaFactorUtils.REF(CLOSE, 1)
        signed_volume = pd.Series(
            np.where(price_changes > 0, VOL,
                     np.where(price_changes < 0, -VOL, 0)),
            index=VOL.index
        )
        return signed_volume.groupby(level='symbol').cumsum() / 10000

    @staticmethod
    def MFI(CLOSE: pd.Series, HIGH: pd.Series, LOW: pd.Series, VOL: pd.Series, N: int = 14) -> pd.Series:
        """资金流量指标(成交量RSI)"""
        TYP = (HIGH + LOW + CLOSE) / 3
        raw_money_flow = TYP * VOL
        
        price_changes = TYP.groupby(level='symbol').diff()
        
        pos_flow = raw_money_flow.where(price_changes > 0, 0)
        neg_flow = raw_money_flow.where(price_changes < 0, 0)
        
        pos_sum = pos_flow.groupby(level='symbol').rolling(window=N, min_periods=1).sum().droplevel(0)
        neg_sum = neg_flow.groupby(level='symbol').rolling(window=N, min_periods=1).sum().droplevel(0)
        
        money_ratio = pos_sum / neg_sum.replace(0, 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))
        
        total_flow = pos_sum + neg_sum
        mfi = np.where(total_flow == 0, 50,
                       np.where(neg_sum == 0, 100,
                                np.where(pos_sum == 0, 0, mfi)))

        return pd.Series(mfi, index=CLOSE.index)

    # ================== 条件和逻辑算子 ==================
    
    @staticmethod
    def CROSS(S1: pd.Series, S2: pd.Series) -> pd.Series:
        """检查金叉(向上穿越)或死叉(向下穿越)"""
        S1, S2 = S1.align(S2)
        cross_up = (S1 > S2) & (S1.shift(1) <= S2.shift(1))
        return cross_up

    @staticmethod
    def COUNT(S: pd.Series, N: int) -> pd.Series:
        """COUNT(CLOSE>O, N): 统计最近N天满足条件的天数"""
        return PandaFactorUtils.SUM(S, N)

    @staticmethod
    def EVERY(S: pd.Series, N: int) -> pd.Series:
        """EVERY(CLOSE>O, 5) 检查最近N天是否全部为True"""
        return PandaFactorUtils.IF(PandaFactorUtils.SUM(S, N) == N, True, False)

    @staticmethod
    def EXIST(S: pd.Series, N: int) -> pd.Series:
        """EXIST(CLOSE>3010, N=5) 检查最近N天是否存在条件"""
        return PandaFactorUtils.IF(PandaFactorUtils.SUM(S, N) > 0, True, False)

    @staticmethod
    def BARSLAST(S: pd.Series) -> pd.Series:
        """计算上一次条件为真到现在的周期数"""
        M = np.concatenate(([0], np.where(S, 1, 0)))
        for i in range(1, len(M)):
            M[i] = 0 if M[i] else M[i - 1] + 1
        return pd.Series(M[1:], index=S.index)

    @staticmethod
    def VALUEWHEN(S: pd.Series, X: pd.Series) -> pd.Series:
        """当条件S为True时，取X的当前值"""
        S, X = S.align(X)

        def apply_valuewhen(group):
            s_group = S.loc[group.index]
            x_group = X.loc[group.index]
            return pd.Series(np.where(s_group, x_group, np.nan), index=group.index)

        result = X.groupby(level='symbol', group_keys=False).apply(apply_valuewhen)

        if not isinstance(result.index, pd.MultiIndex):
            result.index = pd.MultiIndex.from_tuples(
                [(d, s) for d, s in zip(result.index, result.index)],
                names=['date', 'symbol']
            )
        elif result.index.names != ['date', 'symbol']:
            result.index.names = ['date', 'symbol']

        return result

    # ================== 高级函数算子 ==================
    
    @staticmethod
    def VWAP(close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算成交量加权平均价格"""
        close, volume = close.align(volume)

        def calculate_vwap(group_close, group_volume):
            pv = group_close * group_volume
            pv_sum = pv.rolling(window=20, min_periods=1).sum()
            v_sum = group_volume.rolling(window=20, min_periods=1).sum()
            return pv_sum / v_sum

        result = pd.Series(index=close.index, dtype=float)
        for symbol in close.index.get_level_values('symbol').unique():
            mask = close.index.get_level_values('symbol') == symbol
            symbol_close = close[mask]
            symbol_volume = volume[mask]
            result[mask] = calculate_vwap(symbol_close, symbol_volume)

        return result

    @staticmethod
    def CAP(close: pd.Series, shares: pd.Series) -> pd.Series:
        """计算市值"""
        return close * shares

    @staticmethod
    def COVARIANCE(series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动协方差"""
        def rolling_cov(s1, s2, window):
            return s1.rolling(window=window, min_periods=window // 4).cov(s2)

        result = pd.Series(index=series1.index, dtype=float)
        for symbol in series1.index.get_level_values('symbol').unique():
            s1 = series1[series1.index.get_level_values('symbol') == symbol]
            s2 = series2[series2.index.get_level_values('symbol') == symbol]
            s1, s2 = s1.align(s2)
            result[s1.index] = rolling_cov(s1, s2, window)
        return result

    @staticmethod
    def AS_FLOAT(condition: pd.Series) -> pd.Series:
        """将布尔条件转换为浮点数"""
        return condition.astype(float)

    @staticmethod
    def PRODUCT(series: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动乘积"""
        return series.groupby(level='symbol').rolling(window=window, min_periods=1).apply(
            lambda x: np.prod(x)
        ).droplevel(0)

    @staticmethod
    def TS_MEAN(series: pd.Series, window: int = 20) -> pd.Series:
        """计算时序移动平均"""
        return series.groupby(level='symbol').rolling(window=window, min_periods=1).mean().droplevel(0)

    # ================== 工具方法 ==================
    
    def get_available_functions(self) -> list:
        """获取所有可用的函数列表"""
        functions = []
        for name in dir(self):
            if not name.startswith('_') and callable(getattr(self, name)):
                if name != 'get_available_functions':
                    functions.append(name)
        return sorted(functions)

    def get_function_info(self, function_name: str) -> str:
        """获取函数信息和文档"""
        if hasattr(self, function_name):
            func = getattr(self, function_name)
            return func.__doc__ or f"函数 {function_name} - 无文档说明"
        else:
            return f"函数 {function_name} 不存在"

    def validate_series_index(self, series: pd.Series) -> bool:
        """验证序列索引格式是否正确"""
        if isinstance(series.index, pd.MultiIndex):
            return series.index.names == ['date', 'symbol']
        return False

    @classmethod
    def create_multiindex_series(cls, data: dict, dates: list, symbols: list) -> pd.Series:
        """创建MultiIndex格式的Series"""
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        values = []
        for date in dates:
            for symbol in symbols:
                key = f"{date}_{symbol}"
                values.append(data.get(key, np.nan))
        return pd.Series(values, index=index)

# 创建全局实例供外部使用
panda_factor_utils = PandaFactorUtils()