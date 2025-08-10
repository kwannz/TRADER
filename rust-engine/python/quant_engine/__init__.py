"""
QuantAnalyzer Pro - 高性能量化分析引擎

这是一个基于Rust的高性能量化因子计算和回测引擎的Python绑定。
提供了各种技术分析指标和统计因子的快速计算功能。

主要功能：
- 技术指标计算 (RSI, MACD, SMA, EMA, 布林带等)
- 统计因子计算 (相关性, 贝塔系数, 夏普比率等)
- 基本面因子计算 (P/E, P/B, ROE, ROA等)
- 高性能回测引擎

示例用法:
    >>> import quant_engine as qe
    >>> import numpy as np
    >>> 
    >>> # 生成示例价格数据
    >>> prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> 
    >>> # 计算RSI
    >>> rsi = qe.calculate_rsi(prices, period=5)
    >>> print(f"RSI: {rsi[-1]:.2f}")
    >>> 
    >>> # 计算MACD
    >>> macd, signal, histogram = qe.calculate_macd(prices, 5, 10, 3)
    >>> print(f"MACD: {macd[-1]:.4f}")
"""

from typing import Tuple, Optional, List
import numpy as np

try:
    from ._core import (
        py_calculate_rsi as _calculate_rsi,
        py_calculate_macd as _calculate_macd,
        py_calculate_ma as _calculate_ma,
        py_calculate_sma as _calculate_sma,
        py_calculate_ema as _calculate_ema,
        py_calculate_bollinger_bands as _calculate_bollinger_bands,
        __version__,
    )
    _rust_available = True
except ImportError as e:
    _rust_available = False
    _import_error = e

__version__ = "0.1.0"

def _ensure_rust_available():
    """确保Rust扩展模块可用"""
    if not _rust_available:
        raise ImportError(
            f"Rust扩展模块不可用: {_import_error}. "
            "请确保已正确编译了Rust扩展。运行 'maturin develop' 来构建。"
        )

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算相对强弱指数 (RSI)
    
    RSI是一个动量振荡器，用于衡量价格变化的速度和幅度。
    RSI值在0-100之间，通常70以上认为超买，30以下认为超卖。
    
    参数:
        prices: 价格序列 (numpy数组)
        period: 计算周期，默认14
        
    返回:
        RSI值序列，前period-1个值为NaN
        
    示例:
        >>> import numpy as np
        >>> prices = np.array([44, 44.25, 44.5, 43.75, 44.5, 45, 45.25, 45.5, 45.75, 46])
        >>> rsi = calculate_rsi(prices, period=5)
        >>> print(rsi[-1])  # 最新的RSI值
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    if len(prices) < period + 1:
        raise ValueError(f"价格序列长度({len(prices)})不足，至少需要{period + 1}个数据点")
    
    return _calculate_rsi(prices, period)

def calculate_macd(
    prices: np.ndarray, 
    fast_period: int = 12, 
    slow_period: int = 26, 
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算MACD (指数平滑移动平均收敛/发散)
    
    MACD是一个趋势跟踪动量指标，由MACD线、信号线和柱状图组成。
    
    参数:
        prices: 价格序列
        fast_period: 快线EMA周期，默认12
        slow_period: 慢线EMA周期，默认26
        signal_period: 信号线EMA周期，默认9
        
    返回:
        tuple: (MACD线, 信号线, 柱状图)
        
    示例:
        >>> macd, signal, histogram = calculate_macd(prices)
        >>> # 当MACD线上穿信号线时，可能是买入信号
        >>> if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
        ...     print("MACD金叉，可能的买入信号")
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    if fast_period >= slow_period:
        raise ValueError("快线周期必须小于慢线周期")
    
    if len(prices) < slow_period + signal_period:
        raise ValueError(f"价格序列长度不足，至少需要{slow_period + signal_period}个数据点")
    
    return _calculate_macd(prices, fast_period, slow_period, signal_period)

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """
    计算简单移动平均 (SMA)
    
    SMA是最基本的移动平均类型，将指定周期内的价格简单平均。
    
    参数:
        prices: 价格序列
        period: 移动平均周期
        
    返回:
        SMA值序列，前period-1个值为NaN
        
    示例:
        >>> sma_20 = calculate_sma(prices, 20)
        >>> # 价格在SMA上方通常表示上升趋势
        >>> if prices[-1] > sma_20[-1]:
        ...     print("价格在20日均线上方")
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    return _calculate_sma(prices, period)

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    计算指数移动平均 (EMA)
    
    EMA给予近期价格更高的权重，对价格变化比SMA更敏感。
    
    参数:
        prices: 价格序列
        period: EMA周期
        
    返回:
        EMA值序列
        
    示例:
        >>> ema_12 = calculate_ema(prices, 12)
        >>> ema_26 = calculate_ema(prices, 26)
        >>> # EMA交叉策略
        >>> if ema_12[-1] > ema_26[-1]:
        ...     print("短期EMA在长期EMA上方，可能的上升趋势")
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    return _calculate_ema(prices, period)

def calculate_ma(prices: np.ndarray, period: int, ma_type: str = "sma") -> np.ndarray:
    """
    计算移动平均 (通用函数)
    
    参数:
        prices: 价格序列
        period: 移动平均周期
        ma_type: 移动平均类型，"sma"或"ema"
        
    返回:
        移动平均值序列
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    if ma_type not in ["sma", "ema", "simple", "exponential"]:
        raise ValueError("ma_type必须是'sma'、'ema'、'simple'或'exponential'")
    
    return _calculate_ma(prices, period, ma_type)

def calculate_bollinger_bands(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算布林带
    
    布林带由中轨(移动平均)和上下轨(中轨±标准差倍数)组成，
    用于判断价格的相对高低和市场波动性。
    
    参数:
        prices: 价格序列
        period: 移动平均周期，默认20
        std_dev: 标准差倍数，默认2.0
        
    返回:
        tuple: (上轨, 中轨, 下轨)
        
    示例:
        >>> upper, middle, lower = calculate_bollinger_bands(prices)
        >>> # 价格触及上轨可能超买，触及下轨可能超卖
        >>> if prices[-1] > upper[-1]:
        ...     print("价格突破布林带上轨，可能超买")
        >>> elif prices[-1] < lower[-1]:
        ...     print("价格跌破布林带下轨，可能超卖")
    """
    _ensure_rust_available()
    
    if not isinstance(prices, np.ndarray):
        prices = np.array(prices, dtype=np.float64)
    
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    
    if std_dev <= 0:
        raise ValueError("标准差倍数必须大于0")
    
    return _calculate_bollinger_bands(prices, period, std_dev)

# 便捷的别名
rsi = calculate_rsi
macd = calculate_macd
sma = calculate_sma
ema = calculate_ema
bollinger_bands = calculate_bollinger_bands

# 导出的公共API
__all__ = [
    'calculate_rsi',
    'calculate_macd', 
    'calculate_sma',
    'calculate_ema',
    'calculate_ma',
    'calculate_bollinger_bands',
    'rsi',
    'macd',
    'sma', 
    'ema',
    'bollinger_bands',
    '__version__',
]

# 模块级文档字符串
def get_info():
    """获取模块信息"""
    return {
        'version': __version__,
        'rust_available': _rust_available,
        'description': '高性能量化分析引擎',
        'supported_indicators': [
            'RSI - 相对强弱指数',
            'MACD - 指数平滑移动平均收敛/发散', 
            'SMA - 简单移动平均',
            'EMA - 指数移动平均',
            'Bollinger Bands - 布林带',
        ]
    }