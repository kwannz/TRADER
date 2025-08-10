"""
Unified Factor Interface
统一因子接口 - 整合传统PandaFactor算子与AI增强功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging
from datetime import datetime, timedelta

from ..traditional.factor_utils import PandaFactorUtils
from ..traditional.factor_base import (
    BaseFactor, FactorSeries, FormulaFactor, PythonFactor, 
    FactorLibrary, factor_library
)


class DataReader:
    """
    数据读取器 - 适配PandaFactor的数据读取接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("DataReader")
        
        # 这里将来可以集成PandaFactor的FactorReader
        # from panda_data.panda_data.factor.factor_reader import FactorReader
        # self.panda_reader = FactorReader(config)
    
    def get_base_factors(self, symbols: List[str], start_date: str, end_date: str,
                        factors: List[str] = None) -> Dict[str, pd.Series]:
        """
        获取基础因子数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            factors: 因子列表，默认获取OHLCV数据
            
        Returns:
            因子数据字典，键为因子名，值为MultiIndex Series (date, symbol)
        """
        if factors is None:
            factors = ['open', 'close', 'high', 'low', 'volume', 'amount']
        
        # TODO: 集成真实的数据读取逻辑
        # 这里提供模拟数据生成逻辑
        return self._generate_mock_data(symbols, start_date, end_date, factors)
    
    def _generate_mock_data(self, symbols: List[str], start_date: str, end_date: str,
                           factors: List[str]) -> Dict[str, pd.Series]:
        """生成模拟数据用于测试"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 创建MultiIndex
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        
        data = {}
        np.random.seed(42)  # 固定随机种子保证可重复性
        
        for factor in factors:
            if factor == 'close':
                # 生成价格序列（随机游走）
                base_price = 100
                returns = np.random.normal(0, 0.02, len(index))
                prices = base_price * np.exp(np.cumsum(returns))
                data[factor] = pd.Series(prices, index=index)
                
            elif factor == 'open':
                # 开盘价基于收盘价生成
                if 'close' in data:
                    gap = np.random.normal(0, 0.01, len(index))
                    data[factor] = data['close'] * (1 + gap)
                else:
                    data[factor] = pd.Series(np.random.normal(100, 20, len(index)), index=index)
                    
            elif factor == 'high':
                # 最高价
                base = data.get('close', pd.Series(100, index=index))
                high_ratio = np.random.uniform(1.0, 1.05, len(index))
                data[factor] = base * high_ratio
                
            elif factor == 'low':
                # 最低价
                base = data.get('close', pd.Series(100, index=index))
                low_ratio = np.random.uniform(0.95, 1.0, len(index))
                data[factor] = base * low_ratio
                
            elif factor == 'volume':
                # 成交量
                base_vol = 1000000
                vol_noise = np.random.lognormal(0, 0.5, len(index))
                data[factor] = pd.Series(base_vol * vol_noise, index=index)
                
            elif factor == 'amount':
                # 成交额 = 价格 * 成交量
                price = data.get('close', pd.Series(100, index=index))
                volume = data.get('volume', pd.Series(1000000, index=index))
                data[factor] = price * volume
                
            else:
                # 其他因子生成随机数据
                data[factor] = pd.Series(np.random.normal(0, 1, len(index)), index=index)
        
        return data


class UnifiedFactorInterface:
    """
    统一因子接口 - 集成传统算子、公式因子、Python因子和AI增强功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedFactorInterface")
        
        # 初始化组件
        self.utils = PandaFactorUtils()
        self.factor_library = factor_library
        self.data_reader = DataReader(self.config)
        
        # 加载AI增强模块
        self._initialize_ai_components()
        
        self.logger.info("Unified Factor Interface initialized")
    
    def _initialize_ai_components(self):
        """初始化AI增强组件"""
        try:
            # TODO: 集成AI因子发现模块
            # from ..ai_enhanced.ai_factor_discovery import AIFactorDiscovery
            # self.ai_discovery = AIFactorDiscovery(self.config)
            self.ai_discovery = None
            self.logger.info("AI components will be integrated in future phases")
        except ImportError as e:
            self.logger.warning(f"AI components not available: {e}")
            self.ai_discovery = None
    
    # ================== 数据获取接口 ==================
    
    def get_data(self, symbols: List[str], start_date: str, end_date: str,
                 factors: List[str] = None) -> Dict[str, pd.Series]:
        """
        获取基础数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factors: 需要的基础因子列表
            
        Returns:
            因子数据字典
        """
        return self.data_reader.get_base_factors(symbols, start_date, end_date, factors)
    
    # ================== 传统算子接口 ==================
    
    def rank(self, series: pd.Series) -> pd.Series:
        """横截面排名"""
        return self.utils.RANK(series)
    
    def returns(self, close: pd.Series, period: int = 1) -> pd.Series:
        """计算收益率"""
        return self.utils.RETURNS(close, period)
    
    def future_returns(self, close: pd.Series, period: int = 1) -> pd.Series:
        """计算未来收益率"""
        return self.utils.FUTURE_RETURNS(close, period)
    
    def correlation(self, series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动相关系数"""
        return self.utils.CORRELATION(series1, series2, window)
    
    def stddev(self, series: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动标准差"""
        return self.utils.STDDEV(series, window)
    
    def scale(self, series: pd.Series) -> pd.Series:
        """标准化到[-1, 1]区间"""
        return self.utils.SCALE(series)
    
    def ts_rank(self, series: pd.Series, window: int = 20) -> pd.Series:
        """计算时序排名"""
        return self.utils.TS_RANK(series, window)
    
    def decay_linear(self, series: pd.Series, window: int = 20) -> pd.Series:
        """线性衰减加权平均"""
        return self.utils.DECAY_LINEAR(series, window)
    
    def delay(self, series: pd.Series, period: int = 1) -> pd.Series:
        """计算滞后值"""
        return self.utils.DELAY(series, period)
    
    def delta(self, series: pd.Series, period: int = 1) -> pd.Series:
        """计算差分"""
        return self.utils.DELTA(series, period)
    
    # ================== 技术指标接口 ==================
    
    def macd(self, close: pd.Series, short: int = 12, long: int = 26, m: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        return self.utils.MACD(close, short, long, m)
    
    def rsi(self, close: pd.Series, n: int = 24) -> pd.Series:
        """计算RSI指标"""
        return self.utils.RSI(close, n)
    
    def kdj(self, close: pd.Series, high: pd.Series, low: pd.Series,
            n: int = 9, m1: int = 3, m2: int = 3) -> pd.Series:
        """计算KDJ指标"""
        return self.utils.KDJ(close, high, low, n, m1, m2)
    
    def bollinger_bands(self, close: pd.Series, n: int = 20, p: int = 2) -> pd.Series:
        """计算布林带"""
        return self.utils.BOLL(close, n, p)
    
    def atr(self, close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
        """计算ATR指标"""
        return self.utils.ATR(close, high, low, n)
    
    # ================== 因子管理接口 ==================
    
    def create_formula_factor(self, formula: str, name: str) -> str:
        """
        创建公式因子
        
        Args:
            formula: 因子公式，如 "RANK((CLOSE / DELAY(CLOSE, 20)) - 1)"
            name: 因子名称
            
        Returns:
            创建的因子名称
        """
        try:
            self.factor_library.register_formula_factor(formula, name)
            self.logger.info(f"Created formula factor: {name}")
            return name
        except Exception as e:
            self.logger.error(f"Error creating formula factor {name}: {str(e)}")
            raise
    
    def create_python_factor(self, code: str, name: str) -> str:
        """
        创建Python因子
        
        Args:
            code: Python因子代码
            name: 因子名称
            
        Returns:
            创建的因子名称
        """
        try:
            self.factor_library.register_python_factor(code, name)
            self.logger.info(f"Created Python factor: {name}")
            return name
        except Exception as e:
            self.logger.error(f"Error creating Python factor {name}: {str(e)}")
            raise
    
    def calculate_factor(self, factor_name: str, symbols: List[str], 
                        start_date: str, end_date: str) -> FactorSeries:
        """
        计算指定因子
        
        Args:
            factor_name: 因子名称
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            计算结果
        """
        # 获取基础数据
        base_data = self.get_data(symbols, start_date, end_date)
        
        # 计算因子
        result = self.factor_library.calculate_factor(factor_name, base_data)
        
        self.logger.info(f"Calculated factor {factor_name} for {len(symbols)} symbols")
        return result
    
    def calculate_multiple_factors(self, factor_names: List[str], symbols: List[str],
                                  start_date: str, end_date: str) -> Dict[str, FactorSeries]:
        """
        批量计算多个因子
        
        Args:
            factor_names: 因子名称列表
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            因子计算结果字典
        """
        # 获取基础数据
        base_data = self.get_data(symbols, start_date, end_date)
        
        # 批量计算因子
        results = self.factor_library.calculate_multiple_factors(factor_names, base_data)
        
        self.logger.info(f"Calculated {len(results)} factors for {len(symbols)} symbols")
        return results
    
    def list_available_factors(self) -> List[str]:
        """列出所有可用的因子"""
        return self.factor_library.list_factors()
    
    def list_available_functions(self) -> List[str]:
        """列出所有可用的算子函数"""
        return self.utils.get_available_functions()
    
    # ================== 因子分析接口 ==================
    
    def analyze_factor_ic(self, factor: FactorSeries, returns: pd.Series, 
                         periods: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        计算因子IC (Information Coefficient)
        
        Args:
            factor: 因子值
            returns: 收益率数据
            periods: 分析的持有期列表
            
        Returns:
            各期间的IC值
        """
        ic_results = {}
        
        for period in periods:
            try:
                # 计算未来收益率
                future_ret = self.future_returns(returns, period)
                
                # 计算相关系数
                ic = self.correlation(factor.series, future_ret, window=len(factor.series))
                ic_mean = ic.mean()
                
                ic_results[f'ic_{period}d'] = ic_mean
                
            except Exception as e:
                self.logger.warning(f"Error calculating IC for period {period}: {str(e)}")
                ic_results[f'ic_{period}d'] = np.nan
        
        return ic_results
    
    def factor_performance_analysis(self, factor: FactorSeries, returns: pd.Series) -> Dict[str, Any]:
        """
        综合因子性能分析
        
        Args:
            factor: 因子值
            returns: 收益率数据
            
        Returns:
            因子性能分析结果
        """
        try:
            analysis_results = {
                'factor_name': factor.name,
                'data_range': {
                    'start_date': factor.index.get_level_values('date').min(),
                    'end_date': factor.index.get_level_values('date').max(),
                    'symbol_count': len(factor.index.get_level_values('symbol').unique()),
                    'total_observations': len(factor.series)
                },
                'basic_stats': {
                    'mean': factor.series.mean(),
                    'std': factor.series.std(),
                    'skewness': factor.series.skew(),
                    'kurtosis': factor.series.kurtosis(),
                    'missing_ratio': factor.series.isna().sum() / len(factor.series)
                },
                'ic_analysis': self.analyze_factor_ic(factor, returns)
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in factor performance analysis: {str(e)}")
            raise
    
    # ================== AI增强接口 (未来实现) ==================
    
    async def ai_factor_discovery(self, market_data: Dict[str, pd.Series], 
                                 target_returns: pd.Series) -> List[str]:
        """
        AI驱动的因子发现 (占位符)
        
        Args:
            market_data: 市场数据
            target_returns: 目标收益率
            
        Returns:
            发现的因子公式列表
        """
        if self.ai_discovery is None:
            self.logger.warning("AI discovery module not available")
            return []
        
        # TODO: 实现AI因子发现逻辑
        return []
    
    async def ai_factor_optimization(self, factor_name: str, 
                                   optimization_target: str = 'ic') -> str:
        """
        AI驱动的因子优化 (占位符)
        
        Args:
            factor_name: 要优化的因子名称
            optimization_target: 优化目标 ('ic', 'sharpe', 'return')
            
        Returns:
            优化后的因子公式
        """
        if self.ai_discovery is None:
            self.logger.warning("AI discovery module not available")
            return ""
        
        # TODO: 实现AI因子优化逻辑
        return ""
    
    # ================== 工具接口 ==================
    
    def get_factor_info(self, factor_name: str) -> Dict[str, Any]:
        """获取因子详细信息"""
        factor = self.factor_library.get_factor(factor_name)
        if factor is None:
            return {"error": f"Factor '{factor_name}' not found"}
        
        info = {
            "name": factor.name,
            "type": type(factor).__name__,
            "description": getattr(factor, '__doc__', 'No description available')
        }
        
        if isinstance(factor, FormulaFactor):
            info["formula"] = factor.formula
        elif isinstance(factor, PythonFactor):
            info["code_length"] = len(factor.code)
        
        return info
    
    def validate_formula(self, formula: str) -> Dict[str, Any]:
        """验证公式语法"""
        try:
            # 创建临时因子进行验证
            temp_factor = FormulaFactor(formula, "temp_validation")
            return {"valid": True, "message": "Formula syntax is correct"}
        except Exception as e:
            return {"valid": False, "message": f"Formula error: {str(e)}"}


# 创建全局统一接口实例
unified_interface = UnifiedFactorInterface()