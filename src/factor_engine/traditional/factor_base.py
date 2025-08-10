"""
Factor Base Classes
因子基础类 - 基于PandaFactor原始架构适配
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import logging
from ..traditional.factor_utils import PandaFactorUtils


class FactorSeries:
    """封装因子序列，提供更好的操作接口"""
    
    def __init__(self, series: pd.Series, name: str = "factor"):
        """
        初始化因子序列
        
        Args:
            series: pandas Series with MultiIndex (date, symbol)
            name: 因子名称
        """
        self.series = series
        self.name = name
        
        # 验证索引格式
        if not isinstance(series.index, pd.MultiIndex):
            raise ValueError("FactorSeries requires MultiIndex with (date, symbol)")
        
        if series.index.names != ['date', 'symbol']:
            series.index.names = ['date', 'symbol']
            
    def __repr__(self):
        return f"FactorSeries(name='{self.name}', shape={self.series.shape})"
    
    def __getitem__(self, key):
        return self.series[key]
    
    def __setitem__(self, key, value):
        self.series[key] = value
        
    @property
    def index(self):
        return self.series.index
    
    @property
    def values(self):
        return self.series.values
        
    def dropna(self):
        """删除缺失值"""
        return FactorSeries(self.series.dropna(), self.name)
    
    def fillna(self, value=0):
        """填充缺失值"""
        return FactorSeries(self.series.fillna(value), self.name)
    
    def rank(self):
        """计算横截面排名"""
        return FactorSeries(PandaFactorUtils.RANK(self.series), f"{self.name}_rank")
    
    def scale(self):
        """标准化到[-1, 1]区间"""
        return FactorSeries(PandaFactorUtils.SCALE(self.series), f"{self.name}_scaled")
    
    def returns(self, period: int = 1):
        """计算收益率"""
        return FactorSeries(
            PandaFactorUtils.RETURNS(self.series, period), 
            f"{self.name}_ret{period}"
        )
    
    def correlation(self, other: 'FactorSeries', window: int = 20):
        """计算与其他因子的相关性"""
        return FactorSeries(
            PandaFactorUtils.CORRELATION(self.series, other.series, window),
            f"{self.name}_corr_{other.name}"
        )


class BaseFactor(ABC):
    """
    因子基础类 - 所有因子的抽象基类
    基于PandaFactor设计模式
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"Factor.{self.name}")
        self.utils = PandaFactorUtils()
        
    @abstractmethod
    def calculate(self, factors: Dict[str, pd.Series]) -> pd.Series:
        """
        计算因子值
        
        Args:
            factors: 基础因子数据字典，键为因子名，值为MultiIndex Series
            
        Returns:
            计算后的因子值Series
        """
        pass
    
    def validate_input(self, factors: Dict[str, pd.Series]) -> bool:
        """验证输入数据格式"""
        for name, series in factors.items():
            if not isinstance(series, pd.Series):
                self.logger.error(f"Factor {name} is not a pandas Series")
                return False
                
            if not isinstance(series.index, pd.MultiIndex):
                self.logger.error(f"Factor {name} does not have MultiIndex")
                return False
                
            if series.index.names != ['date', 'symbol']:
                self.logger.warning(f"Factor {name} index names are not ['date', 'symbol'], will be corrected")
                series.index.names = ['date', 'symbol']
        
        return True
    
    def process(self, factors: Dict[str, pd.Series]) -> FactorSeries:
        """
        处理因子计算的完整流程
        
        Args:
            factors: 基础因子数据
            
        Returns:
            FactorSeries对象
        """
        try:
            # 验证输入
            if not self.validate_input(factors):
                raise ValueError("Input validation failed")
            
            # 执行计算
            result = self.calculate(factors)
            
            # 包装为FactorSeries
            if isinstance(result, pd.Series):
                return FactorSeries(result, self.name)
            else:
                raise ValueError("calculate method must return pandas Series")
                
        except Exception as e:
            self.logger.error(f"Error processing factor {self.name}: {str(e)}")
            raise


class FormulaFactor(BaseFactor):
    """
    公式因子 - 支持类似WorldQuant Alpha表达式
    """
    
    def __init__(self, formula: str, name: str = None):
        super().__init__(name)
        self.formula = formula
        self.name = name or f"Formula_{hash(formula) % 10000}"
        
        # 构建安全的执行环境
        self._build_safe_namespace()
    
    def _build_safe_namespace(self):
        """构建安全的公式执行环境"""
        self.safe_namespace = {
            # 基础算子
            'RANK': PandaFactorUtils.RANK,
            'RETURNS': PandaFactorUtils.RETURNS,
            'FUTURE_RETURNS': PandaFactorUtils.FUTURE_RETURNS,
            'STDDEV': PandaFactorUtils.STDDEV,
            'CORRELATION': PandaFactorUtils.CORRELATION,
            'IF': PandaFactorUtils.IF,
            'DELAY': PandaFactorUtils.DELAY,
            'SUM': PandaFactorUtils.SUM,
            
            # 时序算子
            'TS_ARGMAX': PandaFactorUtils.TS_ARGMAX,
            'TS_RANK': PandaFactorUtils.TS_RANK,
            'DELTA': PandaFactorUtils.DELTA,
            'ADV': PandaFactorUtils.ADV,
            'TS_MIN': PandaFactorUtils.TS_MIN,
            'TS_MAX': PandaFactorUtils.TS_MAX,
            'DECAY_LINEAR': PandaFactorUtils.DECAY_LINEAR,
            'SCALE': PandaFactorUtils.SCALE,
            
            # 数学算子
            'MIN': PandaFactorUtils.MIN,
            'MAX': PandaFactorUtils.MAX,
            'ABS': PandaFactorUtils.ABS,
            'LOG': PandaFactorUtils.LOG,
            'POWER': PandaFactorUtils.POWER,
            'SIGN': PandaFactorUtils.SIGN,
            'SIGNEDPOWER': PandaFactorUtils.SIGNEDPOWER,
            
            # PandaFactor特色算子
            'RD': PandaFactorUtils.RD,
            'REF': PandaFactorUtils.REF,
            'DIFF': PandaFactorUtils.DIFF,
            'STD': PandaFactorUtils.STD,
            'MA': PandaFactorUtils.MA,
            'EMA': PandaFactorUtils.EMA,
            'SMA': PandaFactorUtils.SMA,
            'WMA': PandaFactorUtils.WMA,
            'HHV': PandaFactorUtils.HHV,
            'LLV': PandaFactorUtils.LLV,
            'SLOPE': PandaFactorUtils.SLOPE,
            
            # 技术指标
            'MACD': PandaFactorUtils.MACD,
            'RSI': PandaFactorUtils.RSI,
            'KDJ': PandaFactorUtils.KDJ,
            'BOLL': PandaFactorUtils.BOLL,
            'CCI': PandaFactorUtils.CCI,
            'ATR': PandaFactorUtils.ATR,
            'ROC': PandaFactorUtils.ROC,
            'OBV': PandaFactorUtils.OBV,
            'MFI': PandaFactorUtils.MFI,
            
            # 条件和逻辑算子
            'CROSS': PandaFactorUtils.CROSS,
            'COUNT': PandaFactorUtils.COUNT,
            'EVERY': PandaFactorUtils.EVERY,
            'EXIST': PandaFactorUtils.EXIST,
            'BARSLAST': PandaFactorUtils.BARSLAST,
            'VALUEWHEN': PandaFactorUtils.VALUEWHEN,
            
            # 高级函数
            'VWAP': PandaFactorUtils.VWAP,
            'CAP': PandaFactorUtils.CAP,
            'COVARIANCE': PandaFactorUtils.COVARIANCE,
            'AS_FLOAT': PandaFactorUtils.AS_FLOAT,
            'PRODUCT': PandaFactorUtils.PRODUCT,
            'TS_MEAN': PandaFactorUtils.TS_MEAN,
            
            # 常用库
            'np': np,
            'pd': pd,
        }
    
    def calculate(self, factors: Dict[str, pd.Series]) -> pd.Series:
        """
        执行公式计算
        
        Args:
            factors: 基础因子数据，支持的键包括：
                    'close', 'open', 'high', 'low', 'volume', 'amount', 
                    'market_cap', 'turnover' 等
        
        Returns:
            计算结果Series
        """
        try:
            # 将因子数据添加到执行环境
            namespace = self.safe_namespace.copy()
            
            # 标准化因子名称映射
            factor_mapping = {
                'CLOSE': factors.get('close', factors.get('CLOSE')),
                'OPEN': factors.get('open', factors.get('OPEN')),
                'HIGH': factors.get('high', factors.get('HIGH')),
                'LOW': factors.get('low', factors.get('LOW')),
                'VOLUME': factors.get('volume', factors.get('VOLUME')),
                'VOL': factors.get('volume', factors.get('VOL')),  # 别名
                'AMOUNT': factors.get('amount', factors.get('AMOUNT')),
                'MARKET_CAP': factors.get('market_cap', factors.get('MARKET_CAP')),
                'TURNOVER': factors.get('turnover', factors.get('TURNOVER')),
            }
            
            # 添加到命名空间
            for key, value in factor_mapping.items():
                if value is not None:
                    namespace[key] = value
            
            # 执行公式
            result = eval(self.formula, {"__builtins__": {}}, namespace)
            
            if not isinstance(result, pd.Series):
                raise ValueError(f"Formula result must be pandas Series, got {type(result)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing formula '{self.formula}': {str(e)}")
            raise


class PythonFactor(BaseFactor):
    """
    Python代码因子 - 支持自定义Python类
    """
    
    def __init__(self, code: str, name: str = None):
        super().__init__(name)
        self.code = code
        self.name = name or f"Python_{hash(code) % 10000}"
        self.factor_class = None
        
        # 编译并实例化因子类
        self._compile_factor_class()
    
    def _compile_factor_class(self):
        """编译Python代码并创建因子类实例"""
        try:
            # 构建安全的执行环境
            safe_builtins = {
                '__build_class__': __builtins__['__build_class__'],
                '__name__': __name__,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
            }
            
            namespace = {
                'BaseFactor': BaseFactor,
                'Factor': BaseFactor,  # 别名
                'pd': pd,
                'np': np,
                'PandaFactorUtils': PandaFactorUtils,
                # 添加所有工具函数作为全局函数
                **{name: getattr(PandaFactorUtils, name) 
                   for name in dir(PandaFactorUtils) 
                   if not name.startswith('_') and callable(getattr(PandaFactorUtils, name))}
            }
            
            # 执行代码
            exec(self.code, {"__builtins__": safe_builtins}, namespace)
            
            # 查找Factor类
            factor_classes = [obj for name, obj in namespace.items() 
                             if isinstance(obj, type) and issubclass(obj, BaseFactor) and obj != BaseFactor]
            
            if not factor_classes:
                raise ValueError("No Factor class found in code")
            
            if len(factor_classes) > 1:
                self.logger.warning(f"Multiple Factor classes found, using the first one")
            
            # 实例化因子类
            self.factor_class = factor_classes[0]()
            
        except Exception as e:
            self.logger.error(f"Error compiling Python factor: {str(e)}")
            raise
    
    def calculate(self, factors: Dict[str, pd.Series]) -> pd.Series:
        """
        执行Python因子计算
        """
        if self.factor_class is None:
            raise ValueError("Factor class not compiled")
        
        return self.factor_class.calculate(factors)


class FactorLibrary:
    """
    因子库 - 管理所有因子的注册和调用
    """
    
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}
        self.logger = logging.getLogger("FactorLibrary")
    
    def register_factor(self, factor: BaseFactor, name: str = None):
        """注册因子"""
        name = name or factor.name
        self.factors[name] = factor
        self.logger.info(f"Registered factor: {name}")
    
    def register_formula_factor(self, formula: str, name: str):
        """注册公式因子"""
        factor = FormulaFactor(formula, name)
        self.register_factor(factor, name)
    
    def register_python_factor(self, code: str, name: str):
        """注册Python因子"""
        factor = PythonFactor(code, name)
        self.register_factor(factor, name)
    
    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """获取因子"""
        return self.factors.get(name)
    
    def list_factors(self) -> List[str]:
        """列出所有已注册的因子"""
        return list(self.factors.keys())
    
    def calculate_factor(self, name: str, factors: Dict[str, pd.Series]) -> FactorSeries:
        """计算指定因子"""
        factor = self.get_factor(name)
        if factor is None:
            raise ValueError(f"Factor '{name}' not found")
        
        return factor.process(factors)
    
    def calculate_multiple_factors(self, factor_names: List[str], 
                                 factors: Dict[str, pd.Series]) -> Dict[str, FactorSeries]:
        """批量计算多个因子"""
        results = {}
        for name in factor_names:
            try:
                results[name] = self.calculate_factor(name, factors)
            except Exception as e:
                self.logger.error(f"Error calculating factor {name}: {str(e)}")
                
        return results


# 创建全局因子库实例
factor_library = FactorLibrary()