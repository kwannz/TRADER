#!/usr/bin/env python3
"""
Phase 1 Integration Demo
PandaFactor传统算子集成演示
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.factor_engine import (
    UnifiedFactorInterface, unified_interface,
    PandaFactorUtils, panda_factor_utils,
    FormulaFactor, PythonFactor
)


def demo_basic_operators():
    """演示基础算子功能"""
    print("=" * 60)
    print("Phase 1 演示: PandaFactor 基础算子集成")
    print("=" * 60)
    
    # 创建示例数据
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # 生成模拟价格数据
    np.random.seed(42)
    prices = 100 + np.random.randn(len(index)) * 5 + np.cumsum(np.random.randn(len(index)) * 0.1)
    close_series = pd.Series(prices, index=index)
    
    print(f"数据维度: {len(dates)} 日期 × {len(symbols)} 股票 = {len(close_series)} 观测值")
    print("\n样本数据:")
    print(close_series.head(10))
    
    # 测试基础算子
    print("\n" + "="*40)
    print("基础算子测试")
    print("="*40)
    
    # 1. 横截面排名
    rank_result = panda_factor_utils.RANK(close_series)
    print(f"\n1. RANK算子:")
    print(f"   原始数据范围: [{close_series.min():.2f}, {close_series.max():.2f}]")
    print(f"   排名结果范围: [{rank_result.min():.2f}, {rank_result.max():.2f}]")
    print("   前5个排名结果:")
    print(rank_result.head().round(3))
    
    # 2. 收益率计算
    returns_1d = panda_factor_utils.RETURNS(close_series, 1)
    returns_5d = panda_factor_utils.RETURNS(close_series, 5)
    print(f"\n2. RETURNS算子:")
    print(f"   1日收益率均值: {returns_1d.mean():.4f}, 标准差: {returns_1d.std():.4f}")
    print(f"   5日收益率均值: {returns_5d.mean():.4f}, 标准差: {returns_5d.std():.4f}")
    
    # 3. 滚动标准差
    volatility = panda_factor_utils.STDDEV(returns_1d, 5)
    print(f"\n3. STDDEV算子:")
    print(f"   5日滚动标准差均值: {volatility.mean():.4f}")
    print("   前5个波动率结果:")
    print(volatility.dropna().head().round(4))
    
    # 4. 时序排名
    ts_rank_result = panda_factor_utils.TS_RANK(close_series, 5)
    print(f"\n4. TS_RANK算子:")
    print(f"   时序排名范围: [{ts_rank_result.min():.2f}, {ts_rank_result.max():.2f}]")
    print("   前5个时序排名结果:")
    print(ts_rank_result.dropna().head().round(3))


def demo_technical_indicators():
    """演示技术指标功能"""
    print("\n" + "="*40)
    print("技术指标测试")
    print("="*40)
    
    # 创建更长的时间序列用于技术指标
    dates = pd.date_range('2024-01-01', '2024-03-01', freq='D')
    symbols = ['AAPL']
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # 生成OHLCV数据
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.02, len(index))
    close = pd.Series(base_price * np.exp(np.cumsum(returns)), index=index)
    
    high = close * (1 + np.random.uniform(0, 0.03, len(index)))
    low = close * (1 - np.random.uniform(0, 0.03, len(index)))
    volume = pd.Series(np.random.lognormal(13, 0.5, len(index)), index=index)
    
    # 1. MACD指标
    dif, dea, macd = panda_factor_utils.MACD(close, 12, 26, 9)
    print(f"\n1. MACD指标:")
    print(f"   DIF范围: [{dif.min():.4f}, {dif.max():.4f}]")
    print(f"   DEA范围: [{dea.min():.4f}, {dea.max():.4f}]")
    print(f"   MACD范围: [{macd.min():.4f}, {macd.max():.4f}]")
    
    # 2. RSI指标
    rsi = panda_factor_utils.RSI(close, 14)
    print(f"\n2. RSI指标:")
    print(f"   RSI范围: [{rsi.min():.2f}, {rsi.max():.2f}]")
    print(f"   RSI均值: {rsi.mean():.2f}")
    
    # 3. KDJ指标
    kdj_k = panda_factor_utils.KDJ(close, high, low, 9, 3, 3)
    print(f"\n3. KDJ指标:")
    print(f"   K线范围: [{kdj_k.min():.2f}, {kdj_k.max():.2f}]")
    
    # 4. 布林带
    boll_mid = panda_factor_utils.BOLL(close, 20, 2)
    print(f"\n4. 布林带:")
    print(f"   中轨均值: {boll_mid.mean():.2f}")


def demo_formula_factor():
    """演示公式因子功能"""
    print("\n" + "="*40)
    print("公式因子测试")
    print("="*40)
    
    # 获取测试数据
    interface = unified_interface
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-01-20'
    
    # 创建经典动量因子公式
    momentum_formula = "RANK((CLOSE / DELAY(CLOSE, 10)) - 1)"
    
    try:
        # 注册公式因子
        factor_name = interface.create_formula_factor(momentum_formula, "momentum_10d")
        print(f"\n创建公式因子: {factor_name}")
        print(f"公式: {momentum_formula}")
        
        # 计算因子值
        result = interface.calculate_factor(factor_name, symbols, start_date, end_date)
        
        print(f"\n因子计算结果:")
        print(f"   因子名称: {result.name}")
        print(f"   数据维度: {result.series.shape}")
        print(f"   值域范围: [{result.series.min():.4f}, {result.series.max():.4f}]")
        print(f"   均值: {result.series.mean():.4f}")
        print(f"   缺失值比例: {result.series.isna().sum() / len(result.series) * 100:.2f}%")
        
        # 显示部分结果
        print("\n前10个因子值:")
        print(result.series.dropna().head(10).round(4))
        
    except Exception as e:
        print(f"公式因子演示出错: {str(e)}")


def demo_python_factor():
    """演示Python因子功能"""
    print("\n" + "="*40)
    print("Python因子测试")
    print("="*40)
    
    # 定义一个复合因子的Python代码
    python_code = '''
class ComplexMomentumFactor(BaseFactor):
    """复合动量因子 - 结合价格动量和成交量信号"""
    
    def calculate(self, factors):
        close = factors['close']
        volume = factors['volume']
        
        # 计算价格动量
        price_momentum = (close / DELAY(close, 20)) - 1
        
        # 计算成交量相对变化
        volume_ratio = volume / DELAY(volume, 5)
        
        # 计算波动率调整
        volatility = STDDEV(RETURNS(close, 1), 10)
        vol_adjustment = 1 / (volatility + 0.01)  # 避免除零
        
        # 组合信号
        momentum_signal = RANK(price_momentum)
        volume_signal = RANK(volume_ratio)
        
        # 最终因子值
        result = momentum_signal * 0.6 + volume_signal * 0.4
        result = result * vol_adjustment
        
        return SCALE(result)  # 标准化到[-1, 1]
'''
    
    try:
        interface = unified_interface
        symbols = ['AAPL', 'GOOGL']
        start_date = '2024-01-01' 
        end_date = '2024-01-15'
        
        # 注册Python因子
        factor_name = interface.create_python_factor(python_code, "complex_momentum")
        print(f"\n创建Python因子: {factor_name}")
        
        # 计算因子值
        result = interface.calculate_factor(factor_name, symbols, start_date, end_date)
        
        print(f"\n因子计算结果:")
        print(f"   因子名称: {result.name}")
        print(f"   数据维度: {result.series.shape}")
        print(f"   值域范围: [{result.series.min():.4f}, {result.series.max():.4f}]")
        print(f"   均值: {result.series.mean():.4f}")
        
        # 显示部分结果
        print("\n前8个因子值:")
        valid_results = result.series.dropna()
        if len(valid_results) > 0:
            print(valid_results.head(8).round(4))
        else:
            print("所有计算结果都是NaN，可能需要更长的数据序列")
            
    except Exception as e:
        print(f"Python因子演示出错: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_system_info():
    """演示系统信息查询"""
    print("\n" + "="*40)
    print("系统功能信息")
    print("="*40)
    
    interface = unified_interface
    
    # 显示可用函数
    functions = interface.list_available_functions()
    print(f"\n可用算子函数数量: {len(functions)}")
    print("主要算子函数:")
    core_functions = [f for f in functions if f in [
        'RANK', 'RETURNS', 'STDDEV', 'CORRELATION', 'TS_RANK', 'MACD', 'RSI', 'SCALE'
    ]]
    for func in core_functions:
        print(f"   - {func}")
    
    # 显示已注册的因子
    registered_factors = interface.list_available_factors()
    print(f"\n已注册因子数量: {len(registered_factors)}")
    if registered_factors:
        print("已注册的因子:")
        for factor in registered_factors:
            info = interface.get_factor_info(factor)
            print(f"   - {factor}: {info.get('type', 'Unknown')}")
    
    # 测试公式验证
    print(f"\n公式验证测试:")
    valid_formula = "RANK(CLOSE / DELAY(CLOSE, 5))"
    invalid_formula = "RANK(UNKNOWN_FUNC(CLOSE))"
    
    result1 = interface.validate_formula(valid_formula)
    result2 = interface.validate_formula(invalid_formula)
    
    print(f"   有效公式 '{valid_formula}': {result1['message']}")
    print(f"   无效公式 '{invalid_formula[:20]}...': {result2['message'][:50]}...")


def main():
    """主演示函数"""
    print("PandaFactor Phase 1 集成演示")
    print("集成70+专业算子，支持公式和Python因子")
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 基础算子演示
        demo_basic_operators()
        
        # 技术指标演示
        demo_technical_indicators()
        
        # 公式因子演示
        demo_formula_factor()
        
        # Python因子演示
        demo_python_factor()
        
        # 系统信息演示
        demo_system_info()
        
        print("\n" + "="*60)
        print("Phase 1 演示完成！")
        print("✅ 传统算子集成: 70+ PandaFactor专业算子")
        print("✅ 公式因子支持: WorldQuant Alpha风格表达式")
        print("✅ Python因子支持: 自定义因子类开发")
        print("✅ 统一接口: 整合传统和AI增强功能")
        print("\n接下来: Phase 2 - LLM服务统一与数据管理扩展")
        print("="*60)
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()