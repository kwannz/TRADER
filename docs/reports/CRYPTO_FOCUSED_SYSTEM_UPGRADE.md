"""
Crypto-Specific Factor Library
加密货币专用因子库 - 针对数字资产市场特性设计的专业算子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta

from ..traditional.factor_utils import PandaFactorUtils


class CryptoFactorUtils:
    """加密货币专用因子工具类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.traditional_utils = PandaFactorUtils()
    
    # ================== 加密货币市场特色算子 ==================
    
    @staticmethod
    def FUNDING_RATE_MOMENTUM(funding_rates: pd.Series, window: int = 24) -> pd.Series:
        """
        资金费率动量因子
        加密货币永续合约特有的资金费率数据分析
        
        Args:
            funding_rates: 资金费率序列 (8小时数据)
            window: 滚动窗口 (默认24 = 3天数据)
            
        Returns:
            资金费率动量因子
        """
        # 资金费率移动平均
        ma_funding = funding_rates.rolling(window).mean()
        
        # 当前资金费率与历史均值的偏离度
        funding_deviation = (funding_rates - ma_funding) / (ma_funding.abs() + 1e-8)
        
        # 极端资金费率标识
        extreme_funding = np.where(
            funding_rates.abs() > 0.01,  # 1%资金费率视为极端
            np.sign(funding_rates) * 2,
            funding_deviation
        )
        
        return pd.Series(extreme_funding, index=funding_rates.index, name='funding_momentum')
    
    @staticmethod
    def WHALE_ALERT(volume: pd.Series, amount: pd.Series, threshold_std: float = 3.0) -> pd.Series:
        """
        巨鲸交易预警因子
        检测异常大额交易，可能影响价格走势
        
        Args:
            volume: 成交量
            amount: 成交额
            threshold_std: 异常值标准差倍数
            
        Returns:
            巨鲸交易强度因子
        """
        # 计算平均单笔交易金额
        avg_trade_size = amount / (volume + 1e-8)
        
        # 滚动平均和标准差
        rolling_mean = avg_trade_size.rolling(window=24*7).mean()  # 7天均值
        rolling_std = avg_trade_size.rolling(window=24*7).std()    # 7天标准差
        
        # 标准化异常值检测
        z_score = (avg_trade_size - rolling_mean) / (rolling_std + 1e-8)
        
        # 巨鲸交易标识
        whale_trades = np.where(
            z_score.abs() > threshold_std,
            np.sign(z_score) * np.log(1 + z_score.abs()),  # 对数缩放
            0
        )
        
        return pd.Series(whale_trades, index=volume.index, name='whale_alert')
    
    @staticmethod
    def FEAR_GREED_INDEX(price: pd.Series, volume: pd.Series, 
                        social_sentiment: Optional[pd.Series] = None) -> pd.Series:
        """
        恐惧贪婪指数
        结合价格、成交量和社交媒体情绪的综合情绪指标
        
        Args:
            price: 价格序列
            volume: 成交量序列
            social_sentiment: 社交媒体情绪得分 (可选)
            
        Returns:
            恐惧贪婪指数 [0-100]
        """
        # 1. 价格动量组件 (25%)
        returns_14d = price.pct_change(periods=24*14)  # 14天收益率
        price_momentum = (returns_14d.rank(pct=True) * 100).fillna(50)
        
        # 2. 波动率组件 (25%)
        volatility = price.pct_change().rolling(window=24).std() * np.sqrt(24*365)
        vol_percentile = (1 - volatility.rank(pct=True)) * 100  # 低波动 = 贪婪
        
        # 3. 成交量组件 (25%)
        volume_sma = volume.rolling(window=24*7).mean()
        volume_ratio = volume / volume_sma
        volume_component = np.clip(volume_ratio.rank(pct=True) * 100, 0, 100)
        
        # 4. 社交情绪组件 (25%, 如果可用)
        if social_sentiment is not None:
            sentiment_component = social_sentiment
        else:
            # 使用价格趋势代替
            price_trend = price.rolling(window=24*7).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
            sentiment_component = (price_trend.rank(pct=True) * 100).fillna(50)
        
        # 综合指数
        fear_greed = (
            price_momentum * 0.25 + 
            vol_percentile * 0.25 + 
            volume_component * 0.25 + 
            sentiment_component * 0.25
        )
        
        # 应用指数平滑
        fear_greed = fear_greed.ewm(alpha=0.1).mean()
        
        return pd.Series(np.clip(fear_greed, 0, 100), 
                        index=price.index, name='fear_greed_index')
    
    @staticmethod
    def MARKET_CAP_RANK(price: pd.Series, circulating_supply: pd.Series) -> pd.Series:
        """
        市值排名因子
        计算实时市值排名，适用于多币种分析
        
        Args:
            price: 价格序列
            circulating_supply: 流通供应量
            
        Returns:
            市值排名因子 (标准化到[0,1])
        """
        # 计算市值
        market_cap = price * circulating_supply
        
        # 按时间横截面排名
        def rank_by_date(group):
            return group.rank(method='dense', ascending=False, pct=True)
        
        market_cap_rank = market_cap.groupby(level='date').apply(rank_by_date)
        
        return market_cap_rank
    
    @staticmethod
    def DEFI_TVL_CORRELATION(price: pd.Series, tvl_data: pd.Series, window: int = 30*24) -> pd.Series:
        """
        DeFi总锁仓价值(TVL)关联因子
        分析价格与DeFi生态系统健康度的关系
        
        Args:
            price: 代币价格
            tvl_data: 总锁仓价值数据
            window: 滚动相关性计算窗口
            
        Returns:
            TVL相关性因子
        """
        # 对数收益率
        price_returns = np.log(price / price.shift(1))
        tvl_returns = np.log(tvl_data / tvl_data.shift(1))
        
        # 滚动相关性
        rolling_corr = price_returns.rolling(window=window, min_periods=window//2).corr(tvl_returns)
        
        # 相关性强度加权
        correlation_strength = rolling_corr.abs() * np.sign(rolling_corr)
        
        return pd.Series(correlation_strength, name='defi_tvl_correlation')
    
    @staticmethod
    def EXCHANGE_FLOW_PRESSURE(net_exchange_flow: pd.Series, window: int = 24) -> pd.Series:
        """
        交易所资金流向压力
        分析资金净流入/流出对价格的潜在影响
        
        Args:
            net_exchange_flow: 交易所净流入 (正值=流入, 负值=流出)
            window: 平滑窗口
            
        Returns:
            资金流压力指标
        """
        # 累积资金流
        cumulative_flow = net_exchange_flow.cumsum()
        
        # 资金流变化率
        flow_velocity = net_exchange_flow.rolling(window=window).sum()
        
        # 压力指标：结合累积量和流速
        flow_ma = flow_velocity.rolling(window=window*3).mean()
        flow_std = flow_velocity.rolling(window=window*3).std()
        
        # 标准化压力指标
        pressure_index = (flow_velocity - flow_ma) / (flow_std + 1e-8)
        
        # 应用指数衰减
        pressure_smoothed = pressure_index.ewm(alpha=0.1).mean()
        
        return pd.Series(pressure_smoothed, name='exchange_flow_pressure')
    
    @staticmethod
    def MINER_CAPITULATION(hash_rate: pd.Series, price: pd.Series, 
                          difficulty: pd.Series, window: int = 24*14) -> pd.Series:
        """
        矿工投降指标 (适用于PoW币种如BTC)
        检测矿工可能的抛售压力
        
        Args:
            hash_rate: 网络算力
            price: 币价
            difficulty: 挖矿难度
            window: 分析窗口
            
        Returns:
            矿工投降风险指标
        """
        # 挖矿收益率 = 价格 / 难度
        mining_profitability = price / difficulty
        
        # 算力变化率
        hash_rate_change = hash_rate.pct_change(periods=window)
        
        # 收益率变化
        profitability_change = mining_profitability.pct_change(periods=window)
        
        # 矿工压力指标
        # 算力下降 + 收益率恶化 = 投降风险
        capitulation_risk = (
            -hash_rate_change * 0.6 +  # 算力下降权重60%
            -profitability_change * 0.4  # 收益率下降权重40%
        )
        
        # 标准化到[0,1]区间
        capitulation_normalized = (capitulation_risk.rank(pct=True))
        
        return pd.Series(capitulation_normalized, name='miner_capitulation')
    
    @staticmethod
    def STABLECOIN_DOMINANCE(btc_price: pd.Series, total_stable_mcap: pd.Series, 
                            total_crypto_mcap: pd.Series) -> pd.Series:
        """
        稳定币主导度因子
        衡量稳定币在加密市场中的影响力
        
        Args:
            btc_price: 比特币价格 (市场基准)
            total_stable_mcap: 稳定币总市值
            total_crypto_mcap: 加密货币总市值
            
        Returns:
            稳定币主导度指标
        """
        # 稳定币市值占比
        stable_dominance = total_stable_mcap / total_crypto_mcap
        
        # BTC价格变化
        btc_returns = btc_price.pct_change()
        
        # 稳定币需求压力：BTC下跌时稳定币占比上升
        demand_pressure = stable_dominance.rolling(window=24*7).corr(-btc_returns)
        
        # 综合主导度指标
        dominance_factor = stable_dominance * (1 + demand_pressure.fillna(0))
        
        return pd.Series(dominance_factor, name='stablecoin_dominance')
    
    @staticmethod
    def LIQUIDATION_CASCADE_RISK(price: pd.Series, open_interest: pd.Series,
                                 funding_rate: pd.Series, window: int = 24*3) -> pd.Series:
        """
        清算瀑布风险指标
        评估期货市场大规模清算的可能性
        
        Args:
            price: 现货价格
            open_interest: 期货持仓量
            funding_rate: 资金费率
            window: 分析窗口
            
        Returns:
            清算风险指标
        """
        # 持仓量变化率
        oi_growth = open_interest.pct_change(periods=window)
        
        # 资金费率极端程度
        funding_extremity = funding_rate.abs().rolling(window=window).quantile(0.8)
        current_funding_pct = funding_rate.abs().rolling(window=window).rank(pct=True)
        
        # 价格波动率
        price_volatility = price.pct_change().rolling(window=window).std()
        vol_percentile = price_volatility.rolling(window=window*4).rank(pct=True)
        
        # 清算风险综合得分
        liquidation_risk = (
            oi_growth.clip(0, np.inf) * 0.4 +  # 持仓快速增长
            current_funding_pct * 0.3 +       # 极端资金费率
            vol_percentile * 0.3               # 高波动环境
        )
        
        # 应用指数平滑
        risk_smoothed = liquidation_risk.ewm(alpha=0.15).mean()
        
        return pd.Series(risk_smoothed, name='liquidation_cascade_risk')
    
    # ================== 加密货币技术分析增强算子 ==================
    
    @staticmethod
    def CRYPTO_RSI_DIVERGENCE(price: pd.Series, rsi_period: int = 14, 
                             divergence_window: int = 24*5) -> pd.Series:
        """
        加密货币RSI背离检测
        考虑加密市场的高波动特性
        
        Args:
            price: 价格序列
            rsi_period: RSI计算周期
            divergence_window: 背离检测窗口
            
        Returns:
            背离强度指标
        """
        # 计算RSI
        rsi = PandaFactorUtils.RSI(price, rsi_period)
        
        # 价格和RSI的趋势
        price_trend = price.rolling(window=divergence_window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        rsi_trend = rsi.rolling(window=divergence_window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # 背离检测
        bullish_divergence = (price_trend < 0) & (rsi_trend > 0)  # 价格下跌但RSI上升
        bearish_divergence = (price_trend > 0) & (rsi_trend < 0)  # 价格上涨但RSI下跌
        
        # 背离强度
        divergence_strength = np.where(
            bullish_divergence, -price_trend * rsi_trend,
            np.where(bearish_divergence, price_trend * (-rsi_trend), 0)
        )
        
        return pd.Series(divergence_strength, name='rsi_divergence')
    
    @staticmethod
    def FLASH_CRASH_DETECTOR(price: pd.Series, volume: pd.Series, 
                            crash_threshold: float = -0.1, 
                            recovery_window: int = 6) -> pd.Series:
        """
        闪崩检测器
        识别加密市场的快速暴跌和恢复模式
        
        Args:
            price: 价格序列
            volume: 成交量序列
            crash_threshold: 闪崩阈值 (如-10%)
            recovery_window: 恢复检测窗口 (小时)
            
        Returns:
            闪崩恢复强度指标
        """
        # 计算收益率
        returns = price.pct_change()
        
        # 检测闪崩
        crash_events = returns < crash_threshold
        
        # 成交量激增
        volume_ma = volume.rolling(window=24).mean()
        volume_spike = volume / volume_ma > 3  # 3倍成交量激增
        
        # 闪崩确认：价格暴跌 + 成交量激增
        confirmed_crashes = crash_events & volume_spike
        
        # 恢复强度检测
        recovery_strength = []
        for i in range(len(price)):
            if confirmed_crashes.iloc[i]:
                # 检测后续恢复
                end_window = min(i + recovery_window + 1, len(price))
                future_returns = returns.iloc[i+1:end_window]
                
                if len(future_returns) > 0:
                    recovery = future_returns.sum()  # 累计恢复幅度
                    recovery_strength.append(recovery)
                else:
                    recovery_strength.append(0)
            else:
                recovery_strength.append(0)
        
        return pd.Series(recovery_strength, index=price.index, name='flash_crash_recovery')
    
    # ================== 跨链和DeFi特色算子 ==================
    
    @staticmethod
    def CROSS_CHAIN_CORRELATION(price_chain1: pd.Series, price_chain2: pd.Series,
                               window: int = 24*7) -> pd.Series:
        """
        跨链关联性分析
        分析同一资产在不同区块链上的价格关联
        
        Args:
            price_chain1: 链1上的价格
            price_chain2: 链2上的价格
            window: 滚动相关性窗口
            
        Returns:
            跨链套利机会指标
        """
        # 价格差异
        price_spread = (price_chain1 - price_chain2) / price_chain1
        
        # 滚动相关性
        returns1 = price_chain1.pct_change()
        returns2 = price_chain2.pct_change()
        rolling_corr = returns1.rolling(window=window).corr(returns2)
        
        # 套利机会：低相关性 + 显著价差
        arbitrage_opportunity = price_spread.abs() * (1 - rolling_corr.abs())
        
        return pd.Series(arbitrage_opportunity, name='cross_chain_arbitrage')
    
    @staticmethod
    def YIELD_FARMING_PRESSURE(token_price: pd.Series, apy_data: pd.Series,
                              tvl_data: pd.Series, window: int = 24*3) -> pd.Series:
        """
        收益农场压力指标
        分析DeFi收益对代币价格的影响
        
        Args:
            token_price: 代币价格
            apy_data: 年化收益率数据
            tvl_data: 锁仓价值数据
            window: 分析窗口
            
        Returns:
            收益农场影响指标
        """
        # APY变化率
        apy_change = apy_data.pct_change(periods=window)
        
        # TVL流入流出
        tvl_flow = tvl_data.pct_change()
        
        # 价格对APY的敏感性
        price_returns = token_price.pct_change()
        apy_sensitivity = price_returns.rolling(window=window).corr(apy_change)
        
        # 综合压力指标
        farming_pressure = (
            apy_change * 0.4 +
            tvl_flow * 0.3 + 
            apy_sensitivity * 0.3
        )
        
        return pd.Series(farming_pressure, name='yield_farming_pressure')


# ================== 专用数据处理工具 ==================

class CryptoDataProcessor:
    """加密货币数据处理专用工具"""
    
    @staticmethod
    def detect_market_regime(price: pd.Series, volume: pd.Series) -> pd.Series:
        """
        市场制度识别
        识别牛市、熊市、横盘等不同市场状态
        """
        returns = price.pct_change()
        volatility = returns.rolling(24*7).std()
        
        # 趋势强度
        trend = price.rolling(24*7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # 制度分类
        regimes = pd.Series(index=price.index, dtype=str)
        regimes.loc[(trend > 0) & (volatility < volatility.quantile(0.5))] = 'bull_quiet'
        regimes.loc[(trend > 0) & (volatility >= volatility.quantile(0.5))] = 'bull_volatile'  
        regimes.loc[(trend < 0) & (volatility < volatility.quantile(0.5))] = 'bear_quiet'
        regimes.loc[(trend < 0) & (volatility >= volatility.quantile(0.5))] = 'bear_volatile'
        regimes.loc[trend.abs() < trend.std()] = 'sideways'
        
        return regimes
    
    @staticmethod
    def clean_crypto_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        加密货币数据清洗
        处理异常值、间断数据等问题
        """
        # 处理极端价格跳跃 (>50%变化视为异常)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                returns = data[col].pct_change()
                outliers = returns.abs() > 0.5
                data.loc[outliers, col] = np.nan
                data[col] = data[col].interpolate(method='linear')
        
        # 处理零成交量
        if 'volume' in data.columns:
            data.loc[data['volume'] == 0, 'volume'] = data['volume'].rolling(24).median()
        
        # 确保OHLC逻辑关系
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        return data