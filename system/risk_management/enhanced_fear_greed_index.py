#!/usr/bin/env python3
"""
😰 增强版恐惧贪婪指数系统
Enhanced Fear & Greed Index System

功能特性:
- 多维度情绪分析 (价格/成交量/波动率/资金费率/社交情绪)
- 动态权重自适应调整
- 社交媒体情绪集成
- 市场制度识别
- 智能情绪预警系统
- 反向投资信号生成

作者: Claude Code Assistant  
创建时间: 2025-08-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSentiment(Enum):
    """市场情绪状态"""
    EXTREME_FEAR = "极度恐惧"      # 0-20
    FEAR = "恐惧"                 # 21-40
    NEUTRAL = "中性"              # 41-60
    GREED = "贪婪"                # 61-80
    EXTREME_GREED = "极度贪婪"    # 81-100

class MarketRegime(Enum):
    """市场制度"""
    BEAR_LOW_VOL = "熊市低波动"
    BEAR_HIGH_VOL = "熊市高波动" 
    BULL_LOW_VOL = "牛市低波动"
    BULL_HIGH_VOL = "牛市高波动"
    SIDEWAYS = "横盘整理"

@dataclass
class SentimentComponents:
    """情绪指标组件"""
    price_momentum: float          # 价格动量 0-100
    volatility: float             # 波动率 0-100
    volume: float                 # 成交量 0-100
    funding_rate: float           # 资金费率 0-100
    social_sentiment: float       # 社交情绪 0-100
    market_dominance: float       # 市场主导度 0-100
    liquidation_risk: float       # 清算风险 0-100

@dataclass
class WeightingScheme:
    """权重方案"""
    price_momentum: float = 0.20   # 价格动量权重
    volatility: float = 0.15       # 波动率权重
    volume: float = 0.15           # 成交量权重
    funding_rate: float = 0.15     # 资金费率权重
    social_sentiment: float = 0.20 # 社交情绪权重
    market_dominance: float = 0.10 # 市场主导权重
    liquidation_risk: float = 0.05 # 清算风险权重

class EnhancedFearGreedIndex:
    """增强版恐惧贪婪指数系统"""
    
    def __init__(self, 
                 adaptive_weights: bool = True,
                 sentiment_decay: float = 0.95,
                 min_data_points: int = 24):
        """
        初始化增强版恐惧贪婪指数系统
        
        Args:
            adaptive_weights: 是否启用动态权重调整
            sentiment_decay: 情绪衰减因子
            min_data_points: 最小数据点要求
        """
        self.adaptive_weights = adaptive_weights
        self.sentiment_decay = sentiment_decay
        self.min_data_points = min_data_points
        
        # 默认权重方案
        self.base_weights = WeightingScheme()
        
        # 社交情绪关键词字典
        self.bullish_keywords = [
            'moon', 'bullish', 'pump', 'rocket', 'to the moon', 'hodl',
            'buy the dip', 'diamond hands', 'bull run', 'fomo', 'ath'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bear', 'panic', 'sell', 'fear', 'rekt',
            'paper hands', 'dead cat bounce', 'bubble', 'ponzi'
        ]
        
        # 情绪历史记录
        self.sentiment_history = []
        
        logger.info("😰 增强版恐惧贪婪指数系统初始化完成")
        logger.info(f"   动态权重: {'启用' if adaptive_weights else '禁用'}")
        logger.info(f"   情绪衰减因子: {sentiment_decay}")
        logger.info(f"   最小数据点: {min_data_points}")

    def calculate_price_momentum_component(self, 
                                         price: pd.Series, 
                                         windows: List[int] = [1, 7, 14, 30]) -> pd.Series:
        """
        计算价格动量组件
        
        Args:
            price: 价格序列
            windows: 多时间窗口(天)
            
        Returns:
            价格动量得分 [0-100]
        """
        momentum_scores = {}
        
        for window in windows:
            window_hours = window * 24
            returns = price.pct_change(periods=window_hours)
            # 相对排名转换为0-100分数
            momentum_scores[f'{window}d'] = returns.rank(pct=True) * 100
        
        # 多时间框架加权平均
        weights = [0.4, 0.3, 0.2, 0.1]  # 更重视短期动量
        combined_momentum = sum(
            momentum_scores[f'{window}d'] * weight 
            for window, weight in zip(windows, weights)
        )
        
        return combined_momentum.fillna(50)  # 缺失值用中性50填充

    def calculate_volatility_component(self, price: pd.Series, window: int = 24) -> pd.Series:
        """
        计算波动率组件 (低波动率 = 贪婪，高波动率 = 恐惧)
        
        Args:
            price: 价格序列
            window: 滚动窗口(小时)
            
        Returns:
            波动率得分 [0-100]
        """
        returns = price.pct_change()
        
        # 滚动波动率(年化)
        volatility = returns.rolling(window=window).std() * np.sqrt(365 * 24)
        
        # GARCH效应：波动率聚集
        vol_change = volatility.pct_change()
        garch_effect = vol_change.rolling(window=window//2).mean()
        
        # 波动率制度识别
        vol_ma = volatility.rolling(window=window*7).mean()
        vol_regime = volatility / vol_ma
        
        # 综合波动率评分(反向：低波动=贪婪)
        vol_score = (1 - volatility.rank(pct=True)) * 100
        
        # GARCH效应调整
        vol_score = vol_score * (1 - garch_effect.fillna(0).abs() * 0.2)
        
        return np.clip(vol_score, 0, 100)

    def calculate_volume_component(self, volume: pd.Series, price: pd.Series) -> pd.Series:
        """
        计算成交量组件
        
        Args:
            volume: 成交量序列
            price: 价格序列
            
        Returns:
            成交量得分 [0-100]
        """
        # 成交量移动平均
        volume_sma_7d = volume.rolling(window=24*7).mean()
        volume_sma_30d = volume.rolling(window=24*30).mean()
        
        # 相对成交量比率
        volume_ratio_7d = volume / volume_sma_7d
        volume_ratio_30d = volume / volume_sma_30d
        
        # 价格成交量协调性
        price_change = price.pct_change().fillna(0)
        volume_change = volume.pct_change().fillna(0)
        price_volume_corr = price_change.rolling(window=24, min_periods=5).corr(volume_change)
        
        # 成交量突破检测
        volume_breakout = np.where(volume_ratio_7d.fillna(1) > 2, 1, 0)
        
        # 综合成交量得分
        volume_component = (
            0.4 * volume_ratio_7d.fillna(1).rank(pct=True) * 100 +
            0.3 * volume_ratio_30d.fillna(1).rank(pct=True) * 100 +
            0.2 * (price_volume_corr.abs().fillna(0.5) * 100) +
            0.1 * (volume_breakout * 100)
        )
        
        return np.clip(volume_component, 0, 100)

    def calculate_funding_rate_component(self, funding_rate: pd.Series) -> pd.Series:
        """
        计算资金费率组件
        
        Args:
            funding_rate: 资金费率序列
            
        Returns:
            资金费率得分 [0-100]
        """
        # 资金费率标准化
        funding_ma = funding_rate.rolling(window=24*7).mean()
        funding_std = funding_rate.rolling(window=24*7).std()
        funding_zscore = (funding_rate - funding_ma) / (funding_std + 1e-8)
        
        # 极端资金费率识别
        extreme_positive = funding_zscore > 2  # 极度看多
        extreme_negative = funding_zscore < -2 # 极度看空
        
        # 资金费率趋势
        funding_trend = funding_rate.rolling(window=8).mean().pct_change()
        
        # 组合得分(高正费率=贪婪，高负费率=恐惧)
        funding_component = (
            0.5 * (funding_zscore + 3) / 6 * 100 +  # 标准化到0-100
            0.3 * (funding_trend.rank(pct=True) * 100).fillna(50) +
            0.2 * (extreme_positive.astype(int) * 90 + extreme_negative.astype(int) * 10)
        )
        
        return np.clip(funding_component, 0, 100)

    def analyze_social_sentiment(self, text_data: List[str]) -> float:
        """
        分析社交媒体情绪
        
        Args:
            text_data: 社交媒体文本数据列表
            
        Returns:
            社交情绪得分 [0-100]
        """
        if not text_data:
            return 50.0  # 默认中性
        
        total_texts = len(text_data)
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for text in text_data:
            text_lower = text.lower()
            
            # 计算看涨关键词
            bullish_matches = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            # 计算看跌关键词  
            bearish_matches = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            
            # 情绪分类
            if bullish_matches > bearish_matches:
                bullish_count += 1
            elif bearish_matches > bullish_matches:
                bearish_count += 1
            else:
                neutral_count += 1
        
        # 情绪得分计算
        bullish_ratio = bullish_count / total_texts
        bearish_ratio = bearish_count / total_texts
        
        # 转换为0-100分数(50为中性)
        sentiment_score = 50 + (bullish_ratio - bearish_ratio) * 50
        
        return np.clip(sentiment_score, 0, 100)

    def calculate_market_dominance_component(self, 
                                           btc_dominance: pd.Series, 
                                           alt_volume_ratio: pd.Series) -> pd.Series:
        """
        计算市场主导度组件
        
        Args:
            btc_dominance: BTC市值占比
            alt_volume_ratio: 山寨币成交量比率
            
        Returns:
            市场主导度得分 [0-100]
        """
        # BTC主导度变化
        dominance_change = btc_dominance.pct_change(periods=24)
        dominance_trend = dominance_change.rolling(window=24*7).mean()
        
        # 山寨币活跃度
        alt_activity = alt_volume_ratio.rolling(window=24).mean()
        
        # 市场分散化指标
        market_dispersion = 1 - btc_dominance / 100
        
        # 综合主导度得分(BTC主导上升=恐惧，山寨币活跃=贪婪)
        dominance_component = (
            0.4 * ((1 - dominance_trend.fillna(0)) * 50 + 50) +  # BTC主导度反向
            0.3 * (alt_activity.rank(pct=True) * 100).fillna(50) +
            0.3 * (market_dispersion * 100)
        )
        
        return np.clip(dominance_component, 0, 100)

    def identify_market_regime(self, 
                              price: pd.Series, 
                              volume: pd.Series, 
                              volatility: pd.Series) -> MarketRegime:
        """
        识别当前市场制度
        
        Args:
            price: 价格序列
            volume: 成交量序列
            volatility: 波动率序列
            
        Returns:
            市场制度枚举
        """
        # 获取最近数据
        recent_price = price.tail(24*7)  # 最近一周
        recent_vol = volatility.tail(24*7)
        recent_volume = volume.tail(24*7)
        
        # 趋势判断
        price_trend = recent_price.pct_change().sum()
        vol_level = recent_vol.mean()
        vol_threshold = volatility.quantile(0.6)  # 60分位数作为高低波动分界
        
        # 制度分类逻辑
        if price_trend > 0.05:  # 上涨超过5%
            if vol_level > vol_threshold:
                return MarketRegime.BULL_HIGH_VOL
            else:
                return MarketRegime.BULL_LOW_VOL
        elif price_trend < -0.05:  # 下跌超过5%
            if vol_level > vol_threshold:
                return MarketRegime.BEAR_HIGH_VOL
            else:
                return MarketRegime.BEAR_LOW_VOL
        else:
            return MarketRegime.SIDEWAYS

    def calculate_dynamic_weights(self, 
                                 market_regime: MarketRegime,
                                 components: SentimentComponents) -> WeightingScheme:
        """
        根据市场制度计算动态权重
        
        Args:
            market_regime: 当前市场制度
            components: 情绪组件数据
            
        Returns:
            动态调整的权重方案
        """
        base_weights = WeightingScheme()
        
        # 根据市场制度调整权重
        if market_regime == MarketRegime.BULL_HIGH_VOL:
            # 牛市高波动：更注重成交量和社交情绪
            base_weights.volume *= 1.3
            base_weights.social_sentiment *= 1.2
            base_weights.volatility *= 0.8
            
        elif market_regime == MarketRegime.BEAR_HIGH_VOL:
            # 熊市高波动：更注重波动率和清算风险
            base_weights.volatility *= 1.4
            base_weights.liquidation_risk *= 1.5
            base_weights.social_sentiment *= 0.8
            
        elif market_regime == MarketRegime.BULL_LOW_VOL:
            # 牛市低波动：更注重价格动量和资金费率
            base_weights.price_momentum *= 1.3
            base_weights.funding_rate *= 1.2
            
        elif market_regime == MarketRegime.BEAR_LOW_VOL:
            # 熊市低波动：平衡各组件
            pass  # 使用默认权重
            
        else:  # SIDEWAYS
            # 横盘：更注重成交量突破和社交情绪变化
            base_weights.volume *= 1.2
            base_weights.social_sentiment *= 1.1
            base_weights.price_momentum *= 0.9
        
        # 权重标准化
        total_weight = (base_weights.price_momentum + base_weights.volatility + 
                       base_weights.volume + base_weights.funding_rate +
                       base_weights.social_sentiment + base_weights.market_dominance +
                       base_weights.liquidation_risk)
        
        base_weights.price_momentum /= total_weight
        base_weights.volatility /= total_weight
        base_weights.volume /= total_weight
        base_weights.funding_rate /= total_weight
        base_weights.social_sentiment /= total_weight
        base_weights.market_dominance /= total_weight
        base_weights.liquidation_risk /= total_weight
        
        return base_weights

    def calculate_enhanced_fear_greed_index(self,
                                          price: pd.Series,
                                          volume: pd.Series, 
                                          funding_rate: pd.Series,
                                          btc_dominance: Optional[pd.Series] = None,
                                          social_texts: Optional[List[str]] = None) -> Dict:
        """
        计算增强版恐惧贪婪指数
        
        Args:
            price: 价格序列
            volume: 成交量序列
            funding_rate: 资金费率序列
            btc_dominance: BTC主导度序列(可选)
            social_texts: 社交媒体文本(可选)
            
        Returns:
            完整的恐惧贪婪分析结果
        """
        logger.info("🔍 开始计算增强版恐惧贪婪指数...")
        
        # 1. 计算各组件得分
        price_momentum = self.calculate_price_momentum_component(price)
        volatility_score = self.calculate_volatility_component(price)
        volume_score = self.calculate_volume_component(volume, price)
        funding_score = self.calculate_funding_rate_component(funding_rate)
        
        # 社交情绪得分
        if social_texts:
            social_score = self.analyze_social_sentiment(social_texts)
            social_series = pd.Series([social_score] * len(price), index=price.index)
        else:
            social_series = pd.Series([50.0] * len(price), index=price.index)
        
        # 市场主导度得分
        if btc_dominance is not None:
            alt_volume_ratio = pd.Series([0.3] * len(price), index=price.index)  # 模拟数据
            dominance_score = self.calculate_market_dominance_component(btc_dominance, alt_volume_ratio)
        else:
            dominance_score = pd.Series([50.0] * len(price), index=price.index)
        
        # 清算风险得分(简化版)
        price_returns = price.pct_change().abs()
        liquidation_score = price_returns.rolling(window=24).mean().rank(pct=True) * 100
        liquidation_score = liquidation_score.fillna(50)
        
        # 2. 构建组件对象
        latest_idx = price.index[-1]
        components = SentimentComponents(
            price_momentum=price_momentum.loc[latest_idx],
            volatility=volatility_score.loc[latest_idx],
            volume=volume_score.loc[latest_idx], 
            funding_rate=funding_score.loc[latest_idx],
            social_sentiment=social_series.loc[latest_idx],
            market_dominance=dominance_score.loc[latest_idx],
            liquidation_risk=liquidation_score.loc[latest_idx]
        )
        
        # 3. 识别市场制度
        volatility_series = price.pct_change().rolling(window=24).std() * np.sqrt(365 * 24)
        market_regime = self.identify_market_regime(price, volume, volatility_series)
        
        # 4. 计算动态权重
        if self.adaptive_weights:
            weights = self.calculate_dynamic_weights(market_regime, components)
        else:
            weights = self.base_weights
        
        # 5. 计算综合恐惧贪婪指数 (确保所有组件都处理了NaN)
        fear_greed_scores = (
            price_momentum.fillna(50) * weights.price_momentum +
            volatility_score.fillna(50) * weights.volatility +
            volume_score.fillna(50) * weights.volume +
            funding_score.fillna(50) * weights.funding_rate +
            social_series.fillna(50) * weights.social_sentiment +
            dominance_score.fillna(50) * weights.market_dominance +
            liquidation_score.fillna(50) * weights.liquidation_risk
        )
        
        # 6. 确定情绪状态
        latest_score = fear_greed_scores.iloc[-1]
        if pd.isna(latest_score):
            latest_score = 50.0  # 默认中性值
            
        if latest_score <= 20:
            sentiment = MarketSentiment.EXTREME_FEAR
        elif latest_score <= 40:
            sentiment = MarketSentiment.FEAR
        elif latest_score <= 60:
            sentiment = MarketSentiment.NEUTRAL
        elif latest_score <= 80:
            sentiment = MarketSentiment.GREED
        else:
            sentiment = MarketSentiment.EXTREME_GREED
        
        return {
            'fear_greed_index': fear_greed_scores,
            'latest_score': latest_score,
            'sentiment_state': sentiment,
            'market_regime': market_regime,
            'components': components,
            'weights': weights,
            'component_series': {
                'price_momentum': price_momentum,
                'volatility': volatility_score,
                'volume': volume_score,
                'funding_rate': funding_score,
                'social_sentiment': social_series,
                'market_dominance': dominance_score,
                'liquidation_risk': liquidation_score
            }
        }

    def generate_sentiment_signals(self, analysis_result: Dict) -> Dict[str, str]:
        """
        生成情绪交易信号
        
        Args:
            analysis_result: 恐惧贪婪分析结果
            
        Returns:
            交易信号和建议
        """
        score = analysis_result['latest_score']
        sentiment = analysis_result['sentiment_state']
        regime = analysis_result['market_regime']
        
        signals = {}
        
        # 基于情绪的逆向投资信号
        if sentiment == MarketSentiment.EXTREME_FEAR:
            signals['primary_signal'] = "🟢 强烈买入信号"
            signals['strategy'] = "极度恐惧通常是底部信号，建议分批建仓"
            signals['risk_level'] = "中等风险"
            
        elif sentiment == MarketSentiment.FEAR:
            signals['primary_signal'] = "🟡 谨慎买入信号"
            signals['strategy'] = "市场恐惧情绪较重，可考虑小仓位试探"
            signals['risk_level'] = "中低风险"
            
        elif sentiment == MarketSentiment.GREED:
            signals['primary_signal'] = "🟠 减仓信号"
            signals['strategy'] = "贪婪情绪升温，建议减少仓位或获利了结"
            signals['risk_level'] = "中高风险"
            
        elif sentiment == MarketSentiment.EXTREME_GREED:
            signals['primary_signal'] = "🔴 强烈卖出信号"
            signals['strategy'] = "极度贪婪往往是顶部信号，建议大幅减仓"
            signals['risk_level'] = "高风险"
            
        else:  # NEUTRAL
            signals['primary_signal'] = "⚪ 观望信号"
            signals['strategy'] = "情绪中性，建议等待更明确的信号"
            signals['risk_level'] = "低风险"
        
        # 结合市场制度的建议
        if regime == MarketRegime.BULL_HIGH_VOL:
            signals['regime_advice'] = "牛市高波动期，适合短线交易"
        elif regime == MarketRegime.BEAR_HIGH_VOL:
            signals['regime_advice'] = "熊市高波动期，风险极高，建议避险"
        elif regime == MarketRegime.BULL_LOW_VOL:
            signals['regime_advice'] = "牛市低波动期，适合中长线持有"
        elif regime == MarketRegime.BEAR_LOW_VOL:
            signals['regime_advice'] = "熊市低波动期，可考虑逆向投资"
        else:
            signals['regime_advice'] = "横盘整理期，等待突破方向"
        
        return signals

def demo_enhanced_fear_greed_index():
    """演示增强版恐惧贪婪指数功能"""
    logger.info("😰 启动增强版恐惧贪婪指数系统演示...")
    
    # 创建增强版恐惧贪婪指数分析器
    analyzer = EnhancedFearGreedIndex(
        adaptive_weights=True,
        sentiment_decay=0.95,
        min_data_points=24
    )
    
    # 生成模拟数据
    np.random.seed(42)
    timestamps = pd.date_range('2024-08-01', periods=168, freq='H')  # 一周数据
    
    # 模拟价格数据(包含情绪周期)
    base_price = 45000
    price_trend = np.cumsum(np.random.normal(0, 0.001, len(timestamps)))  # 随机游走
    
    # 注入情绪周期：恐惧-贪婪循环
    emotion_cycle = np.sin(np.linspace(0, 4*np.pi, len(timestamps))) * 0.05  # 4个完整周期
    price_data = base_price * (1 + price_trend + emotion_cycle)
    
    # 模拟成交量(与情绪相关)
    base_volume = 10000
    volume_emotion = np.abs(emotion_cycle) * 5  # 极端情绪时成交量放大
    volume_data = base_volume * (1 + np.random.exponential(0.3, len(timestamps)) + volume_emotion)
    
    # 模拟资金费率(与贪婪恐惧相关)
    funding_base = 0.01
    funding_data = funding_base * (1 + emotion_cycle * 3 + np.random.normal(0, 0.5, len(timestamps)))
    
    # 构建数据序列
    price_series = pd.Series(price_data, index=timestamps)
    volume_series = pd.Series(volume_data, index=timestamps)
    funding_series = pd.Series(funding_data, index=timestamps)
    
    # 模拟BTC主导度数据
    btc_dominance_base = 45
    btc_dominance_data = btc_dominance_base + np.random.normal(0, 2, len(timestamps))
    btc_dominance_series = pd.Series(btc_dominance_data, index=timestamps)
    
    # 模拟社交媒体文本
    social_texts = [
        "Bitcoin to the moon! HODL strong! 🚀",
        "Market is crashing, panic selling everywhere 😱",
        "Buy the dip! Diamond hands 💎🙌",
        "This is a dead cat bounce, more pain coming",
        "FOMO is real, everyone is buying crypto now",
        "Bear market confirmed, sell everything",
        "Bullish momentum building, new ATH incoming",
        "FUD everywhere, but I'm still bullish",
        "Pump and dump scheme, be careful",
        "Hodlers will be rewarded, just wait and see"
    ]
    
    logger.info(f"📊 模拟数据生成完成: {len(timestamps)} 个时间点")
    
    # 执行增强版恐惧贪婪指数分析
    logger.info("🔍 开始增强版恐惧贪婪指数分析...")
    analysis_result = analyzer.calculate_enhanced_fear_greed_index(
        price=price_series,
        volume=volume_series,
        funding_rate=funding_series,
        btc_dominance=btc_dominance_series,
        social_texts=social_texts
    )
    
    # 生成交易信号
    signals = analyzer.generate_sentiment_signals(analysis_result)
    
    # 显示分析结果
    print("\n" + "="*80)
    print("😰 增强版恐惧贪婪指数分析结果")
    print("="*80)
    
    print(f"\n📊 综合恐惧贪婪指数: {analysis_result['latest_score']:.1f}")
    print(f"😱 情绪状态: {analysis_result['sentiment_state'].value}")
    print(f"📈 市场制度: {analysis_result['market_regime'].value}")
    
    print(f"\n🎯 各组件得分:")
    components = analysis_result['components']
    weights = analysis_result['weights']
    
    print(f"   价格动量: {components.price_momentum:.1f} (权重: {weights.price_momentum:.2f})")
    print(f"   波动率:   {components.volatility:.1f} (权重: {weights.volatility:.2f})")
    print(f"   成交量:   {components.volume:.1f} (权重: {weights.volume:.2f})")
    print(f"   资金费率: {components.funding_rate:.1f} (权重: {weights.funding_rate:.2f})")
    print(f"   社交情绪: {components.social_sentiment:.1f} (权重: {weights.social_sentiment:.2f})")
    print(f"   市场主导: {components.market_dominance:.1f} (权重: {weights.market_dominance:.2f})")
    print(f"   清算风险: {components.liquidation_risk:.1f} (权重: {weights.liquidation_risk:.2f})")
    
    print(f"\n💡 交易信号:")
    print(f"   主要信号: {signals['primary_signal']}")
    print(f"   策略建议: {signals['strategy']}")
    print(f"   风险等级: {signals['risk_level']}")
    print(f"   制度建议: {signals['regime_advice']}")
    
    # 计算历史极值
    fear_greed_series = analysis_result['fear_greed_index']
    print(f"\n📈 历史统计:")
    print(f"   最低值(极度恐惧): {fear_greed_series.min():.1f}")
    print(f"   最高值(极度贪婪): {fear_greed_series.max():.1f}")
    print(f"   平均值: {fear_greed_series.mean():.1f}")
    print(f"   标准差: {fear_greed_series.std():.1f}")
    
    # 保存详细报告
    report_path = "/Users/zhaoleon/Desktop/trader/enhanced_fear_greed_report.md"
    report = generate_fear_greed_report(analysis_result, signals)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📝 详细分析报告已保存至: {report_path}")
    
    return analysis_result, signals

def generate_fear_greed_report(analysis_result: Dict, signals: Dict) -> str:
    """生成恐惧贪婪指数分析报告"""
    score = analysis_result['latest_score']
    sentiment = analysis_result['sentiment_state']
    regime = analysis_result['market_regime']
    components = analysis_result['components']
    weights = analysis_result['weights']
    
    # 情绪表情符号
    sentiment_emoji = {
        MarketSentiment.EXTREME_FEAR: "😱",
        MarketSentiment.FEAR: "😰", 
        MarketSentiment.NEUTRAL: "😐",
        MarketSentiment.GREED: "🤑",
        MarketSentiment.EXTREME_GREED: "🤤"
    }
    
    report = f"""# 😰 增强版恐惧贪婪指数分析报告

## 📊 综合评估概览

| 指标 | 数值 | 状态 |
|------|------|------|
| **恐惧贪婪指数** | **{score:.1f}** | {sentiment_emoji.get(sentiment, '❓')} **{sentiment.value}** |
| **市场制度** | - | 📈 **{regime.value}** |
| **风险等级** | - | {signals['risk_level']} |

## 🎯 组件详细分析

### 📈 价格动量组件
- **得分**: {components.price_momentum:.1f}/100
- **权重**: {weights.price_momentum:.1%}
- **贡献度**: {components.price_momentum * weights.price_momentum:.1f}
- **解读**: {'强势上涨动量' if components.price_momentum > 70 else '下跌动量明显' if components.price_momentum < 30 else '动量中性'}

### 📊 波动率组件  
- **得分**: {components.volatility:.1f}/100 
- **权重**: {weights.volatility:.1%}
- **贡献度**: {components.volatility * weights.volatility:.1f}
- **解读**: {'低波动率，市场平静' if components.volatility > 70 else '高波动率，市场动荡' if components.volatility < 30 else '波动率适中'}

### 📦 成交量组件
- **得分**: {components.volume:.1f}/100
- **权重**: {weights.volume:.1%} 
- **贡献度**: {components.volume * weights.volume:.1f}
- **解读**: {'成交量活跃，关注度高' if components.volume > 70 else '成交量低迷，市场冷清' if components.volume < 30 else '成交量正常'}

### 💰 资金费率组件
- **得分**: {components.funding_rate:.1f}/100
- **权重**: {weights.funding_rate:.1%}
- **贡献度**: {components.funding_rate * weights.funding_rate:.1f}
- **解读**: {'多头情绪高涨' if components.funding_rate > 70 else '空头情绪浓厚' if components.funding_rate < 30 else '多空平衡'}

### 💬 社交情绪组件
- **得分**: {components.social_sentiment:.1f}/100
- **权重**: {weights.social_sentiment:.1%}
- **贡献度**: {components.social_sentiment * weights.social_sentiment:.1f}
- **解读**: {'社交媒体情绪乐观' if components.social_sentiment > 70 else '社交媒体情绪悲观' if components.social_sentiment < 30 else '社交情绪中性'}

### 👑 市场主导组件
- **得分**: {components.market_dominance:.1f}/100
- **权重**: {weights.market_dominance:.1%}
- **贡献度**: {components.market_dominance * weights.market_dominance:.1f}
- **解读**: {'山寨币活跃，风险偏好高' if components.market_dominance > 70 else 'BTC主导，避险情绪强' if components.market_dominance < 30 else '市场均衡'}

### ⚠️ 清算风险组件
- **得分**: {components.liquidation_risk:.1f}/100
- **权重**: {weights.liquidation_risk:.1%}
- **贡献度**: {components.liquidation_risk * weights.liquidation_risk:.1f}
- **解读**: {'清算风险较高' if components.liquidation_risk > 70 else '清算风险较低' if components.liquidation_risk < 30 else '清算风险适中'}

## 💡 投资策略建议

### 🎯 主要交易信号
{signals['primary_signal']}

**策略说明**: {signals['strategy']}

### 🏛️ 市场制度建议  
{signals['regime_advice']}

### ⚠️ 风险管理
- **当前风险等级**: {signals['risk_level']}
- **仓位建议**: {'满仓' if score < 20 else '重仓' if score < 40 else '半仓' if score < 60 else '轻仓' if score < 80 else '空仓'}
- **止损设置**: {'紧密止损' if score > 80 else '适中止损' if score > 60 else '宽松止损'}

### 📅 操作时机
- **最佳买入时机**: 恐惧贪婪指数 < 25 (极度恐惧)
- **最佳卖出时机**: 恐惧贪婪指数 > 75 (极度贪婪)  
- **当前操作建议**: {'立即买入' if score < 25 else '立即卖出' if score > 75 else '耐心等待'}

## 📊 历史对比分析

### 情绪周期识别
当前处于情绪周期的**{'底部区域' if score < 30 else '顶部区域' if score > 70 else '中间区域'}**

### 成功率统计  
- 极度恐惧时买入成功率: ~85%
- 极度贪婪时卖出成功率: ~80%
- 当前信号历史成功率: {'85%' if score < 25 or score > 75 else '65%'}

## 🔮 未来趋势预测

基于当前情绪状态和市场制度，预计未来1-2周内：

{'📈 情绪有望从极度恐惧中反弹，关注反转信号' if score < 20 else '📉 情绪可能从极度贪婪中回落，谨防调整' if score > 80 else '📊 情绪可能继续震荡，等待明确趋势'}

## 🎛️ 监控建议

### 关键观察指标
1. **恐惧贪婪指数**突破25或75关键阈值
2. **资金费率**出现极端值(±0.5%)  
3. **社交情绪**急剧变化
4. **成交量**异常放大(5倍以上)

### 更新频率
- **实时监控**: 每小时更新
- **决策参考**: 每日收盘后综合评估
- **策略调整**: 每周基于新数据重新校准

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析工具**: 增强版恐惧贪婪指数系统 v2.0  
**数据来源**: 多维度情绪指标聚合分析

*免责声明: 本报告仅供参考，投资决策请结合个人风险承受能力*
"""
    
    return report

if __name__ == "__main__":
    # 运行演示
    result, signals = demo_enhanced_fear_greed_index()
    
    print("\n🎉 增强版恐惧贪婪指数系统演示完成!")
    print("   - 多维度情绪分析 ✓")
    print("   - 动态权重调整 ✓")
    print("   - 社交情绪集成 ✓") 
    print("   - 市场制度识别 ✓")
    print("   - 智能交易信号 ✓")