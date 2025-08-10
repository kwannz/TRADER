"""
Coinglass数据AI分析器
为量化交易系统提供基于市场情绪、资金流向、持仓数据的智能分析
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .data_manager import data_manager

logger = get_logger()

@dataclass
class MarketSentimentSignal:
    """市场情绪信号"""
    sentiment_score: float  # -100 到 100，负数表示恐惧，正数表示贪婪
    confidence: float      # 0 到 1，信号置信度
    trend: str            # 'bullish', 'bearish', 'neutral'
    strength: str         # 'weak', 'moderate', 'strong'
    components: Dict[str, float]  # 各组件得分
    timestamp: datetime
    
@dataclass
class FundingRateSignal:
    """资金费率信号"""
    overall_rate: float    # 整体资金费率
    rate_trend: str       # 'increasing', 'decreasing', 'stable'
    market_heat: str      # 'cold', 'normal', 'hot', 'overheated'
    divergence_score: float  # 交易所间分歧程度
    exchanges_data: List[Dict]  # 各交易所数据
    timestamp: datetime

@dataclass
class OpenInterestSignal:
    """持仓数据信号"""
    total_oi: float       # 总持仓量
    oi_trend: str        # 'increasing', 'decreasing', 'stable'
    oi_change_24h: float # 24小时变化百分比
    market_structure: str # 'accumulation', 'distribution', 'neutral'
    risk_level: str      # 'low', 'moderate', 'high', 'extreme'
    exchanges_data: List[Dict]
    timestamp: datetime

@dataclass
class ETFFlowSignal:
    """ETF资金流向信号"""
    btc_flow: float      # BTC ETF净流入
    eth_flow: float      # ETH ETF净流入
    flow_trend: str      # 'inflow', 'outflow', 'neutral'
    institutional_sentiment: str  # 'bullish', 'bearish', 'neutral'
    flow_strength: str   # 'weak', 'moderate', 'strong'
    timestamp: datetime

@dataclass
class CoinglassCompositeSignal:
    """Coinglass综合信号"""
    overall_score: float  # 综合得分 -100 到 100
    signal_strength: str  # 'weak', 'moderate', 'strong'
    market_regime: str    # 'bull', 'bear', 'sideways'
    risk_assessment: str  # 'low', 'moderate', 'high', 'extreme'
    key_factors: List[str]  # 关键影响因素
    
    sentiment_signal: MarketSentimentSignal
    funding_signal: FundingRateSignal
    oi_signal: OpenInterestSignal
    etf_signal: ETFFlowSignal
    
    timestamp: datetime
    confidence: float

class CoinglassAnalyzer:
    """Coinglass数据分析器"""
    
    def __init__(self):
        self.is_initialized = False
        self.last_analysis_time = None
        self.analysis_cache = {}
        self.cache_duration = 300  # 5分钟缓存
        
    async def initialize(self):
        """初始化分析器"""
        try:
            # 确保数据库连接
            if not data_manager._initialized:
                await data_manager.initialize()
            
            self.is_initialized = True
            logger.info("Coinglass分析器初始化完成")
            
        except Exception as e:
            logger.error(f"Coinglass分析器初始化失败: {e}")
            raise
    
    async def analyze_market_sentiment(self, lookback_days: int = 7) -> MarketSentimentSignal:
        """分析市场情绪"""
        try:
            # 获取恐惧贪婪指数数据
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=lookback_days)
            
            collection = data_manager.db["fear_greed_index"]
            cursor = collection.find({
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }).sort("timestamp", -1).limit(100)
            
            fear_greed_data = await cursor.to_list(length=None)
            
            if not fear_greed_data:
                logger.warning("没有找到恐惧贪婪指数数据")
                return self._create_default_sentiment_signal()
            
            # 计算情绪指标
            recent_values = [item["value"] for item in fear_greed_data[:10]]  # 最近10个数据点
            current_value = recent_values[0] if recent_values else 50
            
            # 情绪得分转换 (0-100 -> -50 to +50)
            sentiment_score = (current_value - 50)
            
            # 计算趋势
            trend = self._calculate_sentiment_trend(recent_values)
            
            # 计算强度
            strength = self._calculate_sentiment_strength(current_value, recent_values)
            
            # 计算置信度
            confidence = self._calculate_sentiment_confidence(fear_greed_data, lookback_days)
            
            return MarketSentimentSignal(
                sentiment_score=sentiment_score,
                confidence=confidence,
                trend=trend,
                strength=strength,
                components={
                    "fear_greed_index": current_value,
                    "trend_momentum": self._calculate_momentum(recent_values),
                    "volatility": self._calculate_volatility(recent_values)
                },
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"分析市场情绪失败: {e}")
            return self._create_default_sentiment_signal()
    
    async def analyze_funding_rates(self, symbols: List[str] = None) -> FundingRateSignal:
        """分析资金费率"""
        try:
            symbols = symbols or ["BTC", "ETH", "BNB", "ADA", "SOL"]
            
            # 获取最近的资金费率数据
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            collection = data_manager.db["funding_rates"]
            cursor = collection.find({
                "symbol": {"$in": symbols},
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }).sort("timestamp", -1)
            
            funding_data = await cursor.to_list(length=None)
            
            if not funding_data:
                logger.warning("没有找到资金费率数据")
                return self._create_default_funding_signal()
            
            # 按交易所和符号聚合数据
            exchanges_data = self._aggregate_funding_data(funding_data)
            
            # 计算整体资金费率
            overall_rate = self._calculate_weighted_funding_rate(exchanges_data)
            
            # 分析趋势
            rate_trend = self._analyze_funding_trend(funding_data)
            
            # 评估市场热度
            market_heat = self._assess_market_heat(overall_rate, exchanges_data)
            
            # 计算分歧程度
            divergence_score = self._calculate_funding_divergence(exchanges_data)
            
            return FundingRateSignal(
                overall_rate=overall_rate,
                rate_trend=rate_trend,
                market_heat=market_heat,
                divergence_score=divergence_score,
                exchanges_data=exchanges_data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"分析资金费率失败: {e}")
            return self._create_default_funding_signal()
    
    async def analyze_open_interest(self, symbols: List[str] = None) -> OpenInterestSignal:
        """分析持仓数据"""
        try:
            symbols = symbols or ["BTC", "ETH", "BNB", "ADA", "SOL"]
            
            # 获取持仓数据
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=3)
            
            collection = data_manager.db["open_interest"]
            cursor = collection.find({
                "symbol": {"$in": symbols},
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }).sort("timestamp", -1)
            
            oi_data = await cursor.to_list(length=None)
            
            if not oi_data:
                logger.warning("没有找到持仓数据")
                return self._create_default_oi_signal()
            
            # 聚合和分析数据
            exchanges_data = self._aggregate_oi_data(oi_data)
            total_oi = sum(item["open_interest"] for item in exchanges_data)
            
            # 计算变化
            oi_change_24h = self._calculate_oi_change(oi_data)
            
            # 分析趋势
            oi_trend = self._analyze_oi_trend(oi_change_24h)
            
            # 评估市场结构
            market_structure = self._assess_market_structure(oi_change_24h, exchanges_data)
            
            # 评估风险水平
            risk_level = self._assess_oi_risk_level(total_oi, oi_change_24h)
            
            return OpenInterestSignal(
                total_oi=total_oi,
                oi_trend=oi_trend,
                oi_change_24h=oi_change_24h,
                market_structure=market_structure,
                risk_level=risk_level,
                exchanges_data=exchanges_data,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"分析持仓数据失败: {e}")
            return self._create_default_oi_signal()
    
    async def analyze_etf_flows(self) -> ETFFlowSignal:
        """分析ETF资金流向"""
        try:
            # 获取ETF流向数据
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            collection = data_manager.db["etf_flows"]
            cursor = collection.find({
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }).sort("timestamp", -1)
            
            etf_data = await cursor.to_list(length=None)
            
            if not etf_data:
                logger.warning("没有找到ETF数据")
                return self._create_default_etf_signal()
            
            # 分析BTC和ETH流向
            btc_flow = self._calculate_asset_flow(etf_data, "BTC")
            eth_flow = self._calculate_asset_flow(etf_data, "ETH")
            
            # 判断整体流向趋势
            total_flow = btc_flow + eth_flow
            if total_flow > 100_000_000:  # 1亿美元
                flow_trend = "inflow"
            elif total_flow < -100_000_000:
                flow_trend = "outflow"
            else:
                flow_trend = "neutral"
            
            # 评估机构情绪
            institutional_sentiment = self._assess_institutional_sentiment(btc_flow, eth_flow)
            
            # 评估流向强度
            flow_strength = self._assess_flow_strength(abs(total_flow))
            
            return ETFFlowSignal(
                btc_flow=btc_flow,
                eth_flow=eth_flow,
                flow_trend=flow_trend,
                institutional_sentiment=institutional_sentiment,
                flow_strength=flow_strength,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"分析ETF流向失败: {e}")
            return self._create_default_etf_signal()
    
    async def generate_composite_signal(self) -> CoinglassCompositeSignal:
        """生成Coinglass综合信号"""
        cache_key = "composite_signal"
        
        # 检查缓存
        if self._is_cache_valid(cache_key):
            return self.analysis_cache[cache_key]
        
        try:
            # 获取各个子信号
            sentiment_signal = await self.analyze_market_sentiment()
            funding_signal = await self.analyze_funding_rates()
            oi_signal = await self.analyze_open_interest()
            etf_signal = await self.analyze_etf_flows()
            
            # 计算综合得分
            overall_score = self._calculate_composite_score(
                sentiment_signal, funding_signal, oi_signal, etf_signal
            )
            
            # 评估信号强度
            signal_strength = self._assess_signal_strength(overall_score, [
                sentiment_signal.confidence,
                0.8,  # funding signal confidence (assumed)
                0.8,  # oi signal confidence (assumed)
                0.7   # etf signal confidence (assumed)
            ])
            
            # 判断市场状态
            market_regime = self._determine_market_regime(overall_score, sentiment_signal, funding_signal)
            
            # 风险评估
            risk_assessment = self._assess_overall_risk(oi_signal, funding_signal, sentiment_signal)
            
            # 识别关键因素
            key_factors = self._identify_key_factors(sentiment_signal, funding_signal, oi_signal, etf_signal)
            
            # 计算综合置信度
            confidence = self._calculate_composite_confidence([
                sentiment_signal.confidence,
                0.8, 0.8, 0.7  # 其他信号的假设置信度
            ])
            
            composite_signal = CoinglassCompositeSignal(
                overall_score=overall_score,
                signal_strength=signal_strength,
                market_regime=market_regime,
                risk_assessment=risk_assessment,
                key_factors=key_factors,
                sentiment_signal=sentiment_signal,
                funding_signal=funding_signal,
                oi_signal=oi_signal,
                etf_signal=etf_signal,
                timestamp=datetime.utcnow(),
                confidence=confidence
            )
            
            # 缓存结果
            self.analysis_cache[cache_key] = composite_signal
            self.analysis_cache[f"{cache_key}_time"] = datetime.utcnow()
            
            return composite_signal
            
        except Exception as e:
            logger.error(f"生成综合信号失败: {e}")
            raise
    
    def _calculate_composite_score(self, sentiment: MarketSentimentSignal, 
                                 funding: FundingRateSignal,
                                 oi: OpenInterestSignal, 
                                 etf: ETFFlowSignal) -> float:
        """计算综合得分"""
        # 权重分配
        weights = {
            "sentiment": 0.3,
            "funding": 0.25,
            "oi": 0.25,
            "etf": 0.2
        }
        
        # 标准化各个信号到 -100 到 100 范围
        sentiment_norm = sentiment.sentiment_score
        
        funding_norm = 0
        if funding.market_heat == "overheated":
            funding_norm = -40
        elif funding.market_heat == "hot":
            funding_norm = -20
        elif funding.market_heat == "cold":
            funding_norm = 20
        
        oi_norm = 0
        if oi.risk_level == "extreme":
            oi_norm = -50
        elif oi.risk_level == "high":
            oi_norm = -25
        elif oi.risk_level == "low":
            oi_norm = 25
        
        etf_norm = 0
        if etf.institutional_sentiment == "bullish":
            etf_norm = 30
        elif etf.institutional_sentiment == "bearish":
            etf_norm = -30
        
        # 加权平均
        composite_score = (
            sentiment_norm * weights["sentiment"] +
            funding_norm * weights["funding"] +
            oi_norm * weights["oi"] +
            etf_norm * weights["etf"]
        )
        
        return max(-100, min(100, composite_score))
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.analysis_cache:
            return False
        
        cache_time_key = f"{cache_key}_time"
        if cache_time_key not in self.analysis_cache:
            return False
        
        cache_time = self.analysis_cache[cache_time_key]
        return (datetime.utcnow() - cache_time).total_seconds() < self.cache_duration
    
    # 辅助方法实现
    def _create_default_sentiment_signal(self) -> MarketSentimentSignal:
        return MarketSentimentSignal(
            sentiment_score=0, confidence=0, trend="neutral", strength="weak",
            components={}, timestamp=datetime.utcnow()
        )
    
    def _create_default_funding_signal(self) -> FundingRateSignal:
        return FundingRateSignal(
            overall_rate=0, rate_trend="stable", market_heat="normal",
            divergence_score=0, exchanges_data=[], timestamp=datetime.utcnow()
        )
    
    def _create_default_oi_signal(self) -> OpenInterestSignal:
        return OpenInterestSignal(
            total_oi=0, oi_trend="stable", oi_change_24h=0,
            market_structure="neutral", risk_level="moderate",
            exchanges_data=[], timestamp=datetime.utcnow()
        )
    
    def _create_default_etf_signal(self) -> ETFFlowSignal:
        return ETFFlowSignal(
            btc_flow=0, eth_flow=0, flow_trend="neutral",
            institutional_sentiment="neutral", flow_strength="weak",
            timestamp=datetime.utcnow()
        )
    
    def _calculate_sentiment_trend(self, values: List[int]) -> str:
        if len(values) < 2:
            return "neutral"
        
        recent_avg = statistics.mean(values[:5]) if len(values) >= 5 else statistics.mean(values)
        older_avg = statistics.mean(values[5:]) if len(values) > 5 else recent_avg
        
        if recent_avg > older_avg + 5:
            return "bullish"
        elif recent_avg < older_avg - 5:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_sentiment_strength(self, current: int, values: List[int]) -> str:
        if current <= 20 or current >= 80:
            return "strong"
        elif current <= 35 or current >= 65:
            return "moderate"
        else:
            return "weak"
    
    def _calculate_sentiment_confidence(self, data: List[Dict], lookback_days: int) -> float:
        # 基于数据点数量和时间跨度计算置信度
        data_points = len(data)
        expected_points = lookback_days * 2  # 假设每天2个数据点
        
        data_coverage = min(1.0, data_points / expected_points)
        return max(0.3, data_coverage)
    
    def _calculate_momentum(self, values: List[int]) -> float:
        if len(values) < 3:
            return 0
        
        changes = [values[i] - values[i+1] for i in range(len(values)-1)]
        return statistics.mean(changes) if changes else 0
    
    def _calculate_volatility(self, values: List[int]) -> float:
        if len(values) < 2:
            return 0
        
        return statistics.stdev(values)
    
    # 更多辅助方法的简化实现...
    def _aggregate_funding_data(self, data: List[Dict]) -> List[Dict]:
        return []  # 简化实现
    
    def _calculate_weighted_funding_rate(self, exchanges_data: List[Dict]) -> float:
        return 0.0  # 简化实现
    
    def _analyze_funding_trend(self, data: List[Dict]) -> str:
        return "stable"  # 简化实现
    
    def _assess_market_heat(self, rate: float, exchanges_data: List[Dict]) -> str:
        return "normal"  # 简化实现
    
    def _calculate_funding_divergence(self, exchanges_data: List[Dict]) -> float:
        return 0.0  # 简化实现
    
    def _aggregate_oi_data(self, data: List[Dict]) -> List[Dict]:
        return []  # 简化实现
    
    def _calculate_oi_change(self, data: List[Dict]) -> float:
        return 0.0  # 简化实现
    
    def _analyze_oi_trend(self, change: float) -> str:
        if change > 5:
            return "increasing"
        elif change < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _assess_market_structure(self, change: float, exchanges_data: List[Dict]) -> str:
        return "neutral"  # 简化实现
    
    def _assess_oi_risk_level(self, total_oi: float, change: float) -> str:
        return "moderate"  # 简化实现
    
    def _calculate_asset_flow(self, data: List[Dict], asset: str) -> float:
        return 0.0  # 简化实现
    
    def _assess_institutional_sentiment(self, btc_flow: float, eth_flow: float) -> str:
        total_flow = btc_flow + eth_flow
        if total_flow > 50_000_000:
            return "bullish"
        elif total_flow < -50_000_000:
            return "bearish"
        else:
            return "neutral"
    
    def _assess_flow_strength(self, flow: float) -> str:
        if flow > 500_000_000:
            return "strong"
        elif flow > 100_000_000:
            return "moderate"
        else:
            return "weak"
    
    def _assess_signal_strength(self, score: float, confidences: List[float]) -> str:
        avg_confidence = statistics.mean(confidences)
        if abs(score) > 50 and avg_confidence > 0.8:
            return "strong"
        elif abs(score) > 25 and avg_confidence > 0.6:
            return "moderate"
        else:
            return "weak"
    
    def _determine_market_regime(self, score: float, sentiment: MarketSentimentSignal, funding: FundingRateSignal) -> str:
        if score > 30:
            return "bull"
        elif score < -30:
            return "bear"
        else:
            return "sideways"
    
    def _assess_overall_risk(self, oi: OpenInterestSignal, funding: FundingRateSignal, sentiment: MarketSentimentSignal) -> str:
        risk_factors = 0
        
        if oi.risk_level in ["high", "extreme"]:
            risk_factors += 1
        if funding.market_heat == "overheated":
            risk_factors += 1
        if abs(sentiment.sentiment_score) > 70:  # 极端情绪
            risk_factors += 1
        
        if risk_factors >= 2:
            return "extreme"
        elif risk_factors == 1:
            return "high"
        else:
            return "moderate"
    
    def _identify_key_factors(self, sentiment: MarketSentimentSignal, funding: FundingRateSignal, 
                            oi: OpenInterestSignal, etf: ETFFlowSignal) -> List[str]:
        factors = []
        
        if abs(sentiment.sentiment_score) > 40:
            factors.append(f"Extreme market sentiment ({sentiment.sentiment_score})")
        
        if funding.market_heat in ["hot", "overheated"]:
            factors.append(f"High funding rates ({funding.market_heat})")
        
        if oi.risk_level in ["high", "extreme"]:
            factors.append(f"Elevated open interest risk ({oi.risk_level})")
        
        if etf.flow_strength == "strong":
            factors.append(f"Strong ETF flows ({etf.flow_trend})")
        
        return factors
    
    def _calculate_composite_confidence(self, confidences: List[float]) -> float:
        return statistics.mean(confidences)

# 全局分析器实例
coinglass_analyzer = CoinglassAnalyzer()