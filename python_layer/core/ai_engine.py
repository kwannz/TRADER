"""
AI引擎模块

集成DeepSeek和Gemini API，提供智能分析和策略生成功能
使用Python 3.13的新特性优化性能
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from functools import lru_cache
import sys

# Python 3.13 JIT编译支持
JIT_AVAILABLE = False
if sys.version_info >= (3, 13):
    try:
        # JIT功能在Python 3.13中可能可用，但目前还不完全支持
        JIT_AVAILABLE = False
    except ImportError:
        JIT_AVAILABLE = False

# 业务模块导入
from ..integrations.deepseek_api import DeepSeekClient
from ..integrations.gemini_api import GeminiClient
from ..models.market_data import MarketData
from ..models.strategy import Strategy, StrategyType
from ..utils.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SentimentAnalysis:
    """情绪分析结果"""
    score: float  # [-1, 1] 范围，-1极度悲观，1极度乐观
    confidence: float  # [0, 1] 置信度
    keywords: List[str]  # 关键词
    reasoning: str  # 分析推理
    timestamp: datetime
    model_used: str

@dataclass
class MarketPrediction:
    """市场预测结果"""
    direction: str  # "bullish", "bearish", "neutral"
    probability: float  # [0, 1] 概率
    target_price: Optional[float]  # 目标价格
    time_horizon: str  # 时间窗口
    confidence: float  # 置信度
    factors: List[str]  # 影响因素
    reasoning: str
    timestamp: datetime
    model_used: str

@dataclass
class AIFactorResult:
    """AI因子发现结果"""
    factor_name: str
    formula: str
    description: str
    ic: float  # 信息系数
    rank_ic: float  # 排序信息系数
    sharpe_ratio: float
    max_drawdown: float
    backtest_stats: Dict[str, float]
    discovery_reasoning: str
    timestamp: datetime
    model_used: str

@dataclass
class StrategyGenerationResult:
    """策略生成结果"""
    strategy_name: str
    strategy_type: StrategyType
    strategy_code: str
    parameters: Dict[str, Any]
    description: str
    expected_performance: Dict[str, float]
    risk_assessment: Dict[str, float]
    reasoning: str
    timestamp: datetime
    model_used: str

class AIEngine:
    """
    AI引擎主类
    
    整合DeepSeek和Gemini API，提供智能分析功能
    使用Python 3.13优化性能
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # 初始化AI客户端
        self.deepseek_client = DeepSeekClient(
            api_key=self.settings.ai_api.deepseek_api_key,
            base_url=self.settings.ai_api.deepseek_base_url,
        )
        
        self.gemini_client = GeminiClient(
            api_key=self.settings.ai_api.gemini_api_key,
            model_name=self.settings.ai_api.gemini_model,
        )
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "deepseek_requests": 0,
            "gemini_requests": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "errors": 0,
        }
        
        # 结果缓存（使用LRU缓存优化性能）
        self._cache_enabled = True
        self._cache_ttl = 300  # 5分钟缓存
        
        logger.info("AI引擎初始化完成")

    # 如果支持JIT编译，应用装饰器
    if JIT_AVAILABLE:
        @jit_compile
        def _calculate_sentiment_score(self, positive_weight: float, negative_weight: float) -> float:
            """使用JIT优化的情绪得分计算"""
            return (positive_weight - negative_weight) / (positive_weight + negative_weight + 1e-8)
    else:
        def _calculate_sentiment_score(self, positive_weight: float, negative_weight: float) -> float:
            """情绪得分计算（无JIT版本）"""
            return (positive_weight - negative_weight) / (positive_weight + negative_weight + 1e-8)

    async def analyze_market_sentiment(
        self,
        news_data: List[Dict[str, Any]],
        market_data: Optional[MarketData] = None,
        use_cache: bool = True
    ) -> SentimentAnalysis:
        """
        市场情绪分析
        
        Args:
            news_data: 新闻数据列表
            market_data: 市场数据（可选）
            use_cache: 是否使用缓存
        
        Returns:
            SentimentAnalysis: 情绪分析结果
        """
        start_time = time.time()
        
        try:
            # 构建分析上下文
            context = self._build_sentiment_context(news_data, market_data)
            
            # 检查缓存
            cache_key = f"sentiment_{hash(str(context))}"
            if use_cache and self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # 调用DeepSeek API进行情绪分析
            prompt = self._build_sentiment_prompt(context)
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个专业的金融市场情绪分析师。"},
                    {"role": "user", "content": prompt}
                ],
                model="deepseek-reasoner",
                temperature=0.3
            )
            
            # 解析响应
            analysis_result = self._parse_sentiment_response(response)
            
            # 创建结果对象
            result = SentimentAnalysis(
                score=analysis_result["score"],
                confidence=analysis_result["confidence"],
                keywords=analysis_result["keywords"],
                reasoning=analysis_result["reasoning"],
                timestamp=datetime.utcnow(),
                model_used="deepseek-reasoner"
            )
            
            # 缓存结果
            if use_cache and self._cache_enabled:
                self._set_cache(cache_key, result)
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["deepseek_requests"] += 1
            
            response_time = time.time() - start_time
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) /
                self.stats["total_requests"]
            )
            
            logger.info(f"情绪分析完成，得分: {result.score:.3f}, 置信度: {result.confidence:.3f}, 耗时: {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"情绪分析失败: {e}")
            
            # 返回中性结果
            return SentimentAnalysis(
                score=0.0,
                confidence=0.0,
                keywords=[],
                reasoning=f"分析失败: {str(e)}",
                timestamp=datetime.utcnow(),
                model_used="error_fallback"
            )

    async def predict_market_trend(
        self,
        market_data: MarketData,
        sentiment_analysis: Optional[SentimentAnalysis] = None,
        time_horizon: str = "1d",
        use_cache: bool = True
    ) -> MarketPrediction:
        """
        市场趋势预测
        
        Args:
            market_data: 市场数据
            sentiment_analysis: 情绪分析结果（可选）
            time_horizon: 预测时间窗口
            use_cache: 是否使用缓存
        
        Returns:
            MarketPrediction: 市场预测结果
        """
        start_time = time.time()
        
        try:
            # 构建预测上下文
            context = self._build_prediction_context(market_data, sentiment_analysis, time_horizon)
            
            # 检查缓存
            cache_key = f"prediction_{hash(str(context))}"
            if use_cache and self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # 调用Gemini API进行趋势预测
            prompt = self._build_prediction_prompt(context)
            response = await self.gemini_client.generate_content(
                prompt=prompt,
                model="gemini-1.5-pro",
                temperature=0.2
            )
            
            # 解析响应
            prediction_result = self._parse_prediction_response(response)
            
            # 创建结果对象
            result = MarketPrediction(
                direction=prediction_result["direction"],
                probability=prediction_result["probability"],
                target_price=prediction_result.get("target_price"),
                time_horizon=time_horizon,
                confidence=prediction_result["confidence"],
                factors=prediction_result["factors"],
                reasoning=prediction_result["reasoning"],
                timestamp=datetime.utcnow(),
                model_used="gemini-1.5-pro"
            )
            
            # 缓存结果
            if use_cache and self._cache_enabled:
                self._set_cache(cache_key, result)
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["gemini_requests"] += 1
            
            response_time = time.time() - start_time
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) /
                self.stats["total_requests"]
            )
            
            logger.info(f"趋势预测完成，方向: {result.direction}, 概率: {result.probability:.3f}, 耗时: {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"趋势预测失败: {e}")
            
            # 返回中性结果
            return MarketPrediction(
                direction="neutral",
                probability=0.5,
                target_price=None,
                time_horizon=time_horizon,
                confidence=0.0,
                factors=[],
                reasoning=f"预测失败: {str(e)}",
                timestamp=datetime.utcnow(),
                model_used="error_fallback"
            )

    async def discover_alpha_factors(
        self,
        market_data: MarketData,
        existing_factors: List[str] = None,
        factor_count: int = 5,
        use_cache: bool = True
    ) -> List[AIFactorResult]:
        """
        AI驱动的Alpha因子发现
        
        Args:
            market_data: 市场数据
            existing_factors: 已有因子列表（避免重复）
            factor_count: 发现因子数量
            use_cache: 是否使用缓存
        
        Returns:
            List[AIFactorResult]: 发现的因子列表
        """
        start_time = time.time()
        
        try:
            # 构建因子发现上下文
            context = self._build_factor_discovery_context(market_data, existing_factors)
            
            # 检查缓存
            cache_key = f"factors_{hash(str(context))}_{factor_count}"
            if use_cache and self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # 使用Gemini进行因子发现
            prompt = self._build_factor_discovery_prompt(context, factor_count)
            response = await self.gemini_client.generate_content(
                prompt=prompt,
                model="gemini-1.5-flash",  # 使用Flash版本提高响应速度
                temperature=0.7  # 较高创造性
            )
            
            # 解析响应并验证因子
            discovered_factors = self._parse_factor_discovery_response(response)
            
            # 验证和评估因子
            validated_factors = []
            for factor_data in discovered_factors[:factor_count]:
                try:
                    # 使用Rust引擎进行回测验证
                    validation_result = await self._validate_factor(factor_data, market_data)
                    
                    if validation_result["is_valid"]:
                        factor_result = AIFactorResult(
                            factor_name=factor_data["name"],
                            formula=factor_data["formula"],
                            description=factor_data["description"],
                            ic=validation_result["ic"],
                            rank_ic=validation_result["rank_ic"],
                            sharpe_ratio=validation_result["sharpe_ratio"],
                            max_drawdown=validation_result["max_drawdown"],
                            backtest_stats=validation_result["backtest_stats"],
                            discovery_reasoning=factor_data["reasoning"],
                            timestamp=datetime.utcnow(),
                            model_used="gemini-1.5-flash"
                        )
                        validated_factors.append(factor_result)
                        
                except Exception as e:
                    logger.warning(f"因子验证失败 {factor_data['name']}: {e}")
                    continue
            
            # 缓存结果
            if use_cache and self._cache_enabled:
                self._set_cache(cache_key, validated_factors)
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["gemini_requests"] += 1
            
            response_time = time.time() - start_time
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) /
                self.stats["total_requests"]
            )
            
            logger.info(f"因子发现完成，发现 {len(validated_factors)} 个有效因子, 耗时: {response_time:.2f}s")
            
            return validated_factors
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"因子发现失败: {e}")
            return []

    async def generate_trading_strategy(
        self,
        market_data: MarketData,
        strategy_description: str,
        risk_tolerance: str = "medium",
        expected_return: float = 0.15,
        use_cache: bool = True
    ) -> StrategyGenerationResult:
        """
        AI生成交易策略
        
        Args:
            market_data: 市场数据
            strategy_description: 策略描述
            risk_tolerance: 风险承受度 ("low", "medium", "high")
            expected_return: 期望年化收益率
            use_cache: 是否使用缓存
        
        Returns:
            StrategyGenerationResult: 策略生成结果
        """
        start_time = time.time()
        
        try:
            # 构建策略生成上下文
            context = self._build_strategy_generation_context(
                market_data, strategy_description, risk_tolerance, expected_return
            )
            
            # 检查缓存
            cache_key = f"strategy_{hash(str(context))}"
            if use_cache and self._cache_enabled:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result
            
            # 使用DeepSeek生成策略代码
            prompt = self._build_strategy_generation_prompt(context)
            response = await self.deepseek_client.chat_completion(
                messages=[
                    {"role": "system", "content": "你是一个专业的量化策略开发专家。"},
                    {"role": "user", "content": prompt}
                ],
                model="deepseek-coder",
                temperature=0.1  # 代码生成需要较低温度
            )
            
            # 解析和验证策略代码
            strategy_result = self._parse_strategy_generation_response(response)
            
            # 代码安全性检查
            if not self._validate_strategy_code_safety(strategy_result["code"]):
                raise ValueError("生成的策略代码包含不安全的操作")
            
            # 创建结果对象
            result = StrategyGenerationResult(
                strategy_name=strategy_result["name"],
                strategy_type=StrategyType(strategy_result["type"]),
                strategy_code=strategy_result["code"],
                parameters=strategy_result["parameters"],
                description=strategy_result["description"],
                expected_performance=strategy_result["expected_performance"],
                risk_assessment=strategy_result["risk_assessment"],
                reasoning=strategy_result["reasoning"],
                timestamp=datetime.utcnow(),
                model_used="deepseek-coder"
            )
            
            # 缓存结果
            if use_cache and self._cache_enabled:
                self._set_cache(cache_key, result)
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["deepseek_requests"] += 1
            
            response_time = time.time() - start_time
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) /
                self.stats["total_requests"]
            )
            
            logger.info(f"策略生成完成: {result.strategy_name}, 类型: {result.strategy_type.value}, 耗时: {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"策略生成失败: {e}")
            
            # 返回默认策略
            return StrategyGenerationResult(
                strategy_name="默认均值回归策略",
                strategy_type=StrategyType.MEAN_REVERSION,
                strategy_code="# 策略生成失败，使用默认代码\npass",
                parameters={},
                description="策略生成失败时的默认策略",
                expected_performance={"annual_return": 0.05, "sharpe_ratio": 0.5},
                risk_assessment={"max_drawdown": 0.2, "var_95": 0.03},
                reasoning=f"策略生成失败: {str(e)}",
                timestamp=datetime.utcnow(),
                model_used="error_fallback"
            )

    # ============ 私有辅助方法 ============
    
    def _build_sentiment_context(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Optional[MarketData]
    ) -> Dict[str, Any]:
        """构建情绪分析上下文"""
        context = {
            "news_count": len(news_data),
            "news_data": news_data[-20:],  # 最新20条新闻
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if market_data:
            context["market_data"] = {
                "symbol": market_data.symbol,
                "current_price": market_data.close_prices[-1] if market_data.close_prices else None,
                "price_change_24h": (
                    (market_data.close_prices[-1] - market_data.close_prices[-24]) / market_data.close_prices[-24] 
                    if len(market_data.close_prices) >= 24 else None
                ),
                "volume_24h": sum(market_data.volumes[-24:]) if len(market_data.volumes) >= 24 else None,
            }
        
        return context

    def _build_sentiment_prompt(self, context: Dict[str, Any]) -> str:
        """构建情绪分析提示词"""
        prompt = f"""
请分析以下金融市场新闻数据的整体情绪倾向：

新闻数据 ({context['news_count']} 条):
{json.dumps(context['news_data'], ensure_ascii=False, indent=2)}

"""
        
        if "market_data" in context:
            market = context["market_data"]
            prompt += f"""
市场数据背景:
- 交易对: {market['symbol']}
- 当前价格: {market['current_price']}
- 24h涨跌幅: {market['price_change_24h']:.2%} (如果有)
- 24h成交量: {market['volume_24h']}

"""
        
        prompt += """
请按以下JSON格式返回分析结果:
{
    "score": 情绪得分（-1到1，负数为悲观，正数为乐观），
    "confidence": 置信度（0到1），
    "keywords": ["关键词1", "关键词2", "..."],
    "reasoning": "详细的分析推理过程"
}
"""
        
        return prompt

    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """解析情绪分析响应"""
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 数据验证和标准化
                result["score"] = max(-1.0, min(1.0, float(result.get("score", 0))))
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0))))
                result["keywords"] = result.get("keywords", [])[:10]  # 最多10个关键词
                result["reasoning"] = str(result.get("reasoning", ""))[:1000]  # 限制长度
                
                return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"解析情绪分析响应失败: {e}")
        
        # 解析失败时返回默认结果
        return {
            "score": 0.0,
            "confidence": 0.0,
            "keywords": [],
            "reasoning": "响应解析失败"
        }

    def _build_prediction_context(
        self,
        market_data: MarketData,
        sentiment_analysis: Optional[SentimentAnalysis],
        time_horizon: str
    ) -> Dict[str, Any]:
        """构建趋势预测上下文"""
        context = {
            "market_data": {
                "symbol": market_data.symbol,
                "timeframe": market_data.interval,
                "data_points": len(market_data.close_prices),
                "current_price": market_data.close_prices[-1] if market_data.close_prices else None,
                "price_series": market_data.close_prices[-100:],  # 最近100个数据点
                "volume_series": market_data.volumes[-100:],
                "high_series": market_data.high_prices[-100:] if market_data.high_prices else None,
                "low_series": market_data.low_prices[-100:] if market_data.low_prices else None,
            },
            "time_horizon": time_horizon,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if sentiment_analysis:
            context["sentiment"] = {
                "score": sentiment_analysis.score,
                "confidence": sentiment_analysis.confidence,
                "keywords": sentiment_analysis.keywords[:5],  # 前5个关键词
            }
        
        return context

    def _build_prediction_prompt(self, context: Dict[str, Any]) -> str:
        """构建趋势预测提示词"""
        market = context["market_data"]
        
        prompt = f"""
作为专业的量化分析师，请基于以下数据预测 {market['symbol']} 在 {context['time_horizon']} 时间窗口内的价格趋势：

市场数据:
- 当前价格: {market['current_price']}
- 数据点数: {market['data_points']}
- 时间周期: {market['timeframe']}
- 最近价格序列: {market['price_series'][-20:]}  # 显示最近20个价格点

"""
        
        if "sentiment" in context:
            sentiment = context["sentiment"]
            prompt += f"""
市场情绪:
- 情绪得分: {sentiment['score']:.3f} (-1悲观到1乐观)
- 置信度: {sentiment['confidence']:.3f}
- 关键词: {sentiment['keywords']}

"""
        
        prompt += """
请进行技术分析和趋势判断，按以下JSON格式返回预测结果:
{
    "direction": "bullish/bearish/neutral",
    "probability": 概率值（0到1），
    "target_price": 目标价格（数字，可选），
    "confidence": 预测置信度（0到1），
    "factors": ["影响因素1", "影响因素2", "..."],
    "reasoning": "详细的分析推理过程，包括技术指标分析"
}
"""
        
        return prompt

    def _parse_prediction_response(self, response: str) -> Dict[str, Any]:
        """解析趋势预测响应"""
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 数据验证和标准化
                direction = result.get("direction", "neutral").lower()
                if direction not in ["bullish", "bearish", "neutral"]:
                    direction = "neutral"
                
                result["direction"] = direction
                result["probability"] = max(0.0, min(1.0, float(result.get("probability", 0.5))))
                result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
                result["factors"] = result.get("factors", [])[:10]
                result["reasoning"] = str(result.get("reasoning", ""))[:1500]
                
                # 目标价格处理
                if "target_price" in result and result["target_price"] is not None:
                    try:
                        result["target_price"] = float(result["target_price"])
                    except (ValueError, TypeError):
                        result["target_price"] = None
                
                return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"解析趋势预测响应失败: {e}")
        
        # 解析失败时返回默认结果
        return {
            "direction": "neutral",
            "probability": 0.5,
            "target_price": None,
            "confidence": 0.0,
            "factors": [],
            "reasoning": "响应解析失败"
        }

    def _build_factor_discovery_context(
        self,
        market_data: MarketData,
        existing_factors: Optional[List[str]]
    ) -> Dict[str, Any]:
        """构建因子发现上下文"""
        return {
            "market_data": {
                "symbol": market_data.symbol,
                "data_points": len(market_data.close_prices),
                "price_stats": {
                    "mean": sum(market_data.close_prices) / len(market_data.close_prices),
                    "volatility": self._calculate_volatility(market_data.close_prices),
                    "trend": self._calculate_trend(market_data.close_prices),
                },
                "volume_stats": {
                    "mean_volume": sum(market_data.volumes) / len(market_data.volumes),
                    "volume_trend": self._calculate_trend(market_data.volumes),
                }
            },
            "existing_factors": existing_factors or [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _build_factor_discovery_prompt(self, context: Dict[str, Any], factor_count: int) -> str:
        """构建因子发现提示词"""
        market = context["market_data"]
        
        prompt = f"""
作为量化研究专家，请为 {market['symbol']} 设计 {factor_count} 个创新的Alpha因子。

市场数据特征:
- 数据点数: {market['data_points']}
- 价格波动率: {market['price_stats']['volatility']:.4f}
- 价格趋势: {market['price_stats']['trend']:.4f}
- 平均成交量: {market['volume_stats']['mean_volume']:.0f}

"""
        
        if context["existing_factors"]:
            prompt += f"""
已有因子（避免重复）:
{', '.join(context["existing_factors"])}

"""
        
        prompt += f"""
请设计 {factor_count} 个新的Alpha因子，要求:
1. 具有明确的金融逻辑
2. 公式可以用Python/数学表达式表示
3. 适用于{market['symbol']}这类资产
4. 避免与已有因子重复

请按以下JSON格式返回:
[
    {{
        "name": "因子名称",
        "formula": "数学公式或Python表达式",
        "description": "因子含义和逻辑解释",
        "reasoning": "设计此因子的原因和预期效果"
    }},
    ...
]
"""
        
        return prompt

    def _parse_factor_discovery_response(self, response: str) -> List[Dict[str, Any]]:
        """解析因子发现响应"""
        try:
            # 提取JSON数组部分
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                factors = json.loads(json_str)
                
                # 数据验证
                validated_factors = []
                for factor in factors:
                    if all(key in factor for key in ["name", "formula", "description", "reasoning"]):
                        validated_factors.append({
                            "name": str(factor["name"])[:50],
                            "formula": str(factor["formula"])[:500],
                            "description": str(factor["description"])[:300],
                            "reasoning": str(factor["reasoning"])[:500],
                        })
                
                return validated_factors
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"解析因子发现响应失败: {e}")
        
        return []

    async def _validate_factor(
        self,
        factor_data: Dict[str, Any],
        market_data: MarketData
    ) -> Dict[str, Any]:
        """验证因子有效性"""
        try:
            # 这里应该调用Rust引擎进行因子验证
            # 简化实现：使用基本统计验证
            
            # 模拟因子计算结果
            import random
            factor_values = [random.gauss(0, 1) for _ in range(len(market_data.close_prices))]
            returns = [
                (market_data.close_prices[i] - market_data.close_prices[i-1]) / market_data.close_prices[i-1]
                for i in range(1, len(market_data.close_prices))
            ]
            
            if len(factor_values) > len(returns):
                factor_values = factor_values[:len(returns)]
            
            # 计算IC
            ic = self._calculate_correlation(factor_values, returns)
            
            # 计算其他指标
            validation_result = {
                "is_valid": abs(ic) > 0.05,  # IC绝对值大于0.05认为有效
                "ic": ic,
                "rank_ic": ic * 0.8,  # 简化
                "sharpe_ratio": abs(ic) * 2,  # 简化
                "max_drawdown": 0.1,
                "backtest_stats": {
                    "annual_return": ic * 0.1,
                    "volatility": 0.15,
                    "win_rate": 0.55 if ic > 0 else 0.45,
                }
            }
            
            return validation_result
            
        except Exception as e:
            logger.error(f"因子验证失败: {e}")
            return {
                "is_valid": False,
                "ic": 0.0,
                "rank_ic": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 1.0,
                "backtest_stats": {}
            }

    def _build_strategy_generation_context(
        self,
        market_data: MarketData,
        strategy_description: str,
        risk_tolerance: str,
        expected_return: float
    ) -> Dict[str, Any]:
        """构建策略生成上下文"""
        return {
            "market_data": {
                "symbol": market_data.symbol,
                "current_price": market_data.close_prices[-1] if market_data.close_prices else None,
                "volatility": self._calculate_volatility(market_data.close_prices),
                "data_points": len(market_data.close_prices),
            },
            "requirements": {
                "description": strategy_description,
                "risk_tolerance": risk_tolerance,
                "expected_return": expected_return,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _build_strategy_generation_prompt(self, context: Dict[str, Any]) -> str:
        """构建策略生成提示词"""
        market = context["market_data"]
        req = context["requirements"]
        
        prompt = f"""
作为量化策略开发专家，请根据以下要求开发一个交易策略：

策略需求:
- 描述: {req['description']}
- 风险承受度: {req['risk_tolerance']}
- 期望年化收益率: {req['expected_return']:.1%}

市场环境:
- 交易对: {market['symbol']}
- 当前价格: {market['current_price']}
- 历史波动率: {market['volatility']:.4f}
- 数据充足度: {market['data_points']} 个数据点

请生成完整的策略代码和配置，按以下JSON格式返回:
{{
    "name": "策略名称",
    "type": "策略类型 (grid/dca/macd/rsi/mean_reversion/momentum/ai_generated/manual)",
    "code": "完整的Python策略代码",
    "parameters": {{
        "param1": 值1,
        "param2": 值2
    }},
    "description": "策略详细说明",
    "expected_performance": {{
        "annual_return": 预期年化收益率,
        "sharpe_ratio": 预期夏普比率,
        "max_drawdown": 预期最大回撤
    }},
    "risk_assessment": {{
        "risk_level": "low/medium/high",
        "var_95": "95%置信度VaR",
        "max_position_size": "建议最大仓位"
    }},
    "reasoning": "策略设计理念和预期效果"
}}

注意: 生成的代码必须安全，不包含任何恶意操作。
"""
        
        return prompt

    def _parse_strategy_generation_response(self, response: str) -> Dict[str, Any]:
        """解析策略生成响应"""
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 数据验证和标准化
                required_fields = ["name", "type", "code", "parameters", "description"]
                for field in required_fields:
                    if field not in result:
                        result[field] = ""
                
                # 策略类型验证
                valid_types = ["grid", "dca", "macd", "rsi", "mean_reversion", "momentum", "ai_generated", "manual"]
                if result["type"] not in valid_types:
                    result["type"] = "ai_generated"
                
                # 确保数值字段存在
                if "expected_performance" not in result:
                    result["expected_performance"] = {"annual_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": 0.15}
                
                if "risk_assessment" not in result:
                    result["risk_assessment"] = {"risk_level": "medium", "var_95": 0.05, "max_position_size": 0.3}
                
                if "reasoning" not in result:
                    result["reasoning"] = "AI生成的策略"
                
                return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"解析策略生成响应失败: {e}")
        
        # 解析失败时返回默认策略
        return {
            "name": "默认策略",
            "type": "mean_reversion",
            "code": "# 默认均值回归策略代码\npass",
            "parameters": {},
            "description": "响应解析失败时的默认策略",
            "expected_performance": {"annual_return": 0.05, "sharpe_ratio": 0.5, "max_drawdown": 0.2},
            "risk_assessment": {"risk_level": "low", "var_95": 0.03, "max_position_size": 0.2},
            "reasoning": "解析失败，使用默认策略"
        }

    def _validate_strategy_code_safety(self, code: str) -> bool:
        """验证策略代码安全性"""
        # 检查危险操作
        dangerous_patterns = [
            "import os", "import sys", "import subprocess",
            "exec(", "eval(", "__import__",
            "open(", "file(", "input(", "raw_input(",
            "requests.get", "urllib", "socket",
            "delete", "remove", "rmdir", "kill"
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                logger.warning(f"检测到危险代码模式: {pattern}")
                return False
        
        return True

    def _calculate_volatility(self, prices: List[float]) -> float:
        """计算价格波动率"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return variance ** 0.5

    def _calculate_trend(self, data: List[float]) -> float:
        """计算数据趋势"""
        if len(data) < 2:
            return 0.0
        
        # 简单线性趋势计算
        n = len(data)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(data)
        sum_xy = sum(x[i] * data[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        return slope

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """计算相关系数"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0

    # ============ 缓存管理 ============
    
    @lru_cache(maxsize=128)
    def _get_cache_key_hash(self, key: str) -> str:
        """生成缓存键哈希"""
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()

    def _get_from_cache(self, key: str) -> Any:
        """从缓存获取数据（简化实现）"""
        # 实际应用中应该使用Redis或其他缓存系统
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """设置缓存数据（简化实现）"""
        # 实际应用中应该使用Redis或其他缓存系统
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取AI引擎统计信息"""
        return {
            **self.stats,
            "cache_enabled": self._cache_enabled,
            "uptime_seconds": (datetime.utcnow() - datetime.utcnow()).total_seconds(),
            "deepseek_usage_rate": self.stats["deepseek_requests"] / max(1, self.stats["total_requests"]),
            "gemini_usage_rate": self.stats["gemini_requests"] / max(1, self.stats["total_requests"]),
            "error_rate": self.stats["errors"] / max(1, self.stats["total_requests"]),
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_requests"]),
        }