"""
DeepSeek API客户端
负责情绪分析、市场预测和风险评估
增强版本：包含完整的错误处理、重试机制和速率限制
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
from loguru import logger
import time
import hashlib
from functools import wraps

from config.settings import settings

class DeepSeekError(Exception):
    """DeepSeek API异常"""
    pass

class RateLimitError(DeepSeekError):
    """速率限制异常"""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"API速率限制，请等待{retry_after}秒后重试")

class DeepSeekClient:
    """DeepSeek API客户端 - 增强版"""
    
    def __init__(self):
        self.config = settings.get_ai_config("deepseek")
        self.client = httpx.AsyncClient(
            base_url=self.config["base_url"],
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config["timeout"], 
                write=10.0,
                pool=30.0
            ),
            headers={
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json",
                "User-Agent": "AI-Trader/1.0"
            },
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )
        
        # 速率限制和缓存
        self._request_times = []
        self._cache = {}
        self._cache_expiry = {}
        self._max_requests_per_minute = 20
        self._cache_duration = 300  # 5分钟缓存
        
        # 状态统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited": 0,
            "avg_response_time": 0.0,
            "last_request_time": None,
            "health_status": "healthy"
        }
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def _get_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """生成缓存键"""
        content = json.dumps(messages, sort_keys=True) + json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """获取缓存的响应"""
        if cache_key in self._cache:
            if datetime.utcnow() < self._cache_expiry[cache_key]:
                self.stats["cached_responses"] += 1
                logger.debug(f"使用缓存响应: {cache_key[:8]}...")
                return self._cache[cache_key]
            else:
                # 清理过期缓存
                del self._cache[cache_key]
                del self._cache_expiry[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str):
        """缓存响应"""
        self._cache[cache_key] = response
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(seconds=self._cache_duration)
        
        # 限制缓存大小
        if len(self._cache) > 100:
            # 清理最旧的缓存项
            oldest_key = min(self._cache_expiry.keys(), key=lambda k: self._cache_expiry[k])
            del self._cache[oldest_key]
            del self._cache_expiry[oldest_key]
    
    async def _check_rate_limit(self):
        """检查速率限制"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # 清理1分钟前的请求记录
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        if len(self._request_times) >= self._max_requests_per_minute:
            self.stats["rate_limited"] += 1
            self.stats["health_status"] = "rate_limited"
            raise RateLimitError(60)
        
        self._request_times.append(now)
    
    async def _make_request_with_retry(self, messages: List[Dict], 
                                     max_tokens: Optional[int] = None,
                                     temperature: float = 0.7,
                                     max_retries: int = 3,
                                     use_cache: bool = True) -> Optional[str]:
        """发送请求到DeepSeek API (带重试和缓存)"""
        start_time = time.time()
        
        try:
            # 检查缓存
            if use_cache:
                cache_key = self._get_cache_key(messages, max_tokens=max_tokens, temperature=temperature)
                cached_response = self._get_cached_response(cache_key)
                if cached_response:
                    return cached_response
            
            # 更新统计
            self.stats["total_requests"] += 1
            self.stats["last_request_time"] = datetime.utcnow().isoformat()
            
            for attempt in range(max_retries):
                try:
                    # 检查速率限制
                    await self._check_rate_limit()
                    
                    payload = {
                        "model": self.config["model"],
                        "messages": messages,
                        "max_tokens": max_tokens or self.config["max_tokens"],
                        "temperature": temperature,
                        "stream": False
                    }
                    
                    logger.debug(f"DeepSeek API请求 (尝试 {attempt + 1}/{max_retries})")
                    response = await self.client.post("/chat/completions", json=payload)
                    
                    # 处理响应状态
                    if response.status_code == 429:  # Too Many Requests
                        retry_after = int(response.headers.get("Retry-After", 60))
                        if attempt < max_retries - 1:
                            logger.warning(f"API速率限制，等待{retry_after}秒后重试...")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise RateLimitError(retry_after)
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # 验证响应格式
                    if "choices" not in result or not result["choices"]:
                        raise DeepSeekError("API响应格式错误：缺少choices字段")
                    
                    content = result["choices"][0]["message"]["content"].strip()
                    if not content:
                        raise DeepSeekError("API返回空内容")
                    
                    # 缓存响应
                    if use_cache:
                        self._cache_response(cache_key, content)
                    
                    # 更新统计
                    self.stats["successful_requests"] += 1
                    self.stats["health_status"] = "healthy"
                    
                    # 更新响应时间统计
                    response_time = time.time() - start_time
                    if self.stats["avg_response_time"] == 0:
                        self.stats["avg_response_time"] = response_time
                    else:
                        self.stats["avg_response_time"] = (
                            self.stats["avg_response_time"] * 0.8 + response_time * 0.2
                        )
                    
                    logger.debug(f"DeepSeek API请求成功，响应时间: {response_time:.2f}秒")
                    return content
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [401, 403]:
                        # 认证错误，不重试
                        logger.error(f"DeepSeek API认证错误: {e.response.status_code}")
                        self.stats["health_status"] = "auth_error"
                        raise DeepSeekError(f"认证错误: {e.response.status_code}")
                    elif e.response.status_code == 429:
                        # 速率限制，已在上面处理
                        continue
                    elif e.response.status_code >= 500:
                        # 服务器错误，可以重试
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 指数退避
                            logger.warning(f"服务器错误 {e.response.status_code}，{wait_time}秒后重试...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"DeepSeek服务器错误: {e.response.status_code}")
                            self.stats["health_status"] = "server_error"
                            raise DeepSeekError(f"服务器错误: {e.response.status_code}")
                    else:
                        logger.error(f"DeepSeek API HTTP错误: {e.response.status_code} - {e.response.text}")
                        raise DeepSeekError(f"HTTP错误: {e.response.status_code}")
                
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"网络请求失败，{wait_time}秒后重试: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"DeepSeek网络请求失败: {e}")
                        self.stats["health_status"] = "network_error"
                        raise DeepSeekError(f"网络错误: {e}")
                        
                except RateLimitError:
                    # 重新抛出速率限制异常
                    raise
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"未知错误，{wait_time}秒后重试: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"DeepSeek请求失败: {e}")
                        self.stats["health_status"] = "unknown_error"
                        raise DeepSeekError(f"未知错误: {e}")
            
            # 如果所有重试都失败
            self.stats["failed_requests"] += 1
            return None
            
        except (RateLimitError, DeepSeekError):
            self.stats["failed_requests"] += 1
            raise
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"DeepSeek请求异常: {e}")
            return None
    
    async def _make_request(self, messages: List[Dict], 
                          max_tokens: Optional[int] = None,
                          temperature: float = 0.7) -> Optional[str]:
        """发送请求到DeepSeek API (兼容性包装器)"""
        try:
            return await self._make_request_with_retry(
                messages, max_tokens, temperature, 
                max_retries=3, use_cache=True
            )
        except (RateLimitError, DeepSeekError):
            return None
        except Exception as e:
            logger.error(f"DeepSeek API请求失败: {e}")
            return None
    
    async def analyze_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """新闻情绪分析"""
        try:
            # 构建新闻文本
            news_text = ""
            for news in news_data[:10]:  # 最多分析10条新闻
                news_text += f"标题: {news.get('title', '')}\n内容: {news.get('content', '')[:200]}...\n\n"
            
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的加密货币市场情绪分析师。请分析以下新闻内容，并返回JSON格式的分析结果：
                    {
                        "sentiment_score": 数值（-1到1，-1极度悲观，0中性，1极度乐观）,
                        "confidence": 置信度（0到1）,
                        "key_factors": [影响情绪的关键因素列表],
                        "market_impact": "对市场影响的评估",
                        "recommendation": "基于情绪的交易建议"
                    }"""
                },
                {
                    "role": "user", 
                    "content": f"请分析以下加密货币相关新闻的市场情绪：\n\n{news_text}"
                }
            ]
            
            response = await self._make_request(messages, temperature=0.3)
            if not response:
                return self._get_default_sentiment()
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    # 验证结果格式
                    if self._validate_sentiment_result(result):
                        result["timestamp"] = datetime.utcnow().isoformat()
                        result["source"] = "deepseek"
                        logger.info(f"情绪分析完成: 得分={result['sentiment_score']}, 置信度={result['confidence']}")
                        return result
                        
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek情绪分析JSON解析失败: {e}")
            
            return self._get_default_sentiment()
            
        except Exception as e:
            logger.error(f"DeepSeek情绪分析失败: {e}")
            return self._get_default_sentiment()
    
    async def predict_market_trend(self, market_data: Dict, 
                                  technical_indicators: Dict) -> Dict[str, Any]:
        """市场趋势预测"""
        try:
            # 构建市场数据描述
            data_summary = f"""
            当前市场数据:
            - BTC价格: ${market_data.get('BTC-USDT', {}).get('price', 'N/A')}
            - ETH价格: ${market_data.get('ETH-USDT', {}).get('price', 'N/A')}
            - 24h成交量: {market_data.get('total_volume_24h', 'N/A')}
            - 恐慌指数: {market_data.get('fear_greed_index', 'N/A')}
            
            技术指标:
            - RSI: {technical_indicators.get('rsi', 'N/A')}
            - MACD: {technical_indicators.get('macd', 'N/A')}
            - 布林带: {technical_indicators.get('bollinger', 'N/A')}
            - 移动平均线: {technical_indicators.get('ma', 'N/A')}
            """
            
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的加密货币市场分析师。基于提供的市场数据和技术指标，预测短期市场趋势。请返回JSON格式：
                    {
                        "trend_direction": "up/down/sideways",
                        "confidence": 置信度（0到1）,
                        "time_horizon": "预测时间范围",
                        "key_levels": {"support": 支撑位, "resistance": 阻力位},
                        "risk_level": "low/medium/high",
                        "reasoning": "分析理由",
                        "action_suggestion": "建议操作"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"请分析以下市场数据并预测趋势：\n{data_summary}"
                }
            ]
            
            response = await self._make_request(messages, temperature=0.5)
            if not response:
                return self._get_default_prediction()
                
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    if self._validate_prediction_result(result):
                        result["timestamp"] = datetime.utcnow().isoformat()
                        result["source"] = "deepseek"
                        logger.info(f"市场预测完成: 趋势={result['trend_direction']}, 置信度={result['confidence']}")
                        return result
                        
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek市场预测JSON解析失败: {e}")
                
            return self._get_default_prediction()
            
        except Exception as e:
            logger.error(f"DeepSeek市场预测失败: {e}")
            return self._get_default_prediction()
    
    async def assess_risk(self, portfolio_data: Dict, 
                         market_conditions: Dict) -> Dict[str, Any]:
        """风险评估"""
        try:
            risk_summary = f"""
            投资组合信息:
            - 总资产: {portfolio_data.get('total_value', 'N/A')} USDT
            - 仓位占比: {portfolio_data.get('position_ratio', 'N/A')}%
            - 当前PnL: {portfolio_data.get('unrealized_pnl', 'N/A')} USDT
            - 运行策略数: {portfolio_data.get('active_strategies', 'N/A')}
            
            市场条件:
            - 波动率: {market_conditions.get('volatility', 'N/A')}
            - 流动性: {market_conditions.get('liquidity', 'N/A')}
            - 市场情绪: {market_conditions.get('sentiment_score', 'N/A')}
            """
            
            messages = [
                {
                    "role": "system",
                    "content": """你是专业的量化交易风险管理专家。评估当前投资组合风险并给出建议。返回JSON格式：
                    {
                        "risk_score": 风险评分（0-100，100最高风险）,
                        "risk_level": "low/medium/high/extreme",
                        "main_risks": [主要风险因素列表],
                        "risk_mitigation": [风险缓解建议],
                        "position_adjustment": "仓位调整建议",
                        "stop_loss_suggestion": "止损建议",
                        "urgent_action_needed": true/false
                    }"""
                },
                {
                    "role": "user",
                    "content": f"请评估以下投资组合的风险：\n{risk_summary}"
                }
            ]
            
            response = await self._make_request(messages, temperature=0.2)
            if not response:
                return self._get_default_risk_assessment()
                
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    if self._validate_risk_result(result):
                        result["timestamp"] = datetime.utcnow().isoformat()
                        result["source"] = "deepseek"
                        logger.info(f"风险评估完成: 风险等级={result['risk_level']}, 评分={result['risk_score']}")
                        return result
                        
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek风险评估JSON解析失败: {e}")
                
            return self._get_default_risk_assessment()
            
        except Exception as e:
            logger.error(f"DeepSeek风险评估失败: {e}")
            return self._get_default_risk_assessment()
    
    async def discover_factors(self, market_data: Dict, 
                             price_history: Dict) -> Dict[str, Any]:
        """AI因子发现"""
        try:
            data_description = f"""
            市场数据样本:
            - 价格数据: {len(price_history.get('prices', []))}个数据点
            - 成交量数据: {len(price_history.get('volumes', []))}个数据点
            - 时间范围: {price_history.get('time_range', 'N/A')}
            - 相关性分析: {market_data.get('correlation_matrix', 'N/A')}
            """
            
            messages = [
                {
                    "role": "system",
                    "content": """你是AI量化因子发现专家。基于市场数据发现潜在的alpha因子。返回JSON格式：
                    {
                        "discovered_factors": [
                            {
                                "name": "因子名称",
                                "formula": "因子计算公式",
                                "description": "因子描述",
                                "expected_ic": "预期IC值",
                                "rationale": "发现理由"
                            }
                        ],
                        "factor_combinations": [因子组合建议],
                        "research_direction": "进一步研究方向",
                        "confidence": 置信度
                    }"""
                },
                {
                    "role": "user",
                    "content": f"基于以下市场数据发现新的量化因子：\n{data_description}"
                }
            ]
            
            response = await self._make_request(messages, temperature=0.8)
            if not response:
                return self._get_default_factor_discovery()
                
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    result["timestamp"] = datetime.utcnow().isoformat()
                    result["source"] = "deepseek"
                    logger.info(f"因子发现完成: 发现{len(result.get('discovered_factors', []))}个因子")
                    return result
                    
            except json.JSONDecodeError as e:
                logger.warning(f"DeepSeek因子发现JSON解析失败: {e}")
                
            return self._get_default_factor_discovery()
            
        except Exception as e:
            logger.error(f"DeepSeek因子发现失败: {e}")
            return self._get_default_factor_discovery()
    
    # 默认返回值方法
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """默认情绪分析结果"""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "key_factors": ["数据不足"],
            "market_impact": "中性",
            "recommendation": "观望",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "deepseek_fallback"
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """默认市场预测结果"""
        return {
            "trend_direction": "sideways",
            "confidence": 0.5,
            "time_horizon": "1-4小时",
            "key_levels": {"support": 0, "resistance": 0},
            "risk_level": "medium",
            "reasoning": "数据不足，采用保守预测",
            "action_suggestion": "观望",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "deepseek_fallback"
        }
    
    def _get_default_risk_assessment(self) -> Dict[str, Any]:
        """默认风险评估结果"""
        return {
            "risk_score": 50,
            "risk_level": "medium",
            "main_risks": ["市场波动", "流动性风险"],
            "risk_mitigation": ["分散投资", "设置止损"],
            "position_adjustment": "保持当前仓位",
            "stop_loss_suggestion": "设置10%止损",
            "urgent_action_needed": False,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "deepseek_fallback"
        }
    
    def _get_default_factor_discovery(self) -> Dict[str, Any]:
        """默认因子发现结果"""
        return {
            "discovered_factors": [
                {
                    "name": "价量背离因子",
                    "formula": "correlation(price_change, volume_change, 20)",
                    "description": "价格与成交量变化的负相关性",
                    "expected_ic": "0.03-0.05",
                    "rationale": "经典技术分析理论"
                }
            ],
            "factor_combinations": ["单因子使用"],
            "research_direction": "结合更多市场数据",
            "confidence": 0.6,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "deepseek_fallback"
        }
    
    # 验证方法
    def _validate_sentiment_result(self, result: Dict) -> bool:
        """验证情绪分析结果"""
        required_keys = ["sentiment_score", "confidence", "key_factors", "market_impact", "recommendation"]
        return all(key in result for key in required_keys)
    
    def _validate_prediction_result(self, result: Dict) -> bool:
        """验证市场预测结果"""
        required_keys = ["trend_direction", "confidence", "time_horizon", "key_levels", "risk_level"]
        return all(key in result for key in required_keys)
    
    def _validate_risk_result(self, result: Dict) -> bool:
        """验证风险评估结果"""
        required_keys = ["risk_score", "risk_level", "main_risks", "urgent_action_needed"]
        return all(key in result for key in required_keys)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 简单的健康检查请求
            test_messages = [
                {"role": "system", "content": "你是一个测试助手。"},
                {"role": "user", "content": "请回复'健康检查正常'"}
            ]
            
            start_time = time.time()
            response = await self._make_request_with_retry(
                test_messages, max_tokens=50, temperature=0, 
                max_retries=1, use_cache=False
            )
            response_time = time.time() - start_time
            
            is_healthy = response is not None and "健康" in response
            
            health_status = {
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time": response_time,
                "api_available": is_healthy,
                "stats": self.stats.copy(),
                "cache_size": len(self._cache),
                "active_requests": len(self._request_times),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if not is_healthy:
                health_status["error"] = "健康检查失败"
                logger.warning("DeepSeek健康检查失败")
            else:
                logger.debug("DeepSeek健康检查正常")
            
            return health_status
            
        except Exception as e:
            logger.error(f"DeepSeek健康检查异常: {e}")
            return {
                "status": "error",
                "error": str(e),
                "api_available": False,
                "stats": self.stats.copy(),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats.update({
            "cache_size": len(self._cache),
            "active_requests": len(self._request_times),
            "cache_hit_rate": (
                stats["cached_responses"] / max(stats["total_requests"], 1) * 100
            ),
            "success_rate": (
                stats["successful_requests"] / max(stats["total_requests"], 1) * 100
            )
        })
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        cache_size = len(self._cache)
        self._cache.clear()
        self._cache_expiry.clear()
        logger.info(f"DeepSeek缓存已清理，释放{cache_size}个缓存项")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited": 0,
            "avg_response_time": 0.0,
            "last_request_time": None,
            "health_status": "healthy"
        }
        logger.info("DeepSeek统计信息已重置")
    
    async def close(self):
        """关闭客户端"""
        try:
            await self.client.aclose()
            logger.info("DeepSeek客户端已关闭")
        except Exception as e:
            logger.error(f"关闭DeepSeek客户端时出错: {e}")

class DeepSeekClientManager:
    """DeepSeek客户端管理器"""
    
    def __init__(self):
        self._client = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> DeepSeekClient:
        """获取客户端实例"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = DeepSeekClient()
                    logger.info("DeepSeek客户端实例已创建")
        return self._client
    
    async def close(self):
        """关闭客户端管理器"""
        if self._client:
            await self._client.close()
            self._client = None

# 全局DeepSeek客户端管理器
_deepseek_manager = DeepSeekClientManager()

async def get_deepseek_client() -> DeepSeekClient:
    """获取DeepSeek客户端"""
    return await _deepseek_manager.get_client()

# 向后兼容
deepseek_client = DeepSeekClient()