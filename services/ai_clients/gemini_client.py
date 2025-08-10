"""
Google Gemini API客户端
负责策略生成、多模态分析和智能问答
增强版本：包含完整的错误处理、重试机制和速率限制
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import httpx
from loguru import logger
import time
import hashlib

from config.settings import settings

class GeminiError(Exception):
    """Gemini API异常"""
    pass

class GeminiRateLimitError(GeminiError):
    """Gemini速率限制异常"""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Gemini API速率限制，请等待{retry_after}秒后重试")

class GeminiClient:
    """Google Gemini API客户端 - 增强版"""
    
    def __init__(self):
        self.config = settings.get_ai_config("gemini")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config["timeout"],
                write=10.0,
                pool=30.0
            ),
            params={"key": self.config["api_key"]},
            headers={"User-Agent": "AI-Trader/1.0"},
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )
        
        # 速率限制和缓存
        self._request_times = []
        self._cache = {}
        self._cache_expiry = {}
        self._max_requests_per_minute = 15  # Gemini限制更严格
        self._cache_duration = 600  # 10分钟缓存（策略可以缓存更久）
        
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
                logger.debug(f"使用Gemini缓存响应: {cache_key[:8]}...")
                return self._cache[cache_key]
            else:
                # 清理过期缓存
                del self._cache[cache_key]
                del self._cache_expiry[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str, duration: Optional[int] = None):
        """缓存响应"""
        cache_duration = duration or self._cache_duration
        self._cache[cache_key] = response
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(seconds=cache_duration)
        
        # 限制缓存大小
        if len(self._cache) > 50:  # Gemini缓存更小，策略代码较大
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
            raise GeminiRateLimitError(60)
        
        self._request_times.append(now)
    
    async def _make_request_with_retry(self, messages: List[Dict], 
                                     temperature: float = 0.7,
                                     max_tokens: Optional[int] = None,
                                     max_retries: int = 3,
                                     use_cache: bool = True,
                                     cache_duration: Optional[int] = None) -> Optional[str]:
        """发送请求到Gemini API (带重试和缓存)"""
        start_time = time.time()
        
        try:
            # 检查缓存
            if use_cache:
                cache_key = self._get_cache_key(messages, temperature=temperature, max_tokens=max_tokens)
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
                    
                    # 构建Gemini API格式的内容
                    contents = []
                    for message in messages:
                        if message["role"] == "user":
                            contents.append({
                                "parts": [{"text": message["content"]}]
                            })
                    
                    payload = {
                        "contents": contents,
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens or self.config["max_tokens"],
                            "candidateCount": 1
                        }
                    }
                    
                    logger.debug(f"Gemini API请求 (尝试 {attempt + 1}/{max_retries})")
                    response = await self.client.post(
                        f"/models/{self.config['model']}:generateContent",
                        json=payload
                    )
                    
                    # 处理响应状态
                    if response.status_code == 429:  # Too Many Requests
                        retry_after = int(response.headers.get("Retry-After", 60))
                        if attempt < max_retries - 1:
                            logger.warning(f"Gemini API速率限制，等待{retry_after}秒后重试...")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise GeminiRateLimitError(retry_after)
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # 验证响应格式
                    if "candidates" not in result or not result["candidates"]:
                        if "error" in result:
                            error_msg = result["error"].get("message", "Unknown error")
                            raise GeminiError(f"API错误: {error_msg}")
                        else:
                            raise GeminiError("API响应格式错误：缺少candidates字段")
                    
                    candidate = result["candidates"][0]
                    if "content" not in candidate or "parts" not in candidate["content"]:
                        raise GeminiError("API响应格式错误：内容结构不完整")
                    
                    content = candidate["content"]["parts"][0].get("text", "").strip()
                    if not content:
                        raise GeminiError("API返回空内容")
                    
                    # 检查内容过滤
                    if "finishReason" in candidate and candidate["finishReason"] == "SAFETY":
                        logger.warning("Gemini内容被安全过滤器拦截")
                        raise GeminiError("内容被安全过滤器拦截")
                    
                    # 缓存响应
                    if use_cache:
                        self._cache_response(cache_key, content, cache_duration)
                    
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
                    
                    logger.debug(f"Gemini API请求成功，响应时间: {response_time:.2f}秒")
                    return content
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in [400, 403]:
                        # 认证或请求格式错误，不重试
                        logger.error(f"Gemini API错误: {e.response.status_code}")
                        try:
                            error_detail = e.response.json()
                            error_msg = error_detail.get("error", {}).get("message", "Unknown error")
                        except:
                            error_msg = e.response.text
                        self.stats["health_status"] = "auth_error" if e.response.status_code == 403 else "client_error"
                        raise GeminiError(f"请求错误 ({e.response.status_code}): {error_msg}")
                    elif e.response.status_code == 429:
                        # 速率限制，已在上面处理
                        continue
                    elif e.response.status_code >= 500:
                        # 服务器错误，可以重试
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 指数退避
                            logger.warning(f"Gemini服务器错误 {e.response.status_code}，{wait_time}秒后重试...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"Gemini服务器错误: {e.response.status_code}")
                            self.stats["health_status"] = "server_error"
                            raise GeminiError(f"服务器错误: {e.response.status_code}")
                    else:
                        logger.error(f"Gemini API HTTP错误: {e.response.status_code} - {e.response.text}")
                        raise GeminiError(f"HTTP错误: {e.response.status_code}")
                
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Gemini网络请求失败，{wait_time}秒后重试: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Gemini网络请求失败: {e}")
                        self.stats["health_status"] = "network_error"
                        raise GeminiError(f"网络错误: {e}")
                        
                except GeminiRateLimitError:
                    # 重新抛出速率限制异常
                    raise
                    
                except GeminiError:
                    # 重新抛出Gemini特定错误
                    raise
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Gemini未知错误，{wait_time}秒后重试: {e}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Gemini请求失败: {e}")
                        self.stats["health_status"] = "unknown_error"
                        raise GeminiError(f"未知错误: {e}")
            
            # 如果所有重试都失败
            self.stats["failed_requests"] += 1
            return None
            
        except (GeminiRateLimitError, GeminiError):
            self.stats["failed_requests"] += 1
            raise
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Gemini请求异常: {e}")
            return None
    
    async def _make_request(self, messages: List[Dict], 
                          temperature: float = 0.7,
                          max_tokens: Optional[int] = None) -> Optional[str]:
        """发送请求到Gemini API (兼容性包装器)"""
        try:
            return await self._make_request_with_retry(
                messages, temperature, max_tokens, 
                max_retries=3, use_cache=True
            )
        except (GeminiRateLimitError, GeminiError):
            return None
        except Exception as e:
            logger.error(f"Gemini API请求失败: {e}")
            return None
    
    async def generate_strategy(self, strategy_requirements: Dict) -> Dict[str, Any]:
        """生成交易策略代码"""
        try:
            requirements_text = f"""
            策略需求:
            - 策略类型: {strategy_requirements.get('strategy_type', 'N/A')}
            - 交易对: {strategy_requirements.get('symbols', 'N/A')}
            - 资金限制: {strategy_requirements.get('max_capital', 'N/A')} USDT
            - 风险偏好: {strategy_requirements.get('risk_level', 'medium')}
            - 时间框架: {strategy_requirements.get('timeframe', '1h')}
            - 特殊要求: {strategy_requirements.get('special_requirements', '无')}
            """
            
            messages = [
                {
                    "role": "user",
                    "content": f"""请根据以下需求生成一个完整的Python量化交易策略代码：

{requirements_text}

请返回JSON格式的结果：
{{
    "strategy_name": "策略名称",
    "strategy_type": "策略类型",
    "code": "完整的Python策略代码",
    "description": "策略描述",
    "parameters": {{"参数名": "参数说明"}},
    "risk_management": "风控措施说明",
    "expected_performance": "预期表现",
    "usage_instructions": "使用说明"
}}

策略代码要求：
1. 使用标准的量化交易库（pandas, numpy, ta等）
2. 包含完整的入场和出场逻辑
3. 实现风险管理机制
4. 代码要能直接运行
5. 包含详细注释"""
                }
            ]
            
            response = await self._make_request(messages, temperature=0.5)
            if not response:
                return self._get_default_strategy()
            
            # 提取JSON部分
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 尝试直接提取JSON
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx]
                    else:
                        raise ValueError("未找到JSON格式数据")
                
                result = json.loads(json_str)
                
                if self._validate_strategy_result(result):
                    result["timestamp"] = datetime.utcnow().isoformat()
                    result["source"] = "gemini"
                    result["generated_by"] = "AI"
                    logger.info(f"策略生成完成: {result['strategy_name']}")
                    return result
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Gemini策略生成JSON解析失败: {e}")
                # 尝试从文本中提取信息
                return self._extract_strategy_from_text(response)
                
            return self._get_default_strategy()
            
        except Exception as e:
            logger.error(f"Gemini策略生成失败: {e}")
            return self._get_default_strategy()
    
    async def optimize_strategy(self, strategy_code: str, 
                               performance_data: Dict) -> Dict[str, Any]:
        """优化策略代码"""
        try:
            performance_summary = f"""
            当前策略表现:
            - 收益率: {performance_data.get('return_rate', 'N/A')}%
            - 胜率: {performance_data.get('win_rate', 'N/A')}%
            - 最大回撤: {performance_data.get('max_drawdown', 'N/A')}%
            - 夏普比率: {performance_data.get('sharpe_ratio', 'N/A')}
            - 交易次数: {performance_data.get('trade_count', 'N/A')}
            - 主要问题: {performance_data.get('issues', '无')}
            """
            
            messages = [
                {
                    "role": "user",
                    "content": f"""请分析并优化以下交易策略：

原策略代码:
```python
{strategy_code}
```

策略表现数据:
{performance_summary}

请返回JSON格式的优化建议：
{{
    "optimized_code": "优化后的完整代码",
    "optimization_summary": "优化摘要",
    "key_changes": ["主要修改点列表"],
    "expected_improvements": "预期改进效果",
    "risk_assessment": "风险评估",
    "testing_recommendations": "测试建议"
}}

优化重点：
1. 提高胜率和收益率
2. 减少最大回撤
3. 优化参数设置
4. 加强风险控制
5. 提升代码效率"""
                }
            ]
            
            response = await self._make_request(messages, temperature=0.3)
            if not response:
                return self._get_default_optimization()
                
            try:
                # 提取JSON
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    result["timestamp"] = datetime.utcnow().isoformat()
                    result["source"] = "gemini"
                    logger.info("策略优化完成")
                    return result
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Gemini策略优化JSON解析失败: {e}")
                
            return self._get_default_optimization()
            
        except Exception as e:
            logger.error(f"Gemini策略优化失败: {e}")
            return self._get_default_optimization()
    
    async def analyze_market_multimodal(self, chart_data: str, 
                                      market_context: Dict) -> Dict[str, Any]:
        """多模态市场分析"""
        try:
            context_text = f"""
            市场背景信息:
            - 当前价格: {market_context.get('current_price', 'N/A')}
            - 24h变化: {market_context.get('price_change_24h', 'N/A')}%
            - 成交量: {market_context.get('volume_24h', 'N/A')}
            - 技术指标: {market_context.get('technical_indicators', {})}
            - 新闻情绪: {market_context.get('news_sentiment', 'N/A')}
            """
            
            messages = [
                {
                    "role": "user",
                    "content": f"""请分析以下市场图表和数据：

图表数据（ASCII格式）:
{chart_data}

{context_text}

请返回JSON格式的分析结果：
{{
    "technical_analysis": "技术分析结论",
    "pattern_recognition": "识别的图表模式",
    "support_resistance": {{"support": "支撑位", "resistance": "阻力位"}},
    "trend_analysis": "趋势分析",
    "volume_analysis": "成交量分析", 
    "trading_signals": ["交易信号列表"],
    "confidence_level": "分析置信度",
    "risk_warning": "风险提示"
}}"""
                }
            ]
            
            response = await self._make_request(messages, temperature=0.4)
            if not response:
                return self._get_default_multimodal_analysis()
                
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    result["timestamp"] = datetime.utcnow().isoformat()
                    result["source"] = "gemini"
                    logger.info("多模态分析完成")
                    return result
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Gemini多模态分析JSON解析失败: {e}")
                
            return self._get_default_multimodal_analysis()
            
        except Exception as e:
            logger.error(f"Gemini多模态分析失败: {e}")
            return self._get_default_multimodal_analysis()
    
    async def chat_assistant(self, user_message: str, 
                           context_data: Dict) -> Dict[str, Any]:
        """智能助手对话"""
        try:
            system_context = f"""
            当前交易系统状态:
            - 账户余额: {context_data.get('account_balance', 'N/A')} USDT
            - 运行策略: {context_data.get('active_strategies', 'N/A')}个
            - 当日PnL: {context_data.get('daily_pnl', 'N/A')} USDT
            - 市场状况: {context_data.get('market_status', 'N/A')}
            """
            
            messages = [
                {
                    "role": "user",
                    "content": f"""你是专业的量化交易助手。当前系统状态：

{system_context}

用户问题: {user_message}

请提供专业的回答和建议。如果涉及交易决策，请给出具体的操作建议和风险提示。

返回JSON格式：
{{
    "response": "回答内容",
    "suggestions": ["具体建议列表"],
    "risk_warning": "风险提示",
    "follow_up_questions": ["推荐的后续问题"],
    "confidence": "回答置信度"
}}"""
                }
            ]
            
            response = await self._make_request(messages, temperature=0.6)
            if not response:
                return self._get_default_chat_response(user_message)
                
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    
                    result["timestamp"] = datetime.utcnow().isoformat()
                    result["source"] = "gemini"
                    result["user_message"] = user_message
                    logger.info("AI助手回复完成")
                    return result
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Gemini助手回复JSON解析失败: {e}")
                
            return self._get_default_chat_response(user_message)
            
        except Exception as e:
            logger.error(f"Gemini助手对话失败: {e}")
            return self._get_default_chat_response(user_message)
    
    # 默认返回值和辅助方法
    def _get_default_strategy(self) -> Dict[str, Any]:
        """默认策略生成结果"""
        return {
            "strategy_name": "简单移动平均策略",
            "strategy_type": "趋势跟随",
            "code": """
# 简单移动平均线交易策略
import pandas as pd
import numpy as np

class SimpleMAStrategy:
    def __init__(self, short_window=10, long_window=30):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        data['signal'] = 0
        data['signal'][self.short_window:] = np.where(
            data['short_ma'][self.short_window:] > data['long_ma'][self.short_window:], 1, 0
        )
        data['positions'] = data['signal'].diff()
        
        return data
            """,
            "description": "基于短期和长期移动平均线交叉的趋势跟随策略",
            "parameters": {
                "short_window": "短期移动平均周期",
                "long_window": "长期移动平均周期"
            },
            "risk_management": "固定止损和止盈，最大仓位限制",
            "expected_performance": "中等风险，稳定收益",
            "usage_instructions": "适用于趋势明显的市场",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gemini_fallback"
        }
    
    def _get_default_optimization(self) -> Dict[str, Any]:
        """默认优化建议"""
        return {
            "optimized_code": "# 代码优化建议请参考原代码",
            "optimization_summary": "建议增加风险控制和参数优化",
            "key_changes": ["添加止损机制", "优化移动平均参数", "增加仓位管理"],
            "expected_improvements": "预期提升10-15%收益率",
            "risk_assessment": "风险等级：中等",
            "testing_recommendations": "建议在历史数据上充分回测",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gemini_fallback"
        }
    
    def _get_default_multimodal_analysis(self) -> Dict[str, Any]:
        """默认多模态分析结果"""
        return {
            "technical_analysis": "技术分析数据不足，建议观望",
            "pattern_recognition": "未识别明显模式",
            "support_resistance": {"support": "待确定", "resistance": "待确定"},
            "trend_analysis": "横盘整理",
            "volume_analysis": "成交量正常",
            "trading_signals": ["观望"],
            "confidence_level": "低",
            "risk_warning": "市场数据不足，谨慎操作",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gemini_fallback"
        }
    
    def _get_default_chat_response(self, user_message: str) -> Dict[str, Any]:
        """默认聊天回复"""
        return {
            "response": "抱歉，我现在无法处理您的请求。请稍后再试或重新表述您的问题。",
            "suggestions": ["检查网络连接", "重新提问", "查看帮助文档"],
            "risk_warning": "在做任何交易决策前请仔细评估风险",
            "follow_up_questions": ["需要什么帮助？", "想了解哪个功能？"],
            "confidence": "低",
            "user_message": user_message,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gemini_fallback"
        }
    
    def _validate_strategy_result(self, result: Dict) -> bool:
        """验证策略生成结果"""
        required_keys = ["strategy_name", "strategy_type", "code", "description"]
        return all(key in result for key in required_keys)
    
    def _extract_strategy_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取策略信息"""
        # 简单的文本提取逻辑
        code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else "# 代码提取失败"
        
        return {
            "strategy_name": "AI生成策略",
            "strategy_type": "自定义",
            "code": code,
            "description": "基于AI分析生成的交易策略",
            "parameters": {},
            "risk_management": "请自行添加风险控制",
            "expected_performance": "待评估",
            "usage_instructions": "请仔细回测后使用",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "gemini_text_extract"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 简单的健康检查请求
            test_messages = [
                {"role": "user", "content": "请回复'健康检查正常'"}
            ]
            
            start_time = time.time()
            response = await self._make_request_with_retry(
                test_messages, temperature=0, max_tokens=50,
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
                logger.warning("Gemini健康检查失败")
            else:
                logger.debug("Gemini健康检查正常")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Gemini健康检查异常: {e}")
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
        logger.info(f"Gemini缓存已清理，释放{cache_size}个缓存项")
    
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
        logger.info("Gemini统计信息已重置")
    
    async def close(self):
        """关闭客户端"""
        try:
            await self.client.aclose()
            logger.info("Gemini客户端已关闭")
        except Exception as e:
            logger.error(f"关闭Gemini客户端时出错: {e}")

class GeminiClientManager:
    """Gemini客户端管理器"""
    
    def __init__(self):
        self._client = None
        self._lock = asyncio.Lock()
    
    async def get_client(self) -> GeminiClient:
        """获取客户端实例"""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = GeminiClient()
                    logger.info("Gemini客户端实例已创建")
        return self._client
    
    async def close(self):
        """关闭客户端管理器"""
        if self._client:
            await self._client.close()
            self._client = None

# 全局Gemini客户端管理器
_gemini_manager = GeminiClientManager()

async def get_gemini_client() -> GeminiClient:
    """获取Gemini客户端"""
    return await _gemini_manager.get_client()

# 向后兼容
gemini_client = GeminiClient()