"""
DeepSeek API客户端

集成DeepSeek AI API，提供情绪分析、策略生成等功能
使用Python 3.13优化异步性能
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

from ..utils.logger import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)

@dataclass
class DeepSeekConfig:
    """DeepSeek API配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-reasoner"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class DeepSeekAPIError(Exception):
    """DeepSeek API异常"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class DeepSeekClient:
    """
    DeepSeek API客户端
    
    提供异步API调用，支持聊天完成、流式响应等功能
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.config = DeepSeekConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
        )
        
        # 会话管理
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_created_at: Optional[datetime] = None
        
        # 请求统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "rate_limit_hits": 0,
        }
        
        # 速率限制
        self._request_times: List[float] = []
        self._max_requests_per_minute = 60
        
        logger.info(f"DeepSeek客户端初始化完成，Base URL: {self.config.base_url}")

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self) -> None:
        """确保HTTP会话存在"""
        if (self._session is None or 
            self._session.closed or 
            (self._session_created_at and 
             datetime.utcnow() - self._session_created_at > timedelta(hours=1))):
            
            if self._session and not self._session.closed:
                await self._session.close()
            
            # 使用Python 3.13优化的连接器配置
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                force_close=False,
                # Python 3.13新特性：更好的连接池管理
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DeepSeek-Python-Client/1.0.0",
            }
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                raise_for_status=False,
            )
            
            self._session_created_at = datetime.utcnow()
            logger.debug("HTTP会话已创建")

    async def close(self) -> None:
        """关闭HTTP会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP会话已关闭")

    async def _check_rate_limit(self) -> None:
        """检查速率限制"""
        current_time = time.time()
        
        # 清理超过1分钟的请求记录
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # 检查是否超过速率限制
        if len(self._request_times) >= self._max_requests_per_minute:
            wait_time = 60 - (current_time - self._request_times[0])
            if wait_time > 0:
                logger.warning(f"达到速率限制，等待 {wait_time:.2f} 秒")
                self.stats["rate_limit_hits"] += 1
                await asyncio.sleep(wait_time)
        
        self._request_times.append(current_time)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """发送HTTP请求"""
        await self._ensure_session()
        await self._check_rate_limit()
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                self.stats["total_requests"] += 1
                
                async with self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        # 成功响应
                        response_time = time.time() - start_time
                        self.stats["successful_requests"] += 1
                        self._update_avg_response_time(response_time)
                        
                        # 更新令牌使用统计
                        if "usage" in response_data:
                            usage = response_data["usage"]
                            self.stats["total_tokens_used"] += usage.get("total_tokens", 0)
                            
                            # 估算成本（DeepSeek定价）
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            
                            # DeepSeek价格（每1K token）：输入$0.14，输出$0.28
                            cost = (prompt_tokens * 0.00014) + (completion_tokens * 0.00028)
                            self.stats["total_cost"] += cost
                        
                        logger.debug(f"DeepSeek API调用成功，耗时: {response_time:.2f}s")
                        return response_data
                    
                    elif response.status == 429:
                        # 速率限制
                        retry_after = int(response.headers.get("retry-after", self.config.retry_delay))
                        logger.warning(f"API速率限制，等待 {retry_after} 秒后重试")
                        await asyncio.sleep(retry_after)
                        last_error = DeepSeekAPIError(f"Rate limit exceeded", response.status, response_data)
                        
                    elif response.status >= 500:
                        # 服务器错误，可重试
                        logger.warning(f"服务器错误 {response.status}，第 {attempt + 1} 次重试")
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        last_error = DeepSeekAPIError(f"Server error: {response.status}", response.status, response_data)
                        
                    else:
                        # 客户端错误，不重试
                        self.stats["failed_requests"] += 1
                        error_msg = response_data.get("error", {}).get("message", f"HTTP {response.status}")
                        raise DeepSeekAPIError(error_msg, response.status, response_data)
            
            except aiohttp.ClientError as e:
                last_error = DeepSeekAPIError(f"网络错误: {str(e)}")
                logger.warning(f"网络错误，第 {attempt + 1} 次重试: {e}")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            except asyncio.TimeoutError:
                last_error = DeepSeekAPIError("请求超时")
                logger.warning(f"请求超时，第 {attempt + 1} 次重试")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            except Exception as e:
                last_error = DeepSeekAPIError(f"未知错误: {str(e)}")
                logger.error(f"未知错误: {e}")
                break
        
        # 所有重试都失败
        self.stats["failed_requests"] += 1
        if last_error:
            raise last_error
        else:
            raise DeepSeekAPIError("所有重试都失败")

    def _update_avg_response_time(self, response_time: float) -> None:
        """更新平均响应时间"""
        total_requests = self.stats["successful_requests"]
        if total_requests == 1:
            self.stats["avg_response_time"] = response_time
        else:
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (total_requests - 1) + response_time) / total_requests
            )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-reasoner",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天完成API
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否流式响应
            **kwargs: 其他参数
        
        Returns:
            Dict: API响应
        """
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        logger.debug(f"发送聊天完成请求: {len(messages)} 条消息, 模型: {model}")
        
        return await self._make_request("POST", "/chat/completions", data)

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-reasoner",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式聊天完成API
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大令牌数
            **kwargs: 其他参数
        
        Yields:
            Dict: 流式响应数据
        """
        await self._ensure_session()
        await self._check_rate_limit()
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        url = f"{self.config.base_url}/chat/completions"
        
        try:
            self.stats["total_requests"] += 1
            start_time = time.time()
            
            async with self._session.post(url, json=data) as response:
                if response.status != 200:
                    self.stats["failed_requests"] += 1
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                    raise DeepSeekAPIError(error_msg, response.status, error_data)
                
                # 处理流式响应
                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    chunk_data = chunk.decode('utf-8')
                    buffer += chunk_data
                    
                    # 处理SSE格式
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # 保留不完整的行
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data_json = json.loads(data_str)
                                yield data_json
                            except json.JSONDecodeError:
                                continue
                
                # 更新统计
                response_time = time.time() - start_time
                self.stats["successful_requests"] += 1
                self._update_avg_response_time(response_time)
                
        except Exception as e:
            self.stats["failed_requests"] += 1
            if isinstance(e, DeepSeekAPIError):
                raise
            else:
                raise DeepSeekAPIError(f"流式请求失败: {str(e)}")

    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None,
        model: str = "deepseek-reasoner"
    ) -> Dict[str, Any]:
        """
        文本情绪分析
        
        Args:
            text: 要分析的文本
            context: 上下文信息
            model: 使用的模型
        
        Returns:
            Dict: 情绪分析结果
        """
        system_prompt = """你是一个专业的金融文本情绪分析师。请分析给定文本的情绪倾向，特别关注对金融市场的影响。

请返回JSON格式的结果：
{
    "sentiment": "positive/negative/neutral",
    "score": 情绪得分(-1到1),
    "confidence": 置信度(0到1),
    "keywords": ["关键词列表"],
    "reasoning": "分析原因"
}"""
        
        user_prompt = f"请分析以下文本的市场情绪：\n\n{text}"
        
        if context:
            user_prompt += f"\n\n背景信息：{context}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3
        )
        
        return response

    async def generate_strategy_code(
        self,
        strategy_description: str,
        market_data_info: Dict[str, Any],
        risk_parameters: Dict[str, float],
        model: str = "deepseek-coder"
    ) -> Dict[str, Any]:
        """
        生成交易策略代码
        
        Args:
            strategy_description: 策略描述
            market_data_info: 市场数据信息
            risk_parameters: 风险参数
            model: 使用的模型
        
        Returns:
            Dict: 策略代码生成结果
        """
        system_prompt = """你是一个专业的量化交易策略开发专家。请根据需求生成安全、高效的Python交易策略代码。

代码要求：
1. 使用标准的量化库（pandas, numpy等）
2. 包含完整的风险控制逻辑
3. 代码结构清晰，包含注释
4. 不包含任何恶意或危险操作
5. 返回JSON格式结果

JSON格式：
{
    "strategy_name": "策略名称",
    "strategy_code": "完整Python代码",
    "parameters": {"参数名": 参数值},
    "description": "策略说明",
    "risk_warnings": ["风险提示列表"]
}"""
        
        user_prompt = f"""
请开发交易策略：

策略需求：
{strategy_description}

市场数据：
{json.dumps(market_data_info, ensure_ascii=False, indent=2)}

风险参数：
{json.dumps(risk_parameters, ensure_ascii=False, indent=2)}

请生成完整的策略代码。
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.1  # 代码生成使用低温度
        )
        
        return response

    async def explain_market_data(
        self,
        market_data: Dict[str, Any],
        analysis_type: str = "technical",
        model: str = "deepseek-reasoner"
    ) -> Dict[str, Any]:
        """
        市场数据解读
        
        Args:
            market_data: 市场数据
            analysis_type: 分析类型 (technical/fundamental)
            model: 使用的模型
        
        Returns:
            Dict: 市场数据解读结果
        """
        system_prompt = f"""你是一个专业的{analysis_type}分析师。请分析给定的市场数据，提供专业的市场洞察。

请返回JSON格式的分析结果：
{{
    "summary": "市场概况总结",
    "key_findings": ["主要发现1", "主要发现2", "..."],
    "trend_analysis": "趋势分析",
    "support_resistance": {{"support": 支撑位, "resistance": 阻力位}},
    "trading_suggestions": ["建议1", "建议2", "..."],
    "risk_factors": ["风险因素1", "风险因素2", "..."]
}}"""
        
        user_prompt = f"""
请分析以下市场数据：

{json.dumps(market_data, ensure_ascii=False, indent=2)}

分析重点：
- 价格走势和成交量
- 技术指标信号
- 市场强度和动能
- 潜在的交易机会和风险
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=0.3
        )
        
        return response

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            **self.stats,
            "session_active": self._session is not None and not self._session.closed,
            "session_age_minutes": (
                (datetime.utcnow() - self._session_created_at).total_seconds() / 60
                if self._session_created_at else 0
            ),
            "requests_per_minute": len(self._request_times),
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_requests"])
            ),
            "avg_cost_per_request": (
                self.stats["total_cost"] / max(1, self.stats["successful_requests"])
            ),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "rate_limit_hits": 0,
        }
        self._request_times.clear()
        logger.info("DeepSeek客户端统计信息已重置")

# 便捷的上下文管理器
@asynccontextmanager
async def deepseek_client(api_key: str, base_url: str = "https://api.deepseek.com/v1"):
    """DeepSeek客户端上下文管理器"""
    client = DeepSeekClient(api_key, base_url)
    try:
        yield client
    finally:
        await client.close()