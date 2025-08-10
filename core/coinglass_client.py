"""
Coinglass API客户端
提供市场情绪、资金费率、持仓数据、ETF流向等数据获取功能
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class CoinglassClient:
    """Coinglass API客户端"""
    
    def __init__(self, api_key: str = None, options: Dict = None):
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY", "")
        self.base_url = "https://open-api-v4.coinglass.com"
        
        # 配置选项
        options = options or {}
        self.requests_per_second = options.get("requests_per_second", 10)
        self.retry_attempts = options.get("retry_attempts", 3)
        self.retry_delay = options.get("retry_delay", 1000)
        self.timeout = options.get("timeout", 30)
        
        # 请求队列和限流
        self.request_queue = deque()
        self.is_processing_queue = False
        self.last_request_time = 0
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    async def make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """发送API请求"""
        return await self._queue_request(endpoint, params or {})
    
    async def _queue_request(self, endpoint: str, params: Dict) -> Dict:
        """将请求加入队列"""
        future = asyncio.Future()
        
        self.request_queue.append({
            "endpoint": endpoint,
            "params": params,
            "future": future,
            "attempts": 0,
            "created_at": time.time()
        })
        
        if not self.is_processing_queue:
            asyncio.create_task(self._process_queue())
        
        return await future
    
    async def _process_queue(self):
        """处理请求队列"""
        if self.is_processing_queue:
            return
            
        self.is_processing_queue = True
        
        try:
            while self.request_queue:
                request = self.request_queue.popleft()
                await self._execute_request(request)
                
                # 限流延迟
                await self._rate_limit_delay()
                
        finally:
            self.is_processing_queue = False
    
    async def _execute_request(self, request: Dict):
        """执行单个请求"""
        endpoint = request["endpoint"]
        params = request["params"]
        future = request["future"]
        
        try:
            self.total_requests += 1
            
            # 构建请求头
            headers = {
                "accept": "application/json",
                "User-Agent": "AI-Quant-Trading-System/1.0"
            }
            
            if self.api_key:
                headers["CG-API-KEY"] = self.api_key
            
            # 发送HTTP请求
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                start_time = time.time()
                
                async with session.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers
                ) as response:
                    duration = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # 检查API响应码
                        if data.get("code") == "0":
                            self.successful_requests += 1
                            logger.debug(f"Coinglass API成功: {endpoint} ({duration:.2f}s)")
                            future.set_result(data)
                        else:
                            # API返回错误码
                            error_msg = data.get("msg", "Unknown API error")
                            error = CoinglassAPIError(f"API Error: {error_msg}", data.get("code"))
                            
                            if self._should_retry(request, error):
                                await self._retry_request(request)
                            else:
                                self.failed_requests += 1
                                logger.error(f"Coinglass API错误: {endpoint} - {error_msg}")
                                future.set_exception(error)
                    else:
                        # HTTP状态码错误
                        error_text = await response.text()
                        error = CoinglassHTTPError(f"HTTP {response.status}: {error_text}")
                        
                        if self._should_retry(request, error):
                            await self._retry_request(request)
                        else:
                            self.failed_requests += 1
                            logger.error(f"Coinglass HTTP错误: {endpoint} - {response.status}")
                            future.set_exception(error)
                            
        except Exception as e:
            if self._should_retry(request, e):
                await self._retry_request(request)
            else:
                self.failed_requests += 1
                logger.error(f"Coinglass请求异常: {endpoint} - {str(e)}")
                future.set_exception(e)
    
    def _should_retry(self, request: Dict, error: Exception) -> bool:
        """判断是否应该重试"""
        if request["attempts"] >= self.retry_attempts:
            return False
        
        # API限流错误应该重试
        if isinstance(error, CoinglassAPIError) and "rate limit" in str(error).lower():
            return True
        
        # 网络错误应该重试
        if isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError)):
            return True
        
        # HTTP 5xx错误应该重试
        if isinstance(error, CoinglassHTTPError) and "5" in str(error)[:1]:
            return True
        
        return False
    
    async def _retry_request(self, request: Dict):
        """重试请求"""
        request["attempts"] += 1
        retry_delay = self.retry_delay * request["attempts"] / 1000  # 转换为秒
        
        logger.warning(f"Coinglass请求重试: {request['endpoint']} (尝试 {request['attempts']}/{self.retry_attempts})")
        
        await asyncio.sleep(retry_delay)
        self.request_queue.appendleft(request)  # 放回队列前端优先处理
    
    async def _rate_limit_delay(self):
        """执行限流延迟"""
        now = time.time()
        time_since_last = now - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
    
    # ===================
    # API方法
    # ===================
    
    async def get_supported_coins(self) -> Dict:
        """获取支持的币种列表"""
        return await self.make_request("/api/futures/supported-coins")
    
    async def get_supported_exchange_pairs(self) -> Dict:
        """获取支持的交易所和交易对"""
        return await self.make_request("/api/futures/supported-exchange-pairs")
    
    async def get_contract_markets(self, symbol: str = "all") -> Dict:
        """获取合约市场数据"""
        try:
            return await self.make_request("/api/futures/coins-markets", {"symbol": symbol})
        except CoinglassAPIError as e:
            if "upgrade plan" in str(e).lower():
                logger.warning("合约市场数据需要升级API计划")
                return {"data": [], "message": "需要升级API计划"}
            raise
    
    async def get_pair_markets(self, symbol: str) -> Dict:
        """获取交易对市场数据"""
        if not symbol:
            raise ValueError("symbol参数是必需的")
        return await self.make_request("/api/futures/pairs-markets", {"symbol": symbol})
    
    async def get_fear_greed_index(self) -> Dict:
        """获取恐惧贪婪指数"""
        try:
            return await self.make_request("/api/index/fear-greed-index")
        except CoinglassAPIError as e:
            logger.warning(f"恐惧贪婪指数API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_fear_greed_history(self, limit: int = 100) -> Dict:
        """获取恐惧贪婪指数历史数据"""
        try:
            return await self.make_request("/api/index/fear-greed-history", {"limit": limit})
        except CoinglassAPIError as e:
            logger.warning(f"恐惧贪婪历史数据API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_funding_rates(self, symbol: str = "BTC") -> Dict:
        """获取资金费率数据"""
        try:
            return await self.make_request("/api/futures/funding-rates", {"symbol": symbol})
        except CoinglassAPIError as e:
            logger.warning(f"资金费率API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_funding_rates_history(self, symbol: str = "BTC", limit: int = 100) -> Dict:
        """获取资金费率历史数据"""
        try:
            return await self.make_request("/api/futures/funding-rates-history", {
                "symbol": symbol,
                "limit": limit
            })
        except CoinglassAPIError as e:
            logger.warning(f"资金费率历史API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_open_interest(self, symbol: str = "BTC") -> Dict:
        """获取持仓数据"""
        try:
            return await self.make_request("/api/futures/open-interest", {"symbol": symbol})
        except CoinglassAPIError as e:
            logger.warning(f"持仓数据API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_open_interest_history(self, symbol: str = "BTC", limit: int = 100) -> Dict:
        """获取持仓历史数据"""
        try:
            return await self.make_request("/api/futures/open-interest-history", {
                "symbol": symbol,
                "limit": limit
            })
        except CoinglassAPIError as e:
            logger.warning(f"持仓历史API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_liquidation_data(self, symbol: str = "BTC", period: str = "24h") -> Dict:
        """获取爆仓数据"""
        try:
            return await self.make_request("/api/futures/liquidation", {
                "symbol": symbol,
                "period": period
            })
        except CoinglassAPIError as e:
            logger.warning(f"爆仓数据API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_btc_etf_netflow(self) -> Dict:
        """获取BTC ETF净流入数据"""
        try:
            return await self.make_request("/api/etf/btc-netflow")
        except CoinglassAPIError as e:
            logger.warning(f"BTC ETF数据API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_eth_etf_netflow(self) -> Dict:
        """获取ETH ETF净流入数据"""
        try:
            return await self.make_request("/api/etf/eth-netflow")
        except CoinglassAPIError as e:
            logger.warning(f"ETH ETF数据API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_ahr999_indicator(self) -> Dict:
        """获取AHR999指标"""
        try:
            return await self.make_request("/api/indicator/ahr999")
        except CoinglassAPIError as e:
            logger.warning(f"AHR999指标API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    async def get_rainbow_chart(self) -> Dict:
        """获取彩虹图指标"""
        try:
            return await self.make_request("/api/indicator/rainbow-chart")
        except CoinglassAPIError as e:
            logger.warning(f"彩虹图指标API暂不可用: {e}")
            return {"data": [], "message": "API端点暂不可用"}
    
    # ===================
    # 工具方法
    # ===================
    
    async def test_connection(self) -> Dict:
        """测试API连接"""
        try:
            result = await self.get_supported_coins()
            coin_count = len(result.get("data", [])) if result.get("data") else 0
            
            return {
                "success": True,
                "message": "连接成功",
                "coin_count": coin_count,
                "api_key_configured": bool(self.api_key)
            }
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "error": e.__class__.__name__
            }
    
    def get_status(self) -> Dict:
        """获取客户端状态"""
        success_rate = 0
        if self.total_requests > 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
        
        return {
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "requests_per_second": self.requests_per_second,
            "queue_length": len(self.request_queue),
            "is_processing": self.is_processing_queue,
            "statistics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": f"{success_rate:.2f}%"
            }
        }
    
    async def health_check(self) -> Dict:
        """健康检查"""
        try:
            # 测试基础连接
            connection_test = await self.test_connection()
            
            # 测试各个端点的可用性
            endpoints_status = {}
            
            test_endpoints = [
                ("supported_coins", self.get_supported_coins),
                ("fear_greed_index", self.get_fear_greed_index),
                ("funding_rates", lambda: self.get_funding_rates("BTC")),
                ("open_interest", lambda: self.get_open_interest("BTC"))
            ]
            
            for name, func in test_endpoints:
                try:
                    await asyncio.wait_for(func(), timeout=10)
                    endpoints_status[name] = "available"
                except Exception as e:
                    endpoints_status[name] = f"unavailable: {str(e)[:50]}"
            
            return {
                "status": "healthy" if connection_test["success"] else "unhealthy",
                "connection": connection_test,
                "endpoints": endpoints_status,
                "client_stats": self.get_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

class CoinglassAPIError(Exception):
    """Coinglass API错误"""
    
    def __init__(self, message: str, code: str = None):
        super().__init__(message)
        self.code = code

class CoinglassHTTPError(Exception):
    """Coinglass HTTP错误"""
    pass

# 全局Coinglass客户端实例
coinglass_client = CoinglassClient()