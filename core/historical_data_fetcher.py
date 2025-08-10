"""
历史数据获取器
支持从OKX和Binance获取历史K线数据用于AI模型训练
"""

import asyncio
import aiohttp
import hmac
import hashlib
import base64
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .data_manager import data_manager

logger = get_logger()

class OKXHistoricalDataFetcher:
    """OKX历史数据获取器"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        self.api_key = os.getenv("OKX_API_KEY", "")
        self.secret_key = os.getenv("OKX_SECRET_KEY", "")
        self.passphrase = os.getenv("OKX_PASSPHRASE", "")
        self.sandbox = os.getenv("OKX_SANDBOX", "true").lower() == "true"
        
        if self.sandbox:
            self.base_url = "https://www.okx.com"  # OKX testnet uses same endpoint
    
    def _create_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """创建OKX签名"""
        try:
            message = timestamp + method + request_path + body
            signature = base64.b64encode(
                hmac.new(
                    self.secret_key.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            return signature
        except Exception as e:
            logger.error(f"创建OKX签名失败: {e}")
            return ""
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """获取请求头"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 如果有API密钥，添加签名头
        if self.api_key and self.secret_key and self.passphrase:
            timestamp = str(time.time())
            signature = self._create_signature(timestamp, method, request_path, body)
            
            headers.update({
                'OK-ACCESS-KEY': self.api_key,
                'OK-ACCESS-SIGN': signature,
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self.passphrase
            })
        
        return headers
    
    async def fetch_kline_data(self, 
                              symbol: str,
                              timeframe: str = "1m",
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              limit: int = 300) -> List[Dict]:
        """获取K线数据"""
        try:
            # 构建请求参数
            params = {
                "instId": symbol,
                "bar": timeframe,
                "limit": str(limit)
            }
            
            if start_time:
                params["after"] = str(int(start_time.timestamp() * 1000))
            if end_time:
                params["before"] = str(int(end_time.timestamp() * 1000))
            
            request_path = "/api/v5/market/candles"
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            full_path = f"{request_path}?{query_string}"
            
            headers = self._get_headers("GET", full_path)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}{full_path}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0" and data.get("data"):
                            return self._parse_okx_klines(data["data"])
                        else:
                            logger.error(f"OKX API错误: {data}")
                            return []
                    else:
                        logger.error(f"OKX API请求失败: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"获取OKX历史数据失败: {e}")
            return []
    
    def _parse_okx_klines(self, raw_data: List[List]) -> List[Dict]:
        """解析OKX K线数据"""
        parsed_data = []
        for item in raw_data:
            try:
                parsed_data.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "volume_ccy": float(item[6]) if len(item) > 6 else 0,
                    "volume_ccy_quote": float(item[7]) if len(item) > 7 else 0,
                    "confirm": int(item[8]) if len(item) > 8 else 1
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"解析OKX K线数据项失败: {e}")
                continue
        return parsed_data

class BinanceHistoricalDataFetcher:
    """Binance历史数据获取器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY", "")
        self.sandbox = os.getenv("BINANCE_SANDBOX", "true").lower() == "true"
        
        if self.sandbox:
            self.base_url = "https://testnet.binance.vision"
    
    def _create_signature(self, query_string: str) -> str:
        """创建Binance签名"""
        try:
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            logger.error(f"创建Binance签名失败: {e}")
            return ""
    
    async def fetch_kline_data(self,
                              symbol: str,
                              timeframe: str = "1m",
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              limit: int = 1000) -> List[Dict]:
        """获取K线数据"""
        try:
            # 构建请求参数
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            # 如果需要签名（私有端点）
            if self.api_key and self.secret_key:
                timestamp = int(time.time() * 1000)
                query_string += f"&timestamp={timestamp}"
                signature = self._create_signature(query_string)
                query_string += f"&signature={signature}"
            
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            
            request_path = "/api/v3/klines"
            url = f"{self.base_url}{request_path}?{query_string}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_binance_klines(data)
                    else:
                        error_text = await response.text()
                        logger.error(f"Binance API请求失败: {response.status}, {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"获取Binance历史数据失败: {e}")
            return []
    
    def _parse_binance_klines(self, raw_data: List[List]) -> List[Dict]:
        """解析Binance K线数据"""
        parsed_data = []
        for item in raw_data:
            try:
                parsed_data.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": int(item[6]),
                    "quote_asset_volume": float(item[7]),
                    "number_of_trades": int(item[8]),
                    "taker_buy_base_asset_volume": float(item[9]),
                    "taker_buy_quote_asset_volume": float(item[10])
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"解析Binance K线数据项失败: {e}")
                continue
        return parsed_data

class HistoricalDataManager:
    """历史数据管理器"""
    
    def __init__(self):
        self.okx_fetcher = OKXHistoricalDataFetcher()
        self.binance_fetcher = BinanceHistoricalDataFetcher()
        
    async def fetch_training_data(self, 
                                 symbols: List[str] = None,
                                 exchanges: List[str] = None,
                                 timeframes: List[str] = None,
                                 days_back: int = 30) -> Dict[str, Dict]:
        """获取AI训练数据"""
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT"]
        if exchanges is None:
            exchanges = ["okx", "binance"]
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        training_data = {}
        
        for symbol in symbols:
            training_data[symbol] = {}
            
            for exchange in exchanges:
                training_data[symbol][exchange] = {}
                
                for timeframe in timeframes:
                    logger.info(f"获取{exchange}:{symbol}:{timeframe}历史数据...")
                    
                    try:
                        if exchange.lower() == "okx":
                            # OKX符号格式转换
                            okx_symbol = symbol.replace("/", "-")
                            data = await self.okx_fetcher.fetch_kline_data(
                                okx_symbol, timeframe, start_time, end_time
                            )
                        elif exchange.lower() == "binance":
                            # Binance符号格式转换  
                            binance_symbol = symbol.replace("/", "")
                            data = await self.binance_fetcher.fetch_kline_data(
                                binance_symbol, timeframe, start_time, end_time
                            )
                        else:
                            logger.warning(f"不支持的交易所: {exchange}")
                            continue
                        
                        if data:
                            # 保存到数据库
                            if hasattr(data_manager, '_initialized') and data_manager._initialized:
                                await data_manager.time_series_manager.insert_kline_data(
                                    f"{exchange.upper()}:{symbol}", timeframe, data
                                )
                            
                            training_data[symbol][exchange][timeframe] = data
                            logger.info(f"获取到{len(data)}条{exchange}:{symbol}:{timeframe}数据")
                        else:
                            logger.warning(f"未获取到{exchange}:{symbol}:{timeframe}数据")
                        
                        # 避免频率限制
                        await asyncio.sleep(0.2)
                        
                    except Exception as e:
                        logger.error(f"获取{exchange}:{symbol}:{timeframe}数据失败: {e}")
                        continue
        
        return training_data
    
    async def create_training_dataset(self, 
                                    symbols: List[str] = None,
                                    target_file: str = "data/training_dataset.csv") -> str:
        """创建AI训练数据集"""
        try:
            # 获取历史数据
            training_data = await self.fetch_training_data(symbols)
            
            # 合并所有数据
            combined_data = []
            
            for symbol in training_data:
                for exchange in training_data[symbol]:
                    for timeframe in training_data[symbol][exchange]:
                        data_list = training_data[symbol][exchange][timeframe]
                        
                        for item in data_list:
                            combined_data.append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "timeframe": timeframe,
                                "timestamp": item["timestamp"],
                                "datetime": datetime.fromtimestamp(item["timestamp"] / 1000),
                                "open": item["open"],
                                "high": item["high"],
                                "low": item["low"],
                                "close": item["close"],
                                "volume": item["volume"]
                            })
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(combined_data)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            df.to_csv(target_file, index=False)
            logger.info(f"训练数据集已保存到: {target_file}, 共{len(df)}条记录")
            
            return target_file
            
        except Exception as e:
            logger.error(f"创建训练数据集失败: {e}")
            raise
    
    async def get_latest_data(self, 
                            symbol: str,
                            exchange: str = "binance",
                            timeframe: str = "1m",
                            limit: int = 100) -> List[Dict]:
        """获取最新数据"""
        try:
            if exchange.lower() == "okx":
                okx_symbol = symbol.replace("/", "-")
                return await self.okx_fetcher.fetch_kline_data(okx_symbol, timeframe, limit=limit)
            elif exchange.lower() == "binance":
                binance_symbol = symbol.replace("/", "")
                return await self.binance_fetcher.fetch_kline_data(binance_symbol, timeframe, limit=limit)
            else:
                logger.error(f"不支持的交易所: {exchange}")
                return []
                
        except Exception as e:
            logger.error(f"获取最新数据失败: {e}")
            return []

# 全局历史数据管理器实例
historical_data_manager = HistoricalDataManager()

# 为了向后兼容，提供HistoricalDataFetcher别名
HistoricalDataFetcher = HistoricalDataManager