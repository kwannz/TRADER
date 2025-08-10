"""
Coinglass数据收集器
实现市场情绪、资金费率、持仓数据等的定时收集和存储
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .coinglass_client import coinglass_client
from .data_manager import data_manager

logger = get_logger()

class BaseCoinglassCollector:
    """Coinglass数据收集器基类"""
    
    def __init__(self, name: str, collection_name: str):
        self.name = name
        self.collection_name = collection_name
        self.is_running = False
        self.last_run_time = None
        self.error_count = 0
        self.success_count = 0
        self.total_documents = 0
        
    async def collect(self) -> Dict[str, Any]:
        """执行数据收集"""
        if self.is_running:
            raise RuntimeError(f"Collector {self.name} is already running")
        
        start_time = datetime.utcnow()
        self.is_running = True
        
        try:
            logger.info(f"开始收集{self.name}数据...")
            
            # 执行具体的数据收集逻辑
            result = await self._perform_collection()
            
            # 保存数据到数据库
            if result.get("data"):
                save_result = await self._save_data(result["data"])
                result.update(save_result)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.last_run_time = datetime.utcnow()
            self.success_count += 1
            
            final_result = {
                "success": True,
                "collector": self.name,
                "collection": self.collection_name,
                "count": result.get("count", 0),
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat(),
                **result
            }
            
            logger.info(f"{self.name}数据收集成功: {final_result['count']}条记录，耗时{duration:.2f}秒")
            return final_result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.error_count += 1
            
            error_result = {
                "success": False,
                "collector": self.name,
                "collection": self.collection_name,
                "count": 0,
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.error(f"{self.name}数据收集失败: {str(e)}")
            raise
        finally:
            self.is_running = False
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """执行具体的收集逻辑 - 子类实现"""
        raise NotImplementedError("Subclasses must implement _perform_collection")
    
    async def _save_data(self, data: List[Dict]) -> Dict[str, Any]:
        """保存数据到MongoDB"""
        if not data:
            return {"inserted_count": 0}
        
        try:
            # 确保数据库连接
            if not data_manager._initialized:
                await data_manager.initialize()
            
            # 数据预处理
            processed_data = []
            for item in data:
                processed_item = {
                    **item,
                    "collector": self.name,
                    "collected_at": datetime.utcnow(),
                    "source": "coinglass"
                }
                processed_data.append(processed_item)
            
            # 保存到对应集合
            collection = data_manager.db[self.collection_name]
            
            # 使用upsert避免重复数据
            operations = []
            for item in processed_data:
                # 根据时间戳和符号创建唯一性过滤器
                filter_key = self._get_upsert_filter(item)
                operations.append({
                    "updateOne": {
                        "filter": filter_key,
                        "update": {"$set": item},
                        "upsert": True
                    }
                })
            
            if operations:
                result = await collection.bulk_write(operations, ordered=False)
                inserted_count = result.upserted_count + result.modified_count
                self.total_documents += inserted_count
                
                logger.debug(f"{self.name}: 保存了{inserted_count}条记录到{self.collection_name}")
                return {"inserted_count": inserted_count}
            
            return {"inserted_count": 0}
            
        except Exception as e:
            logger.error(f"{self.name}保存数据失败: {e}")
            raise
    
    def _get_upsert_filter(self, item: Dict) -> Dict:
        """获取upsert的过滤条件"""
        # 默认使用时间戳和符号作为唯一性标识
        filter_key = {}
        
        if "timestamp" in item:
            filter_key["timestamp"] = item["timestamp"]
        if "symbol" in item:
            filter_key["symbol"] = item["symbol"]
        if "date" in item:
            filter_key["date"] = item["date"]
            
        return filter_key or {"collected_at": item["collected_at"]}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取收集器统计信息"""
        total_runs = self.success_count + self.error_count
        success_rate = (self.success_count / total_runs * 100) if total_runs > 0 else 0
        
        return {
            "name": self.name,
            "collection": self.collection_name,
            "is_running": self.is_running,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "total_runs": total_runs,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": f"{success_rate:.2f}%",
            "total_documents": self.total_documents
        }

class FearGreedIndexCollector(BaseCoinglassCollector):
    """恐惧贪婪指数收集器"""
    
    def __init__(self):
        super().__init__("FearGreedIndex", "fear_greed_index")
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """收集恐惧贪婪指数数据"""
        try:
            # 获取当前指数
            current_result = await coinglass_client.get_fear_greed_index()
            
            # 获取历史数据
            history_result = await coinglass_client.get_fear_greed_history(limit=30)
            
            collected_data = []
            
            # 处理当前数据
            if current_result.get("data"):
                current_data = {
                    "type": "current",
                    "value": current_result["data"],
                    "timestamp": datetime.utcnow(),
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                }
                collected_data.append(current_data)
            
            # 处理历史数据
            if history_result.get("data") and isinstance(history_result["data"], list):
                for item in history_result["data"]:
                    if isinstance(item, dict):
                        history_data = {
                            "type": "historical",
                            "value": item.get("value", 0),
                            "date": item.get("date", ""),
                            "timestamp": datetime.fromisoformat(item.get("date", "")) if item.get("date") else datetime.utcnow(),
                            "classification": item.get("classification", "")
                        }
                        collected_data.append(history_data)
            
            return {
                "data": collected_data,
                "count": len(collected_data),
                "current_available": bool(current_result.get("data")),
                "history_available": bool(history_result.get("data"))
            }
            
        except Exception as e:
            logger.error(f"收集恐惧贪婪指数数据失败: {e}")
            return {"data": [], "count": 0, "error": str(e)}

class FundingRateCollector(BaseCoinglassCollector):
    """资金费率收集器"""
    
    def __init__(self):
        super().__init__("FundingRate", "funding_rates")
        self.symbols = ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "LINK", "UNI", "AVAX", "ATOM"]
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """收集资金费率数据"""
        try:
            collected_data = []
            
            for symbol in self.symbols:
                try:
                    # 获取当前资金费率
                    current_result = await coinglass_client.get_funding_rates(symbol)
                    
                    if current_result.get("data"):
                        for exchange_data in current_result["data"]:
                            funding_data = {
                                "symbol": symbol,
                                "exchange": exchange_data.get("exchange", ""),
                                "funding_rate": float(exchange_data.get("rate", 0)),
                                "next_funding_time": exchange_data.get("next_funding_time", ""),
                                "timestamp": datetime.utcnow(),
                                "date": datetime.utcnow().strftime("%Y-%m-%d")
                            }
                            collected_data.append(funding_data)
                    
                    # 小延迟避免频率限制
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"收集{symbol}资金费率失败: {e}")
                    continue
            
            return {
                "data": collected_data,
                "count": len(collected_data),
                "symbols_processed": len(self.symbols)
            }
            
        except Exception as e:
            logger.error(f"收集资金费率数据失败: {e}")
            return {"data": [], "count": 0, "error": str(e)}

class OpenInterestCollector(BaseCoinglassCollector):
    """持仓数据收集器"""
    
    def __init__(self):
        super().__init__("OpenInterest", "open_interest")
        self.symbols = ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "LINK", "UNI", "AVAX", "ATOM"]
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """收集持仓数据"""
        try:
            collected_data = []
            
            for symbol in self.symbols:
                try:
                    # 获取持仓数据
                    result = await coinglass_client.get_open_interest(symbol)
                    
                    if result.get("data"):
                        for exchange_data in result["data"]:
                            oi_data = {
                                "symbol": symbol,
                                "exchange": exchange_data.get("exchange", ""),
                                "open_interest": float(exchange_data.get("open_interest", 0)),
                                "open_interest_value": float(exchange_data.get("open_interest_value", 0)),
                                "change_24h": float(exchange_data.get("change_24h", 0)),
                                "change_24h_percent": float(exchange_data.get("change_24h_percent", 0)),
                                "timestamp": datetime.utcnow(),
                                "date": datetime.utcnow().strftime("%Y-%m-%d")
                            }
                            collected_data.append(oi_data)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"收集{symbol}持仓数据失败: {e}")
                    continue
            
            return {
                "data": collected_data,
                "count": len(collected_data),
                "symbols_processed": len(self.symbols)
            }
            
        except Exception as e:
            logger.error(f"收集持仓数据失败: {e}")
            return {"data": [], "count": 0, "error": str(e)}

class LiquidationCollector(BaseCoinglassCollector):
    """爆仓数据收集器"""
    
    def __init__(self):
        super().__init__("Liquidation", "liquidations")
        self.symbols = ["BTC", "ETH", "BNB", "ADA", "SOL"]
        self.periods = ["1h", "4h", "24h"]
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """收集爆仓数据"""
        try:
            collected_data = []
            
            for symbol in self.symbols:
                for period in self.periods:
                    try:
                        result = await coinglass_client.get_liquidation_data(symbol, period)
                        
                        if result.get("data"):
                            liquidation_data = {
                                "symbol": symbol,
                                "period": period,
                                "total_liquidation": float(result["data"].get("total", 0)),
                                "long_liquidation": float(result["data"].get("long", 0)),
                                "short_liquidation": float(result["data"].get("short", 0)),
                                "long_percentage": float(result["data"].get("long_pct", 0)),
                                "short_percentage": float(result["data"].get("short_pct", 0)),
                                "timestamp": datetime.utcnow(),
                                "date": datetime.utcnow().strftime("%Y-%m-%d")
                            }
                            collected_data.append(liquidation_data)
                        
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"收集{symbol}-{period}爆仓数据失败: {e}")
                        continue
            
            return {
                "data": collected_data,
                "count": len(collected_data),
                "symbols_processed": len(self.symbols),
                "periods_processed": len(self.periods)
            }
            
        except Exception as e:
            logger.error(f"收集爆仓数据失败: {e}")
            return {"data": [], "count": 0, "error": str(e)}

class ETFFlowCollector(BaseCoinglassCollector):
    """ETF资金流向收集器"""
    
    def __init__(self):
        super().__init__("ETFFlow", "etf_flows")
    
    async def _perform_collection(self) -> Dict[str, Any]:
        """收集ETF资金流向数据"""
        try:
            collected_data = []
            
            # 收集BTC ETF数据
            try:
                btc_result = await coinglass_client.get_btc_etf_netflow()
                if btc_result.get("data"):
                    btc_data = {
                        "asset": "BTC",
                        "total_netflow": float(btc_result["data"].get("total", 0)),
                        "daily_netflow": float(btc_result["data"].get("daily", 0)),
                        "weekly_netflow": float(btc_result["data"].get("weekly", 0)),
                        "monthly_netflow": float(btc_result["data"].get("monthly", 0)),
                        "timestamp": datetime.utcnow(),
                        "date": datetime.utcnow().strftime("%Y-%m-%d")
                    }
                    collected_data.append(btc_data)
            except Exception as e:
                logger.warning(f"收集BTC ETF数据失败: {e}")
            
            # 收集ETH ETF数据
            try:
                eth_result = await coinglass_client.get_eth_etf_netflow()
                if eth_result.get("data"):
                    eth_data = {
                        "asset": "ETH",
                        "total_netflow": float(eth_result["data"].get("total", 0)),
                        "daily_netflow": float(eth_result["data"].get("daily", 0)),
                        "weekly_netflow": float(eth_result["data"].get("weekly", 0)),
                        "monthly_netflow": float(eth_result["data"].get("monthly", 0)),
                        "timestamp": datetime.utcnow(),
                        "date": datetime.utcnow().strftime("%Y-%m-%d")
                    }
                    collected_data.append(eth_data)
            except Exception as e:
                logger.warning(f"收集ETH ETF数据失败: {e}")
            
            return {
                "data": collected_data,
                "count": len(collected_data),
                "btc_available": any(item["asset"] == "BTC" for item in collected_data),
                "eth_available": any(item["asset"] == "ETH" for item in collected_data)
            }
            
        except Exception as e:
            logger.error(f"收集ETF数据失败: {e}")
            return {"data": [], "count": 0, "error": str(e)}

class CoinglassCollectorManager:
    """Coinglass收集器管理器"""
    
    def __init__(self):
        self.collectors = {
            "fear_greed": FearGreedIndexCollector(),
            "funding_rate": FundingRateCollector(),
            "open_interest": OpenInterestCollector(),
            "liquidation": LiquidationCollector(),
            "etf_flow": ETFFlowCollector()
        }
        
        self.is_running = False
        self.collection_intervals = {
            "fear_greed": 300,      # 5分钟
            "funding_rate": 900,    # 15分钟
            "open_interest": 1800,  # 30分钟
            "liquidation": 300,     # 5分钟
            "etf_flow": 3600        # 1小时
        }
        
        self.collection_tasks = {}
    
    async def start_collection(self, collectors: List[str] = None):
        """启动数据收集"""
        if self.is_running:
            logger.warning("Coinglass收集器已在运行")
            return
        
        self.is_running = True
        collectors_to_start = collectors or list(self.collectors.keys())
        
        logger.info(f"启动Coinglass数据收集: {collectors_to_start}")
        
        for collector_name in collectors_to_start:
            if collector_name in self.collectors:
                interval = self.collection_intervals[collector_name]
                task = asyncio.create_task(
                    self._collection_loop(collector_name, interval)
                )
                self.collection_tasks[collector_name] = task
                logger.info(f"启动{collector_name}收集器，间隔{interval}秒")
    
    async def stop_collection(self):
        """停止数据收集"""
        self.is_running = False
        
        for name, task in self.collection_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"停止{name}收集器")
        
        # 等待所有任务完成
        if self.collection_tasks:
            await asyncio.gather(*self.collection_tasks.values(), return_exceptions=True)
        
        self.collection_tasks.clear()
        logger.info("Coinglass数据收集已停止")
    
    async def _collection_loop(self, collector_name: str, interval: int):
        """数据收集循环"""
        collector = self.collectors[collector_name]
        
        while self.is_running:
            try:
                # 执行数据收集
                result = await collector.collect()
                logger.debug(f"{collector_name}收集完成: {result}")
                
                # 等待下次收集
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info(f"{collector_name}收集器被取消")
                break
            except Exception as e:
                logger.error(f"{collector_name}收集器异常: {e}")
                # 出错后等待较短时间再重试
                await asyncio.sleep(min(interval, 60))
    
    async def collect_once(self, collector_name: str = None) -> Dict[str, Any]:
        """执行一次数据收集"""
        if collector_name:
            if collector_name not in self.collectors:
                raise ValueError(f"未知的收集器: {collector_name}")
            
            collector = self.collectors[collector_name]
            return await collector.collect()
        else:
            # 收集所有数据
            results = {}
            for name, collector in self.collectors.items():
                try:
                    result = await collector.collect()
                    results[name] = result
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
            
            return results
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        collector_stats = {
            name: collector.get_stats()
            for name, collector in self.collectors.items()
        }
        
        return {
            "is_running": self.is_running,
            "active_tasks": len(self.collection_tasks),
            "collectors": collector_stats,
            "intervals": self.collection_intervals
        }

# 全局收集器管理器实例
coinglass_collector_manager = CoinglassCollectorManager()

# 为了向后兼容，提供CoinglassCollector别名
CoinglassCollector = CoinglassCollectorManager