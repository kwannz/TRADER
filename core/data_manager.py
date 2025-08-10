"""
数据管理器 - 统一数据访问和存储
支持MongoDB时序数据、Redis缓存和实时数据流
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import redis.asyncio as aioredis
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

# 导入优化器和缓存管理器
try:
    from .database_optimizer import create_database_optimizer
    from .enhanced_cache_manager import create_enhanced_cache_manager, EnhancedCacheManager
except ImportError:
    create_database_optimizer = None
    create_enhanced_cache_manager = None
    EnhancedCacheManager = None

# 简化设置管理
class Settings:
    MONGODB_URL = "mongodb://localhost:27017"
    MONGODB_DB_NAME = "quanttrader"
    REDIS_URL = "redis://localhost:6379"
    
settings = Settings()

class MongoTimeSeriesManager:
    """MongoDB时序数据管理器"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.kline_collection = self.db.kline_data
        self.tick_collection = self.db.tick_data
        
    async def ensure_time_series_collections(self):
        """确保时序集合存在并配置正确"""
        try:
            # 创建K线数据时序集合
            try:
                await self.db.create_collection(
                    "kline_data",
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "symbol",
                        "granularity": "seconds"
                    }
                )
                logger.info("创建K线时序集合成功")
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"K线集合创建警告: {e}")
            
            # 创建Tick数据时序集合  
            try:
                await self.db.create_collection(
                    "tick_data", 
                    timeseries={
                        "timeField": "timestamp",
                        "metaField": "symbol", 
                        "granularity": "seconds"
                    }
                )
                logger.info("创建Tick时序集合成功")
            except Exception as e:
                if "already exists" not in str(e):
                    logger.warning(f"Tick集合创建警告: {e}")
                    
            # 创建索引
            await self.kline_collection.create_index([("symbol", 1), ("timestamp", -1)])
            await self.tick_collection.create_index([("symbol", 1), ("timestamp", -1)])
            
        except Exception as e:
            logger.error(f"时序集合初始化失败: {e}")
    
    async def insert_kline_data(self, symbol: str, timeframe: str, data: List[Dict]):
        """插入K线数据"""
        try:
            if not data:
                return
                
            # 处理数据格式
            processed_data = []
            for item in data:
                processed_item = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.fromtimestamp(item["timestamp"] / 1000),
                    "open": float(item["open"]),
                    "high": float(item["high"]), 
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item["volume"]),
                    "created_at": datetime.utcnow()
                }
                processed_data.append(processed_item)
            
            # 批量插入
            result = await self.kline_collection.insert_many(processed_data, ordered=False)
            logger.debug(f"插入K线数据: {symbol} {timeframe} - {len(result.inserted_ids)}条")
            
        except Exception as e:
            logger.error(f"插入K线数据失败 {symbol}: {e}")
    
    async def get_kline_data(self, symbol: str, timeframe: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 1000) -> pd.DataFrame:
        """获取K线数据"""
        try:
            # 构建查询条件
            query = {"symbol": symbol, "timeframe": timeframe}
            
            if start_time or end_time:
                time_filter = {}
                if start_time:
                    time_filter["$gte"] = start_time
                if end_time:
                    time_filter["$lte"] = end_time
                query["timestamp"] = time_filter
            
            # 查询数据
            cursor = self.kline_collection.find(query).sort("timestamp", -1).limit(limit)
            data = await cursor.to_list(length=None)
            
            # 转换为DataFrame
            if data:
                df = pd.DataFrame(data)
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取K线数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    async def insert_tick_data(self, symbol: str, data: Dict):
        """插入Tick数据"""
        try:
            processed_data = {
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(data["timestamp"] / 1000),
                "price": float(data["price"]),
                "volume": float(data["volume"]),
                "side": data.get("side", ""),
                "created_at": datetime.utcnow()
            }
            
            await self.tick_collection.insert_one(processed_data)
            
        except Exception as e:
            logger.error(f"插入Tick数据失败 {symbol}: {e}")

class LegacyRedisCache:
    """Legacy Redis缓存管理器 - 作为fallback使用"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        
    async def set(self, key: str, value: Any, ttl: int = 300):
        """设置缓存"""
        try:
            await self.redis.set(key, str(value), ex=ttl)
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            result = await self.redis.get(key)
            return result.decode() if result else None
        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {e}")
            return None
    
    async def delete(self, key: str):
        """删除缓存"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
    
    async def set_market_data(self, symbol: str, data: Dict, expire: int = 300):
        """设置市场数据缓存"""
        await self.set(f"market:{symbol}", data, expire)
    
    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """获取市场数据缓存"""
        try:
            result = await self.get(f"market:{symbol}")
            return eval(result) if result else None
        except:
            return None
    
    async def set_ai_analysis(self, key: str, analysis: Dict, expire: int = 1800):
        """设置AI分析结果缓存"""
        await self.set(f"ai_analysis:{key}", analysis, expire)
    
    async def get_ai_analysis(self, key: str) -> Optional[Dict]:
        """获取AI分析结果缓存"""
        try:
            result = await self.get(f"ai_analysis:{key}")
            return eval(result) if result else None
        except:
            return None
    
    async def set_strategy_state(self, strategy_id: str, state: Dict):
        """设置策略状态缓存"""
        await self.set(f"strategy_state:{strategy_id}", state, 3600)
    
    async def get_strategy_state(self, strategy_id: str) -> Optional[Dict]:
        """获取策略状态缓存"""
        try:
            result = await self.get(f"strategy_state:{strategy_id}")
            return eval(result) if result else None
        except:
            return None

class DataManager:
    """统一数据管理器"""
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.time_series_manager: Optional[MongoTimeSeriesManager] = None
        self.cache_manager = None  # 使用增强缓存管理器
        self.db_optimizer = None
        self._initialized = False
    
    async def initialize(self):
        """初始化数据管理器"""
        try:
            # 初始化MongoDB
            await self._init_mongodb()
            
            # 初始化Redis
            await self._init_redis()
            
            # 初始化管理器
            self.time_series_manager = MongoTimeSeriesManager(self.db)
            
            # 使用增强缓存管理器
            if create_enhanced_cache_manager:
                try:
                    self.cache_manager = await create_enhanced_cache_manager(self.redis_client)
                    logger.info("✅ 增强缓存管理器已启用")
                except Exception as e:
                    logger.warning(f"⚠️ 增强缓存管理器启用失败: {e}, 回退到基础缓存")
                    self.cache_manager = LegacyRedisCache(self.redis_client)
            else:
                self.cache_manager = LegacyRedisCache(self.redis_client)
                logger.info("⚠️ 使用基础缓存管理器")
            
            # 初始化数据库优化器
            if create_database_optimizer:
                self.db_optimizer = create_database_optimizer(self.db)
            
            # 确保时序集合存在
            await self.time_series_manager.ensure_time_series_collections()
            
            # 创建优化索引
            if self.db_optimizer:
                try:
                    await self.db_optimizer.create_indexes()
                    logger.info("✅ 数据库索引优化完成")
                except Exception as e:
                    logger.warning(f"⚠️ 索引创建失败: {e}")
            
            # 启动缓存预热
            if hasattr(self.cache_manager, 'preload_cache'):
                try:
                    await self.cache_manager.preload_cache()
                    logger.info("✅ 缓存预热完成")
                except Exception as e:
                    logger.warning(f"⚠️ 缓存预热失败: {e}")
            
            # 启动缓存监控
            if hasattr(self.cache_manager, 'start_monitoring'):
                try:
                    await self.cache_manager.start_monitoring()
                    logger.info("✅ 缓存监控已启动")
                except Exception as e:
                    logger.warning(f"⚠️ 缓存监控启动失败: {e}")
            
            self._initialized = True
            logger.info("数据管理器初始化完成")
            
        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
            raise
    
    async def _init_mongodb(self):
        """初始化MongoDB连接"""
        try:
            self.mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=50,
                minPoolSize=10
            )
            
            # 测试连接
            await self.mongo_client.admin.command('ping')
            
            # 获取数据库
            self.db = self.mongo_client[settings.MONGODB_DB_NAME]
            
            logger.info("MongoDB连接成功")
            
        except Exception as e:
            logger.error(f"MongoDB连接失败: {e}")
            raise
    
    async def _init_redis(self):
        """初始化Redis连接"""
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                max_connections=50,
                decode_responses=False
            )
            
            # 测试连接
            await self.redis_client.ping()
            
            logger.info("Redis连接成功")
            
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("数据库连接已关闭")
    
    # 策略数据管理
    async def save_strategy(self, strategy_data: Dict) -> str:
        """保存策略"""
        try:
            result = await self.db.strategies.insert_one({
                **strategy_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"策略保存成功: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"保存策略失败: {e}")
            raise
    
    async def get_strategies(self, status: Optional[str] = None) -> List[Dict]:
        """获取策略列表"""
        try:
            query = {}
            if status:
                query["status"] = status
                
            cursor = self.db.strategies.find(query).sort("created_at", -1)
            strategies = await cursor.to_list(length=None)
            
            # 转换ObjectId为字符串
            for strategy in strategies:
                strategy["_id"] = str(strategy["_id"])
                
            return strategies
        except Exception as e:
            logger.error(f"获取策略失败: {e}")
            return []
    
    async def update_strategy(self, strategy_id: str, update_data: Dict):
        """更新策略"""
        try:
            from bson import ObjectId
            await self.db.strategies.update_one(
                {"_id": ObjectId(strategy_id)},
                {
                    "$set": {
                        **update_data,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            logger.info(f"策略更新成功: {strategy_id}")
        except Exception as e:
            logger.error(f"更新策略失败: {e}")
            raise
    
    # 交易记录管理
    async def save_trade(self, trade_data: Dict):
        """保存交易记录"""
        try:
            await self.db.trades.insert_one({
                **trade_data,
                "created_at": datetime.utcnow()
            })
            logger.debug(f"交易记录保存: {trade_data.get('symbol', 'Unknown')}")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
    
    async def get_trades(self, symbol: Optional[str] = None, 
                        strategy_id: Optional[str] = None,
                        limit: int = 100) -> List[Dict]:
        """获取交易记录"""
        try:
            query = {}
            if symbol:
                query["symbol"] = symbol
            if strategy_id:
                query["strategy_id"] = strategy_id
                
            cursor = self.db.trades.find(query).sort("created_at", -1).limit(limit)
            trades = await cursor.to_list(length=None)
            
            for trade in trades:
                trade["_id"] = str(trade["_id"])
                
            return trades
        except Exception as e:
            logger.error(f"获取交易记录失败: {e}")
            return []
    
    # 因子数据管理
    async def save_factor(self, factor_data: Dict) -> str:
        """保存因子"""
        try:
            result = await self.db.factors.insert_one({
                **factor_data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"因子保存成功: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"保存因子失败: {e}")
            raise
    
    async def get_factors(self, status: Optional[str] = None) -> List[Dict]:
        """获取因子列表"""
        try:
            query = {}
            if status:
                query["status"] = status
                
            cursor = self.db.factors.find(query).sort("ic_mean", -1)
            factors = await cursor.to_list(length=None)
            
            for factor in factors:
                factor["_id"] = str(factor["_id"])
                
            return factors
        except Exception as e:
            logger.error(f"获取因子失败: {e}")
            return []
    
    # 新闻数据管理
    async def save_news(self, news_data: Dict):
        """保存新闻数据"""
        try:
            await self.db.news.insert_one({
                **news_data,
                "created_at": datetime.utcnow()
            })
            logger.debug(f"新闻保存: {news_data.get('title', 'Unknown')[:50]}")
        except Exception as e:
            logger.error(f"保存新闻失败: {e}")
    
    async def get_recent_news(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """获取最近新闻"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            query = {"created_at": {"$gte": start_time}}
            
            cursor = self.db.news.find(query).sort("created_at", -1).limit(limit)
            news = await cursor.to_list(length=None)
            
            for item in news:
                item["_id"] = str(item["_id"])
                
            return news
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return []
    
    # 健康检查
    async def health_check(self) -> Dict[str, bool]:
        """检查数据库健康状态"""
        status = {
            "mongodb": False,
            "redis": False,
            "initialized": self._initialized
        }
        
        try:
            # 检查MongoDB
            await self.mongo_client.admin.command('ping')
            status["mongodb"] = True
        except:
            pass
            
        try:
            # 检查Redis
            await self.redis_client.ping()
            status["redis"] = True
        except:
            pass
            
        return status

# 全局数据管理器实例
data_manager = DataManager()