"""
数据库优化工具
MongoDB索引、查询优化和分片策略
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pymongo
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class MongoDBOptimizer:
    """MongoDB优化器"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
        self.optimization_results = {}
        
    async def create_indexes(self):
        """创建优化索引"""
        logger.info("🚀 开始创建MongoDB优化索引...")
        
        try:
            # 1. 时序数据集合索引
            await self._create_timeseries_indexes()
            
            # 2. 市场数据索引
            await self._create_market_data_indexes()
            
            # 3. Coinglass数据索引
            await self._create_coinglass_indexes()
            
            # 4. 系统性能索引
            await self._create_system_indexes()
            
            logger.info("✅ MongoDB索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ 创建索引失败: {e}")
            raise
    
    async def _create_timeseries_indexes(self):
        """创建时序数据索引"""
        logger.info("📊 创建时序数据索引...")
        
        # tick_data时序集合
        tick_collection = self.db["tick_data"]
        
        try:
            # 复合索引：符号 + 时间戳（降序）
            await tick_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_timestamp_desc", background=True)
            
            # 复合索引：交易所 + 符号 + 时间戳
            await tick_collection.create_index([
                ("exchange", pymongo.ASCENDING),
                ("symbol", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="exchange_symbol_timestamp", background=True)
            
            # 价格范围查询索引
            await tick_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("price", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_price_timestamp", background=True)
            
            # 数据质量索引
            await tick_collection.create_index([
                ("data_quality_level", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="quality_timestamp", background=True)
            
            logger.info("✅ tick_data索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ tick_data索引创建失败: {e}")
        
        # kline_data时序集合
        kline_collection = self.db["kline_data"]
        
        try:
            # 复合索引：符号 + 时间周期 + 时间戳
            await kline_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("timeframe", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_timeframe_timestamp", background=True)
            
            # OHLC范围查询索引
            await kline_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("close", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_close_timestamp", background=True)
            
            # 成交量排序索引
            await kline_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("volume", pymongo.DESCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_volume_desc", background=True)
            
            logger.info("✅ kline_data索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ kline_data索引创建失败: {e}")
    
    async def _create_market_data_indexes(self):
        """创建市场数据索引"""
        logger.info("📈 创建市场数据索引...")
        
        # 创建实时市场数据视图（如果不存在）
        try:
            await self.db.create_collection("market_data_view", viewOn="tick_data", pipeline=[
                {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=24)}}},
                {"$sort": {"timestamp": -1}},
                {"$group": {
                    "_id": {"symbol": "$symbol", "exchange": "$exchange"},
                    "latest_price": {"$first": "$price"},
                    "latest_timestamp": {"$first": "$timestamp"},
                    "volume_24h": {"$sum": "$volume_24h"},
                    "high_24h": {"$max": "$high_24h"},
                    "low_24h": {"$min": "$low_24h"},
                    "data_quality_avg": {"$avg": "$data_quality_score"}
                }}
            ])
            logger.info("✅ 创建market_data_view视图")
        except Exception as e:
            logger.debug(f"market_data_view可能已存在: {e}")
        
        # AI分析结果集合
        ai_collection = self.db["ai_analysis"]
        
        try:
            # AI分析时间索引
            await ai_collection.create_index([
                ("analysis_type", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="analysis_type_timestamp", background=True)
            
            # 信号强度索引
            await ai_collection.create_index([
                ("signal_strength", pymongo.ASCENDING),
                ("confidence", pymongo.DESCENDING)
            ], name="signal_confidence", background=True)
            
            logger.info("✅ ai_analysis索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ ai_analysis索引创建失败: {e}")
    
    async def _create_coinglass_indexes(self):
        """创建Coinglass数据索引"""
        logger.info("🧠 创建Coinglass数据索引...")
        
        # 恐贪指数集合
        fear_greed_collection = self.db["fear_greed_index"]
        
        try:
            # 时间索引（降序，用于获取最新数据）
            await fear_greed_collection.create_index([
                ("timestamp", pymongo.DESCENDING)
            ], name="timestamp_desc", background=True)
            
            # 日期索引（用于按日期查询）
            await fear_greed_collection.create_index([
                ("date", pymongo.ASCENDING)
            ], name="date_asc", background=True)
            
            # 值范围索引
            await fear_greed_collection.create_index([
                ("value", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="value_timestamp", background=True)
            
            # 分类索引
            await fear_greed_collection.create_index([
                ("classification", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="classification_timestamp", background=True)
            
            logger.info("✅ fear_greed_index索引创建完成")
            
        except Exception as e:
            logger.error(f"❌ fear_greed_index索引创建失败: {e}")
        
        # 资金费率集合（如果存在）
        funding_collection = self.db["funding_rates"]
        
        try:
            await funding_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_exchange_timestamp", background=True)
            
            logger.info("✅ funding_rates索引创建完成")
            
        except Exception as e:
            logger.debug(f"funding_rates集合可能不存在: {e}")
        
        # 持仓数据集合（如果存在）
        oi_collection = self.db["open_interest"]
        
        try:
            await oi_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="symbol_timestamp_desc", background=True)
            
            logger.info("✅ open_interest索引创建完成")
            
        except Exception as e:
            logger.debug(f"open_interest集合可能不存在: {e}")
    
    async def _create_system_indexes(self):
        """创建系统性能索引"""
        logger.info("⚙️ 创建系统性能索引...")
        
        # 系统日志集合（如果存在）
        logs_collection = self.db["system_logs"]
        
        try:
            # 时间 + 级别索引
            await logs_collection.create_index([
                ("level", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="level_timestamp", background=True)
            
            # TTL索引（自动删除30天前的日志）
            await logs_collection.create_index([
                ("timestamp", pymongo.ASCENDING)
            ], name="log_ttl", background=True, expireAfterSeconds=30*24*3600)
            
            logger.info("✅ system_logs索引创建完成")
            
        except Exception as e:
            logger.debug(f"system_logs集合可能不存在: {e}")
        
        # 性能指标集合
        metrics_collection = self.db["performance_metrics"]
        
        try:
            # 指标类型 + 时间索引
            await metrics_collection.create_index([
                ("metric_type", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ], name="metric_timestamp", background=True)
            
            # TTL索引（自动删除7天前的性能指标）
            await metrics_collection.create_index([
                ("timestamp", pymongo.ASCENDING)
            ], name="metrics_ttl", background=True, expireAfterSeconds=7*24*3600)
            
            logger.info("✅ performance_metrics索引创建完成")
            
        except Exception as e:
            logger.debug(f"performance_metrics集合可能不存在: {e}")
    
    async def optimize_queries(self):
        """优化查询性能"""
        logger.info("🔍 开始优化查询性能...")
        
        try:
            # 1. 分析慢查询
            await self._analyze_slow_queries()
            
            # 2. 优化聚合管道
            await self._optimize_aggregation_pipelines()
            
            # 3. 配置数据库参数
            await self._configure_database_settings()
            
            logger.info("✅ 查询优化完成")
            
        except Exception as e:
            logger.error(f"❌ 查询优化失败: {e}")
    
    async def _analyze_slow_queries(self):
        """分析慢查询"""
        try:
            # 获取慢查询分析
            db_stats = await self.db.command("dbStats")
            logger.info(f"📊 数据库统计: {db_stats.get('collections')}个集合, {db_stats.get('dataSize', 0)/1024/1024:.2f}MB数据")
            
            # 检查索引使用情况
            collections = await self.db.list_collection_names()
            for collection_name in collections:
                if not collection_name.startswith("system."):
                    try:
                        collection = self.db[collection_name]
                        index_stats = await collection.index_stats().to_list(None)
                        
                        unused_indexes = [idx for idx in index_stats if idx.get("accesses", {}).get("ops", 0) == 0]
                        if unused_indexes:
                            logger.info(f"⚠️ {collection_name}集合有{len(unused_indexes)}个未使用的索引")
                    except Exception as e:
                        logger.debug(f"无法获取{collection_name}的索引统计: {e}")
                        
        except Exception as e:
            logger.error(f"慢查询分析失败: {e}")
    
    async def _optimize_aggregation_pipelines(self):
        """优化聚合管道"""
        logger.info("⚡ 优化聚合查询管道...")
        
        # 创建预计算视图用于常用查询
        try:
            # 24小时价格统计视图
            await self.db.create_collection("price_stats_24h", viewOn="tick_data", pipeline=[
                {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=24)}}},
                {"$group": {
                    "_id": "$symbol",
                    "avg_price": {"$avg": "$price"},
                    "max_price": {"$max": "$price"},
                    "min_price": {"$min": "$price"},
                    "total_volume": {"$sum": "$volume_24h"},
                    "count": {"$sum": 1},
                    "last_update": {"$max": "$timestamp"}
                }},
                {"$sort": {"total_volume": -1}}
            ])
            logger.info("✅ 创建price_stats_24h视图")
        except Exception as e:
            logger.debug(f"price_stats_24h视图可能已存在: {e}")
        
        # 数据质量统计视图
        try:
            await self.db.create_collection("quality_stats", viewOn="tick_data", pipeline=[
                {"$match": {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)}}},
                {"$group": {
                    "_id": {
                        "exchange": "$exchange",
                        "quality_level": "$data_quality_level"
                    },
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$data_quality_score"},
                    "min_score": {"$min": "$data_quality_score"},
                    "max_score": {"$max": "$data_quality_score"}
                }},
                {"$sort": {"_id.exchange": 1, "_id.quality_level": 1}}
            ])
            logger.info("✅ 创建quality_stats视图")
        except Exception as e:
            logger.debug(f"quality_stats视图可能已存在: {e}")
    
    async def _configure_database_settings(self):
        """配置数据库设置"""
        logger.info("⚙️ 优化数据库配置...")
        
        try:
            # 设置分析器收集查询统计（仅在开发环境）
            if os.getenv("ENVIRONMENT", "production") == "development":
                await self.db.command("profile", 2, slowOpThresholdMs=100)
                logger.info("✅ 启用查询分析器")
            
            # 优化连接池设置已在连接字符串中配置
            logger.info("✅ 数据库配置优化完成")
            
        except Exception as e:
            logger.error(f"数据库配置失败: {e}")
    
    async def create_sharding_strategy(self):
        """创建分片策略（企业级功能）"""
        logger.info("🗂️ 设计分片策略...")
        
        # 这里定义分片策略，实际分片需要MongoDB集群
        sharding_recommendations = {
            "tick_data": {
                "shard_key": {"symbol": 1, "timestamp": 1},
                "reason": "按交易对和时间分片，支持高并发写入",
                "zones": [
                    {"range": {"symbol": "BTC/USDT"}, "zone": "crypto_major"},
                    {"range": {"symbol": "ETH/USDT"}, "zone": "crypto_major"},
                ]
            },
            "kline_data": {
                "shard_key": {"symbol": 1, "timeframe": 1, "timestamp": 1},
                "reason": "按交易对、时间周期和时间分片",
                "zones": [
                    {"range": {"timeframe": "1m"}, "zone": "high_freq"},
                    {"range": {"timeframe": "1d"}, "zone": "low_freq"},
                ]
            },
            "fear_greed_index": {
                "shard_key": {"date": 1},
                "reason": "按日期分片，历史数据查询优化",
                "zones": []
            }
        }
        
        logger.info("📋 分片策略设计完成:")
        for collection, strategy in sharding_recommendations.items():
            logger.info(f"  {collection}: {strategy['reason']}")
        
        return sharding_recommendations
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        logger.info("📊 生成数据库优化报告...")
        
        try:
            report = {
                "optimization_time": datetime.utcnow().isoformat(),
                "database_info": {},
                "collections": {},
                "indexes": {},
                "performance": {}
            }
            
            # 数据库基本信息
            db_stats = await self.db.command("dbStats")
            report["database_info"] = {
                "collections": db_stats.get("collections", 0),
                "data_size_mb": db_stats.get("dataSize", 0) / 1024 / 1024,
                "index_size_mb": db_stats.get("indexSize", 0) / 1024 / 1024,
                "total_size_mb": db_stats.get("storageSize", 0) / 1024 / 1024
            }
            
            # 各集合统计
            collections = await self.db.list_collection_names()
            for collection_name in collections:
                if not collection_name.startswith("system."):
                    try:
                        collection = self.db[collection_name]
                        stats = await self.db.command("collStats", collection_name)
                        
                        report["collections"][collection_name] = {
                            "count": stats.get("count", 0),
                            "size_mb": stats.get("size", 0) / 1024 / 1024,
                            "avg_obj_size": stats.get("avgObjSize", 0),
                            "total_index_size_mb": stats.get("totalIndexSize", 0) / 1024 / 1024,
                            "indexes": stats.get("nindexes", 0)
                        }
                        
                        # 索引信息
                        indexes = await collection.list_indexes().to_list(None)
                        report["indexes"][collection_name] = [
                            {
                                "name": idx.get("name"),
                                "key": idx.get("key"),
                                "unique": idx.get("unique", False),
                                "background": idx.get("background", False)
                            }
                            for idx in indexes
                        ]
                        
                    except Exception as e:
                        logger.debug(f"无法获取{collection_name}的统计: {e}")
            
            logger.info("✅ 数据库优化报告生成完成")
            return report
            
        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            return {"error": str(e)}

class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        self.db = database
    
    async def get_optimized_market_data_pipeline(self, symbol: str = None, 
                                                exchange: str = None, 
                                                hours: int = 24) -> List[Dict]:
        """获取优化的市场数据聚合管道"""
        pipeline = []
        
        # 时间范围过滤（使用索引）
        match_stage = {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}}
        
        if symbol:
            match_stage["symbol"] = symbol
        if exchange:
            match_stage["exchange"] = exchange
            
        pipeline.append({"$match": match_stage})
        
        # 排序（使用索引）
        pipeline.append({"$sort": {"timestamp": -1}})
        
        # 聚合统计
        pipeline.extend([
            {"$group": {
                "_id": {"symbol": "$symbol", "exchange": "$exchange"},
                "latest_price": {"$first": "$price"},
                "latest_timestamp": {"$first": "$timestamp"},
                "high": {"$max": "$price"},
                "low": {"$min": "$price"},
                "volume": {"$sum": "$volume_24h"},
                "count": {"$sum": 1},
                "avg_quality": {"$avg": "$data_quality_score"}
            }},
            {"$sort": {"volume": -1}}
        ])
        
        return pipeline
    
    async def get_optimized_candle_pipeline(self, symbol: str, timeframe: str, 
                                           limit: int = 100) -> List[Dict]:
        """获取优化的K线数据管道"""
        pipeline = [
            # 精确匹配（使用复合索引）
            {"$match": {
                "symbol": symbol,
                "timeframe": timeframe
            }},
            # 排序（使用索引）
            {"$sort": {"timestamp": -1}},
            # 限制结果数量
            {"$limit": limit},
            # 只返回需要的字段
            {"$project": {
                "timestamp": 1,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "data_quality_score": 1
            }}
        ]
        
        return pipeline

# 创建全局优化器实例
def create_database_optimizer(database: AsyncIOMotorDatabase) -> MongoDBOptimizer:
    """创建数据库优化器实例"""
    return MongoDBOptimizer(database)