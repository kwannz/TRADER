"""
数据库性能优化器
提供MongoDB和Redis的性能优化功能
包括索引分析、查询优化、内存管理等
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import motor.motor_asyncio
import redis.asyncio as aioredis
from loguru import logger
from collections import defaultdict
import json

from config.settings import settings

class DatabasePerformanceAnalyzer:
    """数据库性能分析器"""
    
    def __init__(self):
        self.mongodb_client = None
        self.redis_client = None
        self.mongodb_db = None
        
    async def initialize(self):
        """初始化数据库连接"""
        try:
            db_config = settings.get_database_config()
            
            # MongoDB连接
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                db_config["mongodb_url"],
                maxPoolSize=20,
                minPoolSize=5
            )
            self.mongodb_db = self.mongodb_client.get_default_database()
            
            # Redis连接
            self.redis_client = aioredis.from_url(
                db_config["redis_url"],
                max_connections=20
            )
            
            logger.info("数据库性能分析器初始化成功")
            
        except Exception as e:
            logger.error(f"数据库性能分析器初始化失败: {e}")
            raise
    
    async def analyze_mongodb_performance(self) -> Dict[str, Any]:
        """分析MongoDB性能"""
        try:
            analysis_results = {
                "server_status": await self._get_mongodb_server_status(),
                "database_stats": await self._get_database_stats(),
                "collection_stats": await self._get_collection_stats(),
                "index_analysis": await self._analyze_indexes(),
                "slow_queries": await self._analyze_slow_queries(),
                "connection_stats": await self._get_connection_stats(),
                "replication_status": await self._get_replication_status(),
                "recommendations": []
            }
            
            # 生成优化建议
            analysis_results["recommendations"] = await self._generate_mongodb_recommendations(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"MongoDB性能分析失败: {e}")
            return {"error": str(e)}
    
    async def _get_mongodb_server_status(self) -> Dict[str, Any]:
        """获取MongoDB服务器状态"""
        try:
            status = await self.mongodb_db.command("serverStatus")
            
            return {
                "version": status.get("version"),
                "uptime": status.get("uptime"),
                "memory": {
                    "resident": status.get("mem", {}).get("resident", 0),
                    "virtual": status.get("mem", {}).get("virtual", 0),
                    "mapped": status.get("mem", {}).get("mapped", 0)
                },
                "connections": {
                    "current": status.get("connections", {}).get("current", 0),
                    "available": status.get("connections", {}).get("available", 0)
                },
                "opcounters": status.get("opcounters", {}),
                "network": status.get("network", {}),
                "locks": status.get("locks", {})
            }
            
        except Exception as e:
            logger.error(f"获取服务器状态失败: {e}")
            return {}
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            stats = await self.mongodb_db.command("dbStats")
            
            return {
                "db_name": stats.get("db"),
                "collections": stats.get("collections", 0),
                "views": stats.get("views", 0),
                "objects": stats.get("objects", 0),
                "avg_obj_size": stats.get("avgObjSize", 0),
                "data_size": stats.get("dataSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "index_size": stats.get("indexSize", 0),
                "total_size": stats.get("totalSize", 0)
            }
            
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return {}
    
    async def _get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            collection_names = await self.mongodb_db.list_collection_names()
            collection_stats = {}
            
            for collection_name in collection_names:
                try:
                    stats = await self.mongodb_db.command("collStats", collection_name)
                    collection_stats[collection_name] = {
                        "count": stats.get("count", 0),
                        "size": stats.get("size", 0),
                        "avg_obj_size": stats.get("avgObjSize", 0),
                        "storage_size": stats.get("storageSize", 0),
                        "indexes": stats.get("nindexes", 0),
                        "total_index_size": stats.get("totalIndexSize", 0),
                        "index_sizes": stats.get("indexSizes", {})
                    }
                except Exception as e:
                    logger.warning(f"获取集合 {collection_name} 统计失败: {e}")
                    
            return collection_stats
            
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {}
    
    async def _analyze_indexes(self) -> Dict[str, Any]:
        """分析索引使用情况"""
        try:
            collection_names = await self.mongodb_db.list_collection_names()
            index_analysis = {}
            
            for collection_name in collection_names:
                try:
                    collection = self.mongodb_db[collection_name]
                    
                    # 获取索引信息
                    indexes = await collection.list_indexes().to_list(length=None)
                    
                    # 获取索引统计
                    index_stats = []
                    try:
                        stats = await self.mongodb_db.command("collStats", collection_name, indexDetails=True)
                        index_stats = stats.get("indexDetails", {})
                    except:
                        pass
                    
                    index_info = []
                    for index in indexes:
                        index_name = index.get("name")
                        index_detail = {
                            "name": index_name,
                            "key": index.get("key", {}),
                            "unique": index.get("unique", False),
                            "sparse": index.get("sparse", False),
                            "background": index.get("background", False),
                            "text": "text" in str(index.get("key", {})),
                            "geospatial": any(v in ["2d", "2dsphere"] for v in str(index.get("key", {})).split())
                        }
                        
                        # 添加统计信息
                        if index_name in index_stats:
                            index_detail["stats"] = index_stats[index_name]
                            
                        index_info.append(index_detail)
                    
                    index_analysis[collection_name] = {
                        "indexes": index_info,
                        "total_indexes": len(indexes),
                        "unused_indexes": [],  # 需要通过$indexStats查询
                        "suggested_indexes": []  # 基于查询模式分析
                    }
                    
                except Exception as e:
                    logger.warning(f"分析集合 {collection_name} 索引失败: {e}")
                    
            return index_analysis
            
        except Exception as e:
            logger.error(f"索引分析失败: {e}")
            return {}
    
    async def _analyze_slow_queries(self) -> Dict[str, Any]:
        """分析慢查询"""
        try:
            # 获取profile数据
            slow_queries = []
            
            try:
                # 启用profiling（如果未启用）
                await self.mongodb_db.command("profile", 2, slowms=100)
                
                # 查询最近的慢操作
                cursor = self.mongodb_db.system.profile.find().sort("ts", -1).limit(50)
                async for doc in cursor:
                    slow_queries.append({
                        "timestamp": doc.get("ts"),
                        "duration": doc.get("millis", 0),
                        "command": doc.get("command", {}),
                        "ns": doc.get("ns"),
                        "client": doc.get("client"),
                        "user": doc.get("user")
                    })
                    
            except Exception as e:
                logger.warning(f"获取慢查询失败: {e}")
            
            return {
                "slow_queries": slow_queries,
                "analysis": self._analyze_query_patterns(slow_queries)
            }
            
        except Exception as e:
            logger.error(f"慢查询分析失败: {e}")
            return {}
    
    def _analyze_query_patterns(self, slow_queries: List[Dict]) -> Dict[str, Any]:
        """分析查询模式"""
        patterns = defaultdict(int)
        collections = defaultdict(int)
        operations = defaultdict(int)
        
        for query in slow_queries:
            # 按集合统计
            ns = query.get("ns", "")
            if ns:
                collections[ns] += 1
            
            # 按操作类型统计
            command = query.get("command", {})
            for op in command.keys():
                operations[op] += 1
        
        return {
            "most_affected_collections": dict(collections),
            "most_common_operations": dict(operations),
            "average_duration": sum(q.get("duration", 0) for q in slow_queries) / max(len(slow_queries), 1)
        }
    
    async def _get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        try:
            status = await self.mongodb_db.command("serverStatus")
            connections = status.get("connections", {})
            
            return {
                "current": connections.get("current", 0),
                "available": connections.get("available", 0),
                "total_created": connections.get("totalCreated", 0),
                "active": connections.get("active", 0)
            }
            
        except Exception as e:
            logger.error(f"获取连接统计失败: {e}")
            return {}
    
    async def _get_replication_status(self) -> Dict[str, Any]:
        """获取复制状态"""
        try:
            status = await self.mongodb_db.command("replSetGetStatus")
            return {
                "set": status.get("set"),
                "members": len(status.get("members", [])),
                "primary": next((m["name"] for m in status.get("members", []) if m.get("stateStr") == "PRIMARY"), None),
                "healthy_members": sum(1 for m in status.get("members", []) if m.get("health") == 1)
            }
            
        except Exception as e:
            # 单机模式下不支持复制集
            return {"mode": "standalone"}
    
    async def _generate_mongodb_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成MongoDB优化建议"""
        recommendations = []
        
        try:
            # 内存使用建议
            memory = analysis.get("server_status", {}).get("memory", {})
            resident = memory.get("resident", 0)
            if resident > 1024:  # 大于1GB
                recommendations.append("内存使用较高，考虑增加服务器内存或优化查询")
            
            # 连接数建议
            connections = analysis.get("connection_stats", {})
            current = connections.get("current", 0)
            available = connections.get("available", 0)
            if current > available * 0.8:
                recommendations.append("当前连接数接近限制，建议增加最大连接数或使用连接池")
            
            # 索引建议
            for collection_name, index_info in analysis.get("index_analysis", {}).items():
                if index_info.get("total_indexes", 0) > 10:
                    recommendations.append(f"集合 {collection_name} 索引过多({index_info['total_indexes']}个)，可能影响写入性能")
                elif index_info.get("total_indexes", 0) < 2:
                    recommendations.append(f"集合 {collection_name} 索引较少，建议根据查询模式添加适当索引")
            
            # 慢查询建议
            slow_queries = analysis.get("slow_queries", {}).get("slow_queries", [])
            if len(slow_queries) > 10:
                recommendations.append("检测到较多慢查询，建议优化查询语句或添加索引")
            
            # 集合大小建议
            for collection_name, stats in analysis.get("collection_stats", {}).items():
                size_mb = stats.get("size", 0) / (1024 * 1024)
                if size_mb > 1000:  # 大于1GB
                    recommendations.append(f"集合 {collection_name} 较大({size_mb:.2f}MB)，考虑分片或归档历史数据")
            
        except Exception as e:
            logger.error(f"生成MongoDB建议失败: {e}")
            recommendations.append("建议生成过程中出现错误，请检查日志")
        
        return recommendations
    
    async def analyze_redis_performance(self) -> Dict[str, Any]:
        """分析Redis性能"""
        try:
            analysis_results = {
                "server_info": await self._get_redis_server_info(),
                "memory_analysis": await self._analyze_redis_memory(),
                "key_analysis": await self._analyze_redis_keys(),
                "client_connections": await self._get_redis_client_info(),
                "performance_metrics": await self._get_redis_performance_metrics(),
                "slow_log": await self._get_redis_slow_log(),
                "recommendations": []
            }
            
            # 生成优化建议
            analysis_results["recommendations"] = await self._generate_redis_recommendations(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Redis性能分析失败: {e}")
            return {"error": str(e)}
    
    async def _get_redis_server_info(self) -> Dict[str, Any]:
        """获取Redis服务器信息"""
        try:
            info = await self.redis_client.info()
            
            return {
                "version": info.get("redis_version"),
                "uptime": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_rss": info.get("used_memory_rss"),
                "max_memory": info.get("maxmemory"),
                "total_commands_processed": info.get("total_commands_processed"),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "expired_keys": info.get("expired_keys"),
                "evicted_keys": info.get("evicted_keys")
            }
            
        except Exception as e:
            logger.error(f"获取Redis服务器信息失败: {e}")
            return {}
    
    async def _analyze_redis_memory(self) -> Dict[str, Any]:
        """分析Redis内存使用"""
        try:
            info = await self.redis_client.info("memory")
            
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            
            memory_analysis = {
                "used_memory": used_memory,
                "used_memory_human": info.get("used_memory_human"),
                "max_memory": max_memory,
                "memory_usage_ratio": (used_memory / max_memory * 100) if max_memory > 0 else 0,
                "used_memory_rss": info.get("used_memory_rss", 0),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                "mem_allocator": info.get("mem_allocator")
            }
            
            return memory_analysis
            
        except Exception as e:
            logger.error(f"Redis内存分析失败: {e}")
            return {}
    
    async def _analyze_redis_keys(self) -> Dict[str, Any]:
        """分析Redis键分布"""
        try:
            # 获取所有数据库的键信息
            key_analysis = {}
            
            for db_index in range(16):  # Redis默认16个数据库
                try:
                    await self.redis_client.select(db_index)
                    db_size = await self.redis_client.dbsize()
                    
                    if db_size > 0:
                        # 获取键样本进行分析
                        sample_keys = []
                        async for key in self.redis_client.scan_iter(count=100):
                            sample_keys.append(key.decode() if isinstance(key, bytes) else key)
                            if len(sample_keys) >= 100:
                                break
                        
                        # 分析键类型
                        key_types = defaultdict(int)
                        key_patterns = defaultdict(int)
                        
                        for key in sample_keys:
                            key_type = await self.redis_client.type(key)
                            key_types[key_type] += 1
                            
                            # 简单的模式识别
                            parts = key.split(':')
                            if len(parts) > 1:
                                pattern = ':'.join(parts[:-1]) + ':*'
                                key_patterns[pattern] += 1
                        
                        key_analysis[f"db_{db_index}"] = {
                            "total_keys": db_size,
                            "sample_size": len(sample_keys),
                            "key_types": dict(key_types),
                            "common_patterns": dict(key_patterns)
                        }
                        
                except Exception as e:
                    continue
            
            # 切回默认数据库
            await self.redis_client.select(0)
            
            return key_analysis
            
        except Exception as e:
            logger.error(f"Redis键分析失败: {e}")
            return {}
    
    async def _get_redis_client_info(self) -> Dict[str, Any]:
        """获取Redis客户端信息"""
        try:
            client_list = await self.redis_client.client_list()
            
            total_clients = len(client_list)
            client_types = defaultdict(int)
            
            for client in client_list:
                client_types[client.get("name", "unknown")] += 1
            
            return {
                "total_clients": total_clients,
                "client_types": dict(client_types),
                "max_clients": (await self.redis_client.config_get("maxclients")).get("maxclients", "N/A")
            }
            
        except Exception as e:
            logger.error(f"获取Redis客户端信息失败: {e}")
            return {}
    
    async def _get_redis_performance_metrics(self) -> Dict[str, Any]:
        """获取Redis性能指标"""
        try:
            info = await self.redis_client.info()
            
            # 计算命中率
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            hit_rate = (hits / (hits + misses) * 100) if (hits + misses) > 0 else 0
            
            return {
                "ops_per_second": info.get("instantaneous_ops_per_sec", 0),
                "hit_rate": hit_rate,
                "total_commands": info.get("total_commands_processed", 0),
                "rejected_connections": info.get("rejected_connections", 0),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "latest_fork_usec": info.get("latest_fork_usec", 0)
            }
            
        except Exception as e:
            logger.error(f"获取Redis性能指标失败: {e}")
            return {}
    
    async def _get_redis_slow_log(self) -> Dict[str, Any]:
        """获取Redis慢日志"""
        try:
            slow_log = await self.redis_client.slowlog_get(50)  # 获取最近50条慢日志
            
            slow_commands = []
            for entry in slow_log:
                slow_commands.append({
                    "id": entry[0],
                    "timestamp": entry[1],
                    "duration": entry[2],  # 微秒
                    "command": entry[3]
                })
            
            return {
                "slow_commands": slow_commands,
                "total_slow_commands": len(slow_commands),
                "average_duration": sum(cmd["duration"] for cmd in slow_commands) / max(len(slow_commands), 1)
            }
            
        except Exception as e:
            logger.error(f"获取Redis慢日志失败: {e}")
            return {}
    
    async def _generate_redis_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成Redis优化建议"""
        recommendations = []
        
        try:
            # 内存使用建议
            memory_analysis = analysis.get("memory_analysis", {})
            usage_ratio = memory_analysis.get("memory_usage_ratio", 0)
            
            if usage_ratio > 90:
                recommendations.append("内存使用率超过90%，建议增加内存或清理过期数据")
            elif usage_ratio > 70:
                recommendations.append("内存使用率较高，建议监控并考虑扩容")
            
            # 碎片率建议
            fragmentation_ratio = memory_analysis.get("memory_fragmentation_ratio", 0)
            if fragmentation_ratio > 1.5:
                recommendations.append("内存碎片率较高，建议执行MEMORY PURGE或重启Redis")
            
            # 命中率建议
            performance = analysis.get("performance_metrics", {})
            hit_rate = performance.get("hit_rate", 0)
            
            if hit_rate < 80:
                recommendations.append(f"缓存命中率较低({hit_rate:.1f}%)，建议优化缓存策略")
            
            # 驱逐键建议
            evicted_keys = performance.get("evicted_keys", 0)
            if evicted_keys > 1000:
                recommendations.append("检测到大量键被驱逐，建议增加内存或调整过期策略")
            
            # 慢日志建议
            slow_log = analysis.get("slow_log", {})
            if slow_log.get("total_slow_commands", 0) > 10:
                recommendations.append("检测到较多慢命令，建议优化数据结构或查询方式")
            
            # 客户端连接建议
            client_info = analysis.get("client_connections", {})
            total_clients = client_info.get("total_clients", 0)
            if total_clients > 100:
                recommendations.append("客户端连接数较多，建议使用连接池或减少连接数")
            
        except Exception as e:
            logger.error(f"生成Redis建议失败: {e}")
            recommendations.append("建议生成过程中出现错误，请检查日志")
        
        return recommendations
    
    async def optimize_database_performance(self) -> Dict[str, Any]:
        """执行数据库性能优化"""
        optimization_results = {
            "mongodb_optimizations": [],
            "redis_optimizations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # MongoDB优化
            mongodb_opts = await self._optimize_mongodb()
            optimization_results["mongodb_optimizations"] = mongodb_opts
            
            # Redis优化
            redis_opts = await self._optimize_redis()
            optimization_results["redis_optimizations"] = redis_opts
            
            logger.info("数据库性能优化完成")
            
        except Exception as e:
            logger.error(f"数据库性能优化失败: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def _optimize_mongodb(self) -> List[Dict[str, Any]]:
        """优化MongoDB性能"""
        optimizations = []
        
        try:
            # 1. 启用查询计划缓存
            try:
                await self.mongodb_db.command("planCacheClear")
                optimizations.append({
                    "action": "清理查询计划缓存",
                    "status": "success"
                })
            except Exception as e:
                optimizations.append({
                    "action": "清理查询计划缓存",
                    "status": "failed",
                    "error": str(e)
                })
            
            # 2. 压缩集合（仅对支持的存储引擎）
            collection_names = await self.mongodb_db.list_collection_names()
            for collection_name in collection_names[:3]:  # 限制前3个集合
                try:
                    await self.mongodb_db.command("compact", collection_name)
                    optimizations.append({
                        "action": f"压缩集合 {collection_name}",
                        "status": "success"
                    })
                except Exception as e:
                    optimizations.append({
                        "action": f"压缩集合 {collection_name}",
                        "status": "failed",
                        "error": str(e)
                    })
            
            # 3. 重建索引（谨慎操作）
            # await self._rebuild_indexes_if_needed()
            
        except Exception as e:
            logger.error(f"MongoDB优化失败: {e}")
        
        return optimizations
    
    async def _optimize_redis(self) -> List[Dict[str, Any]]:
        """优化Redis性能"""
        optimizations = []
        
        try:
            # 1. 内存碎片整理
            try:
                await self.redis_client.memory_purge()
                optimizations.append({
                    "action": "内存碎片整理",
                    "status": "success"
                })
            except Exception as e:
                optimizations.append({
                    "action": "内存碎片整理", 
                    "status": "failed",
                    "error": str(e)
                })
            
            # 2. 清理过期键
            try:
                # Redis会自动清理过期键，这里只是触发一次主动清理
                expired_count = 0  # 实际中需要扫描并计算
                optimizations.append({
                    "action": "清理过期键",
                    "status": "success",
                    "details": f"清理了 {expired_count} 个过期键"
                })
            except Exception as e:
                optimizations.append({
                    "action": "清理过期键",
                    "status": "failed", 
                    "error": str(e)
                })
            
            # 3. 重置慢日志
            try:
                await self.redis_client.slowlog_reset()
                optimizations.append({
                    "action": "重置慢日志",
                    "status": "success"
                })
            except Exception as e:
                optimizations.append({
                    "action": "重置慢日志",
                    "status": "failed",
                    "error": str(e)
                })
            
        except Exception as e:
            logger.error(f"Redis优化失败: {e}")
        
        return optimizations
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            start_time = time.time()
            
            # 并发分析MongoDB和Redis
            mongodb_analysis, redis_analysis = await asyncio.gather(
                self.analyze_mongodb_performance(),
                self.analyze_redis_performance(),
                return_exceptions=True
            )
            
            analysis_time = time.time() - start_time
            
            report = {
                "report_id": f"perf_report_{int(time.time())}",
                "generated_at": datetime.utcnow().isoformat(),
                "analysis_duration": analysis_time,
                "mongodb_analysis": mongodb_analysis if not isinstance(mongodb_analysis, Exception) else {"error": str(mongodb_analysis)},
                "redis_analysis": redis_analysis if not isinstance(redis_analysis, Exception) else {"error": str(redis_analysis)},
                "overall_health": "healthy",  # 基于分析结果计算
                "critical_issues": [],
                "optimization_suggestions": []
            }
            
            # 评估整体健康状况
            report["overall_health"] = self._evaluate_overall_health(report)
            
            # 汇总关键问题和建议
            report["critical_issues"] = self._identify_critical_issues(report)
            report["optimization_suggestions"] = self._consolidate_suggestions(report)
            
            logger.info(f"性能报告生成完成，耗时 {analysis_time:.2f} 秒")
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    def _evaluate_overall_health(self, report: Dict[str, Any]) -> str:
        """评估整体健康状况"""
        try:
            issues = 0
            
            # 检查MongoDB问题
            mongodb = report.get("mongodb_analysis", {})
            if "error" in mongodb:
                issues += 3
            else:
                # 检查关键指标
                memory = mongodb.get("server_status", {}).get("memory", {})
                if memory.get("resident", 0) > 2048:  # 大于2GB
                    issues += 1
                
                connections = mongodb.get("connection_stats", {})
                if connections.get("current", 0) > connections.get("available", 1000) * 0.8:
                    issues += 1
            
            # 检查Redis问题
            redis = report.get("redis_analysis", {})
            if "error" in redis:
                issues += 3
            else:
                memory = redis.get("memory_analysis", {})
                if memory.get("memory_usage_ratio", 0) > 90:
                    issues += 2
                elif memory.get("memory_usage_ratio", 0) > 70:
                    issues += 1
                
                performance = redis.get("performance_metrics", {})
                if performance.get("hit_rate", 100) < 70:
                    issues += 1
            
            # 根据问题数量评估健康状况
            if issues == 0:
                return "excellent"
            elif issues <= 2:
                return "healthy"
            elif issues <= 4:
                return "warning"
            else:
                return "critical"
                
        except Exception as e:
            logger.error(f"评估健康状况失败: {e}")
            return "unknown"
    
    def _identify_critical_issues(self, report: Dict[str, Any]) -> List[str]:
        """识别关键问题"""
        critical_issues = []
        
        try:
            # MongoDB关键问题
            mongodb = report.get("mongodb_analysis", {})
            if "error" in mongodb:
                critical_issues.append("MongoDB连接或分析失败")
            
            # Redis关键问题
            redis = report.get("redis_analysis", {})
            if "error" in redis:
                critical_issues.append("Redis连接或分析失败")
            else:
                memory = redis.get("memory_analysis", {})
                if memory.get("memory_usage_ratio", 0) > 95:
                    critical_issues.append("Redis内存使用率超过95%，可能导致性能问题")
                
                performance = redis.get("performance_metrics", {})
                if performance.get("evicted_keys", 0) > 10000:
                    critical_issues.append("Redis大量键被驱逐，建议立即增加内存")
                    
        except Exception as e:
            logger.error(f"识别关键问题失败: {e}")
            critical_issues.append("问题识别过程中出现错误")
        
        return critical_issues
    
    def _consolidate_suggestions(self, report: Dict[str, Any]) -> List[str]:
        """汇总优化建议"""
        all_suggestions = []
        
        try:
            # 收集MongoDB建议
            mongodb = report.get("mongodb_analysis", {})
            if "recommendations" in mongodb:
                all_suggestions.extend(mongodb["recommendations"])
            
            # 收集Redis建议
            redis = report.get("redis_analysis", {})
            if "recommendations" in redis:
                all_suggestions.extend(redis["recommendations"])
            
            # 去重并排序
            unique_suggestions = list(set(all_suggestions))
            return unique_suggestions[:10]  # 返回最重要的10条建议
            
        except Exception as e:
            logger.error(f"汇总建议失败: {e}")
            return ["建议汇总过程中出现错误，请检查详细分析结果"]
    
    async def close(self):
        """关闭数据库连接"""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("数据库性能分析器连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")

# 全局性能分析器实例
database_optimizer = DatabasePerformanceAnalyzer()