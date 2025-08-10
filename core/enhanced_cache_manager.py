"""
增强版缓存管理器
实现多级缓存、智能预热、分布式缓存同步
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as aioredis
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class CacheLevel(Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"      # 内存缓存 (最快)
    L2_REDIS = "l2_redis"        # Redis缓存 (快速)
    L3_DATABASE = "l3_database"  # 数据库缓存 (较慢)

class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"                  # 最近最少使用
    LFU = "lfu"                  # 最不常用
    TTL = "ttl"                  # 基于时间过期
    WRITE_THROUGH = "write_through"    # 写穿透
    WRITE_BACK = "write_back"          # 写回
    WRITE_AROUND = "write_around"      # 写绕过

@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    deletes: int = 0
    evictions: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    total_requests: int = 0
    avg_response_time_ms: float = 0.0
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.hits + self.misses
        return (self.hits / max(total, 1)) * 100
    
    def get_l1_hit_rate(self) -> float:
        """获取L1缓存命中率"""
        return (self.l1_hits / max(self.total_requests, 1)) * 100
    
    def get_l2_hit_rate(self) -> float:
        """获取L2缓存命中率"""
        return (self.l2_hits / max(self.total_requests, 1)) * 100

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    level: CacheLevel = CacheLevel.L1_MEMORY
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if not self.ttl:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl
    
    def access(self):
        """记录访问"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

class L1MemoryCache:
    """L1内存缓存"""
    
    def __init__(self, max_size: int = 1000, policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # LRU队列
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        start_time = time.time()
        
        try:
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查过期
                if entry.is_expired():
                    await self.delete(key)
                    self.stats.misses += 1
                    return None
                
                # 更新访问记录
                entry.access()
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.stats.hits += 1
                self.stats.l1_hits += 1
                return entry.value
            
            self.stats.misses += 1
            return None
            
        finally:
            response_time = (time.time() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * self.stats.total_requests + response_time) /
                (self.stats.total_requests + 1)
            )
            self.stats.total_requests += 1
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            # 检查容量限制
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                level=CacheLevel.L1_MEMORY
            )
            
            self.cache[key] = entry
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats.writes += 1
            return True
            
        except Exception as e:
            logger.error(f"L1缓存设置失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存数据"""
        try:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                self.stats.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"L1缓存删除失败: {e}")
            return False
    
    async def _evict(self):
        """缓存淘汰"""
        if not self.access_order:
            return
        
        try:
            if self.policy == CachePolicy.LRU:
                # 淘汰最近最少使用的
                evict_key = self.access_order[0]
            elif self.policy == CachePolicy.LFU:
                # 淘汰最不常用的
                evict_key = min(self.cache.keys(), 
                              key=lambda k: self.cache[k].access_count)
            else:
                # 默认LRU
                evict_key = self.access_order[0]
            
            await self.delete(evict_key)
            self.stats.evictions += 1
            
        except Exception as e:
            logger.error(f"缓存淘汰失败: {e}")
    
    def get_size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    async def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()

class SmartRedisCache:
    """智能Redis缓存管理器"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.stats = CacheStats()
        
        # 缓存配置
        self.default_ttl = 300  # 默认5分钟
        self.market_data_ttl = 60    # 市场数据1分钟
        self.ai_analysis_ttl = 1800  # AI分析30分钟
        self.static_data_ttl = 3600  # 静态数据1小时
        
        # 键前缀
        self.prefixes = {
            "market": "market:",
            "ai": "ai:",
            "config": "config:",
            "cache_meta": "meta:",
            "stats": "stats:"
        }
    
    def _get_key_info(self, key: str) -> Tuple[str, int]:
        """获取键信息和TTL"""
        if key.startswith(self.prefixes["market"]):
            return key, self.market_data_ttl
        elif key.startswith(self.prefixes["ai"]):
            return key, self.ai_analysis_ttl
        elif key.startswith(self.prefixes["config"]):
            return key, self.static_data_ttl
        else:
            return key, self.default_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        start_time = time.time()
        
        try:
            # 尝试获取JSON格式数据
            value = await self.redis.get(key)
            
            if value is None:
                self.stats.misses += 1
                return None
            
            # 尝试JSON解码
            try:
                data = json.loads(value.decode() if isinstance(value, bytes) else value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果不是JSON，尝试pickle
                try:
                    data = pickle.loads(value)
                except:
                    data = value.decode() if isinstance(value, bytes) else value
            
            self.stats.hits += 1
            self.stats.l2_hits += 1
            return data
            
        except Exception as e:
            logger.error(f"Redis获取失败 {key}: {e}")
            self.stats.misses += 1
            return None
            
        finally:
            response_time = (time.time() - start_time) * 1000
            self.stats.avg_response_time_ms = (
                (self.stats.avg_response_time_ms * self.stats.total_requests + response_time) /
                (self.stats.total_requests + 1)
            )
            self.stats.total_requests += 1
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            _, default_ttl = self._get_key_info(key)
            expire_time = ttl or default_ttl
            
            # 智能序列化
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str)
            elif isinstance(value, (str, int, float, bool)):
                serialized = json.dumps(value)
            else:
                # 复杂对象使用pickle
                serialized = pickle.dumps(value)
            
            # 设置缓存
            success = await self.redis.set(key, serialized, ex=expire_time)
            
            if success:
                self.stats.writes += 1
                
                # 更新元数据
                await self._update_cache_metadata(key, expire_time)
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Redis设置失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存数据"""
        try:
            result = await self.redis.delete(key)
            if result > 0:
                self.stats.deletes += 1
                await self._remove_cache_metadata(key)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis删除失败 {key}: {e}")
            return False
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        try:
            if not keys:
                return {}
            
            values = await self.redis.mget(keys)
            result = {}
            
            for i, (key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    try:
                        result[key] = json.loads(value.decode() if isinstance(value, bytes) else value)
                        self.stats.hits += 1
                    except:
                        try:
                            result[key] = pickle.loads(value)
                            self.stats.hits += 1
                        except:
                            result[key] = value.decode() if isinstance(value, bytes) else value
                            self.stats.hits += 1
                else:
                    self.stats.misses += 1
            
            self.stats.total_requests += len(keys)
            return result
            
        except Exception as e:
            logger.error(f"Redis批量获取失败: {e}")
            self.stats.misses += len(keys)
            return {}
    
    async def mset(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置"""
        try:
            if not data:
                return True
            
            # 准备批量数据
            pipe = self.redis.pipeline()
            
            for key, value in data.items():
                _, default_ttl = self._get_key_info(key)
                expire_time = ttl or default_ttl
                
                # 序列化
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value, default=str)
                elif isinstance(value, (str, int, float, bool)):
                    serialized = json.dumps(value)
                else:
                    serialized = pickle.dumps(value)
                
                pipe.set(key, serialized, ex=expire_time)
            
            results = await pipe.execute()
            success_count = sum(1 for r in results if r)
            
            self.stats.writes += success_count
            return success_count == len(data)
            
        except Exception as e:
            logger.error(f"Redis批量设置失败: {e}")
            return False
    
    async def _update_cache_metadata(self, key: str, ttl: int):
        """更新缓存元数据"""
        try:
            metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "ttl": ttl,
                "access_count": 0
            }
            
            meta_key = f"{self.prefixes['cache_meta']}{key}"
            await self.redis.hset(meta_key, mapping=metadata)
            await self.redis.expire(meta_key, ttl + 60)  # 元数据存活时间比数据长一点
            
        except Exception as e:
            logger.debug(f"更新缓存元数据失败: {e}")
    
    async def _remove_cache_metadata(self, key: str):
        """移除缓存元数据"""
        try:
            meta_key = f"{self.prefixes['cache_meta']}{key}"
            await self.redis.delete(meta_key)
        except Exception as e:
            logger.debug(f"移除缓存元数据失败: {e}")
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        try:
            info = await self.redis.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self.stats.get_hit_rate(),
                "total_keys": await self._count_keys(),
                "avg_response_time_ms": self.stats.avg_response_time_ms
            }
            
        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")
            return {}
    
    async def _count_keys(self) -> int:
        """统计键数量"""
        try:
            # 分别统计各类型键的数量
            total = 0
            for prefix in self.prefixes.values():
                keys = await self.redis.keys(f"{prefix}*")
                total += len(keys)
            return total
        except:
            return 0

class EnhancedCacheManager:
    """增强版缓存管理器"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.l1_cache = L1MemoryCache(max_size=5000)  # 增加L1缓存大小
        self.l2_cache = SmartRedisCache(redis_client)
        self.stats = CacheStats()
        self.cache = self  # 为了保持向后兼容
        
        # 缓存预热配置
        self.preload_enabled = True
        self.preload_patterns = [
            "market:*",
            "ai:sentiment:*",
            "config:*"
        ]
        self._initialized = False
    
    async def initialize(self):
        """初始化增强缓存管理器"""
        if self._initialized:
            return
            
        try:
            # 初始化L2缓存连接测试
            await self.l2_cache.redis.ping()
            
            # 启动监控任务
            self._monitoring_task = None  # 监控任务占位符
            
            self._initialized = True
            logger.info("✅ 增强缓存管理器初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 增强缓存管理器初始化失败: {e}")
            raise
    
    async def start_monitoring(self):
        """启动缓存监控"""
        try:
            # 启动后台监控任务（可选功能）
            if not hasattr(self, '_monitoring_task') or not self._monitoring_task:
                logger.info("📊 缓存监控任务已准备就绪")
                # 这里可以添加后台监控逻辑，暂时简化
                self._monitoring_task = True
        except Exception as e:
            logger.warning(f"⚠️ 监控启动失败: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        start_time = time.time()
        
        try:
            # L1内存缓存
            value = await self.l1_cache.get(key)
            if value is not None:
                return value
            
            # L2 Redis缓存
            value = await self.l2_cache.get(key)
            if value is not None:
                # 回填到L1缓存
                await self.l1_cache.set(key, value, ttl=60)  # L1缓存1分钟
                return value
            
            return None
            
        finally:
            self._update_global_stats(start_time)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """多级缓存设置"""
        start_time = time.time()
        
        try:
            # 同时写入L1和L2
            l1_success = await self.l1_cache.set(key, value, min(ttl or 300, 300))
            l2_success = await self.l2_cache.set(key, value, ttl)
            
            return l1_success and l2_success
            
        finally:
            self._update_global_stats(start_time)
    
    async def delete(self, key: str) -> bool:
        """多级缓存删除"""
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)
        return l1_success or l2_success
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取"""
        if not keys:
            return {}
        
        result = {}
        missing_keys = []
        
        # 先从L1缓存获取
        for key in keys:
            value = await self.l1_cache.get(key)
            if value is not None:
                result[key] = value
            else:
                missing_keys.append(key)
        
        # 从L2缓存获取剩余的键
        if missing_keys:
            l2_results = await self.l2_cache.mget(missing_keys)
            
            # 回填到L1缓存
            for key, value in l2_results.items():
                result[key] = value
                await self.l1_cache.set(key, value, ttl=60)
        
        return result
    
    async def mset(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """批量设置"""
        if not data:
            return True
        
        # 同时写入L1和L2
        l1_tasks = [self.l1_cache.set(k, v, min(ttl or 300, 300)) for k, v in data.items()]
        l2_success = await self.l2_cache.mset(data, ttl)
        
        l1_results = await asyncio.gather(*l1_tasks, return_exceptions=True)
        l1_success = all(r is True for r in l1_results if not isinstance(r, Exception))
        
        return l1_success and l2_success
    
    async def preload_cache(self):
        """缓存预热"""
        if not self.preload_enabled:
            return
        
        logger.info("🔥 开始缓存预热...")
        
        try:
            # 预热市场数据
            await self._preload_market_data()
            
            # 预热配置数据
            await self._preload_config_data()
            
            # 预热AI分析数据
            await self._preload_ai_data()
            
            logger.info("✅ 缓存预热完成")
            
        except Exception as e:
            logger.error(f"❌ 缓存预热失败: {e}")
    
    async def _preload_market_data(self):
        """预热市场数据"""
        try:
            # 预设一些常用的市场数据键
            common_symbols = ["BTC/USDT", "ETH/USDT"]
            exchanges = ["OKX", "Binance"]
            
            for symbol in common_symbols:
                for exchange in exchanges:
                    key = f"market:{exchange}:{symbol.replace('/', '')}"
                    # 如果Redis中有数据但L1缓存中没有，进行预热
                    if await self.l2_cache.get(key) is not None:
                        value = await self.l2_cache.get(key)
                        await self.l1_cache.set(key, value, ttl=60)
                        
        except Exception as e:
            logger.debug(f"市场数据预热失败: {e}")
    
    async def _preload_config_data(self):
        """预热配置数据"""
        try:
            config_keys = [
                "config:trading_pairs",
                "config:api_settings",
                "config:risk_limits"
            ]
            
            for key in config_keys:
                value = await self.l2_cache.get(key)
                if value is not None:
                    await self.l1_cache.set(key, value, ttl=300)
                    
        except Exception as e:
            logger.debug(f"配置数据预热失败: {e}")
    
    async def _preload_ai_data(self):
        """预热AI数据"""
        try:
            ai_keys = [
                "ai:sentiment:latest",
                "ai:composite_signal:latest",
                "ai:market_regime:latest"
            ]
            
            for key in ai_keys:
                value = await self.l2_cache.get(key)
                if value is not None:
                    await self.l1_cache.set(key, value, ttl=180)
                    
        except Exception as e:
            logger.debug(f"AI数据预热失败: {e}")
    
    def _update_global_stats(self, start_time: float):
        """更新全局统计"""
        response_time = (time.time() - start_time) * 1000
        self.stats.avg_response_time_ms = (
            (self.stats.avg_response_time_ms * self.stats.total_requests + response_time) /
            (self.stats.total_requests + 1)
        )
        self.stats.total_requests += 1
        
        # 合并L1和L2的统计
        self.stats.l1_hits = self.l1_cache.stats.l1_hits
        self.stats.l2_hits = self.l2_cache.stats.l2_hits
        self.stats.hits = self.l1_cache.stats.hits + self.l2_cache.stats.hits
        self.stats.misses = self.l1_cache.stats.misses + self.l2_cache.stats.misses
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计"""
        cache_info = await self.l2_cache.get_cache_info()
        
        return {
            "overall": {
                "total_requests": self.stats.total_requests,
                "hit_rate": self.stats.get_hit_rate(),
                "avg_response_time_ms": self.stats.avg_response_time_ms
            },
            "l1_memory": {
                "size": self.l1_cache.get_size(),
                "max_size": self.l1_cache.max_size,
                "hit_rate": self.stats.get_l1_hit_rate(),
                "evictions": self.l1_cache.stats.evictions
            },
            "l2_redis": {
                "hit_rate": self.stats.get_l2_hit_rate(),
                "used_memory": cache_info.get("used_memory", "N/A"),
                "connected_clients": cache_info.get("connected_clients", 0),
                "total_keys": cache_info.get("total_keys", 0)
            },
            "performance": {
                "l1_response_time_ms": self.l1_cache.stats.avg_response_time_ms,
                "l2_response_time_ms": self.l2_cache.stats.avg_response_time_ms
            }
        }
    
    async def clear_all_caches(self):
        """清空所有缓存"""
        await self.l1_cache.clear()
        # Redis清空需要谨慎，这里只清空我们的键
        for prefix in self.l2_cache.prefixes.values():
            keys = await self.l2_cache.redis.keys(f"{prefix}*")
            if keys:
                await self.l2_cache.redis.delete(*keys)

# 市场数据专用缓存接口
class MarketDataCache:
    """市场数据专用缓存接口"""
    
    def __init__(self, cache_manager: EnhancedCacheManager):
        self.cache = cache_manager
    
    async def set_market_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """设置市场数据"""
        key = f"market:{exchange}:{symbol.replace('/', '')}"
        return await self.cache.set(key, data, ttl=60)  # 1分钟过期
    
    async def get_market_data(self, exchange: str, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        key = f"market:{exchange}:{symbol.replace('/', '')}"
        return await self.cache.get(key)
    
    async def get_latest_prices(self, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """获取最新价格"""
        symbols = symbols or ["BTC/USDT", "ETH/USDT"]
        exchanges = ["OKX", "Binance"]
        
        keys = []
        for symbol in symbols:
            for exchange in exchanges:
                key = f"market:{exchange}:{symbol.replace('/', '')}"
                keys.append(key)
        
        cached_data = await self.cache.mget(keys)
        
        # 重新组织数据结构
        result = {}
        for symbol in symbols:
            result[symbol] = {}
            for exchange in exchanges:
                key = f"market:{exchange}:{symbol.replace('/', '')}"
                result[symbol][exchange.lower()] = cached_data.get(key)
        
        return result

# 创建全局增强缓存管理器实例的工厂函数
async def create_enhanced_cache_manager(redis_client: aioredis.Redis) -> EnhancedCacheManager:
    """创建增强缓存管理器"""
    cache_manager = EnhancedCacheManager(redis_client)
    await cache_manager.initialize()
    return cache_manager