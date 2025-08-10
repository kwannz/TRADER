# VERSION: 3.0.0
# NAME: Redis Cache Setup
# DESCRIPTION: Initialize Redis cache structure and patterns
# DATABASE: redis
# DEPENDENCIES: 2.0.0

"""
Redis缓存设置迁移
初始化Redis缓存结构、键命名模式和TTL设置
"""

async def up(mongodb_db, redis_client):
    """执行迁移 - 设置Redis缓存结构"""
    
    # 1. 设置缓存配置
    cache_config = {
        "market_data_ttl": 60,  # 市场数据缓存1分钟
        "ai_analysis_ttl": 300,  # AI分析缓存5分钟
        "user_session_ttl": 3600,  # 用户会话缓存1小时
        "strategy_cache_ttl": 1800,  # 策略缓存30分钟
        "factor_cache_ttl": 600,  # 因子缓存10分钟
        "system_status_ttl": 30  # 系统状态缓存30秒
    }
    
    # 存储缓存配置
    await redis_client.hset("system:cache_config", mapping=cache_config)
    
    # 2. 初始化缓存命名空间
    namespaces = {
        "market": "market_data",
        "ai": "ai_analysis", 
        "user": "user_sessions",
        "strategy": "strategies",
        "factor": "factors",
        "system": "system_status",
        "trading": "trading_signals",
        "risk": "risk_monitoring"
    }
    
    await redis_client.hset("system:cache_namespaces", mapping=namespaces)
    
    # 3. 设置分布式锁配置
    lock_config = {
        "default_timeout": 300,  # 默认锁超时5分钟
        "migration_lock_timeout": 1800,  # 迁移锁超时30分钟
        "trading_lock_timeout": 60,  # 交易锁超时1分钟
        "ai_processing_lock_timeout": 900  # AI处理锁超时15分钟
    }
    
    await redis_client.hset("system:lock_config", mapping=lock_config)
    
    # 4. 初始化计数器
    counters = {
        "api_requests": 0,
        "successful_trades": 0,
        "failed_trades": 0,
        "ai_analysis_requests": 0,
        "user_logins": 0,
        "system_errors": 0
    }
    
    for counter_name, initial_value in counters.items():
        await redis_client.set(f"counter:{counter_name}", initial_value)
    
    # 5. 设置速率限制配置
    rate_limits = {
        "api_requests_per_minute": 1000,
        "ai_requests_per_minute": 60,
        "trading_requests_per_minute": 100,
        "login_attempts_per_hour": 10
    }
    
    await redis_client.hset("system:rate_limits", mapping=rate_limits)
    
    # 6. 初始化系统状态
    system_status = {
        "database_status": "healthy",
        "ai_engine_status": "healthy", 
        "trading_engine_status": "healthy",
        "websocket_status": "healthy",
        "last_health_check": "2025-01-08T00:00:00Z"
    }
    
    await redis_client.hset("system:status", mapping=system_status)
    await redis_client.expire("system:status", 300)  # 5分钟过期
    
    # 7. 设置会话管理
    session_config = {
        "max_concurrent_sessions": 5,
        "session_timeout": 3600,
        "remember_me_timeout": 2592000  # 30天
    }
    
    await redis_client.hset("system:session_config", mapping=session_config)
    
    # 8. 初始化消息队列配置
    queue_config = {
        "ai_analysis_queue": "queue:ai_analysis",
        "trading_signals_queue": "queue:trading_signals", 
        "risk_alerts_queue": "queue:risk_alerts",
        "system_notifications_queue": "queue:notifications",
        "max_queue_size": 10000,
        "processing_timeout": 300
    }
    
    await redis_client.hset("system:queue_config", mapping=queue_config)
    
    # 9. 设置缓存预热数据
    # 预热常用的市场数据
    popular_symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    for symbol in popular_symbols:
        await redis_client.sadd("system:popular_symbols", symbol)
    
    # 10. 初始化监控指标
    metrics = {
        "cache_hit_ratio": 0.0,
        "avg_response_time": 0.0,
        "active_connections": 0,
        "memory_usage": 0.0,
        "cpu_usage": 0.0
    }
    
    await redis_client.hset("system:metrics", mapping=metrics)
    await redis_client.expire("system:metrics", 60)  # 1分钟过期
    
    print("✅ Redis缓存结构初始化完成")
    return True

async def down(mongodb_db, redis_client):
    """回滚迁移 - 清理Redis缓存设置"""
    
    # 删除创建的Redis键
    keys_to_delete = [
        "system:cache_config",
        "system:cache_namespaces", 
        "system:lock_config",
        "system:rate_limits",
        "system:status",
        "system:session_config",
        "system:queue_config",
        "system:popular_symbols",
        "system:metrics"
    ]
    
    # 删除计数器
    counter_keys = [
        "counter:api_requests",
        "counter:successful_trades",
        "counter:failed_trades",
        "counter:ai_analysis_requests",
        "counter:user_logins",
        "counter:system_errors"
    ]
    
    all_keys = keys_to_delete + counter_keys
    
    for key in all_keys:
        await redis_client.delete(key)
    
    print("✅ Redis缓存设置回滚完成")
    return True