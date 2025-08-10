# VERSION: 1.0.0
# NAME: Initial Database Schema
# DESCRIPTION: Create initial collections and indexes for AI trading system
# DATABASE: mongodb
# DEPENDENCIES: 

"""
初始数据库架构迁移
创建AI量化交易系统的基础集合和索引
"""

async def up(mongodb_db, redis_client):
    """执行迁移 - 创建初始架构"""
    
    # 1. 创建用户集合
    await mongodb_db.users.create_index("username", unique=True)
    await mongodb_db.users.create_index("email", unique=True)
    await mongodb_db.users.create_index("created_at")
    
    # 2. 创建策略集合
    await mongodb_db.strategies.create_index("user_id")
    await mongodb_db.strategies.create_index("strategy_name")
    await mongodb_db.strategies.create_index("status")
    await mongodb_db.strategies.create_index("created_at")
    await mongodb_db.strategies.create_index([("user_id", 1), ("strategy_name", 1)], unique=True)
    
    # 3. 创建交易记录集合
    await mongodb_db.trades.create_index("user_id")
    await mongodb_db.trades.create_index("strategy_id")
    await mongodb_db.trades.create_index("symbol")
    await mongodb_db.trades.create_index("timestamp")
    await mongodb_db.trades.create_index("status")
    await mongodb_db.trades.create_index([("timestamp", -1)])  # 按时间倒序
    
    # 4. 创建市场数据集合
    await mongodb_db.market_data.create_index([("symbol", 1), ("timestamp", 1)], unique=True)
    await mongodb_db.market_data.create_index("timestamp")
    await mongodb_db.market_data.create_index("symbol")
    await mongodb_db.market_data.create_index("data_type")
    
    # 5. 创建AI分析结果集合
    await mongodb_db.ai_analysis.create_index("timestamp")
    await mongodb_db.ai_analysis.create_index("analysis_type")
    await mongodb_db.ai_analysis.create_index("symbol")
    await mongodb_db.ai_analysis.create_index("ai_engine")
    await mongodb_db.ai_analysis.create_index([("timestamp", -1)])
    
    # 6. 创建因子数据集合
    await mongodb_db.factors.create_index([("symbol", 1), ("timestamp", 1), ("factor_name", 1)], unique=True)
    await mongodb_db.factors.create_index("timestamp")
    await mongodb_db.factors.create_index("symbol")
    await mongodb_db.factors.create_index("factor_name")
    await mongodb_db.factors.create_index("factor_category")
    
    # 7. 创建风险监控集合
    await mongodb_db.risk_monitoring.create_index("user_id")
    await mongodb_db.risk_monitoring.create_index("timestamp")
    await mongodb_db.risk_monitoring.create_index("risk_level")
    await mongodb_db.risk_monitoring.create_index([("timestamp", -1)])
    
    # 8. 创建系统日志集合（TTL索引，30天后自动删除）
    await mongodb_db.system_logs.create_index("timestamp", expireAfterSeconds=2592000)  # 30天
    await mongodb_db.system_logs.create_index("level")
    await mongodb_db.system_logs.create_index("module")
    await mongodb_db.system_logs.create_index("user_id")
    
    # 9. 创建配置集合
    await mongodb_db.configurations.create_index("config_key", unique=True)
    await mongodb_db.configurations.create_index("user_id")
    await mongodb_db.configurations.create_index("updated_at")
    
    # 10. 创建API密钥集合
    await mongodb_db.api_keys.create_index("user_id")
    await mongodb_db.api_keys.create_index("key_hash", unique=True)
    await mongodb_db.api_keys.create_index("created_at")
    await mongodb_db.api_keys.create_index("status")
    
    print("✅ 初始数据库架构创建完成")
    return True

async def down(mongodb_db, redis_client):
    """回滚迁移 - 删除创建的索引和集合"""
    
    # 删除集合（谨慎操作）
    collections_to_drop = [
        "users", "strategies", "trades", "market_data", 
        "ai_analysis", "factors", "risk_monitoring", 
        "system_logs", "configurations", "api_keys"
    ]
    
    for collection_name in collections_to_drop:
        await mongodb_db[collection_name].drop()
    
    print("✅ 初始数据库架构回滚完成")
    return True