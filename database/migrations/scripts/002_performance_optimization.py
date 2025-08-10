# VERSION: 2.0.0
# NAME: Performance Optimization Indexes
# DESCRIPTION: Add performance optimization indexes and compound indexes
# DATABASE: mongodb
# DEPENDENCIES: 1.0.0

"""
性能优化迁移
添加复合索引、分片键和性能优化索引
"""

async def up(mongodb_db, redis_client):
    """执行迁移 - 添加性能优化索引"""
    
    # 1. 交易记录性能优化索引
    await mongodb_db.trades.create_index([
        ("user_id", 1),
        ("timestamp", -1),
        ("status", 1)
    ], name="trades_user_time_status_idx")
    
    await mongodb_db.trades.create_index([
        ("strategy_id", 1),
        ("symbol", 1),
        ("timestamp", -1)
    ], name="trades_strategy_symbol_time_idx")
    
    await mongodb_db.trades.create_index([
        ("symbol", 1),
        ("side", 1),
        ("timestamp", -1)
    ], name="trades_symbol_side_time_idx")
    
    # 2. 市场数据查询优化索引
    await mongodb_db.market_data.create_index([
        ("symbol", 1),
        ("data_type", 1),
        ("timestamp", -1)
    ], name="market_data_symbol_type_time_idx")
    
    await mongodb_db.market_data.create_index([
        ("timestamp", -1),
        ("symbol", 1)
    ], name="market_data_time_symbol_idx")
    
    # 3. AI分析结果查询优化
    await mongodb_db.ai_analysis.create_index([
        ("symbol", 1),
        ("analysis_type", 1),
        ("timestamp", -1)
    ], name="ai_analysis_symbol_type_time_idx")
    
    await mongodb_db.ai_analysis.create_index([
        ("ai_engine", 1),
        ("timestamp", -1)
    ], name="ai_analysis_engine_time_idx")
    
    # 4. 因子数据查询优化
    await mongodb_db.factors.create_index([
        ("factor_category", 1),
        ("timestamp", -1)
    ], name="factors_category_time_idx")
    
    await mongodb_db.factors.create_index([
        ("symbol", 1),
        ("factor_category", 1),
        ("timestamp", -1)
    ], name="factors_symbol_category_time_idx")
    
    # 5. 策略性能查询优化
    await mongodb_db.strategies.create_index([
        ("user_id", 1),
        ("status", 1),
        ("created_at", -1)
    ], name="strategies_user_status_time_idx")
    
    await mongodb_db.strategies.create_index([
        ("strategy_type", 1),
        ("status", 1)
    ], name="strategies_type_status_idx")
    
    # 6. 风险监控查询优化
    await mongodb_db.risk_monitoring.create_index([
        ("user_id", 1),
        ("risk_level", 1),
        ("timestamp", -1)
    ], name="risk_user_level_time_idx")
    
    # 7. 系统日志查询优化
    await mongodb_db.system_logs.create_index([
        ("level", 1),
        ("timestamp", -1)
    ], name="logs_level_time_idx")
    
    await mongodb_db.system_logs.create_index([
        ("module", 1),
        ("timestamp", -1)
    ], name="logs_module_time_idx")
    
    await mongodb_db.system_logs.create_index([
        ("user_id", 1),
        ("timestamp", -1)
    ], name="logs_user_time_idx")
    
    # 8. 添加文本搜索索引
    await mongodb_db.strategies.create_index([
        ("strategy_name", "text"),
        ("description", "text")
    ], name="strategies_text_search_idx")
    
    await mongodb_db.system_logs.create_index([
        ("message", "text")
    ], name="logs_text_search_idx")
    
    # 9. 地理位置索引（如果需要）
    # await mongodb_db.users.create_index([("location", "2dsphere")])
    
    print("✅ 性能优化索引创建完成")
    return True

async def down(mongodb_db, redis_client):
    """回滚迁移 - 删除性能优化索引"""
    
    # 删除创建的索引
    indexes_to_drop = [
        ("trades", "trades_user_time_status_idx"),
        ("trades", "trades_strategy_symbol_time_idx"),
        ("trades", "trades_symbol_side_time_idx"),
        ("market_data", "market_data_symbol_type_time_idx"),
        ("market_data", "market_data_time_symbol_idx"),
        ("ai_analysis", "ai_analysis_symbol_type_time_idx"),
        ("ai_analysis", "ai_analysis_engine_time_idx"),
        ("factors", "factors_category_time_idx"),
        ("factors", "factors_symbol_category_time_idx"),
        ("strategies", "strategies_user_status_time_idx"),
        ("strategies", "strategies_type_status_idx"),
        ("risk_monitoring", "risk_user_level_time_idx"),
        ("system_logs", "logs_level_time_idx"),
        ("system_logs", "logs_module_time_idx"),
        ("system_logs", "logs_user_time_idx"),
        ("strategies", "strategies_text_search_idx"),
        ("system_logs", "logs_text_search_idx")
    ]
    
    for collection_name, index_name in indexes_to_drop:
        try:
            await mongodb_db[collection_name].drop_index(index_name)
        except Exception as e:
            print(f"删除索引 {index_name} 失败: {e}")
    
    print("✅ 性能优化索引回滚完成")
    return True