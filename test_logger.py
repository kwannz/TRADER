#!/usr/bin/env python3
"""
日志系统测试脚本
"""

import asyncio
import time
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.unified_logger import (
    get_logger, 
    LogCategory, 
    log_performance, 
    log_errors,
    create_default_logger
)
from config.logger_factory import create_configured_logger


async def test_basic_logging():
    """测试基础日志功能"""
    print("=== 测试基础日志功能 ===")
    
    logger = get_logger()
    
    # 测试不同级别的日志
    logger.trace("这是一条跟踪日志", LogCategory.SYSTEM)
    logger.debug("这是一条调试日志", LogCategory.SYSTEM)
    logger.info("这是一条信息日志", LogCategory.SYSTEM)
    logger.success("这是一条成功日志", LogCategory.SYSTEM)
    logger.warning("这是一条警告日志", LogCategory.SYSTEM)
    logger.error("这是一条错误日志", LogCategory.SYSTEM)
    logger.critical("这是一条严重错误日志", LogCategory.SYSTEM)
    
    print("✅ 基础日志测试完成")


async def test_category_logging():
    """测试分类日志"""
    print("\n=== 测试分类日志 ===")
    
    logger = get_logger()
    
    # 测试不同分类的日志
    logger.api_info("API请求成功", extra_data={"method": "GET", "url": "/api/v1/test"})
    logger.api_error("API请求失败", exception=Exception("连接超时"))
    
    logger.trading_info("下单成功", extra_data={"symbol": "BTC/USDT", "amount": 0.1})
    logger.trading_error("下单失败", exception=Exception("余额不足"))
    
    logger.ai_info("模型推理完成", extra_data={"model": "transformer", "accuracy": 0.95})
    
    logger.security_warning("检测到异常登录", extra_data={"ip": "192.168.1.100", "user": "test"})
    
    logger.performance_info("操作耗时", duration=125.5)
    
    print("✅ 分类日志测试完成")


async def test_context_logging():
    """测试上下文日志"""
    print("\n=== 测试上下文日志 ===")
    
    logger = get_logger()
    
    # 测试上下文管理
    with logger.context(user_id="user123", session_id="session456"):
        logger.info("用户登录", LogCategory.USER)
        
        with logger.context(request_id="req789"):
            logger.api_info("处理用户请求")
            logger.info("执行业务逻辑", LogCategory.SYSTEM)
    
    print("✅ 上下文日志测试完成")


async def test_performance_logging():
    """测试性能日志"""
    print("\n=== 测试性能日志 ===")
    
    logger = get_logger()
    
    # 测试性能上下文
    with logger.performance_context("数据处理"):
        # 模拟一些工作
        await asyncio.sleep(0.1)
        logger.info("处理中间步骤", LogCategory.SYSTEM)
        await asyncio.sleep(0.05)
    
    print("✅ 性能日志测试完成")


@log_performance(LogCategory.SYSTEM, "装饰器测试函数")
async def decorated_async_function():
    """带装饰器的异步函数"""
    await asyncio.sleep(0.1)
    return "async result"


@log_performance(LogCategory.SYSTEM, "同步装饰器测试")
def decorated_sync_function():
    """带装饰器的同步函数"""
    time.sleep(0.05)
    return "sync result"


@log_errors(LogCategory.SYSTEM)
async def error_function():
    """会抛出错误的函数"""
    raise ValueError("测试错误")


async def test_decorators():
    """测试装饰器"""
    print("\n=== 测试装饰器 ===")
    
    # 测试性能装饰器
    result1 = await decorated_async_function()
    result2 = decorated_sync_function()
    
    print(f"异步函数返回: {result1}")
    print(f"同步函数返回: {result2}")
    
    # 测试错误装饰器
    try:
        await error_function()
    except ValueError:
        pass  # 预期的错误
    
    print("✅ 装饰器测试完成")


async def test_configured_logger():
    """测试配置化日志"""
    print("\n=== 测试配置化日志 ===")
    
    try:
        config_logger = create_configured_logger("test_configured")
        
        config_logger.info("配置化日志测试", LogCategory.SYSTEM)
        config_logger.trading_info("交易日志测试", extra_data={"symbol": "ETH/USDT"})
        config_logger.performance_info("性能日志测试", duration=50.0)
        
        print("✅ 配置化日志测试完成")
        
    except Exception as e:
        print(f"⚠️ 配置化日志测试失败: {e}")


async def test_exception_logging():
    """测试异常日志"""
    print("\n=== 测试异常日志 ===")
    
    logger = get_logger()
    
    try:
        # 模拟一个异常
        raise RuntimeError("这是一个测试异常")
    except Exception as e:
        logger.error("捕获异常", LogCategory.SYSTEM, exception=e, 
                    extra_data={"context": "测试异常处理"})
    
    # 测试严重异常
    try:
        raise SystemError("系统级别错误")
    except Exception as e:
        logger.critical("系统严重错误", LogCategory.SYSTEM, exception=e)
    
    print("✅ 异常日志测试完成")


async def test_structured_data():
    """测试结构化数据日志"""
    print("\n=== 测试结构化数据 ===")
    
    logger = get_logger()
    
    # 测试复杂数据结构
    complex_data = {
        "order": {
            "id": "order123",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": 0.1,
            "price": 45000.0,
            "timestamp": "2024-01-01T12:00:00Z"
        },
        "user": {
            "id": "user456",
            "name": "测试用户",
            "level": "VIP"
        },
        "metadata": {
            "source": "web_app",
            "version": "1.0.0",
            "request_id": "req789"
        }
    }
    
    logger.trading_info(
        "复杂订单处理", 
        extra_data=complex_data,
        tags=["order", "btc", "market_buy"]
    )
    
    print("✅ 结构化数据测试完成")


async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始日志系统测试")
    print("=" * 50)
    
    try:
        await test_basic_logging()
        await test_category_logging()
        await test_context_logging()
        await test_performance_logging()
        await test_decorators()
        await test_configured_logger()
        await test_exception_logging()
        await test_structured_data()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成!")
        
        # 检查日志文件
        logs_dir = Path("logs")
        if logs_dir.exists():
            print(f"\n📁 日志文件位置: {logs_dir.absolute()}")
            print("生成的日志文件:")
            for log_file in logs_dir.glob("*.log*"):
                size_kb = log_file.stat().st_size / 1024
                print(f"  - {log_file.name}: {size_kb:.1f} KB")
        
        # 刷新和关闭日志
        logger = get_logger()
        await logger.flush_all()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())