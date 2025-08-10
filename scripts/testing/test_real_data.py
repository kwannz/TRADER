#!/usr/bin/env python3
"""
测试真实数据流连接
验证OKX和Binance数据获取是否正常
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from utils.logger import get_logger
from core.real_data_manager import real_data_manager

# 加载环境变量
load_dotenv()

logger = get_logger(__name__)

async def test_ticker_callback(ticker_data):
    """测试行情数据回调"""
    try:
        symbol = ticker_data.get("symbol", "")
        exchange = ticker_data.get("exchange", "")
        price = ticker_data.get("price", 0)
        change_pct = ticker_data.get("change_24h_pct", 0)
        
        print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] {exchange} {symbol}: ${price:.4f} ({change_pct:+.2f}%)")
        
    except Exception as e:
        logger.error(f"处理行情数据回调失败: {e}")

async def test_candle_callback(symbol, candle_data):
    """测试K线数据回调"""
    try:
        if candle_data:
            latest = candle_data[-1]
            print(f"📈 [{datetime.now().strftime('%H:%M:%S')}] K线: {symbol} - OHLC: {latest.get('open'):.4f}/{latest.get('high'):.4f}/{latest.get('low'):.4f}/{latest.get('close'):.4f}")
            
    except Exception as e:
        logger.error(f"处理K线数据回调失败: {e}")

async def test_real_data_connection():
    """测试真实数据连接"""
    print("🚀 开始测试真实数据连接...")
    
    try:
        # 初始化真实数据管理器
        print("1. 初始化真实数据管理器...")
        await real_data_manager.initialize()
        print("✅ 真实数据管理器初始化完成")
        
        # 添加回调函数
        real_data_manager.add_tick_callback(test_ticker_callback)
        real_data_manager.add_candle_callback(test_candle_callback)
        print("✅ 数据回调函数已添加")
        
        # 启动数据流
        print("2. 启动真实数据流...")
        await asyncio.wait_for(
            real_data_manager.start_real_data_stream(),
            timeout=30.0
        )
        print("✅ 真实数据流启动成功")
        
        # 检查连接状态
        print("\n3. 检查连接状态...")
        connection_status = real_data_manager.get_connection_status()
        print(f"连接状态: {connection_status}")
        
        # 运行60秒接收数据
        print("\n4. 接收实时数据 (60秒)...")
        print("=" * 60)
        
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 60:
            # 获取最新价格
            latest_prices = await real_data_manager.get_latest_prices()
            
            if latest_prices:
                print(f"\n📋 当前缓存价格:")
                for symbol, exchanges in latest_prices.items():
                    if exchanges:
                        for exchange, data in exchanges.items():
                            if data:
                                price = data.get("price", "N/A")
                                print(f"   {exchange.upper()}: {symbol} = ${price}")
            
            await asyncio.sleep(10)  # 每10秒检查一次
        
        print("=" * 60)
        print("✅ 实时数据测试完成")
        
    except asyncio.TimeoutError:
        print("❌ 连接超时，请检查网络和API配置")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"真实数据连接测试失败: {e}")
    
    finally:
        # 停止数据流
        try:
            await real_data_manager.stop_real_data_stream()
            print("🔄 数据流已停止")
        except Exception as e:
            logger.error(f"停止数据流失败: {e}")

async def test_historical_data():
    """测试历史数据获取"""
    print("\n🔄 开始测试历史数据获取...")
    
    try:
        # 获取少量历史数据用于测试
        print("获取最近1天的BTC和ETH历史数据...")
        training_data = await real_data_manager.fetch_historical_training_data(days_back=1)
        
        if training_data:
            print(f"✅ 历史数据获取成功，共获取到{len(training_data)}个交易对的数据")
            
            for symbol in training_data:
                print(f"\n📊 {symbol}:")
                for exchange in training_data[symbol]:
                    for timeframe in training_data[symbol][exchange]:
                        data_count = len(training_data[symbol][exchange][timeframe])
                        if data_count > 0:
                            print(f"   {exchange.upper()}: {timeframe} - {data_count}条记录")
        else:
            print("⚠️ 未获取到历史数据")
            
    except Exception as e:
        print(f"❌ 历史数据测试失败: {e}")
        logger.error(f"历史数据获取测试失败: {e}")

async def run_tests():
    """运行所有测试"""
    print("🔍 AI量化交易系统 - 真实数据测试")
    print("=" * 60)
    
    # 检查环境变量
    print("检查环境配置...")
    required_vars = ["MONGODB_URL", "REDIS_URL"]
    optional_vars = ["OKX_API_KEY", "BINANCE_API_KEY"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: 已配置")
        else:
            print(f"❌ {var}: 未配置")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value and value != "your_okx_api_key" and value != "your_binance_api_key":
            print(f"✅ {var}: 已配置")
        else:
            print(f"⚠️ {var}: 未配置（将使用公共API）")
    
    print("=" * 60)
    
    # 测试历史数据获取
    await test_historical_data()
    
    # 测试实时数据连接
    await test_real_data_connection()
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        logger.error(f"测试运行失败: {e}")