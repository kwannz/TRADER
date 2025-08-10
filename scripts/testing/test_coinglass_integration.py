#!/usr/bin/env python3
"""
测试Coinglass数据集成
验证API客户端、数据收集器和分析器的功能
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
from core.coinglass_client import coinglass_client
from core.coinglass_collector import coinglass_collector_manager
from core.coinglass_analyzer import coinglass_analyzer
from core.real_data_manager import real_data_manager

# 加载环境变量
load_dotenv()

logger = get_logger(__name__)

async def test_coinglass_client():
    """测试Coinglass API客户端"""
    print("🔗 测试Coinglass API客户端...")
    print("=" * 60)
    
    try:
        # 测试连接
        connection_test = await coinglass_client.test_connection()
        print(f"连接测试: {connection_test}")
        
        # 测试支持的币种
        coins = await coinglass_client.get_supported_coins()
        print(f"支持币种数量: {len(coins.get('data', []))}")
        
        # 测试各种API端点
        endpoints_to_test = [
            ("恐惧贪婪指数", coinglass_client.get_fear_greed_index),
            ("BTC资金费率", lambda: coinglass_client.get_funding_rates("BTC")),
            ("BTC持仓数据", lambda: coinglass_client.get_open_interest("BTC")),
            ("BTC ETF流向", coinglass_client.get_btc_etf_netflow)
        ]
        
        for name, func in endpoints_to_test:
            try:
                result = await asyncio.wait_for(func(), timeout=10)
                available = "✅" if result.get("data") else "⚠️"
                print(f"{available} {name}: {len(result.get('data', []))}条数据")
            except Exception as e:
                print(f"❌ {name}: 失败 - {str(e)[:50]}...")
        
        # 客户端状态
        status = coinglass_client.get_status()
        print(f"\n客户端状态: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 客户端测试失败: {e}")
        return False

async def test_coinglass_collectors():
    """测试Coinglass数据收集器"""
    print("\n📊 测试Coinglass数据收集器...")
    print("=" * 60)
    
    try:
        # 执行一次性数据收集
        collectors_to_test = [
            "fear_greed",
            "funding_rate", 
            "open_interest",
            "liquidation",
            "etf_flow"
        ]
        
        for collector_name in collectors_to_test:
            try:
                print(f"测试{collector_name}收集器...")
                result = await coinglass_collector_manager.collect_once(collector_name)
                
                status = "✅" if result.get("success") else "❌"
                count = result.get("count", 0)
                duration = result.get("duration", 0)
                
                print(f"{status} {collector_name}: {count}条记录，耗时{duration:.2f}秒")
                
                if not result.get("success"):
                    print(f"   错误: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {collector_name}: 测试失败 - {e}")
        
        # 收集器状态
        status = coinglass_collector_manager.get_status()
        print(f"\n收集器管理器状态:")
        print(f"运行中: {status['is_running']}")
        print(f"活动任务数: {status['active_tasks']}")
        
        for name, collector_stats in status['collectors'].items():
            print(f"  {name}: 成功率{collector_stats['success_rate']}, 总文档{collector_stats['total_documents']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 收集器测试失败: {e}")
        return False

async def test_coinglass_analyzer():
    """测试Coinglass分析器"""
    print("\n🧠 测试Coinglass分析器...")
    print("=" * 60)
    
    try:
        # 初始化分析器
        if not coinglass_analyzer.is_initialized:
            await coinglass_analyzer.initialize()
        
        # 测试各种分析功能
        analysis_tests = [
            ("市场情绪分析", coinglass_analyzer.analyze_market_sentiment),
            ("资金费率分析", coinglass_analyzer.analyze_funding_rates),
            ("持仓数据分析", coinglass_analyzer.analyze_open_interest),
            ("ETF流向分析", coinglass_analyzer.analyze_etf_flows),
        ]
        
        for name, func in analysis_tests:
            try:
                print(f"执行{name}...")
                result = await func()
                print(f"✅ {name}: 完成")
                
                # 显示关键结果
                if hasattr(result, 'sentiment_score'):
                    print(f"   情绪得分: {result.sentiment_score}")
                elif hasattr(result, 'overall_rate'):
                    print(f"   整体费率: {result.overall_rate}")
                elif hasattr(result, 'total_oi'):
                    print(f"   总持仓: {result.total_oi}")
                elif hasattr(result, 'btc_flow'):
                    print(f"   BTC流向: {result.btc_flow}")
                    
            except Exception as e:
                print(f"❌ {name}: 失败 - {e}")
        
        # 测试综合信号生成
        print("\n生成综合分析信号...")
        try:
            composite_signal = await coinglass_analyzer.generate_composite_signal()
            print(f"✅ 综合信号生成成功")
            print(f"   综合得分: {composite_signal.overall_score}")
            print(f"   信号强度: {composite_signal.signal_strength}")
            print(f"   市场状态: {composite_signal.market_regime}")
            print(f"   风险评估: {composite_signal.risk_assessment}")
            print(f"   置信度: {composite_signal.confidence:.2f}")
            
            if composite_signal.key_factors:
                print(f"   关键因素: {', '.join(composite_signal.key_factors)}")
                
        except Exception as e:
            print(f"❌ 综合信号生成失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析器测试失败: {e}")
        return False

async def test_real_data_manager_integration():
    """测试真实数据管理器集成"""
    print("\n🔌 测试真实数据管理器集成...")
    print("=" * 60)
    
    try:
        # 初始化真实数据管理器
        await real_data_manager.initialize()
        print("✅ 真实数据管理器初始化成功")
        
        # 测试Coinglass分析获取
        coinglass_analysis = await real_data_manager.get_coinglass_analysis()
        
        if coinglass_analysis.get("enabled"):
            print("✅ Coinglass分析集成成功")
            
            composite = coinglass_analysis.get("composite_signal", {})
            print(f"   综合得分: {composite.get('overall_score', 'N/A')}")
            print(f"   市场状态: {composite.get('market_regime', 'N/A')}")
            print(f"   风险评估: {composite.get('risk_assessment', 'N/A')}")
            
            sentiment = coinglass_analysis.get("sentiment", {})
            print(f"   市场情绪: {sentiment.get('score', 'N/A')} ({sentiment.get('trend', 'N/A')})")
            
        else:
            print("⚠️ Coinglass未启用或出现错误")
            print(f"   信息: {coinglass_analysis.get('message', coinglass_analysis.get('error', 'Unknown'))}")
        
        # 健康检查
        health = await real_data_manager.health_check()
        coinglass_health = health.get("coinglass", {})
        
        print(f"\nCoinglass健康状态:")
        print(f"   启用状态: {coinglass_health.get('enabled', False)}")
        
        if coinglass_health.get("status"):
            status = coinglass_health["status"]
            print(f"   运行状态: {status.get('is_running', False)}")
            print(f"   收集器数量: {len(status.get('collectors', {}))}")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

async def test_data_migration():
    """测试数据迁移"""
    print("\n📦 测试Coinglass数据迁移...")
    print("=" * 60)
    
    coinglass_source = "/Users/zhaoleon/Desktop/trader/coinglass_副本"
    
    if not os.path.exists(coinglass_source):
        print(f"⚠️ 跳过数据迁移测试 - 源路径不存在: {coinglass_source}")
        return True
    
    try:
        print(f"开始从{coinglass_source}迁移数据...")
        result = await real_data_manager.migrate_coinglass_historical_data(coinglass_source)
        
        if result.get("success"):
            print("✅ 数据迁移成功")
            stats = result.get("stats", {})
            print(f"   处理文件: {stats.get('files_processed', 0)}")
            print(f"   迁移记录: {stats.get('records_migrated', 0)}")
            print(f"   创建集合: {len(stats.get('collections_created', []))}")
            print(f"   错误数量: {stats.get('errors', 0)}")
            
        else:
            print("❌ 数据迁移失败")
            print(f"   错误: {result.get('error', 'Unknown error')}")
        
        return result.get("success", False)
        
    except Exception as e:
        print(f"❌ 迁移测试失败: {e}")
        return False

async def run_all_tests():
    """运行所有测试"""
    print("🔍 Coinglass集成测试套件")
    print("=" * 80)
    
    tests = [
        ("API客户端", test_coinglass_client),
        ("数据收集器", test_coinglass_collectors),
        ("AI分析器", test_coinglass_analyzer),
        ("系统集成", test_real_data_manager_integration),
        ("数据迁移", test_data_migration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🚀 开始测试: {test_name}")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 测试总结
    print("\n" + "=" * 80)
    print("📋 测试结果总结")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！Coinglass集成成功！")
    else:
        print("⚠️ 部分测试失败，请检查相关配置和网络连接。")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        logger.error(f"测试运行失败: {e}")
        sys.exit(1)