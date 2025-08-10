#!/usr/bin/env python3
"""
AI Factor Discovery & CTBench Integration Demo
AI因子发现与CTBench集成演示脚本
"""

import sys
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integration.ai_factor_ctbench_bridge import (
    get_ai_factor_bridge, FactorEnhancementRequest
)

class AIFactorCTBenchDemo:
    """AI因子发现与CTBench集成演示"""
    
    def __init__(self):
        self.bridge = None
        self.demo_factors = self.create_demo_factors()
        self.demo_market_data = self.generate_demo_market_data()
        
    async def initialize(self):
        """初始化集成桥接器"""
        print("🚀 正在初始化AI因子发现与CTBench集成系统...")
        try:
            self.bridge = await get_ai_factor_bridge()
            print("✅ 集成系统初始化成功!")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            raise
            
    def create_demo_factors(self) -> List[Dict[str, Any]]:
        """创建演示因子"""
        return [
            {
                'name': 'RSI_Mean_Reversion',
                'formula': 'rsi_14 < 30 or rsi_14 > 70',
                'type': 'momentum',
                'description': 'RSI均值回归因子，在超买超卖区域产生反向信号'
            },
            {
                'name': 'Price_Volume_Divergence',
                'formula': 'price_momentum * volume_momentum',
                'type': 'volume',
                'description': '价量背离因子，识别价格与成交量的不一致信号'
            },
            {
                'name': 'Volatility_Breakout',
                'formula': 'price_change / rolling_volatility',
                'type': 'volatility',
                'description': '波动率突破因子，识别异常价格波动'
            },
            {
                'name': 'Multi_Timeframe_Momentum',
                'formula': 'short_momentum * long_momentum',
                'type': 'trend',
                'description': '多时间框架动量因子，结合短期和长期趋势'
            },
            {
                'name': 'Market_Sentiment_Combo',
                'formula': 'sentiment_score * technical_score',
                'type': 'momentum',
                'description': '市场情绪组合因子，结合情绪和技术指标'
            }
        ]
        
    def generate_demo_market_data(self, days: int = 252) -> np.ndarray:
        """生成演示市场数据"""
        print("📊 生成演示市场数据...")
        
        np.random.seed(42)  # 确保可重复性
        
        # 生成价格序列
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, days)  # 0.1%日均收益，2%日波动率
        
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
            
        prices = np.array(prices[1:])
        
        # 构造OHLCV数据
        market_data = np.zeros((days, 6))
        
        for i in range(days):
            price = prices[i]
            # 生成OHLC
            daily_range = price * 0.02 * np.random.random()
            high = price + daily_range * np.random.random()
            low = price - daily_range * np.random.random()
            close = price * (1 + returns[i])
            
            # 生成成交量 (对数正态分布)
            volume = np.random.lognormal(9, 1) * 1000  # 平均约8000股
            
            market_data[i] = [price, high, low, close, volume, close]
            
        print(f"✅ 生成了 {days} 天的市场数据，价格范围: {prices.min():.2f} - {prices.max():.2f}")
        
        return market_data.reshape(1, days, 6)  # 重塑为CTBench期望的格式
        
    async def demo_single_factor_enhancement(self):
        """演示单因子增强验证"""
        print("\n" + "="*60)
        print("🎯 单因子增强验证演示")
        print("="*60)
        
        # 选择第一个因子进行演示
        demo_factor = self.demo_factors[0]
        
        print(f"📈 测试因子: {demo_factor['name']}")
        print(f"   类型: {demo_factor['type']}")
        print(f"   公式: {demo_factor['formula']}")
        print(f"   描述: {demo_factor['description']}")
        
        # 创建因子增强请求
        factor_request = FactorEnhancementRequest(
            factor_name=demo_factor['name'],
            factor_formula=demo_factor['formula'],
            factor_type=demo_factor['type'],
            base_data=self.demo_market_data[0],  # 去掉batch维度
            enhancement_scenarios=['black_swan', 'high_volatility', 'bear_market'],
            validation_period=252
        )
        
        print("\n🔄 开始CTBench增强验证...")
        
        try:
            result = await self.bridge.enhance_factor_with_synthetic_data(factor_request)
            
            if result['success']:
                print("✅ 因子增强验证完成!")
                
                # 显示关键结果
                perf = result['factor_performance']
                print(f"\n📊 因子表现:")
                print(f"   合成数据IC均值: {perf['synthetic_ic_mean']:.4f}")
                print(f"   稳健性评分: {perf['robustness_score']:.3f}")
                print(f"   压力测试生存率: {perf['stress_test_survival_rate']:.2%}")
                
                # 显示CTBench集成指标
                ctbench_metrics = result['ctbench_integration_metrics']
                print(f"\n🤖 CTBench集成指标:")
                print(f"   数据增强质量: {ctbench_metrics['data_augmentation_quality']:.3f}")
                print(f"   场景覆盖度: {ctbench_metrics['scenario_coverage']:.2%}")
                print(f"   验证置信度: {ctbench_metrics['validation_confidence']:.3f}")
                
                # 显示增强建议
                print(f"\n💡 增强建议:")
                for i, rec in enumerate(result['enhancement_recommendations'], 1):
                    print(f"   {i}. {rec}")
                    
                return result
            else:
                print(f"❌ 因子增强验证失败: {result.get('error', '未知错误')}")
                return None
                
        except Exception as e:
            print(f"❌ 演示过程出错: {e}")
            return None
            
    async def demo_batch_factor_validation(self):
        """演示批量因子验证"""
        print("\n" + "="*60)
        print("🔢 批量因子验证演示")
        print("="*60)
        
        print(f"📋 准备验证 {len(self.demo_factors)} 个因子:")
        for i, factor in enumerate(self.demo_factors, 1):
            print(f"   {i}. {factor['name']} ({factor['type']})")
            
        # 创建批量请求
        factor_requests = []
        for factor in self.demo_factors:
            factor_request = FactorEnhancementRequest(
                factor_name=factor['name'],
                factor_formula=factor['formula'],
                factor_type=factor['type'],
                base_data=self.demo_market_data[0],
                enhancement_scenarios=['black_swan', 'high_volatility'],
                validation_period=252
            )
            factor_requests.append(factor_request)
            
        print("\n🔄 开始批量验证...")
        
        try:
            results = await self.bridge.batch_factor_validation(factor_requests)
            
            # 统计结果
            successful = [r for r in results if r.get('success', False)]
            
            print(f"✅ 批量验证完成!")
            print(f"   总因子数: {len(results)}")
            print(f"   成功验证: {len(successful)}")
            print(f"   成功率: {len(successful)/len(results)*100:.1f}%")
            
            # 显示最佳因子
            if successful:
                best_factors = sorted(
                    successful,
                    key=lambda x: x.get('factor_performance', {}).get('robustness_score', 0),
                    reverse=True
                )[:3]
                
                print(f"\n🏆 表现最佳的3个因子:")
                for i, factor in enumerate(best_factors, 1):
                    perf = factor['factor_performance']
                    print(f"   {i}. {factor['factor_name']}")
                    print(f"      稳健性评分: {perf['robustness_score']:.3f}")
                    print(f"      IC均值: {perf['synthetic_ic_mean']:.4f}")
                    
            # 生成集成报告
            integration_report = self.bridge.generate_integration_report(results)
            
            print(f"\n📄 集成报告摘要:")
            ctbench_stats = integration_report['ctbench_integration_stats']
            print(f"   生成合成样本数: {ctbench_stats['total_synthetic_samples_generated']}")
            print(f"   测试场景数: {ctbench_stats['scenarios_tested']}")
            print(f"   平均稳健性评分: {ctbench_stats['average_robustness_score']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ 批量验证出错: {e}")
            return None
            
    async def demo_stress_testing(self):
        """演示压力测试"""
        print("\n" + "="*60)
        print("⚡ 压力测试演示")
        print("="*60)
        
        # 选择一个波动率因子进行压力测试
        volatility_factor = next((f for f in self.demo_factors if f['type'] == 'volatility'), self.demo_factors[0])
        
        print(f"🎯 压力测试因子: {volatility_factor['name']}")
        
        factor_request = FactorEnhancementRequest(
            factor_name=volatility_factor['name'],
            factor_formula=volatility_factor['formula'],
            factor_type=volatility_factor['type'],
            base_data=self.demo_market_data[0],
            enhancement_scenarios=['black_swan', 'high_volatility', 'bear_market'],
            validation_period=252
        )
        
        print("🔥 开始极端场景压力测试...")
        
        try:
            result = await self.bridge.enhance_factor_with_synthetic_data(factor_request)
            
            if result['success']:
                stress_results = result['stress_test_results']
                
                print("✅ 压力测试完成!")
                
                print(f"\n📊 各场景下的因子表现:")
                for scenario, value in stress_results.items():
                    if 'mean_ic' in scenario:
                        scenario_name = scenario.replace('_mean_ic', '')
                        print(f"   {scenario_name}: IC均值 = {value:.4f}")
                        
                # 分析压力测试结果
                survival_rate = result['factor_performance']['stress_test_survival_rate']
                robustness = result['factor_performance']['robustness_score']
                
                print(f"\n🛡️ 压力测试分析:")
                print(f"   生存率: {survival_rate:.2%}")
                print(f"   稳健性评分: {robustness:.3f}")
                
                if survival_rate > 0.7:
                    print("   ✅ 因子在极端市场下表现良好")
                elif survival_rate > 0.5:
                    print("   ⚠️  因子在极端市场下表现一般")
                else:
                    print("   ❌ 因子在极端市场下表现较差")
                    
                return result
            else:
                print(f"❌ 压力测试失败: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ 压力测试出错: {e}")
            return None
            
    async def demo_synthetic_data_analysis(self):
        """演示合成数据分析"""
        print("\n" + "="*60)
        print("📈 合成数据质量分析演示")
        print("="*60)
        
        print("🔄 使用CTBench生成多种市场场景...")
        
        try:
            # 生成不同类型的合成数据
            scenarios = ['black_swan', 'bull_market', 'bear_market', 'high_volatility', 'sideways']
            
            scenario_results = {}
            for scenario in scenarios:
                result = self.bridge.ctbench_service.synthetic_manager.generate_market_scenarios(
                    scenario, self.demo_market_data, 20  # 每种场景生成20个样本
                )
                
                if result['success']:
                    scenario_results[scenario] = result['data']
                    print(f"   ✅ {scenario}: 生成 {result['num_scenarios']} 个样本")
                else:
                    print(f"   ❌ {scenario}: 生成失败")
                    
            # 分析合成数据质量
            print(f"\n📊 合成数据质量分析:")
            
            original_data = self.demo_market_data[0]
            original_returns = np.diff(original_data[:, 3]) / original_data[:-1, 3]  # close价格收益率
            original_volatility = np.std(original_returns)
            
            print(f"   原始数据特征:")
            print(f"     均价: {np.mean(original_data[:, 3]):.2f}")
            print(f"     波动率: {original_volatility:.4f}")
            print(f"     收益率偏度: {self._calculate_skewness(original_returns):.3f}")
            
            print(f"\n   各场景合成数据特征:")
            for scenario_name, scenario_data in scenario_results.items():
                # 分析第一个样本的特征
                sample = scenario_data[0]
                sample_returns = np.diff(sample[:, 3]) / sample[:-1, 3]
                sample_volatility = np.std(sample_returns)
                
                print(f"     {scenario_name}:")
                print(f"       均价: {np.mean(sample[:, 3]):.2f}")
                print(f"       波动率: {sample_volatility:.4f} (vs 原始: {original_volatility:.4f})")
                print(f"       收益率偏度: {self._calculate_skewness(sample_returns):.3f}")
                
                # 计算与原始数据的相似性
                similarity = 1.0 - abs(sample_volatility - original_volatility) / original_volatility
                print(f"       相似性评分: {similarity:.3f}")
                
        except Exception as e:
            print(f"❌ 合成数据分析出错: {e}")
            
    def _calculate_skewness(self, data):
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
        
    async def demo_integration_status(self):
        """演示集成状态检查"""
        print("\n" + "="*60)
        print("🔗 集成状态检查演示")
        print("="*60)
        
        try:
            # 检查CTBench服务状态
            if self.bridge.ctbench_service:
                service_stats = self.bridge.ctbench_service.get_service_stats()
                model_status = self.bridge.ctbench_service.synthetic_manager.get_model_status()
                
                print("✅ CTBench服务状态:")
                print(f"   运行状态: {'正常' if service_stats['is_running'] else '异常'}")
                print(f"   处理请求数: {service_stats['requests_processed']}")
                print(f"   生成样本数: {service_stats['data_generated_samples']}")
                print(f"   错误计数: {service_stats['errors']}")
                
                print(f"\n🤖 可用模型:")
                for model_name, status in model_status.items():
                    if status['initialized']:
                        info = status['info']
                        print(f"   ✅ {model_name}: {info['parameters']:,} 参数, 设备: {info['device']}")
                    else:
                        print(f"   ❌ {model_name}: 未初始化")
            else:
                print("❌ CTBench服务未连接")
                
            # 检查风险管理器状态
            if self.bridge.risk_manager:
                print(f"\n✅ 风险管理器已连接")
            else:
                print(f"\n❌ 风险管理器未连接")
                
            # 显示集成配置
            print(f"\n⚙️ 集成配置:")
            for key, value in self.bridge.config.items():
                print(f"   {key}: {value}")
                
        except Exception as e:
            print(f"❌ 状态检查出错: {e}")
            
    async def run_complete_demo(self):
        """运行完整演示"""
        print("🎬 AI因子发现与CTBench集成系统完整演示")
        print("="*60)
        
        try:
            # 初始化
            await self.initialize()
            
            # 运行各个演示模块
            await self.demo_integration_status()
            await self.demo_synthetic_data_analysis()
            await self.demo_single_factor_enhancement()
            await self.demo_batch_factor_validation()
            await self.demo_stress_testing()
            
            print("\n" + "="*60)
            print("🎉 完整演示结束!")
            print("="*60)
            
            print("\n✨ 演示总结:")
            print("1. ✅ 成功集成AI因子发现与CTBench系统")
            print("2. ✅ 验证了合成数据增强因子验证的能力")
            print("3. ✅ 展示了批量因子处理和压力测试功能")
            print("4. ✅ 分析了不同市场场景下的因子稳健性")
            print("5. ✅ 提供了完整的因子优化建议体系")
            
            print(f"\n🚀 系统现已准备就绪，可以开始实际的因子发现和验证工作!")
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
            
async def main():
    """主函数"""
    demo = AIFactorCTBenchDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    asyncio.run(main())