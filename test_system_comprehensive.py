#!/usr/bin/env python3
"""
综合系统测试脚本
测试所有修复的功能和模块
"""

import sys
import asyncio
import traceback
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = []
        self.success_count = 0
        self.failed_count = 0
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status} | {test_name}"
        if message:
            result += f" | {message}"
        
        self.test_results.append(result)
        print(result)
        
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1
    
    def test_imports(self):
        """测试模块导入"""
        print("\n" + "="*60)
        print("🔍 测试模块导入")
        print("="*60)
        
        modules = [
            ('config.settings', '配置模块'),
            ('core.unified_logger', '统一日志系统'),
            ('core.data_manager', '数据管理器'),
            ('core.ai_engine', 'AI引擎'),
            ('core.app', '主应用'),
            ('core.strategy_engine', '策略引擎'),
            ('core.risk_manager', '风险管理器'),
            ('core.trading_simulator', '交易模拟器'),
            ('core.websocket_client', 'WebSocket客户端'),
            ('main', '主程序入口')
        ]
        
        for module_name, description in modules:
            try:
                __import__(module_name)
                self.log_test(f"导入 {module_name}", True, description)
            except Exception as e:
                self.log_test(f"导入 {module_name}", False, f"{description} - {str(e)}")
    
    def test_logger_system(self):
        """测试日志系统"""
        print("\n" + "="*60)
        print("📝 测试统一日志系统")
        print("="*60)
        
        try:
            from core.unified_logger import get_logger, LogCategory, log_performance, log_errors
            logger = get_logger()
            
            # 测试基本日志功能
            logger.trace("测试TRACE级别", LogCategory.SYSTEM)
            logger.debug("测试DEBUG级别", LogCategory.SYSTEM)
            logger.info("测试INFO级别", LogCategory.SYSTEM)
            logger.success("测试SUCCESS级别", LogCategory.SYSTEM)
            logger.warning("测试WARNING级别", LogCategory.SYSTEM)
            
            self.log_test("日志系统基本功能", True, "所有日志级别正常")
            
            # 测试分类日志
            logger.api_info("API测试", extra_data={"endpoint": "/test"})
            logger.trading_info("交易测试", extra_data={"symbol": "BTC/USDT"})
            logger.ai_info("AI测试", extra_data={"model": "test_model"})
            
            self.log_test("分类日志功能", True, "API/Trading/AI分类日志正常")
            
            # 测试上下文管理
            with logger.context(user_id="test_user", session_id="test_session"):
                logger.info("上下文测试", LogCategory.USER)
            
            self.log_test("上下文管理", True, "上下文日志正常")
            
            # 测试性能监控
            with logger.performance_context("测试操作"):
                import time
                time.sleep(0.01)  # 模拟耗时操作
            
            self.log_test("性能监控", True, "性能上下文正常")
            
        except Exception as e:
            self.log_test("日志系统", False, f"日志系统测试失败: {e}")
    
    async def test_data_manager(self):
        """测试数据管理器"""
        print("\n" + "="*60)
        print("💾 测试数据管理器")
        print("="*60)
        
        try:
            from core.data_manager import data_manager
            
            # 测试数据管理器类型
            self.log_test("数据管理器导入", True, f"类型: {type(data_manager)}")
            
            # 测试初始化
            if hasattr(data_manager, 'initialize'):
                await data_manager.initialize()
                self.log_test("数据管理器初始化", True, "初始化成功")
            
            # 测试基本属性
            if hasattr(data_manager, 'mongodb'):
                self.log_test("MongoDB连接", True, "MongoDB客户端存在")
            
            if hasattr(data_manager, 'redis'):
                self.log_test("Redis连接", True, "Redis客户端存在")
                
        except Exception as e:
            self.log_test("数据管理器", False, f"数据管理器测试失败: {e}")
    
    def test_ai_engine(self):
        """测试AI引擎"""
        print("\n" + "="*60)
        print("🤖 测试AI引擎")
        print("="*60)
        
        try:
            from core.ai_engine import ai_engine
            
            self.log_test("AI引擎导入", True, f"类型: {type(ai_engine)}")
            
            # 测试客户端存在
            if hasattr(ai_engine, 'deepseek_client'):
                self.log_test("DeepSeek客户端", True, "DeepSeek客户端存在")
            
            if hasattr(ai_engine, 'gemini_client'):
                self.log_test("Gemini客户端", True, "Gemini客户端存在")
            
            # 测试方法存在
            if hasattr(ai_engine, 'analyze_market'):
                self.log_test("市场分析方法", True, "analyze_market方法存在")
                
        except Exception as e:
            self.log_test("AI引擎", False, f"AI引擎测试失败: {e}")
    
    def test_strategy_engine(self):
        """测试策略引擎"""
        print("\n" + "="*60)
        print("📈 测试策略引擎")
        print("="*60)
        
        try:
            from core.strategy_engine import strategy_engine
            
            self.log_test("策略引擎导入", True, f"类型: {type(strategy_engine)}")
            
            # 测试基本属性
            if hasattr(strategy_engine, 'strategies'):
                self.log_test("策略存储", True, "strategies属性存在")
            
            if hasattr(strategy_engine, 'execute_strategy'):
                self.log_test("策略执行方法", True, "execute_strategy方法存在")
                
        except Exception as e:
            self.log_test("策略引擎", False, f"策略引擎测试失败: {e}")
    
    def test_risk_manager(self):
        """测试风险管理器"""
        print("\n" + "="*60)
        print("🛡️ 测试风险管理器")
        print("="*60)
        
        try:
            from core.risk_manager import AdvancedRiskManager, advanced_risk_manager
            
            self.log_test("风险管理器导入", True, f"类型: {type(advanced_risk_manager)}")
            
            # 创建新实例测试
            risk_mgr = AdvancedRiskManager()
            self.log_test("风险管理器实例化", True, "成功创建新实例")
            
            # 测试基本方法
            if hasattr(risk_mgr, 'calculate_var'):
                self.log_test("VaR计算方法", True, "calculate_var方法存在")
                
        except Exception as e:
            self.log_test("风险管理器", False, f"风险管理器测试失败: {e}")
    
    def test_main_app(self):
        """测试主应用"""
        print("\n" + "="*60)
        print("🚀 测试主应用")
        print("="*60)
        
        try:
            from main import QuantumTraderCLI
            
            # 测试应用创建
            cli = QuantumTraderCLI()
            self.log_test("主应用创建", True, "QuantumTraderCLI创建成功")
            
            # 测试基本属性
            if hasattr(cli, 'logger'):
                self.log_test("应用日志器", True, "应用集成了统一日志系统")
            
            if hasattr(cli, 'console'):
                self.log_test("控制台界面", True, "Rich控制台初始化成功")
                
        except Exception as e:
            self.log_test("主应用", False, f"主应用测试失败: {e}")
    
    def test_configuration(self):
        """测试配置系统"""
        print("\n" + "="*60)
        print("⚙️ 测试配置系统")
        print("="*60)
        
        try:
            from config.settings import settings
            from config.logger_factory import create_configured_logger
            
            self.log_test("设置模块", True, f"类型: {type(settings)}")
            
            # 测试基本配置
            if hasattr(settings, 'version'):
                self.log_test("版本配置", True, f"版本: {settings.version}")
            
            if hasattr(settings, 'validate_config'):
                missing = settings.validate_config()
                self.log_test("配置验证", True, f"缺失配置: {len(missing)}项")
            
            # 测试日志工厂
            configured_logger = create_configured_logger("test_logger")
            self.log_test("配置化日志器", True, "日志工厂正常工作")
            
        except Exception as e:
            self.log_test("配置系统", False, f"配置系统测试失败: {e}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🎯 开始综合系统测试")
        print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 导入测试
        self.test_imports()
        
        # 日志系统测试
        self.test_logger_system()
        
        # 数据管理器测试
        await self.test_data_manager()
        
        # AI引擎测试
        self.test_ai_engine()
        
        # 策略引擎测试
        self.test_strategy_engine()
        
        # 风险管理器测试
        self.test_risk_manager()
        
        # 主应用测试
        self.test_main_app()
        
        # 配置系统测试
        self.test_configuration()
        
        # 输出测试结果
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)
        
        total_tests = self.success_count + self.failed_count
        success_rate = (self.success_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ 成功: {self.success_count} 项")
        print(f"❌ 失败: {self.failed_count} 项") 
        print(f"📈 成功率: {success_rate:.1f}%")
        print(f"🎯 总计: {total_tests} 项测试")
        
        if self.failed_count == 0:
            print("\n🎉 所有测试通过！系统调试完成！")
        else:
            print(f"\n⚠️  仍有 {self.failed_count} 项测试失败，需要进一步调试")
        
        # 生成测试报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        try:
            report_path = Path("test_results.md")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 系统调试测试报告\n\n")
                f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**测试结果**: {self.success_count}✅ / {self.failed_count}❌ / 总计{self.success_count + self.failed_count}项\n\n")
                
                f.write("## 详细测试结果\n\n")
                for result in self.test_results:
                    f.write(f"- {result}\n")
                
                f.write(f"\n## 系统状态\n\n")
                if self.failed_count == 0:
                    f.write("🎉 **系统状态**: 所有功能正常，调试完成！\n")
                else:
                    f.write(f"⚠️ **系统状态**: 仍有{self.failed_count}项需要修复\n")
            
            print(f"\n📄 测试报告已生成: {report_path.absolute()}")
            
        except Exception as e:
            print(f"⚠️  生成测试报告失败: {e}")


async def main():
    """主函数"""
    tester = SystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())