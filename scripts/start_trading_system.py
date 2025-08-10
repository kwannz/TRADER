"""
完整交易系统启动器

启动和协调所有系统组件：
- 市场数据模拟器
- AI交易引擎  
- 策略执行引擎
- 数据管理器
- CLI界面
"""

import asyncio
import signal
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_manager import data_manager
from core.strategy_engine import strategy_engine
from core.market_simulator import market_simulator
from core.ai_trading_engine import ai_trading_engine
from python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class TradingSystemManager:
    """交易系统管理器"""
    
    def __init__(self):
        self.components = {
            'data_manager': data_manager,
            'market_simulator': market_simulator,
            'strategy_engine': strategy_engine,
            'ai_trading_engine': ai_trading_engine
        }
        
        self.startup_order = [
            'data_manager',
            'market_simulator', 
            'strategy_engine',
            'ai_trading_engine'
        ]
        
        self.shutdown_order = list(reversed(self.startup_order))
        
        self.running = False
        self.startup_tasks = []
        
    async def start_system(self) -> None:
        """启动完整交易系统"""
        try:
            logger.info("🚀 启动AI量化交易系统...")
            
            # 按顺序启动组件
            for component_name in self.startup_order:
                await self._start_component(component_name)
            
            # 创建示例策略
            await self._create_demo_strategies()
            
            self.running = True
            logger.info("✅ 交易系统启动完成！")
            
            # 显示系统状态
            await self._display_system_status()
            
        except Exception as e:
            logger.error(f"❌ 系统启动失败: {e}")
            await self.shutdown_system()
            raise
    
    async def _start_component(self, component_name: str) -> None:
        """启动单个组件"""
        try:
            component = self.components[component_name]
            logger.info(f"启动组件: {component_name}")
            
            if component_name == 'data_manager':
                await component.initialize()
                
            elif component_name == 'market_simulator':
                # 启动市场模拟器
                task = asyncio.create_task(component.start_simulation())
                self.startup_tasks.append(task)
                # 等待一会儿让数据生成
                await asyncio.sleep(2)
                
            elif component_name == 'strategy_engine':
                await component.initialize()
                
            elif component_name == 'ai_trading_engine':
                await component.start_engine()
            
            logger.info(f"✅ {component_name} 启动成功")
            
        except Exception as e:
            logger.error(f"❌ 启动{component_name}失败: {e}")
            raise
    
    async def _create_demo_strategies(self) -> None:
        """创建演示策略"""
        try:
            logger.info("创建演示策略...")
            
            # 创建网格策略
            from core.strategy_engine import GridStrategy
            grid_strategy = GridStrategy(
                strategy_id="demo_grid_btc",
                name="BTC网格策略演示",
                config={
                    "symbol": "BTC/USDT",
                    "grid_count": 10,
                    "price_range": 0.1,
                    "quantity_per_grid": 0.001
                }
            )
            await strategy_engine.add_strategy(grid_strategy)
            
            # 创建AI策略  
            from core.strategy_engine import AIStrategy
            ai_strategy = AIStrategy(
                strategy_id="demo_ai_eth",
                name="ETH AI策略演示", 
                config={
                    "symbol": "ETH/USDT",
                    "ai_model": "deepseek",
                    "analysis_interval": 300,  # 5分钟
                    "confidence_threshold": 0.7,
                    "position_size": 0.002
                }
            )
            await strategy_engine.add_strategy(ai_strategy)
            
            # 创建DCA策略
            from core.strategy_engine import DCAStrategy  
            dca_strategy = DCAStrategy(
                strategy_id="demo_dca_ada",
                name="ADA定投策略演示",
                config={
                    "symbol": "ADA/USDT", 
                    "interval_minutes": 60,  # 1小时
                    "buy_amount": 0.01
                }
            )
            await strategy_engine.add_strategy(dca_strategy)
            
            logger.info("✅ 演示策略创建完成")
            
        except Exception as e:
            logger.error(f"❌ 创建演示策略失败: {e}")
    
    async def _display_system_status(self) -> None:
        """显示系统状态"""
        try:
            print("\n" + "="*60)
            print("🚀 AI量化交易系统 - 实时仿真模式")
            print("="*60)
            
            # 市场模拟器状态
            market_summary = market_simulator.get_market_summary()
            print("\n📊 市场数据模拟器:")
            print(f"   - 模拟币种: {len(market_summary)}个")
            for symbol, data in list(market_summary.items())[:3]:
                change_pct = data['change_24h'] * 100
                print(f"   - {symbol}: ${data['price']:.2f} ({change_pct:+.2f}%)")
            
            # 策略引擎状态
            strategy_status = strategy_engine.get_strategy_status()
            print(f"\n🤖 策略引擎:")
            print(f"   - 活跃策略: {len(strategy_status)}个")
            for strategy_id, strategy in strategy_status.items():
                print(f"   - {strategy['name']}: {strategy['status']}")
            
            # AI引擎状态
            ai_status = ai_trading_engine.get_engine_status()
            print(f"\n🧠 AI交易引擎:")
            print(f"   - 运行状态: {'🟢 运行中' if ai_status['is_running'] else '🔴 已停止'}")
            print(f"   - 跟踪币种: {len(ai_status['tracked_symbols'])}个")
            print(f"   - 活跃信号: {ai_status['active_signals_count']}个")
            print(f"   - 信号成功率: {ai_status['success_rate']:.1f}%")
            
            # 数据库状态
            db_health = await data_manager.health_check()
            print(f"\n🗄️ 数据库状态:")
            print(f"   - MongoDB: {'🟢 正常' if db_health['mongodb'] else '🔴 异常'}")
            print(f"   - Redis: {'🟢 正常' if db_health['redis'] else '🔴 异常'}")
            
            print("\n🎯 系统功能:")
            print("   - ✅ 实时市场数据仿真 (10Hz Tick + 多时间框架K线)")
            print("   - ✅ AI驱动的智能交易决策")
            print("   - ✅ 多策略并行执行")
            print("   - ✅ 实时风险管理")
            print("   - ✅ Bloomberg风格CLI界面")
            
            print("\n📱 启动CLI界面:")
            print("   python cli_interface/main.py")
            
            print("\n⚡ 系统已就绪，按 Ctrl+C 优雅关闭")
            print("="*60)
            
        except Exception as e:
            logger.error(f"显示系统状态失败: {e}")
    
    async def shutdown_system(self) -> None:
        """关闭交易系统"""
        try:
            logger.info("🛑 开始关闭交易系统...")
            
            self.running = False
            
            # 按逆序关闭组件
            for component_name in self.shutdown_order:
                await self._shutdown_component(component_name)
            
            # 取消启动任务
            for task in self.startup_tasks:
                if not task.done():
                    task.cancel()
            
            if self.startup_tasks:
                await asyncio.gather(*self.startup_tasks, return_exceptions=True)
            
            logger.info("✅ 交易系统已完全关闭")
            
        except Exception as e:
            logger.error(f"关闭系统时出错: {e}")
    
    async def _shutdown_component(self, component_name: str) -> None:
        """关闭单个组件"""
        try:
            component = self.components[component_name]
            logger.info(f"关闭组件: {component_name}")
            
            if component_name == 'ai_trading_engine':
                await component.stop_engine()
                
            elif component_name == 'strategy_engine':
                await component.shutdown()
                
            elif component_name == 'market_simulator':
                await component.stop_simulation()
                
            elif component_name == 'data_manager':
                await component.close()
            
            logger.info(f"✅ {component_name} 已关闭")
            
        except Exception as e:
            logger.error(f"关闭{component_name}失败: {e}")
    
    async def run_forever(self) -> None:
        """保持系统运行"""
        try:
            while self.running:
                # 定期显示状态更新
                await asyncio.sleep(30)
                await self._show_runtime_status()
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，开始关闭系统...")
            await self.shutdown_system()
        except Exception as e:
            logger.error(f"系统运行错误: {e}")
            await self.shutdown_system()
    
    async def _show_runtime_status(self) -> None:
        """显示运行时状态"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 获取最新价格
            prices = market_simulator.get_current_prices()
            btc_price = prices.get('BTC/USDT', 0)
            
            # 获取AI状态  
            ai_status = ai_trading_engine.get_engine_status()
            active_signals = len(ai_status.get('active_signals_count', 0))
            
            # 获取策略状态
            strategies = strategy_engine.get_strategy_status()
            active_strategies = sum(1 for s in strategies.values() if s['status'] == 'active')
            
            print(f"\r[{current_time}] BTC: ${btc_price:.2f} | 活跃策略: {active_strategies} | AI信号: {active_signals}", end="", flush=True)
            
        except Exception:
            pass  # 静默处理状态显示错误

async def main():
    """主函数"""
    system_manager = TradingSystemManager()
    
    # 设置信号处理器
    def signal_handler():
        logger.info("收到关闭信号...")
        asyncio.create_task(system_manager.shutdown_system())
    
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
        loop.add_signal_handler(signal.SIGINT, signal_handler)
    
    try:
        # 启动系统
        await system_manager.start_system()
        
        # 保持运行
        await system_manager.run_forever()
        
    except KeyboardInterrupt:
        logger.info("用户中断系统")
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        # 设置事件循环策略
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 运行系统
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"启动系统失败: {e}")
        sys.exit(1)