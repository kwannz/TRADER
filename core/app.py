"""
核心应用类 - Textual主应用
负责协调各个模块和CLI界面
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Log
from textual.binding import Binding
from textual.screen import Screen

from config.settings import settings
from config.bloomberg_theme import get_color_system, STATUS_INDICATORS
from .unified_logger import get_logger, LogCategory, log_performance, log_errors
from .verbose_logger import get_verbose_logger, trace_execution, trace_async_execution, log_configuration_loaded
from .data_manager import data_manager
from .ai_engine import ai_engine
from .strategy_engine import strategy_engine
from .real_data_manager import real_data_manager
from .trading_simulator import trading_simulator

class QuantumTraderApp(App):
    """AI量化交易系统主应用"""
    
    CSS = """
    Screen {
        background: #0D1B2A;
    }
    
    Header {
        background: #1B263B;
        color: #E0E1DD;
    }
    
    Footer {
        background: #1B263B;
        color: #E0E1DD;
    }
    
    Static {
        background: #1B263B;
        color: #E0E1DD;
        border: solid #277DA1;
    }
    
    Log {
        background: #0D1B2A;
        color: #E0E1DD;
        border: solid #415A77;
    }
    
    .status-panel {
        height: 5;
        background: #1B263B;
        border: solid #277DA1;
        padding: 1;
    }
    
    .main-content {
        height: 1fr;
        margin: 1;
    }
    
    .sidebar {
        width: 30;
        background: #1B263B;
        border: solid #415A77;
    }
    
    .content-area {
        width: 1fr;
        margin-left: 1;
        background: #0D1B2A;
    }
    """
    
    BINDINGS = [
        Binding("1", "show_dashboard", "仪表盘", priority=True),
        Binding("2", "show_strategies", "策略管理", priority=True),
        Binding("3", "show_ai_assistant", "AI助手", priority=True),
        Binding("4", "show_factor_lab", "因子发现", priority=True),
        Binding("5", "show_trading_log", "交易记录", priority=True),
        Binding("6", "show_settings", "系统设置", priority=True),
        Binding("7", "show_backtest", "回测系统", priority=True),
        Binding("8", "show_simulation", "仿真交易", priority=True),
        Binding("q", "quit_app", "退出", priority=True),
        Binding("h", "show_help", "帮助", priority=True),
        Binding("r", "refresh_data", "刷新数据", priority=True),
    ]
    
    def __init__(self):
        super().__init__()
        self.title = "AI量化交易系统 v1.0"
        self.sub_title = "Bloomberg Terminal风格"
        self.current_screen = "dashboard"
        self.system_status = {}
        self.market_data = {}
        self.is_initialized = False
        
        # Initialize loggers
        self.logger = get_logger()
        self.verbose_logger = get_verbose_logger()
        self.logger = get_logger()
        self.backtest_status = {}
        self.simulation_status = {}
        
    def compose(self) -> ComposeResult:
        """构建UI布局"""
        yield Header(show_clock=True)
        
        with Container(classes="main-content"):
            with Horizontal():
                # 侧边栏
                with Vertical(classes="sidebar"):
                    yield Static("🔗 系统状态", classes="status-panel", id="connection-status")
                    yield Static("📊 市场概览", id="market-overview")
                    yield Static("🤖 AI状态", id="ai-status")
                    yield Static("⚡ 策略监控", id="strategy-monitor")
                    yield Static("🔄 回测状态", id="backtest-status")
                
                # 主要内容区域
                with Vertical(classes="content-area"):
                    yield Static("📈 主仪表盘", id="main-display")
                    yield Log(id="system-log")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """应用启动时初始化"""
        try:
            logger.info("Textual应用启动中...")
            
            # 设置定时刷新
            self.set_interval(1/settings.refresh_rate_hz, self.refresh_ui)
            self.set_interval(5.0, self.update_system_status)
            self.set_interval(30.0, self.update_ai_analysis)
            
            # 显示欢迎信息
            await self.update_main_display("正在初始化系统...")
            
            logger.info("Textual应用启动完成")
            
        except Exception as e:
            logger.error(f"应用启动失败: {e}")
    
    async def initialize(self):
        """初始化应用组件"""
        try:
            logger.info("初始化应用组件...")
            
            # 1. 初始化数据管理器
            await data_manager.initialize()
            
            # 2. 初始化AI引擎
            await ai_engine.initialize()
            
            # 3. 初始化策略引擎
            await strategy_engine.initialize()
            
            # 4. 初始化真实数据管理器
            await real_data_manager.initialize()
            
            # 5. 初始化仿真交易机
            await self._initialize_trading_simulator()
            
            # 6. 启动真实数据流
            asyncio.create_task(self._start_real_data_stream())
            
            # 7. 启动后台任务
            asyncio.create_task(self._background_tasks())
            
            self.is_initialized = True
            await self.update_main_display("✅ 系统初始化完成!")
            
            logger.info("应用组件初始化完成")
            
        except Exception as e:
            logger.error(f"应用初始化失败: {e}")
            await self.update_main_display(f"❌ 初始化失败: {e}")
            raise
    
    async def _initialize_trading_simulator(self):
        """初始化仿真交易机"""
        try:
            # 仿真交易机在导入时已经初始化，这里只需要准备一些默认策略
            logger.info("仿真交易机已就绪")
            
            # 初始化回测和仿真状态
            self.backtest_status = {
                "mode": trading_simulator.mode,
                "is_running": trading_simulator.is_running,
                "strategies_count": len(trading_simulator.strategies),
                "last_update": datetime.utcnow().isoformat()
            }
            
            self.simulation_status = {
                "market_data_active": False,
                "price_generators": len(trading_simulator.price_generators),
                "symbols": list(trading_simulator.symbols.keys()),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"仿真交易机初始化失败: {e}")
    
    async def _start_real_data_stream(self):
        """启动真实数据流"""
        try:
            # 添加市场数据回调
            real_data_manager.add_tick_callback(self._on_market_data)
            real_data_manager.add_candle_callback(self._on_candle_data)
            
            # 启动数据流
            await real_data_manager.start_real_data_stream()
            
            # 获取历史训练数据
            await self.log_message("🔄 开始获取历史训练数据...")
            try:
                training_data = await real_data_manager.fetch_historical_training_data(days_back=7)  # 获取7天数据
                await self.log_message(f"✅ 历史数据获取完成，共获取{len(training_data)}个交易对的数据")
            except Exception as e:
                await self.log_message(f"⚠️ 历史数据获取失败: {e}")
            
            logger.info("真实数据流启动完成")
            
        except Exception as e:
            logger.error(f"真实数据流启动失败: {e}")
            await self.log_message(f"❌ 真实数据流启动失败: {e}")
    
    async def _on_market_data(self, data: Dict[str, Any]):
        """处理市场数据回调"""
        try:
            symbol = data.get("symbol", "")
            exchange = data.get("exchange", "")
            
            # 更新市场数据缓存
            self.market_data[f"{exchange}:{symbol}"] = data
            
            # 记录日志
            price = data.get("price", 0)
            change_pct = data.get("change_24h_pct", 0)
            
            if abs(change_pct) > 5:  # 大幅波动时记录
                await self.log_message(
                    f"💰 {exchange} {symbol}: ${price:.4f} ({change_pct:+.2f}%)",
                    level="info"
                )
                
        except Exception as e:
            logger.error(f"处理市场数据失败: {e}")
    
    async def _on_candle_data(self, symbol: str, candle_data: List[Dict]):
        """处理K线数据回调"""
        try:
            if candle_data:
                latest_candle = candle_data[-1]
                await self.log_message(
                    f"📊 K线数据: {symbol} - Close: ${latest_candle.get('close', 0):.4f}",
                    level="debug"
                )
        except Exception as e:
            logger.error(f"处理K线数据失败: {e}")
    
    async def _background_tasks(self):
        """后台任务"""
        try:
            while True:
                # 检查系统健康状态
                if self.is_initialized:
                    await self._health_check()
                
                # 每分钟执行一次
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("后台任务已取消")
        except Exception as e:
            logger.error(f"后台任务异常: {e}")
    
    async def _health_check(self):
        """系统健康检查"""
        try:
            # 检查数据库连接
            db_status = await data_manager.health_check()
            
            # 检查真实数据连接
            ws_status = real_data_manager.get_connection_status()
            
            # 检查策略状态
            strategy_status = strategy_engine.get_strategy_status()
            
            # 更新系统状态
            self.system_status = {
                "database": db_status,
                "websocket": ws_status,
                "strategies": strategy_status,
                "last_check": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
    
    # ============ UI更新方法 ============
    
    async def refresh_ui(self):
        """刷新UI显示"""
        try:
            if not self.is_initialized:
                return
            
            # 更新连接状态
            await self.update_connection_status()
            
            # 更新市场概览
            await self.update_market_overview()
            
            # 更新AI状态
            await self.update_ai_status()
            
            # 更新策略监控
            await self.update_strategy_monitor()
            
            # 更新回测状态
            await self.update_backtest_status()
            
        except Exception as e:
            logger.debug(f"UI刷新失败: {e}")
    
    async def update_connection_status(self):
        """更新连接状态"""
        try:
            status_widget = self.query_one("#connection-status", Static)
            
            # 获取连接状态
            db_status = self.system_status.get("database", {})
            ws_status = self.system_status.get("websocket", {})
            
            # 构建状态文本
            mongodb_icon = STATUS_INDICATORS["connected"] if db_status.get("mongodb") else STATUS_INDICATORS["disconnected"]
            redis_icon = STATUS_INDICATORS["connected"] if db_status.get("redis") else STATUS_INDICATORS["disconnected"]
            okx_icon = STATUS_INDICATORS["connected"] if ws_status.get("okx") else STATUS_INDICATORS["disconnected"]
            binance_icon = STATUS_INDICATORS["connected"] if ws_status.get("binance") else STATUS_INDICATORS["disconnected"]
            
            status_text = f"""🔗 系统状态
            
{mongodb_icon} MongoDB
{redis_icon} Redis  
{okx_icon} OKX WebSocket
{binance_icon} Binance WebSocket"""
            
            status_widget.update(status_text)
            
        except Exception as e:
            logger.debug(f"更新连接状态失败: {e}")
    
    async def update_market_overview(self):
        """更新市场概览"""
        try:
            overview_widget = self.query_one("#market-overview", Static)
            
            # 获取主要币种数据（优先 OKX，回退 Binance）
            def get_pair(symbol_dash: str) -> dict:
                okx_key = f"OKX:{symbol_dash}"
                binance_key = f"Binance:{symbol_dash.replace('-', '')}"
                data = self.market_data.get(okx_key)
                if not data or not data.get("price"):
                    data = self.market_data.get(binance_key, {})
                return data or {}

            btc = get_pair("BTC-USDT")
            eth = get_pair("ETH-USDT")

            def fmt_price(v):
                try:
                    return f"${float(v):,.2f}"
                except Exception:
                    return "N/A"

            def fmt_pct(v):
                try:
                    return f"{float(v):+.2f}%"
                except Exception:
                    return "+0.00%"

            overview_text = f"""📊 市场概览

BTC/USDT: {fmt_price(btc.get('price'))}
24h: {fmt_pct(btc.get('change_24h_pct', 0))}

ETH/USDT: {fmt_price(eth.get('price'))}
24h: {fmt_pct(eth.get('change_24h_pct', 0))}

更新: {datetime.now().strftime('%H:%M:%S')}"""
            
            overview_widget.update(overview_text)
            
        except Exception as e:
            logger.debug(f"更新市场概览失败: {e}")
    
    async def update_ai_status(self):
        """更新AI状态"""
        try:
            ai_widget = self.query_one("#ai-status", Static)
            
            ai_text = f"""🤖 AI状态

{STATUS_INDICATORS["ai_thinking"]} DeepSeek: 活跃
{STATUS_INDICATORS["ai_thinking"]} Gemini: 活跃

情绪分析: 进行中
因子发现: 待启动"""
            
            ai_widget.update(ai_text)
            
        except Exception as e:
            logger.debug(f"更新AI状态失败: {e}")
    
    async def update_strategy_monitor(self):
        """更新策略监控"""
        try:
            strategy_widget = self.query_one("#strategy-monitor", Static)
            
            strategies = self.system_status.get("strategies", {})
            running_count = sum(1 for s in strategies.values() if s["status"] == "active")
            
            strategy_text = f"""⚡ 策略监控

运行策略: {running_count}/5
总交易: {sum(s["trades_count"] for s in strategies.values())}
系统状态: {STATUS_INDICATORS["running"]}"""
            
            strategy_widget.update(strategy_text)
            
        except Exception as e:
            logger.debug(f"更新策略监控失败: {e}")
    
    async def update_backtest_status(self):
        """更新回测状态"""
        try:
            backtest_widget = self.query_one("#backtest-status", Static)
            
            # 获取仿真交易机状态
            mode = trading_simulator.mode
            is_running = trading_simulator.is_running
            strategies_count = len(trading_simulator.strategies)
            
            # 根据模式显示不同信息
            if mode == "backtest":
                if trading_simulator.backtest_engine:
                    backtest_status = trading_simulator.get_backtest_status()
                    status_text = f"""🔄 回测系统

状态: {backtest_status.get('status', 'idle')}
策略: {strategies_count}个
进度: {backtest_status.get('progress', 0):.1%}"""
                else:
                    status_text = f"""🔄 回测系统

状态: 未配置
策略: {strategies_count}个
需要配置回测参数"""
            else:  # simulation mode
                status_text = f"""🔄 仿真交易

状态: {'运行中' if is_running else '停止'}
策略: {strategies_count}个
市场: 实时数据"""
            
            backtest_widget.update(status_text)
            
        except Exception as e:
            logger.debug(f"更新回测状态失败: {e}")
    
    async def update_main_display(self, content: str):
        """更新主显示区域"""
        try:
            main_widget = self.query_one("#main-display", Static)
            main_widget.update(content)
        except Exception as e:
            logger.debug(f"更新主显示失败: {e}")
    
    async def log_message(self, message: str, level: str = "info"):
        """记录日志消息"""
        try:
            log_widget = self.query_one("#system-log", Log)
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_widget.write_line(f"[{timestamp}] {message}")
        except Exception as e:
            logger.debug(f"记录日志失败: {e}")
    
    # ============ 定时任务 ============
    
    async def update_system_status(self):
        """定期更新系统状态"""
        if self.is_initialized:
            await self._health_check()
    
    async def update_ai_analysis(self):
        """定期更新AI分析"""
        try:
            if self.is_initialized:
                # 执行市场情绪分析
                sentiment = await ai_engine.analyze_market_sentiment()
                
                await self.log_message(
                    f"🧠 AI情绪分析: {sentiment.get('sentiment_score', 0):.2f} "
                    f"({sentiment.get('market_impact', '中性')})",
                    level="ai"
                )
                
        except Exception as e:
            logger.debug(f"AI分析更新失败: {e}")
    
    # ============ 按键绑定处理 ============
    
    async def action_show_dashboard(self) -> None:
        """显示主仪表盘"""
        self.current_screen = "dashboard"
        await self.update_main_display("📈 主仪表盘\n\n欢迎使用AI量化交易系统!\n实时数据正在更新中...")
        await self.log_message("🔄 切换到主仪表盘")
    
    async def action_show_strategies(self) -> None:
        """显示策略管理"""
        self.current_screen = "strategies"
        strategies = strategy_engine.get_strategy_status()
        
        strategy_list = "🚀 策略管理\n\n"
        if strategies:
            for sid, info in strategies.items():
                status_icon = STATUS_INDICATORS.get(info["status"], "❓")
                strategy_list += f"{status_icon} {info['name']} - {info['status']}\n"
        else:
            strategy_list += "暂无运行策略\n"
        
        await self.update_main_display(strategy_list)
        await self.log_message("🔄 切换到策略管理")
    
    async def action_show_ai_assistant(self) -> None:
        """显示AI助手"""
        self.current_screen = "ai_assistant"
        await self.update_main_display("🤖 AI智能助手\n\n您好！我是您的AI交易助手。\n请告诉我您需要什么帮助。")
        await self.log_message("🔄 切换到AI助手")
    
    async def action_show_factor_lab(self) -> None:
        """显示因子发现实验室"""
        self.current_screen = "factor_lab"
        await self.update_main_display("🔬 因子发现实验室\n\nAI因子挖掘功能\n使用DeepSeek+Gemini发现新的Alpha因子")
        await self.log_message("🔄 切换到因子发现实验室")
    
    async def action_show_trading_log(self) -> None:
        """显示交易记录"""
        self.current_screen = "trading_log"
        await self.update_main_display("📋 交易记录\n\n历史交易数据和绩效分析")
        await self.log_message("🔄 切换到交易记录")
    
    async def action_show_settings(self) -> None:
        """显示系统设置"""
        self.current_screen = "settings"
        await self.update_main_display("⚙️ 系统设置\n\nAPI配置、风控参数、界面设置")
        await self.log_message("🔄 切换到系统设置")
    
    async def action_show_backtest(self) -> None:
        """显示回测系统"""
        self.current_screen = "backtest"
        
        # 获取回测状态
        backtest_status = trading_simulator.get_backtest_status()
        
        backtest_text = f"""🔄 量化回测系统

当前模式: {trading_simulator.mode}
系统状态: {backtest_status.get('status', 'idle')}
已注册策略: {len(trading_simulator.strategies)}个

功能菜单:
1. 配置回测参数
2. 添加交易策略  
3. 运行历史回测
4. 查看结果报告
5. 性能分析

使用说明:
- 回测支持多策略并行测试
- 支持历史数据回放和虚拟交易
- 内置整合性能分析器
- 支持多种数据源（OKX、CoinGlass、模拟）"""
        
        await self.update_main_display(backtest_text)
        await self.log_message("🔄 切换到回测系统")
    
    async def action_show_simulation(self) -> None:
        """显示仿真交易"""
        self.current_screen = "simulation"
        
        # 获取仿真状态
        market_summary = trading_simulator.get_market_summary()
        
        simulation_text = f"""🎡 仿真交易系统

当前模式: {trading_simulator.mode}
运行状态: {'活跃' if trading_simulator.is_running else '停止'}
支持品种: {len(trading_simulator.symbols)}个

实时价格:"""
        
        # 添加实时价格信息
        for symbol, data in list(market_summary.items())[:5]:  # 显示前5个
            price = data.get('price', 0)
            change_24h = data.get('change_24h', 0)
            simulation_text += f"\n{symbol}: ${price:.4f} ({change_24h:+.2f}%)"
        
        simulation_text += f"""

仿真特性:
• 高频Tick级数据生成 (10Hz)
• K线数据实时更新
• 新闻事件与市场波动
• 突发事件模拟（闪崩、拉升）
• 做市商和流动性提供者模拟
• 滑点和延迟模拟
• 虚拟订单簿和撮合引擎"""
        
        await self.update_main_display(simulation_text)
        await self.log_message("🔄 切换到仿真交易")
    
    async def action_refresh_data(self) -> None:
        """刷新数据"""
        await self.log_message("🔄 手动刷新数据...")
        await self.refresh_ui()
    
    async def action_show_help(self) -> None:
        """显示帮助"""
        help_text = """❓ 快捷键帮助

[1] 主仪表盘 - 实时行情和系统状态
[2] 策略管理 - 创建和管理交易策略  
[3] AI助手 - 智能分析和建议
[4] 因子发现 - AI挖掘量化因子
[5] 交易记录 - 历史数据和绩效
[6] 系统设置 - 配置和参数
[7] 回测系统 - 历史数据回测测试
[8] 仿真交易 - 实时市场数据仿真
[R] 刷新数据 - 手动更新显示
[H] 显示帮助 - 当前页面
[Q] 安全退出 - 关闭系统

更多功能正在开发中..."""
        
        await self.update_main_display(help_text)
        await self.log_message("❓ 显示帮助信息")
    
    async def action_quit_app(self) -> None:
        """安全退出应用"""
        await self.log_message("👋 正在安全关闭系统...")
        await self.shutdown()
        self.exit()
    
    # ============ 生命周期管理 ============
    
    async def shutdown(self):
        """关闭应用"""
        try:
            logger.info("正在关闭应用...")
            
            # 关闭仿真交易机
            if trading_simulator.is_running:
                await trading_simulator.stop_simulation()
            
            # 关闭真实数据流
            await real_data_manager.stop_real_data_stream()
            
            # 关闭策略引擎
            await strategy_engine.shutdown()
            
            # 关闭AI引擎
            await ai_engine.close()
            
            # 关闭数据管理器
            await data_manager.close()
            
            logger.info("应用已安全关闭")
            
        except Exception as e:
            logger.error(f"关闭应用时发生错误: {e}")

# 便捷函数
async def create_app() -> QuantumTraderApp:
    """创建应用实例"""
    return QuantumTraderApp()