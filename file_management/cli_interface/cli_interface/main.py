"""
CLI主界面应用

使用Rich + Textual构建Bloomberg Terminal风格的量化交易界面
支持4Hz实时数据刷新和Bloomberg主题
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Rich和Textual核心组件
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Button, Input, Select, 
    ProgressBar, DataTable, Log, TabbedContent, TabPane
)
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.binding import Binding
from textual.screen import Screen
from textual.css.query import NoMatches

# 应用模块导入
from .screens.dashboard import DashboardScreen
from .screens.strategy_manager import StrategyManagerScreen
from .screens.ai_assistant import AIAssistantScreen
from .screens.factor_lab import FactorLabScreen
from .screens.trade_history import TradeHistoryScreen
from .screens.settings import SettingsScreen
from .themes.bloomberg import BloombergTheme
from .components.status import StatusBar
from .components.charts import PriceChart, PerformanceChart
from .utils.layout import LayoutManager
from .utils.keyboard import KeyboardHandler
from .utils.animation import AnimationManager
from ..python_layer.utils.config import get_settings
from ..python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class TradingSystemApp(App):
    """
    量化交易系统主应用
    
    使用Textual构建的Terminal用户界面
    """
    
    CSS_PATH = "styles.css"  # CSS样式文件
    TITLE = "AI量化交易系统"
    SUB_TITLE = "Bloomberg Terminal Style"
    
    # 应用绑定键
    BINDINGS = [
        Binding("1", "switch_screen('dashboard')", "主仪表盘", priority=True),
        Binding("2", "switch_screen('strategies')", "策略管理", priority=True),
        Binding("3", "switch_screen('ai_assistant')", "AI助手", priority=True),
        Binding("4", "switch_screen('factor_lab')", "因子发现", priority=True),
        Binding("5", "switch_screen('trade_history')", "交易记录", priority=True),
        Binding("6", "switch_screen('settings')", "系统设置", priority=True),
        Binding("r", "refresh_data", "刷新数据"),
        Binding("h", "toggle_help", "帮助"),
        Binding("q", "quit", "退出", priority=True),
        Binding("ctrl+c", "quit", "强制退出"),
    ]
    
    # 响应式属性
    current_screen = reactive("dashboard")
    is_connected = reactive(True)
    real_time_data = reactive({})
    system_stats = reactive({})
    show_help = reactive(False)
    
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.console = Console(theme=BloombergTheme.get_theme())
        
        # 界面管理器
        self.layout_manager = LayoutManager()
        self.keyboard_handler = KeyboardHandler()
        self.animation_manager = AnimationManager()
        
        # 屏幕实例
        self._screens: Dict[str, Screen] = {}
        
        # 数据更新任务
        self._data_update_task: Optional[asyncio.Task] = None
        self._refresh_rate = 0.25  # 4Hz刷新频率
        
        logger.info("量化交易CLI应用初始化完成")

    def on_mount(self) -> None:
        """应用挂载时的初始化"""
        try:
            # 设置应用主题
            self.theme = BloombergTheme.get_textual_theme()
            
            # 初始化所有屏幕
            self._initialize_screens()
            
            # 启动数据更新任务
            self._start_data_updates()
            
            # 显示启动画面
            self._show_startup_banner()
            
            # 切换到主仪表盘
            self.action_switch_screen("dashboard")
            
        except Exception as e:
            logger.error(f"应用挂载失败: {e}")
            self.exit(1)

    def on_unmount(self) -> None:
        """应用卸载时的清理"""
        try:
            # 停止数据更新任务
            if self._data_update_task:
                self._data_update_task.cancel()
            
            logger.info("量化交易CLI应用已退出")
        except Exception as e:
            logger.error(f"应用退出错误: {e}")

    def _initialize_screens(self) -> None:
        """初始化所有屏幕"""
        try:
            self._screens = {
                "dashboard": DashboardScreen(),
                "strategies": StrategyManagerScreen(), 
                "ai_assistant": AIAssistantScreen(),
                "factor_lab": FactorLabScreen(),
                "trade_history": TradeHistoryScreen(),
                "settings": SettingsScreen()
            }
            
            # 安装所有屏幕
            for name, screen in self._screens.items():
                self.install_screen(screen, name=name)
                
            logger.info("所有屏幕初始化完成")
            
        except Exception as e:
            logger.error(f"屏幕初始化失败: {e}")
            raise

    def _start_data_updates(self) -> None:
        """启动数据更新任务"""
        if self._data_update_task is None or self._data_update_task.done():
            self._data_update_task = asyncio.create_task(self._data_update_loop())
            logger.info("数据更新任务已启动")

    async def _data_update_loop(self) -> None:
        """数据更新循环（4Hz刷新）"""
        while True:
            try:
                # 更新实时数据
                await self._update_real_time_data()
                
                # 更新系统统计
                await self._update_system_stats()
                
                # 通知当前屏幕数据更新
                current_screen_obj = self._screens.get(self.current_screen)
                if hasattr(current_screen_obj, 'on_data_update'):
                    await current_screen_obj.on_data_update(self.real_time_data)
                
                await asyncio.sleep(self._refresh_rate)
                
            except asyncio.CancelledError:
                logger.info("数据更新任务已取消")
                break
            except Exception as e:
                logger.error(f"数据更新错误: {e}")
                await asyncio.sleep(1)  # 错误时延长等待

    async def _update_real_time_data(self) -> None:
        """更新实时市场数据"""
        try:
            # 从市场模拟器获取实时数据
            from ..core.market_simulator import market_simulator
            from ..core.strategy_engine import strategy_engine
            from ..core.ai_trading_engine import ai_trading_engine
            
            # 获取市场数据
            market_summary = market_simulator.get_market_summary()
            
            # 格式化价格数据
            prices = {}
            for symbol, data in market_summary.items():
                prices[symbol] = {
                    "price": data["price"],
                    "change_24h": data["change_24h"],
                    "volume_24h": data["volume_24h"]
                }
            
            # 获取投资组合数据（模拟）
            portfolio_value = 10000  # 初始资金
            daily_pnl = sum(random.uniform(-50, 100) for _ in range(3))  # 模拟3个策略的盈亏
            
            # 获取策略状态
            strategy_status = strategy_engine.get_strategy_status()
            active_count = sum(1 for s in strategy_status.values() if s["status"] == "active")
            total_trades = sum(s["trades_count"] for s in strategy_status.values())
            
            # 计算成功率（模拟）
            success_rate = 0.65 if active_count > 0 else 0
            
            # 获取AI引擎状态
            ai_status = ai_trading_engine.get_engine_status()
            
            self.real_time_data = {
                "prices": prices,
                "portfolio": {
                    "total_value": portfolio_value + daily_pnl,
                    "daily_pnl": daily_pnl,
                    "unrealized_pnl": daily_pnl * 0.3
                },
                "strategies": {
                    "active_count": active_count,
                    "total_trades": total_trades,
                    "success_rate": success_rate,
                    "ai_signals": ai_status.get('active_signals_count', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"更新实时数据失败: {e}")
            # 回退到模拟数据
            import random
            
            self.real_time_data = {
                "prices": {
                    "BTC/USDT": {
                        "price": 45000 + random.uniform(-1000, 1000),
                        "change_24h": random.uniform(-0.05, 0.05),
                        "volume_24h": random.uniform(1000000, 5000000)
                    },
                    "ETH/USDT": {
                        "price": 2800 + random.uniform(-200, 200),
                        "change_24h": random.uniform(-0.08, 0.08),
                        "volume_24h": random.uniform(500000, 2000000)
                    }
                },
                "portfolio": {
                    "total_value": 10000 + random.uniform(-100, 200),
                    "daily_pnl": random.uniform(-50, 100),
                    "unrealized_pnl": random.uniform(-200, 300)
                },
                "strategies": {
                    "active_count": 3,
                    "total_trades": random.randint(150, 160),
                    "success_rate": 0.62 + random.uniform(-0.02, 0.02),
                    "ai_signals": 0
                }
            }

    async def _update_system_stats(self) -> None:
        """更新系统统计信息"""
        try:
            # 获取系统性能指标
            import psutil
            
            self.system_stats = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "connections": {
                    "okx": self.is_connected,
                    "binance": self.is_connected,
                    "mongodb": True,
                    "redis": True
                },
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"更新系统统计失败: {e}")

    def _show_startup_banner(self) -> None:
        """显示启动横幅"""
        banner_text = Text()
        banner_text.append("🚀 ", style="bold green")
        banner_text.append("AI量化交易系统", style="bold cyan")
        banner_text.append(" v1.0", style="dim")
        banner_text.append("\n📊 ", style="bold yellow")
        banner_text.append("Bloomberg Terminal风格界面", style="bold")
        banner_text.append("\n⚡ ", style="bold magenta")
        banner_text.append("Rust引擎 + Python业务 + AI智能", style="bold")
        
        startup_panel = Panel(
            Align.center(banner_text),
            title="[bold green]系统启动[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        
        # 在状态栏显示启动信息
        try:
            self.query_one(StatusBar).show_message("系统启动完成 ✅", duration=3)
        except NoMatches:
            pass

    # ============ 动作处理器 ============

    def action_switch_screen(self, screen_name: str) -> None:
        """切换屏幕"""
        if screen_name in self._screens:
            try:
                self.switch_screen(screen_name)
                self.current_screen = screen_name
                logger.info(f"切换到屏幕: {screen_name}")
                
                # 通知状态栏
                screen_names = {
                    "dashboard": "主仪表盘",
                    "strategies": "策略管理",
                    "ai_assistant": "AI智能助手",
                    "factor_lab": "因子发现实验室",
                    "trade_history": "交易记录",
                    "settings": "系统设置"
                }
                
                try:
                    self.query_one(StatusBar).show_message(
                        f"已切换到 {screen_names.get(screen_name, screen_name)}"
                    )
                except NoMatches:
                    pass
                
            except Exception as e:
                logger.error(f"切换屏幕失败: {e}")
        else:
            logger.warning(f"未知屏幕: {screen_name}")

    def action_refresh_data(self) -> None:
        """手动刷新数据"""
        try:
            # 立即触发数据更新
            if self._data_update_task:
                # 创建新的更新任务
                self._data_update_task.cancel()
                self._start_data_updates()
            
            try:
                self.query_one(StatusBar).show_message("数据已刷新 🔄", duration=2)
            except NoMatches:
                pass
                
            logger.info("手动刷新数据")
            
        except Exception as e:
            logger.error(f"刷新数据失败: {e}")

    def action_toggle_help(self) -> None:
        """切换帮助显示"""
        self.show_help = not self.show_help
        
        if self.show_help:
            self._show_help_modal()
        else:
            try:
                self.pop_screen()
            except:
                pass

    def _show_help_modal(self) -> None:
        """显示帮助模态框"""
        help_content = """
[bold cyan]快捷键说明[/bold cyan]

[bold]导航键:[/bold]
  [yellow]1[/yellow] - 主仪表盘    [yellow]2[/yellow] - 策略管理    [yellow]3[/yellow] - AI助手
  [yellow]4[/yellow] - 因子发现    [yellow]5[/yellow] - 交易记录    [yellow]6[/yellow] - 系统设置

[bold]功能键:[/bold]
  [yellow]R[/yellow] - 刷新数据    [yellow]H[/yellow] - 显示帮助    [yellow]Q[/yellow] - 退出系统
  [yellow]Ctrl+C[/yellow] - 强制退出

[bold]界面特性:[/bold]
  • 🔄 4Hz实时数据刷新
  • 📊 Bloomberg Terminal风格
  • 🚀 Rust高性能引擎集成
  • 🤖 AI智能分析支持
  
[dim]按 H 关闭帮助[/dim]
"""
        
        # 这里应该创建帮助模态屏幕
        # 简化实现：通过状态栏显示
        try:
            self.query_one(StatusBar).show_message("帮助: 按1-6切换界面，R刷新，Q退出", duration=5)
        except NoMatches:
            pass

    def action_quit(self) -> None:
        """退出应用"""
        logger.info("用户请求退出应用")
        self.exit()

    # ============ 响应式属性处理器 ============

    def watch_current_screen(self, old_screen: str, new_screen: str) -> None:
        """监视屏幕切换"""
        logger.debug(f"屏幕切换: {old_screen} -> {new_screen}")

    def watch_real_time_data(self, old_data: Dict, new_data: Dict) -> None:
        """监视实时数据变化"""
        # 数据变化时可以触发界面更新
        pass

    def watch_is_connected(self, old_status: bool, new_status: bool) -> None:
        """监视连接状态变化"""
        if old_status != new_status:
            status_text = "已连接" if new_status else "连接断开"
            status_style = "green" if new_status else "red"
            
            try:
                self.query_one(StatusBar).show_message(
                    f"网络状态: {status_text}",
                    style=status_style,
                    duration=3
                )
            except NoMatches:
                pass
            
            logger.info(f"连接状态变化: {status_text}")

# ============ 主界面屏幕 ============

class MainScreen(Screen):
    """主屏幕布局"""
    
    def compose(self) -> ComposeResult:
        """构建主屏幕组件"""
        # 主布局容器
        with Container(id="main-container"):
            # 顶部状态栏
            yield StatusBar(id="status-bar")
            
            # 主内容区域
            with Container(id="content-area"):
                yield Static("主内容区域", id="main-content")
            
            # 底部快捷键栏
            with Horizontal(id="shortcut-bar"):
                yield Static("[1]仪表盘", classes="shortcut-key")
                yield Static("[2]策略", classes="shortcut-key")
                yield Static("[3]AI助手", classes="shortcut-key")
                yield Static("[4]因子发现", classes="shortcut-key")
                yield Static("[5]交易记录", classes="shortcut-key")
                yield Static("[6]设置", classes="shortcut-key")
                yield Static("[Q]退出", classes="shortcut-key exit-key")

# ============ CLI启动函数 ============

def create_cli_app() -> TradingSystemApp:
    """创建CLI应用实例"""
    return TradingSystemApp()

async def run_cli_async():
    """异步运行CLI应用"""
    app = create_cli_app()
    try:
        await app.run_async()
    except KeyboardInterrupt:
        logger.info("用户中断，退出应用")
    except Exception as e:
        logger.error(f"CLI应用运行错误: {e}")
        raise
    finally:
        logger.info("CLI应用已关闭")

def run_cli():
    """同步运行CLI应用（主入口）"""
    try:
        # 检查终端支持
        if not _check_terminal_support():
            print("❌ 终端不支持高级功能，建议使用现代终端（如 iTerm2, Windows Terminal）")
            return 1
        
        # 显示启动信息
        console = Console()
        with console.status("[bold green]正在启动量化交易系统..."):
            import time
            time.sleep(1)
        
        console.print("[bold green]✅ 系统启动成功！[/bold green]")
        console.print("[dim]使用数字键 1-6 切换功能页面，按 H 查看帮助，按 Q 退出[/dim]\n")
        
        # 运行应用
        if sys.version_info >= (3, 11):
            # Python 3.11+ 异步运行支持
            asyncio.run(run_cli_async())
        else:
            # 兼容旧版本
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_cli_async())
            finally:
                loop.close()
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，正在退出...[/yellow]")
        return 0
    except Exception as e:
        console.print(f"\n[bold red]❌ 启动失败: {e}[/bold red]")
        logger.error(f"CLI启动失败: {e}")
        return 1

def _check_terminal_support() -> bool:
    """检查终端功能支持"""
    try:
        console = Console()
        
        # 检查颜色支持
        if not console.color_system:
            return False
        
        # 检查终端大小
        if console.size.width < 120 or console.size.height < 40:
            print(f"⚠️  终端尺寸太小 ({console.size.width}×{console.size.height})，建议至少 120×40")
            return False
        
        return True
        
    except Exception:
        return False

# ============ 主程序入口 ============

if __name__ == "__main__":
    sys.exit(run_cli())