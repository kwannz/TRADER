"""
仪表盘屏幕 - 实时市场数据展示

Bloomberg Terminal风格的主仪表盘，支持：
- 实时价格行情和图表
- 投资组合概览 
- 策略运行状态
- 市场新闻和分析
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header, Static, DataTable, Label, ProgressBar, 
    TabbedContent, TabPane
)
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.align import Align
from rich.console import Console

from ..components.charts import RealTimePriceChart, PerformanceChart, VolumeChart
from ..components.status import MarketStatusWidget, PortfolioWidget, StrategyStatusWidget
from ..components.tables import WatchlistTable, TopGainersTable, NewsTable
from ..themes.bloomberg import BloombergTheme
from ...core.data_manager import data_manager
from ...core.strategy_engine import strategy_engine
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class DashboardScreen(Screen):
    """
    主仪表盘屏幕
    
    实时显示市场数据、投资组合状态和策略运行信息
    """
    
    CSS_PATH = "dashboard.css"
    BINDINGS = [
        ("escape", "app.pop_screen", "返回"),
        ("r", "refresh", "刷新"),
        ("f1", "toggle_fullscreen_chart", "全屏图表"),
        ("f2", "toggle_watchlist", "监控列表"),
    ]
    
    # 响应式数据
    market_data = reactive({})
    portfolio_data = reactive({})
    strategy_status = reactive({})
    news_data = reactive([])
    system_status = reactive({})
    
    def __init__(self):
        super().__init__()
        self.refresh_interval = 0.25  # 4Hz刷新
        self.update_task: Optional[asyncio.Task] = None
        self.theme = BloombergTheme()
        
    def compose(self) -> ComposeResult:
        """构建仪表盘布局"""
        
        with Container(id="dashboard-container"):
            # 顶部市场状态栏
            with Horizontal(id="market-status-bar", classes="status-bar"):
                yield MarketStatusWidget(id="market-status")
                yield PortfolioWidget(id="portfolio-widget")
                yield StrategyStatusWidget(id="strategy-widget")
            
            # 主内容区域 - 网格布局
            with Grid(id="main-grid"):
                # 左侧价格图表区域
                with Container(id="charts-section", classes="dashboard-panel"):
                    with TabbedContent(id="chart-tabs"):
                        with TabPane("价格走势", id="price-tab"):
                            yield RealTimePriceChart(
                                id="price-chart",
                                symbol="BTC/USDT",
                                timeframe="1m"
                            )
                        with TabPane("交易量", id="volume-tab"):
                            yield VolumeChart(
                                id="volume-chart", 
                                symbol="BTC/USDT"
                            )
                        with TabPane("收益曲线", id="pnl-tab"):
                            yield PerformanceChart(
                                id="performance-chart",
                                period="1d"
                            )
                
                # 中间数据表格区域
                with Vertical(id="tables-section", classes="dashboard-panel"):
                    # 监控列表
                    with Container(id="watchlist-container", classes="table-container"):
                        yield Label("📊 市场监控", classes="panel-title")
                        yield WatchlistTable(id="watchlist-table")
                    
                    # 涨幅榜
                    with Container(id="gainers-container", classes="table-container"):
                        yield Label("🚀 今日涨幅榜", classes="panel-title")
                        yield TopGainersTable(id="gainers-table")
                
                # 右侧信息面板
                with Vertical(id="info-section", classes="dashboard-panel"):
                    # 实时新闻
                    with Container(id="news-container", classes="info-container"):
                        yield Label("📰 市场新闻", classes="panel-title")
                        yield NewsTable(id="news-table")
                    
                    # 系统状态
                    with Container(id="system-container", classes="info-container"):
                        yield Label("⚙️ 系统状态", classes="panel-title")
                        yield Static(id="system-status")
                    
                    # 快速操作
                    with Container(id="actions-container", classes="info-container"):
                        yield Label("🎯 快速操作", classes="panel-title")
                        yield Static(id="quick-actions")

    def on_mount(self) -> None:
        """屏幕挂载时初始化"""
        try:
            self.start_data_updates()
            logger.info("仪表盘屏幕已挂载")
        except Exception as e:
            logger.error(f"仪表盘挂载失败: {e}")

    def on_unmount(self) -> None:
        """屏幕卸载时清理"""
        try:
            self.stop_data_updates()
            logger.info("仪表盘屏幕已卸载")
        except Exception as e:
            logger.error(f"仪表盘卸载失败: {e}")

    def start_data_updates(self) -> None:
        """启动数据更新任务"""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self._update_loop())
            logger.debug("仪表盘数据更新任务已启动")

    def stop_data_updates(self) -> None:
        """停止数据更新任务"""
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()

    async def _update_loop(self) -> None:
        """数据更新循环"""
        while True:
            try:
                # 并行更新所有数据
                await asyncio.gather(
                    self._update_market_data(),
                    self._update_portfolio_data(),
                    self._update_strategy_status(),
                    self._update_news_data(),
                    self._update_system_status(),
                    return_exceptions=True
                )
                
                # 更新界面组件
                self._refresh_widgets()
                
                await asyncio.sleep(self.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"仪表盘数据更新错误: {e}")
                await asyncio.sleep(1)

    async def _update_market_data(self) -> None:
        """更新市场数据"""
        try:
            # 模拟实时价格数据（实际应用中从WebSocket获取）
            import random
            
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
            market_data = {}
            
            for symbol in symbols:
                base_prices = {
                    "BTC/USDT": 45000,
                    "ETH/USDT": 2800,
                    "ADA/USDT": 1.2,
                    "DOT/USDT": 35,
                    "LINK/USDT": 28
                }
                
                base_price = base_prices.get(symbol, 100)
                current_price = base_price * (1 + random.uniform(-0.02, 0.02))
                change_24h = random.uniform(-0.1, 0.1)
                volume_24h = random.uniform(1000000, 10000000)
                
                market_data[symbol] = {
                    "price": current_price,
                    "change_24h": change_24h,
                    "change_24h_abs": current_price * change_24h,
                    "volume_24h": volume_24h,
                    "high_24h": current_price * (1 + abs(change_24h) * 0.8),
                    "low_24h": current_price * (1 - abs(change_24h) * 0.8),
                    "timestamp": datetime.utcnow()
                }
            
            self.market_data = market_data
            
        except Exception as e:
            logger.error(f"更新市场数据失败: {e}")

    async def _update_portfolio_data(self) -> None:
        """更新投资组合数据"""
        try:
            # 模拟投资组合数据（实际应用中从数据库获取）
            import random
            
            total_value = 10000 + random.uniform(-500, 1000)
            daily_pnl = random.uniform(-200, 400)
            unrealized_pnl = random.uniform(-300, 600)
            
            self.portfolio_data = {
                "total_value": total_value,
                "daily_pnl": daily_pnl,
                "daily_pnl_percent": daily_pnl / total_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl / total_value,
                "available_balance": 3000 + random.uniform(-100, 200),
                "margin_used": random.uniform(2000, 5000),
                "positions_count": random.randint(3, 8),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"更新投资组合数据失败: {e}")

    async def _update_strategy_status(self) -> None:
        """更新策略状态"""
        try:
            # 获取策略引擎状态
            strategy_status = strategy_engine.get_strategy_status()
            
            # 添加统计信息
            active_count = sum(1 for s in strategy_status.values() if s["status"] == "active")
            total_trades = sum(s["trades_count"] for s in strategy_status.values())
            
            # 计算平均收益率
            total_pnl = sum(s["pnl"] for s in strategy_status.values())
            avg_pnl_percent = (total_pnl / 10000) if total_pnl else 0  # 假设总资金10000
            
            self.strategy_status = {
                "strategies": strategy_status,
                "active_count": active_count,
                "total_count": len(strategy_status),
                "total_trades": total_trades,
                "avg_pnl_percent": avg_pnl_percent,
                "success_rate": random.uniform(0.55, 0.75),  # 模拟成功率
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"更新策略状态失败: {e}")

    async def _update_news_data(self) -> None:
        """更新新闻数据"""
        try:
            # 模拟新闻数据（实际应用中从新闻API获取）
            import random
            
            news_templates = [
                "比特币突破{price}美元关键阻力位",
                "以太坊网络升级即将完成，价格看涨",
                "机构投资者大量买入{symbol}，市场情绪转好",
                "监管政策明朗化，加密市场迎来新机遇",
                "DeFi协议锁定价值创新高，生态发展强劲"
            ]
            
            news_data = []
            for i in range(10):
                template = random.choice(news_templates)
                title = template.format(
                    price=random.randint(40000, 50000),
                    symbol=random.choice(["BTC", "ETH", "ADA"])
                )
                
                news_data.append({
                    "title": title,
                    "source": random.choice(["CoinDesk", "CoinTelegraph", "币世界", "金色财经"]),
                    "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(1, 120)),
                    "sentiment": random.choice(["positive", "neutral", "negative"]),
                    "impact": random.choice(["high", "medium", "low"])
                })
            
            self.news_data = sorted(news_data, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"更新新闻数据失败: {e}")

    async def _update_system_status(self) -> None:
        """更新系统状态"""
        try:
            import psutil
            
            # 获取系统性能指标
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # 检查数据库连接（如果数据管理器已初始化）
            db_status = {"mongodb": True, "redis": True}
            if hasattr(data_manager, '_initialized') and data_manager._initialized:
                db_status = await data_manager.health_check()
            
            self.system_status = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "connections": {
                    "okx_websocket": True,
                    "binance_websocket": True,
                    "mongodb": db_status.get("mongodb", False),
                    "redis": db_status.get("redis", False)
                },
                "uptime": datetime.utcnow().strftime("%H:%M:%S"),
                "rust_engine_status": "active",
                "ai_engine_status": "active"
            }
            
        except Exception as e:
            logger.error(f"更新系统状态失败: {e}")

    def _refresh_widgets(self) -> None:
        """刷新所有界面组件"""
        try:
            # 更新市场状态栏
            market_widget = self.query_one("#market-status", MarketStatusWidget)
            market_widget.update_data(self.market_data)
            
            # 更新投资组合小部件
            portfolio_widget = self.query_one("#portfolio-widget", PortfolioWidget)
            portfolio_widget.update_data(self.portfolio_data)
            
            # 更新策略状态小部件
            strategy_widget = self.query_one("#strategy-widget", StrategyStatusWidget)
            strategy_widget.update_data(self.strategy_status)
            
            # 更新价格图表
            price_chart = self.query_one("#price-chart", RealTimePriceChart)
            if "BTC/USDT" in self.market_data:
                btc_data = self.market_data["BTC/USDT"]
                price_chart.add_data_point(btc_data["price"], btc_data["timestamp"])
            
            # 更新监控列表
            watchlist = self.query_one("#watchlist-table", WatchlistTable)
            watchlist.update_data(list(self.market_data.values())[:5])
            
            # 更新涨幅榜
            gainers = self.query_one("#gainers-table", TopGainersTable) 
            sorted_data = sorted(
                self.market_data.values(), 
                key=lambda x: x["change_24h"], 
                reverse=True
            )
            gainers.update_data(sorted_data[:5])
            
            # 更新新闻表格
            news_table = self.query_one("#news-table", NewsTable)
            news_table.update_data(self.news_data[:10])
            
            # 更新系统状态
            system_status_widget = self.query_one("#system-status", Static)
            system_status_widget.update(self._format_system_status())
            
        except Exception as e:
            logger.error(f"刷新界面组件失败: {e}")

    def _format_system_status(self) -> str:
        """格式化系统状态显示"""
        if not self.system_status:
            return "正在加载..."
        
        status = self.system_status
        
        # CPU和内存状态
        cpu_color = "green" if status["cpu_percent"] < 80 else "yellow" if status["cpu_percent"] < 95 else "red"
        memory_color = "green" if status["memory_percent"] < 80 else "yellow" if status["memory_percent"] < 95 else "red"
        
        status_text = f"""[bold]系统性能[/bold]
CPU: [{cpu_color}]{status['cpu_percent']:.1f}%[/{cpu_color}]
内存: [{memory_color}]{status['memory_percent']:.1f}%[/{memory_color}] ({status['memory_used_gb']:.1f}GB/{status['memory_total_gb']:.1f}GB)

[bold]连接状态[/bold]"""
        
        # 连接状态
        for conn_name, conn_status in status["connections"].items():
            color = "green" if conn_status else "red"
            symbol = "✅" if conn_status else "❌"
            status_text += f"\n{symbol} {conn_name}: [{color}]{'已连接' if conn_status else '断开'}[/{color}]"
        
        status_text += f"\n\n[bold]引擎状态[/bold]\n🚀 Rust引擎: [green]{status['rust_engine_status']}[/green]"
        status_text += f"\n🤖 AI引擎: [green]{status['ai_engine_status']}[/green]"
        status_text += f"\n⏰ 运行时间: {status['uptime']}"
        
        return status_text

    # ============ 动作处理器 ============

    def action_refresh(self) -> None:
        """手动刷新数据"""
        try:
            # 重启数据更新任务以立即刷新
            self.stop_data_updates()
            self.start_data_updates()
            logger.info("仪表盘数据手动刷新")
        except Exception as e:
            logger.error(f"手动刷新失败: {e}")

    def action_toggle_fullscreen_chart(self) -> None:
        """切换图表全屏显示"""
        # TODO: 实现图表全屏功能
        pass

    def action_toggle_watchlist(self) -> None:
        """切换监控列表显示"""
        # TODO: 实现监控列表开关
        pass

    # ============ 数据更新处理器 ============

    async def on_data_update(self, data: Dict[str, Any]) -> None:
        """外部数据更新回调"""
        try:
            # 处理来自主应用的数据更新
            if "market_data" in data:
                self.market_data = data["market_data"]
            
            if "portfolio_data" in data:
                self.portfolio_data = data["portfolio_data"]
                
            if "strategy_status" in data:
                self.strategy_status = data["strategy_status"]
            
            # 刷新界面
            self._refresh_widgets()
            
        except Exception as e:
            logger.error(f"外部数据更新处理失败: {e}")