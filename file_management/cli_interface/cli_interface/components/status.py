"""
状态显示组件集合

实现Bloomberg Terminal风格的状态显示组件：
- 市场状态栏
- 投资组合小部件
- 策略状态小部件
- 系统状态指示器
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

from textual.widget import Widget
from textual.reactive import reactive
from rich.console import RenderResult
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.align import Align

from ..themes.bloomberg import BloombergTheme
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class StatusBar(Widget):
    """全局状态栏"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.message = ""
        self.message_style = "white"
        self.show_time = True
        
    def show_message(self, message: str, style: str = "white", duration: Optional[float] = None):
        """显示状态消息"""
        self.message = message
        self.message_style = style
        self.refresh()
        
        # TODO: 如果设置了duration，则在指定时间后清除消息
        if duration:
            # 需要实现定时器逻辑
            pass
    
    def render(self) -> RenderResult:
        """渲染状态栏"""
        content = Text()
        
        # 时间显示
        if self.show_time:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content.append(f"🕒 {current_time}", style="dim cyan")
            content.append(" | ", style="dim")
        
        # 状态消息
        if self.message:
            content.append(self.message, style=self.message_style)
        else:
            content.append("就绪", style="green")
        
        return Panel(
            content,
            height=1,
            style=self.theme.STATUS_BAR_STYLE,
            padding=(0, 1)
        )

class MarketStatusWidget(Widget):
    """市场状态小部件"""
    
    market_data = reactive({})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        
    def update_data(self, data: Dict[str, Any]):
        """更新市场数据"""
        self.market_data = data
    
    def render(self) -> RenderResult:
        """渲染市场状态"""
        if not self.market_data:
            return Panel(
                Align.center(Text("📊 连接市场数据...", style="dim")),
                title="[bold cyan]市场状态[/bold cyan]",
                height=5
            )
        
        # 选择主要币种显示
        main_symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        content = Text()
        
        for i, symbol in enumerate(main_symbols):
            if symbol in self.market_data:
                data = self.market_data[symbol]
                price = data.get("price", 0)
                change_24h = data.get("change_24h", 0)
                
                # 选择颜色
                color = "green" if change_24h >= 0 else "red"
                symbol_short = symbol.split("/")[0]
                
                content.append(f"{symbol_short}: ", style="white")
                content.append(f"${price:.2f}", style="bold white")
                content.append(f" ({change_24h:+.2f}%)", style=color)
                
                if i < len(main_symbols) - 1:
                    content.append(" | ", style="dim")
        
        return Panel(
            content,
            title="[bold cyan]📊 市场行情[/bold cyan]",
            height=3,
            border_style="cyan"
        )

class PortfolioWidget(Widget):
    """投资组合小部件"""
    
    portfolio_data = reactive({})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        
    def update_data(self, data: Dict[str, Any]):
        """更新投资组合数据"""
        self.portfolio_data = data
    
    def render(self) -> RenderResult:
        """渲染投资组合状态"""
        if not self.portfolio_data:
            return Panel(
                Align.center(Text("💰 加载投资组合...", style="dim")),
                title="[bold green]投资组合[/bold green]",
                height=5
            )
        
        total_value = self.portfolio_data.get("total_value", 0)
        daily_pnl = self.portfolio_data.get("daily_pnl", 0)
        daily_pnl_percent = self.portfolio_data.get("daily_pnl_percent", 0)
        unrealized_pnl = self.portfolio_data.get("unrealized_pnl", 0)
        
        # 内容构建
        content = Text()
        content.append("总资产: ", style="white")
        content.append(f"${total_value:,.2f}", style="bold white")
        content.append("\n")
        
        # 日盈亏
        pnl_color = "green" if daily_pnl >= 0 else "red"
        content.append("日盈亏: ", style="white") 
        content.append(f"${daily_pnl:+,.2f}", style=pnl_color)
        content.append(f" ({daily_pnl_percent:+.2f}%)", style=pnl_color)
        content.append("\n")
        
        # 浮动盈亏
        unrealized_color = "green" if unrealized_pnl >= 0 else "red"
        content.append("浮动: ", style="dim")
        content.append(f"${unrealized_pnl:+,.2f}", style=unrealized_color)
        
        return Panel(
            content,
            title="[bold green]💰 投资组合[/bold green]",
            height=5,
            border_style="green"
        )

class StrategyStatusWidget(Widget):
    """策略状态小部件"""
    
    strategy_data = reactive({})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        
    def update_data(self, data: Dict[str, Any]):
        """更新策略数据"""
        self.strategy_data = data
    
    def render(self) -> RenderResult:
        """渲染策略状态"""
        if not self.strategy_data:
            return Panel(
                Align.center(Text("🤖 加载策略状态...", style="dim")),
                title="[bold yellow]策略状态[/bold yellow]",
                height=5
            )
        
        active_count = self.strategy_data.get("active_count", 0)
        total_count = self.strategy_data.get("total_count", 0)
        total_trades = self.strategy_data.get("total_trades", 0)
        success_rate = self.strategy_data.get("success_rate", 0)
        avg_pnl_percent = self.strategy_data.get("avg_pnl_percent", 0)
        
        # 内容构建
        content = Text()
        content.append("运行中: ", style="white")
        content.append(f"{active_count}/{total_count}", style="bold yellow")
        content.append(" 策略", style="white")
        content.append("\n")
        
        content.append("交易次数: ", style="white")
        content.append(f"{total_trades}", style="bold white")
        content.append("\n")
        
        # 成功率和平均收益
        success_color = "green" if success_rate > 0.5 else "yellow" if success_rate > 0.4 else "red"
        pnl_color = "green" if avg_pnl_percent >= 0 else "red"
        
        content.append("成功率: ", style="dim")
        content.append(f"{success_rate:.1%}", style=success_color)
        content.append(" | ", style="dim")
        content.append("收益: ", style="dim")
        content.append(f"{avg_pnl_percent:+.2f}%", style=pnl_color)
        
        return Panel(
            content,
            title="[bold yellow]🤖 策略引擎[/bold yellow]",
            height=5,
            border_style="yellow"
        )

class SystemStatusIndicator(Widget):
    """系统状态指示器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.connections = {}
        self.system_metrics = {}
        
    def update_connections(self, connections: Dict[str, bool]):
        """更新连接状态"""
        self.connections = connections
        self.refresh()
        
    def update_system_metrics(self, metrics: Dict[str, Any]):
        """更新系统指标"""
        self.system_metrics = metrics
        self.refresh()
    
    def render(self) -> RenderResult:
        """渲染系统状态"""
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("", style="dim")
        table.add_column("", style="white")
        table.add_column("", style="dim")
        
        # 连接状态
        if self.connections:
            table.add_row("", "[bold]连接状态[/bold]", "")
            for name, status in self.connections.items():
                color = "green" if status else "red"
                symbol = "●" if status else "○"
                table.add_row("", f"[{color}]{symbol}[/{color}] {name}", "")
        
        # 系统指标
        if self.system_metrics:
            table.add_row("", "", "")
            table.add_row("", "[bold]系统负载[/bold]", "")
            
            cpu = self.system_metrics.get("cpu_percent", 0)
            memory = self.system_metrics.get("memory_percent", 0)
            
            cpu_color = "green" if cpu < 70 else "yellow" if cpu < 90 else "red"
            memory_color = "green" if memory < 70 else "yellow" if memory < 90 else "red"
            
            table.add_row("", f"CPU: [{cpu_color}]{cpu:.1f}%[/{cpu_color}]", "")
            table.add_row("", f"内存: [{memory_color}]{memory:.1f}%[/{memory_color}]", "")
        
        return Panel(
            table,
            title="[bold]⚙️ 系统状态[/bold]",
            height=8,
            border_style="blue"
        )

class PerformanceIndicator(Widget):
    """性能指示器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.latency_ms = 0
        self.throughput_per_sec = 0
        self.error_count = 0
        self.uptime_seconds = 0
        
    def update_metrics(self, 
                      latency: float = None,
                      throughput: int = None,
                      errors: int = None,
                      uptime: int = None):
        """更新性能指标"""
        if latency is not None:
            self.latency_ms = latency
        if throughput is not None:
            self.throughput_per_sec = throughput
        if errors is not None:
            self.error_count = errors
        if uptime is not None:
            self.uptime_seconds = uptime
        
        self.refresh()
    
    def render(self) -> RenderResult:
        """渲染性能指标"""
        content = Text()
        
        # 延迟指标
        latency_color = "green" if self.latency_ms < 50 else "yellow" if self.latency_ms < 100 else "red"
        content.append("延迟: ", style="dim")
        content.append(f"{self.latency_ms:.1f}ms", style=latency_color)
        content.append(" | ", style="dim")
        
        # 吞吐量
        throughput_color = "green" if self.throughput_per_sec > 100 else "yellow"
        content.append("处理: ", style="dim")
        content.append(f"{self.throughput_per_sec}/s", style=throughput_color)
        content.append("\n")
        
        # 错误计数
        error_color = "green" if self.error_count == 0 else "yellow" if self.error_count < 10 else "red"
        content.append("错误: ", style="dim")
        content.append(f"{self.error_count}", style=error_color)
        content.append(" | ", style="dim")
        
        # 运行时间
        uptime_hours = self.uptime_seconds // 3600
        uptime_minutes = (self.uptime_seconds % 3600) // 60
        content.append("运行: ", style="dim")
        content.append(f"{uptime_hours:02d}:{uptime_minutes:02d}", style="cyan")
        
        return Panel(
            content,
            title="[bold]📊 性能[/bold]",
            height=4,
            border_style="cyan"
        )