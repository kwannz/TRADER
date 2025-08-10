"""
图表组件集合

实现Bloomberg Terminal风格的实时图表组件：
- 实时价格走势图
- 交易量柱状图  
- 投资组合收益曲线
- 技术指标图表
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import math

from textual.widget import Widget
from textual.reactive import reactive
from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.bar import Bar
from rich.align import Align

from ..themes.bloomberg import BloombergTheme
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class BaseChart(Widget):
    """图表基类"""
    
    def __init__(self, 
                 title: str = "",
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.chart_width = width or 80
        self.chart_height = height or 20
        self.theme = BloombergTheme()
        
    def render(self) -> Panel:
        """渲染图表面板"""
        content = self._render_chart_content()
        
        return Panel(
            content,
            title=self.title,
            border_style=self.theme.CHART_BORDER_STYLE,
            padding=(0, 1),
            expand=False
        )
    
    def _render_chart_content(self) -> str:
        """子类需要实现的图表内容渲染方法"""
        return "图表内容"

class RealTimePriceChart(BaseChart):
    """实时价格走势图"""
    
    def __init__(self, 
                 symbol: str = "BTC/USDT",
                 timeframe: str = "1m",
                 max_points: int = 120,
                 **kwargs):
        super().__init__(title=f"📈 {symbol} - {timeframe}", **kwargs)
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_points = max_points
        
        # 数据存储
        self.price_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 图表状态
        self.current_price = 0.0
        self.price_change = 0.0
        self.price_change_percent = 0.0
        
    def add_data_point(self, price: float, timestamp: datetime) -> None:
        """添加新的价格数据点"""
        try:
            old_price = self.current_price if self.current_price else price
            
            self.price_data.append(price)
            self.timestamps.append(timestamp)
            
            # 更新当前价格和变化
            self.current_price = price
            self.price_change = price - old_price
            self.price_change_percent = (self.price_change / old_price * 100) if old_price else 0
            
            # 触发重新渲染
            self.refresh()
            
        except Exception as e:
            logger.error(f"添加价格数据点失败: {e}")
    
    def _render_chart_content(self) -> str:
        """渲染价格走势图"""
        if len(self.price_data) < 2:
            return self._render_no_data()
        
        try:
            # 计算图表参数
            prices = list(self.price_data)
            min_price = min(prices)
            max_price = max(prices)
            price_range = max_price - min_price or 1
            
            # 生成ASCII图表
            chart_lines = []
            
            # 标题行：显示当前价格和变化
            change_color = "green" if self.price_change >= 0 else "red"
            change_symbol = "▲" if self.price_change >= 0 else "▼"
            
            header_line = f"[bold cyan]{self.symbol}[/bold cyan] "
            header_line += f"[bold white]{self.current_price:.2f}[/bold white] "
            header_line += f"[{change_color}]{change_symbol} {self.price_change:+.2f} ({self.price_change_percent:+.2f}%)[/{change_color}]"
            chart_lines.append(header_line)
            chart_lines.append("")
            
            # 绘制价格曲线
            chart_height = 15
            chart_width = min(len(prices), 80)
            
            # 创建价格网格
            for row in range(chart_height):
                line = ""
                y_value = max_price - (row / (chart_height - 1)) * price_range
                
                # 绘制Y轴标签
                line += f"[dim]{y_value:8.2f}[/dim] │"
                
                # 绘制价格线
                for col in range(chart_width):
                    if col < len(prices):
                        price = prices[col]
                        # 计算价格在图表中的Y位置
                        normalized_price = (price - min_price) / price_range
                        price_row = chart_height - 1 - int(normalized_price * (chart_height - 1))
                        
                        if price_row == row:
                            # 根据价格趋势选择颜色
                            if col > 0 and prices[col] > prices[col-1]:
                                line += "[green]●[/green]"
                            elif col > 0 and prices[col] < prices[col-1]:
                                line += "[red]●[/red]"
                            else:
                                line += "[yellow]●[/yellow]"
                        else:
                            line += " "
                    else:
                        line += " "
                
                chart_lines.append(line)
            
            # 添加X轴
            x_axis = "         └" + "─" * chart_width
            chart_lines.append("[dim]" + x_axis + "[/dim]")
            
            # 添加时间标签
            if len(self.timestamps) > 0:
                latest_time = self.timestamps[-1].strftime("%H:%M:%S")
                earliest_time = self.timestamps[0].strftime("%H:%M:%S") if len(self.timestamps) > 1 else latest_time
                time_line = f"[dim]          {earliest_time}" + " " * (chart_width - 16) + f"{latest_time}[/dim]"
                chart_lines.append(time_line)
            
            return "\n".join(chart_lines)
            
        except Exception as e:
            logger.error(f"渲染价格图表失败: {e}")
            return f"图表渲染错误: {e}"
    
    def _render_no_data(self) -> str:
        """渲染无数据状态"""
        return Align.center(Text(
            "📊 等待市场数据...\n正在连接实时数据源",
            style="dim"
        ))

class VolumeChart(BaseChart):
    """交易量柱状图"""
    
    def __init__(self, 
                 symbol: str = "BTC/USDT",
                 max_bars: int = 24,
                 **kwargs):
        super().__init__(title=f"📊 {symbol} - 交易量", **kwargs)
        self.symbol = symbol
        self.max_bars = max_bars
        
        # 数据存储
        self.volume_data = deque(maxlen=max_bars)
        self.timestamps = deque(maxlen=max_bars)
        
    def add_volume_data(self, volume: float, timestamp: datetime) -> None:
        """添加交易量数据"""
        try:
            self.volume_data.append(volume)
            self.timestamps.append(timestamp)
            self.refresh()
        except Exception as e:
            logger.error(f"添加交易量数据失败: {e}")
    
    def _render_chart_content(self) -> str:
        """渲染交易量柱状图"""
        if not self.volume_data:
            return "📊 等待交易量数据..."
        
        try:
            volumes = list(self.volume_data)
            max_volume = max(volumes) or 1
            
            chart_lines = []
            
            # 标题行
            avg_volume = sum(volumes) / len(volumes)
            latest_volume = volumes[-1]
            
            header = f"[bold cyan]{self.symbol}[/bold cyan] "
            header += f"[bold white]当前: {latest_volume:,.0f}[/bold white] "
            header += f"[dim]平均: {avg_volume:,.0f}[/dim]"
            chart_lines.append(header)
            chart_lines.append("")
            
            # 绘制柱状图
            chart_height = 12
            for row in range(chart_height):
                line = f"[dim]{max_volume * (1 - row/chart_height):8.0f}[/dim] │"
                
                for vol in volumes:
                    bar_height = int((vol / max_volume) * chart_height)
                    if bar_height > (chart_height - 1 - row):
                        # 根据交易量大小选择颜色
                        if vol > avg_volume * 1.5:
                            line += "[bold red]█[/bold red]"
                        elif vol > avg_volume:
                            line += "[yellow]█[/yellow]"
                        else:
                            line += "[green]█[/green]"
                    else:
                        line += " "
                
                chart_lines.append(line)
            
            # X轴
            x_axis = "         └" + "─" * len(volumes)
            chart_lines.append("[dim]" + x_axis + "[/dim]")
            
            return "\n".join(chart_lines)
            
        except Exception as e:
            logger.error(f"渲染交易量图表失败: {e}")
            return f"图表渲染错误: {e}"

class PerformanceChart(BaseChart):
    """投资组合收益曲线图"""
    
    def __init__(self, 
                 period: str = "1d",
                 max_points: int = 100,
                 **kwargs):
        super().__init__(title=f"📈 投资组合收益 - {period}", **kwargs)
        self.period = period
        self.max_points = max_points
        
        # 数据存储
        self.pnl_data = deque(maxlen=max_points)
        self.equity_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 统计数据
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
    def add_performance_data(self, 
                           equity: float, 
                           pnl: float, 
                           timestamp: datetime) -> None:
        """添加收益数据"""
        try:
            self.equity_data.append(equity)
            self.pnl_data.append(pnl)
            self.timestamps.append(timestamp)
            
            # 更新统计指标
            self._update_stats()
            self.refresh()
            
        except Exception as e:
            logger.error(f"添加收益数据失败: {e}")
    
    def _update_stats(self) -> None:
        """更新统计指标"""
        try:
            if len(self.equity_data) < 2:
                return
            
            equities = list(self.equity_data)
            initial_equity = equities[0]
            current_equity = equities[-1]
            
            # 总收益率
            self.total_return = (current_equity - initial_equity) / initial_equity * 100
            
            # 最大回撤
            peak = equities[0]
            max_drawdown = 0
            
            for equity in equities:
                if equity > peak:
                    peak = equity
                else:
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            self.max_drawdown = max_drawdown * 100
            
            # 简化夏普比率计算
            if len(equities) > 10:
                returns = [(equities[i] - equities[i-1]) / equities[i-1] 
                          for i in range(1, len(equities))]
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
                self.sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0
            
        except Exception as e:
            logger.error(f"更新统计指标失败: {e}")
    
    def _render_chart_content(self) -> str:
        """渲染收益曲线图"""
        if not self.equity_data:
            return "📈 等待收益数据..."
        
        try:
            equities = list(self.equity_data)
            
            chart_lines = []
            
            # 统计信息标题
            return_color = "green" if self.total_return >= 0 else "red"
            header = f"[bold white]总收益: [{return_color}]{self.total_return:+.2f}%[/{return_color}][/bold white] "
            header += f"[dim]最大回撤: {self.max_drawdown:.2f}%[/dim] "
            header += f"[dim]夏普: {self.sharpe_ratio:.2f}[/dim]"
            chart_lines.append(header)
            chart_lines.append("")
            
            # 绘制收益曲线
            if len(equities) >= 2:
                min_equity = min(equities)
                max_equity = max(equities)
                equity_range = max_equity - min_equity or 1
                
                chart_height = 12
                chart_width = min(len(equities), 60)
                
                for row in range(chart_height):
                    line = ""
                    y_value = max_equity - (row / (chart_height - 1)) * equity_range
                    
                    # Y轴标签
                    line += f"[dim]{y_value:8.0f}[/dim] │"
                    
                    # 绘制曲线
                    for col in range(chart_width):
                        if col < len(equities):
                            equity = equities[col]
                            normalized_equity = (equity - min_equity) / equity_range
                            equity_row = chart_height - 1 - int(normalized_equity * (chart_height - 1))
                            
                            if equity_row == row:
                                # 根据盈亏情况选择颜色
                                if col > 0:
                                    if equities[col] > equities[col-1]:
                                        line += "[green]●[/green]"
                                    elif equities[col] < equities[col-1]:
                                        line += "[red]●[/red]"
                                    else:
                                        line += "[yellow]●[/yellow]"
                                else:
                                    line += "[cyan]●[/cyan]"
                            else:
                                line += " "
                        else:
                            line += " "
                    
                    chart_lines.append(line)
                
                # X轴
                x_axis = "         └" + "─" * chart_width
                chart_lines.append("[dim]" + x_axis + "[/dim]")
                
            return "\n".join(chart_lines)
            
        except Exception as e:
            logger.error(f"渲染收益图表失败: {e}")
            return f"图表渲染错误: {e}"

class IndicatorChart(BaseChart):
    """技术指标图表"""
    
    def __init__(self, 
                 indicator_name: str = "RSI",
                 symbol: str = "BTC/USDT",
                 max_points: int = 100,
                 **kwargs):
        super().__init__(title=f"📊 {symbol} - {indicator_name}", **kwargs)
        self.indicator_name = indicator_name
        self.symbol = symbol
        self.max_points = max_points
        
        # 指标数据存储
        self.indicator_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 指标参数
        self.upper_bound = 100 if indicator_name == "RSI" else None
        self.lower_bound = 0 if indicator_name == "RSI" else None
        self.overbought_level = 70 if indicator_name == "RSI" else None
        self.oversold_level = 30 if indicator_name == "RSI" else None
        
    def add_indicator_data(self, value: float, timestamp: datetime) -> None:
        """添加技术指标数据"""
        try:
            self.indicator_data.append(value)
            self.timestamps.append(timestamp)
            self.refresh()
        except Exception as e:
            logger.error(f"添加指标数据失败: {e}")
    
    def _render_chart_content(self) -> str:
        """渲染技术指标图表"""
        if not self.indicator_data:
            return f"📊 等待{self.indicator_name}数据..."
        
        try:
            values = list(self.indicator_data)
            current_value = values[-1]
            
            chart_lines = []
            
            # 指标状态标题
            status_color = self._get_indicator_color(current_value)
            status_text = self._get_indicator_status(current_value)
            
            header = f"[bold cyan]{self.indicator_name}[/bold cyan] "
            header += f"[bold white]{current_value:.2f}[/bold white] "
            header += f"[{status_color}]{status_text}[/{status_color}]"
            chart_lines.append(header)
            chart_lines.append("")
            
            # 绘制指标曲线
            min_val = self.lower_bound if self.lower_bound is not None else min(values)
            max_val = self.upper_bound if self.upper_bound is not None else max(values)
            val_range = max_val - min_val or 1
            
            chart_height = 10
            chart_width = min(len(values), 60)
            
            for row in range(chart_height):
                line = ""
                y_value = max_val - (row / (chart_height - 1)) * val_range
                
                # Y轴标签和参考线
                line += f"[dim]{y_value:6.1f}[/dim] │"
                
                # 绘制超买/超卖线
                if (self.overbought_level and abs(y_value - self.overbought_level) < val_range * 0.05):
                    line = line[:-1] + "[red]┤[/red]"
                elif (self.oversold_level and abs(y_value - self.oversold_level) < val_range * 0.05):
                    line = line[:-1] + "[green]┤[/green]"
                
                # 绘制指标线
                for col in range(chart_width):
                    if col < len(values):
                        value = values[col]
                        normalized_value = (value - min_val) / val_range
                        value_row = chart_height - 1 - int(normalized_value * (chart_height - 1))
                        
                        if value_row == row:
                            color = self._get_indicator_color(value)
                            line += f"[{color}]●[/{color}]"
                        else:
                            line += " "
                    else:
                        line += " "
                
                chart_lines.append(line)
            
            # X轴
            x_axis = "        └" + "─" * chart_width
            chart_lines.append("[dim]" + x_axis + "[/dim]")
            
            return "\n".join(chart_lines)
            
        except Exception as e:
            logger.error(f"渲染指标图表失败: {e}")
            return f"图表渲染错误: {e}"
    
    def _get_indicator_color(self, value: float) -> str:
        """根据指标值获取颜色"""
        if self.indicator_name == "RSI":
            if value >= 70:
                return "red"
            elif value <= 30:
                return "green"
            else:
                return "yellow"
        return "cyan"
    
    def _get_indicator_status(self, value: float) -> str:
        """根据指标值获取状态文本"""
        if self.indicator_name == "RSI":
            if value >= 70:
                return "超买"
            elif value <= 30:
                return "超卖"
            else:
                return "正常"
        return "正常"