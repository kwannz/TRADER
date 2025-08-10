"""
表格组件集合

实现Bloomberg Terminal风格的数据表格组件：
- 监控列表表格
- 涨幅榜表格  
- 交易记录表格
- 新闻表格
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from textual.widgets import DataTable
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table as RichTable
from rich.console import RenderResult
from rich.panel import Panel

from ..themes.bloomberg import BloombergTheme
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class WatchlistTable(DataTable):
    """监控列表表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "代码", "价格", "涨跌", "涨幅%", "24H量", "更新"
        )
        
    def update_data(self, data: List[Dict[str, Any]]):
        """更新监控列表数据"""
        try:
            # 清空现有数据
            self.clear()
            
            # 添加新数据行
            for item in data:
                symbol = item.get("symbol", "N/A")
                price = item.get("price", 0)
                change_24h = item.get("change_24h", 0)
                change_24h_abs = item.get("change_24h_abs", 0)
                volume_24h = item.get("volume_24h", 0)
                timestamp = item.get("timestamp", datetime.utcnow())
                
                # 格式化数据
                symbol_short = symbol.split("/")[0] if "/" in symbol else symbol
                price_str = f"${price:.2f}" if price >= 1 else f"${price:.4f}"
                change_str = f"{change_24h_abs:+.2f}"
                change_pct_str = f"{change_24h*100:+.2f}%"
                volume_str = f"{volume_24h/1000000:.1f}M" if volume_24h > 1000000 else f"{volume_24h/1000:.0f}K"
                time_str = timestamp.strftime("%H:%M") if isinstance(timestamp, datetime) else "N/A"
                
                # 添加行，使用Rich样式
                self.add_row(
                    Text(symbol_short, style="bold cyan"),
                    Text(price_str, style="bold white"),
                    Text(change_str, style="green" if change_24h_abs >= 0 else "red"),
                    Text(change_pct_str, style="green" if change_24h >= 0 else "red"),
                    Text(volume_str, style="dim yellow"),
                    Text(time_str, style="dim")
                )
                
        except Exception as e:
            logger.error(f"更新监控列表失败: {e}")

class TopGainersTable(DataTable):
    """涨幅榜表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "排名", "代码", "涨幅%", "价格", "成交量"
        )
        
    def update_data(self, data: List[Dict[str, Any]]):
        """更新涨幅榜数据"""
        try:
            # 清空现有数据
            self.clear()
            
            # 添加排序后的数据
            for i, item in enumerate(data, 1):
                symbol = item.get("symbol", "N/A")
                price = item.get("price", 0)
                change_24h = item.get("change_24h", 0)
                volume_24h = item.get("volume_24h", 0)
                
                # 格式化数据
                rank_str = f"#{i}"
                symbol_short = symbol.split("/")[0] if "/" in symbol else symbol
                price_str = f"${price:.2f}" if price >= 1 else f"${price:.4f}"
                change_pct_str = f"{change_24h*100:+.2f}%"
                volume_str = f"{volume_24h/1000000:.1f}M" if volume_24h > 1000000 else f"{volume_24h/1000:.0f}K"
                
                # 选择涨幅颜色
                change_color = "bright_green" if change_24h > 0.05 else "green" if change_24h > 0 else "red"
                
                self.add_row(
                    Text(rank_str, style="dim yellow"),
                    Text(symbol_short, style="bold cyan"),
                    Text(change_pct_str, style=change_color),
                    Text(price_str, style="white"),
                    Text(volume_str, style="dim")
                )
                
        except Exception as e:
            logger.error(f"更新涨幅榜失败: {e}")

class TradeHistoryTable(DataTable):
    """交易历史表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "时间", "代码", "方向", "价格", "数量", "盈亏", "状态"
        )
        
    def update_data(self, trades: List[Dict[str, Any]]):
        """更新交易历史数据"""
        try:
            # 清空现有数据
            self.clear()
            
            for trade in trades:
                timestamp = trade.get("timestamp", datetime.utcnow())
                symbol = trade.get("symbol", "N/A")
                side = trade.get("side", "N/A")
                price = trade.get("price", 0)
                quantity = trade.get("quantity", 0)
                pnl = trade.get("pnl", 0)
                status = trade.get("status", "pending")
                
                # 格式化数据
                time_str = timestamp.strftime("%H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp)
                symbol_short = symbol.split("/")[0] if "/" in symbol else symbol
                side_str = "买入" if side == "buy" else "卖出" if side == "sell" else side
                price_str = f"${price:.2f}" if price >= 1 else f"${price:.4f}"
                quantity_str = f"{quantity:.4f}"
                pnl_str = f"{pnl:+.2f}"
                status_str = {"filled": "已成交", "pending": "待成交", "cancelled": "已取消"}.get(status, status)
                
                # 选择颜色
                side_color = "green" if side == "buy" else "red"
                pnl_color = "green" if pnl >= 0 else "red"
                status_color = "green" if status == "filled" else "yellow" if status == "pending" else "red"
                
                self.add_row(
                    Text(time_str, style="dim"),
                    Text(symbol_short, style="cyan"),
                    Text(side_str, style=side_color),
                    Text(price_str, style="white"),
                    Text(quantity_str, style="white"),
                    Text(pnl_str, style=pnl_color),
                    Text(status_str, style=status_color)
                )
                
        except Exception as e:
            logger.error(f"更新交易历史失败: {e}")

class NewsTable(DataTable):
    """新闻表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "时间", "标题", "来源", "影响"
        )
        
    def update_data(self, news_list: List[Dict[str, Any]]):
        """更新新闻数据"""
        try:
            # 清空现有数据
            self.clear()
            
            for news in news_list:
                timestamp = news.get("timestamp", datetime.utcnow())
                title = news.get("title", "无标题")
                source = news.get("source", "未知")
                impact = news.get("impact", "low")
                sentiment = news.get("sentiment", "neutral")
                
                # 格式化数据
                time_str = timestamp.strftime("%H:%M") if isinstance(timestamp, datetime) else str(timestamp)
                
                # 截断标题长度
                title_display = title[:50] + "..." if len(title) > 50 else title
                
                # 影响级别显示
                impact_symbols = {"high": "🔥", "medium": "⚠️", "low": "ℹ️"}
                impact_display = impact_symbols.get(impact, "ℹ️")
                
                # 情绪颜色
                sentiment_colors = {"positive": "green", "negative": "red", "neutral": "white"}
                title_color = sentiment_colors.get(sentiment, "white")
                
                self.add_row(
                    Text(time_str, style="dim cyan"),
                    Text(title_display, style=title_color),
                    Text(source, style="dim yellow"),
                    Text(impact_display, style="white")
                )
                
        except Exception as e:
            logger.error(f"更新新闻数据失败: {e}")

class StrategyTable(DataTable):
    """策略管理表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "名称", "类型", "状态", "收益率", "交易次数", "最后信号", "操作"
        )
        
    def update_data(self, strategies: Dict[str, Dict[str, Any]]):
        """更新策略数据"""
        try:
            # 清空现有数据
            self.clear()
            
            for strategy_id, strategy in strategies.items():
                name = strategy.get("name", "未命名")
                strategy_type = strategy.get("type", "unknown")
                status = strategy.get("status", "draft")
                pnl = strategy.get("pnl", 0)
                trades_count = strategy.get("trades_count", 0)
                last_signal_time = strategy.get("last_signal_time")
                
                # 格式化数据
                name_display = name[:15] + "..." if len(name) > 15 else name
                
                # 策略类型中文化
                type_map = {"grid": "网格", "dca": "定投", "ai_generated": "AI策略"}
                type_display = type_map.get(strategy_type, strategy_type)
                
                # 状态中文化和颜色
                status_map = {
                    "draft": ("草稿", "dim"),
                    "active": ("运行中", "green"),
                    "paused": ("已暂停", "yellow"),
                    "stopped": ("已停止", "red"),
                    "error": ("错误", "bright_red")
                }
                status_text, status_color = status_map.get(status, (status, "white"))
                
                # 收益率
                pnl_percent = (pnl / 10000 * 100) if pnl != 0 else 0  # 假设初始资金10000
                pnl_str = f"{pnl_percent:+.2f}%"
                pnl_color = "green" if pnl_percent >= 0 else "red"
                
                # 最后信号时间
                if last_signal_time:
                    if isinstance(last_signal_time, str):
                        try:
                            last_time = datetime.fromisoformat(last_signal_time.replace('Z', '+00:00'))
                            time_str = last_time.strftime("%H:%M:%S")
                        except:
                            time_str = "无效时间"
                    else:
                        time_str = last_signal_time.strftime("%H:%M:%S")
                else:
                    time_str = "无"
                
                self.add_row(
                    Text(name_display, style="bold cyan"),
                    Text(type_display, style="yellow"),
                    Text(status_text, style=status_color),
                    Text(pnl_str, style=pnl_color),
                    Text(str(trades_count), style="white"),
                    Text(time_str, style="dim"),
                    Text("管理", style="blue underline")  # 操作按钮
                )
                
        except Exception as e:
            logger.error(f"更新策略表格失败: {e}")

class FactorTable(DataTable):
    """因子分析表格"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        self.cursor_type = "row"
        
        # 设置表格列
        self.add_columns(
            "因子名称", "IC均值", "IC_IR", "最大回撤", "年化收益", "夏普比率", "状态"
        )
        
    def update_data(self, factors: List[Dict[str, Any]]):
        """更新因子数据"""
        try:
            # 清空现有数据
            self.clear()
            
            for factor in factors:
                name = factor.get("name", "未命名")
                ic_mean = factor.get("ic_mean", 0)
                ic_ir = factor.get("ic_ir", 0)
                max_drawdown = factor.get("max_drawdown", 0)
                annual_return = factor.get("annual_return", 0)
                sharpe_ratio = factor.get("sharpe_ratio", 0)
                status = factor.get("status", "testing")
                
                # 格式化数据
                name_display = name[:20] + "..." if len(name) > 20 else name
                ic_mean_str = f"{ic_mean:.3f}"
                ic_ir_str = f"{ic_ir:.2f}"
                drawdown_str = f"{max_drawdown:.1%}"
                return_str = f"{annual_return:.1%}"
                sharpe_str = f"{sharpe_ratio:.2f}"
                
                # 状态颜色
                status_colors = {
                    "active": "green",
                    "testing": "yellow", 
                    "failed": "red",
                    "archived": "dim"
                }
                status_color = status_colors.get(status, "white")
                
                # IC均值颜色（绝对值越大越好）
                ic_color = "green" if abs(ic_mean) > 0.05 else "yellow" if abs(ic_mean) > 0.02 else "red"
                
                # 夏普比率颜色
                sharpe_color = "green" if sharpe_ratio > 1.5 else "yellow" if sharpe_ratio > 1.0 else "red"
                
                self.add_row(
                    Text(name_display, style="bold cyan"),
                    Text(ic_mean_str, style=ic_color),
                    Text(ic_ir_str, style="white"),
                    Text(drawdown_str, style="red" if max_drawdown > 0.1 else "yellow"),
                    Text(return_str, style="green" if annual_return > 0 else "red"),
                    Text(sharpe_str, style=sharpe_color),
                    Text(status, style=status_color)
                )
                
        except Exception as e:
            logger.error(f"更新因子表格失败: {e}")

class CustomDataTable(Widget):
    """自定义数据表格基类"""
    
    def __init__(self, 
                 title: str = "",
                 columns: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.columns = columns or []
        self.data = []
        self.theme = BloombergTheme()
        
    def set_data(self, data: List[Dict[str, Any]]):
        """设置表格数据"""
        self.data = data
        self.refresh()
    
    def add_row(self, row_data: Dict[str, Any]):
        """添加行数据"""
        self.data.append(row_data)
        self.refresh()
        
    def clear_data(self):
        """清空数据"""
        self.data = []
        self.refresh()
    
    def render(self) -> RenderResult:
        """渲染表格"""
        if not self.columns:
            return Text("表格未配置列", style="red")
        
        table = RichTable(
            title=self.title if self.title else None,
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=None
        )
        
        # 添加列
        for column in self.columns:
            table.add_column(column, style="white")
        
        # 添加数据行
        for row_data in self.data:
            row_values = []
            for column in self.columns:
                value = row_data.get(column, "")
                if isinstance(value, Text):
                    row_values.append(value)
                else:
                    row_values.append(str(value))
            table.add_row(*row_values)
        
        return Panel(table, expand=True)