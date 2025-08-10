"""
主仪表盘界面 - Bloomberg风格实时数据展示
提供系统状态、市场行情、策略监控的综合视图
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from rich.console import Group, Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, DataTable, ProgressBar
from textual.screen import Screen
from loguru import logger

from config.bloomberg_theme import BLOOMBERG_COLORS, STATUS_INDICATORS, get_color
from core.data_manager import data_manager
from core.ai_engine import ai_engine
from core.strategy_engine import strategy_engine
from core.websocket_client import websocket_manager

class MarketOverviewWidget(Static):
    """市场概览小组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.market_data = {}
        
    def compose(self) -> ComposeResult:
        yield Static("📊 实时行情", classes="panel-title")
    
    async def update_market_data(self, data: Dict[str, Any]):
        """更新市场数据"""
        self.market_data.update(data)
        await self.refresh_display()
    
    async def refresh_display(self):
        """刷新显示内容"""
        try:
            # 创建市场数据表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("交易对", style="cyan", no_wrap=True)
            table.add_column("价格", style="white", justify="right")
            table.add_column("24h变化", justify="right")
            table.add_column("成交量", style="blue", justify="right")
            
            # 主要币种数据
            symbols = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT"]
            
            for symbol in symbols:
                data = self.market_data.get(f"OKX:{symbol}", {})
                if data:
                    price = f"${float(data.get('price', 0)):.4f}"
                    change_pct = float(data.get('change_24h_pct', 0))
                    volume_24h = data.get('volume_24h', 0)
                    
                    # 根据涨跌设置颜色
                    if change_pct > 0:
                        change_color = "green"
                        change_text = f"+{change_pct:.2f}%"
                    elif change_pct < 0:
                        change_color = "red"  
                        change_text = f"{change_pct:.2f}%"
                    else:
                        change_color = "white"
                        change_text = "0.00%"
                    
                    # 格式化成交量
                    if volume_24h > 1e9:
                        volume_str = f"{volume_24h/1e9:.1f}B"
                    elif volume_24h > 1e6:
                        volume_str = f"{volume_24h/1e6:.1f}M"
                    else:
                        volume_str = f"{volume_24h:.0f}"
                    
                    table.add_row(
                        symbol,
                        price,
                        Text(change_text, style=change_color),
                        volume_str
                    )
                else:
                    table.add_row(symbol, "N/A", "N/A", "N/A")
            
            # 更新显示
            content = Panel(
                Align.center(table),
                title="📊 实时行情",
                border_style="bright_blue",
                padding=(1, 1)
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"市场概览更新失败: {e}")

class AIAnalysisWidget(Static):
    """AI分析小组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sentiment_data = {}
        self.prediction_data = {}
        
    async def update_ai_analysis(self):
        """更新AI分析"""
        try:
            # 获取最新AI分析
            sentiment = await ai_engine.analyze_market_sentiment()
            prediction = await ai_engine.predict_market_movement(["BTC-USDT", "ETH-USDT"])
            
            self.sentiment_data = sentiment
            self.prediction_data = prediction
            
            await self.refresh_display()
            
        except Exception as e:
            logger.error(f"AI分析更新失败: {e}")
    
    async def refresh_display(self):
        """刷新AI分析显示"""
        try:
            # 情绪分析部分
            sentiment_score = self.sentiment_data.get("sentiment_score", 0)
            confidence = self.sentiment_data.get("confidence", 0)
            
            # 根据情绪得分选择emoji和颜色
            if sentiment_score > 0.3:
                sentiment_emoji = "😊"
                sentiment_color = "green"
                sentiment_text = "乐观"
            elif sentiment_score < -0.3:
                sentiment_emoji = "😰"
                sentiment_color = "red"
                sentiment_text = "悲观"
            else:
                sentiment_emoji = "😐"
                sentiment_color = "yellow"
                sentiment_text = "中性"
            
            # 市场预测部分
            trend = self.prediction_data.get("trend_direction", "sideways")
            pred_confidence = self.prediction_data.get("confidence", 0)
            
            trend_map = {
                "up": ("📈", "green", "上涨"),
                "down": ("📉", "red", "下跌"),
                "sideways": ("➡️", "yellow", "横盘")
            }
            trend_emoji, trend_color, trend_text = trend_map.get(trend, ("❓", "white", "未知"))
            
            # 创建AI分析表格
            table = Table(show_header=False, box=None)
            table.add_column("项目", style="cyan")
            table.add_column("状态", justify="center")
            table.add_column("详情", style="white")
            
            table.add_row(
                "市场情绪",
                f"{sentiment_emoji}",
                Text(f"{sentiment_text} ({sentiment_score:.2f})", style=sentiment_color)
            )
            
            table.add_row(
                "AI预测",
                f"{trend_emoji}",
                Text(f"{trend_text} ({pred_confidence:.0%})", style=trend_color)
            )
            
            table.add_row(
                "推荐操作",
                "💡",
                self.sentiment_data.get("recommendation", "观望")
            )
            
            # 更新显示
            content = Panel(
                table,
                title="🤖 AI智能分析",
                border_style="bright_magenta",
                padding=(1, 1)
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"AI分析显示失败: {e}")

class StrategyStatusWidget(Static):
    """策略状态小组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_data = {}
        
    async def update_strategy_status(self):
        """更新策略状态"""
        try:
            self.strategy_data = strategy_engine.get_strategy_status()
            await self.refresh_display()
        except Exception as e:
            logger.error(f"策略状态更新失败: {e}")
    
    async def refresh_display(self):
        """刷新策略状态显示"""
        try:
            # 创建策略状态卡片
            if not self.strategy_data:
                content = Panel(
                    Align.center("暂无运行策略\n\n按 [2] 进入策略管理"),
                    title="🚀 策略状态",
                    border_style="bright_green",
                    padding=(1, 1)
                )
                self.update(content)
                return
            
            # 策略状态统计
            total_strategies = len(self.strategy_data)
            running_strategies = sum(1 for s in self.strategy_data.values() if s["status"] == "active")
            total_trades = sum(s["trades_count"] for s in self.strategy_data.values())
            
            # 创建策略卡片
            strategy_cards = []
            
            for strategy_id, info in self.strategy_data.items():
                status = info["status"]
                name = info["name"]
                position = info["position"]
                pnl = info["pnl"]
                
                # 状态指示器
                status_indicators = {
                    "active": ("🟢", "green"),
                    "paused": ("🟡", "yellow"), 
                    "stopped": ("🔴", "red"),
                    "error": ("❌", "bright_red")
                }
                
                status_icon, status_color = status_indicators.get(status, ("❓", "white"))
                
                # PnL颜色
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_text = f"{pnl:+.2f} USDT"
                
                card_text = f"{status_icon} {name}\n状态: {status}\n仓位: {position:.4f}\nPnL: "
                card_with_pnl = Text(card_text)
                card_with_pnl.append(pnl_text, style=pnl_color)
                
                strategy_cards.append(Panel(
                    card_with_pnl,
                    border_style=status_color,
                    padding=(0, 1)
                ))
            
            # 组合所有策略卡片
            if len(strategy_cards) <= 2:
                cards_layout = Columns(strategy_cards, equal=True, expand=True)
            else:
                # 多于2个策略时分行显示
                cards_layout = Group(
                    Columns(strategy_cards[:2], equal=True, expand=True),
                    Columns(strategy_cards[2:4], equal=True, expand=True) if len(strategy_cards) > 2 else ""
                )
            
            # 顶部统计信息
            stats_text = f"总策略: {total_strategies} | 运行: {running_strategies} | 总交易: {total_trades}"
            
            main_content = Group(
                Align.center(Text(stats_text, style="bold cyan")),
                "",
                cards_layout
            )
            
            content = Panel(
                main_content,
                title="🚀 策略状态",
                border_style="bright_green",
                padding=(1, 1)
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"策略状态显示失败: {e}")

class SystemStatusWidget(Static):
    """系统状态小组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_status = {}
        
    async def update_system_status(self):
        """更新系统状态"""
        try:
            # 获取各组件状态
            db_status = await data_manager.health_check()
            ws_status = websocket_manager.get_connection_status()
            
            self.system_status = {
                "database": db_status,
                "websocket": ws_status,
                "timestamp": datetime.now()
            }
            
            await self.refresh_display()
            
        except Exception as e:
            logger.error(f"系统状态更新失败: {e}")
    
    async def refresh_display(self):
        """刷新系统状态显示"""
        try:
            # 连接状态表格
            table = Table(show_header=False, box=None)
            table.add_column("组件", style="cyan")
            table.add_column("状态", justify="center")
            
            # 数据库状态
            db_status = self.system_status.get("database", {})
            mongodb_status = STATUS_INDICATORS["connected"] if db_status.get("mongodb") else STATUS_INDICATORS["disconnected"]
            redis_status = STATUS_INDICATORS["connected"] if db_status.get("redis") else STATUS_INDICATORS["disconnected"]
            
            table.add_row("MongoDB", mongodb_status)
            table.add_row("Redis", redis_status)
            
            # WebSocket状态
            ws_status = self.system_status.get("websocket", {})
            for exchange, connected in ws_status.items():
                ws_icon = STATUS_INDICATORS["connected"] if connected else STATUS_INDICATORS["disconnected"]
                table.add_row(f"{exchange} WS", ws_icon)
            
            # 最后更新时间
            timestamp = self.system_status.get("timestamp", datetime.now())
            update_time = timestamp.strftime("%H:%M:%S")
            
            content = Panel(
                Group(
                    table,
                    "",
                    Align.center(Text(f"更新: {update_time}", style="dim"))
                ),
                title="🔗 系统状态",
                border_style="bright_blue",
                padding=(1, 1)
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"系统状态显示失败: {e}")

class NewsWidget(Static):
    """新闻和日志小组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_data = []
        self.log_messages = []
        
    async def update_news_data(self):
        """更新新闻数据"""
        try:
            # 获取最新新闻
            recent_news = await data_manager.get_recent_news(hours=2, limit=5)
            self.news_data = recent_news
            await self.refresh_display()
        except Exception as e:
            logger.error(f"新闻数据更新失败: {e}")
    
    def add_log_message(self, message: str, level: str = "info"):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        
        # 保持最新20条消息
        if len(self.log_messages) > 20:
            self.log_messages = self.log_messages[-20:]
    
    async def refresh_display(self):
        """刷新新闻和日志显示"""
        try:
            # 新闻部分
            if self.news_data:
                news_text = []
                for news in self.news_data[:3]:  # 显示3条最新新闻
                    title = news.get("title", "")[:40] + "..." if len(news.get("title", "")) > 40 else news.get("title", "")
                    news_text.append(f"📰 {title}")
            else:
                news_text = ["📰 暂无最新新闻"]
            
            # 系统日志部分
            if self.log_messages:
                log_text = []
                for log in self.log_messages[-5:]:  # 显示最新5条日志
                    level_icons = {
                        "info": "ℹ️",
                        "warning": "⚠️",
                        "error": "❌",
                        "success": "✅",
                        "ai": "🧠"
                    }
                    icon = level_icons.get(log["level"], "📝")
                    log_text.append(f"{log['timestamp']} {icon} {log['message']}")
            else:
                log_text = ["📝 系统启动中..."]
            
            # 组合内容
            content_text = Group(
                Text("📰 财经快讯", style="bold yellow"),
                *[Text(line, style="white") for line in news_text],
                "",
                Text("📝 系统日志", style="bold cyan"),
                *[Text(line, style="dim white") for line in log_text]
            )
            
            content = Panel(
                content_text,
                title="📰 新闻与日志",
                border_style="bright_yellow",
                padding=(1, 1)
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"新闻日志显示失败: {e}")

class DashboardScreen(Screen):
    """主仪表盘屏幕"""
    
    CSS = f"""
    .dashboard-grid {{
        layout: grid;
        grid-size: 3 3;
        grid-gutter: 1;
        margin: 1;
        height: 1fr;
    }}
    
    .market-widget {{
        row-span: 2;
        column-span: 1;
    }}
    
    .ai-widget {{
        row-span: 1;
        column-span: 1;
    }}
    
    .strategy-widget {{
        row-span: 2;
        column-span: 1;
    }}
    
    .system-widget {{
        row-span: 1;
        column-span: 1;
    }}
    
    .news-widget {{
        row-span: 3;
        column-span: 1;
    }}
    
    .footer-widget {{
        row-span: 1;
        column-span: 2;
    }}
    """
    
    def __init__(self):
        super().__init__()
        self.market_widget = None
        self.ai_widget = None
        self.strategy_widget = None
        self.system_widget = None
        self.news_widget = None
        
    def compose(self) -> ComposeResult:
        """构建仪表盘布局"""
        with Container(classes="dashboard-grid"):
            # 第一列：市场概览 (2行)
            self.market_widget = MarketOverviewWidget(classes="market-widget")
            yield self.market_widget
            
            # 第二列：AI分析
            self.ai_widget = AIAnalysisWidget(classes="ai-widget")
            yield self.ai_widget
            
            # 第二列：系统状态  
            self.system_widget = SystemStatusWidget(classes="system-widget")
            yield self.system_widget
            
            # 第三列：策略状态 (2行)
            self.strategy_widget = StrategyStatusWidget(classes="strategy-widget")
            yield self.strategy_widget
            
            # 第三列：新闻日志 (3行)
            self.news_widget = NewsWidget(classes="news-widget")
            yield self.news_widget
    
    async def on_mount(self):
        """仪表盘挂载时初始化"""
        try:
            logger.info("主仪表盘初始化...")
            
            # 设置定时更新
            self.set_interval(0.25, self.update_real_time_data)    # 4Hz实时数据
            self.set_interval(5.0, self.update_system_status)      # 5秒系统状态
            self.set_interval(30.0, self.update_ai_analysis)       # 30秒AI分析
            self.set_interval(60.0, self.update_news_data)         # 60秒新闻数据
            
            # 初始更新
            await self.initial_update()
            
            logger.info("主仪表盘初始化完成")
            
        except Exception as e:
            logger.error(f"仪表盘初始化失败: {e}")
    
    async def initial_update(self):
        """初始数据更新"""
        try:
            # 并行更新所有组件
            await asyncio.gather(
                self.update_system_status(),
                self.update_ai_analysis(),
                self.update_news_data(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"初始数据更新失败: {e}")
    
    async def update_real_time_data(self):
        """更新实时数据 (4Hz)"""
        try:
            # 更新策略状态
            if self.strategy_widget:
                await self.strategy_widget.update_strategy_status()
            
        except Exception as e:
            logger.debug(f"实时数据更新失败: {e}")
    
    async def update_system_status(self):
        """更新系统状态"""
        try:
            if self.system_widget:
                await self.system_widget.update_system_status()
        except Exception as e:
            logger.debug(f"系统状态更新失败: {e}")
    
    async def update_ai_analysis(self):
        """更新AI分析"""
        try:
            if self.ai_widget:
                await self.ai_widget.update_ai_analysis()
        except Exception as e:
            logger.debug(f"AI分析更新失败: {e}")
    
    async def update_news_data(self):
        """更新新闻数据"""
        try:
            if self.news_widget:
                await self.news_widget.update_news_data()
        except Exception as e:
            logger.debug(f"新闻数据更新失败: {e}")
    
    async def on_market_data_update(self, data: Dict[str, Any]):
        """处理市场数据更新"""
        if self.market_widget:
            await self.market_widget.update_market_data(data)
    
    def log_message(self, message: str, level: str = "info"):
        """添加日志消息"""
        if self.news_widget:
            self.news_widget.add_log_message(message, level)

# 创建仪表盘实例的便捷函数
def create_dashboard() -> DashboardScreen:
    """创建仪表盘实例"""
    return DashboardScreen()