"""
AI助手屏幕

Bloomberg Terminal风格的AI智能助手界面，支持：
- 市场分析和预测
- 交易建议和风险评估
- 智能问答和策略优化
- 实时市场情绪分析
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Static, Button, Input, TextArea, Label, Select,
    TabbedContent, TabPane, ProgressBar, Log
)
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich.markdown import Markdown

from ..components.charts import IndicatorChart
from ..components.status import PerformanceIndicator
from ..themes.bloomberg import BloombergTheme
from ...core.ai_engine import ai_engine
from ...core.data_manager import data_manager
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class AIAssistantScreen(Screen):
    """AI智能助手主屏幕"""
    
    CSS_PATH = "ai_assistant.css"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "返回"),
        Binding("enter", "send_message", "发送消息"),
        Binding("ctrl+r", "refresh_analysis", "刷新分析"),
        Binding("ctrl+c", "clear_chat", "清空对话"),
    ]
    
    # 响应式数据
    market_analysis = reactive({})
    trading_suggestions = reactive([])
    sentiment_analysis = reactive({})
    chat_history = reactive([])
    
    def __init__(self):
        super().__init__()
        self.theme = BloombergTheme()
        self.refresh_interval = 30.0  # 30秒刷新一次AI分析
        self.update_task: Optional[asyncio.Task] = None
        self.is_processing = False
        
    def compose(self) -> ComposeResult:
        """构建AI助手界面"""
        
        with Container(id="ai-assistant-container"):
            # 顶部状态栏
            with Horizontal(id="ai-status-bar", classes="status-bar"):
                yield Label("🤖 AI智能助手", classes="title")
                yield PerformanceIndicator(id="ai-performance")
                yield Button("🔄 刷新分析", id="btn-refresh-analysis")
                yield Button("🧹 清空对话", id="btn-clear-chat")
            
            # 主内容区域
            with TabbedContent(id="ai-tabs"):
                # 智能对话页
                with TabPane("智能对话", id="chat-tab"):
                    with Horizontal(id="chat-section"):
                        # 左侧对话区
                        with Vertical(id="chat-container", classes="panel"):
                            yield Label("💬 智能对话", classes="panel-title")
                            yield ScrollableContainer(
                                Log(id="chat-log", auto_scroll=True),
                                id="chat-scroll"
                            )
                            
                            # 输入区域
                            with Horizontal(id="chat-input-section"):
                                yield Input(
                                    placeholder="请输入您的问题或指令...",
                                    id="chat-input"
                                )
                                yield Button("发送", id="btn-send", variant="primary")
                        
                        # 右侧快捷功能
                        with Vertical(id="quick-actions", classes="panel"):
                            yield Label("⚡ 快捷功能", classes="panel-title")
                            yield Button("📊 市场分析", id="btn-market-analysis", classes="quick-btn")
                            yield Button("💡 交易建议", id="btn-trading-advice", classes="quick-btn")
                            yield Button("⚠️ 风险评估", id="btn-risk-assessment", classes="quick-btn")
                            yield Button("📈 技术分析", id="btn-technical-analysis", classes="quick-btn")
                            yield Button("📰 新闻摘要", id="btn-news-summary", classes="quick-btn")
                            yield Button("🎯 策略优化", id="btn-strategy-optimization", classes="quick-btn")
                
                # 市场分析页
                with TabPane("市场分析", id="analysis-tab"):
                    with Horizontal(id="analysis-section"):
                        # 左侧分析结果
                        with Container(id="analysis-container", classes="panel"):
                            yield Label("📊 AI市场分析", classes="panel-title")
                            yield Static(id="market-analysis-content")
                        
                        # 右侧情绪指标
                        with Vertical(id="sentiment-section", classes="panel"):
                            yield Label("😊 市场情绪", classes="panel-title")
                            yield Static(id="sentiment-display")
                            
                            yield Label("📈 技术指标", classes="panel-title")
                            yield IndicatorChart(
                                indicator_name="AI信心指数",
                                symbol="MARKET",
                                id="confidence-chart"
                            )
                
                # 交易建议页
                with TabPane("交易建议", id="suggestions-tab"):
                    with Vertical(id="suggestions-section"):
                        with Horizontal(id="suggestion-controls"):
                            yield Label("选择币种:", classes="form-label")
                            yield Select([
                                ("BTC/USDT", "BTC/USDT"),
                                ("ETH/USDT", "ETH/USDT"),
                                ("ADA/USDT", "ADA/USDT"),
                                ("DOT/USDT", "DOT/USDT")
                            ], id="suggestion-symbol")
                            yield Button("获取建议", id="btn-get-suggestions", variant="primary")
                        
                        with Container(id="suggestions-display", classes="panel"):
                            yield Label("💡 AI交易建议", classes="panel-title")
                            yield Static(id="suggestions-content")
                
                # 风险监控页
                with TabPane("风险监控", id="risk-tab"):
                    with Container(id="risk-section", classes="panel"):
                        yield Label("⚠️ 实时风险监控", classes="panel-title")
                        yield Static(id="risk-analysis-content")

    def on_mount(self) -> None:
        """屏幕挂载时初始化"""
        try:
            self.start_ai_updates()
            self._initialize_welcome_message()
            logger.info("AI助手屏幕已挂载")
        except Exception as e:
            logger.error(f"AI助手屏幕挂载失败: {e}")

    def on_unmount(self) -> None:
        """屏幕卸载时清理"""
        try:
            self.stop_ai_updates()
            logger.info("AI助手屏幕已卸载")
        except Exception as e:
            logger.error(f"AI助手屏幕卸载失败: {e}")

    def start_ai_updates(self) -> None:
        """启动AI数据更新任务"""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self._ai_update_loop())

    def stop_ai_updates(self) -> None:
        """停止AI数据更新任务"""
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()

    async def _ai_update_loop(self) -> None:
        """AI分析更新循环"""
        while True:
            try:
                # 定期更新市场分析
                await self._update_market_analysis()
                
                # 更新情绪分析
                await self._update_sentiment_analysis()
                
                # 更新风险监控
                await self._update_risk_monitoring()
                
                await asyncio.sleep(self.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AI更新循环错误: {e}")
                await asyncio.sleep(5)

    def _initialize_welcome_message(self) -> None:
        """初始化欢迎消息"""
        try:
            chat_log = self.query_one("#chat-log", Log)
            
            welcome_msg = """[bold cyan]🤖 AI智能助手已就绪！[/bold cyan]

我可以为您提供以下服务：

📊 **市场分析**: 实时市场趋势分析和预测
💡 **交易建议**: 基于AI算法的个性化交易建议  
⚠️ **风险评估**: 投资组合风险分析和预警
📈 **技术分析**: 专业技术指标解读
📰 **新闻解读**: 市场新闻智能摘要和影响分析
🎯 **策略优化**: 交易策略智能优化建议

您可以：
• 直接输入问题进行对话
• 点击右侧快捷按钮快速获取分析
• 使用快捷键 Ctrl+R 刷新分析

有什么可以帮助您的吗？"""
            
            chat_log.write(welcome_msg)
            
        except Exception as e:
            logger.error(f"初始化欢迎消息失败: {e}")

    async def _update_market_analysis(self) -> None:
        """更新市场分析"""
        try:
            # 模拟AI市场分析
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            analysis = await ai_engine.analyze_market_trends(symbols)
            
            if analysis:
                self.market_analysis = analysis
                self._update_analysis_display()
                
        except Exception as e:
            logger.error(f"更新市场分析失败: {e}")

    async def _update_sentiment_analysis(self) -> None:
        """更新情绪分析"""
        try:
            # 获取市场情绪数据
            sentiment = await ai_engine.analyze_market_sentiment()
            
            if sentiment:
                self.sentiment_analysis = sentiment
                self._update_sentiment_display()
                
        except Exception as e:
            logger.error(f"更新情绪分析失败: {e}")

    async def _update_risk_monitoring(self) -> None:
        """更新风险监控"""
        try:
            # 获取风险评估数据
            portfolio_data = {
                "total_value": 10000,
                "position_ratio": 0.6,
                "unrealized_pnl": 500,
                "active_strategies": 3
            }
            
            risk_analysis = await ai_engine.assess_portfolio_risk(portfolio_data)
            
            if risk_analysis:
                self._update_risk_display(risk_analysis)
                
        except Exception as e:
            logger.error(f"更新风险监控失败: {e}")

    def _update_analysis_display(self) -> None:
        """更新分析显示"""
        try:
            if not self.market_analysis:
                return
                
            # 格式化市场分析内容
            analysis = self.market_analysis
            
            content = f"""[bold green]🔍 最新市场分析[/bold green]
[dim]更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]

[bold]总体趋势:[/bold] {analysis.get('overall_trend', '分析中...')}
[bold]市场情绪:[/bold] {analysis.get('market_sentiment', '中性')}
[bold]波动率:[/bold] {analysis.get('volatility_level', '正常')}

[bold cyan]主要币种分析:[/bold cyan]"""

            for symbol, data in analysis.get('symbols', {}).items():
                trend = data.get('trend', '横盘')
                confidence = data.get('confidence', 0.5) * 100
                
                trend_color = "green" if trend == "上涨" else "red" if trend == "下跌" else "yellow"
                confidence_color = "green" if confidence > 70 else "yellow" if confidence > 50 else "red"
                
                content += f"\n• [{trend_color}]{symbol}: {trend}[/{trend_color}] (信心度: [{confidence_color}]{confidence:.0f}%[/{confidence_color}])"
                
                if 'support' in data and 'resistance' in data:
                    content += f"\n  支撑: ${data['support']:.2f} | 阻力: ${data['resistance']:.2f}"

            # 添加AI建议
            if 'recommendations' in analysis:
                content += f"\n\n[bold yellow]🎯 AI建议:[/bold yellow]"
                for rec in analysis['recommendations']:
                    content += f"\n• {rec}"

            analysis_widget = self.query_one("#market-analysis-content", Static)
            analysis_widget.update(content)
            
        except Exception as e:
            logger.error(f"更新分析显示失败: {e}")

    def _update_sentiment_display(self) -> None:
        """更新情绪显示"""
        try:
            if not self.sentiment_analysis:
                return
                
            sentiment = self.sentiment_analysis
            
            # 构建情绪显示内容
            overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
            fear_greed_index = sentiment.get('fear_greed_index', 50)
            news_sentiment = sentiment.get('news_sentiment', 0)
            social_sentiment = sentiment.get('social_sentiment', 0)
            
            # 情绪颜色映射
            sentiment_colors = {
                'bullish': 'green',
                'bearish': 'red', 
                'neutral': 'yellow'
            }
            sentiment_color = sentiment_colors.get(overall_sentiment, 'yellow')
            
            # 恐慌贪婪指数颜色
            if fear_greed_index > 75:
                fg_color = "red"
                fg_status = "极度贪婪"
            elif fear_greed_index > 55:
                fg_color = "yellow"
                fg_status = "贪婪"
            elif fear_greed_index > 45:
                fg_color = "green"
                fg_status = "中性"
            elif fear_greed_index > 25:
                fg_color = "yellow"
                fg_status = "恐慌"
            else:
                fg_color = "red"
                fg_status = "极度恐慌"
            
            content = f"""[bold]😊 市场情绪分析[/bold]

[bold]总体情绪:[/bold] [{sentiment_color}]{overall_sentiment.upper()}[/{sentiment_color}]

[bold]📊 恐慌贪婪指数:[/bold]
[{fg_color}]{fear_greed_index}/100 - {fg_status}[/{fg_color}]

[bold]📰 新闻情绪:[/bold] {news_sentiment:+.2f}
[bold]💬 社交媒体:[/bold] {social_sentiment:+.2f}

[bold yellow]情绪指标解读:[/bold yellow]"""
            
            # 添加情绪解读
            interpretations = sentiment.get('interpretations', [])
            for interp in interpretations:
                content += f"\n• {interp}"
                
            sentiment_widget = self.query_one("#sentiment-display", Static)
            sentiment_widget.update(content)
            
            # 更新信心指数图表
            confidence_chart = self.query_one("#confidence-chart", IndicatorChart)
            confidence_chart.add_indicator_data(fear_greed_index, datetime.now())
            
        except Exception as e:
            logger.error(f"更新情绪显示失败: {e}")

    def _update_risk_display(self, risk_analysis: Dict[str, Any]) -> None:
        """更新风险显示"""
        try:
            risk_level = risk_analysis.get('risk_level', 'medium')
            overall_score = risk_analysis.get('overall_score', 0.5) * 100
            
            # 风险等级颜色
            risk_colors = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'red',
                'critical': 'bright_red'
            }
            risk_color = risk_colors.get(risk_level, 'yellow')
            
            content = f"""[bold red]⚠️ 投资组合风险分析[/bold red]
[dim]更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]

[bold]总体风险等级:[/bold] [{risk_color}]{risk_level.upper()}[/{risk_color}]
[bold]风险评分:[/bold] {overall_score:.0f}/100

[bold cyan]风险分解:[/bold cyan]"""
            
            # 风险分项
            risk_factors = risk_analysis.get('risk_factors', {})
            for factor_name, factor_data in risk_factors.items():
                score = factor_data.get('score', 0) * 100
                level = factor_data.get('level', 'medium')
                factor_color = risk_colors.get(level, 'yellow')
                
                content += f"\n• [{factor_color}]{factor_name}: {score:.0f}%[/{factor_color}]"
                
                # 添加说明
                if 'description' in factor_data:
                    content += f"\n  {factor_data['description']}"
            
            # 风险预警
            if risk_analysis.get('urgent_action_needed', False):
                content += f"\n\n[bold bright_red]🚨 紧急风险预警 🚨[/bold bright_red]"
                warnings = risk_analysis.get('warnings', [])
                for warning in warnings:
                    content += f"\n❗ {warning}"
            
            # 风险建议
            recommendations = risk_analysis.get('recommendations', [])
            if recommendations:
                content += f"\n\n[bold yellow]💡 风险缓解建议:[/bold yellow]"
                for rec in recommendations:
                    content += f"\n• {rec}"
            
            risk_widget = self.query_one("#risk-analysis-content", Static)
            risk_widget.update(content)
            
        except Exception as e:
            logger.error(f"更新风险显示失败: {e}")

    # ============ 事件处理器 ============

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击事件处理"""
        button_id = event.button.id
        
        try:
            if button_id == "btn-send":
                self.action_send_message()
            elif button_id == "btn-refresh-analysis":
                self.action_refresh_analysis()
            elif button_id == "btn-clear-chat":
                self.action_clear_chat()
            elif button_id == "btn-market-analysis":
                asyncio.create_task(self._quick_market_analysis())
            elif button_id == "btn-trading-advice":
                asyncio.create_task(self._quick_trading_advice())
            elif button_id == "btn-risk-assessment":
                asyncio.create_task(self._quick_risk_assessment())
            elif button_id == "btn-technical-analysis":
                asyncio.create_task(self._quick_technical_analysis())
            elif button_id == "btn-news-summary":
                asyncio.create_task(self._quick_news_summary())
            elif button_id == "btn-strategy-optimization":
                asyncio.create_task(self._quick_strategy_optimization())
            elif button_id == "btn-get-suggestions":
                asyncio.create_task(self._get_trading_suggestions())
                
        except Exception as e:
            logger.error(f"按钮事件处理失败: {e}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """输入框提交事件"""
        if event.input.id == "chat-input":
            self.action_send_message()

    # ============ 动作处理器 ============

    def action_send_message(self) -> None:
        """发送消息"""
        try:
            chat_input = self.query_one("#chat-input", Input)
            message = chat_input.value.strip()
            
            if not message:
                return
            
            # 清空输入框
            chat_input.value = ""
            
            # 显示用户消息
            self._add_chat_message("User", message, "cyan")
            
            # 处理AI回复
            asyncio.create_task(self._process_ai_response(message))
            
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    def action_refresh_analysis(self) -> None:
        """刷新分析"""
        try:
            # 重启AI更新任务
            self.stop_ai_updates()
            self.start_ai_updates()
            
            self._add_chat_message("System", "🔄 正在刷新AI分析...", "yellow")
            
        except Exception as e:
            logger.error(f"刷新分析失败: {e}")

    def action_clear_chat(self) -> None:
        """清空对话"""
        try:
            chat_log = self.query_one("#chat-log", Log)
            chat_log.clear()
            
            # 重新显示欢迎消息
            self._initialize_welcome_message()
            
        except Exception as e:
            logger.error(f"清空对话失败: {e}")

    # ============ AI处理方法 ============

    async def _process_ai_response(self, user_message: str) -> None:
        """处理AI响应"""
        try:
            if self.is_processing:
                self._add_chat_message("System", "⏳ AI正在处理中，请稍候...", "yellow")
                return
            
            self.is_processing = True
            self._add_chat_message("System", "🤔 AI思考中...", "dim")
            
            # 调用AI引擎处理消息
            response = await ai_engine.process_user_query(user_message)
            
            # 移除思考中的消息并显示回复
            chat_log = self.query_one("#chat-log", Log)
            
            if response:
                self._add_chat_message("AI", response.get('answer', '抱歉，我无法回答这个问题。'), "green")
                
                # 如果有建议，也显示出来
                if 'suggestions' in response:
                    for suggestion in response['suggestions']:
                        self._add_chat_message("AI", f"💡 建议: {suggestion}", "blue")
            else:
                self._add_chat_message("AI", "抱歉，我现在遇到了一些技术问题，请稍后再试。", "red")
            
        except Exception as e:
            logger.error(f"处理AI响应失败: {e}")
            self._add_chat_message("AI", f"处理消息时出现错误: {e}", "red")
        finally:
            self.is_processing = False

    def _add_chat_message(self, sender: str, message: str, style: str) -> None:
        """添加聊天消息"""
        try:
            chat_log = self.query_one("#chat-log", Log)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if sender == "User":
                formatted_message = f"[{style}][{timestamp}] 👤 {sender}:[/{style}] {message}"
            elif sender == "AI":
                formatted_message = f"[{style}][{timestamp}] 🤖 {sender}:[/{style}] {message}"
            else:
                formatted_message = f"[{style}][{timestamp}] {sender}:[/{style}] {message}"
            
            chat_log.write(formatted_message)
            
        except Exception as e:
            logger.error(f"添加聊天消息失败: {e}")

    # ============ 快捷功能方法 ============

    async def _quick_market_analysis(self) -> None:
        """快速市场分析"""
        self._add_chat_message("System", "📊 正在进行市场分析...", "yellow")
        
        try:
            symbols = ["BTC/USDT", "ETH/USDT"]
            analysis = await ai_engine.analyze_market_trends(symbols)
            
            if analysis:
                summary = f"""📊 **快速市场分析**

**总体趋势**: {analysis.get('overall_trend', '分析中')}
**市场情绪**: {analysis.get('market_sentiment', '中性')}
**建议操作**: {analysis.get('recommended_action', '观望')}

详细分析请查看"市场分析"标签页。"""
                
                self._add_chat_message("AI", summary, "green")
            else:
                self._add_chat_message("AI", "市场分析暂时不可用，请稍后重试。", "red")
                
        except Exception as e:
            logger.error(f"快速市场分析失败: {e}")
            self._add_chat_message("AI", f"市场分析失败: {e}", "red")

    async def _quick_trading_advice(self) -> None:
        """快速交易建议"""
        self._add_chat_message("System", "💡 正在生成交易建议...", "yellow")
        
        try:
            # 模拟交易建议
            advice = await ai_engine.generate_trading_advice(["BTC/USDT"])
            
            if advice:
                suggestion_text = f"""💡 **AI交易建议**

**推荐操作**: {advice.get('action', '观望')}
**目标价位**: {advice.get('target_price', 'N/A')}
**止损价位**: {advice.get('stop_loss', 'N/A')}
**信心度**: {advice.get('confidence', 0.5)*100:.0f}%

**理由**: {advice.get('reasoning', '基于当前市场趋势分析')}"""
                
                self._add_chat_message("AI", suggestion_text, "green")
            else:
                self._add_chat_message("AI", "暂时无法生成交易建议。", "red")
                
        except Exception as e:
            logger.error(f"快速交易建议失败: {e}")
            self._add_chat_message("AI", f"生成交易建议失败: {e}", "red")

    async def _quick_risk_assessment(self) -> None:
        """快速风险评估"""
        self._add_chat_message("System", "⚠️ 正在评估投资组合风险...", "yellow")
        
        try:
            portfolio_data = {
                "total_value": 10000,
                "position_ratio": 0.6,
                "unrealized_pnl": 500,
                "active_strategies": 3
            }
            
            risk_analysis = await ai_engine.assess_portfolio_risk(portfolio_data)
            
            if risk_analysis:
                risk_text = f"""⚠️ **风险评估报告**

**风险等级**: {risk_analysis.get('risk_level', 'medium').upper()}
**风险评分**: {risk_analysis.get('overall_score', 0.5)*100:.0f}/100

**主要风险**: {', '.join(risk_analysis.get('main_risks', ['正常市场风险']))}

**建议**: {risk_analysis.get('recommendation', '保持当前仓位配置')}"""
                
                color = "red" if risk_analysis.get('risk_level') in ['high', 'critical'] else "yellow"
                self._add_chat_message("AI", risk_text, color)
            else:
                self._add_chat_message("AI", "风险评估暂时不可用。", "red")
                
        except Exception as e:
            logger.error(f"快速风险评估失败: {e}")
            self._add_chat_message("AI", f"风险评估失败: {e}", "red")

    async def _quick_technical_analysis(self) -> None:
        """快速技术分析"""
        self._add_chat_message("System", "📈 正在进行技术分析...", "yellow")
        
        # 模拟技术分析
        analysis_text = """📈 **技术分析摘要**

**BTC/USDT**:
- RSI: 65 (接近超买)
- MACD: 金叉信号
- 支撑位: $44,500
- 阻力位: $46,800

**ETH/USDT**:
- RSI: 58 (正常区间)
- MA20 > MA50 (短期看涨)
- 支撑位: $2,750
- 阻力位: $2,950

**总结**: 短期内保持谨慎乐观，关注阻力位突破情况。"""
        
        self._add_chat_message("AI", analysis_text, "green")

    async def _quick_news_summary(self) -> None:
        """快速新闻摘要"""
        self._add_chat_message("System", "📰 正在分析最新市场新闻...", "yellow")
        
        # 模拟新闻摘要
        news_text = """📰 **市场新闻摘要**

**今日热点**:
1. 🏛️ 某国央行表示将继续监管加密货币发展
2. 🏢 大型机构增持比特币ETF份额
3. 📊 加密货币市场总市值突破新高
4. ⚡ 以太坊网络升级进展顺利

**影响分析**:
- 监管消息对短期价格影响有限
- 机构资金流入提供长期支撑
- 技术升级增强网络价值

**建议**: 关注监管动态，把握长期投资机会。"""
        
        self._add_chat_message("AI", news_text, "green")

    async def _quick_strategy_optimization(self) -> None:
        """快速策略优化"""
        self._add_chat_message("System", "🎯 正在分析策略优化机会...", "yellow")
        
        # 模拟策略优化建议
        optimization_text = """🎯 **策略优化建议**

**当前策略分析**:
- 网格策略: 表现良好，建议适当扩大网格范围
- DCA策略: 可考虑调整定投频率
- AI策略: 建议提高信心阈值到75%

**优化方向**:
1. 🔄 调整仓位分配比例 (40% 网格 + 35% AI + 25% DCA)
2. ⚡ 优化止盈止损参数
3. 📊 增加相关性低的交易对

**预期收益提升**: 8-15%

是否需要详细的优化方案？"""
        
        self._add_chat_message("AI", optimization_text, "green")

    async def _get_trading_suggestions(self) -> None:
        """获取交易建议"""
        try:
            symbol_select = self.query_one("#suggestion-symbol", Select)
            selected_symbol = symbol_select.value
            
            self._add_chat_message("System", f"💡 正在为 {selected_symbol} 生成交易建议...", "yellow")
            
            # 调用AI引擎生成建议
            suggestions = await ai_engine.generate_trading_advice([selected_symbol])
            
            if suggestions:
                content = f"""💡 **{selected_symbol} 交易建议**

**推荐操作**: {suggestions.get('action', '观望')}
**入场价位**: {suggestions.get('entry_price', 'N/A')}
**目标价位**: {suggestions.get('target_price', 'N/A')}
**止损价位**: {suggestions.get('stop_loss', 'N/A')}
**仓位建议**: {suggestions.get('position_size', 'N/A')}
**持有期限**: {suggestions.get('holding_period', 'N/A')}

**分析依据**:
{suggestions.get('reasoning', '基于技术分析和市场情绪')}

**风险提示**: {suggestions.get('risk_warning', '投资有风险，请谨慎决策')}"""
                
                # 显示在建议内容区域
                suggestions_widget = self.query_one("#suggestions-content", Static)
                suggestions_widget.update(content)
                
                self._add_chat_message("AI", f"已为 {selected_symbol} 生成交易建议，请查看"交易建议"标签页。", "green")
            else:
                self._add_chat_message("AI", "暂时无法生成交易建议，请稍后重试。", "red")
                
        except Exception as e:
            logger.error(f"获取交易建议失败: {e}")
            self._add_chat_message("AI", f"生成交易建议失败: {e}", "red")