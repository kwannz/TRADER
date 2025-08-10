"""
AI智能助手界面 - 对话式AI分析和交易建议
支持自然语言交互，提供专业的量化交易建议
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from rich.text import Text
from rich.panel import Panel
from rich.console import Group
from rich.align import Align
from rich.columns import Columns
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Input, Button, Label, TextArea
from textual.screen import Screen
from textual.binding import Binding
from loguru import logger

from config.bloomberg_theme import BLOOMBERG_COLORS, STATUS_INDICATORS
from core.ai_engine import ai_engine
from core.data_manager import data_manager

class ChatMessage:
    """聊天消息类"""
    
    def __init__(self, content: str, is_user: bool, timestamp: Optional[datetime] = None):
        self.content = content
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now()
        self.id = f"msg_{self.timestamp.timestamp()}"

class ChatHistoryWidget(ScrollableContainer):
    """聊天历史显示组件"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: List[ChatMessage] = []
        
    async def add_message(self, message: ChatMessage):
        """添加新消息"""
        self.messages.append(message)
        await self.refresh_display()
        # 滚动到底部
        self.scroll_end()
    
    async def refresh_display(self):
        """刷新聊天显示"""
        try:
            # 清空现有内容
            await self.remove_children()
            
            # 添加欢迎消息
            if not self.messages:
                welcome_widget = Static(Panel(
                    Group(
                        Align.center("🤖 AI智能助手"),
                        "",
                        Align.center("您好！我是您的AI量化交易助手"),
                        Align.center("我可以帮您："),
                        "",
                        "• 📊 分析市场趋势和情绪",
                        "• 🚀 生成和优化交易策略", 
                        "• 💡 提供个性化交易建议",
                        "• 🔍 解释技术指标和信号",
                        "• 📈 评估投资组合风险",
                        "",
                        Align.center("请输入您的问题或需求...")
                    ),
                    title="欢迎使用AI助手",
                    border_style="bright_magenta",
                    padding=(1, 2)
                ))
                await self.mount(welcome_widget)
                return
            
            # 显示聊天记录
            for message in self.messages:
                message_widget = await self.create_message_widget(message)
                await self.mount(message_widget)
                
        except Exception as e:
            logger.error(f"聊天历史刷新失败: {e}")
    
    async def create_message_widget(self, message: ChatMessage) -> Static:
        """创建消息组件"""
        try:
            timestamp_str = message.timestamp.strftime("%H:%M:%S")
            
            if message.is_user:
                # 用户消息 - 右对齐，蓝色边框
                content = Panel(
                    Group(
                        Text(message.content, style="white"),
                        "",
                        Text(f"🕒 {timestamp_str}", style="dim")
                    ),
                    title="👤 您",
                    title_align="right",
                    border_style="bright_blue",
                    padding=(1, 1),
                    width=60
                )
                
                return Static(Align.right(content))
                
            else:
                # AI消息 - 左对齐，紫色边框
                content = Panel(
                    Group(
                        Text(message.content, style="white"),
                        "",
                        Text(f"🕒 {timestamp_str}", style="dim")
                    ),
                    title="🤖 AI助手",
                    title_align="left",
                    border_style="bright_magenta",
                    padding=(1, 1),
                    width=60
                )
                
                return Static(Align.left(content))
                
        except Exception as e:
            logger.error(f"创建消息组件失败: {e}")
            return Static("消息显示错误")

class QuickActionsWidget(Static):
    """快捷操作组件"""
    
    def __init__(self, on_action_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_action = on_action_callback
        
    def compose(self) -> ComposeResult:
        yield Label("⚡ 快捷操作")
        
        with Horizontal():
            yield Button("市场分析", variant="primary", id="market-analysis-btn")
            yield Button("策略建议", variant="success", id="strategy-advice-btn")
        
        with Horizontal():
            yield Button("风险评估", variant="warning", id="risk-assessment-btn") 
            yield Button("新闻解读", variant="default", id="news-analysis-btn")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """处理快捷操作按钮"""
        action_map = {
            "market-analysis-btn": "请分析当前比特币和以太坊的市场趋势，给出技术面和基本面的判断",
            "strategy-advice-btn": "基于当前市场条件，推荐一个适合的交易策略",
            "risk-assessment-btn": "评估我当前投资组合的风险状况，给出风控建议",
            "news-analysis-btn": "分析最近的加密货币相关新闻，总结市场情绪"
        }
        
        if event.button.id in action_map:
            await self.on_action(action_map[event.button.id])

class AIAssistantScreen(Screen):
    """AI智能助手主屏幕"""
    
    CSS = """
    .ai-layout {
        layout: vertical;
        height: 1fr;
        margin: 1;
    }
    
    .chat-area {
        height: 1fr;
        border: solid $primary;
        margin: 0 0 1 0;
    }
    
    .input-area {
        layout: horizontal;
        height: 5;
    }
    
    .input-field {
        width: 1fr;
        margin-right: 1;
    }
    
    .send-button {
        width: 12;
    }
    
    .quick-actions {
        height: 8;
        margin: 1 0;
        border: solid $secondary;
        padding: 1;
    }
    
    .typing-indicator {
        height: 3;
        margin: 0 0 1 0;
    }
    """
    
    BINDINGS = [
        Binding("enter", "send_message", "发送消息"),
        Binding("ctrl+c", "clear_chat", "清空聊天"),
        Binding("escape", "back", "返回"),
    ]
    
    def __init__(self):
        super().__init__()
        self.chat_history = None
        self.is_ai_thinking = False
        self.conversation_context = []
        
    def compose(self) -> ComposeResult:
        with Container(classes="ai-layout"):
            yield Label("🤖 AI智能助手")
            
            # 快捷操作区
            yield QuickActionsWidget(
                self.handle_quick_action,
                classes="quick-actions"
            )
            
            # 聊天区域
            self.chat_history = ChatHistoryWidget(classes="chat-area")
            yield self.chat_history
            
            # AI思考指示器
            yield Static("", id="typing-indicator", classes="typing-indicator")
            
            # 输入区域
            with Horizontal(classes="input-area"):
                yield TextArea(
                    placeholder="输入您的问题或需求...",
                    id="message-input",
                    classes="input-field"
                )
                yield Button("发送", variant="primary", id="send-btn", classes="send-button")
    
    async def on_mount(self):
        """页面挂载初始化"""
        try:
            logger.info("AI助手界面初始化...")
            
            # 初始化聊天历史
            await self.chat_history.refresh_display()
            
            # 焦点设置到输入框
            self.query_one("#message-input", TextArea).focus()
            
            logger.info("AI助手界面初始化完成")
            
        except Exception as e:
            logger.error(f"AI助手界面初始化失败: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """处理按钮点击"""
        if event.button.id == "send-btn":
            await self.action_send_message()
    
    async def action_send_message(self) -> None:
        """发送消息"""
        try:
            input_widget = self.query_one("#message-input", TextArea)
            message_text = input_widget.text.strip()
            
            if not message_text:
                return
            
            # 清空输入框
            input_widget.text = ""
            
            # 添加用户消息
            user_message = ChatMessage(message_text, is_user=True)
            await self.chat_history.add_message(user_message)
            
            # 显示AI思考状态
            await self.show_ai_thinking()
            
            # 获取AI回复
            ai_response = await self.get_ai_response(message_text)
            
            # 隐藏AI思考状态
            await self.hide_ai_thinking()
            
            # 添加AI回复
            ai_message = ChatMessage(ai_response, is_user=False)
            await self.chat_history.add_message(ai_message)
            
            # 更新对话上下文
            self.conversation_context.append({
                "user": message_text,
                "ai": ai_response,
                "timestamp": datetime.now()
            })
            
            # 保持最近10轮对话
            if len(self.conversation_context) > 10:
                self.conversation_context = self.conversation_context[-10:]
                
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            await self.hide_ai_thinking()
            
            error_message = ChatMessage(
                f"抱歉，处理您的消息时出现错误：{e}",
                is_user=False
            )
            await self.chat_history.add_message(error_message)
    
    async def handle_quick_action(self, action_text: str):
        """处理快捷操作"""
        try:
            # 添加用户消息
            user_message = ChatMessage(action_text, is_user=True)
            await self.chat_history.add_message(user_message)
            
            # 显示AI思考
            await self.show_ai_thinking()
            
            # 获取AI回复
            ai_response = await self.get_ai_response(action_text)
            
            # 隐藏AI思考
            await self.hide_ai_thinking()
            
            # 添加AI回复
            ai_message = ChatMessage(ai_response, is_user=False)
            await self.chat_history.add_message(ai_message)
            
        except Exception as e:
            logger.error(f"快捷操作处理失败: {e}")
    
    async def get_ai_response(self, user_message: str) -> str:
        """获取AI回复"""
        try:
            # 构建对话上下文
            context = await self.build_context()
            
            # 调用AI引擎
            response = await ai_engine.chat_with_assistant(user_message, context)
            
            # 提取回复内容
            if isinstance(response, dict):
                ai_reply = response.get("response", "抱歉，我现在无法回答您的问题。")
                
                # 如果有建议，添加到回复中
                suggestions = response.get("suggestions", [])
                if suggestions:
                    ai_reply += f"\n\n💡 建议：\n" + "\n".join(f"• {s}" for s in suggestions[:3])
                
                # 如果有风险提示，添加到回复中
                risk_warning = response.get("risk_warning")
                if risk_warning:
                    ai_reply += f"\n\n⚠️ 风险提示：{risk_warning}"
                
                return ai_reply
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"AI回复获取失败: {e}")
            return f"抱歉，AI服务暂时不可用：{e}"
    
    async def build_context(self) -> Dict[str, Any]:
        """构建对话上下文"""
        try:
            # 获取系统状态
            context = {
                "timestamp": datetime.now().isoformat(),
                "conversation_history": self.conversation_context[-5:],  # 最近5轮对话
            }
            
            # 获取市场数据
            try:
                # 这里可以添加更多上下文信息
                context.update({
                    "market_session": "active",
                    "user_language": "Chinese"
                })
            except Exception as e:
                logger.warning(f"获取市场上下文失败: {e}")
            
            return context
            
        except Exception as e:
            logger.error(f"构建对话上下文失败: {e}")
            return {}
    
    async def show_ai_thinking(self):
        """显示AI思考状态"""
        try:
            self.is_ai_thinking = True
            indicator = self.query_one("#typing-indicator", Static)
            
            thinking_text = Panel(
                "🧠 AI正在思考中...",
                border_style="bright_magenta",
                padding=(0, 1)
            )
            
            indicator.update(thinking_text)
            
        except Exception as e:
            logger.error(f"显示AI思考状态失败: {e}")
    
    async def hide_ai_thinking(self):
        """隐藏AI思考状态"""
        try:
            self.is_ai_thinking = False
            indicator = self.query_one("#typing-indicator", Static)
            indicator.update("")
        except Exception as e:
            logger.error(f"隐藏AI思考状态失败: {e}")
    
    async def action_clear_chat(self) -> None:
        """清空聊天记录"""
        try:
            self.chat_history.messages.clear()
            self.conversation_context.clear()
            await self.chat_history.refresh_display()
            self.notify("聊天记录已清空", severity="information")
        except Exception as e:
            logger.error(f"清空聊天失败: {e}")
    
    async def action_back(self) -> None:
        """返回上一页"""
        self.app.pop_screen()

# AI助手相关工具函数
def create_ai_assistant() -> AIAssistantScreen:
    """创建AI助手实例"""
    return AIAssistantScreen()