"""
系统设置屏幕

Bloomberg Terminal风格的系统设置界面，支持：
- 系统参数配置
- API密钥管理
- 界面主题设置
- 风控参数调整
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Static, Button, Input, Select, Label, Switch,
    TabbedContent, TabPane
)
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.binding import Binding

from ..themes.bloomberg import BloombergTheme
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class SettingsScreen(Screen):
    """系统设置主屏幕"""
    
    CSS_PATH = "settings.css"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "返回"),
        Binding("ctrl+s", "save_settings", "保存设置"),
        Binding("r", "reset_settings", "重置设置"),
    ]
    
    def __init__(self):
        super().__init__()
        self.theme = BloombergTheme()
        
    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            with TabbedContent(id="settings-tabs"):
                with TabPane("基础设置", id="basic-tab"):
                    with Vertical(classes="settings-section"):
                        yield Label("⚙️ 基础配置", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("刷新频率(Hz):", classes="setting-label")
                            yield Input("4", id="refresh-rate")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("初始资金(USDT):", classes="setting-label")
                            yield Input("10000", id="initial-balance")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("最大仓位比例:", classes="setting-label")
                            yield Input("0.8", id="max-position")
                
                with TabPane("API设置", id="api-tab"):
                    with Vertical(classes="settings-section"):
                        yield Label("🔑 API配置", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("OKX API Key:", classes="setting-label")
                            yield Input("", id="okx-api-key", password=True)
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("Binance API Key:", classes="setting-label")
                            yield Input("", id="binance-api-key", password=True)
                
                with TabPane("界面设置", id="ui-tab"):
                    with Vertical(classes="settings-section"):
                        yield Label("🎨 界面配置", classes="section-title")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("主题:", classes="setting-label")
                            yield Select([("Bloomberg", "bloomberg"), ("Dark", "dark")], id="theme-select")
                        
                        with Horizontal(classes="setting-row"):
                            yield Label("启用声音:", classes="setting-label")
                            yield Switch(id="sound-switch")

    def on_mount(self) -> None:
        """屏幕挂载"""
        try:
            self._load_current_settings()
            logger.info("设置屏幕已挂载")
        except Exception as e:
            logger.error(f"设置屏幕挂载失败: {e}")

    def _load_current_settings(self) -> None:
        """加载当前设置"""
        pass