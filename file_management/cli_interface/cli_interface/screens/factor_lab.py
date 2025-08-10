"""
因子实验室屏幕

Bloomberg Terminal风格的量化因子分析界面，支持：
- Alpha因子发现和验证
- 因子IC分析和回测
- 因子组合优化
- 自定义因子开发
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Static, Button, Input, Select, TextArea, Label,
    TabbedContent, TabPane, ProgressBar
)
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable

from ..components.tables import FactorTable, CustomDataTable
from ..components.charts import IndicatorChart
from ..themes.bloomberg import BloombergTheme
from ...core.data_manager import data_manager
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class FactorLabScreen(Screen):
    """因子实验室主屏幕"""
    
    CSS_PATH = "factor_lab.css"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "返回"),
        Binding("n", "new_factor", "新建因子"),
        Binding("r", "refresh", "刷新"),
        Binding("t", "test_factor", "测试因子"),
        Binding("b", "backtest_factor", "回测因子"),
    ]
    
    factors = reactive([])
    selected_factor = reactive({})
    
    def __init__(self):
        super().__init__()
        self.theme = BloombergTheme()
        
    def compose(self) -> ComposeResult:
        with Container(id="factor-lab-container"):
            with TabbedContent(id="factor-tabs"):
                with TabPane("因子列表", id="factor-list-tab"):
                    with Horizontal():
                        with Container(classes="panel"):
                            yield Label("🧪 Alpha因子列表", classes="panel-title")
                            yield FactorTable(id="factor-table")
                        
                        with Container(classes="panel"):
                            yield Label("📊 因子分析", classes="panel-title")
                            yield Static(id="factor-analysis")
                
                with TabPane("因子开发", id="factor-dev-tab"):
                    with Vertical():
                        yield Label("🔬 自定义因子开发", classes="section-title")
                        yield TextArea("# 输入因子公式\n# 例: close / sma(close, 20) - 1", id="factor-formula")
                        yield Button("测试因子", id="btn-test-factor", variant="primary")
                        yield Static(id="factor-test-results")

    def on_mount(self) -> None:
        """屏幕挂载"""
        try:
            self._load_factors()
            logger.info("因子实验室屏幕已挂载")
        except Exception as e:
            logger.error(f"因子实验室屏幕挂载失败: {e}")

    def _load_factors(self) -> None:
        """加载因子数据"""
        # 模拟因子数据
        factors = [
            {
                "name": "动量因子",
                "ic_mean": 0.085,
                "ic_ir": 1.34,
                "max_drawdown": 0.08,
                "annual_return": 0.156,
                "sharpe_ratio": 1.89,
                "status": "active"
            },
            {
                "name": "均值回归因子",
                "ic_mean": -0.062,
                "ic_ir": -1.12,
                "max_drawdown": 0.12,
                "annual_return": 0.098,
                "sharpe_ratio": 1.23,
                "status": "testing"
            }
        ]
        
        factor_table = self.query_one("#factor-table", FactorTable)
        factor_table.update_data(factors)