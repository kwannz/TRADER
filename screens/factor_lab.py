"""
因子发现实验室 - AI驱动的量化因子挖掘平台
集成CTBench时间序列生成和Alpha101因子计算
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, TaskID
from rich.console import Group
from rich.align import Align
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Static, DataTable, Button, Input, Select, TextArea, 
    Label, ProgressBar, TabPane, TabbedContent, Checkbox
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from loguru import logger

from config.bloomberg_theme import BLOOMBERG_COLORS, STATUS_INDICATORS
from core.data_manager import data_manager
from core.ai_engine import ai_engine
from ctbench.features.alpha101 import alpha101_calculator
from ctbench.models.base_model import create_model

class FactorDiscoveryModal(ModalScreen):
    """因子发现模态对话框"""
    
    CSS = """
    FactorDiscoveryModal {
        align: center middle;
    }
    
    .discovery-modal {
        width: 80;
        height: 30;
        background: $surface;
        border: thick $primary;
    }
    
    .form-field {
        margin: 1 0;
    }
    
    .progress-area {
        height: 8;
        margin: 1 0;
        border: solid $secondary;
        padding: 1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.result = None
        self.discovering = False
        
    def compose(self) -> ComposeResult:
        with Container(classes="discovery-modal"):
            yield Label("🔬 AI因子发现器", classes="form-field")
            
            yield Label("分析标的:", classes="form-field")
            with Horizontal():
                yield Checkbox("BTC/USDT", value=True, id="btc-checkbox")
                yield Checkbox("ETH/USDT", value=True, id="eth-checkbox")
                yield Checkbox("BNB/USDT", value=False, id="bnb-checkbox")
                yield Checkbox("SOL/USDT", value=False, id="sol-checkbox")
            
            yield Label("发现模式:", classes="form-field")
            yield Select([
                ("快速发现 (30个样本)", "fast"),
                ("标准发现 (100个样本)", "standard"),
                ("深度发现 (300个样本)", "deep")
            ], value="standard", id="discovery-mode", classes="form-field")
            
            yield Label("AI模型:", classes="form-field")
            yield Select([
                ("DeepSeek + Gemini", "both"),
                ("仅DeepSeek", "deepseek"),
                ("仅Gemini", "gemini")
            ], value="both", id="ai-model", classes="form-field")
            
            with Container(classes="progress-area"):
                yield Label("发现进度:")
                yield ProgressBar(id="discovery-progress")
                yield Static("", id="progress-status")
            
            with Horizontal():
                yield Button("开始发现", variant="primary", id="start-btn")
                yield Button("取消", variant="default", id="cancel-btn")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn" and not self.discovering:
            await self.start_discovery()
        else:
            self.dismiss()
    
    async def start_discovery(self):
        """开始因子发现"""
        try:
            self.discovering = True
            progress = self.query_one("#discovery-progress", ProgressBar)
            status = self.query_one("#progress-status", Static)
            start_btn = self.query_one("#start-btn", Button)
            
            start_btn.disabled = True
            progress.update(progress=0)
            status.update("准备数据...")
            
            # 获取选中的标的
            symbols = []
            if self.query_one("#btc-checkbox", Checkbox).value:
                symbols.append("BTC-USDT")
            if self.query_one("#eth-checkbox", Checkbox).value:
                symbols.append("ETH-USDT")
            if self.query_one("#bnb-checkbox", Checkbox).value:
                symbols.append("BNB-USDT")
            if self.query_one("#sol-checkbox", Checkbox).value:
                symbols.append("SOL-USDT")
            
            if not symbols:
                self.notify("请至少选择一个交易对", severity="error")
                return
            
            mode = self.query_one("#discovery-mode", Select).value
            ai_model = self.query_one("#ai-model", Select).value
            
            progress.update(progress=20)
            status.update("收集历史数据...")
            
            # 模拟数据收集过程
            await asyncio.sleep(1)
            
            progress.update(progress=40)
            status.update("AI分析中...")
            
            # 调用AI因子发现
            discovered_factors = await ai_engine.discover_alpha_factors(symbols)
            
            progress.update(progress=70)
            status.update("计算因子统计...")
            
            # 计算传统因子作为对比
            traditional_factors = await self.calculate_traditional_factors(symbols)
            
            progress.update(progress=90)
            status.update("生成报告...")
            
            # 合并结果
            all_factors = {
                **discovered_factors,
                "traditional_factors": traditional_factors
            }
            
            progress.update(progress=100)
            status.update("发现完成!")
            
            await asyncio.sleep(1)
            
            self.result = {
                "success": True,
                "factors": all_factors,
                "symbols": symbols,
                "mode": mode,
                "ai_model": ai_model
            }
            
            self.dismiss(self.result)
            
        except Exception as e:
            logger.error(f"因子发现失败: {e}")
            self.notify(f"发现失败: {e}", severity="error")
        finally:
            self.discovering = False
            start_btn.disabled = False
    
    async def calculate_traditional_factors(self, symbols: List[str]) -> Dict[str, Any]:
        """计算传统因子"""
        try:
            factors = []
            
            for symbol in symbols:
                # 获取历史数据
                df = await data_manager.time_series_manager.get_kline_data(
                    symbol, "1h", limit=500
                )
                
                if not df.empty:
                    # 计算Alpha101因子
                    factor_df = alpha101_calculator.calculate_all_factors(df)
                    
                    # 计算因子IC
                    returns = df['close'].pct_change()
                    
                    for col in factor_df.columns:
                        if col.startswith('alpha_'):
                            factor_values = factor_df[col]
                            stats = alpha101_calculator.calculate_factor_statistics(
                                factor_values, returns
                            )
                            
                            if not stats.get("error"):
                                factors.append({
                                    "name": f"{col}_{symbol}",
                                    "formula": f"Alpha101.{col}",
                                    "ic_mean": stats.get("IC_1d", 0),
                                    "ic_ir": stats.get("ICIR", 0),
                                    "symbol": symbol,
                                    "type": "traditional"
                                })
            
            return {"factors": factors, "count": len(factors)}
            
        except Exception as e:
            logger.error(f"传统因子计算失败: {e}")
            return {"factors": [], "count": 0}

class FactorAnalysisWidget(Static):
    """因子分析组件"""
    
    def __init__(self, factor_data: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_data = factor_data
        
    async def on_mount(self):
        """挂载时更新显示"""
        await self.refresh_display()
    
    async def refresh_display(self):
        """刷新因子分析显示"""
        try:
            if not self.factor_data:
                self.update("请选择一个因子查看分析详情")
                return
            
            # 基本信息
            name = self.factor_data.get("name", "未知因子")
            formula = self.factor_data.get("formula", "公式未知")
            description = self.factor_data.get("description", "暂无描述")
            
            # 性能指标
            ic_mean = self.factor_data.get("ic_mean", 0)
            ic_ir = self.factor_data.get("ICIR", 0)
            expected_ic = self.factor_data.get("expected_ic", "未知")
            
            # 创建分析表格
            table = Table(show_header=False, box=None)
            table.add_column("指标", style="cyan")
            table.add_column("数值", justify="right")
            
            table.add_row("因子名称", name)
            table.add_row("计算公式", formula)
            table.add_row("IC均值", f"{ic_mean:.4f}")
            table.add_row("ICIR", f"{ic_ir:.2f}")
            table.add_row("预期IC", str(expected_ic))
            
            # IC评级
            if abs(ic_mean) > 0.05:
                ic_rating = "优秀"
                ic_color = "green"
            elif abs(ic_mean) > 0.03:
                ic_rating = "良好"
                ic_color = "yellow"
            elif abs(ic_mean) > 0.01:
                ic_rating = "一般"
                ic_color = "blue"
            else:
                ic_rating = "较差"
                ic_color = "red"
            
            table.add_row("IC评级", Text(ic_rating, style=ic_color))
            
            # 组合内容
            content = Group(
                Text(f"📊 {name}", style="bold magenta"),
                "",
                table,
                "",
                Text("📝 因子描述:", style="bold"),
                Text(description, style="white"),
                "",
                Text("💡 使用建议:", style="bold"),
                self.generate_usage_suggestion(ic_mean, ic_ir)
            )
            
            panel = Panel(
                content,
                title="🔍 因子详细分析",
                border_style="bright_green",
                padding=(1, 1)
            )
            
            self.update(panel)
            
        except Exception as e:
            logger.error(f"因子分析显示失败: {e}")
            self.update(f"分析显示错误: {e}")
    
    def generate_usage_suggestion(self, ic_mean: float, ic_ir: float) -> Text:
        """生成使用建议"""
        if abs(ic_mean) > 0.05 and ic_ir > 0.5:
            return Text("• 高质量因子，建议重点使用\n• 可作为主要信号源\n• 建议权重: 15-20%", style="green")
        elif abs(ic_mean) > 0.03:
            return Text("• 中等质量因子，可配合使用\n• 建议与其他因子组合\n• 建议权重: 5-10%", style="yellow")
        else:
            return Text("• 因子效果一般，谨慎使用\n• 需要进一步优化\n• 建议权重: <5%", style="red")

class FactorLabScreen(Screen):
    """因子发现实验室主屏幕"""
    
    CSS = """
    .lab-layout {
        layout: horizontal;
        height: 1fr;
        margin: 1;
    }
    
    .left-panel {
        width: 30%;
        layout: vertical;
        margin-right: 1;
    }
    
    .factor-list {
        height: 1fr;
        border: solid $primary;
        margin: 1 0;
    }
    
    .right-panel {
        width: 70%;
        layout: vertical;
    }
    
    .action-buttons {
        layout: horizontal;
        height: 3;
        margin: 1 0;
    }
    
    .detail-tabs {
        height: 1fr;
        border: solid $secondary;
    }
    """
    
    BINDINGS = [
        Binding("d", "discover_factors", "因子发现"),
        Binding("o", "optimize_factors", "因子优化"),
        Binding("t", "test_factors", "因子测试"),
        Binding("r", "refresh_data", "刷新数据"),
        Binding("escape", "back", "返回"),
    ]
    
    def __init__(self):
        super().__init__()
        self.factors_data = {}
        self.selected_factor_id = None
        self.analysis_widget = None
        
    def compose(self) -> ComposeResult:
        with Container(classes="lab-layout"):
            # 左侧面板
            with Vertical(classes="left-panel"):
                yield Label("🔬 因子发现实验室")
                
                with Horizontal(classes="action-buttons"):
                    yield Button("发现", variant="primary", id="discover-btn")
                    yield Button("优化", variant="success", id="optimize-btn")
                    yield Button("测试", variant="warning", id="test-btn")
                
                # 因子列表
                yield DataTable(id="factor-table", classes="factor-list")
            
            # 右侧详情面板
            with Vertical(classes="right-panel"):
                yield Label("📊 因子分析")
                
                with TabbedContent(classes="detail-tabs"):
                    with TabPane("因子详情", id="detail-tab"):
                        yield Static("请选择一个因子查看详情", id="factor-detail")
                    
                    with TabPane("性能图表", id="chart-tab"):
                        yield Static("性能图表功能开发中...", id="factor-chart")
                    
                    with TabPane("回测结果", id="backtest-tab"):
                        yield Static("回测结果功能开发中...", id="factor-backtest")
    
    async def on_mount(self):
        """页面挂载初始化"""
        try:
            logger.info("因子发现实验室初始化...")
            
            # 初始化因子表格
            table = self.query_one("#factor-table", DataTable)
            table.add_columns("因子名称", "类型", "IC", "ICIR", "状态")
            
            # 加载已有因子
            await self.refresh_factors()
            
            # 设置定时刷新
            self.set_interval(30.0, self.refresh_factors)
            
            logger.info("因子发现实验室初始化完成")
            
        except Exception as e:
            logger.error(f"因子发现实验室初始化失败: {e}")
    
    async def refresh_factors(self):
        """刷新因子列表"""
        try:
            # 从数据库获取因子
            db_factors = await data_manager.get_factors()
            
            # 更新因子数据
            self.factors_data = {}
            for factor in db_factors:
                factor_id = factor["_id"]
                self.factors_data[factor_id] = factor
            
            await self.update_factor_table()
            
        except Exception as e:
            logger.error(f"因子列表刷新失败: {e}")
    
    async def update_factor_table(self):
        """更新因子表格"""
        try:
            table = self.query_one("#factor-table", DataTable)
            table.clear()
            
            for factor_id, factor in self.factors_data.items():
                name = factor.get("name", "未知因子")
                factor_type = factor.get("created_by", "unknown")
                ic_mean = factor.get("ic_mean", 0)
                icir = factor.get("ic_ir", 0)
                status = factor.get("status", "inactive")
                
                # 状态图标
                status_icons = {
                    "active": "🟢",
                    "testing": "🟡",
                    "inactive": "⚪",
                    "deprecated": "🔴"
                }
                status_display = f"{status_icons.get(status, '❓')} {status}"
                
                # 类型图标
                type_icons = {
                    "AI": "🤖",
                    "Manual": "👤",
                    "traditional": "📊"
                }
                type_display = f"{type_icons.get(factor_type, '❓')} {factor_type}"
                
                table.add_row(
                    name[:20] + "..." if len(name) > 20 else name,
                    type_display,
                    f"{ic_mean:.4f}",
                    f"{icir:.2f}",
                    status_display,
                    key=factor_id
                )
                
        except Exception as e:
            logger.error(f"因子表格更新失败: {e}")
    
    async def on_data_table_row_selected(self, event):
        """因子表格行选中事件"""
        try:
            if event.data_table.cursor_row >= 0:
                row_key = event.data_table.get_row_at(event.data_table.cursor_row).key
                self.selected_factor_id = row_key
                await self.update_factor_detail()
        except Exception as e:
            logger.error(f"因子选择失败: {e}")
    
    async def update_factor_detail(self):
        """更新因子详情"""
        try:
            if not self.selected_factor_id:
                return
                
            factor_data = self.factors_data.get(self.selected_factor_id)
            if not factor_data:
                return
            
            # 更新详情组件
            detail_widget = self.query_one("#factor-detail", Static)
            
            # 创建分析组件
            self.analysis_widget = FactorAnalysisWidget(factor_data)
            await self.analysis_widget.refresh_display()
            
            detail_widget.update(self.analysis_widget.renderable)
            
        except Exception as e:
            logger.error(f"因子详情更新失败: {e}")
    
    # 按钮事件处理
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击事件"""
        try:
            if event.button.id == "discover-btn":
                await self.action_discover_factors()
            elif event.button.id == "optimize-btn":
                await self.action_optimize_factors()
            elif event.button.id == "test-btn":
                await self.action_test_factors()
        except Exception as e:
            logger.error(f"按钮事件处理失败: {e}")
    
    # 快捷键动作
    async def action_discover_factors(self) -> None:
        """开始因子发现"""
        modal = FactorDiscoveryModal()
        result = await self.app.push_screen_wait(modal)
        
        if result and result.get("success"):
            discovered_count = len(result.get("factors", {}).get("discovered_factors", []))
            self.notify(f"成功发现 {discovered_count} 个新因子!", severity="information")
            await self.refresh_factors()
    
    async def action_optimize_factors(self) -> None:
        """优化因子"""
        if not self.selected_factor_id:
            self.notify("请先选择一个因子", severity="warning")
            return
        
        self.notify("因子优化功能开发中...", severity="information")
    
    async def action_test_factors(self) -> None:
        """测试因子"""
        if not self.selected_factor_id:
            self.notify("请先选择一个因子", severity="warning") 
            return
            
        self.notify("因子回测功能开发中...", severity="information")
    
    async def action_refresh_data(self) -> None:
        """刷新数据"""
        await self.refresh_factors()
        self.notify("因子数据已刷新", severity="information")
    
    async def action_back(self) -> None:
        """返回上一页"""
        self.app.pop_screen()

# 创建因子发现实验室实例的便捷函数
def create_factor_lab() -> FactorLabScreen:
    """创建因子发现实验室实例"""
    return FactorLabScreen()