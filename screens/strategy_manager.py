"""
策略管理界面 - 完整的策略生命周期管理
支持策略创建、编辑、启停、监控和性能分析
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.console import Group
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Static, DataTable, Button, Input, Select, TextArea, 
    Checkbox, Label, ProgressBar, TabPane, TabbedContent
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from loguru import logger

from config.bloomberg_theme import BLOOMBERG_COLORS, STATUS_INDICATORS
from core.data_manager import data_manager
from core.ai_engine import ai_engine
from core.strategy_engine import strategy_engine, StrategyStatus

class StrategyCreateModal(ModalScreen):
    """策略创建模态对话框"""
    
    CSS = """
    StrategyCreateModal {
        align: center middle;
    }
    
    .create-modal {
        width: 80;
        height: 25;
        background: $surface;
        border: thick $primary;
    }
    
    .form-field {
        margin: 1 0;
    }
    
    .button-row {
        layout: horizontal;
        height: 3;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "取消"),
    ]
    
    def __init__(self, strategy_type: str = "manual"):
        super().__init__()
        self.strategy_type = strategy_type
        self.result = None
        
    def compose(self) -> ComposeResult:
        with Container(classes="create-modal"):
            yield Label("🚀 创建新策略", classes="form-field")
            
            yield Label("策略名称:", classes="form-field")
            yield Input(placeholder="输入策略名称...", id="strategy-name", classes="form-field")
            
            yield Label("策略类型:", classes="form-field")
            yield Select([
                ("网格策略", "grid"),
                ("定投策略", "dca"), 
                ("AI生成策略", "ai_generated"),
                ("手动策略", "manual")
            ], value=self.strategy_type, id="strategy-type", classes="form-field")
            
            yield Label("交易对:", classes="form-field")
            yield Select([
                ("BTC/USDT", "BTC-USDT"),
                ("ETH/USDT", "ETH-USDT"),
                ("BNB/USDT", "BNB-USDT"),
                ("SOL/USDT", "SOL-USDT")
            ], id="symbol-select", classes="form-field")
            
            yield Label("初始资金 (USDT):", classes="form-field")
            yield Input(placeholder="100.0", id="initial-capital", classes="form-field")
            
            with Horizontal(classes="button-row"):
                yield Button("创建策略", variant="primary", id="create-btn")
                yield Button("取消", variant="default", id="cancel-btn")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-btn":
            await self.create_strategy()
        else:
            self.dismiss()
    
    async def create_strategy(self):
        """创建策略"""
        try:
            name = self.query_one("#strategy-name", Input).value
            strategy_type = self.query_one("#strategy-type", Select).value
            symbol = self.query_one("#symbol-select", Select).value
            capital = float(self.query_one("#initial-capital", Input).value or "100.0")
            
            if not name:
                self.notify("请输入策略名称", severity="error")
                return
            
            # 构建策略配置
            config = {
                "symbol": symbol,
                "initial_capital": capital,
                "max_position_size": 0.8,
            }
            
            # 根据策略类型添加特定配置
            if strategy_type == "grid":
                config.update({
                    "grid_count": 10,
                    "price_range": 0.1,  # 10%价格区间
                    "quantity_per_grid": 0.001
                })
            elif strategy_type == "dca":
                config.update({
                    "interval_minutes": 60,  # 1小时定投
                    "buy_amount": 0.001
                })
            elif strategy_type == "ai_generated":
                config.update({
                    "ai_model": "gemini",
                    "analysis_interval": 1800,  # 30分钟分析
                    "position_size": 0.001
                })
            
            # 保存策略到数据库
            strategy_data = {
                "name": name,
                "type": strategy_type,
                "config": config,
                "status": "draft",
                "generated_by": "manual"
            }
            
            strategy_id = await data_manager.save_strategy(strategy_data)
            
            self.result = {
                "success": True,
                "strategy_id": strategy_id,
                "strategy_data": strategy_data
            }
            
            self.dismiss(self.result)
            
        except Exception as e:
            logger.error(f"创建策略失败: {e}")
            self.notify(f"创建失败: {e}", severity="error")

class AIStrategyModal(ModalScreen):
    """AI策略生成模态对话框"""
    
    CSS = """
    AIStrategyModal {
        align: center middle;
    }
    
    .ai-modal {
        width: 90;
        height: 30;
        background: $surface;
        border: thick $primary;
    }
    
    .form-field {
        margin: 1 0;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.result = None
        self.generating = False
        
    def compose(self) -> ComposeResult:
        with Container(classes="ai-modal"):
            yield Label("🤖 AI策略生成器", classes="form-field")
            
            yield Label("策略描述 (自然语言):", classes="form-field")
            yield TextArea(
                placeholder="例如：创建一个基于RSI和MACD的比特币交易策略，当RSI低于30时买入，高于70时卖出...",
                id="strategy-description",
                classes="form-field"
            )
            
            yield Label("交易对:", classes="form-field")
            yield Select([
                ("BTC/USDT", "BTC-USDT"),
                ("ETH/USDT", "ETH-USDT")
            ], id="ai-symbol", classes="form-field")
            
            yield Label("风险偏好:", classes="form-field")
            yield Select([
                ("保守型", "conservative"),
                ("平衡型", "balanced"),
                ("激进型", "aggressive")
            ], id="risk-level", classes="form-field")
            
            yield ProgressBar(id="ai-progress", show_eta=False)
            
            with Horizontal():
                yield Button("生成策略", variant="primary", id="generate-btn")
                yield Button("取消", variant="default", id="cancel-btn")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate-btn" and not self.generating:
            await self.generate_ai_strategy()
        else:
            self.dismiss()
    
    async def generate_ai_strategy(self):
        """生成AI策略"""
        try:
            self.generating = True
            progress = self.query_one("#ai-progress", ProgressBar)
            generate_btn = self.query_one("#generate-btn", Button)
            
            generate_btn.disabled = True
            progress.update(progress=10)
            
            # 获取输入参数
            description = self.query_one("#strategy-description", TextArea).text
            symbol = self.query_one("#ai-symbol", Select).value
            risk_level = self.query_one("#risk-level", Select).value
            
            if not description.strip():
                self.notify("请输入策略描述", severity="error")
                return
            
            progress.update(progress=30)
            
            # 准备AI生成参数
            requirements = {
                "strategy_type": "ai_generated",
                "description": description,
                "symbols": [symbol],
                "risk_level": risk_level,
                "max_capital": 500.0,
                "timeframe": "1h"
            }
            
            progress.update(progress=60)
            
            # 调用AI生成策略
            strategy_result = await ai_engine.generate_trading_strategy(requirements)
            
            progress.update(progress=100)
            
            if strategy_result and "code" in strategy_result:
                self.result = {
                    "success": True,
                    "strategy_data": strategy_result
                }
                self.notify("AI策略生成成功!", severity="information")
                self.dismiss(self.result)
            else:
                self.notify("AI策略生成失败", severity="error")
                
        except Exception as e:
            logger.error(f"AI策略生成失败: {e}")
            self.notify(f"生成失败: {e}", severity="error")
        finally:
            self.generating = False
            generate_btn.disabled = False

class StrategyPerformanceWidget(Static):
    """策略性能展示组件"""
    
    def __init__(self, strategy_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy_id = strategy_id
        self.performance_data = {}
        
    async def update_performance(self):
        """更新策略性能数据"""
        try:
            # 获取策略交易记录
            trades = await data_manager.get_trades(strategy_id=self.strategy_id, limit=100)
            
            if not trades:
                self.performance_data = {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "max_drawdown": 0,
                    "profit_factor": 0
                }
            else:
                # 计算性能指标
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if float(trade.get("pnl", 0)) > 0)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                total_pnl = sum(float(trade.get("pnl", 0)) for trade in trades)
                avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                profits = [float(trade.get("pnl", 0)) for trade in trades if float(trade.get("pnl", 0)) > 0]
                losses = [abs(float(trade.get("pnl", 0))) for trade in trades if float(trade.get("pnl", 0)) < 0]
                
                gross_profit = sum(profits) if profits else 0
                gross_loss = sum(losses) if losses else 0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
                
                # 简化的最大回撤计算
                cumulative_pnl = []
                running_pnl = 0
                for trade in trades:
                    running_pnl += float(trade.get("pnl", 0))
                    cumulative_pnl.append(running_pnl)
                
                if cumulative_pnl:
                    peak = cumulative_pnl[0]
                    max_drawdown = 0
                    for pnl in cumulative_pnl:
                        if pnl > peak:
                            peak = pnl
                        drawdown = (peak - pnl) / peak if peak != 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)
                else:
                    max_drawdown = 0
                
                self.performance_data = {
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                    "max_drawdown": max_drawdown * 100,
                    "profit_factor": profit_factor
                }
            
            await self.refresh_display()
            
        except Exception as e:
            logger.error(f"策略性能更新失败: {e}")
    
    async def refresh_display(self):
        """刷新性能显示"""
        try:
            # 创建性能指标表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("指标", style="cyan")
            table.add_column("数值", justify="right")
            
            data = self.performance_data
            
            # PnL颜色
            pnl_color = "green" if data.get("total_pnl", 0) >= 0 else "red"
            
            table.add_row("总交易数", str(data.get("total_trades", 0)))
            table.add_row("胜率", f"{data.get('win_rate', 0):.1f}%")
            table.add_row(
                "总盈亏", 
                Text(f"{data.get('total_pnl', 0):.2f} USDT", style=pnl_color)
            )
            table.add_row("平均盈亏", f"{data.get('avg_pnl', 0):.2f} USDT")
            table.add_row("最大回撤", f"{data.get('max_drawdown', 0):.1f}%")
            
            pf = data.get('profit_factor', 0)
            if pf == float('inf'):
                pf_text = "∞"
            else:
                pf_text = f"{pf:.2f}"
            table.add_row("盈亏比", pf_text)
            
            content = Panel(
                table,
                title="📊 策略绩效",
                border_style="bright_green"
            )
            
            self.update(content)
            
        except Exception as e:
            logger.error(f"性能显示刷新失败: {e}")

class StrategyManagerScreen(Screen):
    """策略管理主屏幕"""
    
    CSS = """
    .strategy-layout {
        layout: horizontal;
        height: 1fr;
        margin: 1;
    }
    
    .strategy-list {
        width: 60%;
        margin-right: 1;
    }
    
    .strategy-detail {
        width: 40%;
        layout: vertical;
    }
    
    .action-buttons {
        layout: horizontal;
        height: 3;
        margin: 1 0;
    }
    
    .detail-tabs {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("c", "create_strategy", "创建策略"),
        Binding("a", "ai_strategy", "AI策略"),
        Binding("r", "refresh_data", "刷新数据"),
        Binding("escape", "back", "返回"),
    ]
    
    def __init__(self):
        super().__init__()
        self.strategies = {}
        self.selected_strategy_id = None
        self.performance_widget = None
        
    def compose(self) -> ComposeResult:
        with Container(classes="strategy-layout"):
            # 左侧策略列表
            with Vertical(classes="strategy-list"):
                yield Label("🚀 策略管理")
                
                with Horizontal(classes="action-buttons"):
                    yield Button("创建策略", variant="primary", id="create-btn")
                    yield Button("AI策略", variant="success", id="ai-btn") 
                    yield Button("刷新", variant="default", id="refresh-btn")
                
                # 策略表格
                yield DataTable(id="strategy-table")
            
            # 右侧策略详情
            with Vertical(classes="strategy-detail"):
                yield Label("📋 策略详情")
                
                with TabbedContent(id="detail-tabs", classes="detail-tabs"):
                    with TabPane("基本信息", id="info-tab"):
                        yield Static("请选择一个策略查看详情", id="strategy-info")
                    
                    with TabPane("绩效分析", id="performance-tab"):
                        yield Static("绩效数据加载中...", id="strategy-performance")
                    
                    with TabPane("操作控制", id="control-tab"):
                        with Vertical():
                            yield Button("启动策略", variant="success", id="start-btn")
                            yield Button("暂停策略", variant="warning", id="pause-btn")
                            yield Button("停止策略", variant="error", id="stop-btn")
                            yield Button("删除策略", variant="error", id="delete-btn")
    
    async def on_mount(self):
        """页面挂载时初始化"""
        try:
            # 初始化策略表格
            table = self.query_one("#strategy-table", DataTable)
            table.add_columns("策略名称", "类型", "状态", "交易对", "PnL", "交易数")
            
            # 加载策略数据
            await self.refresh_strategies()
            
            # 设置定时刷新
            self.set_interval(5.0, self.refresh_strategies)
            
        except Exception as e:
            logger.error(f"策略管理页面初始化失败: {e}")
    
    async def refresh_strategies(self):
        """刷新策略列表"""
        try:
            # 获取数据库中的策略
            db_strategies = await data_manager.get_strategies()
            
            # 获取策略引擎中的运行时状态
            engine_status = strategy_engine.get_strategy_status()
            
            # 合并数据
            self.strategies = {}
            for strategy in db_strategies:
                strategy_id = strategy["_id"]
                runtime_status = engine_status.get(strategy_id, {})
                
                self.strategies[strategy_id] = {
                    **strategy,
                    "runtime_status": runtime_status.get("status", strategy["status"]),
                    "runtime_pnl": runtime_status.get("pnl", 0),
                    "runtime_trades": runtime_status.get("trades_count", 0)
                }
            
            await self.update_strategy_table()
            
        except Exception as e:
            logger.error(f"策略列表刷新失败: {e}")
    
    async def update_strategy_table(self):
        """更新策略表格"""
        try:
            table = self.query_one("#strategy-table", DataTable)
            table.clear()
            
            for strategy_id, strategy in self.strategies.items():
                # 状态指示器
                status = strategy.get("runtime_status", "draft")
                status_indicators = {
                    "active": "🟢",
                    "paused": "🟡", 
                    "stopped": "🔴",
                    "draft": "⚪",
                    "error": "❌"
                }
                status_display = f"{status_indicators.get(status, '❓')} {status}"
                
                # PnL显示
                pnl = float(strategy.get("runtime_pnl", 0))
                pnl_display = f"{pnl:+.2f} USDT"
                
                # 交易对
                symbol = strategy.get("config", {}).get("symbol", "N/A")
                
                table.add_row(
                    strategy["name"],
                    strategy["type"],
                    status_display,
                    symbol,
                    pnl_display,
                    str(strategy.get("runtime_trades", 0)),
                    key=strategy_id
                )
                
        except Exception as e:
            logger.error(f"策略表格更新失败: {e}")
    
    async def on_data_table_row_selected(self, event):
        """策略表格行选中事件"""
        try:
            if event.data_table.cursor_row >= 0:
                row_key = event.data_table.get_row_at(event.data_table.cursor_row).key
                self.selected_strategy_id = row_key
                await self.update_strategy_detail()
        except Exception as e:
            logger.error(f"策略选择失败: {e}")
    
    async def update_strategy_detail(self):
        """更新策略详情"""
        try:
            if not self.selected_strategy_id:
                return
                
            strategy = self.strategies.get(self.selected_strategy_id)
            if not strategy:
                return
            
            # 更新基本信息
            info_widget = self.query_one("#strategy-info", Static)
            
            info_text = f"""策略信息:
            
名称: {strategy['name']}
类型: {strategy['type']}  
状态: {strategy.get('runtime_status', 'draft')}
交易对: {strategy.get('config', {}).get('symbol', 'N/A')}
创建时间: {strategy.get('created_at', 'N/A')[:16]}
最后更新: {strategy.get('updated_at', 'N/A')[:16]}

配置参数:
{self._format_config(strategy.get('config', {}))}"""
            
            info_widget.update(info_text)
            
            # 更新绩效分析
            await self.update_performance_tab()
            
        except Exception as e:
            logger.error(f"策略详情更新失败: {e}")
    
    def _format_config(self, config: Dict) -> str:
        """格式化配置参数"""
        formatted = []
        for key, value in config.items():
            formatted.append(f"  {key}: {value}")
        return "\n".join(formatted) if formatted else "  无特殊配置"
    
    async def update_performance_tab(self):
        """更新绩效标签页"""
        try:
            if not self.selected_strategy_id:
                return
            
            # 创建或更新性能组件
            performance_container = self.query_one("#strategy-performance", Static)
            
            if not self.performance_widget or self.performance_widget.strategy_id != self.selected_strategy_id:
                self.performance_widget = StrategyPerformanceWidget(self.selected_strategy_id)
                
            await self.performance_widget.update_performance()
            
            # 更新显示
            performance_container.update(self.performance_widget.renderable)
            
        except Exception as e:
            logger.error(f"绩效标签页更新失败: {e}")
    
    # 按钮事件处理
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击事件"""
        try:
            if event.button.id == "create-btn":
                await self.action_create_strategy()
            elif event.button.id == "ai-btn":
                await self.action_ai_strategy()
            elif event.button.id == "refresh-btn":
                await self.action_refresh_data()
            elif event.button.id == "start-btn":
                await self.start_selected_strategy()
            elif event.button.id == "pause-btn":
                await self.pause_selected_strategy()
            elif event.button.id == "stop-btn":
                await self.stop_selected_strategy()
            elif event.button.id == "delete-btn":
                await self.delete_selected_strategy()
        except Exception as e:
            logger.error(f"按钮事件处理失败: {e}")
    
    # 快捷键动作
    async def action_create_strategy(self) -> None:
        """创建策略"""
        modal = StrategyCreateModal()
        result = await self.app.push_screen_wait(modal)
        
        if result and result.get("success"):
            self.notify("策略创建成功!", severity="information")
            await self.refresh_strategies()
    
    async def action_ai_strategy(self) -> None:
        """AI策略生成"""
        modal = AIStrategyModal()
        result = await self.app.push_screen_wait(modal)
        
        if result and result.get("success"):
            self.notify("AI策略生成成功!", severity="information")
            await self.refresh_strategies()
    
    async def action_refresh_data(self) -> None:
        """刷新数据"""
        await self.refresh_strategies()
        self.notify("数据已刷新", severity="information")
    
    async def action_back(self) -> None:
        """返回上一页"""
        self.app.pop_screen()
    
    # 策略操作
    async def start_selected_strategy(self):
        """启动选中的策略"""
        if not self.selected_strategy_id:
            self.notify("请先选择一个策略", severity="warning")
            return
            
        try:
            await strategy_engine.start_strategy(self.selected_strategy_id)
            self.notify("策略已启动", severity="information")
            await self.refresh_strategies()
        except Exception as e:
            self.notify(f"启动失败: {e}", severity="error")
    
    async def pause_selected_strategy(self):
        """暂停选中的策略"""
        if not self.selected_strategy_id:
            self.notify("请先选择一个策略", severity="warning")
            return
            
        try:
            await strategy_engine.pause_strategy(self.selected_strategy_id)
            self.notify("策略已暂停", severity="information")
            await self.refresh_strategies()
        except Exception as e:
            self.notify(f"暂停失败: {e}", severity="error")
    
    async def stop_selected_strategy(self):
        """停止选中的策略"""
        if not self.selected_strategy_id:
            self.notify("请先选择一个策略", severity="warning")
            return
            
        try:
            await strategy_engine.remove_strategy(self.selected_strategy_id)
            self.notify("策略已停止", severity="information")
            await self.refresh_strategies()
        except Exception as e:
            self.notify(f"停止失败: {e}", severity="error")
    
    async def delete_selected_strategy(self):
        """删除选中的策略"""
        if not self.selected_strategy_id:
            self.notify("请先选择一个策略", severity="warning")
            return
        
        # 这里应该弹出确认对话框，简化实现直接删除
        try:
            # 先停止策略
            await strategy_engine.remove_strategy(self.selected_strategy_id)
            
            # 从数据库删除
            await data_manager.update_strategy(self.selected_strategy_id, {"status": "deleted"})
            
            self.notify("策略已删除", severity="information")
            await self.refresh_strategies()
            
            # 清空选择
            self.selected_strategy_id = None
            
        except Exception as e:
            self.notify(f"删除失败: {e}", severity="error")

# 创建策略管理器实例的便捷函数
def create_strategy_manager() -> StrategyManagerScreen:
    """创建策略管理器实例"""
    return StrategyManagerScreen()