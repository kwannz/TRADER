"""
策略管理屏幕

Bloomberg Terminal风格的策略管理界面，支持：
- 策略创建和编辑
- 策略启动/暂停/停止控制
- 实时策略状态监控  
- 策略回测和优化
- 策略模板管理
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from textual.app import ComposeResult
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Static, Button, Input, Select, TextArea, Label,
    TabbedContent, TabPane, Switch, RadioSet, RadioButton,
    Collapsible, Rule
)
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.reactive import reactive
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align

from ..components.tables import StrategyTable, CustomDataTable
from ..components.status import PerformanceIndicator
from ..components.charts import PerformanceChart, IndicatorChart
from ..themes.bloomberg import BloombergTheme
from ...core.strategy_engine import strategy_engine, GridStrategy, DCAStrategy, AIStrategy
from ...core.data_manager import data_manager
from ...python_layer.utils.logger import get_logger

logger = get_logger(__name__)

class StrategyManagerScreen(Screen):
    """策略管理主屏幕"""
    
    CSS_PATH = "strategy_manager.css"
    BINDINGS = [
        Binding("escape", "app.pop_screen", "返回"),
        Binding("n", "new_strategy", "新建策略"),
        Binding("r", "refresh", "刷新"),
        Binding("s", "start_selected", "启动选中"),
        Binding("p", "pause_selected", "暂停选中"),
        Binding("t", "stop_selected", "停止选中"),
        Binding("d", "delete_selected", "删除选中"),
        Binding("b", "backtest_selected", "回测选中"),
    ]
    
    # 响应式数据
    strategies = reactive({})
    selected_strategy_id = reactive("")
    performance_data = reactive({})
    
    def __init__(self):
        super().__init__()
        self.theme = BloombergTheme()
        self.refresh_interval = 1.0  # 1Hz刷新频率
        self.update_task: Optional[asyncio.Task] = None
        
    def compose(self) -> ComposeResult:
        """构建策略管理界面"""
        
        with Container(id="strategy-manager-container"):
            # 顶部操作栏
            with Horizontal(id="strategy-toolbar", classes="toolbar"):
                yield Button("📝 新建策略", id="btn-new", variant="success")
                yield Button("▶️ 启动", id="btn-start", variant="primary")
                yield Button("⏸️ 暂停", id="btn-pause", variant="warning") 
                yield Button("⏹️ 停止", id="btn-stop", variant="error")
                yield Button("🗑️ 删除", id="btn-delete", variant="error")
                yield Button("📊 回测", id="btn-backtest", variant="default")
                yield Button("🔄 刷新", id="btn-refresh", variant="default")
            
            # 主内容区域
            with TabbedContent(id="strategy-tabs"):
                # 策略列表页
                with TabPane("策略列表", id="strategy-list-tab"):
                    with Horizontal(id="strategy-list-section"):
                        # 左侧策略表格
                        with Container(id="strategy-table-container", classes="panel"):
                            yield Label("🤖 策略列表", classes="panel-title")
                            yield StrategyTable(id="strategy-table")
                        
                        # 右侧策略详情
                        with Vertical(id="strategy-details-section", classes="panel"):
                            yield Label("📋 策略详情", classes="panel-title")
                            yield Static(id="strategy-details")
                            
                            yield Label("📈 实时状态", classes="panel-title")
                            yield PerformanceIndicator(id="strategy-performance")
                
                # 策略创建页
                with TabPane("创建策略", id="create-strategy-tab"):
                    yield StrategyCreationForm(id="strategy-creation-form")
                
                # 策略回测页
                with TabPane("回测分析", id="backtest-tab"):
                    with Vertical(id="backtest-section"):
                        with Horizontal(id="backtest-controls"):
                            yield Label("回测参数:", classes="form-label")
                            yield Input(placeholder="开始日期 (YYYY-MM-DD)", id="backtest-start-date")
                            yield Input(placeholder="结束日期 (YYYY-MM-DD)", id="backtest-end-date")
                            yield Button("开始回测", id="btn-run-backtest", variant="primary")
                        
                        with Container(id="backtest-results", classes="panel"):
                            yield Label("📊 回测结果", classes="panel-title")
                            yield Static(id="backtest-output")
                
                # 策略模板页
                with TabPane("模板管理", id="template-tab"):
                    with Horizontal(id="template-section"):
                        # 模板列表
                        with Container(id="template-list-container", classes="panel"):
                            yield Label("📋 策略模板", classes="panel-title")
                            yield CustomDataTable(
                                title="",
                                columns=["名称", "类型", "描述", "使用次数"],
                                id="template-table"
                            )
                        
                        # 模板详情
                        with Container(id="template-details-container", classes="panel"):
                            yield Label("🔍 模板详情", classes="panel-title")
                            yield Static(id="template-details")
                            yield Button("使用模板", id="btn-use-template", variant="success")

    def on_mount(self) -> None:
        """屏幕挂载时初始化"""
        try:
            self.start_data_updates()
            self._load_strategy_templates()
            logger.info("策略管理屏幕已挂载")
        except Exception as e:
            logger.error(f"策略管理屏幕挂载失败: {e}")

    def on_unmount(self) -> None:
        """屏幕卸载时清理"""
        try:
            self.stop_data_updates()
            logger.info("策略管理屏幕已卸载")
        except Exception as e:
            logger.error(f"策略管理屏幕卸载失败: {e}")

    def start_data_updates(self) -> None:
        """启动数据更新任务"""
        if self.update_task is None or self.update_task.done():
            self.update_task = asyncio.create_task(self._update_loop())

    def stop_data_updates(self) -> None:
        """停止数据更新任务"""
        if self.update_task and not self.update_task.done():
            self.update_task.cancel()

    async def _update_loop(self) -> None:
        """数据更新循环"""
        while True:
            try:
                # 获取策略状态
                self.strategies = strategy_engine.get_strategy_status()
                
                # 更新策略表格
                strategy_table = self.query_one("#strategy-table", StrategyTable)
                strategy_table.update_data(self.strategies)
                
                # 更新选中策略的详情
                if self.selected_strategy_id and self.selected_strategy_id in self.strategies:
                    self._update_strategy_details()
                
                await asyncio.sleep(self.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"策略管理数据更新错误: {e}")
                await asyncio.sleep(1)

    def _update_strategy_details(self) -> None:
        """更新策略详情显示"""
        try:
            if not self.selected_strategy_id or self.selected_strategy_id not in self.strategies:
                return
            
            strategy = self.strategies[self.selected_strategy_id]
            
            # 格式化策略详情
            details_text = f"""[bold cyan]策略名称:[/bold cyan] {strategy['name']}
[bold yellow]策略类型:[/bold yellow] {strategy.get('type', 'Unknown')}
[bold white]当前状态:[/bold white] {strategy['status']}
[bold green]总收益率:[/bold green] {strategy['pnl']:+.2f} USDT
[bold blue]交易次数:[/bold blue] {strategy['trades_count']}
[bold magenta]创建时间:[/bold magenta] {strategy['created_at']}
[bold dim]更新时间:[/bold dim] {strategy['updated_at']}

[bold]策略配置:[/bold]
{self._format_strategy_config(strategy)}"""
            
            details_widget = self.query_one("#strategy-details", Static)
            details_widget.update(details_text)
            
            # 更新性能指标
            perf_widget = self.query_one("#strategy-performance", PerformanceIndicator)
            perf_widget.update_metrics(
                latency=50.0,  # 模拟延迟
                throughput=100,  # 模拟吞吐量
                errors=0,
                uptime=3600  # 模拟运行时间
            )
            
        except Exception as e:
            logger.error(f"更新策略详情失败: {e}")

    def _format_strategy_config(self, strategy: Dict[str, Any]) -> str:
        """格式化策略配置信息"""
        try:
            # 这里应该根据策略类型格式化配置
            config = strategy.get('config', {})
            if not config:
                return "无配置信息"
            
            config_lines = []
            for key, value in config.items():
                config_lines.append(f"  {key}: {value}")
            
            return "\n".join(config_lines)
            
        except Exception as e:
            logger.error(f"格式化策略配置失败: {e}")
            return "配置信息解析错误"

    def _load_strategy_templates(self) -> None:
        """加载策略模板"""
        try:
            # 预定义策略模板
            templates = [
                {
                    "名称": "经典网格策略",
                    "类型": "网格交易",
                    "描述": "适合震荡市场的网格交易策略",
                    "使用次数": "23"
                },
                {
                    "名称": "DCA定投策略", 
                    "类型": "定投",
                    "描述": "定期定额投资策略，降低平均成本",
                    "使用次数": "15"
                },
                {
                    "名称": "AI趋势跟踪",
                    "类型": "AI策略", 
                    "描述": "基于机器学习的趋势识别和跟踪",
                    "使用次数": "8"
                },
                {
                    "名称": "RSI均值回归",
                    "类型": "技术指标",
                    "描述": "基于RSI指标的均值回归策略",
                    "使用次数": "12"
                }
            ]
            
            template_table = self.query_one("#template-table", CustomDataTable)
            template_table.set_data(templates)
            
        except Exception as e:
            logger.error(f"加载策略模板失败: {e}")

    # ============ 事件处理器 ============

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """按钮点击事件处理"""
        button_id = event.button.id
        
        try:
            if button_id == "btn-new":
                self.action_new_strategy()
            elif button_id == "btn-start":
                self.action_start_selected()
            elif button_id == "btn-pause":
                self.action_pause_selected()
            elif button_id == "btn-stop":
                self.action_stop_selected()
            elif button_id == "btn-delete":
                self.action_delete_selected()
            elif button_id == "btn-backtest":
                self.action_backtest_selected()
            elif button_id == "btn-refresh":
                self.action_refresh()
            elif button_id == "btn-run-backtest":
                self._run_backtest()
            elif button_id == "btn-use-template":
                self._use_template()
                
        except Exception as e:
            logger.error(f"按钮事件处理失败: {e}")

    def on_data_table_row_selected(self, event) -> None:
        """表格行选择事件"""
        try:
            # 获取选中的策略ID
            if event.data_table.id == "strategy-table":
                row_index = event.row_index
                if 0 <= row_index < len(self.strategies):
                    strategy_ids = list(self.strategies.keys())
                    self.selected_strategy_id = strategy_ids[row_index]
                    self._update_strategy_details()
                    
        except Exception as e:
            logger.error(f"表格选择事件处理失败: {e}")

    # ============ 动作处理器 ============

    def action_new_strategy(self) -> None:
        """新建策略"""
        try:
            # 切换到创建策略标签页
            tabs = self.query_one("#strategy-tabs", TabbedContent)
            tabs.active = "create-strategy-tab"
            logger.info("切换到策略创建页面")
        except Exception as e:
            logger.error(f"新建策略失败: {e}")

    def action_start_selected(self) -> None:
        """启动选中的策略"""
        if self.selected_strategy_id:
            asyncio.create_task(self._start_strategy(self.selected_strategy_id))

    def action_pause_selected(self) -> None:
        """暂停选中的策略"""
        if self.selected_strategy_id:
            asyncio.create_task(self._pause_strategy(self.selected_strategy_id))

    def action_stop_selected(self) -> None:
        """停止选中的策略"""
        if self.selected_strategy_id:
            asyncio.create_task(self._stop_strategy(self.selected_strategy_id))

    def action_delete_selected(self) -> None:
        """删除选中的策略"""
        if self.selected_strategy_id:
            # TODO: 显示确认对话框
            asyncio.create_task(self._delete_strategy(self.selected_strategy_id))

    def action_backtest_selected(self) -> None:
        """回测选中的策略"""
        if self.selected_strategy_id:
            # 切换到回测标签页
            tabs = self.query_one("#strategy-tabs", TabbedContent)
            tabs.active = "backtest-tab"

    def action_refresh(self) -> None:
        """刷新策略数据"""
        try:
            # 重启更新任务
            self.stop_data_updates()
            self.start_data_updates()
            logger.info("策略数据已刷新")
        except Exception as e:
            logger.error(f"刷新策略数据失败: {e}")

    # ============ 策略操作方法 ============

    async def _start_strategy(self, strategy_id: str) -> None:
        """启动策略"""
        try:
            await strategy_engine.start_strategy(strategy_id)
            logger.info(f"策略已启动: {strategy_id}")
        except Exception as e:
            logger.error(f"启动策略失败: {e}")

    async def _pause_strategy(self, strategy_id: str) -> None:
        """暂停策略"""
        try:
            await strategy_engine.pause_strategy(strategy_id)
            logger.info(f"策略已暂停: {strategy_id}")
        except Exception as e:
            logger.error(f"暂停策略失败: {e}")

    async def _stop_strategy(self, strategy_id: str) -> None:
        """停止策略"""
        try:
            await strategy_engine.remove_strategy(strategy_id)
            logger.info(f"策略已停止: {strategy_id}")
            self.selected_strategy_id = ""
        except Exception as e:
            logger.error(f"停止策略失败: {e}")

    async def _delete_strategy(self, strategy_id: str) -> None:
        """删除策略"""
        try:
            await strategy_engine.remove_strategy(strategy_id)
            # TODO: 同时从数据库删除
            logger.info(f"策略已删除: {strategy_id}")
            self.selected_strategy_id = ""
        except Exception as e:
            logger.error(f"删除策略失败: {e}")

    def _run_backtest(self) -> None:
        """运行回测"""
        try:
            # 获取回测参数
            start_date = self.query_one("#backtest-start-date", Input).value
            end_date = self.query_one("#backtest-end-date", Input).value
            
            if not start_date or not end_date:
                self._show_backtest_result("请输入有效的开始和结束日期")
                return
            
            # 模拟回测结果
            result = f"""[bold green]回测完成![/bold green]

[bold]回测参数:[/bold]
策略: {self.strategies.get(self.selected_strategy_id, {}).get('name', '未选择')}
时间范围: {start_date} 至 {end_date}

[bold]回测结果:[/bold]
总收益率: [green]+15.8%[/green]
最大回撤: [red]-5.2%[/red]
夏普比率: [yellow]1.34[/yellow]
胜率: [green]68.5%[/green]
总交易次数: 156
平均持仓时间: 2.3天

[bold]风险指标:[/bold]
年化波动率: 12.8%
最大连续亏损: 3次
最大单次亏损: -1.8%

[dim]回测基于历史数据，实际结果可能不同[/dim]"""
            
            self._show_backtest_result(result)
            
        except Exception as e:
            logger.error(f"回测运行失败: {e}")
            self._show_backtest_result(f"回测失败: {e}")

    def _show_backtest_result(self, result: str) -> None:
        """显示回测结果"""
        try:
            output_widget = self.query_one("#backtest-output", Static)
            output_widget.update(result)
        except Exception as e:
            logger.error(f"显示回测结果失败: {e}")

    def _use_template(self) -> None:
        """使用选中的模板"""
        try:
            # TODO: 获取选中的模板并填充到创建表单
            # 切换到创建策略页面
            tabs = self.query_one("#strategy-tabs", TabbedContent)
            tabs.active = "create-strategy-tab"
            
            logger.info("使用策略模板")
        except Exception as e:
            logger.error(f"使用模板失败: {e}")

class StrategyCreationForm(Container):
    """策略创建表单"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme = BloombergTheme()
        
    def compose(self) -> ComposeResult:
        """构建策略创建表单"""
        
        with Vertical(id="strategy-form"):
            # 基本信息
            with Container(id="basic-info-section", classes="form-section"):
                yield Label("📋 基本信息", classes="section-title")
                
                with Horizontal(classes="form-row"):
                    yield Label("策略名称:", classes="form-label")
                    yield Input(placeholder="输入策略名称", id="strategy-name")
                
                with Horizontal(classes="form-row"):
                    yield Label("策略类型:", classes="form-label")
                    yield Select([
                        ("网格策略", "grid"),
                        ("定投策略", "dca"), 
                        ("AI策略", "ai_generated")
                    ], id="strategy-type")
                
                with Horizontal(classes="form-row"):
                    yield Label("交易币种:", classes="form-label")
                    yield Input(placeholder="如: BTC/USDT", value="BTC/USDT", id="trading-symbol")
            
            # 策略参数
            with Container(id="strategy-params-section", classes="form-section"):
                yield Label("⚙️ 策略参数", classes="section-title")
                
                # 参数将根据策略类型动态显示
                yield Container(id="dynamic-params")
            
            # 风控设置
            with Container(id="risk-section", classes="form-section"):
                yield Label("🛡️ 风控设置", classes="section-title")
                
                with Horizontal(classes="form-row"):
                    yield Label("单次交易金额:", classes="form-label")
                    yield Input(placeholder="USDT", value="100", id="trade-amount")
                
                with Horizontal(classes="form-row"):
                    yield Label("止损比例:", classes="form-label")
                    yield Input(placeholder="如: 0.05 (5%)", value="0.05", id="stop-loss")
                
                with Horizontal(classes="form-row"):
                    yield Label("最大仓位比例:", classes="form-label")
                    yield Input(placeholder="如: 0.2 (20%)", value="0.2", id="max-position")
            
            # 操作按钮
            with Horizontal(id="form-actions", classes="form-actions"):
                yield Button("🚀 创建并启动", id="btn-create-start", variant="success")
                yield Button("💾 保存草稿", id="btn-save-draft", variant="primary")
                yield Button("🔄 重置表单", id="btn-reset", variant="default")
                yield Button("❌ 取消", id="btn-cancel", variant="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """策略类型选择改变事件"""
        if event.select.id == "strategy-type":
            self._update_dynamic_params(event.value)

    def _update_dynamic_params(self, strategy_type: str) -> None:
        """根据策略类型更新动态参数"""
        try:
            params_container = self.query_one("#dynamic-params", Container)
            
            # 清空现有参数
            params_container.remove_children()
            
            if strategy_type == "grid":
                # 网格策略参数
                with params_container:
                    with Horizontal(classes="form-row"):
                        yield Label("网格数量:", classes="form-label")
                        yield Input(placeholder="如: 10", value="10", id="grid-count")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("价格区间(%):", classes="form-label")  
                        yield Input(placeholder="如: 0.1 (10%)", value="0.1", id="price-range")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("每格交易量:", classes="form-label")
                        yield Input(placeholder="USDT", value="50", id="grid-amount")
                        
            elif strategy_type == "dca":
                # 定投策略参数
                with params_container:
                    with Horizontal(classes="form-row"):
                        yield Label("定投间隔(分钟):", classes="form-label")
                        yield Input(placeholder="如: 60", value="60", id="dca-interval")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("每次投入金额:", classes="form-label")
                        yield Input(placeholder="USDT", value="100", id="dca-amount")
                        
            elif strategy_type == "ai_generated":
                # AI策略参数
                with params_container:
                    with Horizontal(classes="form-row"):
                        yield Label("AI模型:", classes="form-label")
                        yield Select([
                            ("DeepSeek", "deepseek"),
                            ("Gemini", "gemini")
                        ], id="ai-model")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("分析间隔(分钟):", classes="form-label")
                        yield Input(placeholder="如: 30", value="30", id="ai-interval")
                    
                    with Horizontal(classes="form-row"):
                        yield Label("信心阈值:", classes="form-label")
                        yield Input(placeholder="如: 0.7", value="0.7", id="confidence-threshold")
            
        except Exception as e:
            logger.error(f"更新动态参数失败: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """表单按钮事件处理"""
        button_id = event.button.id
        
        if button_id == "btn-create-start":
            self._create_strategy(start=True)
        elif button_id == "btn-save-draft":
            self._create_strategy(start=False)
        elif button_id == "btn-reset":
            self._reset_form()
        elif button_id == "btn-cancel":
            # 返回策略列表
            pass

    def _create_strategy(self, start: bool = False) -> None:
        """创建策略"""
        try:
            # 收集表单数据
            form_data = self._collect_form_data()
            
            if not self._validate_form_data(form_data):
                return
            
            # 创建策略实例
            asyncio.create_task(self._create_strategy_async(form_data, start))
            
        except Exception as e:
            logger.error(f"创建策略失败: {e}")

    def _collect_form_data(self) -> Dict[str, Any]:
        """收集表单数据"""
        try:
            data = {
                "name": self.query_one("#strategy-name", Input).value,
                "type": self.query_one("#strategy-type", Select).value,
                "symbol": self.query_one("#trading-symbol", Input).value,
                "trade_amount": float(self.query_one("#trade-amount", Input).value or "0"),
                "stop_loss": float(self.query_one("#stop-loss", Input).value or "0"),
                "max_position": float(self.query_one("#max-position", Input).value or "0"),
            }
            
            # 根据策略类型收集特定参数
            strategy_type = data["type"]
            if strategy_type == "grid":
                data.update({
                    "grid_count": int(self.query_one("#grid-count", Input).value or "0"),
                    "price_range": float(self.query_one("#price-range", Input).value or "0"),
                    "quantity_per_grid": float(self.query_one("#grid-amount", Input).value or "0") / data["trade_amount"] if data["trade_amount"] > 0 else 0
                })
            elif strategy_type == "dca":
                data.update({
                    "interval_minutes": int(self.query_one("#dca-interval", Input).value or "0"),
                    "buy_amount": float(self.query_one("#dca-amount", Input).value or "0") / data["trade_amount"] if data["trade_amount"] > 0 else 0
                })
            elif strategy_type == "ai_generated":
                data.update({
                    "ai_model": self.query_one("#ai-model", Select).value,
                    "analysis_interval": int(self.query_one("#ai-interval", Input).value or "0") * 60,  # 转换为秒
                    "confidence_threshold": float(self.query_one("#confidence-threshold", Input).value or "0"),
                    "position_size": data["trade_amount"] / 50000 if data["trade_amount"] > 0 else 0.001  # 假设BTC价格50000
                })
            
            return data
            
        except Exception as e:
            logger.error(f"收集表单数据失败: {e}")
            return {}

    def _validate_form_data(self, data: Dict[str, Any]) -> bool:
        """验证表单数据"""
        try:
            if not data.get("name"):
                # TODO: 显示错误消息
                logger.error("策略名称不能为空")
                return False
            
            if not data.get("type"):
                logger.error("必须选择策略类型")
                return False
            
            if not data.get("symbol"):
                logger.error("交易币种不能为空")
                return False
            
            if data.get("trade_amount", 0) <= 0:
                logger.error("交易金额必须大于0")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证表单数据失败: {e}")
            return False

    async def _create_strategy_async(self, form_data: Dict[str, Any], start: bool) -> None:
        """异步创建策略"""
        try:
            import uuid
            
            # 生成策略ID
            strategy_id = str(uuid.uuid4())
            
            # 准备策略配置
            config = {k: v for k, v in form_data.items() 
                     if k not in ["name", "type"]}
            
            # 根据类型创建策略实例
            strategy_type = form_data["type"]
            if strategy_type == "grid":
                strategy = GridStrategy(strategy_id, form_data["name"], config)
            elif strategy_type == "dca":
                strategy = DCAStrategy(strategy_id, form_data["name"], config)
            elif strategy_type == "ai_generated":
                strategy = AIStrategy(strategy_id, form_data["name"], config)
            else:
                raise ValueError(f"不支持的策略类型: {strategy_type}")
            
            # 添加到策略引擎
            await strategy_engine.add_strategy(strategy)
            
            # 保存到数据库
            if hasattr(data_manager, '_initialized') and data_manager._initialized:
                strategy_data = {
                    "name": form_data["name"],
                    "type": strategy_type,
                    "config": config,
                    "status": "active" if start else "draft"
                }
                await data_manager.save_strategy(strategy_data)
            
            # 如果需要启动
            if start:
                await strategy_engine.start_strategy(strategy_id)
            
            logger.info(f"策略创建成功: {form_data['name']}")
            
            # 清空表单
            self._reset_form()
            
        except Exception as e:
            logger.error(f"异步创建策略失败: {e}")

    def _reset_form(self) -> None:
        """重置表单"""
        try:
            # 清空所有输入框
            for input_widget in self.query("Input"):
                if input_widget.id != "trading-symbol":  # 保留默认的交易对
                    input_widget.value = ""
            
            # 重置选择框
            for select_widget in self.query("Select"):
                select_widget.value = select_widget.options[0][1]
            
            # 清空动态参数
            params_container = self.query_one("#dynamic-params", Container)
            params_container.remove_children()
            
            logger.info("表单已重置")
            
        except Exception as e:
            logger.error(f"重置表单失败: {e}")