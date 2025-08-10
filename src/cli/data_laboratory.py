"""
Data Laboratory CLI Interface
数据实验室CLI界面 - Bloomberg风格的时序数据生成实验室
"""

import asyncio
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.live import Live
from rich.columns import Columns
from rich.tree import Tree

from ..integration.model_service import get_ctbench_service, DataGenerationRequest
from ..integration.risk_control.enhanced_risk_manager import EnhancedRiskManager

class DataLaboratory:
    """数据实验室CLI界面"""
    
    def __init__(self):
        self.console = Console()
        self.ctbench_service = None
        self.risk_manager = None
        self.current_data = None
        self.generated_scenarios = {}
        self.experiment_history = []
        
    async def initialize(self):
        """初始化服务"""
        try:
            self.ctbench_service = await get_ctbench_service()
            self.risk_manager = EnhancedRiskManager()
            await self.risk_manager.initialize()
            
            self.console.print(Panel.fit(
                "[bold green]🧪 CTBench数据实验室已启动[/bold green]\n"
                "[dim]Bloomberg风格时序数据生成与分析平台[/dim]",
                border_style="green"
            ))
        except Exception as e:
            self.console.print(f"[red]初始化失败: {e}[/red]")
            raise
            
    def display_header(self):
        """显示标题界面"""
        header_text = Text()
        header_text.append("█▀▀ ▀█▀ █▄▄ █▀▀ █▄░█ █▀▀ █░█\n", style="bold cyan")
        header_text.append("█▄▄ ░█░ █▄█ ██▄ █░▀█ █▄▄ █▀█\n", style="bold cyan")
        header_text.append("DATA LABORATORY", style="bold white")
        
        info_panel = Panel(
            header_text,
            title="[bold blue]CTBench时序生成实验室[/bold blue]",
            border_style="blue"
        )
        
        self.console.print(info_panel)
        
    def show_main_menu(self) -> str:
        """显示主菜单"""
        menu_table = Table(title="📊 实验室主菜单", border_style="cyan")
        menu_table.add_column("选项", style="cyan", width=3)
        menu_table.add_column("功能", style="white", width=20)
        menu_table.add_column("描述", style="dim", width=40)
        
        menu_items = [
            ("1", "模型管理", "初始化、训练、查看TSG模型状态"),
            ("2", "数据生成", "生成合成时序数据和市场场景"),
            ("3", "风险实验", "压力测试和风险评估实验"),
            ("4", "数据分析", "分析生成数据的质量和特性"),
            ("5", "实时监控", "实时数据生成和风险监控"),
            ("6", "实验历史", "查看和管理实验历史记录"),
            ("7", "配置管理", "模型参数和系统配置"),
            ("0", "退出", "退出数据实验室")
        ]
        
        for option, feature, desc in menu_items:
            menu_table.add_row(option, feature, desc)
            
        self.console.print(menu_table)
        
        return Prompt.ask(
            "\n[cyan]请选择功能[/cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6", "7"],
            default="1"
        )
        
    async def model_management_menu(self):
        """模型管理菜单"""
        while True:
            self.console.clear()
            self.console.print(Panel("🤖 [bold]模型管理中心[/bold]", border_style="green"))
            
            # 显示模型状态
            await self.display_model_status()
            
            choice = Prompt.ask(
                "\n[green]选择操作[/green]",
                choices=["1", "2", "3", "4", "0"],
                default="1"
            )
            
            if choice == "1":
                await self.initialize_model()
            elif choice == "2":
                await self.train_model()
            elif choice == "3":
                await self.view_model_details()
            elif choice == "4":
                await self.load_save_model()
            elif choice == "0":
                break
                
    async def display_model_status(self):
        """显示模型状态"""
        try:
            status = self.ctbench_service.synthetic_manager.get_model_status()
            
            status_table = Table(title="📈 TSG模型状态", border_style="green")
            status_table.add_column("模型类型", style="cyan")
            status_table.add_column("状态", style="white")
            status_table.add_column("参数量", style="yellow")
            status_table.add_column("设备", style="magenta")
            
            for model_type, info in status.items():
                if info['initialized']:
                    model_info = info['info']
                    status_table.add_row(
                        model_type,
                        "[green]✓ 已初始化[/green]",
                        f"{model_info['parameters']:,}",
                        model_info['device']
                    )
                else:
                    status_table.add_row(
                        model_type,
                        "[dim]○ 未初始化[/dim]",
                        "-",
                        "-"
                    )
                    
            self.console.print(status_table)
            
            # 显示操作选项
            options_table = Table(border_style="dim")
            options_table.add_column("选项", width=3)
            options_table.add_column("操作")
            
            options = [
                ("1", "初始化模型"),
                ("2", "训练模型"),
                ("3", "查看详细信息"),
                ("4", "加载/保存模型"),
                ("0", "返回主菜单")
            ]
            
            for opt, desc in options:
                options_table.add_row(opt, desc)
                
            self.console.print(options_table)
            
        except Exception as e:
            self.console.print(f"[red]获取模型状态失败: {e}[/red]")
            
    async def initialize_model(self):
        """初始化模型"""
        model_types = ["timevae", "quantgan", "diffusion", "fourier"]
        
        model_type = Prompt.ask(
            "[cyan]选择要初始化的模型类型[/cyan]",
            choices=model_types,
            default="timevae"
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"正在初始化 {model_type} 模型...", total=None)
            
            try:
                success = self.ctbench_service.synthetic_manager.initialize_model(model_type)
                
                if success:
                    self.console.print(f"[green]✓ 模型 {model_type} 初始化成功![/green]")
                else:
                    self.console.print(f"[red]✗ 模型 {model_type} 初始化失败![/red]")
                    
            except Exception as e:
                self.console.print(f"[red]初始化错误: {e}[/red]")
                
        Prompt.ask("\n按回车键继续...", default="")
        
    async def data_generation_menu(self):
        """数据生成菜单"""
        while True:
            self.console.clear()
            self.console.print(Panel("🎲 [bold]数据生成实验室[/bold]", border_style="blue"))
            
            choice = Prompt.ask(
                "[blue]选择生成类型[/blue]\n"
                "1. 基础合成数据\n"
                "2. 市场场景生成\n"
                "3. 压力测试场景\n"
                "4. 批量场景实验\n"
                "0. 返回主菜单",
                choices=["1", "2", "3", "4", "0"],
                default="1"
            )
            
            if choice == "1":
                await self.generate_basic_synthetic_data()
            elif choice == "2":
                await self.generate_market_scenarios()
            elif choice == "3":
                await self.generate_stress_scenarios()
            elif choice == "4":
                await self.batch_scenario_experiment()
            elif choice == "0":
                break
                
    async def generate_basic_synthetic_data(self):
        """生成基础合成数据"""
        # 获取可用模型
        status = self.ctbench_service.synthetic_manager.get_model_status()
        available_models = [name for name, info in status.items() if info['initialized']]
        
        if not available_models:
            self.console.print("[red]没有可用的已初始化模型！[/red]")
            return
            
        model_type = Prompt.ask(
            "[blue]选择生成模型[/blue]",
            choices=available_models,
            default=available_models[0]
        )
        
        num_samples = IntPrompt.ask("[blue]生成样本数量[/blue]", default=100)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"生成 {num_samples} 个合成样本...", total=100)
            
            try:
                result = self.ctbench_service.synthetic_manager.generate_synthetic_data(
                    model_type, num_samples
                )
                
                progress.update(task, advance=100)
                
                if result['success']:
                    self.current_data = result['data']
                    self.console.print(f"[green]✓ 成功生成 {result['num_samples']} 个样本![/green]")
                    self.console.print(f"数据形状: {result['shape']}")
                    
                    # 显示数据统计
                    self.display_data_statistics(self.current_data)
                    
                    # 询问是否保存
                    if Confirm.ask("是否保存生成的数据?"):
                        await self.save_generated_data(self.current_data, f"{model_type}_synthetic")
                        
                else:
                    self.console.print(f"[red]✗ 生成失败: {result.get('error', '未知错误')}[/red]")
                    
            except Exception as e:
                self.console.print(f"[red]生成错误: {e}[/red]")
                
        Prompt.ask("\n按回车键继续...", default="")
        
    async def generate_market_scenarios(self):
        """生成市场场景"""
        scenario_types = ['black_swan', 'bull_market', 'bear_market', 'high_volatility', 'sideways']
        
        scenario_type = Prompt.ask(
            "[blue]选择市场场景类型[/blue]",
            choices=scenario_types,
            default="black_swan"
        )
        
        # 需要基础数据
        if self.current_data is None:
            self.console.print("[yellow]需要基础数据！正在生成示例基础数据...[/yellow]")
            await self.generate_sample_base_data()
            
        num_scenarios = IntPrompt.ask("[blue]生成场景数量[/blue]", default=50)
        
        with Progress() as progress:
            task = progress.add_task(f"生成 {scenario_type} 场景...", total=100)
            
            try:
                # 准备基础数据（取第一个样本作为基础）
                base_data = self.current_data[0] if len(self.current_data.shape) == 3 else self.current_data
                
                result = self.ctbench_service.synthetic_manager.generate_market_scenarios(
                    scenario_type, base_data, num_scenarios
                )
                
                progress.update(task, advance=100)
                
                if result['success']:
                    scenario_data = result['data']
                    self.generated_scenarios[scenario_type] = scenario_data
                    
                    self.console.print(f"[green]✓ 成功生成 {scenario_type} 场景![/green]")
                    self.console.print(f"场景数据形状: {scenario_data.shape}")
                    
                    # 分析场景特性
                    self.analyze_scenario_characteristics(scenario_data, scenario_type)
                    
                else:
                    self.console.print(f"[red]✗ 场景生成失败: {result.get('error')}[/red]")
                    
            except Exception as e:
                self.console.print(f"[red]场景生成错误: {e}[/red]")
                
        Prompt.ask("\n按回车键继续...", default="")
        
    async def generate_sample_base_data(self):
        """生成示例基础数据"""
        # 生成简单的价格数据作为基础
        days = 60
        initial_price = 100
        
        # 生成随机价格走势
        returns = np.random.normal(0.001, 0.02, days)  # 0.1%均值，2%波动率
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
            
        prices = np.array(prices[1:])
        
        # 构造OHLCV数据
        ohlcv_data = np.zeros((days, 6))
        
        for i in range(days):
            price = prices[i]
            # 简化的OHLC生成
            daily_range = price * 0.02 * np.random.random()
            high = price + daily_range * np.random.random()
            low = price - daily_range * np.random.random()
            
            ohlcv_data[i] = [
                price,  # Open
                high,   # High
                low,    # Low
                price * (1 + returns[i]),  # Close
                np.random.uniform(1000, 10000),  # Volume
                price   # Adjusted close
            ]
            
        self.current_data = ohlcv_data.reshape(1, days, 6)
        self.console.print("[green]✓ 示例基础数据已生成[/green]")
        
    def display_data_statistics(self, data: np.ndarray):
        """显示数据统计信息"""
        stats_table = Table(title="📊 数据统计", border_style="cyan")
        stats_table.add_column("指标", style="cyan")
        stats_table.add_column("数值", style="white")
        
        if len(data.shape) == 3:
            # (samples, sequence, features)
            avg_data = np.mean(data, axis=0)  # 对样本维度求平均
            feature_means = np.mean(avg_data, axis=0)  # 对序列维度求平均
            feature_stds = np.std(avg_data, axis=0)
            
            stats_table.add_row("样本数量", str(data.shape[0]))
            stats_table.add_row("序列长度", str(data.shape[1]))
            stats_table.add_row("特征数量", str(data.shape[2]))
            
            for i in range(min(6, data.shape[2])):
                feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
                name = feature_names[i] if i < len(feature_names) else f'Feature_{i}'
                stats_table.add_row(f"{name} 均值", f"{feature_means[i]:.4f}")
                stats_table.add_row(f"{name} 标准差", f"{feature_stds[i]:.4f}")
                
        self.console.print(stats_table)
        
    def analyze_scenario_characteristics(self, scenario_data: np.ndarray, scenario_type: str):
        """分析场景特性"""
        char_table = Table(title=f"🎯 {scenario_type} 场景特性", border_style="yellow")
        char_table.add_column("特性", style="yellow")
        char_table.add_column("数值", style="white")
        char_table.add_column("描述", style="dim")
        
        try:
            # 计算收益率分布
            all_returns = []
            max_drawdowns = []
            volatilities = []
            
            for scenario in scenario_data:
                prices = scenario[:, 0]  # 假设第0列是价格
                returns = np.diff(prices) / prices[:-1]
                all_returns.extend(returns)
                
                # 最大回撤
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdowns.append(np.min(drawdown))
                
                # 波动率
                volatilities.append(np.std(returns))
                
            all_returns = np.array(all_returns)
            
            # 统计特性
            char_table.add_row("场景数量", str(len(scenario_data)), "生成的场景总数")
            char_table.add_row("平均收益率", f"{np.mean(all_returns):.6f}", "所有场景平均日收益")
            char_table.add_row("收益率标准差", f"{np.std(all_returns):.6f}", "收益率波动程度")
            char_table.add_row("最大单日涨幅", f"{np.max(all_returns):.4f}", "最大单日涨幅")
            char_table.add_row("最大单日跌幅", f"{np.min(all_returns):.4f}", "最大单日跌幅")
            char_table.add_row("平均最大回撤", f"{np.mean(max_drawdowns):.4f}", "平均最大回撤幅度")
            char_table.add_row("平均波动率", f"{np.mean(volatilities):.4f}", "平均日波动率")
            
            # 极端事件统计
            extreme_up = np.sum(all_returns > 0.05)  # 5%以上涨幅
            extreme_down = np.sum(all_returns < -0.05)  # 5%以上跌幅
            char_table.add_row("极端上涨事件", str(extreme_up), "单日涨幅>5%的次数")
            char_table.add_row("极端下跌事件", str(extreme_down), "单日跌幅>5%的次数")
            
            self.console.print(char_table)
            
        except Exception as e:
            self.console.print(f"[red]场景特性分析失败: {e}[/red]")
            
    async def risk_experiment_menu(self):
        """风险实验菜单"""
        while True:
            self.console.clear()
            self.console.print(Panel("⚠️ [bold]风险实验中心[/bold]", border_style="red"))
            
            choice = Prompt.ask(
                "[red]选择实验类型[/red]\n"
                "1. 压力测试实验\n"
                "2. 黑天鹅概率分析\n"
                "3. 投资组合风险评估\n"
                "4. VaR反测试\n"
                "0. 返回主菜单",
                choices=["1", "2", "3", "4", "0"],
                default="1"
            )
            
            if choice == "1":
                await self.stress_test_experiment()
            elif choice == "2":
                await self.black_swan_analysis()
            elif choice == "3":
                await self.portfolio_risk_assessment()
            elif choice == "4":
                await self.var_backtesting()
            elif choice == "0":
                break
                
    async def stress_test_experiment(self):
        """压力测试实验"""
        if self.current_data is None:
            await self.generate_sample_base_data()
            
        self.console.print("[yellow]准备压力测试数据...[/yellow]")
        
        with Progress() as progress:
            task = progress.add_task("运行压力测试...", total=100)
            
            try:
                base_data = self.current_data[0] if len(self.current_data.shape) == 3 else self.current_data
                
                result = await self.ctbench_service.generate_stress_test_scenarios(
                    base_data.reshape(1, base_data.shape[0], base_data.shape[1])
                )
                
                progress.update(task, advance=100)
                
                if result['success']:
                    stress_scenarios = result['stress_scenarios']
                    
                    # 分析压力测试结果
                    stress_table = Table(title="🔥 压力测试结果", border_style="red")
                    stress_table.add_column("场景类型", style="red")
                    stress_table.add_column("场景数量", style="white")
                    stress_table.add_column("最大损失", style="yellow")
                    stress_table.add_column("平均损失", style="cyan")
                    stress_table.add_column("损失分布", style="dim")
                    
                    for scenario_type, scenarios in stress_scenarios.items():
                        # 计算损失分布
                        scenario_losses = []
                        for scenario in scenarios:
                            returns = np.diff(scenario[:, 0]) / scenario[:-1, 0]
                            total_loss = -np.sum(returns)
                            scenario_losses.append(total_loss)
                            
                        max_loss = np.max(scenario_losses)
                        avg_loss = np.mean(scenario_losses)
                        loss_95 = np.percentile(scenario_losses, 95)
                        
                        stress_table.add_row(
                            scenario_type,
                            str(len(scenarios)),
                            f"{max_loss:.2%}",
                            f"{avg_loss:.2%}",
                            f"95%分位: {loss_95:.2%}"
                        )
                        
                    self.console.print(stress_table)
                    
                    # 保存压力测试结果
                    self.generated_scenarios.update(stress_scenarios)
                    
                else:
                    self.console.print(f"[red]✗ 压力测试失败: {result.get('error')}[/red]")
                    
            except Exception as e:
                self.console.print(f"[red]压力测试错误: {e}[/red]")
                
        Prompt.ask("\n按回车键继续...", default="")
        
    async def save_generated_data(self, data: np.ndarray, filename_prefix: str):
        """保存生成的数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.npy"
            
            np.save(filename, data)
            self.console.print(f"[green]✓ 数据已保存到 {filename}[/green]")
            
            # 同时保存元数据
            metadata = {
                'timestamp': timestamp,
                'shape': list(data.shape),
                'filename': filename,
                'type': filename_prefix
            }
            
            with open(f"{filename_prefix}_{timestamp}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.console.print(f"[red]保存数据失败: {e}[/red]")
            
    def show_real_time_monitor(self):
        """显示实时监控界面"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # 创建实时更新的组件
        def make_header():
            return Panel(
                f"🔴 LIVE | CTBench实时监控 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="bold red"
            )
            
        def make_status_panel():
            service_stats = {"running": True, "models_loaded": 2, "requests_processed": 150}
            
            status_text = Text()
            status_text.append("🟢 服务状态: 运行中\n", style="green")
            status_text.append(f"📊 已加载模型: {service_stats['models_loaded']}\n", style="cyan")
            status_text.append(f"📈 处理请求: {service_stats['requests_processed']}\n", style="yellow")
            
            return Panel(status_text, title="服务状态", border_style="green")
            
        def make_metrics_panel():
            # 模拟实时指标
            metrics = {
                'generation_rate': f"{np.random.uniform(15, 25):.1f}/min",
                'avg_latency': f"{np.random.uniform(0.1, 0.5):.2f}s",
                'queue_size': np.random.randint(0, 10),
                'memory_usage': f"{np.random.uniform(45, 65):.1f}%"
            }
            
            metrics_table = Table(border_style="blue")
            metrics_table.add_column("指标", style="blue")
            metrics_table.add_column("数值", style="white")
            
            metrics_table.add_row("生成速率", metrics['generation_rate'])
            metrics_table.add_row("平均延迟", metrics['avg_latency'])
            metrics_table.add_row("队列长度", str(metrics['queue_size']))
            metrics_table.add_row("内存使用", metrics['memory_usage'])
            
            return Panel(metrics_table, title="性能指标", border_style="blue")
            
        def make_footer():
            return Panel(
                "按 [bold]Ctrl+C[/bold] 退出实时监控",
                style="dim"
            )
            
        # 实时更新循环
        with Live(layout, refresh_per_second=4) as live:
            try:
                while True:
                    layout["header"].update(make_header())
                    layout["left"].update(make_status_panel())
                    layout["right"].update(make_metrics_panel())
                    layout["footer"].update(make_footer())
                    
                    asyncio.sleep(0.25)  # 4Hz更新
            except KeyboardInterrupt:
                self.console.print("\n[yellow]实时监控已停止[/yellow]")
                
    async def run(self):
        """运行数据实验室"""
        await self.initialize()
        
        try:
            while True:
                self.console.clear()
                self.display_header()
                
                choice = self.show_main_menu()
                
                if choice == "0":
                    self.console.print("[yellow]感谢使用CTBench数据实验室![/yellow]")
                    break
                elif choice == "1":
                    await self.model_management_menu()
                elif choice == "2":
                    await self.data_generation_menu()
                elif choice == "3":
                    await self.risk_experiment_menu()
                elif choice == "4":
                    self.analyze_generated_data()
                elif choice == "5":
                    self.show_real_time_monitor()
                elif choice == "6":
                    self.show_experiment_history()
                elif choice == "7":
                    self.configuration_menu()
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]程序已退出[/yellow]")
        except Exception as e:
            self.console.print(f"[red]运行错误: {e}[/red]")
            
    def analyze_generated_data(self):
        """分析生成数据"""
        if not self.generated_scenarios and self.current_data is None:
            self.console.print("[red]没有可分析的数据！[/red]")
            Prompt.ask("\n按回车键继续...", default="")
            return
            
        self.console.clear()
        self.console.print(Panel("📊 [bold]数据分析报告[/bold]", border_style="cyan"))
        
        if self.current_data is not None:
            self.display_data_statistics(self.current_data)
            
        if self.generated_scenarios:
            for scenario_type, scenario_data in self.generated_scenarios.items():
                self.analyze_scenario_characteristics(scenario_data, scenario_type)
                
        Prompt.ask("\n按回车键继续...", default="")
        
    def show_experiment_history(self):
        """显示实验历史"""
        if not self.experiment_history:
            self.console.print("[yellow]没有实验历史记录[/yellow]")
        else:
            history_table = Table(title="📋 实验历史", border_style="magenta")
            history_table.add_column("时间", style="magenta")
            history_table.add_column("实验类型", style="white")
            history_table.add_column("结果", style="green")
            
            for exp in self.experiment_history[-10:]:  # 显示最近10条
                history_table.add_row(
                    exp['timestamp'],
                    exp['type'],
                    exp['result']
                )
                
            self.console.print(history_table)
            
        Prompt.ask("\n按回车键继续...", default="")
        
    def configuration_menu(self):
        """配置管理菜单"""
        self.console.print("[yellow]配置管理功能开发中...[/yellow]")
        Prompt.ask("\n按回车键继续...", default="")

async def main():
    """主函数"""
    lab = DataLaboratory()
    await lab.run()

if __name__ == "__main__":
    asyncio.run(main())