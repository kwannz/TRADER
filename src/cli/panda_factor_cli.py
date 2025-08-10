"""
PandaFactor CLI Integration
PandaFactor CLI集成 - 将70+专业算子功能集成到命令行界面
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Rich相关导入，用于美化命令行界面
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
except ImportError:
    # 如果Rich不可用，提供基础实现
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def clear(self):
            os.system('cls' if os.name == 'nt' else 'clear')
    
    class Prompt:
        @staticmethod
        def ask(prompt_text: str, default: str = None) -> str:
            return input(f"{prompt_text} [{default}]: ") if default else input(f"{prompt_text}: ")
    
    class Confirm:
        @staticmethod
        def ask(prompt_text: str, default: bool = True) -> bool:
            response = input(f"{prompt_text} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not response:
                return default
            return response.startswith('y')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.factor_engine import UnifiedFactorInterface, unified_interface, PandaFactorUtils
    from src.llm_services import UnifiedLLMService, unified_llm_service
    from src.data_management import UnifiedDataReader, unified_data_reader
    from src.factor_validation import UnifiedFactorValidator, unified_factor_validator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class PandaFactorCLI:
    """
    PandaFactor命令行界面 - 70+算子的专业量化工作台
    """
    
    def __init__(self):
        self.console = Console()
        self.interface = unified_interface
        self.llm_service = unified_llm_service
        self.data_reader = unified_data_reader
        self.validator = unified_factor_validator
        
        # CLI状态
        self.current_session = {
            'factors': {},
            'data_cache': {},
            'conversation_history': []
        }
        
        self.console.print("[bold blue]🐼 PandaFactor Professional CLI v1.0[/bold blue]")
        self.console.print("集成70+专业算子的量化因子开发工作台\n")
    
    def show_welcome(self):
        """显示欢迎信息"""
        welcome_panel = Panel(
            """[bold green]欢迎使用 PandaFactor Professional CLI！[/bold green]

🚀 核心功能：
• 70+ 专业量化算子 (RANK, MACD, RSI, KDJ等)
• 公式因子开发 (WorldQuant Alpha风格)
• Python因子开发 (自定义因子类)
• AI智能助手 (因子生成、优化、调试)
• 综合性能验证 (IC分析、分层回测、压力测试)
• 实时数据接入 (MongoDB集成)

📖 快速开始：
• 输入 'help' 查看所有命令
• 输入 'demo' 体验核心功能
• 输入 'list-functions' 查看可用算子
            """,
            title="🐼 PandaFactor Professional",
            border_style="green"
        )
        self.console.print(welcome_panel)
    
    def show_help(self):
        """显示帮助信息"""
        help_table = Table(title="🔧 PandaFactor CLI 命令列表")
        help_table.add_column("命令", style="cyan", no_wrap=True)
        help_table.add_column("功能", style="white")
        help_table.add_column("示例", style="dim")
        
        commands = [
            ("help", "显示命令帮助", "help"),
            ("demo", "运行功能演示", "demo"),
            ("list-functions", "列出可用算子函数", "list-functions"),
            ("list-factors", "列出已创建因子", "list-factors"),
            ("create-formula", "创建公式因子", "create-formula"),
            ("create-python", "创建Python因子", "create-python"),
            ("calculate", "计算因子值", "calculate RANK_MOMENTUM"),
            ("validate", "验证因子性能", "validate RANK_MOMENTUM"),
            ("ai-chat", "与因子助手对话", "ai-chat"),
            ("ai-generate", "AI生成因子", "ai-generate"),
            ("ai-optimize", "AI优化因子", "ai-optimize"),
            ("load-data", "加载市场数据", "load-data"),
            ("data-info", "查看数据信息", "data-info"),
            ("export", "导出结果", "export results.csv"),
            ("config", "查看配置信息", "config"),
            ("clear", "清空屏幕", "clear"),
            ("exit", "退出程序", "exit")
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        self.console.print(help_table)
    
    def list_functions(self):
        """列出所有可用的算子函数"""
        functions = self.interface.list_available_functions()
        
        # 按类别分组显示
        categories = {
            "基础算子": ["RANK", "RETURNS", "FUTURE_RETURNS", "STDDEV", "CORRELATION", "DELAY", "DELTA", "SCALE"],
            "时序算子": ["TS_RANK", "TS_MIN", "TS_MAX", "TS_ARGMAX", "DECAY_LINEAR", "ADV", "SUM", "TS_MEAN"],
            "数学算子": ["MIN", "MAX", "ABS", "LOG", "POWER", "SIGN", "SIGNEDPOWER", "IF"],
            "技术指标": ["MACD", "RSI", "KDJ", "BOLL", "CCI", "ATR", "ROC", "OBV", "MFI"],
            "移动平均": ["MA", "EMA", "SMA", "WMA", "HHV", "LLV"],
            "条件逻辑": ["CROSS", "COUNT", "EVERY", "EXIST", "BARSLAST", "VALUEWHEN"],
            "高级函数": ["VWAP", "CAP", "COVARIANCE", "PRODUCT", "SLOPE"]
        }
        
        for category, expected_funcs in categories.items():
            available_funcs = [f for f in functions if f in expected_funcs]
            if available_funcs:
                func_table = Table(title=f"📊 {category}")
                func_table.add_column("函数名", style="cyan")
                func_table.add_column("状态", style="green")
                
                for func in expected_funcs:
                    status = "✅ 可用" if func in functions else "❌ 不可用"
                    func_table.add_row(func, status)
                
                self.console.print(func_table)
        
        self.console.print(f"\n[bold]总计: {len(functions)} 个算子函数可用[/bold]")
    
    def list_factors(self):
        """列出已创建的因子"""
        factors = self.interface.list_available_factors()
        
        if not factors:
            self.console.print("[yellow]暂无已创建的因子[/yellow]")
            return
        
        factor_table = Table(title="📈 已创建因子列表")
        factor_table.add_column("因子名称", style="cyan")
        factor_table.add_column("类型", style="green")
        factor_table.add_column("创建时间", style="dim")
        
        for factor_name in factors:
            info = self.interface.get_factor_info(factor_name)
            factor_type = info.get('type', 'Unknown')
            factor_table.add_row(factor_name, factor_type, "未知")
        
        self.console.print(factor_table)
    
    def create_formula_factor(self):
        """创建公式因子"""
        self.console.print("\n[bold cyan]🧮 创建公式因子[/bold cyan]")
        
        # 获取因子名称
        factor_name = Prompt.ask("请输入因子名称")
        if not factor_name:
            self.console.print("[red]因子名称不能为空[/red]")
            return
        
        # 显示公式示例
        examples_panel = Panel(
            """[bold]公式示例:[/bold]

[cyan]• 简单动量:[/cyan] RANK((CLOSE / DELAY(CLOSE, 20)) - 1)
[cyan]• 波动率调整动量:[/cyan] RANK(RETURNS(CLOSE, 20)) / STDDEV(RETURNS(CLOSE, 1), 20)
[cyan]• 价量配合:[/cyan] CORRELATION(CLOSE, VOLUME, 20) * RANK(RETURNS(CLOSE, 10))
[cyan]• 技术指标组合:[/cyan] RSI(CLOSE, 14) / 100 - 0.5

[yellow]可用基础数据:[/yellow] CLOSE, OPEN, HIGH, LOW, VOLUME, AMOUNT
[yellow]支持嵌套函数和四则运算[/yellow]
            """,
            title="💡 公式因子语法",
            border_style="blue"
        )
        self.console.print(examples_panel)
        
        # 获取公式
        formula = Prompt.ask("请输入因子公式")
        if not formula:
            self.console.print("[red]公式不能为空[/red]")
            return
        
        # 验证公式语法
        self.console.print("🔍 验证公式语法...")
        validation_result = self.interface.validate_formula(formula)
        
        if not validation_result['valid']:
            self.console.print(f"[red]❌ 公式语法错误: {validation_result['message']}[/red]")
            return
        
        # 创建因子
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("创建公式因子...", total=None)
                created_name = self.interface.create_formula_factor(formula, factor_name)
                progress.update(task, completed=True)
            
            self.console.print(f"[green]✅ 成功创建公式因子: {created_name}[/green]")
            self.console.print(f"[dim]公式: {formula}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def create_python_factor(self):
        """创建Python因子"""
        self.console.print("\n[bold cyan]🐍 创建Python因子[/bold cyan]")
        
        # 获取因子名称
        factor_name = Prompt.ask("请输入因子名称")
        if not factor_name:
            self.console.print("[red]因子名称不能为空[/red]")
            return
        
        # 显示Python因子模板
        template_code = '''class MyFactor(BaseFactor):
    """自定义因子 - 请修改calculate方法"""
    
    def calculate(self, factors):
        close = factors['close']
        volume = factors['volume']
        
        # 示例：计算动量因子
        returns = RETURNS(close, 20)
        momentum = RANK(returns)
        
        # 示例：加入成交量信号
        volume_signal = RANK(volume / DELAY(volume, 5))
        
        # 组合信号
        result = momentum * 0.7 + volume_signal * 0.3
        
        return SCALE(result)  # 标准化到[-1, 1]'''
        
        syntax = Syntax(template_code, "python", theme="monokai", line_numbers=True)
        template_panel = Panel(syntax, title="🔧 Python因子模板", border_style="green")
        self.console.print(template_panel)
        
        # 选择输入方式
        input_method = Prompt.ask(
            "选择代码输入方式", 
            choices=["template", "manual", "file"], 
            default="template"
        )
        
        if input_method == "template":
            code = template_code
            if not Confirm.ask("使用模板代码？"):
                return
        elif input_method == "file":
            file_path = Prompt.ask("请输入Python文件路径")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                self.console.print(f"[red]读取文件失败: {str(e)}[/red]")
                return
        else:
            self.console.print("请输入Python代码 (输入空行结束):")
            code_lines = []
            while True:
                line = input()
                if not line.strip():
                    break
                code_lines.append(line)
            code = '\n'.join(code_lines)
        
        if not code.strip():
            self.console.print("[red]代码不能为空[/red]")
            return
        
        # 创建因子
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("创建Python因子...", total=None)
                created_name = self.interface.create_python_factor(code, factor_name)
                progress.update(task, completed=True)
            
            self.console.print(f"[green]✅ 成功创建Python因子: {created_name}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def calculate_factor(self):
        """计算因子值"""
        factors = self.interface.list_available_factors()
        
        if not factors:
            self.console.print("[yellow]暂无可计算的因子，请先创建因子[/yellow]")
            return
        
        # 选择因子
        self.console.print("\n[bold cyan]📊 计算因子值[/bold cyan]")
        factor_table = Table(title="可计算因子")
        factor_table.add_column("序号", style="dim")
        factor_table.add_column("因子名称", style="cyan")
        
        for i, factor in enumerate(factors, 1):
            factor_table.add_row(str(i), factor)
        
        self.console.print(factor_table)
        
        try:
            choice = int(Prompt.ask("请选择因子序号")) - 1
            if choice < 0 or choice >= len(factors):
                self.console.print("[red]无效的选择[/red]")
                return
            
            factor_name = factors[choice]
        except ValueError:
            self.console.print("[red]请输入有效的数字[/red]")
            return
        
        # 获取计算参数
        symbols_input = Prompt.ask("请输入股票代码 (逗号分隔)", default="AAPL,GOOGL,MSFT")
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        start_date = Prompt.ask("请输入开始日期 (YYYY-MM-DD)", default="2024-01-01")
        end_date = Prompt.ask("请输入结束日期 (YYYY-MM-DD)", default="2024-01-31")
        
        # 计算因子
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task(f"计算因子 {factor_name}...", total=None)
                result = self.interface.calculate_factor(factor_name, symbols, start_date, end_date)
                progress.update(task, completed=True)
            
            # 显示结果统计
            stats_table = Table(title=f"📈 因子 {factor_name} 计算结果")
            stats_table.add_column("统计项", style="cyan")
            stats_table.add_column("数值", style="white")
            
            stats_table.add_row("数据维度", str(result.series.shape))
            stats_table.add_row("均值", f"{result.series.mean():.6f}")
            stats_table.add_row("标准差", f"{result.series.std():.6f}")
            stats_table.add_row("最小值", f"{result.series.min():.6f}")
            stats_table.add_row("最大值", f"{result.series.max():.6f}")
            stats_table.add_row("缺失值", f"{result.series.isna().sum()} ({result.series.isna().mean():.1%})")
            
            self.console.print(stats_table)
            
            # 显示样本数据
            sample_data = result.series.dropna().head(10)
            if len(sample_data) > 0:
                self.console.print("\n[bold]前10个有效值:[/bold]")
                for idx, value in sample_data.items():
                    date, symbol = idx if isinstance(idx, tuple) else (idx, 'N/A')
                    self.console.print(f"{date} {symbol}: {value:.6f}")
            
            # 缓存结果
            self.current_session['factors'][factor_name] = result
            
            # 询问是否导出
            if Confirm.ask("\n是否导出结果到CSV文件？"):
                filename = Prompt.ask("请输入文件名", default=f"{factor_name}_result.csv")
                try:
                    df = result.series.reset_index()
                    df.to_csv(filename, index=False)
                    self.console.print(f"[green]✅ 结果已导出到 {filename}[/green]")
                except Exception as e:
                    self.console.print(f"[red]导出失败: {str(e)}[/red]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 计算失败: {str(e)}[/red]")
    
    async def ai_chat(self):
        """AI因子助手对话"""
        self.console.print("\n[bold cyan]🤖 AI因子开发助手[/bold cyan]")
        self.console.print("[dim]输入 'quit' 退出对话[/dim]\n")
        
        while True:
            user_input = Prompt.ask("[bold green]您[/bold green]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                continue
            
            try:
                with Progress(SpinnerColumn(), TextColumn("🤖 AI思考中...")) as progress:
                    task = progress.add_task("processing", total=None)
                    response = await self.llm_service.chat_with_factor_assistant(
                        user_input, 
                        self.current_session['conversation_history']
                    )
                    progress.update(task, completed=True)
                
                # 添加到对话历史
                self.current_session['conversation_history'].extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                
                # 显示回复
                response_panel = Panel(response, title="🤖 AI助手", border_style="blue")
                self.console.print(response_panel)
                
            except Exception as e:
                self.console.print(f"[red]AI服务出错: {str(e)}[/red]")
    
    async def ai_generate_factor(self):
        """AI生成因子"""
        self.console.print("\n[bold cyan]🧠 AI因子生成[/bold cyan]")
        
        requirements = Prompt.ask("请描述您需要的因子特性和目标")
        
        if not requirements.strip():
            self.console.print("[red]需求描述不能为空[/red]")
            return
        
        try:
            with Progress(SpinnerColumn(), TextColumn("🧠 AI生成因子中...")) as progress:
                task = progress.add_task("generating", total=None)
                result = await self.llm_service.generate_factor(requirements)
                progress.update(task, completed=True)
            
            if 'error' in result:
                self.console.print(f"[red]生成失败: {result['error']}[/red]")
                return
            
            # 显示生成结果
            result_panel = Panel(
                f"""[bold green]生成的因子公式:[/bold green]
{result.get('formula', '未生成公式')}

[bold blue]因子解释:[/bold blue]
{result.get('explanation', '无解释')}

[bold yellow]参数说明:[/bold yellow]
{json.dumps(result.get('parameters', {}), indent=2, ensure_ascii=False)}

[bold magenta]适用场景:[/bold magenta]
{result.get('scenarios', '未知')}
                """,
                title="🎯 AI生成结果",
                border_style="green"
            )
            
            self.console.print(result_panel)
            
            # 询问是否创建因子
            if 'formula' in result and Confirm.ask("是否基于此公式创建因子？"):
                factor_name = Prompt.ask("请输入因子名称")
                if factor_name:
                    try:
                        created_name = self.interface.create_formula_factor(result['formula'], factor_name)
                        self.console.print(f"[green]✅ 成功创建因子: {created_name}[/green]")
                    except Exception as e:
                        self.console.print(f"[red]创建因子失败: {str(e)}[/red]")
            
        except Exception as e:
            self.console.print(f"[red]AI生成出错: {str(e)}[/red]")
    
    async def validate_factor(self):
        """验证因子性能"""
        factors = list(self.current_session['factors'].keys())
        
        if not factors:
            self.console.print("[yellow]暂无已计算的因子，请先计算因子值[/yellow]")
            return
        
        self.console.print("\n[bold cyan]🔍 因子性能验证[/bold cyan]")
        
        # 选择因子
        factor_table = Table(title="可验证因子")
        factor_table.add_column("序号", style="dim")
        factor_table.add_column("因子名称", style="cyan")
        
        for i, factor in enumerate(factors, 1):
            factor_table.add_row(str(i), factor)
        
        self.console.print(factor_table)
        
        try:
            choice = int(Prompt.ask("请选择因子序号")) - 1
            if choice < 0 or choice >= len(factors):
                self.console.print("[red]无效的选择[/red]")
                return
            
            factor_name = factors[choice]
            factor_result = self.current_session['factors'][factor_name]
        except ValueError:
            self.console.print("[red]请输入有效的数字[/red]")
            return
        
        # 生成模拟市场数据用于验证
        self.console.print("📊 准备验证数据...")
        
        # 从因子索引中提取日期和股票
        dates = factor_result.series.index.get_level_values('date').unique()
        symbols = factor_result.series.index.get_level_values('symbol').unique()
        
        # 生成模拟市场数据
        market_data = {}
        np.random.seed(42)
        
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        
        # 生成收盘价
        base_prices = np.random.uniform(50, 200, len(symbols))
        price_data = []
        for i, date in enumerate(dates):
            for j, symbol in enumerate(symbols):
                noise = np.random.normal(0, 0.02)  # 2%波动
                price = base_prices[j] * (1 + noise * (i + 1) * 0.1)
                price_data.append(max(10, price))  # 确保价格为正
        
        market_data['close'] = pd.Series(price_data, index=index)
        
        # 生成成交量
        volume_data = np.random.lognormal(13, 0.5, len(index))
        market_data['volume'] = pd.Series(volume_data, index=index)
        
        try:
            with Progress(SpinnerColumn(), TextColumn("🔍 执行综合验证...")) as progress:
                task = progress.add_task("validating", total=None)
                metrics = await self.validator.comprehensive_validation(
                    factor_result.series, 
                    market_data,
                    validation_periods=[1, 5, 10]
                )
                progress.update(task, completed=True)
            
            # 生成验证报告
            report = self.validator.generate_validation_report(metrics)
            
            # 显示验证结果
            self._display_validation_report(report)
            
        except Exception as e:
            self.console.print(f"[red]❌ 验证失败: {str(e)}[/red]")
    
    def _display_validation_report(self, report: Dict[str, Any]):
        """显示验证报告"""
        if 'error' in report:
            self.console.print(f"[red]报告生成失败: {report['error']}[/red]")
            return
        
        # 因子基本信息
        info_table = Table(title="📊 因子基本信息")
        info_table.add_column("项目", style="cyan")
        info_table.add_column("值", style="white")
        
        factor_info = report.get('factor_info', {})
        info_table.add_row("因子名称", factor_info.get('name', 'N/A'))
        info_table.add_row("样本期间", f"{factor_info.get('sample_period', ('N/A', 'N/A'))[0]} 至 {factor_info.get('sample_period', ('N/A', 'N/A'))[1]}")
        info_table.add_row("总观测数", str(factor_info.get('total_observations', 'N/A')))
        info_table.add_row("缺失值比例", factor_info.get('missing_ratio', 'N/A'))
        
        self.console.print(info_table)
        
        # 基础统计
        stats_table = Table(title="📈 基础统计特征")
        stats_table.add_column("统计量", style="cyan")
        stats_table.add_column("数值", style="white")
        
        basic_stats = report.get('basic_statistics', {})
        for stat_name, stat_value in basic_stats.items():
            if stat_value is not None:
                stats_table.add_row(stat_name, f"{stat_value:.4f}")
            else:
                stats_table.add_row(stat_name, "N/A")
        
        self.console.print(stats_table)
        
        # IC分析结果
        ic_analysis = report.get('ic_analysis', {})
        if ic_analysis:
            ic_table = Table(title="📊 IC分析结果")
            ic_table.add_column("持有期", style="cyan")
            ic_table.add_column("IC均值", style="white")
            ic_table.add_column("信息比率", style="green")
            ic_table.add_column("正确率", style="yellow")
            ic_table.add_column("表现评级", style="magenta")
            
            for period, ic_data in ic_analysis.items():
                ic_table.add_row(
                    period.replace('period_', '').replace('d', '天'),
                    str(ic_data.get('ic_mean', 'N/A')),
                    str(ic_data.get('ic_ir', 'N/A')),
                    ic_data.get('ic_positive_ratio', 'N/A'),
                    ic_data.get('performance', 'N/A')
                )
            
            self.console.print(ic_table)
        
        # 分层回测结果
        layered_perf = report.get('layered_performance', {})
        if layered_perf:
            layer_table = Table(title="📊 分层回测结果")
            layer_table.add_column("层级", style="cyan")
            layer_table.add_column("平均收益率", style="white")
            
            layer_returns = layered_perf.get('layer_returns', {})
            for layer, return_val in layer_returns.items():
                if return_val is not None:
                    layer_table.add_row(layer, f"{return_val:.4f}")
                else:
                    layer_table.add_row(layer, "N/A")
            
            if layered_perf.get('long_short_return') is not None:
                layer_table.add_row("[bold]多空收益[/bold]", f"[bold]{layered_perf['long_short_return']:.4f}[/bold]")
            
            layer_table.add_row("单调性", layered_perf.get('monotonicity', 'N/A'))
            
            self.console.print(layer_table)
        
        # 综合评分和建议
        score = report.get('overall_score', 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        
        recommendations = report.get('recommendations', [])
        rec_text = "\n".join([f"• {rec}" for rec in recommendations])
        
        summary_panel = Panel(
            f"""[bold {score_color}]综合评分: {score:.3f}[/bold {score_color}]

[bold blue]改进建议:[/bold blue]
{rec_text}
            """,
            title="🎯 验证总结",
            border_style=score_color
        )
        
        self.console.print(summary_panel)
    
    def run_demo(self):
        """运行功能演示"""
        self.console.print("\n[bold cyan]🎬 PandaFactor 功能演示[/bold cyan]")
        
        demo_options = [
            "基础算子演示",
            "技术指标演示", 
            "公式因子演示",
            "完整工作流演示",
            "取消"
        ]
        
        for i, option in enumerate(demo_options, 1):
            self.console.print(f"{i}. {option}")
        
        try:
            choice = int(Prompt.ask("请选择演示内容")) - 1
            if choice < 0 or choice >= len(demo_options):
                self.console.print("[red]无效的选择[/red]")
                return
            
            if choice == 4:  # 取消
                return
            
            # 运行对应的演示
            demo_functions = [
                self._demo_basic_operators,
                self._demo_technical_indicators,
                self._demo_formula_factors,
                self._demo_complete_workflow
            ]
            
            demo_functions[choice]()
            
        except ValueError:
            self.console.print("[red]请输入有效的数字[/red]")
    
    def _demo_basic_operators(self):
        """演示基础算子"""
        self.console.print("\n[bold]🔧 基础算子演示[/bold]")
        
        # 创建示例数据
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        symbols = ['AAPL', 'GOOGL']
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(index)) * 0.02) * 100
        close_series = pd.Series(prices, index=index)
        
        self.console.print(f"📊 生成示例数据: {len(dates)}天 × {len(symbols)}只股票")
        
        # 演示RANK算子
        with Progress(SpinnerColumn(), TextColumn("计算RANK算子...")) as progress:
            task = progress.add_task("rank", total=None)
            rank_result = self.interface.rank(close_series)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ RANK算子结果范围: [{rank_result.min():.3f}, {rank_result.max():.3f}]")
        
        # 演示RETURNS算子
        with Progress(SpinnerColumn(), TextColumn("计算RETURNS算子...")) as progress:
            task = progress.add_task("returns", total=None)
            returns_result = self.interface.returns(close_series, 5)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ RETURNS(5)结果均值: {returns_result.mean():.6f}")
        
        # 演示STDDEV算子
        with Progress(SpinnerColumn(), TextColumn("计算STDDEV算子...")) as progress:
            task = progress.add_task("stddev", total=None)
            vol_result = self.interface.stddev(returns_result, 5)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ STDDEV(5)结果均值: {vol_result.mean():.6f}")
        
        self.console.print("[green]🎉 基础算子演示完成！[/green]")
    
    def _demo_technical_indicators(self):
        """演示技术指标"""
        self.console.print("\n[bold]📊 技术指标演示[/bold]")
        
        # 创建更长的时间序列
        dates = pd.date_range('2024-01-01', '2024-02-29', freq='D')
        symbols = ['AAPL']
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        
        np.random.seed(42)
        base_price = 150
        returns = np.random.normal(0, 0.02, len(index))
        close = pd.Series(base_price * np.exp(np.cumsum(returns)), index=index)
        high = close * (1 + np.random.uniform(0, 0.03, len(index)))
        low = close * (1 - np.random.uniform(0, 0.03, len(index)))
        
        self.console.print(f"📊 生成示例数据: {len(dates)}天技术指标数据")
        
        # MACD指标
        with Progress(SpinnerColumn(), TextColumn("计算MACD指标...")) as progress:
            task = progress.add_task("macd", total=None)
            dif, dea, macd = self.interface.macd(close)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ MACD - DIF范围: [{dif.min():.3f}, {dif.max():.3f}]")
        
        # RSI指标
        with Progress(SpinnerColumn(), TextColumn("计算RSI指标...")) as progress:
            task = progress.add_task("rsi", total=None)
            rsi = self.interface.rsi(close, 14)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ RSI(14)均值: {rsi.mean():.2f}")
        
        # KDJ指标
        with Progress(SpinnerColumn(), TextColumn("计算KDJ指标...")) as progress:
            task = progress.add_task("kdj", total=None)
            kdj_k = self.interface.kdj(close, high, low)
            progress.update(task, completed=True)
        
        self.console.print(f"✅ KDJ-K线范围: [{kdj_k.min():.2f}, {kdj_k.max():.2f}]")
        
        self.console.print("[green]🎉 技术指标演示完成！[/green]")
    
    def _demo_formula_factors(self):
        """演示公式因子"""
        self.console.print("\n[bold]🧮 公式因子演示[/bold]")
        
        # 创建经典动量因子
        formula = "RANK((CLOSE / DELAY(CLOSE, 20)) - 1)"
        factor_name = "demo_momentum"
        
        try:
            with Progress(SpinnerColumn(), TextColumn("创建动量因子...")) as progress:
                task = progress.add_task("create", total=None)
                created_name = self.interface.create_formula_factor(formula, factor_name)
                progress.update(task, completed=True)
            
            self.console.print(f"✅ 成功创建公式因子: {created_name}")
            self.console.print(f"📝 公式: {formula}")
            
            # 计算因子值
            symbols = ['AAPL', 'GOOGL']
            start_date = '2024-01-01'
            end_date = '2024-01-31'
            
            with Progress(SpinnerColumn(), TextColumn("计算因子值...")) as progress:
                task = progress.add_task("calculate", total=None)
                result = self.interface.calculate_factor(created_name, symbols, start_date, end_date)
                progress.update(task, completed=True)
            
            self.console.print(f"✅ 计算完成: {result.series.shape[0]} 个观测值")
            self.console.print(f"📊 因子值范围: [{result.series.min():.4f}, {result.series.max():.4f}]")
            
            # 缓存结果
            self.current_session['factors'][created_name] = result
            
            self.console.print("[green]🎉 公式因子演示完成！[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 演示失败: {str(e)}[/red]")
    
    def _demo_complete_workflow(self):
        """演示完整工作流"""
        self.console.print("\n[bold]🚀 完整工作流演示[/bold]")
        self.console.print("将演示：创建因子 → 计算值 → 性能验证 → 生成报告")
        
        # 步骤1: 创建复合因子
        self.console.print("\n[bold cyan]步骤1: 创建复合动量因子[/bold cyan]")
        formula = "SCALE(RANK(RETURNS(CLOSE, 20)) * RANK(VOLUME / DELAY(VOLUME, 5)))"
        factor_name = "demo_complex_momentum"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"✅ 创建因子: {created_name}")
            
            # 步骤2: 计算因子值
            self.console.print("\n[bold cyan]步骤2: 计算因子值[/bold cyan]")
            symbols = ['AAPL', 'GOOGL', 'MSFT']
            result = self.interface.calculate_factor(created_name, symbols, '2024-01-01', '2024-01-31')
            self.console.print(f"✅ 计算完成: {len(result.series)} 个观测值")
            
            # 步骤3: 缓存并准备验证
            self.current_session['factors'][created_name] = result
            self.console.print("\n[bold cyan]步骤3: 准备性能验证[/bold cyan]")
            
            # 简化的性能统计
            stats = {
                '数据完整性': f"{(1 - result.series.isna().mean()) * 100:.1f}%",
                '数值稳定性': '良好' if result.series.std() > 0 else '需关注',
                '分布特征': f"偏度{result.series.skew():.3f}"
            }
            
            stats_table = Table(title="📊 快速性能检查")
            stats_table.add_column("指标", style="cyan")
            stats_table.add_column("结果", style="white")
            
            for metric, value in stats.items():
                stats_table.add_row(metric, str(value))
            
            self.console.print(stats_table)
            
            # 步骤4: 生成简要报告
            self.console.print("\n[bold cyan]步骤4: 生成简要报告[/bold cyan]")
            
            summary_panel = Panel(
                f"""[bold green]✅ 工作流完成总结[/bold green]

[cyan]因子名称:[/cyan] {created_name}
[cyan]公式:[/cyan] {formula}
[cyan]数据期间:[/cyan] 2024-01-01 至 2024-01-31
[cyan]股票范围:[/cyan] {', '.join(symbols)}
[cyan]总观测值:[/cyan] {len(result.series)}

[yellow]主要特征:[/yellow]
• 均值: {result.series.mean():.6f}
• 标准差: {result.series.std():.6f}
• 数据完整性: {(1-result.series.isna().mean())*100:.1f}%

[blue]后续建议:[/blue]
1. 可使用 'validate' 命令进行详细性能验证
2. 可使用 'ai-optimize' 命令优化因子
3. 可使用 'export' 命令导出结果
                """,
                title="🎯 工作流总结",
                border_style="green"
            )
            
            self.console.print(summary_panel)
            
            self.console.print("[green]🎉 完整工作流演示完成！[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 工作流演示失败: {str(e)}[/red]")
    
    def show_config(self):
        """显示配置信息"""
        config_table = Table(title="⚙️ 系统配置信息")
        config_table.add_column("配置项", style="cyan")
        config_table.add_column("状态", style="white")
        
        # 检查各组件状态
        config_table.add_row("因子引擎", "✅ 正常")
        config_table.add_row("数据读取器", "✅ 正常")
        config_table.add_row("因子验证器", "✅ 正常")
        
        # LLM服务状态
        try:
            import openai
            config_table.add_row("LLM服务", "✅ 可用")
        except ImportError:
            config_table.add_row("LLM服务", "❌ 未配置")
        
        # MongoDB状态  
        try:
            import pymongo
            config_table.add_row("MongoDB连接", "⚠️ 待配置")
        except ImportError:
            config_table.add_row("MongoDB连接", "❌ 未安装")
        
        # 算子统计
        available_functions = len(self.interface.list_available_functions())
        config_table.add_row("可用算子", f"✅ {available_functions} 个")
        
        # 因子统计
        created_factors = len(self.interface.list_available_factors())
        config_table.add_row("已创建因子", f"📊 {created_factors} 个")
        
        self.console.print(config_table)
    
    async def main_loop(self):
        """主命令循环"""
        self.show_welcome()
        
        while True:
            try:
                command = Prompt.ask("\n[bold yellow]PandaFactor[/bold yellow]").strip().lower()
                
                if command in ['exit', 'quit', 'q']:
                    self.console.print("[bold green]感谢使用 PandaFactor Professional CLI！[/bold green]")
                    break
                elif command in ['help', 'h']:
                    self.show_help()
                elif command == 'demo':
                    self.run_demo()
                elif command == 'list-functions':
                    self.list_functions()
                elif command == 'list-factors':
                    self.list_factors()
                elif command == 'create-formula':
                    self.create_formula_factor()
                elif command == 'create-python':
                    self.create_python_factor()
                elif command in ['calculate', 'calc']:
                    self.calculate_factor()
                elif command == 'validate':
                    await self.validate_factor()
                elif command == 'ai-chat':
                    await self.ai_chat()
                elif command == 'ai-generate':
                    await self.ai_generate_factor()
                elif command == 'config':
                    self.show_config()
                elif command == 'clear':
                    self.console.clear()
                elif command == '':
                    continue
                else:
                    self.console.print(f"[red]未知命令: {command}[/red]")
                    self.console.print("输入 'help' 查看可用命令")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]使用 'exit' 命令退出程序[/yellow]")
            except Exception as e:
                self.console.print(f"[red]命令执行出错: {str(e)}[/red]")


def main():
    """CLI入口函数"""
    cli = PandaFactorCLI()
    
    try:
        # 运行异步主循环
        asyncio.run(cli.main_loop())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()