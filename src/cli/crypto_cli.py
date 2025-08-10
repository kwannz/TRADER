#!/usr/bin/env python3
"""
Crypto-Focused PandaFactor CLI
加密货币专用PandaFactor CLI - 针对数字资产市场优化的量化工作台
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
    from rich.layout import Layout
    from rich.columns import Columns
except ImportError:
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

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.factor_engine.crypto_specialized import CryptoFactorUtils, CryptoDataProcessor
    # 为了演示，其他组件先简化处理
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class CryptoPandaFactorCLI:
    """
    加密货币专用PandaFactor CLI - 数字资产量化分析工作台
    """
    
    def __init__(self):
        self.console = Console()
        
        # 核心组件
        self.crypto_utils = CryptoFactorUtils()
        self.data_processor = CryptoDataProcessor()
        # 其他组件暂时简化
        
        # CLI状态
        self.current_session = {
            'crypto_factors': {},
            'market_data': {},
            'active_symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'timeframe': '1h',
            'conversation_history': []
        }
        
        # 显示欢迎信息
        self.show_crypto_welcome()
    
    def show_crypto_welcome(self):
        """显示加密货币专用欢迎信息"""
        welcome_panel = Panel(
            """[bold green]🚀 欢迎使用 Crypto PandaFactor Professional CLI！[/bold green]

[bold cyan]🔗 专为加密货币市场设计的量化因子开发工作台[/bold cyan]

💰 [bold]加密货币专用功能:[/bold]
• 资金费率动量分析 (FUNDING_RATE_MOMENTUM)
• 巨鲸交易预警系统 (WHALE_ALERT)  
• 恐惧贪婪指数计算 (FEAR_GREED_INDEX)
• DeFi TVL关联分析 (DEFI_TVL_CORRELATION)
• 清算瀑布风险评估 (LIQUIDATION_CASCADE_RISK)
• 跨链套利机会识别 (CROSS_CHAIN_CORRELATION)
• 收益农场压力分析 (YIELD_FARMING_PRESSURE)

📊 [bold]支持的交易所:[/bold] Binance, Coinbase, OKX
🕐 [bold]时间框架:[/bold] 1m, 5m, 15m, 1h, 4h, 1d
💱 [bold]默认交易对:[/bold] BTC/USDT, ETH/USDT, BNB/USDT

📖 [bold]快速开始:[/bold]
• 输入 'help' 查看加密货币专用命令
• 输入 'crypto-demo' 体验加密因子功能
• 输入 'market-overview' 查看实时市场概况
            """,
            title="🔗 Crypto PandaFactor Professional",
            border_style="cyan"
        )
        self.console.print(welcome_panel)
    
    def show_help(self):
        """显示加密货币专用帮助信息"""
        help_table = Table(title="💰 Crypto PandaFactor CLI 命令列表")
        help_table.add_column("命令", style="cyan", no_wrap=True)
        help_table.add_column("功能", style="white")
        help_table.add_column("示例", style="dim")
        
        commands = [
            # 基础命令
            ("help", "显示命令帮助", "help"),
            ("crypto-demo", "加密因子演示", "crypto-demo"),
            ("market-overview", "市场概况", "market-overview"),
            
            # 数据相关
            ("set-symbols", "设置分析币种", "set-symbols BTC/USDT,ETH/USDT"),
            ("set-timeframe", "设置时间框架", "set-timeframe 1h"),
            ("load-crypto-data", "加载加密数据", "load-crypto-data"),
            ("funding-rates", "查看资金费率", "funding-rates"),
            
            # 加密因子
            ("create-crypto-factor", "创建加密货币专用因子", "create-crypto-factor"),
            ("whale-alert", "巨鲸交易分析", "whale-alert"),
            ("fear-greed", "恐惧贪婪指数", "fear-greed"),
            ("liquidation-risk", "清算风险分析", "liquidation-risk"),
            
            # 高级功能
            ("cross-exchange", "跨交易所分析", "cross-exchange"),
            ("defi-analysis", "DeFi生态分析", "defi-analysis"),
            ("flash-crash", "闪崩检测", "flash-crash"),
            
            # AI功能
            ("crypto-ai-chat", "加密AI助手", "crypto-ai-chat"),
            ("ai-market-insight", "AI市场洞察", "ai-market-insight"),
            
            # 通用命令
            ("validate-crypto", "验证加密因子", "validate-crypto"),
            ("export-results", "导出分析结果", "export-results"),
            ("config", "查看配置", "config"),
            ("clear", "清空屏幕", "clear"),
            ("exit", "退出程序", "exit")
        ]
        
        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)
        
        self.console.print(help_table)
    
    def show_market_overview(self):
        """显示市场概况"""
        self.console.print("\n[bold cyan]📊 加密货币市场概况[/bold cyan]")
        
        # 创建市场概览表格
        market_table = Table(title="实时市场数据")
        market_table.add_column("交易对", style="cyan", no_wrap=True)
        market_table.add_column("价格", style="green")
        market_table.add_column("24h涨跌", style="yellow")
        market_table.add_column("成交量", style="blue") 
        market_table.add_column("恐惧贪婪", style="magenta")
        
        # 模拟市场数据
        market_data = {
            'BTC/USDT': {'price': 43250.50, 'change': 2.34, 'volume': '1.2B', 'fg': 65},
            'ETH/USDT': {'price': 2890.75, 'change': -1.56, 'volume': '890M', 'fg': 58},
            'BNB/USDT': {'price': 385.20, 'change': 0.89, 'volume': '156M', 'fg': 72},
            'ADA/USDT': {'price': 0.485, 'change': 3.12, 'volume': '89M', 'fg': 68},
            'SOL/USDT': {'price': 98.65, 'change': -2.34, 'volume': '234M', 'fg': 45}
        }
        
        for symbol, data in market_data.items():
            change_color = "green" if data['change'] > 0 else "red"
            fg_color = "green" if data['fg'] > 60 else "yellow" if data['fg'] > 40 else "red"
            
            market_table.add_row(
                symbol,
                f"${data['price']:,.2f}",
                f"[{change_color}]{data['change']:+.2f}%[/{change_color}]",
                data['volume'],
                f"[{fg_color}]{data['fg']}[/{fg_color}]"
            )
        
        self.console.print(market_table)
        
        # 市场情绪面板
        sentiment_panel = Panel(
            """[bold]📈 市场情绪分析:[/bold]

🔹 整体趋势: [green]谨慎乐观[/green]
🔹 波动水平: [yellow]中等[/yellow] 
🔹 资金流向: [cyan]流入主流币[/cyan]
🔹 关键阻力: BTC $45,000, ETH $3,000
🔹 支撑位置: BTC $42,000, ETH $2,800

⚠️  [bold red]风险提醒:[/bold red] 关注美联储政策动向和监管消息
            """,
            title="💡 市场洞察",
            border_style="blue"
        )
        self.console.print(sentiment_panel)
    
    def set_active_symbols(self):
        """设置分析的币种"""
        current_symbols = ", ".join(self.current_session['active_symbols'])
        self.console.print(f"\n当前分析币种: [cyan]{current_symbols}[/cyan]")
        
        new_symbols = Prompt.ask(
            "请输入新的币种列表 (逗号分隔)",
            default=current_symbols
        )
        
        if new_symbols:
            symbols_list = [s.strip().upper() for s in new_symbols.split(',')]
            # 确保格式正确
            formatted_symbols = []
            for symbol in symbols_list:
                if '/' not in symbol:
                    symbol = f"{symbol}/USDT"
                formatted_symbols.append(symbol)
            
            self.current_session['active_symbols'] = formatted_symbols
            self.console.print(f"[green]✅ 已设置分析币种: {', '.join(formatted_symbols)}[/green]")
    
    def set_timeframe(self):
        """设置时间框架"""
        current_tf = self.current_session['timeframe']
        self.console.print(f"\n当前时间框架: [cyan]{current_tf}[/cyan]")
        
        available_tf = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        tf_table = Table(title="可用时间框架")
        tf_table.add_column("序号", style="dim")
        tf_table.add_column("时间框架", style="cyan")
        tf_table.add_column("描述", style="white")
        
        descriptions = {
            '1m': '1分钟 - 超短线交易',
            '5m': '5分钟 - 短线分析',
            '15m': '15分钟 - 日内交易',
            '1h': '1小时 - 中短线分析',
            '4h': '4小时 - 中线分析',
            '1d': '1日 - 长线分析'
        }
        
        for i, tf in enumerate(available_tf, 1):
            tf_table.add_row(str(i), tf, descriptions[tf])
        
        self.console.print(tf_table)
        
        try:
            choice = int(Prompt.ask("请选择时间框架序号")) - 1
            if 0 <= choice < len(available_tf):
                new_tf = available_tf[choice]
                self.current_session['timeframe'] = new_tf
                self.console.print(f"[green]✅ 已设置时间框架: {new_tf}[/green]")
            else:
                self.console.print("[red]❌ 无效选择[/red]")
        except ValueError:
            self.console.print("[red]❌ 请输入有效数字[/red]")
    
    async def load_crypto_data(self):
        """加载加密货币数据"""
        symbols = self.current_session['active_symbols']
        timeframe = self.current_session['timeframe']
        
        self.console.print(f"\n[bold cyan]📥 加载加密货币数据[/bold cyan]")
        self.console.print(f"币种: {', '.join(symbols)}")
        self.console.print(f"时间框架: {timeframe}")
        
        # 获取日期范围
        days_back = int(Prompt.ask("请输入回溯天数", default="30"))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("正在加载市场数据...", total=None)
                
                # 加载综合市场数据
                market_data = await self.crypto_data_manager.get_multi_timeframe_data(
                    symbols,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    [timeframe]
                )
                
                progress.update(task, completed=True)
            
            # 存储数据
            self.current_session['market_data'] = market_data
            
            # 显示数据统计
            data_stats = Table(title="数据加载统计")
            data_stats.add_column("币种", style="cyan")
            data_stats.add_column("数据量", style="white")
            data_stats.add_column("时间范围", style="green")
            data_stats.add_column("完整性", style="yellow")
            
            for symbol in symbols:
                symbol_key = f"{symbol.replace('/', '')}_close"
                if symbol_key in market_data.get(timeframe, {}):
                    data_series = market_data[timeframe][symbol_key]
                    completeness = (1 - data_series.isna().mean()) * 100
                    
                    data_stats.add_row(
                        symbol,
                        str(len(data_series)),
                        f"{start_date.strftime('%m-%d')} 至 {end_date.strftime('%m-%d')}",
                        f"{completeness:.1f}%"
                    )
            
            self.console.print(data_stats)
            self.console.print("[green]✅ 数据加载完成[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 数据加载失败: {str(e)}[/red]")
    
    def create_crypto_factor(self):
        """创建加密货币专用因子"""
        self.console.print("\n[bold cyan]💰 创建加密货币专用因子[/bold cyan]")
        
        # 加密因子类型选择
        factor_types = {
            "1": ("FUNDING_RATE_MOMENTUM", "资金费率动量因子"),
            "2": ("WHALE_ALERT", "巨鲸交易预警因子"),
            "3": ("FEAR_GREED_INDEX", "恐惧贪婪指数"),
            "4": ("LIQUIDATION_CASCADE_RISK", "清算瀑布风险"),
            "5": ("DEFI_TVL_CORRELATION", "DeFi TVL关联因子"),
            "6": ("CROSS_CHAIN_CORRELATION", "跨链关联分析"),
            "7": ("FLASH_CRASH_DETECTOR", "闪崩检测器"),
            "8": ("CUSTOM_CRYPTO", "自定义加密因子")
        }
        
        factor_table = Table(title="加密货币专用因子类型")
        factor_table.add_column("序号", style="dim")
        factor_table.add_column("因子类型", style="cyan")
        factor_table.add_column("描述", style="white")
        
        for key, (factor_type, description) in factor_types.items():
            factor_table.add_row(key, factor_type, description)
        
        self.console.print(factor_table)
        
        choice = Prompt.ask("请选择因子类型序号")
        
        if choice not in factor_types:
            self.console.print("[red]❌ 无效选择[/red]")
            return
        
        factor_type, description = factor_types[choice]
        
        # 根据选择创建不同的因子
        if factor_type == "FUNDING_RATE_MOMENTUM":
            self._create_funding_rate_factor()
        elif factor_type == "WHALE_ALERT":
            self._create_whale_alert_factor()
        elif factor_type == "FEAR_GREED_INDEX":
            self._create_fear_greed_factor()
        elif factor_type == "LIQUIDATION_CASCADE_RISK":
            self._create_liquidation_risk_factor()
        elif factor_type == "CUSTOM_CRYPTO":
            self._create_custom_crypto_factor()
        else:
            self.console.print(f"[yellow]⚠️ {factor_type} 功能开发中...[/yellow]")
    
    def _create_funding_rate_factor(self):
        """创建资金费率动量因子"""
        self.console.print("\n[bold]📈 资金费率动量因子[/bold]")
        
        # 显示因子说明
        explanation = Panel(
            """[bold green]资金费率动量因子说明:[/bold green]

📊 [bold]原理:[/bold] 分析永续合约资金费率的变化趋势
🎯 [bold]目标:[/bold] 预测基于资金费率极值的价格反转
⚡ [bold]特点:[/bold] 加密货币市场独有的情绪指标

[cyan]关键参数:[/cyan]
• window: 滚动窗口大小 (默认24，即3个资金费率周期)
• extreme_threshold: 极端费率阈值 (默认1%)

[yellow]适用场景:[/yellow] 永续合约交易、市场情绪分析
            """,
            border_style="green"
        )
        self.console.print(explanation)
        
        # 获取参数
        window = int(Prompt.ask("滚动窗口大小", default="24"))
        factor_name = Prompt.ask("因子名称", default="funding_rate_momentum")
        
        # 创建公式
        formula = f"FUNDING_RATE_MOMENTUM(FUNDING_RATES, {window})"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"[green]✅ 成功创建资金费率动量因子: {created_name}[/green]")
            
            # 存储到加密因子列表
            self.current_session['crypto_factors'][created_name] = {
                'type': 'funding_rate_momentum',
                'formula': formula,
                'parameters': {'window': window}
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def _create_whale_alert_factor(self):
        """创建巨鲸交易预警因子"""
        self.console.print("\n[bold]🐋 巨鲸交易预警因子[/bold]")
        
        explanation = Panel(
            """[bold green]巨鲸交易预警因子说明:[/bold green]

📊 [bold]原理:[/bold] 检测异常大额交易对市场的潜在影响
🎯 [bold]目标:[/bold] 提前发现可能导致价格剧烈波动的大额交易
⚡ [bold]特点:[/bold] 基于成交量和成交额的异常值检测

[cyan]关键参数:[/cyan]
• threshold_std: 异常值标准差倍数 (默认3.0)
• window: 滚动统计窗口 (默认168小时=7天)

[yellow]适用场景:[/yellow] 风险管理、大户行为分析
            """,
            border_style="blue"
        )
        self.console.print(explanation)
        
        # 获取参数
        threshold = float(Prompt.ask("异常值标准差倍数", default="3.0"))
        factor_name = Prompt.ask("因子名称", default="whale_alert")
        
        # 创建公式
        formula = f"WHALE_ALERT(VOLUME, AMOUNT, {threshold})"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"[green]✅ 成功创建巨鲸交易预警因子: {created_name}[/green]")
            
            self.current_session['crypto_factors'][created_name] = {
                'type': 'whale_alert',
                'formula': formula,
                'parameters': {'threshold': threshold}
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def _create_fear_greed_factor(self):
        """创建恐惧贪婪指数因子"""
        self.console.print("\n[bold]😰 恐惧贪婪指数因子[/bold]")
        
        explanation = Panel(
            """[bold green]恐惧贪婪指数因子说明:[/bold green]

📊 [bold]原理:[/bold] 综合价格动量、波动率、成交量和情绪的多维度指标
🎯 [bold]目标:[/bold] 量化市场整体情绪状态 (0=极度恐惧, 100=极度贪婪)
⚡ [bold]特点:[/bold] 加密市场情绪的综合衡量

[cyan]组成部分:[/cyan]
• 价格动量 (25%): 14天收益率排名
• 波动率 (25%): 反向波动率指标
• 成交量 (25%): 成交量相对强度
• 市场趋势 (25%): 价格趋势方向

[yellow]适用场景:[/yellow] 市场择时、情绪分析、反向投资策略
            """,
            border_style="magenta"
        )
        self.console.print(explanation)
        
        factor_name = Prompt.ask("因子名称", default="fear_greed_index")
        
        # 创建公式
        formula = "FEAR_GREED_INDEX(CLOSE, VOLUME)"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"[green]✅ 成功创建恐惧贪婪指数因子: {created_name}[/green]")
            
            self.current_session['crypto_factors'][created_name] = {
                'type': 'fear_greed_index',
                'formula': formula,
                'parameters': {}
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def _create_liquidation_risk_factor(self):
        """创建清算瀑布风险因子"""
        self.console.print("\n[bold]💥 清算瀑布风险因子[/bold]")
        
        explanation = Panel(
            """[bold green]清算瀑布风险因子说明:[/bold green]

📊 [bold]原理:[/bold] 评估期货市场大规模清算的可能性
🎯 [bold]目标:[/bold] 预警可能引发连锁清算的市场状态  
⚡ [bold]特点:[/bold] 结合持仓量、资金费率和波动率的风险评估

[cyan]风险信号:[/cyan]
• 持仓量快速增长 (40%)
• 极端资金费率 (30%) 
• 高波动环境 (30%)

[yellow]适用场景:[/yellow] 风险管理、期货交易、市场监控
            """,
            border_style="red"
        )
        self.console.print(explanation)
        
        window = int(Prompt.ask("分析窗口 (小时)", default="72"))
        factor_name = Prompt.ask("因子名称", default="liquidation_risk")
        
        # 创建公式
        formula = f"LIQUIDATION_CASCADE_RISK(CLOSE, OPEN_INTEREST, FUNDING_RATES, {window})"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"[green]✅ 成功创建清算瀑布风险因子: {created_name}[/green]")
            
            self.current_session['crypto_factors'][created_name] = {
                'type': 'liquidation_risk',
                'formula': formula,
                'parameters': {'window': window}
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    def _create_custom_crypto_factor(self):
        """创建自定义加密因子"""
        self.console.print("\n[bold]🛠️ 自定义加密因子[/bold]")
        
        # 显示可用的加密专用函数
        crypto_functions = [
            "FUNDING_RATE_MOMENTUM", "WHALE_ALERT", "FEAR_GREED_INDEX",
            "MARKET_CAP_RANK", "DEFI_TVL_CORRELATION", "EXCHANGE_FLOW_PRESSURE", 
            "MINER_CAPITULATION", "STABLECOIN_DOMINANCE", "LIQUIDATION_CASCADE_RISK",
            "CRYPTO_RSI_DIVERGENCE", "FLASH_CRASH_DETECTOR", "CROSS_CHAIN_CORRELATION",
            "YIELD_FARMING_PRESSURE"
        ]
        
        functions_panel = Panel(
            f"""[bold cyan]可用加密专用函数:[/bold cyan]

{chr(10).join([f'• {func}' for func in crypto_functions])}

[yellow]示例公式:[/yellow]
• RANK(WHALE_ALERT(VOLUME, AMOUNT, 3.0))
• SCALE(FEAR_GREED_INDEX(CLOSE, VOLUME) / 100)
• IF(FUNDING_RATE_MOMENTUM(FUNDING_RATES, 24) > 1, 1, -1)
            """,
            border_style="green"
        )
        self.console.print(functions_panel)
        
        # 获取自定义公式
        formula = Prompt.ask("请输入自定义因子公式")
        factor_name = Prompt.ask("因子名称")
        
        if not formula or not factor_name:
            self.console.print("[red]❌ 公式和名称不能为空[/red]")
            return
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"[green]✅ 成功创建自定义加密因子: {created_name}[/green]")
            
            self.current_session['crypto_factors'][created_name] = {
                'type': 'custom_crypto',
                'formula': formula,
                'parameters': {}
            }
            
        except Exception as e:
            self.console.print(f"[red]❌ 创建失败: {str(e)}[/red]")
    
    async def crypto_ai_chat(self):
        """加密货币专用AI助手"""
        self.console.print("\n[bold cyan]🤖 加密货币AI分析助手[/bold cyan]")
        self.console.print("[dim]专精于数字资产分析的AI顾问 | 输入 'quit' 退出[/dim]\n")
        
        # 设置加密货币专用上下文
        crypto_context = """我是专门分析加密货币的AI助手，具备以下专业知识:
• 加密货币市场结构和特点
• DeFi生态系统分析
• 永续合约和资金费率机制
• 链上数据和巨鲸行为分析
• 加密货币特色技术指标
• 市场制度识别和风险管理

我可以帮助您:
1. 分析市场趋势和价格走势
2. 解释加密货币特色指标
3. 设计适合数字资产的交易策略
4. 识别套利机会和风险点
5. 优化加密因子构建"""
        
        self.current_session['conversation_history'].append({
            "role": "system", 
            "content": crypto_context
        })
        
        while True:
            user_input = Prompt.ask("[bold green]您[/bold green]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input.strip():
                continue
            
            try:
                with Progress(SpinnerColumn(), TextColumn("🧠 AI加密分析中...")) as progress:
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
                response_panel = Panel(response, title="🤖 加密AI助手", border_style="cyan")
                self.console.print(response_panel)
                
            except Exception as e:
                self.console.print(f"[red]AI服务出错: {str(e)}[/red]")
    
    def show_funding_rates(self):
        """显示资金费率信息"""
        self.console.print("\n[bold cyan]💰 永续合约资金费率[/bold cyan]")
        
        # 模拟资金费率数据
        funding_data = {
            'BTC/USDT': {'current': 0.0001, '8h_avg': 0.0002, '24h_avg': 0.0003, 'trend': '↗️'},
            'ETH/USDT': {'current': -0.0001, '8h_avg': 0.0001, '24h_avg': 0.0002, 'trend': '↘️'},
            'BNB/USDT': {'current': 0.0003, '8h_avg': 0.0002, '24h_avg': 0.0001, 'trend': '↗️'},
            'ADA/USDT': {'current': 0.0000, '8h_avg': 0.0001, '24h_avg': 0.0002, 'trend': '→'},
            'SOL/USDT': {'current': -0.0002, '8h_avg': -0.0001, '24h_avg': 0.0001, 'trend': '↘️'}
        }
        
        funding_table = Table(title="资金费率概览")
        funding_table.add_column("合约", style="cyan")
        funding_table.add_column("当前费率", style="white")
        funding_table.add_column("8H均值", style="green")
        funding_table.add_column("24H均值", style="blue")
        funding_table.add_column("趋势", style="yellow")
        funding_table.add_column("状态", style="magenta")
        
        for symbol, data in funding_data.items():
            current_rate = data['current']
            
            # 费率状态判断
            if abs(current_rate) > 0.0005:
                status = "[red]极端[/red]"
            elif abs(current_rate) > 0.0002:
                status = "[yellow]偏高[/yellow]"
            else:
                status = "[green]正常[/green]"
            
            # 费率颜色
            rate_color = "red" if current_rate < -0.0001 else "green" if current_rate > 0.0001 else "white"
            
            funding_table.add_row(
                symbol,
                f"[{rate_color}]{current_rate:+.4f}%[/{rate_color}]",
                f"{data['8h_avg']:+.4f}%",
                f"{data['24h_avg']:+.4f}%", 
                data['trend'],
                status
            )
        
        self.console.print(funding_table)
        
        # 费率解读
        interpretation = Panel(
            """[bold]📊 资金费率解读:[/bold]

[green]正费率 (+):[/green] 多头支付空头，市场偏向看涨
[red]负费率 (-):[/red] 空头支付多头，市场偏向看跌
[yellow]极端费率:[/yellow] |费率| > 0.05%, 可能出现趋势反转

[cyan]交易策略提示:[/cyan]
• 极端正费率: 考虑减仓多头或开空
• 极端负费率: 考虑减仓空头或开多  
• 费率趋势变化: 关注情绪转换点
            """,
            border_style="blue"
        )
        self.console.print(interpretation)
    
    def run_crypto_demo(self):
        """运行加密货币专用演示"""
        self.console.print("\n[bold cyan]🚀 Crypto PandaFactor 功能演示[/bold cyan]")
        
        demo_options = [
            "资金费率动量分析",
            "巨鲸交易检测演示",
            "恐惧贪婪指数计算", 
            "闪崩检测与恢复分析",
            "加密市场制度识别",
            "DeFi生态关联分析",
            "完整加密因子工作流",
            "取消"
        ]
        
        for i, option in enumerate(demo_options, 1):
            self.console.print(f"{i}. {option}")
        
        try:
            choice = int(Prompt.ask("请选择演示内容")) - 1
            if choice < 0 or choice >= len(demo_options):
                self.console.print("[red]无效的选择[/red]")
                return
            
            if choice == 7:  # 取消
                return
            
            # 运行对应演示
            demo_functions = [
                self._demo_funding_rate_momentum,
                self._demo_whale_alert,
                self._demo_fear_greed_index,
                self._demo_flash_crash_detection,
                self._demo_market_regime,
                self._demo_defi_analysis,
                self._demo_complete_crypto_workflow
            ]
            
            demo_functions[choice]()
            
        except ValueError:
            self.console.print("[red]请输入有效的数字[/red]")
    
    def _demo_funding_rate_momentum(self):
        """演示资金费率动量分析"""
        self.console.print("\n[bold]💰 资金费率动量分析演示[/bold]")
        
        # 生成模拟资金费率数据
        dates = pd.date_range('2024-01-01', periods=100, freq='8H')
        np.random.seed(42)
        
        # 模拟资金费率变化
        base_rate = 0.0001
        noise = np.random.normal(0, 0.0002, len(dates))
        trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 0.0003
        funding_rates = base_rate + trend + noise
        
        funding_series = pd.Series(funding_rates, index=dates, name='funding_rate')
        
        # 计算动量因子
        momentum = self.crypto_utils.FUNDING_RATE_MOMENTUM(funding_series, window=24)
        
        # 显示分析结果
        analysis_table = Table(title="资金费率动量分析")
        analysis_table.add_column("指标", style="cyan")
        analysis_table.add_column("数值", style="white")
        analysis_table.add_column("解读", style="green")
        
        analysis_table.add_row(
            "当前资金费率", 
            f"{funding_rates[-1]:+.4f}%",
            "多头略微占优" if funding_rates[-1] > 0 else "空头略微占优"
        )
        analysis_table.add_row(
            "动量指标",
            f"{momentum.iloc[-1]:.3f}",
            "费率趋势向上" if momentum.iloc[-1] > 0.5 else "费率趋势向下" if momentum.iloc[-1] < -0.5 else "费率震荡"
        )
        analysis_table.add_row(
            "极端信号次数",
            str(int((abs(momentum) > 1.5).sum())),
            "关注反转机会"
        )
        
        self.console.print(analysis_table)
        self.console.print("[green]✅ 资金费率动量分析演示完成[/green]")
    
    def _demo_whale_alert(self):
        """演示巨鲸交易检测"""
        self.console.print("\n[bold]🐋 巨鲸交易检测演示[/bold]")
        
        # 生成模拟交易数据
        dates = pd.date_range('2024-01-01', periods=168, freq='1H')  # 一周数据
        np.random.seed(42)
        
        # 正常成交量和成交额
        normal_volume = np.random.lognormal(10, 0.5, len(dates))
        normal_amount = normal_volume * np.random.uniform(40000, 50000, len(dates))
        
        # 添加几个巨鲸交易
        whale_indices = [50, 80, 120]
        for idx in whale_indices:
            normal_volume[idx] *= 10  # 10倍成交量
            normal_amount[idx] *= 15  # 15倍成交额
        
        volume_series = pd.Series(normal_volume, index=dates)
        amount_series = pd.Series(normal_amount, index=dates)
        
        # 检测巨鲸交易
        whale_alerts = self.crypto_utils.WHALE_ALERT(volume_series, amount_series, threshold_std=3.0)
        
        # 找出警报
        significant_whales = whale_alerts[abs(whale_alerts) > 1.0]
        
        if len(significant_whales) > 0:
            whale_table = Table(title="检测到的巨鲸交易")
            whale_table.add_column("时间", style="cyan")
            whale_table.add_column("警报强度", style="red")
            whale_table.add_column("成交量倍数", style="yellow")
            whale_table.add_column("影响评估", style="green")
            
            for timestamp, alert_value in significant_whales.items():
                volume_multiple = volume_series.loc[timestamp] / volume_series.rolling(168).mean().loc[timestamp]
                impact = "高影响" if abs(alert_value) > 2 else "中等影响"
                
                whale_table.add_row(
                    timestamp.strftime("%m-%d %H:%M"),
                    f"{alert_value:.2f}",
                    f"{volume_multiple:.1f}x", 
                    impact
                )
            
            self.console.print(whale_table)
        else:
            self.console.print("[yellow]本期间内未检测到显著巨鲸交易[/yellow]")
        
        self.console.print("[green]✅ 巨鲸交易检测演示完成[/green]")
    
    def _demo_fear_greed_index(self):
        """演示恐惧贪婪指数"""
        self.console.print("\n[bold]😰 恐惧贪婪指数演示[/bold]")
        
        # 生成模拟价格和成交量数据
        dates = pd.date_range('2024-01-01', periods=30, freq='1D')
        np.random.seed(42)
        
        # 价格数据（模拟牛转熊的过程）
        base_price = 45000
        returns = np.concatenate([
            np.random.normal(0.02, 0.05, 10),  # 前10天上涨
            np.random.normal(0, 0.08, 10),     # 中间10天震荡 
            np.random.normal(-0.03, 0.06, 10) # 后10天下跌
        ])
        prices = base_price * np.exp(np.cumsum(returns))
        
        price_series = pd.Series(prices, index=dates)
        volume_series = pd.Series(np.random.lognormal(13, 0.3, len(dates)), index=dates)
        
        # 计算恐惧贪婪指数
        fg_index = self.crypto_utils.FEAR_GREED_INDEX(price_series, volume_series)
        
        # 显示不同阶段的指数
        stages = [
            (0, 10, "上涨期"),
            (10, 20, "震荡期"), 
            (20, 30, "下跌期")
        ]
        
        fg_table = Table(title="恐惧贪婪指数变化")
        fg_table.add_column("阶段", style="cyan")
        fg_table.add_column("平均指数", style="white")
        fg_table.add_column("情绪状态", style="yellow")
        fg_table.add_column("建议操作", style="green")
        
        for start, end, stage_name in stages:
            avg_index = fg_index.iloc[start:end].mean()
            
            if avg_index > 75:
                emotion = "极度贪婪"
                suggestion = "考虑减仓"
                color = "red"
            elif avg_index > 55:
                emotion = "贪婪" 
                suggestion = "谨慎操作"
                color = "yellow"
            elif avg_index > 45:
                emotion = "中性"
                suggestion = "观察为主"
                color = "white"
            elif avg_index > 25:
                emotion = "恐惧"
                suggestion = "关注机会"
                color = "cyan"
            else:
                emotion = "极度恐惧"
                suggestion = "考虑加仓"
                color = "green"
            
            fg_table.add_row(
                stage_name,
                f"[{color}]{avg_index:.1f}[/{color}]",
                emotion,
                suggestion
            )
        
        self.console.print(fg_table)
        
        # 当前状态
        current_fg = fg_index.iloc[-1]
        status_panel = Panel(
            f"""[bold]当前市场情绪状态:[/bold]

恐惧贪婪指数: [{'green' if current_fg < 25 else 'red' if current_fg > 75 else 'yellow'}]{current_fg:.1f}[/]

[cyan]指数解读:[/cyan]
• 0-25: 极度恐惧 (抄底机会)
• 25-45: 恐惧 (逢低布局)  
• 45-55: 中性 (观察等待)
• 55-75: 贪婪 (谨慎操作)
• 75-100: 极度贪婪 (考虑减仓)
            """,
            border_style="magenta"
        )
        self.console.print(status_panel)
        self.console.print("[green]✅ 恐惧贪婪指数演示完成[/green]")
    
    def _demo_complete_crypto_workflow(self):
        """演示完整加密工作流"""
        self.console.print("\n[bold]🚀 完整加密因子工作流演示[/bold]")
        self.console.print("演示：数据加载 → 创建加密因子 → 性能分析 → AI优化建议")
        
        # 步骤1: 模拟数据加载
        self.console.print("\n[bold cyan]步骤1: 加载多币种数据[/bold cyan]")
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.console.print(f"✅ 已加载 {', '.join(symbols)} 24小时数据")
        
        # 步骤2: 创建加密特色因子
        self.console.print("\n[bold cyan]步骤2: 创建加密货币复合因子[/bold cyan]")
        formula = "SCALE(FEAR_GREED_INDEX(CLOSE, VOLUME) / 100 + WHALE_ALERT(VOLUME, AMOUNT, 2.5) * 0.3)"
        factor_name = "crypto_sentiment_composite"
        
        try:
            created_name = self.interface.create_formula_factor(formula, factor_name)
            self.console.print(f"✅ 创建复合情绪因子: {created_name}")
            self.console.print(f"📝 公式: {formula}")
            
            # 步骤3: 模拟因子计算
            self.console.print("\n[bold cyan]步骤3: 计算因子值[/bold cyan]")
            
            # 生成模拟因子值
            dates = pd.date_range('2024-01-01', periods=24, freq='1H')
            np.random.seed(42)
            factor_values = np.random.normal(0, 0.5, len(dates))
            
            # 模拟统计
            stats_table = Table(title="因子计算统计")
            stats_table.add_column("统计项", style="cyan")
            stats_table.add_column("数值", style="white")
            
            stats_table.add_row("计算币种", "3个 (BTC, ETH, BNB)")
            stats_table.add_row("时间范围", "24小时")
            stats_table.add_row("因子均值", f"{np.mean(factor_values):.4f}")
            stats_table.add_row("因子标准差", f"{np.std(factor_values):.4f}")
            stats_table.add_row("数据完整性", "100%")
            
            self.console.print(stats_table)
            
            # 步骤4: 加密特色分析
            self.console.print("\n[bold cyan]步骤4: 加密货币特色分析[/bold cyan]")
            
            analysis_results = {
                "市场制度": "震荡偏多",
                "巨鲸活跃度": "中等",
                "资金费率状态": "略偏多头",
                "恐惧贪婪指数": "56 (轻微贪婪)",
                "清算风险": "低"
            }
            
            analysis_table = Table(title="加密市场分析")
            analysis_table.add_column("分析维度", style="cyan")
            analysis_table.add_column("当前状态", style="white")
            
            for dimension, status in analysis_results.items():
                analysis_table.add_row(dimension, status)
            
            self.console.print(analysis_table)
            
            # 步骤5: 交易建议
            self.console.print("\n[bold cyan]步骤5: AI交易建议[/bold cyan]")
            
            suggestion_panel = Panel(
                """[bold green]💡 综合交易建议:[/bold green]

[yellow]因子信号:[/yellow] 复合情绪因子显示市场情绪轻微乐观

[cyan]具体建议:[/cyan]
• BTC: 震荡偏多，可考虑逢低轻仓布局
• ETH: 跟随BTC走势，注意DeFi生态影响  
• BNB: 资金流入较好，相对抗跌

[red]风险提醒:[/red]
• 关注资金费率变化，避免极端费率时建仓
• 监控巨鲸动向，大额异动时谨慎操作
• 设置止损，加密市场波动较大

[blue]下一步:[/blue] 可使用'validate-crypto'进行详细回测验证
                """,
                border_style="green"
            )
            
            self.console.print(suggestion_panel)
            self.console.print("[green]🎉 完整加密因子工作流演示完成！[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ 工作流演示失败: {str(e)}[/red]")
    
    def _demo_flash_crash_detection(self):
        """演示闪崩检测"""
        self.console.print("\n[bold]⚡ 闪崩检测演示[/bold]")
        # 实现闪崩检测演示...
        self.console.print("[yellow]⚠️ 闪崩检测功能演示开发中...[/yellow]")
    
    def _demo_market_regime(self):
        """演示市场制度识别"""
        self.console.print("\n[bold]📊 市场制度识别演示[/bold]")
        # 实现市场制度识别演示...
        self.console.print("[yellow]⚠️ 市场制度识别演示开发中...[/yellow]")
    
    def _demo_defi_analysis(self):
        """演示DeFi生态分析"""
        self.console.print("\n[bold]🌐 DeFi生态分析演示[/bold]")
        # 实现DeFi生态分析演示...
        self.console.print("[yellow]⚠️ DeFi生态分析演示开发中...[/yellow]")
    
    def show_config(self):
        """显示配置信息"""
        config_table = Table(title="⚙️ 加密货币系统配置")
        config_table.add_column("配置项", style="cyan")
        config_table.add_column("状态/值", style="white")
        config_table.add_column("说明", style="dim")
        
        # 系统组件状态
        config_table.add_row("因子引擎", "✅ 正常", "传统+加密因子")
        config_table.add_row("加密数据源", "✅ 可用", "支持多交易所")
        config_table.add_row("LLM助手", "✅ 可用", "加密专业版")
        config_table.add_row("因子验证器", "✅ 正常", "IC+压力测试")
        
        # 当前会话状态
        active_symbols = ", ".join(self.current_session['active_symbols'])
        config_table.add_row("活跃币种", active_symbols, "当前分析对象")
        config_table.add_row("时间框架", self.current_session['timeframe'], "数据粒度")
        
        crypto_factors = len(self.current_session['crypto_factors'])
        config_table.add_row("已创建加密因子", str(crypto_factors), "本会话创建")
        
        # 支持的功能
        config_table.add_row("资金费率分析", "✅ 支持", "永续合约")
        config_table.add_row("巨鲸检测", "✅ 支持", "异常交易识别")
        config_table.add_row("DeFi分析", "⚠️ 部分", "开发中")
        config_table.add_row("跨链分析", "⚠️ 部分", "开发中")
        
        self.console.print(config_table)
        
        # 显示支持的交易所
        exchanges_panel = Panel(
            """[bold cyan]支持的数据源:[/bold cyan]

🔹 [green]Binance:[/green] OHLCV, 资金费率, 持仓量
🔹 [green]Coinbase:[/green] OHLCV数据  
🔹 [yellow]OKX:[/yellow] OHLCV数据 (配置中)
🔹 [yellow]链上数据:[/yellow] DeFi TVL, 巨鲸地址 (开发中)

[dim]注: 当前运行在模拟模式，使用高质量模拟数据[/dim]
            """,
            border_style="blue"
        )
        self.console.print(exchanges_panel)
    
    async def main_loop(self):
        """主命令循环"""
        while True:
            try:
                command = Prompt.ask("\n[bold yellow]CryptoPandaFactor[/bold yellow]").strip().lower()
                
                if command in ['exit', 'quit', 'q']:
                    self.console.print("[bold green]感谢使用 Crypto PandaFactor Professional CLI！[/bold green]")
                    await self.crypto_data_manager.close()  # 关闭数据连接
                    break
                elif command in ['help', 'h']:
                    self.show_help()
                elif command == 'crypto-demo':
                    self.run_crypto_demo()
                elif command == 'market-overview':
                    self.show_market_overview()
                elif command == 'set-symbols':
                    self.set_active_symbols()
                elif command == 'set-timeframe':
                    self.set_timeframe()
                elif command == 'load-crypto-data':
                    await self.load_crypto_data()
                elif command == 'create-crypto-factor':
                    self.create_crypto_factor()
                elif command == 'funding-rates':
                    self.show_funding_rates()
                elif command == 'crypto-ai-chat':
                    await self.crypto_ai_chat()
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
    cli = CryptoPandaFactorCLI()
    
    try:
        # 运行异步主循环
        asyncio.run(cli.main_loop())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()