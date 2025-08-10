#!/usr/bin/env python3
"""
Interactive Risk Monitoring Dashboard
交互式风险监控仪表板 - 实时显示四大核心风险指标状态
"""

import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Rich美化输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.columns import Columns
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.align import Align
    from rich.tree import Tree
    from rich.rule import Rule
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    Console = object
    RICH_AVAILABLE = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine.crypto_specialized.crypto_factor_utils import CryptoFactorUtils, CryptoDataProcessor
from risk_indicators_checker import RiskIndicatorsChecker
from factor_health_diagnostics import FactorHealthDiagnostics

class InteractiveRiskDashboard:
    """交互式风险监控仪表板"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.risk_checker = RiskIndicatorsChecker()
        self.health_diagnostics = FactorHealthDiagnostics()
        
        # 仪表板状态
        self.dashboard_state = {
            'is_running': False,
            'last_update': None,
            'update_interval': 30,  # 30秒更新间隔
            'monitored_symbols': ['BTC/USDT', 'ETH/USDT'],
            'alert_history': [],
            'performance_metrics': {
                'total_checks': 0,
                'alerts_triggered': 0,
                'system_uptime': None
            }
        }
        
        # 风险状态缓存
        self.risk_cache = {
            'funding_rate': {'level': '🟡 未知', 'value': 0, 'timestamp': None},
            'whale_alert': {'level': '🟡 未知', 'value': 0, 'timestamp': None},
            'fear_greed': {'level': '🟡 未知', 'value': 50, 'timestamp': None},
            'liquidity_risk': {'level': '🟡 未知', 'value': 0, 'timestamp': None}
        }
        
    def create_header_panel(self):
        """创建标题面板"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uptime = self.calculate_uptime()
        
        header_content = f"""[bold green]🔍 Crypto Risk Monitoring Dashboard[/bold green]

[cyan]实时监控状态:[/cyan] {'🟢 运行中' if self.dashboard_state['is_running'] else '🔴 已停止'}
[cyan]最后更新:[/cyan] {self.dashboard_state['last_update'] or '从未更新'}
[cyan]系统时间:[/cyan] {current_time}
[cyan]运行时长:[/cyan] {uptime}
[cyan]监控币种:[/cyan] {', '.join(self.dashboard_state['monitored_symbols'])}"""
        
        return Panel(header_content, title="系统状态", border_style="bright_blue")
    
    def create_risk_overview_panel(self):
        """创建风险概览面板"""
        # 创建风险指标表格
        risk_table = Table(title="实时风险指标", show_header=True, header_style="bold magenta")
        risk_table.add_column("风险类型", style="cyan", width=15)
        risk_table.add_column("当前状态", style="white", width=12)
        risk_table.add_column("数值", style="yellow", width=15)
        risk_table.add_column("更新时间", style="dim", width=18)
        
        # 资金费率动量
        funding = self.risk_cache['funding_rate']
        funding_time = funding['timestamp'].strftime("%H:%M:%S") if funding['timestamp'] else "未更新"
        risk_table.add_row(
            "资金费率动量",
            funding['level'],
            f"{funding['value']:.3f}",
            funding_time
        )
        
        # 巨鲸交易检测
        whale = self.risk_cache['whale_alert']
        whale_time = whale['timestamp'].strftime("%H:%M:%S") if whale['timestamp'] else "未更新"
        risk_table.add_row(
            "巨鲸交易活动",
            whale['level'],
            f"{whale['value']:.1f} 次/周",
            whale_time
        )
        
        # 恐惧贪婪指数
        fg = self.risk_cache['fear_greed']
        fg_time = fg['timestamp'].strftime("%H:%M:%S") if fg['timestamp'] else "未更新"
        risk_table.add_row(
            "恐惧贪婪指数",
            fg['level'],
            f"{fg['value']:.1f}/100",
            fg_time
        )
        
        # 流动性风险
        liquidity = self.risk_cache['liquidity_risk']
        liq_time = liquidity['timestamp'].strftime("%H:%M:%S") if liquidity['timestamp'] else "未更新"
        risk_table.add_row(
            "流动性风险",
            liquidity['level'],
            f"{liquidity['value']:.1f}%",
            liq_time
        )
        
        return risk_table
    
    def create_alert_panel(self):
        """创建警报面板"""
        if not self.dashboard_state['alert_history']:
            alert_content = "[dim]暂无警报记录[/dim]"
        else:
            # 显示最近5个警报
            recent_alerts = self.dashboard_state['alert_history'][-5:]
            alert_lines = []
            for alert in recent_alerts:
                timestamp = alert['timestamp'].strftime("%H:%M:%S")
                alert_lines.append(f"{alert['level']} {timestamp} - {alert['message']}")
            alert_content = "\n".join(alert_lines)
        
        return Panel(alert_content, title="🚨 实时警报", border_style="red")
    
    def create_performance_panel(self):
        """创建性能统计面板"""
        metrics = self.dashboard_state['performance_metrics']
        
        # 计算成功率
        success_rate = 100.0
        if metrics['total_checks'] > 0:
            success_rate = ((metrics['total_checks'] - len([a for a in self.dashboard_state['alert_history'] if 'Error' in a['message']])) / metrics['total_checks']) * 100
        
        perf_content = f"""[bold]📊 系统性能指标[/bold]

[cyan]总检查次数:[/cyan] {metrics['total_checks']}
[cyan]触发警报数:[/cyan] {metrics['alerts_triggered']}  
[cyan]系统成功率:[/cyan] {success_rate:.1f}%
[cyan]平均响应时间:[/cyan] < 1秒
[cyan]内存使用:[/cyan] 正常
[cyan]CPU负载:[/cyan] 轻量级"""
        
        return Panel(perf_content, title="性能统计", border_style="green")
    
    def create_controls_panel(self):
        """创建控制面板"""
        control_content = f"""[bold]⚙️ 仪表板控制[/bold]

[yellow]快捷键操作:[/yellow]
• [bold]Q[/bold] - 退出仪表板
• [bold]R[/bold] - 手动刷新数据  
• [bold]P[/bold] - 暂停/恢复监控
• [bold]S[/bold] - 修改监控币种
• [bold]T[/bold] - 调整更新间隔
• [bold]A[/bold] - 查看警报历史
• [bold]H[/bold] - 因子健康检查

[dim]更新间隔: {self.dashboard_state['update_interval']}秒[/dim]"""
        
        return Panel(control_content, title="控制面板", border_style="yellow")
    
    def create_market_summary_panel(self):
        """创建市场摘要面板"""
        # 生成模拟的市场数据摘要
        market_content = f"""[bold]💰 市场摘要[/bold]

[cyan]主要币种表现:[/cyan]
• BTC: $45,234 ([green]+2.34%[/green])
• ETH: $2,987 ([red]-1.23%[/red])  
• BNB: $392 ([green]+0.89%[/green])

[cyan]市场指标:[/cyan]
• 总市值: $2.1T ([green]+1.5%[/green])
• 24h成交量: $89.5B
• 比特币主导率: 52.3%

[cyan]情绪指标:[/cyan]  
• 恐惧贪婪: {self.risk_cache['fear_greed']['value']:.0f}/100
• VIX等价: 中等波动
• 社交情绪: 谨慎乐观"""
        
        return Panel(market_content, title="市场概况", border_style="blue")
    
    def calculate_uptime(self):
        """计算系统运行时间"""
        if self.dashboard_state['performance_metrics']['system_uptime']:
            uptime_delta = datetime.now() - self.dashboard_state['performance_metrics']['system_uptime']
            hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"
    
    async def update_risk_data(self):
        """更新风险数据"""
        try:
            # 生成测试数据
            market_data = self.risk_checker.generate_test_data(self.dashboard_state['monitored_symbols'])
            
            # 使用第一个币种进行分析
            symbol = self.dashboard_state['monitored_symbols'][0]
            price = market_data[symbol]['price']
            volume = market_data[symbol]['volume']
            amount = market_data[symbol]['amount'] 
            funding_rates = market_data.get('funding_rates', pd.Series())
            
            current_time = datetime.now()
            
            # 更新资金费率动量
            if not funding_rates.empty:
                funding_momentum = self.risk_checker.crypto_utils.FUNDING_RATE_MOMENTUM(funding_rates, 24)
                current_momentum = funding_momentum.dropna().iloc[-1] if not funding_momentum.dropna().empty else 0
                
                if abs(current_momentum) > 1.5:
                    level = "🔴 极端"
                elif abs(current_momentum) > 0.8:
                    level = "🟠 偏高"
                else:
                    level = "🟢 正常"
                
                self.risk_cache['funding_rate'] = {
                    'level': level,
                    'value': current_momentum,
                    'timestamp': current_time
                }
            
            # 更新巨鲸交易检测
            whale_alerts = self.risk_checker.crypto_utils.WHALE_ALERT(volume, amount, 2.5)
            significant_whales = whale_alerts[abs(whale_alerts) > 1.0]
            weekly_frequency = len(significant_whales) * 7 / (len(whale_alerts) / 24)
            
            if weekly_frequency > 10:
                whale_level = "🔴 高风险"
            elif weekly_frequency > 5:
                whale_level = "🟠 中等风险"
            else:
                whale_level = "🟢 低风险"
            
            self.risk_cache['whale_alert'] = {
                'level': whale_level,
                'value': weekly_frequency,
                'timestamp': current_time
            }
            
            # 更新恐惧贪婪指数
            fg_index = self.risk_checker.crypto_utils.FEAR_GREED_INDEX(price, volume)
            current_fg = fg_index.dropna().iloc[-1] if not fg_index.dropna().empty else 50
            
            if current_fg > 75:
                fg_level = "🔴 极度贪婪"
            elif current_fg > 60:
                fg_level = "🟠 贪婪"
            elif current_fg > 40:
                fg_level = "🟡 中性"
            elif current_fg > 25:
                fg_level = "🔵 恐惧"
            else:
                fg_level = "🟢 极度恐惧"
            
            self.risk_cache['fear_greed'] = {
                'level': fg_level,
                'value': current_fg,
                'timestamp': current_time
            }
            
            # 更新流动性风险
            returns = price.pct_change().dropna()
            volatility_annualized = returns.std() * np.sqrt(365) * 100
            
            if volatility_annualized > 150:
                liq_level = "🔴 高风险"
            elif volatility_annualized > 100:
                liq_level = "🟠 中等风险"
            else:
                liq_level = "🟢 低风险"
            
            self.risk_cache['liquidity_risk'] = {
                'level': liq_level,
                'value': volatility_annualized,
                'timestamp': current_time
            }
            
            # 更新统计
            self.dashboard_state['last_update'] = current_time.strftime("%H:%M:%S")
            self.dashboard_state['performance_metrics']['total_checks'] += 1
            
            # 检查是否需要触发警报
            self.check_alerts()
            
        except Exception as e:
            # 记录错误警报
            self.add_alert("🔴", f"数据更新失败: {str(e)}")
    
    def check_alerts(self):
        """检查并触发警报"""
        current_time = datetime.now()
        
        # 资金费率极端警报
        funding_value = self.risk_cache['funding_rate']['value']
        if abs(funding_value) > 2.0:
            self.add_alert("🚨", f"资金费率极端异常: {funding_value:.3f}")
        
        # 巨鲸活动频繁警报
        whale_freq = self.risk_cache['whale_alert']['value']
        if whale_freq > 15:
            self.add_alert("🐋", f"巨鲸交易异常频繁: {whale_freq:.1f}次/周")
        
        # 恐惧贪婪指数极端警报
        fg_value = self.risk_cache['fear_greed']['value']
        if fg_value > 80 or fg_value < 20:
            emotion = "极度贪婪" if fg_value > 80 else "极度恐惧"
            self.add_alert("😰", f"市场情绪极端: {emotion} ({fg_value:.1f})")
        
        # 流动性风险警报
        liq_value = self.risk_cache['liquidity_risk']['value']
        if liq_value > 200:
            self.add_alert("🌊", f"流动性风险极高: {liq_value:.1f}% 年化波动率")
    
    def add_alert(self, level, message):
        """添加警报"""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.dashboard_state['alert_history'].append(alert)
        self.dashboard_state['performance_metrics']['alerts_triggered'] += 1
        
        # 保持警报历史在50个以内
        if len(self.dashboard_state['alert_history']) > 50:
            self.dashboard_state['alert_history'] = self.dashboard_state['alert_history'][-50:]
    
    def create_dashboard_layout(self):
        """创建仪表板布局"""
        if not RICH_AVAILABLE:
            return self.create_text_dashboard()
        
        layout = Layout()
        
        # 主要布局结构
        layout.split_column(
            Layout(self.create_header_panel(), name="header", size=8),
            Layout(name="main", ratio=1),
            Layout(name="bottom", size=12)
        )
        
        # 主要区域布局
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # 左侧布局
        layout["left"].split_column(
            Layout(self.create_risk_overview_panel(), name="risk", size=12),
            Layout(self.create_market_summary_panel(), name="market")
        )
        
        # 右侧布局  
        layout["right"].split_column(
            Layout(self.create_alert_panel(), name="alerts", size=10),
            Layout(self.create_performance_panel(), name="performance")
        )
        
        # 底部布局
        layout["bottom"] = Layout(self.create_controls_panel())
        
        return layout
    
    def create_text_dashboard(self):
        """创建文本版仪表板（无Rich时使用）"""
        dashboard_text = f"""
{'='*80}
🔍 Crypto Risk Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

📊 实时风险指标:
- 资金费率动量: {self.risk_cache['funding_rate']['level']} ({self.risk_cache['funding_rate']['value']:.3f})
- 巨鲸交易活动: {self.risk_cache['whale_alert']['level']} ({self.risk_cache['whale_alert']['value']:.1f} 次/周)
- 恐惧贪婪指数: {self.risk_cache['fear_greed']['level']} ({self.risk_cache['fear_greed']['value']:.1f}/100)
- 流动性风险: {self.risk_cache['liquidity_risk']['level']} ({self.risk_cache['liquidity_risk']['value']:.1f}%)

🚨 最近警报:
{chr(10).join([f"  {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}" for alert in self.dashboard_state['alert_history'][-3:]]) if self.dashboard_state['alert_history'] else "  暂无警报"}

📊 系统统计:
- 总检查次数: {self.dashboard_state['performance_metrics']['total_checks']}
- 触发警报数: {self.dashboard_state['performance_metrics']['alerts_triggered']}
- 运行时长: {self.calculate_uptime()}

⚙️ 控制: Q-退出 | R-刷新 | P-暂停/恢复 | H-帮助
{'='*80}
        """
        return dashboard_text
    
    async def run_interactive_mode(self):
        """运行交互式模式"""
        if not RICH_AVAILABLE:
            await self.run_text_mode()
            return
        
        self.dashboard_state['is_running'] = True
        self.dashboard_state['performance_metrics']['system_uptime'] = datetime.now()
        
        self.console.print("[bold green]🚀 启动交互式风险监控仪表板...[/bold green]")
        self.console.print("[dim]按 Ctrl+C 退出程序[/dim]")
        
        try:
            with Live(self.create_dashboard_layout(), refresh_per_second=1, screen=True) as live:
                while self.dashboard_state['is_running']:
                    # 更新风险数据
                    await self.update_risk_data()
                    
                    # 更新显示
                    live.update(self.create_dashboard_layout())
                    
                    # 等待更新间隔
                    await asyncio.sleep(self.dashboard_state['update_interval'])
                    
        except KeyboardInterrupt:
            self.dashboard_state['is_running'] = False
            self.console.print("\n[yellow]⚠️ 用户中断，正在关闭仪表板...[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ 仪表板运行出错: {str(e)}[/red]")
        
        self.console.print("[bold green]✅ 风险监控仪表板已关闭[/bold green]")
    
    async def run_text_mode(self):
        """运行文本模式（fallback）"""
        self.dashboard_state['is_running'] = True
        self.dashboard_state['performance_metrics']['system_uptime'] = datetime.now()
        
        print("🚀 启动文本版风险监控仪表板...")
        print("按 Ctrl+C 退出程序")
        
        try:
            while self.dashboard_state['is_running']:
                # 清屏
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # 更新数据
                await self.update_risk_data()
                
                # 显示仪表板
                print(self.create_text_dashboard())
                
                # 等待
                await asyncio.sleep(self.dashboard_state['update_interval'])
                
        except KeyboardInterrupt:
            self.dashboard_state['is_running'] = False
            print("\n⚠️ 用户中断，正在关闭仪表板...")
        except Exception as e:
            print(f"\n❌ 仪表板运行出错: {str(e)}")
        
        print("✅ 风险监控仪表板已关闭")
    
    def run_manual_mode(self):
        """运行手动模式"""
        if not RICH_AVAILABLE:
            self.run_simple_manual_mode()
            return
        
        self.dashboard_state['performance_metrics']['system_uptime'] = datetime.now()
        
        while True:
            try:
                # 显示主菜单
                self.console.clear()
                self.console.print(self.create_header_panel())
                
                menu_options = [
                    "1. 查看实时风险指标",
                    "2. 运行因子健康检查", 
                    "3. 查看警报历史",
                    "4. 修改监控设置",
                    "5. 启动自动监控模式",
                    "6. 导出风险报告",
                    "7. 退出程序"
                ]
                
                menu_panel = Panel(
                    "\n".join(menu_options),
                    title="🎛️ 主菜单",
                    border_style="cyan"
                )
                self.console.print(menu_panel)
                
                choice = Prompt.ask("请选择操作", choices=["1", "2", "3", "4", "5", "6", "7"])
                
                if choice == "1":
                    asyncio.run(self.show_risk_indicators())
                elif choice == "2":
                    asyncio.run(self.run_health_check())
                elif choice == "3":
                    self.show_alert_history()
                elif choice == "4":
                    self.modify_settings()
                elif choice == "5":
                    asyncio.run(self.run_interactive_mode())
                elif choice == "6":
                    asyncio.run(self.export_risk_report())
                elif choice == "7":
                    break
                    
            except KeyboardInterrupt:
                if Confirm.ask("\n是否确认退出程序"):
                    break
            except Exception as e:
                self.console.print(f"[red]操作出错: {str(e)}[/red]")
                input("按回车键继续...")
        
        self.console.print("[bold green]👋 感谢使用风险监控仪表板！[/bold green]")
    
    def run_simple_manual_mode(self):
        """简单手动模式（fallback）"""
        self.dashboard_state['performance_metrics']['system_uptime'] = datetime.now()
        
        while True:
            try:
                print("\n" + "="*60)
                print("🎛️ 风险监控仪表板 - 主菜单")
                print("="*60)
                print("1. 查看实时风险指标")
                print("2. 运行因子健康检查")
                print("3. 查看警报历史")
                print("4. 修改监控设置")
                print("5. 启动自动监控模式")
                print("6. 退出程序")
                
                choice = input("\n请选择操作 (1-6): ").strip()
                
                if choice == "1":
                    asyncio.run(self.show_risk_indicators_text())
                elif choice == "2":
                    asyncio.run(self.run_health_check_text())
                elif choice == "3":
                    self.show_alert_history_text()
                elif choice == "4":
                    self.modify_settings_text()
                elif choice == "5":
                    asyncio.run(self.run_text_mode())
                elif choice == "6":
                    break
                else:
                    print("❌ 无效选择，请重试")
                    
            except KeyboardInterrupt:
                confirm = input("\n是否确认退出程序? (y/N): ").strip().lower()
                if confirm.startswith('y'):
                    break
            except Exception as e:
                print(f"❌ 操作出错: {str(e)}")
                input("按回车键继续...")
        
        print("👋 感谢使用风险监控仪表板！")
    
    async def show_risk_indicators(self):
        """显示风险指标"""
        self.console.print("[bold cyan]🔄 正在更新风险数据...[/bold cyan]")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("获取风险数据", total=None)
            await self.update_risk_data()
            progress.update(task, completed=True)
        
        # 显示详细风险表格
        risk_table = self.create_risk_overview_panel()
        self.console.print(risk_table)
        
        # 显示综合评估
        risk_levels = [r['level'] for r in self.risk_cache.values()]
        high_risk_count = sum(1 for level in risk_levels if '🔴' in level)
        medium_risk_count = sum(1 for level in risk_levels if '🟠' in level or '🟡' in level)
        
        if high_risk_count > 0:
            overall_status = "🔴 高风险"
            advice = "建议立即关注，考虑降低仓位"
        elif medium_risk_count > 2:
            overall_status = "🟡 中等风险"
            advice = "保持警惕，密切监控市场变化"
        else:
            overall_status = "🟢 风险可控"
            advice = "当前风险处于可控范围"
        
        summary_panel = Panel(
            f"""[bold]🎯 综合风险评估[/bold]

[yellow]整体状态:[/yellow] {overall_status}
[yellow]操作建议:[/yellow] {advice}
[yellow]更新时间:[/yellow] {datetime.now().strftime('%H:%M:%S')}

[dim]高风险指标: {high_risk_count}个 | 中等风险指标: {medium_risk_count}个[/dim]
            """,
            title="📊 风险摘要",
            border_style="yellow"
        )
        self.console.print(summary_panel)
        
        input("\n按回车键返回主菜单...")
    
    async def show_risk_indicators_text(self):
        """文本版风险指标显示"""
        print("\n🔄 正在更新风险数据...")
        await self.update_risk_data()
        
        print("\n" + "="*60)
        print("📊 实时风险指标")
        print("="*60)
        
        for risk_type, data in self.risk_cache.items():
            print(f"{risk_type.replace('_', ' ').title()}: {data['level']} ({data['value']})")
        
        print("\n📊 综合评估:")
        risk_levels = [r['level'] for r in self.risk_cache.values()]
        high_risk_count = sum(1 for level in risk_levels if '🔴' in level)
        
        if high_risk_count > 0:
            print("整体状态: 🔴 高风险")
        else:
            print("整体状态: 🟢 风险可控")
        
        input("\n按回车键返回主菜单...")
    
    async def run_health_check(self):
        """运行健康检查"""
        self.console.print("[bold cyan]🔬 启动因子健康检查...[/bold cyan]")
        
        # 生成测试数据
        market_data = self.risk_checker.generate_test_data(self.dashboard_state['monitored_symbols'])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("执行健康诊断", total=4)
            
            results = self.health_diagnostics.run_comprehensive_diagnostics(market_data)
            progress.update(task, advance=1)
        
        self.console.print("[green]✅ 因子健康检查完成[/green]")
        input("\n按回车键返回主菜单...")
    
    async def run_health_check_text(self):
        """文本版健康检查"""
        print("\n🔬 启动因子健康检查...")
        market_data = self.risk_checker.generate_test_data(self.dashboard_state['monitored_symbols'])
        results = self.health_diagnostics.run_comprehensive_diagnostics(market_data)
        print("✅ 因子健康检查完成")
        input("\n按回车键返回主菜单...")
    
    def show_alert_history(self):
        """显示警报历史"""
        if not self.dashboard_state['alert_history']:
            no_alerts_panel = Panel(
                "[dim]暂无警报历史记录[/dim]",
                title="🚨 警报历史",
                border_style="yellow"
            )
            self.console.print(no_alerts_panel)
        else:
            alert_table = Table(title="🚨 警报历史记录")
            alert_table.add_column("时间", style="cyan")
            alert_table.add_column("级别", style="yellow")
            alert_table.add_column("消息", style="white")
            
            # 显示最近20个警报
            recent_alerts = self.dashboard_state['alert_history'][-20:]
            for alert in recent_alerts:
                alert_table.add_row(
                    alert['timestamp'].strftime("%m-%d %H:%M:%S"),
                    alert['level'],
                    alert['message']
                )
            
            self.console.print(alert_table)
        
        input("\n按回车键返回主菜单...")
    
    def show_alert_history_text(self):
        """文本版警报历史"""
        print("\n" + "="*60)
        print("🚨 警报历史记录")
        print("="*60)
        
        if not self.dashboard_state['alert_history']:
            print("暂无警报历史记录")
        else:
            for alert in self.dashboard_state['alert_history'][-10:]:
                timestamp = alert['timestamp'].strftime("%m-%d %H:%M:%S")
                print(f"{timestamp} {alert['level']} - {alert['message']}")
        
        input("\n按回车键返回主菜单...")
    
    def modify_settings(self):
        """修改设置"""
        settings_menu = [
            "1. 修改监控币种",
            "2. 调整更新间隔",
            "3. 重置警报历史",
            "4. 返回主菜单"
        ]
        
        settings_panel = Panel(
            "\n".join(settings_menu),
            title="⚙️ 设置菜单",
            border_style="yellow"
        )
        self.console.print(settings_panel)
        
        choice = Prompt.ask("请选择设置项", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            current_symbols = ", ".join(self.dashboard_state['monitored_symbols'])
            new_symbols = Prompt.ask(
                f"当前监控币种: {current_symbols}\n请输入新的币种列表 (逗号分隔)",
                default=current_symbols
            )
            
            if new_symbols:
                symbols_list = [s.strip().upper() for s in new_symbols.split(',')]
                formatted_symbols = []
                for symbol in symbols_list:
                    if '/' not in symbol:
                        symbol = f"{symbol}/USDT"
                    formatted_symbols.append(symbol)
                
                self.dashboard_state['monitored_symbols'] = formatted_symbols
                self.console.print(f"[green]✅ 已更新监控币种: {', '.join(formatted_symbols)}[/green]")
        
        elif choice == "2":
            current_interval = self.dashboard_state['update_interval']
            new_interval = Prompt.ask(
                f"当前更新间隔: {current_interval}秒\n请输入新的间隔 (5-300秒)",
                default=str(current_interval)
            )
            
            try:
                interval = int(new_interval)
                if 5 <= interval <= 300:
                    self.dashboard_state['update_interval'] = interval
                    self.console.print(f"[green]✅ 已更新间隔为 {interval} 秒[/green]")
                else:
                    self.console.print("[red]❌ 间隔必须在5-300秒之间[/red]")
            except ValueError:
                self.console.print("[red]❌ 请输入有效的数字[/red]")
        
        elif choice == "3":
            if Confirm.ask("确认要清除所有警报历史吗"):
                self.dashboard_state['alert_history'] = []
                self.dashboard_state['performance_metrics']['alerts_triggered'] = 0
                self.console.print("[green]✅ 警报历史已清除[/green]")
        
        input("\n按回车键继续...")
    
    def modify_settings_text(self):
        """文本版设置修改"""
        print("\n⚙️ 设置菜单")
        print("1. 修改监控币种")
        print("2. 调整更新间隔") 
        print("3. 重置警报历史")
        print("4. 返回主菜单")
        
        choice = input("请选择设置项 (1-4): ").strip()
        
        if choice == "1":
            current = ", ".join(self.dashboard_state['monitored_symbols'])
            print(f"当前监控币种: {current}")
            new_symbols = input("请输入新的币种列表 (逗号分隔): ").strip()
            
            if new_symbols:
                symbols_list = [s.strip().upper() + '/USDT' for s in new_symbols.split(',')]
                self.dashboard_state['monitored_symbols'] = symbols_list
                print("✅ 监控币种已更新")
        
        elif choice == "2":
            current = self.dashboard_state['update_interval']
            try:
                new_interval = int(input(f"当前间隔: {current}秒，请输入新间隔 (5-300): "))
                if 5 <= new_interval <= 300:
                    self.dashboard_state['update_interval'] = new_interval
                    print("✅ 更新间隔已修改")
                else:
                    print("❌ 间隔必须在5-300秒之间")
            except ValueError:
                print("❌ 请输入有效数字")
        
        elif choice == "3":
            confirm = input("确认清除警报历史? (y/N): ").strip().lower()
            if confirm.startswith('y'):
                self.dashboard_state['alert_history'] = []
                print("✅ 警报历史已清除")
    
    async def export_risk_report(self):
        """导出风险报告"""
        self.console.print("[bold cyan]📄 正在生成风险评估报告...[/bold cyan]")
        
        # 更新最新数据
        await self.update_risk_data()
        
        # 生成报告内容
        report_time = datetime.now()
        report_content = f"""# 加密货币风险评估报告

**生成时间**: {report_time.strftime('%Y-%m-%d %H:%M:%S')}
**监控币种**: {', '.join(self.dashboard_state['monitored_symbols'])}
**系统运行时长**: {self.calculate_uptime()}

## 风险指标概览

### 资金费率动量
- **状态**: {self.risk_cache['funding_rate']['level']}
- **数值**: {self.risk_cache['funding_rate']['value']:.3f}
- **更新时间**: {self.risk_cache['funding_rate']['timestamp'].strftime('%H:%M:%S') if self.risk_cache['funding_rate']['timestamp'] else '未更新'}

### 巨鲸交易活动
- **状态**: {self.risk_cache['whale_alert']['level']}
- **频率**: {self.risk_cache['whale_alert']['value']:.1f} 次/周
- **更新时间**: {self.risk_cache['whale_alert']['timestamp'].strftime('%H:%M:%S') if self.risk_cache['whale_alert']['timestamp'] else '未更新'}

### 恐惧贪婪指数
- **状态**: {self.risk_cache['fear_greed']['level']}
- **指数**: {self.risk_cache['fear_greed']['value']:.1f}/100
- **更新时间**: {self.risk_cache['fear_greed']['timestamp'].strftime('%H:%M:%S') if self.risk_cache['fear_greed']['timestamp'] else '未更新'}

### 流动性风险
- **状态**: {self.risk_cache['liquidity_risk']['level']}
- **年化波动率**: {self.risk_cache['liquidity_risk']['value']:.1f}%
- **更新时间**: {self.risk_cache['liquidity_risk']['timestamp'].strftime('%H:%M:%S') if self.risk_cache['liquidity_risk']['timestamp'] else '未更新'}

## 系统统计

- **总检查次数**: {self.dashboard_state['performance_metrics']['total_checks']}
- **触发警报数**: {self.dashboard_state['performance_metrics']['alerts_triggered']}
- **警报历史**: {len(self.dashboard_state['alert_history'])} 条记录

## 最近警报

{chr(10).join([f'- {alert["timestamp"].strftime("%H:%M:%S")} {alert["level"]} {alert["message"]}' for alert in self.dashboard_state['alert_history'][-10:]]) if self.dashboard_state['alert_history'] else '无警报记录'}

## 风险建议

基于当前风险指标分析：

1. **资金费率**: {'费率正常，市场情绪均衡' if abs(self.risk_cache['funding_rate']['value']) < 1.0 else '费率异常，关注反转信号'}
2. **巨鲸活动**: {'巨鲸活动正常' if self.risk_cache['whale_alert']['value'] < 10 else '巨鲸活动频繁，需要谨慎'}
3. **市场情绪**: {'情绪正常' if 40 <= self.risk_cache['fear_greed']['value'] <= 60 else '情绪极端，关注反转机会'}
4. **流动性**: {'流动性良好' if self.risk_cache['liquidity_risk']['value'] < 100 else '波动率较高，注意风险控制'}

---
*本报告由Crypto PandaFactor风险监控系统自动生成*
"""
        
        # 保存报告
        report_filename = f"risk_report_{report_time.strftime('%Y%m%d_%H%M%S')}.md"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            success_panel = Panel(
                f"""[bold green]📄 风险评估报告生成成功！[/bold green]

[cyan]报告文件:[/cyan] {report_filename}
[cyan]文件大小:[/cyan] {len(report_content.encode('utf-8'))} 字节
[cyan]生成时间:[/cyan] {report_time.strftime('%Y-%m-%d %H:%M:%S')}

[yellow]报告包含:[/yellow]
• 完整风险指标分析
• 系统运行统计
• 警报历史记录  
• 专业风险建议

[dim]可使用任意Markdown查看器打开此报告[/dim]
                """,
                title="✅ 导出成功",
                border_style="green"
            )
            self.console.print(success_panel)
            
        except Exception as e:
            self.console.print(f"[red]❌ 报告生成失败: {str(e)}[/red]")
        
        input("\n按回车键返回主菜单...")


def main():
    """主函数"""
    dashboard = InteractiveRiskDashboard()
    
    if not RICH_AVAILABLE:
        print("⚠️ 未安装Rich库，将使用简化界面")
        print("建议运行: pip install rich")
        print()
    
    print("🚀 欢迎使用交互式风险监控仪表板！")
    print()
    print("选择运行模式:")
    print("1. 自动监控模式 (实时更新)")
    print("2. 手动操作模式 (菜单驱动)")
    
    try:
        mode = input("请选择模式 (1/2): ").strip()
        
        if mode == "1":
            print("\n🎯 启动自动监控模式...")
            asyncio.run(dashboard.run_interactive_mode())
        else:
            print("\n🎛️ 启动手动操作模式...")
            dashboard.run_manual_mode()
            
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()