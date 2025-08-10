#!/usr/bin/env python3
"""
CLI界面启动脚本

启动Bloomberg Terminal风格的命令行界面
支持实时数据显示、策略管理、AI分析等功能
"""

import os
import sys
import asyncio
from pathlib import Path
import subprocess

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CLILauncher:
    """CLI启动器"""
    
    def __init__(self):
        self.project_root = project_root
        
        # 确定虚拟环境路径
        self.venv_path = self.project_root / 'venv'
        if os.name == 'nt':  # Windows
            self.python_path = self.venv_path / 'Scripts' / 'python.exe'
        else:  # Unix-like
            self.python_path = self.venv_path / 'bin' / 'python'
    
    def check_environment(self) -> bool:
        """检查运行环境"""
        print("🔍 检查CLI运行环境...")
        
        # 检查虚拟环境
        if not self.venv_path.exists():
            print("❌ 未找到虚拟环境，请先运行 python scripts/setup_local.py")
            return False
        
        if not self.python_path.exists():
            print(f"❌ 未找到Python解释器: {self.python_path}")
            return False
        
        # 检查CLI模块
        cli_main = self.project_root / 'cli_interface' / 'main.py'
        if not cli_main.exists():
            print(f"❌ CLI主模块不存在: {cli_main}")
            return False
        
        print("✅ CLI环境检查完成")
        return True
    
    def check_dependencies(self) -> bool:
        """检查CLI依赖"""
        print("📦 检查CLI依赖...")
        
        required_packages = [
            'rich', 'textual', 'asyncio', 'psutil'
        ]
        
        try:
            # 检查包是否已安装
            result = subprocess.run([
                str(self.python_path), '-m', 'pip', 'list', '--format=freeze'
            ], capture_output=True, text=True, check=True)
            
            installed_packages = {line.split('==')[0].lower() for line in result.stdout.split('\n') if '==' in line}
            
            missing_packages = []
            for package in required_packages:
                if package.lower() not in installed_packages:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ 缺少CLI依赖包: {', '.join(missing_packages)}")
                print("   请运行: python scripts/setup_local.py")
                return False
            
            print("✅ CLI依赖包检查通过")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 检查依赖失败: {e}")
            return False
    
    def check_terminal_support(self) -> bool:
        """检查终端支持"""
        print("🖥️  检查终端兼容性...")
        
        try:
            from rich.console import Console
            console = Console()
            
            # 检查颜色支持
            if not console.color_system:
                print("⚠️  终端不支持颜色，建议使用现代终端")
                return False
            
            # 检查终端大小
            if console.size.width < 120 or console.size.height < 40:
                print(f"⚠️  终端尺寸较小 ({console.size.width}×{console.size.height})")
                print("   建议使用至少 120×40 的终端窗口以获得最佳体验")
                
                # 询问用户是否继续
                response = input("是否继续启动CLI? (y/N): ").lower().strip()
                if response not in ['y', 'yes']:
                    return False
            
            print("✅ 终端兼容性检查通过")
            return True
            
        except Exception as e:
            print(f"❌ 终端检查失败: {e}")
            return False
    
    def show_cli_banner(self):
        """显示CLI启动横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                   🖥️  AI量化交易系统                          ║
║                 Bloomberg Terminal 风格界面                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🎯 核心功能:                                                ║
║     • 📊 实时市场数据 (4Hz刷新)                              ║
║     • 🤖 AI智能分析助手                                      ║
║     • 📈 策略管理与执行                                       ║
║     • 🔬 Alpha因子发现实验室                                  ║
║     • 📝 交易记录与分析                                       ║
║                                                              ║
║  ⌨️  快捷键:                                                 ║
║     • 数字键 1-6  切换功能页面                               ║
║     • R          刷新数据                                    ║
║     • H          显示帮助                                    ║
║     • Q          退出系统                                    ║
║     • Ctrl+C     强制退出                                    ║
║                                                              ║
║  💡 使用提示:                                                ║
║     • 支持鼠标点击和键盘导航                                 ║
║     • 实时数据自动更新                                       ║
║     • 支持多窗口和标签页                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_services_running(self) -> dict:
        """检查后端服务状态"""
        print("🔗 检查后端服务状态...")
        
        services_status = {
            'fastapi': False,
            'mongodb': False,
            'redis': False
        }
        
        # 检查FastAPI服务
        try:
            import httpx
            response = httpx.get('http://localhost:8000/health', timeout=3)
            if response.status_code == 200:
                services_status['fastapi'] = True
                print("✅ FastAPI服务运行正常")
            else:
                print("⚠️  FastAPI服务响应异常")
        except Exception:
            print("⚠️  FastAPI服务未启动")
        
        # 检查MongoDB
        try:
            import pymongo
            client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            client.close()
            services_status['mongodb'] = True
            print("✅ MongoDB连接正常")
        except Exception:
            print("⚠️  MongoDB未启动或连接失败")
        
        # 检查Redis
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            client.close()
            services_status['redis'] = True
            print("✅ Redis连接正常")
        except Exception:
            print("⚠️  Redis未启动或连接失败")
        
        return services_status
    
    def start_cli(self, standalone: bool = False):
        """启动CLI界面"""
        try:
            # 环境检查
            if not self.check_environment():
                return False
            
            if not self.check_dependencies():
                return False
            
            if not self.check_terminal_support():
                return False
            
            # 检查后端服务（如果不是独立模式）
            if not standalone:
                services_status = self.check_services_running()
                
                # 如果关键服务未启动，询问用户
                if not services_status['fastapi']:
                    print("\n⚠️  FastAPI后端服务未启动")
                    print("   CLI可以独立运行，但部分功能将受限")
                    
                    response = input("\n选择运行模式:\n1) 独立模式 (有限功能)\n2) 先启动后端服务\n3) 退出\n请选择 (1/2/3): ").strip()
                    
                    if response == '2':
                        print("\n请先运行: python scripts/start_server.py")
                        return False
                    elif response == '3':
                        return False
                    else:
                        standalone = True
                        print("✅ 将以独立模式启动CLI")
            
            # 显示启动横幅
            self.show_cli_banner()
            
            # 启动CLI应用
            print("🚀 正在启动CLI界面...")
            print("   提示: 如果界面显示异常，请确保终端窗口足够大")
            print("   推荐终端尺寸: 120×40 或更大\n")
            
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            if standalone:
                env['CLI_STANDALONE'] = '1'
            
            # 执行CLI主程序
            cli_main = self.project_root / 'cli_interface' / 'main.py'
            
            result = subprocess.run([
                str(self.python_path), str(cli_main)
            ], cwd=self.project_root, env=env)
            
            if result.returncode != 0:
                print(f"\n❌ CLI退出异常 (返回码: {result.returncode})")
                return False
            
            print("\n👋 感谢使用AI量化交易系统!")
            return True
            
        except KeyboardInterrupt:
            print("\n⌨️  用户中断")
            return True
        except Exception as e:
            print(f"\n❌ CLI启动失败: {e}")
            return False
    
    def run_demo_mode(self):
        """运行演示模式"""
        print("🎬 启动演示模式...")
        
        # 这里可以实现一个简化的演示版本
        # 显示一些模拟数据和界面预览
        
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.layout import Layout
            from rich.live import Live
            import time
            import random
            
            console = Console()
            
            def create_demo_layout():
                """创建演示布局"""
                layout = Layout()
                
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="main", ratio=1),
                    Layout(name="footer", size=3)
                )
                
                layout["main"].split_row(
                    Layout(name="left", ratio=2),
                    Layout(name="right", ratio=1)
                )
                
                # 标题
                layout["header"].update(
                    Panel("🤖 AI量化交易系统 - 演示模式", style="bold cyan")
                )
                
                # 底部信息
                layout["footer"].update(
                    Panel("按 Ctrl+C 退出演示模式", style="dim")
                )
                
                return layout
            
            def update_demo_data():
                """更新演示数据"""
                # 创建价格表格
                price_table = Table(title="📊 实时行情")
                price_table.add_column("交易对", style="cyan")
                price_table.add_column("价格", style="green")
                price_table.add_column("涨跌", style="red")
                price_table.add_column("成交量", style="blue")
                
                symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
                for symbol in symbols:
                    price = f"{45000 + random.uniform(-1000, 1000):.2f}"
                    change = f"{random.uniform(-0.05, 0.05):+.2%}"
                    volume = f"{random.uniform(1000000, 5000000):.0f}"
                    price_table.add_row(symbol, price, change, volume)
                
                # 创建策略表格
                strategy_table = Table(title="🎯 策略状态")
                strategy_table.add_column("策略名称", style="cyan")
                strategy_table.add_column("状态", style="green")
                strategy_table.add_column("收益率", style="yellow")
                
                strategies = ["网格策略", "DCA策略", "均线策略"]
                for strategy in strategies:
                    status = "运行中" if random.choice([True, False]) else "暂停"
                    profit = f"{random.uniform(-0.02, 0.08):+.2%}"
                    strategy_table.add_row(strategy, status, profit)
                
                return price_table, strategy_table
            
            layout = create_demo_layout()
            
            with Live(layout, console=console, refresh_per_second=2):
                for _ in range(30):  # 运行30秒
                    price_table, strategy_table = update_demo_data()
                    
                    layout["left"].update(price_table)
                    layout["right"].update(strategy_table)
                    
                    time.sleep(1)
            
            print("\n✅ 演示模式结束")
            print("💡 要体验完整功能，请启动完整系统:")
            print("   1. python scripts/setup_local.py")
            print("   2. python scripts/start_server.py")
            print("   3. python scripts/start_cli.py")
            
        except KeyboardInterrupt:
            print("\n演示模式已退出")
        except ImportError:
            print("❌ 演示模式需要rich库，请安装依赖包")
        except Exception as e:
            print(f"❌ 演示模式错误: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI量化交易系统CLI界面')
    parser.add_argument('--standalone', '-s', action='store_true', 
                       help='独立模式 (不连接后端服务)')
    parser.add_argument('--demo', '-d', action='store_true',
                       help='演示模式 (模拟数据)')
    
    args = parser.parse_args()
    
    print("🖥️  AI量化交易系统 - CLI界面启动程序")
    print("="*60)
    
    launcher = CLILauncher()
    
    if args.demo:
        launcher.run_demo_mode()
    else:
        success = launcher.start_cli(standalone=args.standalone)
        
        if not success:
            print("\n❌ CLI启动失败")
            sys.exit(1)

if __name__ == "__main__":
    main()