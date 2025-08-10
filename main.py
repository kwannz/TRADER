#!/usr/bin/env python3
"""
AI量化交易系统 - 主程序入口
CLI终端应用启动器
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

from config.settings import settings
from config.bloomberg_theme import get_theme, BLOOMBERG_COLORS, STATUS_INDICATORS
from core.app import QuantumTraderApp
from core.unified_logger import get_logger, LogCategory, setup_database_logging
from core.verbose_logger import (
    get_verbose_logger, setup_verbose_logging, log_startup_sequence, 
    log_shutdown_sequence, log_configuration_loaded
)

class QuantumTraderCLI:
    """AI量化交易系统CLI启动器"""
    
    def __init__(self):
        self.console = Console(theme=get_theme())
        self.app: Optional[QuantumTraderApp] = None
        self.logger = get_logger()
        self.verbose_logger = get_verbose_logger()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器 - 优雅关闭"""
        self.logger.info(f"收到信号 {signum}，正在优雅关闭...", LogCategory.SYSTEM)
        if self.app:
            asyncio.create_task(self.app.shutdown())
        sys.exit(0)
    
    def display_startup_banner(self):
        """显示启动横幅"""
        # ASCII艺术Logo
        logo_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║     ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗     ║
    ║    ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║     ║
    ║    ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║     ║
    ║    ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║     ║
    ║    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝     ║
    ║     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝      ║
    ║                                                               ║
    ║            ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗   ║
    ║            ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗  ║
    ║               ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝  ║
    ║               ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗  ║
    ║               ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║  ║
    ║               ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝  ║
    ╚═══════════════════════════════════════════════════════════════╝
        """
        
        title_text = Text()
        title_text.append("AI量化交易系统", style="bold blue")
        title_text.append(" v", style="white")
        title_text.append(settings.version, style="bold green")
        title_text.append(" - ", style="white") 
        title_text.append("Bloomberg Terminal风格", style="bold cyan")
        
        subtitle_text = Text()
        subtitle_text.append("🤖 AI驱动 ", style="bold magenta")
        subtitle_text.append("• 📊 实时交易 ", style="bold green")
        subtitle_text.append("• 🔬 因子发现 ", style="bold blue")
        subtitle_text.append("• ⚡ 智能执行", style="bold yellow")
        
        panel = Panel(
            Align.center(f"{logo_text}\n\n{title_text}\n{subtitle_text}"),
            title="🚀 启动中",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(panel)
        self.console.print()
    
    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        self.console.print("🔍 检查系统要求...", style="bold yellow")
        
        # 检查Python版本
        if sys.version_info < (3, 9):
            self.console.print(f"❌ Python版本过低，需要3.9+，当前: {sys.version}", style="bold red")
            return False
        self.console.print(f"✅ Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", style="green")
        
        # 检查终端尺寸
        try:
            size = self.console.size
            if size.width < settings.min_terminal_width or size.height < settings.min_terminal_height:
                self.console.print(
                    f"⚠️  终端尺寸较小 ({size.width}x{size.height})，建议至少 {settings.min_terminal_width}x{settings.min_terminal_height}",
                    style="bold yellow"
                )
            else:
                self.console.print(f"✅ 终端尺寸: {size.width}x{size.height}", style="green")
        except Exception as e:
            self.logger.warning(f"无法检测终端尺寸: {e}", LogCategory.SYSTEM)
        
        # 检查配置
        missing_configs = settings.validate_config()
        if missing_configs:
            self.console.print("⚠️  以下配置缺失，请在.env文件中配置:", style="bold yellow")
            for config in missing_configs:
                self.console.print(f"   • {config}", style="yellow")
            self.console.print("🔧 系统将在模拟模式下运行", style="bold cyan")
        else:
            self.console.print("✅ 配置文件完整", style="green")
        
        return True
    
    def display_system_info(self):
        """显示系统信息"""
        info_text = f"""
[bold]系统信息[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 数据库: MongoDB + Redis
🤖 AI引擎: DeepSeek + Gemini  
📡 交易所: OKX + Binance WebSocket
🔬 模型: 8个TSG模型 (CTBench集成)
💡 因子: 125+个量化因子
⚡ 刷新: {settings.refresh_rate_hz}Hz 实时数据
🛡️  风控: {settings.hard_stop_loss} USDT硬止损
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[bold cyan]快捷键说明[/bold cyan]
[1] 主仪表盘  [2] 策略管理  [3] AI助手   [4] 因子发现
[5] 交易记录  [6] 系统设置  [Q] 安全退出  [H] 帮助

[bold green]启动完成! 🎉[/bold green]
        """
        
        panel = Panel(
            info_text,
            title="📋 系统配置",
            border_style="bright_green",
            padding=(0, 2)
        )
        
        self.console.print(panel)
    
    async def initialize_app(self):
        """初始化应用"""
        try:
            self.console.print("🔧 初始化应用组件...", style="bold yellow")
            self.logger.info("开始初始化应用组件", LogCategory.SYSTEM)
            
            # 设置数据库日志（如果配置了数据库）
            try:
                db_config = settings.get_database_config()
                setup_database_logging(db_config.get("mongodb_url", ""), db_config.get("redis_url", ""))
                self.logger.success("数据库日志系统已启用", LogCategory.DATABASE)
            except Exception as e:
                self.logger.warning(f"数据库日志系统启用失败，将使用文件日志: {e}", LogCategory.DATABASE)
            
            # 创建应用实例
            self.app = QuantumTraderApp()
            
            # 异步初始化
            await self.app.initialize()
            
            self.console.print("✅ 应用初始化完成!", style="bold green")
            self.logger.success("应用初始化完成", LogCategory.SYSTEM)
            return True
            
        except Exception as e:
            self.logger.error(f"应用初始化失败: {e}", LogCategory.SYSTEM, exception=e)
            self.console.print(f"❌ 初始化失败: {e}", style="bold red")
            return False
    
    async def run(self):
        """运行主程序"""
        try:
            # Initialize verbose logging system
            setup_verbose_logging()
            
            self.logger.info("启动AI量化交易系统", LogCategory.SYSTEM)
            log_startup_sequence("AI量化交易系统", settings.version)
            
            # 显示启动横幅
            self.display_startup_banner()
            
            # 检查系统要求
            if not self.check_system_requirements():
                self.console.print("❌ 系统要求检查失败，退出程序", style="bold red")
                self.logger.error("系统要求检查失败", LogCategory.SYSTEM)
                return 1
            
            # 初始化应用
            if not await self.initialize_app():
                self.console.print("❌ 应用初始化失败，退出程序", style="bold red")
                return 1
            
            # 显示系统信息
            self.display_system_info()
            
            # 启动Textual应用
            self.console.print("🚀 启动CLI界面...", style="bold green")
            self.logger.info("启动CLI界面", LogCategory.SYSTEM)
            await self.app.run_async()
            
            self.logger.success("程序正常退出", LogCategory.SYSTEM)
            return 0
            
        except KeyboardInterrupt:
            self.console.print("\n👋 用户中断，正在退出...", style="bold yellow")
            self.logger.info("用户中断程序运行", LogCategory.SYSTEM)
            return 0
        except Exception as e:
            self.logger.error(f"程序运行错误: {e}", LogCategory.SYSTEM, exception=e)
            self.console.print(f"\n❌ 程序异常: {e}", style="bold red")
            return 1
        finally:
            if self.app:
                self.logger.info("正在关闭应用", LogCategory.SYSTEM)
                log_shutdown_sequence("AI量化交易系统")
                await self.app.shutdown()
            await self.logger.close()

def main():
    """主入口函数"""
    if sys.platform == "win32":
        # Windows下设置事件循环策略
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 创建CLI实例并运行
    cli = QuantumTraderCLI()
    
    try:
        exit_code = asyncio.run(cli.run())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 再见!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()