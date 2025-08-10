#!/usr/bin/env python3
"""
Data Laboratory Startup Script
数据实验室启动脚本
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cli.data_laboratory import DataLaboratory

def setup_basic_logging():
    """设置基础日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def display_laboratory_banner():
    """显示实验室横幅"""
    banner = """
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  ██████╗  █████╗ ████████╗ █████╗     ██╗      █████╗ ██████╗ │
    │  ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██║     ██╔══██╗██╔══██╗│
    │  ██║  ██║███████║   ██║   ███████║    ██║     ███████║██████╔╝│
    │  ██║  ██║██╔══██║   ██║   ██╔══██║    ██║     ██╔══██║██╔══██╗│
    │  ██████╔╝██║  ██║   ██║   ██║  ██║    ███████╗██║  ██║██████╔╝│
    │  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═╝╚═════╝ │
    │                                                             │
    │             🧪 CTBench 数据实验室 🧪                        │
    │                                                             │
    │        Bloomberg风格时序数据生成与分析平台                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(banner)
    print("🔬 正在启动数据实验室...")
    print("=" * 65)

async def main():
    """主函数"""
    try:
        # 显示横幅
        display_laboratory_banner()
        
        # 设置日志
        setup_basic_logging()
        
        # 创建并运行数据实验室
        laboratory = DataLaboratory()
        await laboratory.run()
        
    except KeyboardInterrupt:
        print("\n👋 感谢使用CTBench数据实验室!")
    except Exception as e:
        print(f"❌ 实验室启动失败: {e}")
        logging.error(f"Laboratory startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())