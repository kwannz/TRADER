#!/usr/bin/env python3
"""
PandaFactor Professional CLI 启动脚本
PandaFactor专业版命令行工具 - 启动入口
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from cli.panda_factor_cli import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"启动失败，缺少依赖: {e}")
    print("\n请确保安装了以下依赖包:")
    print("- pandas")
    print("- numpy") 
    print("- rich (可选，用于美化界面)")
    print("- openai (可选，用于AI功能)")
    print("- pymongo (可选，用于MongoDB)")
    print("- scipy (用于统计分析)")
    
    print(f"\n安装命令: pip install pandas numpy rich openai pymongo scipy")
    sys.exit(1)
    
except Exception as e:
    print(f"启动失败: {e}")
    sys.exit(1)