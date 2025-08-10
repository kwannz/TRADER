#!/usr/bin/env python3
"""
AI量化交易系统 - 一键启动脚本
自动安装依赖、检查环境、启动实时数据服务器
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🚀 AI量化交易系统 - 实时数据服务器                                        ║
║    Bloomberg风格 | 实时WebSocket数据流 | 4Hz刷新率                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本检查通过: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """安装依赖包"""
    print("📦 检查并安装依赖包...")
    
    try:
        # 检查pip版本
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        
        # 升级pip
        print("   ⬆️  升级pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # 安装依赖
        print("   📥 安装依赖包...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        print("   请手动运行: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ 未找到pip，请检查Python安装")
        return False

def check_files():
    """检查必要文件"""
    required_files = [
        'index.html',
        'styles.css', 
        'app.js',
        'server.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 文件检查完成")
    return True

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试依赖模块...")
    
    test_modules = [
        ('aiohttp', 'Web框架'),
        ('ccxt', '交易所API'),
        ('pandas', '数据处理'),
        ('numpy', '数值计算'),
        ('websockets', 'WebSocket')
    ]
    
    for module, description in test_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} ({description})")
        except ImportError:
            print(f"   ❌ {module} ({description}) - 导入失败")
            return False
    
    print("✅ 模块测试通过")
    return True

def start_server():
    """启动服务器"""
    print("🚀 启动实时数据服务器...")
    
    try:
        # 启动服务器进程
        print("   📡 初始化WebSocket连接...")
        print("   🔌 连接交易所API...")
        print("   ⚡ 启动4Hz实时数据流...")
        
        # 等待1秒后自动打开浏览器
        print("   🌐 准备打开浏览器...")
        time.sleep(1)
        
        # 在新的进程中启动服务器
        server_process = subprocess.Popen([sys.executable, 'server.py'])
        
        # 等待服务器启动
        print("   ⏳ 等待服务器启动...")
        time.sleep(3)
        
        # 打开浏览器
        print("   🔄 打开浏览器...")
        webbrowser.open('http://localhost:8000')
        
        print("\n✅ 服务器启动成功!")
        print("┌─────────────────────────────────────────────────────────┐")
        print("│  🌐 前端界面: http://localhost:8000                     │")
        print("│  🔌 WebSocket: ws://localhost:8000/ws                  │") 
        print("│  🔗 API测试: http://localhost:8000/api/market          │")
        print("│  📊 实时数据: 4Hz刷新频率                               │")
        print("└─────────────────────────────────────────────────────────┘")
        print("\n💡 使用提示:")
        print("   • 实时价格数据来自OKX和Binance交易所")
        print("   • 支持BTC、ETH、BNB、SOL等主流币种")
        print("   • WebSocket自动重连机制")
        print("   • Bloomberg风格专业界面")
        print("\n🛑 按 Ctrl+C 停止服务器")
        
        # 等待服务器进程
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n\n🛑 正在停止服务器...")
            server_process.terminate()
            server_process.wait()
            print("👋 服务器已停止")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False

def main():
    """主函数"""
    print_banner()
    
    # 切换到脚本目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📂 工作目录: {script_dir}")
    print()
    
    # 检查系统环境
    if not check_python_version():
        return False
    
    if not check_files():
        return False
    
    # 询问是否安装依赖
    response = input("📦 是否安装/更新依赖包? (y/n, 默认y): ").lower().strip()
    if response in ('', 'y', 'yes'):
        if not install_dependencies():
            return False
    
    if not test_imports():
        print("\n💡 如果模块导入失败，请尝试:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n🎯 准备启动系统...")
    time.sleep(1)
    
    # 启动服务器
    return start_server()

if __name__ == '__main__':
    try:
        success = main()
        if not success:
            print("\n❌ 启动失败，请检查错误信息")
            input("按Enter键退出...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        input("按Enter键退出...")
        sys.exit(1)