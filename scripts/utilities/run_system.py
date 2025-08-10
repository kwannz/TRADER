#!/usr/bin/env python3
"""
AI量化交易系统 - 一键启动脚本

提供多种启动模式：
1. 完整安装和启动
2. 仅启动服务器
3. 仅启动CLI
4. 运行验证测试
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class SystemLauncher:
    """系统启动器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / 'scripts'
        
        # 确定Python解释器路径
        self.venv_path = self.project_root / 'venv'
        if os.name == 'nt':  # Windows
            self.python_path = self.venv_path / 'Scripts' / 'python.exe'
        else:  # Unix-like
            self.python_path = self.venv_path / 'bin' / 'python'
        
        # 如果没有虚拟环境，使用系统Python
        if not self.python_path.exists():
            self.python_path = sys.executable
    
    def run_script(self, script_name: str, args: list = None) -> int:
        """运行脚本"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return 1
        
        cmd = [str(self.python_path), str(script_path)]
        if args:
            cmd.extend(args)
        
        try:
            print(f"🚀 执行: {' '.join(cmd)}")
            return subprocess.call(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            print("\n⌨️  用户中断")
            return 0
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            return 1
    
    def check_installation(self) -> bool:
        """检查是否已安装"""
        # 检查虚拟环境
        if not self.venv_path.exists():
            return False
        
        # 检查配置文件
        config_files = [
            self.project_root / '.env',
            self.project_root / 'config' / 'local.json'
        ]
        
        return any(config_file.exists() for config_file in config_files)
    
    def show_banner(self):
        """显示横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                   🤖 AI量化交易系统                           ║
║                     一键启动程序                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🚀 技术栈: Python 3.13 + FastAPI + Rust + AI              ║
║  💾 数据库: MongoDB 8.0 + Redis 8.0                        ║
║  🧠 AI引擎: DeepSeek Reasoner + Gemini Pro                 ║
║  🖥️  界面: Bloomberg Terminal风格CLI                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def show_menu(self):
        """显示功能菜单"""
        menu = """
请选择操作模式:

1️⃣  完整安装 - 自动安装所有依赖和服务
2️⃣  启动服务器 - 启动FastAPI Web服务器
3️⃣  启动CLI - 启动Bloomberg风格命令行界面
4️⃣  运行测试 - 验证所有模块功能
5️⃣  演示模式 - 体验CLI界面（无需安装）
6️⃣  状态检查 - 检查系统和服务状态
0️⃣  退出程序

        """
        print(menu)
    
    def handle_full_setup(self):
        """处理完整安装"""
        print("🔧 开始完整系统安装...")
        return self.run_script('setup_local.py')
    
    def handle_start_server(self):
        """处理启动服务器"""
        print("🌐 启动Web服务器...")
        return self.run_script('start_server.py')
    
    def handle_start_cli(self):
        """处理启动CLI"""
        print("🖥️  启动CLI界面...")
        return self.run_script('start_cli.py')
    
    def handle_run_tests(self):
        """处理运行测试"""
        print("🧪 运行系统测试...")
        
        # 检查CLI验证器是否存在
        cli_tester = self.project_root / 'cli_validation' / 'cli_tester.py'
        if cli_tester.exists():
            return subprocess.call([
                str(self.python_path), 
                str(cli_tester),
                '--verbose'
            ], cwd=self.project_root)
        else:
            print("❌ 测试模块不存在，请先运行完整安装")
            return 1
    
    def handle_demo_mode(self):
        """处理演示模式"""
        print("🎬 启动演示模式...")
        return self.run_script('start_cli.py', ['--demo'])
    
    def handle_status_check(self):
        """处理状态检查"""
        print("🔍 检查系统状态...")
        
        status_info = {
            "虚拟环境": "✅" if self.venv_path.exists() else "❌",
            "配置文件": "✅" if (self.project_root / '.env').exists() else "❌",
            "Rust引擎": "✅" if (self.project_root / 'rust_engine' / 'Cargo.toml').exists() else "❌",
            "Python层": "✅" if (self.project_root / 'python_layer').exists() else "❌",
            "FastAPI层": "✅" if (self.project_root / 'fastapi_layer').exists() else "❌",
            "CLI界面": "✅" if (self.project_root / 'cli_interface').exists() else "❌",
        }
        
        print("\n📊 系统组件状态:")
        for component, status in status_info.items():
            print(f"   {status} {component}")
        
        # 检查服务状态
        print("\n🔗 外部服务状态:")
        
        # 检查MongoDB
        try:
            import pymongo
            client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            client.close()
            print("   ✅ MongoDB")
        except:
            print("   ❌ MongoDB")
        
        # 检查Redis
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            client.close()
            print("   ✅ Redis")
        except:
            print("   ❌ Redis")
        
        # 检查FastAPI服务
        try:
            import httpx
            response = httpx.get('http://localhost:8000/health', timeout=2)
            if response.status_code == 200:
                print("   ✅ FastAPI服务")
            else:
                print("   ⚠️  FastAPI服务响应异常")
        except:
            print("   ❌ FastAPI服务")
        
        print("\n💡 如需帮助，请参考:")
        print("   📖 README.md - 项目说明")
        print("   🏗️  FULLSTACK_ARCHITECTURE.md - 系统架构")
        
        return 0
    
    def run_interactive(self):
        """运行交互式菜单"""
        self.show_banner()
        
        # 检查安装状态
        if not self.check_installation():
            print("⚠️  系统尚未安装，建议先选择「完整安装」")
        else:
            print("✅ 系统已安装，可以直接启动服务")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("请输入选择 (0-6): ").strip()
                
                if choice == '0':
                    print("👋 感谢使用AI量化交易系统!")
                    break
                elif choice == '1':
                    self.handle_full_setup()
                elif choice == '2':
                    self.handle_start_server()
                elif choice == '3':
                    self.handle_start_cli()
                elif choice == '4':
                    self.handle_run_tests()
                elif choice == '5':
                    self.handle_demo_mode()
                elif choice == '6':
                    self.handle_status_check()
                else:
                    print("❌ 无效选择，请输入0-6之间的数字")
                
                input("\n按回车键继续...")
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 用户退出")
                break
            except EOFError:
                print("\n\n👋 用户退出")
                break

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI量化交易系统一键启动')
    parser.add_argument('--setup', action='store_true', help='直接运行完整安装')
    parser.add_argument('--server', action='store_true', help='直接启动服务器')
    parser.add_argument('--cli', action='store_true', help='直接启动CLI')
    parser.add_argument('--test', action='store_true', help='直接运行测试')
    parser.add_argument('--demo', action='store_true', help='直接运行演示模式')
    parser.add_argument('--status', action='store_true', help='直接检查状态')
    
    args = parser.parse_args()
    
    launcher = SystemLauncher()
    
    # 如果有直接命令参数，执行对应操作
    if args.setup:
        return launcher.handle_full_setup()
    elif args.server:
        return launcher.handle_start_server()
    elif args.cli:
        return launcher.handle_start_cli()
    elif args.test:
        return launcher.handle_run_tests()
    elif args.demo:
        return launcher.handle_demo_mode()
    elif args.status:
        return launcher.handle_status_check()
    else:
        # 否则运行交互式菜单
        launcher.run_interactive()
        return 0

if __name__ == "__main__":
    sys.exit(main())