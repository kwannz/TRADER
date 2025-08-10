#!/usr/bin/env python3
"""
AI量化交易系统 - 一键启动开发环境
支持热重载的开发服务器启动脚本
"""

import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path

class DevEnvironmentStarter:
    """开发环境启动器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        
    def check_python_version(self):
        """检查Python版本"""
        version = sys.version_info
        if version < (3, 8):
            print(f"❌ Python版本过低: {version.major}.{version.minor}")
            print("需要Python 3.8或更高版本")
            return False
        
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self):
        """检查并安装依赖包"""
        required_packages = [
            'aiohttp>=3.9.0',
            'watchdog>=3.0.0',
            'ccxt>=4.1.0',
            'pandas>=2.1.0',
            'numpy>=1.25.0',
            'websockets>=12.0'
        ]
        
        print("🔍 检查依赖包...")
        
        missing_packages = []
        for package in required_packages:
            package_name = package.split('>=')[0]
            try:
                __import__(package_name)
                print(f"  ✅ {package_name}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package_name} (缺失)")
        
        if missing_packages:
            print(f"\n📦 需要安装依赖包: {len(missing_packages)} 个")
            
            # 询问是否自动安装
            response = input("是否自动安装缺失的依赖包? (y/N): ").lower().strip()
            if response in ['y', 'yes']:
                return self.install_dependencies(missing_packages)
            else:
                print("请手动安装依赖包:")
                print(f"pip install {' '.join(missing_packages)}")
                return False
        
        print("✅ 所有依赖包已安装")
        return True
    
    def install_dependencies(self, packages):
        """安装依赖包"""
        print("📦 正在安装依赖包...")
        
        try:
            cmd = [self.python_executable, '-m', 'pip', 'install'] + packages
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 依赖包安装成功!")
                return True
            else:
                print(f"❌ 依赖包安装失败: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖包安装失败: {e}")
            return False
        except Exception as e:
            print(f"❌ 安装过程出错: {e}")
            return False
    
    def check_project_structure(self):
        """检查项目结构"""
        print("🏗️ 检查项目结构...")
        
        required_files = [
            'dev_server.py',
            'server.py',
            'dev_client.js',
            'file_management/web_interface/index.html',
            'file_management/web_interface/app.js',
            'file_management/web_interface/styles.css'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"  ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"  ⚠️ {file_path} (可选)")
        
        if missing_files:
            print(f"⚠️ 注意: {len(missing_files)} 个文件缺失，但不影响基本功能")
        
        return True
    
    def start_dev_server(self, mode='hot'):
        """启动开发服务器"""
        print(f"\n🚀 启动开发环境 ({mode} 模式)...")
        
        try:
            if mode == 'hot':
                # 启动热重载开发服务器
                script_path = self.project_root / 'dev_server.py'
                cmd = [self.python_executable, str(script_path)]
            else:
                # 启动增强版生产服务器
                script_path = self.project_root / 'server.py'
                cmd = [self.python_executable, str(script_path), '--dev']
            
            print(f"📜 执行命令: {' '.join(cmd)}")
            
            # 启动服务器进程
            subprocess.run(cmd, cwd=str(self.project_root))
            
        except KeyboardInterrupt:
            print("\n🛑 开发服务器已停止")
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
        
        return True
    
    def show_usage_info(self):
        """显示使用说明"""
        print("\n" + "="*60)
        print("🔧 AI量化交易系统 - 开发环境")
        print("="*60)
        print("\n📖 使用说明:")
        print("  • 修改 .py 文件将自动重启后端服务器")
        print("  • 修改 .html/.css/.js 文件将自动刷新浏览器")
        print("  • 在浏览器开发者工具中查看热重载日志")
        print("  • 按 Ctrl+C 停止开发服务器")
        print("\n🌐 访问地址:")
        print("  • 前端界面: http://localhost:8000")
        print("  • API文档: http://localhost:8000/api/dev/status")
        print("\n🛠️ 开发工具:")
        print("  • WebSocket连接状态会显示在浏览器控制台")
        print("  • 页面左下角会显示'开发模式'指示器")
        print("  • 代码更新通知会在页面右上角显示")
        print("\n" + "="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI量化交易系统开发环境启动器')
    parser.add_argument('--mode', choices=['hot', 'enhanced'], default='hot',
                        help='启动模式: hot=热重载服务器, enhanced=增强版生产服务器')
    parser.add_argument('--skip-deps', action='store_true',
                        help='跳过依赖检查')
    parser.add_argument('--no-install', action='store_true',
                        help='不自动安装依赖包')
    
    args = parser.parse_args()
    
    starter = DevEnvironmentStarter()
    
    print("🚀 AI量化交易系统 - 开发环境启动器")
    print(f"📂 项目路径: {starter.project_root}")
    print(f"🐍 Python: {starter.python_executable}")
    print(f"💻 系统: {platform.system()} {platform.release()}")
    print("-" * 50)
    
    # 检查Python版本
    if not starter.check_python_version():
        sys.exit(1)
    
    # 检查依赖包
    if not args.skip_deps:
        if not starter.check_dependencies():
            print("\n💡 如果要跳过依赖检查，请使用 --skip-deps 参数")
            sys.exit(1)
    else:
        print("⚠️ 已跳过依赖检查")
    
    # 检查项目结构
    starter.check_project_structure()
    
    # 显示使用说明
    starter.show_usage_info()
    
    # 启动开发服务器
    input("\n按 Enter 键启动开发服务器...")
    starter.start_dev_server(args.mode)

if __name__ == '__main__':
    main()