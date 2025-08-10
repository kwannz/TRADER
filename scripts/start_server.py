#!/usr/bin/env python3
"""
本地服务器启动脚本

启动完整的AI量化交易系统，包括：
- FastAPI Web服务器
- WebSocket实时数据服务
- 后台数据同步任务
- 系统监控服务
"""

import os
import sys
import asyncio
import signal
import subprocess
from pathlib import Path
from typing import Optional, List
import json
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ServerManager:
    """服务器管理器"""
    
    def __init__(self):
        self.project_root = project_root
        self.processes: List[subprocess.Popen] = []
        self.running = False
        
        # 确定虚拟环境路径
        self.venv_path = self.project_root / 'venv'
        if os.name == 'nt':  # Windows
            self.python_path = self.venv_path / 'Scripts' / 'python.exe'
            self.pip_path = self.venv_path / 'Scripts' / 'pip.exe'
        else:  # Unix-like
            self.python_path = self.venv_path / 'bin' / 'python'
            self.pip_path = self.venv_path / 'bin' / 'pip'
    
    def check_environment(self) -> bool:
        """检查运行环境"""
        print("🔍 检查运行环境...")
        
        # 检查虚拟环境
        if not self.venv_path.exists():
            print("❌ 未找到虚拟环境，请先运行 python scripts/setup_local.py")
            return False
        
        if not self.python_path.exists():
            print(f"❌ 未找到Python解释器: {self.python_path}")
            return False
        
        # 检查配置文件
        config_files = [
            self.project_root / '.env',
            self.project_root / 'config' / 'local.json'
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                print(f"⚠️  配置文件不存在: {config_file}")
                print("   请先运行安装脚本或手动创建配置文件")
        
        print("✅ 环境检查完成")
        return True
    
    def check_dependencies(self) -> bool:
        """检查Python依赖"""
        print("📦 检查Python依赖...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'pymongo', 'redis',
            'rich', 'textual', 'asyncio', 'httpx'
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
                print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
                print("   请运行: python scripts/setup_local.py")
                return False
            
            print("✅ 所有依赖包已安装")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 检查依赖失败: {e}")
            return False
    
    def check_services(self) -> bool:
        """检查外部服务状态"""
        print("🔗 检查外部服务状态...")
        
        # 检查MongoDB
        try:
            import pymongo
            client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            client.close()
            print("✅ MongoDB连接正常")
        except Exception as e:
            print(f"❌ MongoDB连接失败: {e}")
            print("   请启动MongoDB服务")
            return False
        
        # 检查Redis
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            client.close()
            print("✅ Redis连接正常")
        except Exception as e:
            print(f"❌ Redis连接失败: {e}")
            print("   请启动Redis服务")
            return False
        
        return True
    
    def load_config(self) -> dict:
        """加载配置"""
        config = {}
        
        # 加载.env文件
        env_file = self.project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        
        # 加载JSON配置
        json_config_file = self.project_root / 'config' / 'local.json'
        if json_config_file.exists():
            with open(json_config_file, 'r', encoding='utf-8') as f:
                json_config = json.load(f)
                config['json_config'] = json_config
        
        return config
    
    def start_fastapi_server(self, config: dict) -> subprocess.Popen:
        """启动FastAPI服务器"""
        print("🚀 启动FastAPI服务器...")
        
        # 构建启动命令
        host = config.get('API_HOST', '0.0.0.0')
        port = int(config.get('API_PORT', 8000))
        debug = config.get('DEBUG', 'true').lower() == 'true'
        
        cmd = [
            str(self.python_path), '-m', 'uvicorn',
            'fastapi_layer.main:app',
            '--host', host,
            '--port', str(port),
            '--log-level', 'info',
            '--access-log'
        ]
        
        if debug:
            cmd.extend(['--reload', '--reload-dir', str(self.project_root)])
        
        # 设置环境变量
        env = os.environ.copy()
        env.update(config)
        env['PYTHONPATH'] = str(self.project_root)
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ FastAPI服务器已启动 (PID: {process.pid})")
            print(f"   📍 服务地址: http://{host}:{port}")
            print(f"   📚 API文档: http://{host}:{port}/docs")
            print(f"   💚 健康检查: http://{host}:{port}/health")
            
            return process
            
        except Exception as e:
            print(f"❌ 启动FastAPI服务器失败: {e}")
            raise
    
    def start_background_services(self, config: dict) -> List[subprocess.Popen]:
        """启动后台服务"""
        processes = []
        
        # 如果有其他后台服务需要启动，可以在这里添加
        # 例如：数据采集服务、策略执行服务等
        
        return processes
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"\n📡 收到信号 {signum}，正在关闭服务...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """关闭所有服务"""
        print("🔴 正在关闭服务...")
        self.running = False
        
        for i, process in enumerate(self.processes):
            if process.poll() is None:  # 进程仍在运行
                print(f"   停止进程 {i+1} (PID: {process.pid})")
                try:
                    process.terminate()
                    # 等待进程优雅关闭
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"   强制终止进程 {process.pid}")
                    process.kill()
        
        print("✅ 所有服务已关闭")
    
    def monitor_processes(self):
        """监控进程状态"""
        print("👁️  开始监控服务状态...")
        
        while self.running:
            time.sleep(5)
            
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    print(f"⚠️  进程 {i+1} 已意外退出 (返回码: {process.returncode})")
                    
                    # 读取进程输出
                    if process.stdout:
                        output = process.stdout.read()
                        if output:
                            print(f"   进程输出: {output}")
                    
                    # 这里可以实现进程重启逻辑
                    print("   进程监控检测到异常退出")
    
    def show_startup_banner(self, config: dict):
        """显示启动横幅"""
        host = config.get('API_HOST', '0.0.0.0')
        port = config.get('API_PORT', '8000')
        
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                   🤖 AI量化交易系统                           ║
║                    本地服务器启动完成                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🌐 Web服务:    http://{host}:{port}                   ║
║  📚 API文档:    http://{host}:{port}/docs             ║
║  🔍 系统状态:   http://{host}:{port}/health           ║
║  📊 系统指标:   http://{host}:{port}/metrics          ║
║                                                              ║
║  🚀 技术栈:     Python 3.13 + FastAPI + Rust              ║
║  🧠 AI引擎:     DeepSeek + Gemini Pro                      ║
║  💾 数据库:     MongoDB 8.0 + Redis 8.0                   ║
║                                                              ║
║  💡 使用提示:                                                ║
║     • 按 Ctrl+C 优雅关闭服务                                ║
║     • 日志实时显示在控制台                                   ║
║     • 配置文件: .env, config/local.json                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def start(self, debug: bool = False, port: Optional[int] = None):
        """启动服务器"""
        try:
            # 环境检查
            if not self.check_environment():
                return False
            
            if not self.check_dependencies():
                return False
            
            if not self.check_services():
                return False
            
            # 加载配置
            config = self.load_config()
            if port:
                config['API_PORT'] = str(port)
            if debug:
                config['DEBUG'] = 'true'
            
            # 设置信号处理器
            self.setup_signal_handlers()
            
            # 启动服务
            self.running = True
            
            # 启动FastAPI服务器
            fastapi_process = self.start_fastapi_server(config)
            self.processes.append(fastapi_process)
            
            # 启动后台服务
            bg_processes = self.start_background_services(config)
            self.processes.extend(bg_processes)
            
            # 等待服务启动
            time.sleep(2)
            
            # 显示启动信息
            self.show_startup_banner(config)
            
            # 监控进程状态
            self.monitor_processes()
            
        except KeyboardInterrupt:
            print("\n⌨️  用户中断")
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
        finally:
            self.shutdown()
        
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI量化交易系统本地服务器')
    parser.add_argument('--port', '-p', type=int, help='指定端口号 (默认: 8000)')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式')
    parser.add_argument('--no-check', action='store_true', help='跳过环境检查')
    
    args = parser.parse_args()
    
    print("🚀 AI量化交易系统 - 本地服务器启动程序")
    print("="*60)
    
    server_manager = ServerManager()
    success = server_manager.start(debug=args.debug, port=args.port)
    
    if not success:
        print("\n❌ 服务器启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main()