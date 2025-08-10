#!/usr/bin/env python3
"""
AI量化交易系统 - 热重载开发服务器
类似Replit的开发环境，支持代码修改时自动重载
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
import subprocess
import signal
from pathlib import Path
from typing import Set, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import aiohttp
from aiohttp import web, WSMsgType
import webbrowser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

class HotReloadEventHandler(FileSystemEventHandler):
    """文件变化监控处理器"""
    
    def __init__(self, dev_server_instance):
        self.dev_server = dev_server_instance
        self.last_reload_time = 0
        self.reload_cooldown = 1  # 1秒冷却时间防止频繁重载
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        file_ext = Path(file_path).suffix.lower()
        
        # 监控这些文件类型的变化
        watch_extensions = {'.py', '.html', '.css', '.js', '.json'}
        
        if file_ext in watch_extensions:
            current_time = time.time()
            if current_time - self.last_reload_time > self.reload_cooldown:
                self.last_reload_time = current_time
                logger.info(f"🔄 文件已修改: {file_path}")
                
                if file_ext == '.py':
                    # Python文件修改 - 重启服务器
                    asyncio.create_task(self.dev_server.restart_backend())
                else:
                    # 前端文件修改 - 通知浏览器刷新
                    asyncio.create_task(self.dev_server.notify_frontend_reload())

class DevServer:
    """开发服务器"""
    
    def __init__(self):
        self.app = None
        self.runner = None
        self.site = None
        self.observer = None
        self.backend_process = None
        self.websocket_clients: Set = set()
        self.port = 8000
        self.host = 'localhost'
        
    async def create_app(self):
        """创建开发应用"""
        app = web.Application()
        
        # CORS中间件
        @web.middleware
        async def cors_handler(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        app.middlewares.append(cors_handler)
        
        # WebSocket路由 - 用于热重载通知
        app.router.add_get('/dev-ws', self.websocket_handler)
        
        # API路由
        app.router.add_get('/api/dev/status', self.dev_status_handler)
        app.router.add_post('/api/dev/restart', self.restart_handler)
        
        # 静态文件服务
        web_interface_path = Path(__file__).parent / 'file_management' / 'web_interface'
        if web_interface_path.exists():
            app.router.add_static('/', path=str(web_interface_path), name='static')
        else:
            # 如果没有专门的web_interface目录，使用当前目录
            app.router.add_static('/', path=str(Path(__file__).parent), name='static')
        
        return app
    
    async def websocket_handler(self, request):
        """开发WebSocket处理器"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_clients.add(ws)
        logger.info(f"🔗 开发WebSocket连接，总数: {len(self.websocket_clients)}")
        
        try:
            await ws.send_str(json.dumps({
                'type': 'dev_connected',
                'message': '开发模式已连接',
                'timestamp': int(time.time() * 1000)
            }))
            
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get('type') == 'ping':
                            await ws.send_str(json.dumps({'type': 'pong'}))
                    except json.JSONDecodeError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'开发WebSocket错误: {ws.exception()}')
                    break
        
        except Exception as e:
            logger.error(f"开发WebSocket处理异常: {e}")
        
        finally:
            self.websocket_clients.discard(ws)
            logger.info(f"🔗 开发WebSocket断开，剩余: {len(self.websocket_clients)}")
        
        return ws
    
    async def dev_status_handler(self, request):
        """开发状态API"""
        return web.json_response({
            'success': True,
            'status': 'running',
            'mode': 'development',
            'connected_clients': len(self.websocket_clients),
            'watching_files': True
        })
    
    async def restart_handler(self, request):
        """手动重启API"""
        await self.restart_backend()
        return web.json_response({
            'success': True,
            'message': '后端服务器重启中...'
        })
    
    async def notify_frontend_reload(self):
        """通知前端重载"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': 'reload_frontend',
            'message': '前端文件已更新，正在刷新页面...',
            'timestamp': int(time.time() * 1000)
        }
        
        # 发送给所有连接的客户端
        disconnected_clients = set()
        for ws in self.websocket_clients.copy():
            try:
                await ws.send_str(json.dumps(message))
            except Exception as e:
                disconnected_clients.add(ws)
        
        # 移除断开的连接
        self.websocket_clients -= disconnected_clients
        logger.info("🔄 已通知前端刷新")
    
    async def restart_backend(self):
        """重启后端服务"""
        if not self.websocket_clients:
            return
            
        message = {
            'type': 'backend_restarting',
            'message': 'Python代码已更新，后端正在重启...',
            'timestamp': int(time.time() * 1000)
        }
        
        # 通知客户端后端重启
        disconnected_clients = set()
        for ws in self.websocket_clients.copy():
            try:
                await ws.send_str(json.dumps(message))
            except Exception as e:
                disconnected_clients.add(ws)
        
        self.websocket_clients -= disconnected_clients
        logger.info("🔄 Python代码更新，准备重启后端...")
        
        # 延迟2秒后发送重启完成通知
        await asyncio.sleep(2)
        
        restart_message = {
            'type': 'backend_restarted', 
            'message': '后端重启完成',
            'timestamp': int(time.time() * 1000)
        }
        
        for ws in self.websocket_clients.copy():
            try:
                await ws.send_str(json.dumps(restart_message))
            except Exception:
                pass
    
    def start_file_watcher(self):
        """启动文件监控"""
        event_handler = HotReloadEventHandler(self)
        self.observer = Observer()
        
        # 监控当前目录及子目录
        watch_paths = [
            str(Path(__file__).parent),  # 主目录
            str(Path(__file__).parent / 'file_management'),  # Web界面
            str(Path(__file__).parent / 'core'),  # 核心代码
            str(Path(__file__).parent / 'src'),   # 源代码
        ]
        
        for path in watch_paths:
            if Path(path).exists():
                self.observer.schedule(event_handler, path, recursive=True)
                logger.info(f"👀 监控目录: {path}")
        
        self.observer.start()
        logger.info("🔍 文件监控已启动")
    
    def stop_file_watcher(self):
        """停止文件监控"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("🔍 文件监控已停止")
    
    async def start(self):
        """启动开发服务器"""
        logger.info("🚀 启动热重载开发服务器...")
        
        # 创建应用
        self.app = await self.create_app()
        
        # 启动服务器
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        # 启动文件监控
        self.start_file_watcher()
        
        # 打开浏览器
        url = f"http://{self.host}:{self.port}"
        
        logger.info("✅ 开发服务器启动成功!")
        logger.info(f"🌐 前端界面: {url}")
        logger.info(f"🔗 开发WebSocket: ws://{self.host}:{self.port}/dev-ws")
        logger.info(f"📊 开发API: {url}/api/dev/status")
        logger.info("🔥 热重载模式已激活")
        logger.info("   - 修改Python文件将重启后端")
        logger.info("   - 修改HTML/CSS/JS文件将刷新浏览器")
        logger.info("按 Ctrl+C 停止开发服务器")
        
        # 延迟打开浏览器，避免服务器还没完全启动
        await asyncio.sleep(1)
        try:
            webbrowser.open(url)
            logger.info(f"🌐 已自动打开浏览器: {url}")
        except Exception:
            logger.info(f"💡 请手动打开浏览器访问: {url}")
        
        try:
            # 保持运行
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 正在关闭开发服务器...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """清理资源"""
        self.stop_file_watcher()
        
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("✅ 开发服务器已关闭")

def check_dependencies():
    """检查开发环境依赖"""
    required_packages = [
        'aiohttp',
        'watchdog',
        'webbrowser'  # 通常是内置模块
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'webbrowser':
                import webbrowser
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少开发环境依赖: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

async def main():
    """主函数"""
    if not check_dependencies():
        sys.exit(1)
    
    dev_server = DevServer()
    await dev_server.start()

if __name__ == '__main__':
    # 设置信号处理
    def signal_handler(signum, frame):
        logger.info("🛑 收到停止信号")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    asyncio.run(main())