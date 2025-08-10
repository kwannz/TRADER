#!/usr/bin/env python3
"""
AI量化交易系统 - 实时数据服务器 
支持WebSocket实时数据流、REST API、静态文件服务、开发模式
"""

import asyncio
import json
import logging
import os
import sys
import time
import aiohttp
import websockets
from aiohttp import web, WSMsgType
import ccxt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimeDataManager:
    """实时数据管理器"""
    
    def __init__(self):
        self.exchanges = {}
        self.websocket_clients = set()
        self.market_data = {}
        self.running = False
        
    async def initialize_exchanges(self):
        """初始化交易所连接"""
        try:
            # 初始化OKX (使用公共API，无需密钥)
            self.exchanges['okx'] = ccxt.okx({
                'sandbox': False,  # 使用正式环境的公共API
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
            
            # 初始化Binance (使用公共API)
            self.exchanges['binance'] = ccxt.binance({
                'sandbox': False,
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
            
            logger.info("✅ 交易所API初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 交易所初始化失败: {e}")
            return False
    
    async def get_market_data(self, symbol: str = "BTC/USDT") -> Dict:
        """获取实时市场数据 - 仅真实数据"""
        # 先尝试OKX API
        try:
            okx_ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.exchanges['okx'].fetch_ticker, symbol
            )
            
            market_data = {
                'symbol': symbol,
                'price': float(okx_ticker['last']),
                'volume_24h': float(okx_ticker['baseVolume']),
                'change_24h': float(okx_ticker['change']),
                'change_24h_pct': float(okx_ticker['percentage']),
                'high_24h': float(okx_ticker['high']),
                'low_24h': float(okx_ticker['low']),
                'bid': float(okx_ticker['bid']) if okx_ticker['bid'] else 0,
                'ask': float(okx_ticker['ask']) if okx_ticker['ask'] else 0,
                'timestamp': int(time.time() * 1000),
                'exchange': 'okx',
                'data_source': 'real'
            }
            
            self.market_data[symbol] = market_data
            return market_data
            
        except Exception as okx_error:
            logger.warning(f"OKX API失败 {symbol}: {okx_error}")
            
            # 尝试Binance API作为备用
            try:
                binance_ticker = await asyncio.get_event_loop().run_in_executor(
                    None, self.exchanges['binance'].fetch_ticker, symbol
                )
                
                market_data = {
                    'symbol': symbol,
                    'price': float(binance_ticker['last']),
                    'volume_24h': float(binance_ticker['baseVolume']),
                    'change_24h': float(binance_ticker['change']),
                    'change_24h_pct': float(binance_ticker['percentage']),
                    'high_24h': float(binance_ticker['high']),
                    'low_24h': float(binance_ticker['low']),
                    'bid': float(binance_ticker['bid']) if binance_ticker['bid'] else 0,
                    'ask': float(binance_ticker['ask']) if binance_ticker['ask'] else 0,
                    'timestamp': int(time.time() * 1000),
                    'exchange': 'binance',
                    'data_source': 'real'
                }
                
                self.market_data[symbol] = market_data
                return market_data
                
            except Exception as binance_error:
                logger.error(f"Binance API也失败 {symbol}: {binance_error}")
                # 不返回模拟数据，而是抛出异常
                raise Exception(f"无法从任何交易所获取 {symbol} 的数据：OKX({okx_error}), Binance({binance_error})")
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[Dict]:
        """获取历史K线数据 - 仅真实数据"""
        # 先尝试OKX
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, self.exchanges['okx'].fetch_ohlcv, symbol, timeframe, None, limit
            )
            
            candles = []
            for candle in ohlcv:
                candles.append({
                    'timestamp': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'exchange': 'okx',
                    'data_source': 'real'
                })
            
            return candles
            
        except Exception as okx_error:
            logger.warning(f"OKX历史数据失败 {symbol}: {okx_error}")
            
            # 尝试Binance
            try:
                ohlcv = await asyncio.get_event_loop().run_in_executor(
                    None, self.exchanges['binance'].fetch_ohlcv, symbol, timeframe, None, limit
                )
                
                candles = []
                for candle in ohlcv:
                    candles.append({
                        'timestamp': int(candle[0]),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'exchange': 'binance', 
                        'data_source': 'real'
                    })
                
                return candles
                
            except Exception as binance_error:
                logger.error(f"Binance历史数据也失败 {symbol}: {binance_error}")
                raise Exception(f"无法获取 {symbol} 历史数据：OKX({okx_error}), Binance({binance_error})")
    
    async def start_data_stream(self):
        """启动数据流"""
        self.running = True
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
        
        logger.info("🚀 启动实时数据流...")
        
        while self.running:
            try:
                # 获取所有币种的实时数据
                tasks = [self.get_market_data(symbol) for symbol in symbols]
                market_updates = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 发送数据到所有WebSocket客户端
                if self.websocket_clients:
                    for i, update in enumerate(market_updates):
                        if isinstance(update, dict):  # 只发送成功获取的数据
                            message = {
                                'type': 'market_update',
                                'data': update
                            }
                            
                            # 发送给所有连接的客户端
                            disconnected_clients = set()
                            for ws in self.websocket_clients.copy():
                                try:
                                    await ws.send_str(json.dumps(message))
                                except Exception:
                                    disconnected_clients.add(ws)
                            
                            # 移除断开的连接
                            self.websocket_clients -= disconnected_clients
                        
                        elif isinstance(update, Exception):
                            # API失败时发送错误信息给客户端
                            error_symbol = symbols[i]
                            error_message = {
                                'type': 'data_error',
                                'symbol': error_symbol,
                                'message': f'{error_symbol} 数据获取失败，正在重试...',
                                'timestamp': int(time.time() * 1000)
                            }
                            
                            disconnected_clients = set()
                            for ws in self.websocket_clients.copy():
                                try:
                                    await ws.send_str(json.dumps(error_message))
                                except Exception:
                                    disconnected_clients.add(ws)
                            
                            self.websocket_clients -= disconnected_clients
                
                # 4Hz刷新率 (每250ms更新一次)
                await asyncio.sleep(0.25)
                
            except Exception as e:
                logger.error(f"数据流异常: {e}")
                await asyncio.sleep(1)
    
    def stop_data_stream(self):
        """停止数据流"""
        self.running = False
        logger.info("⏹️ 数据流已停止")

# 全局数据管理器实例
data_manager = RealTimeDataManager()

async def websocket_handler(request):
    """WebSocket处理器"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # 添加到客户端集合
    data_manager.websocket_clients.add(ws)
    logger.info(f"📱 WebSocket客户端连接，总数: {len(data_manager.websocket_clients)}")
    
    try:
        # 发送初始数据
        initial_data = {
            'type': 'connection_success',
            'message': '实时数据连接成功',
            'timestamp': int(time.time() * 1000)
        }
        await ws.send_str(json.dumps(initial_data))
        
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data.get('type') == 'subscribe':
                        # 处理订阅请求
                        symbols = data.get('symbols', ['BTC/USDT'])
                        logger.info(f"客户端订阅: {symbols}")
                        
                        # 立即发送当前数据
                        for symbol in symbols:
                            current_data = await data_manager.get_market_data(symbol)
                            response = {
                                'type': 'market_update',
                                'data': current_data
                            }
                            await ws.send_str(json.dumps(response))
                        
                except json.JSONDecodeError:
                    await ws.send_str(json.dumps({
                        'type': 'error',
                        'message': '无效的JSON格式'
                    }))
            
            elif msg.type == WSMsgType.ERROR:
                logger.error(f'WebSocket错误: {ws.exception()}')
                break
    
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
    
    finally:
        # 移除客户端
        data_manager.websocket_clients.discard(ws)
        logger.info(f"📱 WebSocket客户端断开，剩余: {len(data_manager.websocket_clients)}")
    
    return ws

async def api_market_data(request):
    """API: 获取市场数据"""
    try:
        symbol = request.query.get('symbol', 'BTC/USDT')
        data = await data_manager.get_market_data(symbol)
        
        return web.json_response({
            'success': True,
            'data': data
        })
    except Exception as e:
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_ai_analysis(request):
    """API: AI分析 - 需要真实AI API"""
    try:
        symbol = request.query.get('symbol', 'BTC')
        
        # 这里应该调用真实的AI API (DeepSeek, Gemini等)
        # 暂时返回错误，提醒用户配置真实AI API
        return web.json_response({
            'success': False,
            'error': 'AI分析功能需要配置真实的AI API密钥 (DeepSeek/Gemini)',
            'message': '请在环境变量中设置 DEEPSEEK_API_KEY 或 GEMINI_API_KEY'
        }, status=501)  # 501 Not Implemented
        
    except Exception as e:
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def api_dev_status(request):
    """API: 开发模式状态"""
    try:
        return web.json_response({
            'success': True,
            'mode': 'development',
            'status': 'running',
            'server': 'aiohttp',
            'connected_ws_clients': len(data_manager.websocket_clients),
            'market_data_count': len(data_manager.market_data),
            'exchanges_active': len(data_manager.exchanges),
            'timestamp': int(time.time() * 1000)
        })
    except Exception as e:
        return web.json_response({
            'success': False,
            'error': str(e)
        }, status=500)

async def create_app(dev_mode=False):
    """创建应用"""
    app = web.Application()
    
    # 简单的CORS处理中间件
    @web.middleware
    async def cors_handler(request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        # 开发模式下禁用缓存
        if dev_mode:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        
        return response
    
    app.middlewares.append(cors_handler)
    
    # WebSocket路由
    app.router.add_get('/ws', websocket_handler)
    
    # API路由
    app.router.add_get('/api/market', api_market_data)
    app.router.add_get('/api/ai/analysis', api_ai_analysis)
    
    # 开发模式API
    if dev_mode:
        app.router.add_get('/api/dev/status', api_dev_status)
    
    # 静态文件服务 - 优先从web_interface目录提供服务
    web_interface_path = Path(__file__).parent / 'file_management' / 'web_interface'
    if web_interface_path.exists():
        app.router.add_static('/', path=str(web_interface_path), name='static')
        logger.info(f"📁 静态文件服务: {web_interface_path}")
    else:
        app.router.add_static('/', path='.', name='static')
        logger.info("📁 静态文件服务: 当前目录")
    
    return app

async def main(dev_mode=False):
    """主函数"""
    mode_text = "开发模式" if dev_mode else "生产模式"
    logger.info(f"🚀 启动AI量化交易系统 ({mode_text})...")
    
    # 初始化数据管理器
    if not await data_manager.initialize_exchanges():
        logger.error("❌ 交易所API初始化失败，系统无法启动")
        print("❌ 无法连接到交易所API，系统将无法获取真实数据")
        print("请检查网络连接或稍后重试")
        return
    
    # 启动数据流
    asyncio.create_task(data_manager.start_data_stream())
    
    # 创建应用
    app = await create_app(dev_mode=dev_mode)
    
    # 启动服务器
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, 'localhost', 8888)
    await site.start()
    
    logger.info("✅ 服务器启动成功!")
    logger.info("📊 前端界面: http://localhost:8888")
    logger.info("🔌 WebSocket: ws://localhost:8888/ws")
    logger.info("🔗 API测试: http://localhost:8888/api/market")
    
    if dev_mode:
        logger.info("🔧 开发模式API: http://localhost:8888/api/dev/status")
        logger.info("🔄 缓存已禁用，文件将实时更新")
    
    logger.info("🚨 纯真实数据模式: 仅使用OKX/Binance真实API数据")
    logger.info("按 Ctrl+C 停止服务器")
    
    try:
        # 保持运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 正在关闭服务器...")
        data_manager.stop_data_stream()
        await runner.cleanup()

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        ('aiohttp', 'aiohttp'),
        ('aiohttp_cors', 'aiohttp-cors'), 
        ('ccxt', 'ccxt'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('websockets', 'websockets')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == '__main__':
    # 检查命令行参数
    dev_mode = '--dev' in sys.argv or '-d' in sys.argv
    
    # 检查依赖 (暂时跳过以进行测试)
    # if not check_dependencies():
    #     sys.exit(1)
    
    # 检查必要文件 (开发模式下更宽松)
    web_interface_path = Path(__file__).parent / 'file_management' / 'web_interface'
    
    if web_interface_path.exists():
        required_files = [
            str(web_interface_path / 'index.html'),
            str(web_interface_path / 'styles.css'), 
            str(web_interface_path / 'app.js')
        ]
    else:
        required_files = ['index.html', 'styles.css', 'app.js']
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files and not dev_mode:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        print("💡 使用 --dev 参数启动开发模式可忽略某些文件检查")
        sys.exit(1)
    elif missing_files and dev_mode:
        print(f"⚠️ 开发模式：缺少文件 {', '.join(missing_files)} 但继续启动")
    else:
        print("✅ 所有文件检查完成")
    
    if dev_mode:
        print("🔧 开发模式已启用")
    
    # 启动异步服务器
    asyncio.run(main(dev_mode=dev_mode))