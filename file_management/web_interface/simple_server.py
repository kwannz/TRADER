#!/usr/bin/env python3
"""
AI量化交易系统 - 简化前端服务器
提供静态文件服务和WebSocket模拟
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import aiohttp
import aiohttp_cors
from aiohttp import web, WSMsgType
import aiofiles

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWebServer:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.app = None
        self.websockets = set()
        self.market_data = {}
        self.ai_analysis = {}
        
        # 模拟数据生成器
        self.data_generators = {
            'market': self.generate_market_data,
            'ai': self.generate_ai_analysis
        }

    def setup_app(self):
        """设置应用和路由"""
        self.app = web.Application()
        
        # 设置CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # 静态文件路由
        static_path = Path(__file__).parent
        self.app.router.add_static('/', static_path, name='static')
        
        # API路由
        self.app.router.add_get('/api/v1/market/latest', self.api_market_data)
        self.app.router.add_get('/api/v1/strategies', self.api_strategies)
        self.app.router.add_get('/api/v1/ai/sentiment', self.api_ai_sentiment)
        self.app.router.add_get('/api/v1/trades/history', self.api_trade_history)
        self.app.router.add_get('/api/v1/health', self.api_health)
        
        # WebSocket路由
        self.app.router.add_get('/ws/market-data', self.websocket_market_data)
        self.app.router.add_get('/ws/ai-analysis', self.websocket_ai_analysis)
        self.app.router.add_get('/dev-ws', self.websocket_dev)
        
        # 添加CORS到所有路由
        for route in list(self.app.router.routes()):
            cors.add(route)

    async def api_market_data(self, request):
        """获取市场数据API"""
        try:
            data = {
                "success": True,
                "data": {
                    "timestamp": int(time.time() * 1000),
                    "symbols": {
                        "BTC-USDT": {
                            "price": 45000 + random.uniform(-1000, 1000),
                            "change24h": random.uniform(-0.1, 0.1),
                            "volume24h": random.uniform(800000, 1200000)
                        },
                        "ETH-USDT": {
                            "price": 2800 + random.uniform(-200, 200),
                            "change24h": random.uniform(-0.08, 0.08),
                            "volume24h": random.uniform(600000, 900000)
                        },
                        "BNB-USDT": {
                            "price": 300 + random.uniform(-30, 30),
                            "change24h": random.uniform(-0.06, 0.06),
                            "volume24h": random.uniform(200000, 400000)
                        },
                        "SOL-USDT": {
                            "price": 100 + random.uniform(-10, 10),
                            "change24h": random.uniform(-0.12, 0.12),
                            "volume24h": random.uniform(300000, 500000)
                        }
                    }
                }
            }
            return web.json_response(data)
        except Exception as e:
            logger.error(f"API错误: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def api_strategies(self, request):
        """获取策略列表API"""
        strategies = [
            {
                "id": "strategy_001",
                "name": "BTC网格策略",
                "type": "grid",
                "symbol": "BTC-USDT",
                "status": "running",
                "pnl": 247.85,
                "win_rate": 0.687,
                "created_at": "2024-01-27T10:30:00Z"
            },
            {
                "id": "strategy_002", 
                "name": "ETH AI策略",
                "type": "ai",
                "symbol": "ETH-USDT",
                "status": "running",
                "pnl": 156.42,
                "win_rate": 0.723,
                "created_at": "2024-01-26T15:45:00Z"
            },
            {
                "id": "strategy_003",
                "name": "多币种DCA",
                "type": "dca", 
                "symbol": "MULTI",
                "status": "paused",
                "pnl": -23.17,
                "win_rate": 0.542,
                "created_at": "2024-01-25T09:15:00Z"
            }
        ]
        
        return web.json_response({
            "success": True,
            "data": {
                "strategies": strategies,
                "total_count": len(strategies)
            }
        })

    async def api_ai_sentiment(self, request):
        """获取AI情绪分析API"""
        sentiment_data = {
            "sentiment": {
                "score": random.uniform(-1, 1),
                "trend": random.choice(["bullish", "bearish", "neutral"]),
                "confidence": random.uniform(0.6, 0.95)
            },
            "prediction": {
                "direction": random.choice(["up", "down", "sideways"]),
                "timeframe": "24h",
                "confidence": random.uniform(0.5, 0.9),
                "target_price": 45000 + random.uniform(-2000, 2000)
            },
            "news_sentiment": {
                "score": random.uniform(-0.5, 0.8),
                "source_count": random.randint(15, 45)
            },
            "social_sentiment": {
                "score": random.uniform(-0.3, 0.6),
                "mention_count": random.randint(1200, 3500)
            }
        }
        
        return web.json_response({
            "success": True,
            "data": sentiment_data,
            "timestamp": int(time.time() * 1000)
        })

    async def api_trade_history(self, request):
        """获取交易历史API"""
        trades = []
        for i in range(20):
            trades.append({
                "id": f"trade_{i+1:03d}",
                "timestamp": int(time.time() * 1000) - i * 3600000,
                "symbol": random.choice(["BTC-USDT", "ETH-USDT", "BNB-USDT"]),
                "side": random.choice(["buy", "sell"]),
                "amount": round(random.uniform(0.01, 1.0), 6),
                "price": round(random.uniform(40000, 50000), 2),
                "fee": round(random.uniform(0.1, 5.0), 2),
                "pnl": round(random.uniform(-50, 100), 2),
                "strategy": random.choice(["BTC网格策略", "ETH AI策略", "多币种DCA"])
            })
        
        return web.json_response({
            "success": True,
            "data": {
                "trades": trades,
                "total_pnl": sum(trade["pnl"] for trade in trades),
                "total_trades": len(trades)
            }
        })

    async def api_health(self, request):
        """健康检查API"""
        return web.json_response({
            "status": "healthy",
            "timestamp": int(time.time() * 1000),
            "services": {
                "web_server": "running",
                "websocket": "running",
                "data_simulation": "running"
            },
            "version": "2.0.0"
        })

    async def websocket_market_data(self, request):
        """市场数据WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        logger.info("市场数据WebSocket连接已建立")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get('type') == 'subscribe':
                        # 处理订阅请求
                        await ws.send_str(json.dumps({
                            "type": "subscription_confirmed",
                            "channels": data.get('channels', [])
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket错误: {ws.exception()}')
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}")
        finally:
            self.websockets.discard(ws)
            logger.info("市场数据WebSocket连接已关闭")
        
        return ws

    async def websocket_ai_analysis(self, request):
        """AI分析WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        logger.info("AI分析WebSocket连接已建立")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # 处理消息
                    pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket错误: {ws.exception()}')
        except Exception as e:
            logger.error(f"WebSocket处理错误: {e}")
        finally:
            self.websockets.discard(ws)
            logger.info("AI分析WebSocket连接已关闭")
        
        return ws

    async def websocket_dev(self, request):
        """开发模式WebSocket"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        logger.info("开发模式WebSocket连接已建立")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await ws.send_str('{"type":"pong"}')
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'开发WebSocket错误: {ws.exception()}')
        except Exception as e:
            logger.error(f"开发WebSocket处理错误: {e}")
        finally:
            logger.info("开发模式WebSocket连接已关闭")
        
        return ws

    def generate_market_data(self) -> Dict[str, Any]:
        """生成模拟市场数据"""
        symbols = ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT"]
        base_prices = {"BTC-USDT": 45000, "ETH-USDT": 2800, "BNB-USDT": 300, "SOL-USDT": 100}
        
        data = {}
        for symbol in symbols:
            base_price = base_prices[symbol]
            data[symbol] = {
                "price": base_price + random.uniform(-base_price*0.02, base_price*0.02),
                "change24h": random.uniform(-0.1, 0.1),
                "volume24h": random.uniform(500000, 1500000),
                "timestamp": int(time.time() * 1000)
            }
        
        return data

    def generate_ai_analysis(self) -> Dict[str, Any]:
        """生成模拟AI分析数据"""
        return {
            "sentiment": {
                "score": random.uniform(-1, 1),
                "confidence": random.uniform(0.6, 0.95)
            },
            "prediction": {
                "direction": random.choice(["bullish", "bearish", "neutral"]),
                "confidence": random.uniform(0.5, 0.9)
            },
            "timestamp": int(time.time() * 1000)
        }

    async def broadcast_data(self):
        """定期广播数据到WebSocket客户端"""
        while True:
            try:
                if self.websockets:
                    # 生成市场数据
                    market_data = self.generate_market_data()
                    market_message = json.dumps({
                        "type": "market_update",
                        "data": market_data
                    })
                    
                    # 生成AI分析数据
                    ai_data = self.generate_ai_analysis()
                    ai_message = json.dumps({
                        "type": "ai_analysis",
                        "data": ai_data
                    })
                    
                    # 广播给所有连接的客户端
                    disconnected = set()
                    for ws in self.websockets:
                        try:
                            await ws.send_str(market_message)
                            await asyncio.sleep(0.1)  # 小延迟避免消息冲突
                            await ws.send_str(ai_message)
                        except Exception as e:
                            logger.error(f"广播数据失败: {e}")
                            disconnected.add(ws)
                    
                    # 清理断开的连接
                    self.websockets -= disconnected
                
                # 每2秒更新一次
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"数据广播错误: {e}")
                await asyncio.sleep(5)

    async def start_server(self):
        """启动服务器"""
        self.setup_app()
        
        # 启动数据广播任务
        asyncio.create_task(self.broadcast_data())
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"🚀 AI量化交易系统前端服务器启动成功!")
        logger.info(f"📡 服务地址: http://{self.host}:{self.port}")
        logger.info(f"🌐 请在浏览器中打开上述地址访问系统")
        logger.info(f"📊 WebSocket数据模拟已启动")
        logger.info(f"🔧 开发模式热重载已启用")

async def main():
    """主函数"""
    server = SimpleWebServer()
    await server.start_server()
    
    try:
        # 保持服务器运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("服务器正在关闭...")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")