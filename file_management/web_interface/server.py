#!/usr/bin/env python3
"""
AI量化交易系统 - 前端服务器
简单的HTTP服务器，支持静态文件和API模拟
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import threading
import os
import socket

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemHandler(SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        
        # API路由处理
        if parsed_path.path.startswith('/api/'):
            self.handle_api_request(parsed_path)
        else:
            # 静态文件处理
            super().do_GET()
    
    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path.startswith('/api/'):
            self.handle_api_request(parsed_path)
        else:
            self.send_error(404)
    
    def handle_api_request(self, parsed_path):
        """处理API请求"""
        try:
            # 设置响应头
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # 路由分发
            if parsed_path.path == '/api/v1/market/latest':
                response = self.get_market_data()
            elif parsed_path.path == '/api/v1/strategies':
                response = self.get_strategies()
            elif parsed_path.path == '/api/v1/ai/sentiment':
                response = self.get_ai_sentiment()
            elif parsed_path.path == '/api/v1/trades/history':
                response = self.get_trade_history()
            elif parsed_path.path == '/api/v1/health':
                response = self.get_health_check()
            else:
                response = {"success": False, "error": "API endpoint not found"}
            
            # 发送响应
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            logger.error(f"API请求处理错误: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def get_market_data(self):
        """获取市场数据"""
        return {
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
    
    def get_strategies(self):
        """获取策略列表"""
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
        
        return {
            "success": True,
            "data": {
                "strategies": strategies,
                "total_count": len(strategies)
            }
        }
    
    def get_ai_sentiment(self):
        """获取AI情绪分析"""
        return {
            "success": True,
            "data": {
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
            },
            "timestamp": int(time.time() * 1000)
        }
    
    def get_trade_history(self):
        """获取交易历史"""
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
        
        return {
            "success": True,
            "data": {
                "trades": trades,
                "total_pnl": sum(trade["pnl"] for trade in trades),
                "total_trades": len(trades)
            }
        }
    
    def get_health_check(self):
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": int(time.time() * 1000),
            "services": {
                "web_server": "running",
                "data_simulation": "running"
            },
            "version": "2.0.0"
        }
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"{self.address_string()} - {format % args}")

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程HTTP服务器"""
    allow_reuse_address = True
    daemon_threads = True

def find_free_port(start_port=8000, max_attempts=10):
    """寻找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"无法找到可用端口 (尝试范围: {start_port}-{start_port + max_attempts})")

def main():
    """主函数"""
    # 切换到脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 寻找可用端口
    try:
        port = find_free_port(8000)
    except RuntimeError as e:
        logger.error(e)
        return
    
    # 创建服务器
    server = ThreadedHTTPServer(('', port), TradingSystemHandler)
    
    # 打印启动信息
    print("\n" + "="*60)
    print("🚀 AI量化交易系统前端服务器")
    print("="*60)
    print(f"📡 服务地址: http://localhost:{port}")
    print(f"🌐 请在浏览器中打开上述地址访问系统")
    print(f"📊 模拟API数据已启动")
    print(f"🔧 静态文件服务已启用")
    print("="*60)
    print("按 Ctrl+C 停止服务器\n")
    
    try:
        # 启动服务器
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭服务器...")
        server.shutdown()
        server.server_close()
        print("✅ 服务器已停止")

if __name__ == '__main__':
    main()