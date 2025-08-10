#!/usr/bin/env python3
"""
AI量化交易系统 - 模拟API服务器
为前端提供模拟的API数据
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random
import time
from urllib.parse import urlparse, parse_qs
import threading

class MockAPIHandler(BaseHTTPRequestHandler):
    """模拟API处理器"""
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        
        # 设置CORS头
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # 路由处理
        if parsed_path.path == '/api/v1/market/latest':
            response = self.get_market_data()
        elif parsed_path.path == '/api/v1/strategies':
            response = self.get_strategies()
        elif parsed_path.path == '/api/v1/ai/sentiment':
            response = self.get_ai_sentiment()
        elif parsed_path.path == '/api/v1/trades/history':
            response = self.get_trade_history()
        elif parsed_path.path == '/api/v1/health':
            response = self.get_health()
        else:
            response = {"success": False, "error": "API endpoint not found"}
        
        # 发送响应
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        """处理OPTIONS请求（CORS预检）"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_market_data(self):
        """模拟市场数据"""
        return {
            "success": True,
            "data": {
                "timestamp": int(time.time() * 1000),
                "symbols": {
                    "BTC-USDT": {
                        "price": round(45000 + random.uniform(-1000, 1000), 2),
                        "change24h": round(random.uniform(-0.1, 0.1), 4),
                        "volume24h": round(random.uniform(800000, 1200000), 2)
                    },
                    "ETH-USDT": {
                        "price": round(2800 + random.uniform(-200, 200), 2),
                        "change24h": round(random.uniform(-0.08, 0.08), 4),
                        "volume24h": round(random.uniform(600000, 900000), 2)
                    },
                    "BNB-USDT": {
                        "price": round(300 + random.uniform(-30, 30), 2),
                        "change24h": round(random.uniform(-0.06, 0.06), 4),
                        "volume24h": round(random.uniform(200000, 400000), 2)
                    }
                }
            }
        }
    
    def get_strategies(self):
        """模拟策略数据"""
        return {
            "success": True,
            "data": {
                "strategies": [
                    {
                        "id": "strategy_001",
                        "name": "BTC网格策略",
                        "type": "grid",
                        "symbol": "BTC-USDT",
                        "status": "running",
                        "pnl": round(random.uniform(100, 500), 2),
                        "win_rate": round(random.uniform(0.6, 0.8), 3)
                    },
                    {
                        "id": "strategy_002",
                        "name": "ETH AI策略",
                        "type": "ai",
                        "symbol": "ETH-USDT",
                        "status": "running", 
                        "pnl": round(random.uniform(50, 300), 2),
                        "win_rate": round(random.uniform(0.65, 0.85), 3)
                    }
                ],
                "total_count": 2
            }
        }
    
    def get_ai_sentiment(self):
        """模拟AI情绪分析"""
        return {
            "success": True,
            "data": {
                "sentiment": {
                    "score": round(random.uniform(-1, 1), 3),
                    "confidence": round(random.uniform(0.6, 0.95), 3)
                },
                "prediction": {
                    "direction": random.choice(["up", "down", "sideways"]),
                    "confidence": round(random.uniform(0.5, 0.9), 3)
                }
            }
        }
    
    def get_trade_history(self):
        """模拟交易历史"""
        trades = []
        for i in range(10):
            trades.append({
                "id": f"trade_{i+1:03d}",
                "timestamp": int(time.time() * 1000) - i * 3600000,
                "symbol": random.choice(["BTC-USDT", "ETH-USDT"]),
                "side": random.choice(["buy", "sell"]),
                "amount": round(random.uniform(0.01, 1.0), 6),
                "price": round(random.uniform(40000, 50000), 2),
                "pnl": round(random.uniform(-50, 100), 2)
            })
        
        return {
            "success": True,
            "data": {
                "trades": trades,
                "total_pnl": round(sum(trade["pnl"] for trade in trades), 2)
            }
        }
    
    def get_health(self):
        """健康检查"""
        return {
            "status": "healthy",
            "timestamp": int(time.time() * 1000),
            "version": "2.0.0"
        }
    
    def log_message(self, format, *args):
        """简化日志输出"""
        pass

def run_mock_api():
    """运行模拟API服务器"""
    server = HTTPServer(('localhost', 8001), MockAPIHandler)
    print(f"🔗 模拟API服务器启动: http://localhost:8001")
    server.serve_forever()

if __name__ == '__main__':
    run_mock_api()