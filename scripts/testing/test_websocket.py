#!/usr/bin/env python3
"""
WebSocket连接测试脚本
"""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    try:
        print("🔌 连接WebSocket:", uri)
        async with websockets.connect(uri) as websocket:
            
            # 发送订阅消息
            subscribe_message = {
                "type": "subscribe",
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
            }
            
            print("📡 发送订阅消息:", subscribe_message)
            await websocket.send(json.dumps(subscribe_message))
            
            # 接收消息
            print("📥 等待接收消息...")
            for i in range(10):  # 接收10条消息
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    print(f"📨 消息 {i+1}:")
                    print(f"   类型: {data.get('type')}")
                    
                    if data.get('type') == 'market_update':
                        market_data = data.get('data', {})
                        print(f"   币种: {market_data.get('symbol')}")
                        print(f"   价格: ${market_data.get('price')}")
                        print(f"   交易所: {market_data.get('exchange')}")
                        print(f"   数据源: {market_data.get('data_source')}")
                        print(f"   时间: {market_data.get('timestamp')}")
                        
                        # 验证数据源必须是真实的
                        if market_data.get('data_source') != 'real':
                            print(f"   ❌ 警告: 数据源不是真实数据!")
                        else:
                            print(f"   ✅ 确认: 真实数据源")
                            
                    elif data.get('type') == 'data_error':
                        print(f"   ⚠️ 数据错误: {data.get('message')}")
                        
                    elif data.get('type') == 'connection_success':
                        print(f"   🎉 连接成功: {data.get('message')}")
                        
                    print()
                    
                except asyncio.TimeoutError:
                    print(f"   ⏰ 超时等待消息 {i+1}")
                    break
                    
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())