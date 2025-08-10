"""
测试辅助工具函数
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Any, Dict, List, Optional

class AsyncContextManager:
    """异步上下文管理器用于测试"""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockWebSocketResponse:
    """模拟WebSocket响应"""
    
    def __init__(self):
        self.messages = []
        self.closed = False
        self.close_code = None
        
    async def send_str(self, data: str):
        """发送字符串消息"""
        self.messages.append(('text', data))
        
    async def send_bytes(self, data: bytes):
        """发送字节消息"""
        self.messages.append(('bytes', data))
        
    async def close(self, code=1000):
        """关闭连接"""
        self.closed = True
        self.close_code = code
        
    def get_messages(self) -> List[tuple]:
        """获取所有消息"""
        return self.messages
        
    def get_last_message(self) -> Optional[tuple]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None

def create_mock_request(query_params: Optional[Dict[str, str]] = None, 
                       json_data: Optional[Dict] = None):
    """创建模拟HTTP请求"""
    mock_request = Mock()
    mock_request.query = query_params or {}
    
    if json_data:
        mock_request.json = AsyncMock(return_value=json_data)
    else:
        mock_request.json = AsyncMock(return_value={})
        
    return mock_request

def create_temp_file(content: str, suffix: str = '.py') -> Path:
    """创建临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)

def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """等待条件满足"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    return False

async def async_wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """异步等待条件满足"""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
    return False

class MockFileSystemEventHandler:
    """模拟文件系统事件处理器"""
    
    def __init__(self):
        self.events = []
        
    def on_modified(self, event):
        """文件修改事件"""
        self.events.append(('modified', event))
        
    def on_created(self, event):
        """文件创建事件"""
        self.events.append(('created', event))
        
    def on_deleted(self, event):
        """文件删除事件"""
        self.events.append(('deleted', event))
        
    def get_events(self):
        """获取所有事件"""
        return self.events
        
    def clear_events(self):
        """清空事件"""
        self.events = []

def assert_websocket_message_sent(mock_ws, expected_message_type: str, expected_content: Optional[Dict] = None):
    """断言WebSocket消息已发送"""
    mock_ws.send_str.assert_called()
    
    if expected_content:
        # 检查最后发送的消息
        last_call = mock_ws.send_str.call_args_list[-1]
        message_data = json.loads(last_call[0][0])
        
        assert message_data.get('type') == expected_message_type
        
        for key, value in expected_content.items():
            assert message_data.get(key) == value

def create_sample_market_data(symbol: str = 'BTC/USDT', price: float = 45000.0) -> Dict[str, Any]:
    """创建示例市场数据"""
    return {
        'symbol': symbol,
        'price': price,
        'volume_24h': 1000000.0,
        'change_24h': 500.0,
        'change_24h_pct': 1.12,
        'high_24h': price + 1000,
        'low_24h': price - 1000,
        'bid': price - 50,
        'ask': price + 50,
        'timestamp': int(time.time() * 1000),
        'exchange': 'test',
        'data_source': 'real'
    }

def create_sample_ohlcv_data(symbol: str = 'BTC/USDT', count: int = 3) -> List[List]:
    """创建示例OHLCV数据"""
    base_time = int(time.time()) * 1000
    data = []
    
    for i in range(count):
        timestamp = base_time + (i * 3600000)  # 每小时
        open_price = 45000 + (i * 100)
        high_price = open_price + 500
        low_price = open_price - 300
        close_price = open_price + 200
        volume = 1000 + (i * 50)
        
        data.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    return data

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_market_data_batch(symbols: List[str], count: int = 10) -> Dict[str, List[Dict]]:
        """生成批量市场数据"""
        data = {}
        for symbol in symbols:
            data[symbol] = []
            for i in range(count):
                base_price = 45000 if 'BTC' in symbol else 2800 if 'ETH' in symbol else 300
                price_variation = (i - count//2) * 100
                
                data[symbol].append(create_sample_market_data(
                    symbol=symbol,
                    price=base_price + price_variation
                ))
        return data
    
    @staticmethod
    def generate_websocket_messages(count: int = 5) -> List[Dict]:
        """生成WebSocket测试消息"""
        messages = []
        message_types = ['market_update', 'reload_frontend', 'backend_restarting', 'dev_connected']
        
        for i in range(count):
            msg_type = message_types[i % len(message_types)]
            messages.append({
                'type': msg_type,
                'message': f'Test message {i}',
                'timestamp': int(time.time() * 1000) + i
            })
        
        return messages

# 装饰器
def skip_if_no_network(func):
    """如果没有网络连接则跳过测试"""
    import socket
    
    def wrapper(*args, **kwargs):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return func(*args, **kwargs)
        except OSError:
            import pytest
            pytest.skip("No network connection available")
    
    return wrapper

def async_test(func):
    """异步测试装饰器"""
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper