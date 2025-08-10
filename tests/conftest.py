"""
pytest配置文件和通用fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json
import logging

# 设置测试日志级别
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """创建临时目录用于测试"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_websocket():
    """模拟WebSocket连接"""
    mock_ws = Mock()
    mock_ws.send_str = Mock()
    mock_ws.close = Mock()
    mock_ws.prepare = Mock()
    return mock_ws

@pytest.fixture
def mock_exchange():
    """模拟加密货币交易所"""
    mock_exchange = Mock()
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 45000.0,
        'baseVolume': 1000000.0,
        'change': 500.0,
        'percentage': 1.12,
        'high': 46000.0,
        'low': 44000.0,
        'bid': 44950.0,
        'ask': 45050.0
    }
    mock_exchange.fetch_ohlcv.return_value = [
        [1640995200000, 45000, 46000, 44000, 45500, 1000],
        [1640998800000, 45500, 46500, 45000, 46000, 1100],
        [1641002400000, 46000, 46200, 45500, 45800, 950]
    ]
    return mock_exchange

@pytest.fixture
def sample_market_data():
    """示例市场数据"""
    return {
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'price': 45000.0,
            'volume_24h': 1000000.0,
            'change_24h': 500.0,
            'change_24h_pct': 1.12,
            'high_24h': 46000.0,
            'low_24h': 44000.0,
            'bid': 44950.0,
            'ask': 45050.0,
            'timestamp': 1641006000000,
            'exchange': 'test',
            'data_source': 'real'
        }
    }

@pytest.fixture
def test_config():
    """测试配置"""
    return {
        'server': {
            'host': 'localhost',
            'port': 8001,  # 使用不同端口避免冲突
            'dev_mode': True
        },
        'hot_reload': {
            'enabled': False,  # 测试时禁用热重载
            'watch_extensions': ['.py', '.html', '.css', '.js']
        },
        'logging': {
            'level': 'DEBUG'
        }
    }

@pytest.fixture
def mock_aiohttp_request():
    """模拟aiohttp请求"""
    mock_request = Mock()
    mock_request.query = {}
    mock_request.json = Mock()
    return mock_request

@pytest.fixture
def mock_file_watcher():
    """模拟文件监控器"""
    with patch('watchdog.observers.Observer') as mock_observer:
        mock_observer_instance = Mock()
        mock_observer.return_value = mock_observer_instance
        yield mock_observer_instance

@pytest.fixture
async def aiohttp_client():
    """aiohttp测试客户端"""
    from aiohttp.test_utils import TestClient, TestServer
    from aiohttp import web
    
    app = web.Application()
    
    # 添加基本路由用于测试
    async def hello(request):
        return web.json_response({'status': 'ok'})
    
    app.router.add_get('/test', hello)
    
    async with TestClient(TestServer(app)) as client:
        yield client

@pytest.fixture
def mock_ccxt_exchanges():
    """模拟CCXT交易所"""
    with patch('ccxt.okx') as mock_okx, patch('ccxt.binance') as mock_binance:
        # 配置OKX模拟
        mock_okx_instance = Mock()
        mock_okx_instance.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 45000.0,
            'baseVolume': 1000000.0,
            'change': 500.0,
            'percentage': 1.12,
            'high': 46000.0,
            'low': 44000.0,
            'bid': 44950.0,
            'ask': 45050.0
        }
        mock_okx.return_value = mock_okx_instance
        
        # 配置Binance模拟
        mock_binance_instance = Mock()
        mock_binance_instance.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 44800.0,
            'baseVolume': 950000.0,
            'change': 300.0,
            'percentage': 0.67,
            'high': 45800.0,
            'low': 44200.0,
            'bid': 44750.0,
            'ask': 44850.0
        }
        mock_binance.return_value = mock_binance_instance
        
        yield {
            'okx': mock_okx_instance,
            'binance': mock_binance_instance
        }

# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "e2e: 端到端测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "network: 需要网络连接的测试")

# 测试前后钩子
def pytest_sessionstart(session):
    """测试会话开始前的设置"""
    print("🚀 开始运行测试套件...")

def pytest_sessionfinish(session, exitstatus):
    """测试会话结束后的清理"""
    print(f"✅ 测试套件运行完成，退出状态: {exitstatus}")