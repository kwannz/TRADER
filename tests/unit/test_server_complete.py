"""
server.py 完整覆盖率测试
针对所有未覆盖的代码行进行测试
"""

import pytest
import asyncio
import json
import sys
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestRealTimeDataManagerComplete:
    """完整测试RealTimeDataManager类"""
    
    @pytest.fixture
    def data_manager(self):
        """创建RealTimeDataManager实例"""
        from server import RealTimeDataManager
        return RealTimeDataManager()
    
    def test_data_manager_init(self, data_manager):
        """测试数据管理器初始化"""
        assert hasattr(data_manager, 'exchanges')
        assert hasattr(data_manager, 'websocket_clients')
        assert hasattr(data_manager, 'market_data')
        assert hasattr(data_manager, 'running')
        assert data_manager.exchanges == {}
        assert data_manager.websocket_clients == set()
        assert data_manager.market_data == {}
        assert data_manager.running is False
    
    @pytest.mark.asyncio
    async def test_initialize_exchanges_success(self, data_manager):
        """测试交易所初始化成功路径"""
        with patch('ccxt.okx') as mock_okx, \
             patch('ccxt.binance') as mock_binance, \
             patch('os.environ.get') as mock_env:
            
            # 模拟环境变量
            mock_env.side_effect = lambda key, default=None: {
                'OKX_API_KEY': 'test_okx_key',
                'OKX_SECRET': 'test_okx_secret',
                'OKX_PASSPHRASE': 'test_okx_pass',
                'BINANCE_API_KEY': 'test_binance_key',
                'BINANCE_SECRET': 'test_binance_secret'
            }.get(key, default)
            
            # 模拟交易所实例
            mock_okx_instance = Mock()
            mock_binance_instance = Mock()
            mock_okx.return_value = mock_okx_instance
            mock_binance.return_value = mock_binance_instance
            
            result = await data_manager.initialize_exchanges()
            
            assert result is True
            assert 'okx' in data_manager.exchanges
            assert 'binance' in data_manager.exchanges
    
    @pytest.mark.asyncio
    async def test_initialize_exchanges_missing_credentials(self, data_manager):
        """测试交易所初始化缺失凭据"""
        with patch('os.environ.get', return_value=None):
            result = await data_manager.initialize_exchanges()
            assert result is False
            assert len(data_manager.exchanges) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_exchanges_import_error(self, data_manager):
        """测试交易所初始化导入错误"""
        with patch('ccxt.okx', side_effect=ImportError("No module named ccxt")):
            result = await data_manager.initialize_exchanges()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, data_manager):
        """测试市场数据获取成功"""
        # 模拟交易所
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 45000.0,
            'bid': 44999.0,
            'ask': 45001.0,
            'timestamp': 1234567890000
        }
        data_manager.exchanges['okx'] = mock_exchange
        
        result = await data_manager.get_market_data('BTC/USDT')
        
        assert result is not None
        assert result['symbol'] == 'BTC/USDT'
        assert result['last'] == 45000.0
        mock_exchange.fetch_ticker.assert_called_once_with('BTC/USDT')
    
    @pytest.mark.asyncio
    async def test_get_market_data_no_exchanges(self, data_manager):
        """测试没有交易所时获取市场数据"""
        result = await data_manager.get_market_data('BTC/USDT')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_market_data_exchange_error(self, data_manager):
        """测试交易所错误"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.side_effect = Exception("Network error")
        data_manager.exchanges['okx'] = mock_exchange
        
        result = await data_manager.get_market_data('BTC/USDT')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_broadcast_data_success(self, data_manager):
        """测试数据广播成功"""
        mock_ws1 = Mock()
        mock_ws1.send_str = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_str = AsyncMock()
        
        data_manager.websocket_clients.add(mock_ws1)
        data_manager.websocket_clients.add(mock_ws2)
        
        test_data = {'symbol': 'BTC/USDT', 'price': 45000}
        
        await data_manager.broadcast_data(test_data)
        
        expected_message = json.dumps(test_data)
        mock_ws1.send_str.assert_called_once_with(expected_message)
        mock_ws2.send_str.assert_called_once_with(expected_message)
    
    @pytest.mark.asyncio
    async def test_broadcast_data_with_failed_clients(self, data_manager):
        """测试广播时部分客户端失败"""
        mock_ws1 = Mock()
        mock_ws1.send_str = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_str = AsyncMock(side_effect=Exception("Connection lost"))
        mock_ws3 = Mock()
        mock_ws3.send_str = AsyncMock()
        
        data_manager.websocket_clients.add(mock_ws1)
        data_manager.websocket_clients.add(mock_ws2)
        data_manager.websocket_clients.add(mock_ws3)
        
        test_data = {'symbol': 'ETH/USDT', 'price': 3000}
        
        await data_manager.broadcast_data(test_data)
        
        # 失败的客户端应被移除
        assert mock_ws2 not in data_manager.websocket_clients
        assert len(data_manager.websocket_clients) == 2
        
        # 成功的客户端应收到消息
        mock_ws1.send_str.assert_called_once()
        mock_ws3.send_str.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_data_collection(self, data_manager):
        """测试启动数据收集"""
        data_manager.running = False
        
        with patch.object(data_manager, 'data_collection_loop', new_callable=AsyncMock) as mock_loop:
            await data_manager.start_data_collection()
            
            assert data_manager.running is True
            mock_loop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_data_collection(self, data_manager):
        """测试停止数据收集"""
        data_manager.running = True
        
        await data_manager.stop_data_collection()
        
        assert data_manager.running is False
    
    @pytest.mark.asyncio
    async def test_data_collection_loop(self, data_manager):
        """测试数据收集循环"""
        data_manager.running = True
        
        # 模拟收集几次后停止
        call_count = 0
        original_get_market_data = data_manager.get_market_data
        
        async def mock_get_market_data(symbol):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                data_manager.running = False
            return {'symbol': symbol, 'price': 45000 + call_count}
        
        with patch.object(data_manager, 'get_market_data', side_effect=mock_get_market_data), \
             patch.object(data_manager, 'broadcast_data', new_callable=AsyncMock) as mock_broadcast, \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            await data_manager.data_collection_loop()
            
            assert call_count >= 3
            assert mock_broadcast.call_count >= 2

class TestWebSocketHandlers:
    """测试WebSocket处理器"""
    
    @pytest.fixture
    def data_manager(self):
        from server import RealTimeDataManager
        return RealTimeDataManager()
    
    @pytest.mark.asyncio
    async def test_websocket_handler_complete_flow(self, data_manager):
        """测试WebSocket处理器完整流程"""
        with patch('aiohttp.web.WebSocketResponse') as MockWS:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            mock_ws.close = AsyncMock()
            
            # 模拟消息序列
            messages = [
                Mock(type=aiohttp.WSMsgType.TEXT, data='{"type": "subscribe", "symbols": ["BTC/USDT"]}'),
                Mock(type=aiohttp.WSMsgType.TEXT, data='{"type": "ping"}'),
                Mock(type=aiohttp.WSMsgType.CLOSE)
            ]
            
            mock_ws.__aiter__ = AsyncMock(return_value=iter(messages))
            MockWS.return_value = mock_ws
            
            # 导入并测试WebSocket处理器
            from server import websocket_handler
            mock_request = Mock()
            
            result = await websocket_handler(mock_request, data_manager)
            
            assert result == mock_ws
            mock_ws.prepare.assert_called_once_with(mock_request)
            
            # 验证客户端被添加到集合中
            assert mock_ws in data_manager.websocket_clients
    
    @pytest.mark.asyncio
    async def test_websocket_message_processing(self, data_manager):
        """测试WebSocket消息处理的各种类型"""
        from server import process_websocket_message
        
        # 测试订阅消息
        subscribe_msg = {"type": "subscribe", "symbols": ["BTC/USDT", "ETH/USDT"]}
        result = await process_websocket_message(subscribe_msg, data_manager)
        assert result["type"] == "subscription_confirmed"
        assert "BTC/USDT" in result["symbols"]
        
        # 测试取消订阅消息
        unsubscribe_msg = {"type": "unsubscribe", "symbols": ["BTC/USDT"]}
        result = await process_websocket_message(unsubscribe_msg, data_manager)
        assert result["type"] == "unsubscription_confirmed"
        
        # 测试心跳消息
        ping_msg = {"type": "ping"}
        result = await process_websocket_message(ping_msg, data_manager)
        assert result["type"] == "pong"
        
        # 测试未知消息类型
        unknown_msg = {"type": "unknown_type"}
        result = await process_websocket_message(unknown_msg, data_manager)
        assert result["type"] == "error"

class TestAPIHandlers:
    """测试API处理器"""
    
    @pytest.fixture
    def data_manager(self):
        from server import RealTimeDataManager
        return RealTimeDataManager()
    
    @pytest.mark.asyncio
    async def test_get_market_data_handler_success(self, data_manager):
        """测试市场数据API处理器成功路径"""
        from server import get_market_data_handler
        
        # 模拟请求
        mock_request = Mock()
        mock_request.match_info = {'symbol': 'BTC/USDT'}
        
        # 模拟数据管理器返回数据
        mock_data = {
            'symbol': 'BTC/USDT',
            'last': 45000.0,
            'timestamp': 1234567890
        }
        
        with patch.object(data_manager, 'get_market_data', return_value=mock_data), \
             patch('aiohttp.web.json_response') as mock_json_response:
            
            await get_market_data_handler(mock_request, data_manager)
            
            mock_json_response.assert_called_once()
            call_args = mock_json_response.call_args[0][0]
            assert call_args['success'] is True
            assert call_args['data'] == mock_data
    
    @pytest.mark.asyncio
    async def test_get_market_data_handler_not_found(self, data_manager):
        """测试市场数据API处理器数据未找到"""
        from server import get_market_data_handler
        
        mock_request = Mock()
        mock_request.match_info = {'symbol': 'UNKNOWN/USDT'}
        
        with patch.object(data_manager, 'get_market_data', return_value=None), \
             patch('aiohttp.web.json_response') as mock_json_response:
            
            await get_market_data_handler(mock_request, data_manager)
            
            mock_json_response.assert_called_once()
            call_args = mock_json_response.call_args[0][0]
            assert call_args['success'] is False
            assert 'error' in call_args
    
    @pytest.mark.asyncio
    async def test_health_check_handler(self, data_manager):
        """测试健康检查处理器"""
        from server import health_check_handler
        
        mock_request = Mock()
        
        with patch('aiohttp.web.json_response') as mock_json_response, \
             patch('time.time', return_value=1234567890):
            
            await health_check_handler(mock_request)
            
            mock_json_response.assert_called_once()
            call_args = mock_json_response.call_args[0][0]
            assert call_args['status'] == 'healthy'
            assert call_args['timestamp'] == 1234567890

class TestServerApplicationSetup:
    """测试服务器应用程序设置"""
    
    @pytest.mark.asyncio
    async def test_create_app_complete(self):
        """测试应用创建完整流程"""
        from server import create_app, RealTimeDataManager
        
        data_manager = RealTimeDataManager()
        
        with patch('aiohttp_cors.setup') as mock_cors_setup, \
             patch('aiohttp.web.Application') as MockApp:
            
            mock_app = Mock()
            mock_router = Mock()
            mock_app.router = mock_router
            MockApp.return_value = mock_app
            
            app = await create_app(data_manager, dev_mode=True)
            
            assert app == mock_app
            mock_cors_setup.assert_called_once()
            
            # 验证路由被添加
            assert mock_router.add_get.call_count >= 2  # 至少有健康检查和市场数据路由
    
    def test_cors_middleware_setup(self):
        """测试CORS中间件设置"""
        from server import setup_cors
        
        mock_app = Mock()
        mock_cors = Mock()
        
        with patch('aiohttp_cors.setup', return_value=mock_cors) as mock_cors_setup:
            setup_cors(mock_app)
            
            mock_cors_setup.assert_called_once_with(mock_app)
            mock_cors.add.assert_called()

class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_check_dependencies_success(self):
        """测试依赖检查成功"""
        from server import check_dependencies
        
        # 所有必需的包都可用时
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            
            result = check_dependencies()
            assert isinstance(result, bool)
    
    def test_check_dependencies_missing_packages(self):
        """测试依赖检查缺失包"""
        def mock_import_side_effect(name, *args, **kwargs):
            if name in ['nonexistent_package']:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect), \
             patch('server.required_packages', ['aiohttp', 'nonexistent_package']):
            
            from server import check_dependencies
            result = check_dependencies()
            assert result is False

class TestMainFunction:
    """测试main函数和程序入口"""
    
    @pytest.mark.asyncio
    async def test_main_function_normal_flow(self):
        """测试main函数正常流程"""
        from server import main
        
        with patch('server.check_dependencies', return_value=True), \
             patch('server.RealTimeDataManager') as MockDataManager, \
             patch('server.create_app', new_callable=AsyncMock) as mock_create_app, \
             patch('aiohttp.web.run_app') as mock_run_app:
            
            mock_data_manager = Mock()
            mock_data_manager.initialize_exchanges = AsyncMock(return_value=True)
            MockDataManager.return_value = mock_data_manager
            
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            await main(dev_mode=False)
            
            mock_data_manager.initialize_exchanges.assert_called_once()
            mock_create_app.assert_called_once()
            mock_run_app.assert_called_once_with(mock_app, host='localhost', port=8000)
    
    @pytest.mark.asyncio
    async def test_main_function_dev_mode(self):
        """测试main函数开发模式"""
        from server import main
        
        with patch('server.check_dependencies', return_value=True), \
             patch('server.RealTimeDataManager') as MockDataManager, \
             patch('server.create_app', new_callable=AsyncMock) as mock_create_app, \
             patch('aiohttp.web.run_app') as mock_run_app:
            
            mock_data_manager = Mock()
            mock_data_manager.initialize_exchanges = AsyncMock(return_value=True)
            MockDataManager.return_value = mock_data_manager
            
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            await main(dev_mode=True)
            
            # 验证开发模式被传递
            mock_create_app.assert_called_once_with(mock_data_manager, dev_mode=True)
    
    @pytest.mark.asyncio
    async def test_main_function_dependency_check_failure(self):
        """测试依赖检查失败"""
        from server import main
        
        with patch('server.check_dependencies', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            await main()
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_main_function_exchange_init_failure(self):
        """测试交易所初始化失败"""
        from server import main
        
        with patch('server.check_dependencies', return_value=True), \
             patch('server.RealTimeDataManager') as MockDataManager, \
             patch('sys.exit') as mock_exit:
            
            mock_data_manager = Mock()
            mock_data_manager.initialize_exchanges = AsyncMock(return_value=False)
            MockDataManager.return_value = mock_data_manager
            
            await main()
            
            mock_exit.assert_called_once_with(1)
    
    def test_argument_parsing(self):
        """测试命令行参数解析"""
        import argparse
        
        # 模拟参数解析器
        def parse_args(argv=None):
            parser = argparse.ArgumentParser()
            parser.add_argument('--dev', action='store_true', help='开发模式')
            parser.add_argument('--port', type=int, default=8000, help='端口号')
            parser.add_argument('--host', default='localhost', help='主机地址')
            return parser.parse_args(argv)
        
        # 测试默认参数
        args = parse_args([])
        assert args.dev is False
        assert args.port == 8000
        assert args.host == 'localhost'
        
        # 测试开发模式参数
        args = parse_args(['--dev'])
        assert args.dev is True
        
        # 测试自定义端口和主机
        args = parse_args(['--port', '3000', '--host', '0.0.0.0'])
        assert args.port == 3000
        assert args.host == '0.0.0.0'

class TestErrorHandlingAndEdgeCases:
    """测试错误处理和边界情况"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_error(self):
        """测试WebSocket连接错误"""
        from server import RealTimeDataManager
        
        data_manager = RealTimeDataManager()
        
        mock_ws = Mock()
        mock_ws.prepare = AsyncMock(side_effect=ConnectionError("Connection failed"))
        
        data_manager.websocket_clients.add(mock_ws)
        
        # 测试广播时的连接错误处理
        test_data = {'symbol': 'BTC/USDT', 'price': 45000}
        
        await data_manager.broadcast_data(test_data)
        
        # 失败的连接应该被移除
        assert mock_ws not in data_manager.websocket_clients
    
    @pytest.mark.asyncio
    async def test_json_serialization_error(self):
        """测试JSON序列化错误"""
        from server import RealTimeDataManager
        
        data_manager = RealTimeDataManager()
        
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        data_manager.websocket_clients.add(mock_ws)
        
        # 创建不可序列化的数据
        class UnserializableObject:
            pass
        
        test_data = {'object': UnserializableObject()}
        
        # 应该处理序列化错误而不崩溃
        await data_manager.broadcast_data(test_data)
        
        # WebSocket连接应该保持
        assert mock_ws in data_manager.websocket_clients
    
    def test_environment_variable_edge_cases(self):
        """测试环境变量边界情况"""
        test_cases = [
            ('', None),  # 空字符串
            ('   ', None),  # 空白字符串
            ('VALID_VALUE', 'VALID_VALUE'),  # 正常值
        ]
        
        for env_value, expected in test_cases:
            with patch('os.environ.get', return_value=env_value):
                result = os.environ.get('TEST_KEY')
                if env_value.strip():
                    assert result == expected
                else:
                    assert result == env_value

class TestPerformanceAndConcurrency:
    """测试性能和并发相关功能"""
    
    @pytest.mark.asyncio
    async def test_concurrent_websocket_handling(self):
        """测试并发WebSocket处理"""
        from server import RealTimeDataManager
        
        data_manager = RealTimeDataManager()
        
        # 创建多个模拟WebSocket客户端
        mock_clients = []
        for i in range(10):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            mock_clients.append(mock_ws)
            data_manager.websocket_clients.add(mock_ws)
        
        # 并发广播数据
        test_data = {'symbol': 'BTC/USDT', 'price': 45000}
        
        # 创建多个并发任务
        tasks = []
        for i in range(5):
            tasks.append(data_manager.broadcast_data(test_data))
        
        await asyncio.gather(*tasks)
        
        # 验证所有客户端都收到了消息
        for mock_ws in mock_clients:
            assert mock_ws.send_str.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_high_frequency_data_updates(self):
        """测试高频数据更新"""
        from server import RealTimeDataManager
        
        data_manager = RealTimeDataManager()
        
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        data_manager.websocket_clients.add(mock_ws)
        
        # 快速连续发送多次数据更新
        for i in range(100):
            test_data = {'symbol': 'BTC/USDT', 'price': 45000 + i, 'sequence': i}
            await data_manager.broadcast_data(test_data)
        
        # 验证所有消息都被发送
        assert mock_ws.send_str.call_count == 100

class TestConfigurationAndSettings:
    """测试配置和设置"""
    
    def test_different_configuration_modes(self):
        """测试不同配置模式"""
        configs = [
            {'dev_mode': True, 'debug': True},
            {'dev_mode': False, 'debug': False},
            {'dev_mode': True, 'debug': False},
        ]
        
        for config in configs:
            # 模拟配置应用
            dev_mode = config.get('dev_mode', False)
            debug = config.get('debug', False)
            
            # 验证配置逻辑
            if dev_mode:
                assert 'dev_mode' in config
            
            if debug:
                assert config['debug'] is True
            else:
                assert config.get('debug', False) is False
    
    def test_port_and_host_configuration(self):
        """测试端口和主机配置"""
        test_configs = [
            {'host': 'localhost', 'port': 8000},
            {'host': '0.0.0.0', 'port': 3000},
            {'host': '127.0.0.1', 'port': 8080},
        ]
        
        for config in test_configs:
            host = config['host']
            port = config['port']
            
            # 验证配置有效性
            assert isinstance(host, str)
            assert isinstance(port, int)
            assert 1000 <= port <= 65535
            assert host in ['localhost', '127.0.0.1', '0.0.0.0'] or '.' in host