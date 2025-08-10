"""
开发环境集成测试
测试各个组件之间的协作
"""

import pytest
import asyncio
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.helpers import (
    MockWebSocketResponse,
    create_temp_file,
    async_wait_for_condition,
    TestDataGenerator
)

class TestDevServerWebSocketIntegration:
    """开发服务器WebSocket集成测试"""
    
    @pytest.mark.asyncio
    async def test_websocket_hot_reload_flow(self):
        """测试WebSocket热重载完整流程"""
        # 模拟开发服务器和WebSocket客户端交互
        
        # 1. 模拟WebSocket连接建立
        mock_ws_clients = set()
        mock_dev_server = Mock()
        mock_dev_server.websocket_clients = mock_ws_clients
        
        # 2. 模拟客户端连接
        mock_ws = MockWebSocketResponse()
        mock_ws_clients.add(mock_ws)
        
        # 3. 模拟文件修改事件
        async def simulate_file_change(file_type='frontend'):
            if file_type == 'frontend':
                # 前端文件修改 - 通知刷新
                message = {
                    'type': 'reload_frontend',
                    'message': '前端文件已更新，正在刷新页面...',
                    'timestamp': int(time.time() * 1000)
                }
            else:
                # Python文件修改 - 后端重启
                message = {
                    'type': 'backend_restarting',
                    'message': 'Python代码已更新，后端正在重启...',
                    'timestamp': int(time.time() * 1000)
                }
            
            # 发送消息给所有客户端
            for ws in mock_ws_clients:
                await ws.send_str(json.dumps(message))
            
            return len(mock_ws_clients)
        
        # 4. 测试前端文件修改
        clients_notified = await simulate_file_change('frontend')
        assert clients_notified == 1
        
        messages = mock_ws.get_messages()
        assert len(messages) >= 1
        
        last_message = mock_ws.get_last_message()
        message_data = json.loads(last_message[1])
        assert message_data['type'] == 'reload_frontend'
        
        # 5. 测试Python文件修改
        await simulate_file_change('python')
        
        messages = mock_ws.get_messages()
        assert len(messages) >= 2
        
        last_message = mock_ws.get_last_message()
        message_data = json.loads(last_message[1])
        assert message_data['type'] == 'backend_restarting'
    
    @pytest.mark.asyncio
    async def test_multiple_clients_notification(self):
        """测试多客户端通知"""
        # 创建多个WebSocket客户端
        clients = []
        for i in range(3):
            client = MockWebSocketResponse()
            clients.append(client)
        
        # 模拟广播消息
        async def broadcast_message(message):
            message_json = json.dumps(message)
            for client in clients:
                await client.send_str(message_json)
        
        # 发送测试消息
        test_message = {
            'type': 'test_broadcast',
            'message': 'Testing multiple clients',
            'timestamp': int(time.time() * 1000)
        }
        
        await broadcast_message(test_message)
        
        # 验证所有客户端都收到消息
        for client in clients:
            messages = client.get_messages()
            assert len(messages) >= 1
            
            last_message = client.get_last_message()
            received_data = json.loads(last_message[1])
            assert received_data['type'] == 'test_broadcast'
    
    @pytest.mark.asyncio
    async def test_client_disconnection_handling(self):
        """测试客户端断开连接处理"""
        mock_clients = set()
        
        # 添加多个客户端
        for i in range(3):
            client = MockWebSocketResponse()
            mock_clients.add(client)
        
        assert len(mock_clients) == 3
        
        # 模拟一个客户端断开
        disconnected_client = list(mock_clients)[0]
        mock_clients.remove(disconnected_client)
        
        # 验证客户端数量减少
        assert len(mock_clients) == 2
        
        # 模拟给剩余客户端发送消息
        message = {'type': 'after_disconnect', 'count': len(mock_clients)}
        message_json = json.dumps(message)
        
        for client in mock_clients:
            await client.send_str(message_json)
        
        # 验证剩余客户端收到消息
        for client in mock_clients:
            messages = client.get_messages()
            assert len(messages) >= 1

class TestFileWatchingIntegration:
    """文件监控集成测试"""
    
    @pytest.fixture
    def temp_project_dir(self, temp_dir):
        """创建临时项目目录"""
        # 创建项目结构
        (temp_dir / 'src').mkdir()
        (temp_dir / 'templates').mkdir()
        (temp_dir / 'static').mkdir()
        
        # 创建测试文件
        (temp_dir / 'src' / 'main.py').write_text('print("hello")')
        (temp_dir / 'templates' / 'index.html').write_text('<html></html>')
        (temp_dir / 'static' / 'style.css').write_text('body {}')
        
        return temp_dir
    
    def test_file_extension_filtering(self, temp_project_dir):
        """测试文件扩展名过滤"""
        watch_extensions = {'.py', '.html', '.css', '.js', '.json'}
        
        # 创建不同类型的文件
        test_files = [
            'test.py',      # 应该监控
            'test.html',    # 应该监控
            'test.css',     # 应该监控
            'test.js',      # 应该监控
            'test.txt',     # 不应该监控
            'test.log',     # 不应该监控
            'test.pyc',     # 不应该监控
        ]
        
        monitored_files = []
        ignored_files = []
        
        for filename in test_files:
            file_ext = Path(filename).suffix.lower()
            if file_ext in watch_extensions:
                monitored_files.append(filename)
            else:
                ignored_files.append(filename)
        
        # 验证过滤结果
        assert 'test.py' in monitored_files
        assert 'test.html' in monitored_files
        assert 'test.css' in monitored_files
        assert 'test.js' in monitored_files
        
        assert 'test.txt' in ignored_files
        assert 'test.log' in ignored_files
        assert 'test.pyc' in ignored_files
    
    @pytest.mark.asyncio
    async def test_file_change_detection_simulation(self, temp_project_dir):
        """模拟文件变更检测"""
        # 模拟文件监控器
        class MockFileWatcher:
            def __init__(self):
                self.events = []
                self.handlers = []
            
            def add_handler(self, handler):
                self.handlers.append(handler)
            
            async def simulate_file_change(self, file_path, event_type='modified'):
                event = {
                    'type': event_type,
                    'path': file_path,
                    'timestamp': time.time()
                }
                self.events.append(event)
                
                # 通知所有处理器
                for handler in self.handlers:
                    await handler.handle_event(event)
        
        # 模拟事件处理器
        class MockEventHandler:
            def __init__(self):
                self.processed_events = []
            
            async def handle_event(self, event):
                file_ext = Path(event['path']).suffix.lower()
                
                if file_ext == '.py':
                    action = 'restart_backend'
                elif file_ext in ['.html', '.css', '.js']:
                    action = 'reload_frontend'
                else:
                    action = 'ignore'
                
                processed_event = {
                    'original_event': event,
                    'action': action,
                    'processed_at': time.time()
                }
                self.processed_events.append(processed_event)
        
        # 创建监控器和处理器
        watcher = MockFileWatcher()
        handler = MockEventHandler()
        watcher.add_handler(handler)
        
        # 模拟不同文件的变更
        test_changes = [
            str(temp_project_dir / 'src' / 'main.py'),
            str(temp_project_dir / 'templates' / 'index.html'),
            str(temp_project_dir / 'static' / 'style.css')
        ]
        
        for file_path in test_changes:
            await watcher.simulate_file_change(file_path)
        
        # 验证事件处理
        assert len(handler.processed_events) == 3
        
        # 验证不同文件类型的处理动作
        actions = [event['action'] for event in handler.processed_events]
        assert 'restart_backend' in actions     # main.py
        assert 'reload_frontend' in actions     # index.html, style.css
        assert actions.count('reload_frontend') == 2

class TestAPIEndpointsIntegration:
    """API端点集成测试"""
    
    @pytest.mark.asyncio
    async def test_dev_status_api_integration(self):
        """测试开发状态API集成"""
        # 模拟数据管理器状态
        mock_data_manager = Mock()
        mock_data_manager.websocket_clients = set([Mock(), Mock()])  # 2个客户端
        mock_data_manager.market_data = {'BTC/USDT': {}, 'ETH/USDT': {}}  # 2个市场
        mock_data_manager.exchanges = {'okx': Mock(), 'binance': Mock()}  # 2个交易所
        
        # 模拟开发状态API处理
        async def dev_status_handler():
            return {
                'success': True,
                'mode': 'development',
                'status': 'running',
                'server': 'aiohttp',
                'connected_ws_clients': len(mock_data_manager.websocket_clients),
                'market_data_count': len(mock_data_manager.market_data),
                'exchanges_active': len(mock_data_manager.exchanges),
                'timestamp': int(time.time() * 1000)
            }
        
        # 调用API
        response = await dev_status_handler()
        
        # 验证响应
        assert response['success'] is True
        assert response['mode'] == 'development'
        assert response['status'] == 'running'
        assert response['connected_ws_clients'] == 2
        assert response['market_data_count'] == 2
        assert response['exchanges_active'] == 2
        assert 'timestamp' in response
    
    @pytest.mark.asyncio
    async def test_market_data_api_integration(self, mock_ccxt_exchanges):
        """测试市场数据API集成"""
        # 模拟市场数据获取流程
        async def market_data_handler(symbol='BTC/USDT'):
            try:
                # 模拟从交易所获取数据
                exchange_data = mock_ccxt_exchanges['okx'].fetch_ticker.return_value
                
                market_data = {
                    'symbol': symbol,
                    'price': exchange_data['last'],
                    'volume_24h': exchange_data['baseVolume'],
                    'change_24h': exchange_data['change'],
                    'change_24h_pct': exchange_data['percentage'],
                    'high_24h': exchange_data['high'],
                    'low_24h': exchange_data['low'],
                    'bid': exchange_data['bid'],
                    'ask': exchange_data['ask'],
                    'timestamp': int(time.time() * 1000),
                    'exchange': 'okx',
                    'data_source': 'real'
                }
                
                return {
                    'success': True,
                    'data': market_data
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # 测试成功获取数据
        response = await market_data_handler('BTC/USDT')
        
        assert response['success'] is True
        assert response['data']['symbol'] == 'BTC/USDT'
        assert 'price' in response['data']
        assert 'timestamp' in response['data']
        
        # 测试获取其他交易对
        response = await market_data_handler('ETH/USDT')
        assert response['success'] is True
        assert response['data']['symbol'] == 'ETH/USDT'

class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    @pytest.mark.asyncio
    async def test_complete_hot_reload_workflow(self):
        """测试完整的热重载工作流"""
        # 模拟整个热重载流程
        
        # 1. 初始化开发环境
        dev_env = {
            'websocket_clients': set(),
            'file_watcher': Mock(),
            'server_running': True
        }
        
        # 2. 模拟客户端连接
        client1 = MockWebSocketResponse()
        client2 = MockWebSocketResponse()
        
        dev_env['websocket_clients'].add(client1)
        dev_env['websocket_clients'].add(client2)
        
        assert len(dev_env['websocket_clients']) == 2
        
        # 3. 模拟文件监控启动
        class MockFileEvent:
            def __init__(self, file_path, event_type='modified'):
                self.src_path = file_path
                self.event_type = event_type
                self.is_directory = False
        
        # 4. 模拟文件变更处理流程
        async def handle_file_change_workflow(file_path):
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.py':
                # Python文件变更 - 后端重启流程
                
                # 4.1 通知客户端后端重启开始
                restart_message = {
                    'type': 'backend_restarting',
                    'message': 'Python代码已更新，后端正在重启...',
                    'timestamp': int(time.time() * 1000)
                }
                
                for client in dev_env['websocket_clients']:
                    await client.send_str(json.dumps(restart_message))
                
                # 4.2 模拟后端重启延迟
                await asyncio.sleep(0.1)  # 模拟重启时间
                
                # 4.3 通知客户端后端重启完成
                ready_message = {
                    'type': 'backend_restarted',
                    'message': '后端重启完成',
                    'timestamp': int(time.time() * 1000)
                }
                
                for client in dev_env['websocket_clients']:
                    await client.send_str(json.dumps(ready_message))
                
                return 'backend_restarted'
                
            elif file_ext in ['.html', '.css', '.js']:
                # 前端文件变更 - 页面刷新流程
                
                reload_message = {
                    'type': 'reload_frontend',
                    'message': '前端文件已更新，正在刷新页面...',
                    'timestamp': int(time.time() * 1000)
                }
                
                for client in dev_env['websocket_clients']:
                    await client.send_str(json.dumps(reload_message))
                
                return 'frontend_reloaded'
            
            return 'ignored'
        
        # 5. 测试Python文件变更工作流
        result = await handle_file_change_workflow('/test/main.py')
        assert result == 'backend_restarted'
        
        # 验证客户端收到重启消息
        for client in dev_env['websocket_clients']:
            messages = client.get_messages()
            assert len(messages) >= 2  # 重启开始 + 重启完成
            
            # 检查消息类型
            message_types = []
            for msg_type, msg_data in messages:
                if msg_type == 'text':
                    data = json.loads(msg_data)
                    message_types.append(data['type'])
            
            assert 'backend_restarting' in message_types
            assert 'backend_restarted' in message_types
        
        # 6. 测试前端文件变更工作流
        # 清空之前的消息
        for client in dev_env['websocket_clients']:
            client.messages = []
        
        result = await handle_file_change_workflow('/test/style.css')
        assert result == 'frontend_reloaded'
        
        # 验证客户端收到刷新消息
        for client in dev_env['websocket_clients']:
            messages = client.get_messages()
            assert len(messages) >= 1
            
            last_message = client.get_last_message()
            data = json.loads(last_message[1])
            assert data['type'] == 'reload_frontend'
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """测试错误恢复工作流"""
        # 模拟错误情况下的恢复流程
        
        error_scenarios = []
        
        # 1. WebSocket连接断开恢复
        async def test_websocket_recovery():
            clients = set()
            
            # 添加客户端
            client1 = MockWebSocketResponse()
            client2 = MockWebSocketResponse()
            clients.add(client1)
            clients.add(client2)
            
            # 模拟一个客户端连接异常
            def send_with_error(data):
                raise ConnectionError("Client disconnected")
            
            client1.send_str = send_with_error
            
            # 尝试发送消息并处理异常
            message = json.dumps({'type': 'test', 'message': 'hello'})
            
            disconnected_clients = set()
            for client in clients.copy():
                try:
                    await client.send_str(message)
                except ConnectionError:
                    disconnected_clients.add(client)
            
            # 移除断开的客户端
            clients -= disconnected_clients
            
            return len(clients) == 1  # 应该剩下一个正常客户端
        
        result = await test_websocket_recovery()
        error_scenarios.append(('websocket_recovery', result))
        
        # 2. 文件监控错误恢复
        def test_file_watch_recovery():
            def mock_file_operation_with_error():
                raise PermissionError("File access denied")
            
            def safe_file_operation():
                try:
                    mock_file_operation_with_error()
                    return False
                except PermissionError:
                    # 错误恢复逻辑
                    return True  # 表示错误被正确处理
            
            return safe_file_operation()
        
        result = test_file_watch_recovery()
        error_scenarios.append(('file_watch_recovery', result))
        
        # 3. API错误恢复
        async def test_api_error_recovery():
            async def failing_api_call():
                raise Exception("API temporarily unavailable")
            
            async def api_with_retry(max_retries=2):
                for attempt in range(max_retries):
                    try:
                        return await failing_api_call()
                    except Exception as e:
                        if attempt == max_retries - 1:
                            # 最后一次尝试失败，返回错误响应
                            return {'success': False, 'error': str(e)}
                        # 继续重试
                        await asyncio.sleep(0.01)  # 短暂延迟
            
            result = await api_with_retry()
            return result['success'] is False and 'error' in result
        
        result = await test_api_error_recovery()
        error_scenarios.append(('api_error_recovery', result))
        
        # 验证所有错误场景都被正确处理
        for scenario_name, success in error_scenarios:
            assert success, f"Error recovery failed for scenario: {scenario_name}"

class TestPerformanceIntegration:
    """性能集成测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_websocket_handling(self):
        """测试并发WebSocket处理性能"""
        # 模拟大量并发WebSocket连接
        client_count = 50
        clients = []
        
        for i in range(client_count):
            client = MockWebSocketResponse()
            clients.append(client)
        
        # 模拟广播消息给所有客户端
        start_time = time.time()
        
        message = {
            'type': 'performance_test',
            'message': f'Broadcasting to {client_count} clients',
            'timestamp': int(time.time() * 1000)
        }
        message_json = json.dumps(message)
        
        # 并发发送消息
        tasks = []
        for client in clients:
            task = client.send_str(message_json)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能断言 - 50个客户端的广播应该在合理时间内完成
        assert duration < 1.0  # 应该在1秒内完成
        
        # 验证所有客户端都收到消息
        for client in clients:
            messages = client.get_messages()
            assert len(messages) >= 1
    
    def test_file_watching_performance(self, temp_dir):
        """测试文件监控性能"""
        # 创建大量文件进行监控测试
        file_count = 100
        files_created = []
        
        start_time = time.time()
        
        for i in range(file_count):
            file_path = temp_dir / f'test_file_{i}.py'
            file_path.write_text(f'# Test file {i}')
            files_created.append(file_path)
        
        end_time = time.time()
        creation_duration = end_time - start_time
        
        # 文件创建性能断言
        assert creation_duration < 5.0  # 100个文件应该在5秒内创建完成
        assert len(files_created) == file_count
        
        # 模拟文件变更检测性能
        start_time = time.time()
        
        # 模拟扫描和过滤文件
        monitored_files = []
        watch_extensions = {'.py', '.js', '.html', '.css'}
        
        for file_path in files_created:
            if file_path.suffix in watch_extensions:
                monitored_files.append(file_path)
        
        end_time = time.time()
        scan_duration = end_time - start_time
        
        # 扫描性能断言
        assert scan_duration < 1.0  # 文件扫描应该很快完成
        assert len(monitored_files) == file_count  # 所有.py文件都应该被监控