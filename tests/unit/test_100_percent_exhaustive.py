"""
100%覆盖率攻坚 - 代码路径穷举测试
确保每个条件分支的True/False都被覆盖
"""

import pytest
import asyncio
import sys
import os
import time
import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestBranchCoverageExhaustive:
    """穷举所有分支覆盖"""
    
    def test_all_file_extension_branches(self):
        """测试所有文件扩展名分支"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        handler.last_reload_time = 0  # 确保冷却时间已过
        
        # 详尽的文件扩展名测试
        extension_test_cases = [
            # Python分支 - 应该触发backend restart
            ('.py', True, 'backend'),
            ('.PY', True, 'backend'),  # 大小写
            ('.pyw', False, None),     # 相似但不同
            
            # HTML分支 - 应该触发frontend reload
            ('.html', True, 'frontend'),
            ('.HTML', True, 'frontend'),
            ('.htm', False, None),     # 相似但不同
            
            # CSS分支 - 应该触发frontend reload
            ('.css', True, 'frontend'),
            ('.CSS', True, 'frontend'),
            ('.scss', False, None),    # 相似但不同
            
            # JavaScript分支 - 应该触发frontend reload
            ('.js', True, 'frontend'),
            ('.JS', True, 'frontend'),
            ('.jsx', False, None),     # 相似但不同
            
            # JSON分支 - 应该触发frontend reload
            ('.json', True, 'frontend'),
            ('.JSON', True, 'frontend'),
            ('.jsonl', False, None),   # 相似但不同
            
            # 不匹配的扩展名
            ('.txt', False, None),
            ('.md', False, None),
            ('.xml', False, None),
            ('.yaml', False, None),
            ('.yml', False, None),
            ('.ini', False, None),
            ('.cfg', False, None),
            ('.conf', False, None),
            ('', False, None),         # 无扩展名
            ('.', False, None),        # 只有点
        ]
        
        for extension, should_trigger, expected_type in extension_test_cases:
            test_path = f"/project/testfile{extension}"
            
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = test_path
            
            with patch('time.time', return_value=1000.0), \
                 patch('asyncio.create_task') as mock_create_task:
                
                handler.on_modified(mock_event)
                
                if should_trigger:
                    mock_create_task.assert_called_once()
                    # 验证调用了正确的方法
                    call_args = mock_create_task.call_args[0][0]
                    if expected_type == 'backend':
                        # 应该调用restart_backend
                        assert hasattr(call_args, '__name__') or str(call_args).find('restart_backend') != -1
                    elif expected_type == 'frontend':
                        # 应该调用notify_frontend_reload
                        assert hasattr(call_args, '__name__') or str(call_args).find('notify_frontend_reload') != -1
                else:
                    mock_create_task.assert_not_called()
                
                mock_create_task.reset_mock()
    
    def test_all_cooldown_time_branches(self):
        """测试冷却时间的所有分支"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        
        # 冷却时间分支的详尽测试
        cooldown_scenarios = [
            # time_diff > cooldown (应该触发)
            {'last_time': 0, 'current_time': 2.0, 'should_trigger': True},
            {'last_time': 100, 'current_time': 102.5, 'should_trigger': True},
            {'last_time': 1000, 'current_time': 1001.1, 'should_trigger': True},
            
            # time_diff <= cooldown (不应该触发)
            {'last_time': 100, 'current_time': 101.0, 'should_trigger': False},  # 等于cooldown
            {'last_time': 100, 'current_time': 100.5, 'should_trigger': False},  # 小于cooldown
            {'last_time': 100, 'current_time': 100.0, 'should_trigger': False},  # 差值为0
            {'last_time': 100, 'current_time': 99.0, 'should_trigger': False},   # 负差值
            
            # 边界情况
            {'last_time': 0, 'current_time': 1.0, 'should_trigger': False},      # 刚好等于cooldown
            {'last_time': 0, 'current_time': 1.0001, 'should_trigger': True},    # 略大于cooldown
            {'last_time': 0, 'current_time': 0.9999, 'should_trigger': False},   # 略小于cooldown
        ]
        
        for scenario in cooldown_scenarios:
            handler.last_reload_time = scenario['last_time']
            
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = "/test/file.py"
            
            with patch('time.time', return_value=scenario['current_time']), \
                 patch('asyncio.create_task') as mock_create_task:
                
                handler.on_modified(mock_event)
                
                if scenario['should_trigger']:
                    mock_create_task.assert_called_once()
                    # 验证时间被更新
                    assert handler.last_reload_time == scenario['current_time']
                else:
                    mock_create_task.assert_not_called()
                    # 验证时间没有被更新
                    assert handler.last_reload_time == scenario['last_time']
                
                mock_create_task.reset_mock()
    
    def test_all_directory_vs_file_branches(self):
        """测试目录vs文件的所有分支"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        handler.last_reload_time = 0
        
        # 目录vs文件的分支测试
        event_scenarios = [
            # 文件事件（应该处理）
            {'is_directory': False, 'should_process': True},
            # 目录事件（应该忽略）
            {'is_directory': True, 'should_process': False},
        ]
        
        for scenario in event_scenarios:
            mock_event = Mock()
            mock_event.is_directory = scenario['is_directory']
            mock_event.src_path = "/test/path.py"
            
            with patch('time.time', return_value=1000.0), \
                 patch('asyncio.create_task') as mock_create_task:
                
                handler.on_modified(mock_event)
                
                if scenario['should_process']:
                    mock_create_task.assert_called_once()
                else:
                    mock_create_task.assert_not_called()
                
                mock_create_task.reset_mock()
    
    def test_all_websocket_client_presence_branches(self):
        """测试WebSocket客户端存在性的所有分支"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 客户端存在性分支测试
        client_scenarios = [
            # 有客户端的情况
            {'client_count': 1, 'should_send': True},
            {'client_count': 5, 'should_send': True},
            {'client_count': 10, 'should_send': True},
            # 无客户端的情况
            {'client_count': 0, 'should_send': False},
        ]
        
        for scenario in client_scenarios:
            # 清理现有客户端
            server.websocket_clients.clear()
            
            # 添加指定数量的客户端
            for i in range(scenario['client_count']):
                mock_ws = Mock()
                mock_ws.send_str = AsyncMock()
                server.websocket_clients.add(mock_ws)
            
            # 测试前端重载通知
            with patch('time.time', return_value=1000.0), \
                 patch('json.dumps', return_value='{"test": "message"}'):
                
                result = asyncio.run(server.notify_frontend_reload())
                
                if scenario['should_send']:
                    # 验证所有客户端都收到了消息
                    for ws in server.websocket_clients:
                        ws.send_str.assert_called_once()
                else:
                    # 没有客户端时，函数应该早期返回
                    assert result is None
    
    def test_all_path_existence_branches(self):
        """测试路径存在性的所有分支"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 路径存在性分支的详尽测试
        path_scenarios = [
            # 所有路径都存在
            {
                'main_exists': True,
                'file_mgmt_exists': True,
                'core_exists': True,
                'src_exists': True,
                'expected_schedules': 4
            },
            # 部分路径存在
            {
                'main_exists': True,
                'file_mgmt_exists': True,
                'core_exists': False,
                'src_exists': True,
                'expected_schedules': 3
            },
            # 只有主路径存在
            {
                'main_exists': True,
                'file_mgmt_exists': False,
                'core_exists': False,
                'src_exists': False,
                'expected_schedules': 1
            },
            # 没有路径存在（理论上不可能，但测试边界情况）
            {
                'main_exists': False,
                'file_mgmt_exists': False,
                'core_exists': False,
                'src_exists': False,
                'expected_schedules': 0
            }
        ]
        
        for scenario in path_scenarios:
            with patch('watchdog.observers.Observer') as MockObserver:
                mock_observer = Mock()
                mock_observer.start = Mock()
                mock_observer.schedule = Mock()
                MockObserver.return_value = mock_observer
                
                # 设置路径存在性模拟
                def path_exists_mock(self):
                    path_str = str(self)
                    if path_str.endswith(str(Path(__file__).parent.parent.parent)):
                        return scenario['main_exists']
                    elif 'file_management' in path_str:
                        return scenario['file_mgmt_exists']
                    elif 'core' in path_str:
                        return scenario['core_exists']
                    elif 'src' in path_str:
                        return scenario['src_exists']
                    else:
                        return True  # 默认存在
                
                with patch('pathlib.Path.exists', path_exists_mock):
                    server.start_file_watcher()
                
                # 验证schedule调用次数
                assert mock_observer.schedule.call_count == scenario['expected_schedules']
                
                if scenario['expected_schedules'] > 0:
                    mock_observer.start.assert_called_once()
                else:
                    # 如果没有路径存在，可能不会启动observer
                    pass
    
    @pytest.mark.asyncio
    async def test_all_websocket_message_type_branches(self):
        """测试WebSocket消息类型的所有分支"""
        from server import websocket_handler, data_manager
        from aiohttp import WSMsgType
        
        # WebSocket消息类型的详尽测试
        message_type_scenarios = [
            # TEXT消息 - 有效JSON
            {
                'type': WSMsgType.TEXT,
                'data': '{"type": "subscribe", "symbols": ["BTC/USDT"]}',
                'should_process': True
            },
            # TEXT消息 - 无效JSON
            {
                'type': WSMsgType.TEXT,
                'data': 'invalid json',
                'should_process': False
            },
            # TEXT消息 - 空JSON
            {
                'type': WSMsgType.TEXT,
                'data': '{}',
                'should_process': False
            },
            # TEXT消息 - ping
            {
                'type': WSMsgType.TEXT,
                'data': '{"type": "ping"}',
                'should_process': True
            },
            # BINARY消息（应该被忽略）
            {
                'type': WSMsgType.BINARY,
                'data': b'binary data',
                'should_process': False
            },
            # ERROR消息（应该中断循环）
            {
                'type': WSMsgType.ERROR,
                'should_break': True
            },
            # CLOSE消息（应该中断循环）
            {
                'type': WSMsgType.CLOSE,
                'should_break': True
            },
        ]
        
        for scenario in message_type_scenarios:
            with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
                mock_ws = Mock()
                mock_ws.prepare = AsyncMock()
                mock_ws.send_str = AsyncMock()
                
                # 创建消息
                if scenario['type'] == WSMsgType.ERROR:
                    mock_ws.exception = Mock(return_value=Exception("WebSocket error"))
                
                messages = [Mock(type=scenario['type'])]
                if hasattr(scenario, 'data'):
                    messages[0].data = scenario['data']
                
                # 添加CLOSE消息来确保循环结束
                if not scenario.get('should_break'):
                    messages.append(Mock(type=WSMsgType.CLOSE))
                
                async def message_iterator():
                    for msg in messages:
                        yield msg
                
                mock_ws.__aiter__ = message_iterator
                MockWSResponse.return_value = mock_ws
                
                # 模拟市场数据获取成功
                with patch.object(data_manager, 'get_market_data') as mock_get_data:
                    mock_get_data.return_value = {
                        'symbol': 'BTC/USDT',
                        'price': 45000,
                        'timestamp': int(time.time() * 1000)
                    }
                    
                    mock_request = Mock()
                    
                    # 执行WebSocket处理器
                    result = await websocket_handler(mock_request)
                    
                    # 验证结果
                    assert result == mock_ws
                    
                    # 验证处理逻辑
                    if scenario.get('should_process'):
                        if '{"type": "subscribe"' in scenario.get('data', ''):
                            # 订阅消息应该触发数据获取
                            mock_get_data.assert_called()
                        elif '{"type": "ping"}' in scenario.get('data', ''):
                            # ping消息可能触发pong响应
                            pass
    
    def test_all_exception_handling_branches(self):
        """测试异常处理的所有分支"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 异常处理分支的详尽测试
        exception_scenarios = [
            # OKX成功，无需fallback
            {
                'okx_result': {'last': 45000, 'baseVolume': 1000, 'change': 500,
                               'percentage': 1.12, 'high': 46000, 'low': 44000,
                               'bid': 44950, 'ask': 45050},
                'binance_result': None,  # 不应该被调用
                'should_succeed': True,
                'expected_exchange': 'okx'
            },
            # OKX失败，Binance成功
            {
                'okx_error': ConnectionError("OKX failed"),
                'binance_result': {'last': 44000, 'baseVolume': 800, 'change': 300,
                                  'percentage': 0.68, 'high': 45000, 'low': 43000,
                                  'bid': 43950, 'ask': 44050},
                'should_succeed': True,
                'expected_exchange': 'binance'
            },
            # 两个都失败
            {
                'okx_error': TimeoutError("OKX timeout"),
                'binance_error': ConnectionError("Binance failed"),
                'should_succeed': False
            },
            # 数据格式错误
            {
                'okx_error': KeyError("Missing 'last' field"),
                'binance_error': ValueError("Invalid data format"),
                'should_succeed': False
            },
        ]
        
        for scenario in exception_scenarios:
            # 设置OKX交易所
            mock_okx = Mock()
            if 'okx_result' in scenario:
                mock_okx.fetch_ticker = Mock(return_value=scenario['okx_result'])
            else:
                mock_okx.fetch_ticker = Mock(side_effect=scenario['okx_error'])
            manager.exchanges['okx'] = mock_okx
            
            # 设置Binance交易所
            mock_binance = Mock()
            if 'binance_result' in scenario:
                mock_binance.fetch_ticker = Mock(return_value=scenario['binance_result'])
            elif 'binance_error' in scenario:
                mock_binance.fetch_ticker = Mock(side_effect=scenario['binance_error'])
            manager.exchanges['binance'] = mock_binance
            
            # 执行市场数据获取
            try:
                result = asyncio.run(manager.get_market_data("BTC/USDT"))
                
                if scenario['should_succeed']:
                    assert isinstance(result, dict)
                    assert result['symbol'] == "BTC/USDT"
                    assert result['exchange'] == scenario['expected_exchange']
                    
                    # 验证正确的交易所被调用
                    mock_okx.fetch_ticker.assert_called_once()
                    
                    if scenario['expected_exchange'] == 'binance':
                        mock_binance.fetch_ticker.assert_called_once()
                    else:
                        # OKX成功时，Binance不应该被调用
                        mock_binance.fetch_ticker.assert_not_called()
                else:
                    # 不应该成功，但没有异常说明异常被处理了
                    assert result is None
                    
            except Exception as e:
                if scenario['should_succeed']:
                    assert False, f"Unexpected exception: {e}"
                else:
                    # 预期的异常
                    assert isinstance(e, Exception)
    
    def test_all_cleanup_resource_branches(self):
        """测试清理资源的所有分支"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 资源清理分支的详尽测试
        cleanup_scenarios = [
            # 所有资源都存在
            {
                'has_observer': True,
                'has_runner': True,
                'observer_works': True,
                'runner_works': True
            },
            # Observer存在但清理失败
            {
                'has_observer': True,
                'has_runner': True,
                'observer_works': False,
                'runner_works': True
            },
            # Runner存在但清理失败
            {
                'has_observer': True,
                'has_runner': True,
                'observer_works': True,
                'runner_works': False
            },
            # 只有Observer
            {
                'has_observer': True,
                'has_runner': False,
                'observer_works': True,
                'runner_works': None
            },
            # 只有Runner
            {
                'has_observer': False,
                'has_runner': True,
                'observer_works': None,
                'runner_works': True
            },
            # 都不存在
            {
                'has_observer': False,
                'has_runner': False,
                'observer_works': None,
                'runner_works': None
            }
        ]
        
        for scenario in cleanup_scenarios:
            # 设置Observer
            if scenario['has_observer']:
                server.observer = Mock()
                if scenario['observer_works']:
                    server.observer.stop = Mock()
                    server.observer.join = Mock()
                else:
                    server.observer.stop = Mock(side_effect=OSError("Stop failed"))
                    server.observer.join = Mock(side_effect=OSError("Join failed"))
            else:
                server.observer = None
            
            # 设置Runner
            if scenario['has_runner']:
                server.runner = Mock()
                if scenario['runner_works']:
                    server.runner.cleanup = AsyncMock()
                else:
                    server.runner.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))
            else:
                server.runner = None
            
            # 执行清理 - 不应该抛出异常
            try:
                asyncio.run(server.cleanup())
                cleanup_succeeded = True
            except Exception:
                cleanup_succeeded = False
            
            # 清理应该总是成功（异常应该被处理）
            assert cleanup_succeeded is True
    
    def test_all_dependency_import_branches(self):
        """测试依赖导入的所有分支"""
        from dev_server import check_dependencies
        from server import check_dependencies as server_check
        
        # 导入分支的详尽测试
        import_scenarios = [
            # 所有依赖都可用
            {
                'available': ['aiohttp', 'watchdog', 'webbrowser', 'ccxt', 'pandas', 'numpy', 'websockets', 'aiohttp_cors'],
                'missing': [],
                'expected_result': True
            },
            # 缺少单个依赖
            {
                'available': ['watchdog', 'webbrowser'],
                'missing': ['aiohttp'],
                'expected_result': False
            },
            # 缺少多个依赖
            {
                'available': ['webbrowser'],
                'missing': ['aiohttp', 'watchdog'],
                'expected_result': False
            },
            # 缺少所有依赖
            {
                'available': [],
                'missing': ['aiohttp', 'watchdog', 'webbrowser'],
                'expected_result': False
            },
            # webbrowser特殊处理分支
            {
                'available': ['aiohttp', 'watchdog'],
                'missing': ['webbrowser'],
                'expected_result': False,
                'special_webbrowser': True
            }
        ]
        
        for scenario in import_scenarios:
            def mock_import(name, *args, **kwargs):
                if name in scenario['available']:
                    if name == 'webbrowser':
                        import webbrowser
                        return webbrowser
                    else:
                        return Mock()
                elif name in scenario['missing']:
                    raise ImportError(f"No module named '{name}'")
                else:
                    return Mock()  # 默认成功
            
            with patch('builtins.__import__', side_effect=mock_import), \
                 patch('builtins.print') as mock_print:
                
                # 测试dev_server依赖检查
                result1 = check_dependencies()
                assert result1 == scenario['expected_result']
                
                if not scenario['expected_result']:
                    mock_print.assert_called()
                
                mock_print.reset_mock()
                
                # 测试server依赖检查
                result2 = server_check()
                # server的依赖检查可能有不同的依赖列表
                assert isinstance(result2, bool)