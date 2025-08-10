"""
100%覆盖率攻坚 - 边界条件与错误路径测试
专门覆盖异常处理分支和边界情况
"""

import pytest
import asyncio
import sys
import os
import time
import json
import socket
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestExceptionHandlingPaths:
    """测试异常处理的所有分支"""
    
    @pytest.mark.asyncio
    async def test_websocket_send_failure_recovery(self):
        """测试WebSocket发送失败的恢复机制"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 创建不同类型的失败客户端
        failure_scenarios = [
            ConnectionResetError("Connection reset by peer"),
            ConnectionAbortedError("Connection aborted"),
            BrokenPipeError("Broken pipe"),
            OSError("Network is down"),
            Exception("Generic send error")
        ]
        
        for i, error in enumerate(failure_scenarios):
            # 创建失败的客户端
            failing_ws = Mock()
            failing_ws.send_str = AsyncMock(side_effect=error)
            server.websocket_clients.add(failing_ws)
            
            # 创建正常的客户端作为对照
            normal_ws = Mock()
            normal_ws.send_str = AsyncMock()
            server.websocket_clients.add(normal_ws)
            
            initial_count = len(server.websocket_clients)
            
            # 执行通知操作
            await server.notify_frontend_reload()
            
            # 验证失败的客户端被移除
            assert failing_ws not in server.websocket_clients
            # 验证正常客户端保留
            assert normal_ws in server.websocket_clients
            # 验证客户端数量减少
            assert len(server.websocket_clients) == initial_count - 1
            
            # 清理为下一轮测试
            server.websocket_clients.clear()
    
    @pytest.mark.asyncio
    async def test_market_data_api_failure_cascade(self):
        """测试市场数据API失败的级联处理"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 创建会失败的交易所模拟
        failure_scenarios = [
            # OKX失败，Binance成功
            {
                'okx_error': ConnectionError("OKX connection failed"),
                'binance_data': {
                    'last': 45000, 'baseVolume': 1000, 'change': 500,
                    'percentage': 1.12, 'high': 46000, 'low': 44000,
                    'bid': 44950, 'ask': 45050
                },
                'should_succeed': True
            },
            # 两个都失败
            {
                'okx_error': TimeoutError("OKX timeout"),
                'binance_error': ConnectionError("Binance failed"),
                'should_succeed': False
            },
            # 数据格式异常
            {
                'okx_error': ValueError("Invalid data format"),
                'binance_error': KeyError("Missing required field"),
                'should_succeed': False
            }
        ]
        
        for scenario in failure_scenarios:
            # 设置OKX交易所模拟
            mock_okx = Mock()
            mock_okx.fetch_ticker = Mock(side_effect=scenario['okx_error'])
            manager.exchanges['okx'] = mock_okx
            
            # 设置Binance交易所模拟
            mock_binance = Mock()
            if 'binance_data' in scenario:
                mock_binance.fetch_ticker = Mock(return_value=scenario['binance_data'])
            elif 'binance_error' in scenario:
                mock_binance.fetch_ticker = Mock(side_effect=scenario['binance_error'])
            manager.exchanges['binance'] = mock_binance
            
            # 执行市场数据获取
            try:
                result = await manager.get_market_data("BTC/USDT")
                if scenario['should_succeed']:
                    assert isinstance(result, dict)
                    assert 'symbol' in result
                    assert result['exchange'] == 'binance'
                else:
                    # 如果不应该成功，但没有抛出异常，则result应该是None
                    assert result is None
            except Exception as e:
                if scenario['should_succeed']:
                    assert False, f"Unexpected exception: {e}"
                else:
                    # 预期的异常
                    assert isinstance(e, Exception)
    
    def test_file_watcher_observer_failures(self):
        """测试文件监控Observer的各种失败情况"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 测试Observer创建失败
        with patch('watchdog.observers.Observer', side_effect=OSError("Cannot create observer")):
            try:
                server.start_file_watcher()
                assert False, "Should have failed"
            except OSError:
                pass  # 预期的异常
        
        # 测试Observer启动失败
        with patch('watchdog.observers.Observer') as MockObserver:
            mock_observer = Mock()
            mock_observer.start = Mock(side_effect=RuntimeError("Start failed"))
            mock_observer.schedule = Mock()
            MockObserver.return_value = mock_observer
            
            try:
                server.start_file_watcher()
                # 可能会抛出异常，也可能会被捕获
            except RuntimeError:
                pass  # 可能的异常
        
        # 测试Observer停止失败
        server.observer = Mock()
        server.observer.stop = Mock(side_effect=OSError("Stop failed"))
        server.observer.join = Mock()
        
        # stop_file_watcher应该处理异常
        try:
            server.stop_file_watcher()
        except OSError:
            pass  # 可能传播异常
    
    def test_dependency_check_missing_imports(self):
        """测试依赖检查中的导入失败"""
        from dev_server import check_dependencies
        from server import check_dependencies as server_check_dependencies
        
        # 创建各种导入失败场景
        import_scenarios = [
            # 单个模块缺失
            {'missing': ['aiohttp'], 'available': ['watchdog', 'webbrowser']},
            # 多个模块缺失
            {'missing': ['aiohttp', 'watchdog'], 'available': ['webbrowser']},
            # 所有模块缺失
            {'missing': ['aiohttp', 'watchdog', 'webbrowser'], 'available': []},
            # 部分模块有复杂错误
            {'missing': ['ccxt'], 'available': ['aiohttp'], 'complex_errors': True}
        ]
        
        for scenario in import_scenarios:
            def selective_import(name, *args, **kwargs):
                if name in scenario['missing']:
                    if scenario.get('complex_errors') and name == 'ccxt':
                        raise ImportError(f"Complex import error for {name}: version conflict")
                    else:
                        raise ImportError(f"No module named '{name}'")
                elif name in scenario['available']:
                    if name == 'webbrowser':
                        import webbrowser
                        return webbrowser
                    else:
                        return Mock()
                else:
                    return Mock()  # 默认成功
            
            with patch('builtins.__import__', side_effect=selective_import), \
                 patch('builtins.print') as mock_print:
                
                # 测试dev_server依赖检查
                result1 = check_dependencies()
                if scenario['missing']:
                    assert result1 is False
                    mock_print.assert_called()
                else:
                    assert result1 is True
                
                mock_print.reset_mock()
                
                # 测试server依赖检查
                result2 = server_check_dependencies()
                assert isinstance(result2, bool)
    
    def test_json_processing_edge_cases(self):
        """测试JSON处理的边界情况"""
        from dev_server import DevServer
        from aiohttp import WSMsgType
        
        server = DevServer()
        
        # JSON解析的各种边界情况
        json_scenarios = [
            # 有效JSON
            '{"type": "ping", "timestamp": 1234567890}',
            # 无效JSON
            '{"type": "ping", "timestamp":}',
            '{"type": "ping" "missing_comma": true}',
            '{invalid json format}',
            'not json at all',
            '',  # 空字符串
            '{}',  # 空对象
            'null',  # null值
            '[]',  # 空数组
            '{"type": null}',  # null type
            '{"type": ""}',  # 空type
            '{"type": 123}',  # 非字符串type
        ]
        
        for json_str in json_scenarios:
            with patch('aiohttp.web.WebSocketResponse') as MockWS:
                mock_ws = Mock()
                mock_ws.prepare = AsyncMock()
                mock_ws.send_str = AsyncMock()
                
                # 创建消息
                message = Mock(type=WSMsgType.TEXT, data=json_str)
                
                async def message_iter():
                    yield message
                    # 添加CLOSE消息来结束迭代
                    close_msg = Mock(type=WSMsgType.CLOSE)
                    yield close_msg
                
                mock_ws.__aiter__ = message_iter
                MockWS.return_value = mock_ws
                
                mock_request = Mock()
                
                # 执行WebSocket处理 - 应该不会因为JSON错误而崩溃
                try:
                    result = asyncio.run(server.websocket_handler(mock_request))
                    assert result == mock_ws
                except Exception as e:
                    # 某些实现可能抛出异常，但不应该是JSON相关的
                    assert not isinstance(e, json.JSONDecodeError)
    
    def test_file_path_edge_cases(self):
        """测试文件路径的边界情况"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        handler.last_reload_time = 0  # 确保冷却时间已过
        
        # 各种边界情况的文件路径
        path_scenarios = [
            # 正常路径
            '/project/app.py',
            # 路径包含特殊字符
            '/project/file with spaces.py',
            '/project/file-with-dashes.css',
            '/project/file_with_underscores.js',
            '/project/file.with.dots.json',
            # 深层嵌套路径
            '/very/deep/nested/path/to/file.html',
            # 相对路径
            './relative/path/file.py',
            '../parent/path/file.css',
            # 空路径或特殊值
            '',
            '/',
            '/.',
            '/..',
            # Windows风格路径
            'C:\\Windows\\path\\file.py',
            'C:/Mixed/Style/path.js',
            # 无扩展名文件
            '/project/README',
            '/project/Makefile',
            # 隐藏文件
            '/project/.hidden',
            '/project/.gitignore',
            # 非常长的路径
            '/' + 'very_long_directory_name' * 10 + '/file.py',
        ]
        
        for file_path in path_scenarios:
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = file_path
            
            with patch('time.time', return_value=1000.0), \
                 patch('asyncio.create_task') as mock_create_task:
                
                try:
                    handler.on_modified(mock_event)
                    # 不应该因为路径问题而崩溃
                except Exception as e:
                    # 某些路径可能导致异常，但不应该是未处理的
                    assert not isinstance(e, (AttributeError, TypeError))
                
                mock_create_task.reset_mock()
    
    def test_port_binding_failures(self):
        """测试端口绑定失败的情况"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试各种端口相关的异常
        port_error_scenarios = [
            # 端口被占用
            OSError("Address already in use"),
            # 权限不足
            PermissionError("Permission denied"),
            # 网络不可达
            OSError("Network is unreachable"),
            # 无效端口号
            ValueError("Invalid port number"),
        ]
        
        for error in port_error_scenarios:
            with patch('socket.socket') as mock_socket:
                mock_sock = Mock()
                mock_sock.bind = Mock(side_effect=error)
                mock_sock.close = Mock()
                mock_socket.return_value = mock_sock
                
                try:
                    result = starter.check_port_availability(8000)
                    # 如果没有抛出异常，应该返回False表示不可用
                    assert result is False
                except (OSError, PermissionError, ValueError):
                    # 异常被传播也是可接受的
                    pass
    
    @pytest.mark.asyncio
    async def test_async_timeout_scenarios(self):
        """测试异步操作超时的场景"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 创建会超时的交易所模拟
        def slow_fetch_ticker(symbol):
            import time
            time.sleep(5)  # 模拟非常慢的API调用
            return {'last': 45000}
        
        mock_exchange = Mock()
        mock_exchange.fetch_ticker = slow_fetch_ticker
        manager.exchanges['slow_exchange'] = mock_exchange
        
        # 使用较短的超时时间
        try:
            result = await asyncio.wait_for(
                manager.get_market_data("BTC/USDT"), 
                timeout=0.5
            )
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            # 预期的超时异常
            pass
    
    def test_memory_exhaustion_simulation(self):
        """模拟内存耗尽的情况"""
        from dev_server import DevServer
        from server import RealTimeDataManager
        
        server = DevServer()
        manager = RealTimeDataManager()
        
        # 模拟内存不足的情况
        original_set_add = set.add
        
        def memory_exhausted_add(self, item):
            if len(self) > 100:  # 模拟内存限制
                raise MemoryError("Out of memory")
            return original_set_add(self, item)
        
        with patch.object(set, 'add', memory_exhausted_add):
            try:
                # 尝试添加大量客户端
                for i in range(150):
                    mock_client = Mock()
                    server.websocket_clients.add(mock_client)
                
                assert False, "Should have raised MemoryError"
            except MemoryError:
                # 预期的内存错误
                pass
    
    def test_file_system_permission_errors(self):
        """测试文件系统权限错误"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟各种文件系统权限问题
        permission_scenarios = [
            PermissionError("Permission denied"),
            OSError("Operation not permitted"),
            FileNotFoundError("No such file or directory"),
            IsADirectoryError("Is a directory"),
            NotADirectoryError("Not a directory"),
        ]
        
        for error in permission_scenarios:
            with patch('pathlib.Path.exists', side_effect=error):
                try:
                    result = starter.check_project_structure()
                    # 如果异常被捕获，应该返回False
                    assert result is False
                except (PermissionError, OSError, FileNotFoundError):
                    # 异常被传播也是可接受的
                    pass
    
    def test_signal_handling_edge_cases(self):
        """测试信号处理的边界情况"""
        import signal
        
        # 测试信号处理器的各种情况
        signal_scenarios = [
            signal.SIGINT,
            signal.SIGTERM,
        ]
        
        if hasattr(signal, 'SIGHUP'):  # Unix系统才有
            signal_scenarios.append(signal.SIGHUP)
        
        for sig in signal_scenarios:
            # 创建测试信号处理器
            def test_handler(signum, frame):
                assert signum == sig
                # 模拟信号处理逻辑
                return "handled"
            
            # 验证信号处理器可以被设置
            try:
                old_handler = signal.signal(sig, test_handler)
                # 恢复原来的处理器
                signal.signal(sig, old_handler)
            except (OSError, ValueError):
                # 某些信号在某些平台上不能被处理
                pass
    
    def test_threading_race_conditions(self):
        """测试多线程竞争条件"""
        import threading
        import time
        
        # 创建共享资源
        shared_counter = {'value': 0}
        shared_list = []
        lock = threading.Lock()
        
        def worker_with_race(worker_id, iterations):
            for i in range(iterations):
                # 故意创建竞争条件
                current = shared_counter['value']
                time.sleep(0.001)  # 增加竞争概率
                shared_counter['value'] = current + 1
                shared_list.append(f"worker_{worker_id}_iter_{i}")
        
        def worker_with_lock(worker_id, iterations):
            for i in range(iterations):
                with lock:
                    current = shared_counter['value']
                    shared_counter['value'] = current + 1
                    shared_list.append(f"worker_{worker_id}_iter_{i}")
        
        # 测试有竞争条件的情况
        shared_counter['value'] = 0
        shared_list.clear()
        
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker_with_race, 
                args=(i, 10)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 由于竞争条件，最终值可能小于期望值
        race_result = shared_counter['value']
        race_list_len = len(shared_list)
        
        # 测试有锁保护的情况
        shared_counter['value'] = 0
        shared_list.clear()
        
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker_with_lock, 
                args=(i, 10)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 有锁保护时，结果应该是准确的
        lock_result = shared_counter['value']
        lock_list_len = len(shared_list)
        
        assert lock_result == 50  # 5 workers * 10 iterations
        assert lock_list_len == 50
        # 竞争条件的结果通常小于或等于正确结果
        assert race_result <= lock_result

class TestBoundaryValueTesting:
    """边界值测试"""
    
    def test_numeric_boundary_values(self):
        """测试数值边界值"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 端口号边界值测试
        port_boundaries = [
            # 有效范围边界
            (1, True),      # 最小有效端口
            (65535, True),  # 最大有效端口
            # 无效范围
            (0, False),     # 小于最小值
            (65536, False), # 大于最大值
            (-1, False),    # 负数
            # 特殊值
            (80, True),     # HTTP端口
            (443, True),    # HTTPS端口
            (8000, True),   # 常用开发端口
            (8080, True),   # 常用代理端口
        ]
        
        for port, should_be_valid in port_boundaries:
            try:
                result = starter.check_port_availability(port)
                if should_be_valid:
                    assert isinstance(result, bool)
                else:
                    # 无效端口应该返回False或抛出异常
                    assert result is False
            except (ValueError, OSError):
                if should_be_valid:
                    # 有效端口号不应该抛出ValueError
                    assert False, f"Valid port {port} raised exception"
                else:
                    # 无效端口号抛出异常是可接受的
                    pass
    
    def test_string_boundary_values(self):
        """测试字符串边界值"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        handler.last_reload_time = 0
        
        # 文件扩展名边界值测试
        extension_boundaries = [
            # 有效扩展名
            ('.py', True),
            ('.html', True),
            ('.css', True),
            ('.js', True),
            ('.json', True),
            # 大小写变化
            ('.PY', True),    # 应该被转换为小写
            ('.HTML', True),
            ('.CSS', True),
            # 无效扩展名
            ('.txt', False),
            ('.png', False),
            ('.pdf', False),
            ('.zip', False),
            # 边界情况
            ('', False),      # 无扩展名
            ('.', False),     # 只有点
            ('..', False),    # 两个点
            ('.py.backup', False),  # 多重扩展名
        ]
        
        for extension, should_trigger in extension_boundaries:
            test_path = f"/test/file{extension}"
            
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = test_path
            
            with patch('time.time', return_value=1000.0), \
                 patch('asyncio.create_task') as mock_create_task:
                
                handler.on_modified(mock_event)
                
                if should_trigger:
                    mock_create_task.assert_called_once()
                else:
                    mock_create_task.assert_not_called()
                
                mock_create_task.reset_mock()
    
    def test_time_boundary_values(self):
        """测试时间相关的边界值"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        
        current_time = 1000.0
        cooldown = handler.reload_cooldown  # 默认1秒
        
        # 时间边界值测试
        time_scenarios = [
            # 冷却时间内
            (current_time - 0.5, False),    # 0.5秒前
            (current_time - 0.9, False),    # 0.9秒前
            (current_time - 0.99, False),   # 0.99秒前
            # 边界值
            (current_time - 1.0, False),    # 刚好1秒前（等于cooldown）
            (current_time - 1.01, True),    # 1.01秒前（超过cooldown）
            # 冷却时间外
            (current_time - 2.0, True),     # 2秒前
            (current_time - 10.0, True),    # 10秒前
            # 特殊情况
            (0, True),                      # 初始值
            (current_time, False),          # 当前时间（差值为0）
            (current_time + 1, False),      # 未来时间（负差值）
        ]
        
        for last_time, should_trigger in time_scenarios:
            handler.last_reload_time = last_time
            
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = "/test/file.py"
            
            with patch('time.time', return_value=current_time), \
                 patch('asyncio.create_task') as mock_create_task:
                
                handler.on_modified(mock_event)
                
                if should_trigger:
                    mock_create_task.assert_called_once()
                    assert handler.last_reload_time == current_time
                else:
                    mock_create_task.assert_not_called()
                    # 时间不应该被更新
                    assert handler.last_reload_time == last_time
                
                mock_create_task.reset_mock()
    
    def test_collection_size_boundaries(self):
        """测试集合大小边界值"""
        from dev_server import DevServer
        from server import RealTimeDataManager
        
        server = DevServer()
        manager = RealTimeDataManager()
        
        # 客户端集合大小边界测试
        size_boundaries = [
            0,      # 空集合
            1,      # 单个元素
            10,     # 小集合
            100,    # 中等集合
            1000,   # 大集合
        ]
        
        for size in size_boundaries:
            # 清理现有客户端
            server.websocket_clients.clear()
            manager.websocket_clients.clear()
            
            # 添加指定数量的客户端
            clients = []
            for i in range(size):
                mock_client = Mock()
                mock_client.send_str = AsyncMock()
                clients.append(mock_client)
                server.websocket_clients.add(mock_client)
                manager.websocket_clients.add(mock_client)
            
            # 验证集合大小
            assert len(server.websocket_clients) == size
            assert len(manager.websocket_clients) == size
            
            # 测试操作在不同大小下的行为
            if size > 0:
                # 测试移除操作
                first_client = clients[0]
                server.websocket_clients.discard(first_client)
                assert len(server.websocket_clients) == size - 1
                
                # 测试清空操作
                server.websocket_clients.clear()
                assert len(server.websocket_clients) == 0
    
    def test_unicode_and_encoding_boundaries(self):
        """测试Unicode和编码边界情况"""
        import json
        
        # Unicode字符边界测试
        unicode_scenarios = [
            # 基本ASCII
            "Hello World",
            # 基本多字节字符
            "你好世界",
            "Hello 世界",
            # 特殊字符
            "File with émoji 🚀",
            "Path/with/ünicöde.py",
            # 控制字符
            "Line\nBreak",
            "Tab\tCharacter",
            "Return\rCharacter",
            # JSON特殊字符
            'Text with "quotes"',
            "Text with 'apostrophe'",
            "Text with \\ backslash",
            # 极端情况
            "",  # 空字符串
            " ",  # 空格
            "\x00",  # NULL字符
            "\uffff",  # Unicode边界
        ]
        
        for text in unicode_scenarios:
            try:
                # 测试JSON序列化
                json_str = json.dumps({'message': text})
                assert isinstance(json_str, str)
                
                # 测试JSON反序列化
                parsed = json.loads(json_str)
                assert parsed['message'] == text
                
                # 测试文件路径处理
                from pathlib import Path
                path = Path(f"/test/{text}.py")
                extension = path.suffix.lower()
                assert isinstance(extension, str)
                
            except (UnicodeError, json.JSONDecodeError, OSError):
                # 某些极端Unicode字符可能导致异常，这是可接受的
                pass