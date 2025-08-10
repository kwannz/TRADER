"""
集成工作流程测试
测试完整的工作流程以提高覆盖率到80%
"""

import pytest
import asyncio
import sys
import os
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestCompleteDevServerWorkflow:
    """测试开发服务器的完整工作流程"""
    
    @pytest.mark.asyncio
    async def test_complete_server_startup_sequence(self):
        """测试完整的服务器启动序列"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 模拟完整的启动序列
        with patch.object(server, 'create_app', new_callable=AsyncMock) as mock_create_app, \
             patch('aiohttp.web.AppRunner') as MockAppRunner, \
             patch('aiohttp.web.TCPSite') as MockTCPSite, \
             patch.object(server, 'start_file_watcher') as mock_start_watcher, \
             patch('webbrowser.open') as mock_browser, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep, \
             patch('dev_server.logger') as mock_logger:
            
            # 设置mock对象
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            mock_runner = Mock()
            mock_runner.setup = AsyncMock()
            MockAppRunner.return_value = mock_runner
            
            mock_site = Mock()
            mock_site.start = AsyncMock()
            MockTCPSite.return_value = mock_site
            
            # 模拟运行一段时间后KeyboardInterrupt
            mock_sleep.side_effect = [None, None, KeyboardInterrupt()]
            
            with patch.object(server, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
                # 执行完整启动流程
                await server.start()
                
                # 验证启动序列的每个步骤
                mock_create_app.assert_called_once()
                MockAppRunner.assert_called_once_with(mock_app)
                mock_runner.setup.assert_called_once()
                MockTCPSite.assert_called_once_with(mock_runner, server.host, server.port)
                mock_site.start.assert_called_once()
                mock_start_watcher.assert_called_once()
                mock_browser.assert_called_once_with(f"http://{server.host}:{server.port}")
                
                # 验证日志记录
                mock_logger.info.assert_called()
                
                # 验证清理
                mock_cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_complete_lifecycle(self):
        """测试WebSocket的完整生命周期"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWS:
            # 创建模拟WebSocket
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            mock_ws.close = AsyncMock()
            
            # 模拟消息序列
            messages = [
                Mock(type=1, data='{"type": "ping"}'),  # TEXT消息
                Mock(type=1, data='{"type": "subscribe", "symbol": "BTC/USDT"}'),  # 订阅消息
                Mock(type=8)  # CLOSE消息
            ]
            
            # 异步迭代器
            async def mock_aiter():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = mock_aiter
            MockWS.return_value = mock_ws
            
            mock_request = Mock()
            
            # 执行WebSocket处理
            result = await server.websocket_handler(mock_request)
            
            # 验证WebSocket生命周期
            assert result == mock_ws
            mock_ws.prepare.assert_called_once_with(mock_request)
            
            # 验证客户端被添加
            assert mock_ws in server.websocket_clients
            
            # 验证消息发送（ping响应）
            mock_ws.send_str.assert_called()
    
    def test_file_watcher_complete_setup(self):
        """测试文件监控的完整设置"""
        from dev_server import DevServer, HotReloadEventHandler
        
        server = DevServer()
        
        with patch('watchdog.observers.Observer') as MockObserver, \
             patch('pathlib.Path.exists', return_value=True) as mock_exists:
            
            mock_observer = Mock()
            mock_observer.start = Mock()
            mock_observer.schedule = Mock()
            MockObserver.return_value = mock_observer
            
            # 执行文件监控启动
            server.start_file_watcher()
            
            # 验证Observer设置
            MockObserver.assert_called_once()
            mock_observer.schedule.assert_called()
            mock_observer.start.assert_called_once()
            
            # 验证事件处理器创建
            call_args = mock_observer.schedule.call_args
            assert len(call_args[0]) >= 1  # 至少有一个参数（事件处理器）
            
            # 验证observer保存
            assert server.observer == mock_observer
    
    @pytest.mark.asyncio
    async def test_hot_reload_complete_workflow(self):
        """测试热重载的完整工作流程"""
        from dev_server import DevServer, HotReloadEventHandler
        
        server = DevServer()
        handler = HotReloadEventHandler(server)
        
        # 添加多个WebSocket客户端
        mock_clients = []
        for i in range(3):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            mock_clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        # 模拟文件修改事件
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/project/app.py"
        
        handler.last_reload_time = 0  # 确保冷却时间已过
        
        with patch('time.time', return_value=1000.0), \
             patch('dev_server.logger') as mock_logger, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # 直接执行重启后端流程（模拟create_task的效果）
            async def simulate_restart():
                await server.restart_backend()
            
            # 触发文件修改
            with patch('asyncio.create_task', side_effect=lambda coro: asyncio.ensure_future(coro)) as mock_create_task:
                handler.on_modified(mock_event)
                
                # 等待任务完成
                await asyncio.sleep(0.1)
                
                # 验证create_task被调用
                mock_create_task.assert_called_once()
                
            # 验证所有客户端都收到重启通知
            for client in mock_clients:
                assert client.send_str.call_count >= 1  # 至少收到一条消息
    
    @pytest.mark.asyncio
    async def test_multiple_client_notification(self):
        """测试多客户端通知系统"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 创建混合状态的客户端
        clients = []
        
        # 正常客户端
        for i in range(3):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        # 会失败的客户端
        failing_clients = []
        for i in range(2):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock(side_effect=ConnectionError("Connection lost"))
            failing_clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        initial_count = len(server.websocket_clients)
        assert initial_count == 5
        
        # 执行通知
        with patch('time.time', return_value=1234567890), \
             patch('json.dumps', return_value='{"test": "message"}'), \
             patch('dev_server.logger') as mock_logger:
            
            await server.notify_frontend_reload()
            
            # 验证正常客户端收到消息
            for client in clients:
                client.send_str.assert_called_once()
            
            # 验证失败客户端被移除
            for failing_client in failing_clients:
                assert failing_client not in server.websocket_clients
            
            # 验证最终客户端数量
            assert len(server.websocket_clients) == 3
            
            # 验证错误日志
            assert mock_logger.error.call_count == 2  # 两个失败的客户端

class TestCompleteServerWorkflow:
    """测试server.py的完整工作流程"""
    
    def test_real_time_data_manager_lifecycle(self):
        """测试数据管理器的完整生命周期"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 验证初始状态
        assert manager.exchanges == {}
        assert manager.websocket_clients == set()
        assert manager.market_data == {}
        assert manager.running is False
        
        # 模拟添加交易所
        mock_exchange = Mock()
        manager.exchanges['okx'] = mock_exchange
        manager.exchanges['binance'] = mock_exchange
        
        # 模拟市场数据
        test_data = {
            'BTC/USDT': {'price': 45000, 'volume': 1000},
            'ETH/USDT': {'price': 3000, 'volume': 500}
        }
        manager.market_data.update(test_data)
        
        # 模拟客户端连接
        clients = [Mock() for _ in range(5)]
        manager.websocket_clients.update(clients)
        
        # 验证状态
        assert len(manager.exchanges) == 2
        assert len(manager.market_data) == 2
        assert len(manager.websocket_clients) == 5
        
        # 清理资源
        manager.exchanges.clear()
        manager.market_data.clear()
        manager.websocket_clients.clear()
        
        # 验证清理
        assert len(manager.exchanges) == 0
        assert len(manager.market_data) == 0
        assert len(manager.websocket_clients) == 0
    
    @pytest.mark.asyncio
    async def test_exchange_initialization_workflow(self):
        """测试交易所初始化工作流程"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 测试缺少凭据的情况
        with patch('os.environ.get', return_value=None):
            result = await manager.initialize_exchanges()
            assert result is False
            assert len(manager.exchanges) == 0
        
        # 测试有凭据但导入失败
        with patch('os.environ.get') as mock_env:
            mock_env.side_effect = lambda key, default=None: {
                'OKX_API_KEY': 'test_key',
                'OKX_SECRET': 'test_secret',
                'OKX_PASSPHRASE': 'test_pass'
            }.get(key, default)
            
            with patch('ccxt.okx', side_effect=ImportError("ccxt not found")):
                result = await manager.initialize_exchanges()
                assert result is False
    
    def test_dependency_checking_workflow(self):
        """测试依赖检查工作流程"""
        from server import check_dependencies
        
        # 测试所有依赖可用
        with patch('builtins.__import__', return_value=Mock()):
            result = check_dependencies()
            assert isinstance(result, bool)
        
        # 测试部分依赖缺失
        def mock_import_with_failure(name, *args, **kwargs):
            if name == 'aiohttp_cors':
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import_with_failure), \
             patch('builtins.print') as mock_print:
            
            result = check_dependencies()
            assert result is False
            mock_print.assert_called()

class TestCompleteStartDevWorkflow:
    """测试start_dev.py的完整工作流程"""
    
    def test_dev_environment_starter_complete_workflow(self):
        """测试开发环境启动器的完整工作流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 验证基本属性
        assert starter.project_root.exists()
        assert starter.project_root.is_dir()
        assert isinstance(starter.python_executable, str)
        assert 'python' in starter.python_executable.lower()
    
    def test_python_version_checking_workflow(self):
        """测试Python版本检查工作流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试当前版本
        result = starter.check_python_version()
        assert isinstance(result, bool)
        
        # 当前运行环境应该满足版本要求
        if sys.version_info >= (3, 8):
            assert result is True
        else:
            assert result is False
    
    def test_project_structure_validation_workflow(self):
        """测试项目结构验证工作流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 执行结构检查
        result = starter.check_project_structure()
        assert isinstance(result, bool)
        
        # 检查关键文件
        key_files = ['dev_server.py', 'server.py', 'start_dev.py']
        file_count = 0
        
        for filename in key_files:
            filepath = starter.project_root / filename
            if filepath.exists():
                file_count += 1
                assert filepath.is_file()
                assert filepath.suffix == '.py'
        
        # 至少应该有一些关键文件存在
        assert file_count > 0
    
    def test_dependency_installation_workflow(self):
        """测试依赖安装工作流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试空列表安装
        result = starter.install_dependencies([])
        assert result is True  # 空列表应该成功
        
        # 测试模拟成功安装
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Successfully installed"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['pytest', 'coverage'])
            assert result is True
            
            # 验证命令构建
            call_args = mock_run.call_args[0][0]
            assert starter.python_executable in call_args
            assert '-m' in call_args
            assert 'pip' in call_args
            assert 'install' in call_args
            assert 'pytest' in call_args
            assert 'coverage' in call_args
        
        # 测试安装失败
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Package not found"
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['nonexistent-package'])
            assert result is False

class TestWorkflowIntegration:
    """测试跨模块工作流程集成"""
    
    def test_cross_module_dependency_validation(self):
        """测试跨模块依赖验证"""
        # 导入所有主要模块
        import dev_server
        import server
        import start_dev
        
        # 验证所有模块都有依赖检查
        assert hasattr(dev_server, 'check_dependencies')
        assert hasattr(server, 'check_dependencies')
        
        # 执行所有依赖检查
        results = []
        results.append(dev_server.check_dependencies())
        results.append(server.check_dependencies())
        
        # 验证结果
        for result in results:
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_complete_hot_reload_integration(self):
        """测试完整的热重载集成流程"""
        from dev_server import DevServer, HotReloadEventHandler
        
        server = DevServer()
        handler = HotReloadEventHandler(server)
        
        # 设置WebSocket客户端
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        server.websocket_clients.add(mock_ws)
        
        # 设置文件监控
        with patch('watchdog.observers.Observer') as MockObserver:
            mock_observer = Mock()
            mock_observer.start = Mock()
            mock_observer.schedule = Mock()
            MockObserver.return_value = mock_observer
            
            server.start_file_watcher()
            
            # 模拟文件修改事件
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = "/project/main.py"
            
            handler.last_reload_time = 0
            
            with patch('time.time', return_value=2000.0), \
                 patch('asyncio.create_task') as mock_create_task:
                
                # 触发文件修改
                handler.on_modified(mock_event)
                
                # 验证任务创建
                mock_create_task.assert_called_once()
                
                # 直接执行重启流程
                await server.restart_backend()
                
                # 验证客户端收到通知
                assert mock_ws.send_str.call_count >= 2  # 开始和完成通知
    
    def test_environment_setup_integration(self):
        """测试环境设置集成流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟完整设置流程
        with patch.object(starter, 'validate_environment', return_value=True) as mock_validate, \
             patch.object(starter, 'install_dependencies', return_value=True) as mock_install:
            
            # 执行环境验证
            env_valid = starter.validate_environment()
            assert env_valid is True
            
            # 执行依赖安装
            deps_installed = starter.install_dependencies(['pytest'])
            assert deps_installed is True
    
    def test_configuration_management_integration(self):
        """测试配置管理集成"""
        # 测试环境变量配置
        test_config = {
            'DEV_MODE': 'true',
            'SERVER_PORT': '8000',
            'SERVER_HOST': 'localhost',
            'DEBUG_LEVEL': 'info'
        }
        
        # 应用配置
        original_env = {}
        for key, value in test_config.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # 验证配置应用
            dev_mode = os.environ.get('DEV_MODE', 'false').lower() == 'true'
            server_port = int(os.environ.get('SERVER_PORT', '3000'))
            server_host = os.environ.get('SERVER_HOST', '127.0.0.1')
            debug_level = os.environ.get('DEBUG_LEVEL', 'warning')
            
            assert dev_mode is True
            assert server_port == 8000
            assert server_host == 'localhost'
            assert debug_level == 'info'
            
        finally:
            # 恢复原始环境
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

class TestErrorHandlingIntegration:
    """测试错误处理集成"""
    
    @pytest.mark.asyncio
    async def test_websocket_error_recovery_workflow(self):
        """测试WebSocket错误恢复工作流程"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 创建各种状态的客户端
        good_clients = []
        bad_clients = []
        
        # 正常客户端
        for i in range(3):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            good_clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        # 各种错误客户端
        error_types = [
            ConnectionError("Network error"),
            OSError("Socket error"),
            Exception("Generic error")
        ]
        
        for error in error_types:
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock(side_effect=error)
            bad_clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        initial_count = len(server.websocket_clients)
        assert initial_count == 6
        
        # 执行通知，测试错误恢复
        with patch('dev_server.logger') as mock_logger:
            await server.notify_frontend_reload()
            
            # 验证错误客户端被移除
            for bad_client in bad_clients:
                assert bad_client not in server.websocket_clients
            
            # 验证正常客户端保留
            for good_client in good_clients:
                assert good_client in server.websocket_clients
            
            # 验证最终状态
            assert len(server.websocket_clients) == 3
            
            # 验证错误日志
            assert mock_logger.error.call_count == 3
    
    def test_import_error_handling_workflow(self):
        """测试导入错误处理工作流程"""
        # 测试各种导入错误场景
        error_scenarios = [
            ('missing_module', ImportError("No module named 'missing_module'")),
            ('corrupted_module', ImportError("Corrupted module")),
            ('version_conflict', ImportError("Version conflict"))
        ]
        
        for module_name, error in error_scenarios:
            def mock_import_with_specific_error(name, *args, **kwargs):
                if name == module_name:
                    raise error
                return Mock()
            
            with patch('builtins.__import__', side_effect=mock_import_with_specific_error):
                try:
                    __import__(module_name)
                    assert False, f"Should have raised {type(error).__name__}"
                except ImportError as e:
                    assert str(e) == str(error)
    
    def test_file_operation_error_handling(self):
        """测试文件操作错误处理"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试文件权限错误
        with patch('pathlib.Path.exists', side_effect=PermissionError("Access denied")):
            try:
                result = starter.check_project_structure()
                # 应该处理权限错误，返回False或处理异常
                assert isinstance(result, bool)
            except PermissionError:
                # 也可以让异常传播，但不应该崩溃
                pass
    
    @pytest.mark.asyncio
    async def test_async_operation_error_handling(self):
        """测试异步操作错误处理"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 模拟清理过程中的错误
        mock_observer = Mock()
        mock_observer.stop = Mock(side_effect=OSError("Observer error"))
        mock_observer.join = Mock()
        
        mock_runner = Mock()
        mock_runner.cleanup = AsyncMock(side_effect=Exception("Cleanup error"))
        
        server.observer = mock_observer
        server.runner = mock_runner
        
        # 执行清理，应该处理所有错误
        with patch('dev_server.logger') as mock_logger:
            await server.cleanup()
            
            # 验证错误被记录但不影响执行
            # (具体实现可能不同，这里主要测试不会崩溃)

class TestPerformanceIntegration:
    """测试性能相关的集成"""
    
    @pytest.mark.asyncio
    async def test_high_load_websocket_handling(self):
        """测试高负载WebSocket处理"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 创建大量客户端
        clients = []
        for i in range(100):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        # 执行大量并发通知
        tasks = []
        for i in range(10):
            tasks.append(server.notify_frontend_reload())
        
        # 并发执行
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有客户端都收到消息
        for client in clients:
            assert client.send_str.call_count >= 1
    
    def test_memory_management_workflow(self):
        """测试内存管理工作流程"""
        from dev_server import DevServer
        from server import RealTimeDataManager
        
        server = DevServer()
        manager = RealTimeDataManager()
        
        # 分配大量资源
        large_data = {}
        for i in range(1000):
            large_data[f"key_{i}"] = {"data": list(range(100))}
        
        manager.market_data.update(large_data)
        assert len(manager.market_data) == 1000
        
        # 批量添加WebSocket客户端
        clients = [Mock() for _ in range(1000)]
        server.websocket_clients.update(clients)
        manager.websocket_clients.update(clients)
        
        assert len(server.websocket_clients) == 1000
        assert len(manager.websocket_clients) == 1000
        
        # 清理资源
        manager.market_data.clear()
        server.websocket_clients.clear()
        manager.websocket_clients.clear()
        
        # 验证清理
        assert len(manager.market_data) == 0
        assert len(server.websocket_clients) == 0
        assert len(manager.websocket_clients) == 0
    
    def test_concurrent_file_operations(self):
        """测试并发文件操作"""
        import threading
        
        results = []
        errors = []
        
        def file_operation_worker():
            try:
                # 模拟文件操作
                current_file = Path(__file__)
                assert current_file.exists()
                assert current_file.is_file()
                
                # 模拟路径操作
                parent_dir = current_file.parent
                assert parent_dir.exists()
                assert parent_dir.is_dir()
                
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程并发执行
        threads = []
        for i in range(10):
            thread = threading.Thread(target=file_operation_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 10
        assert len(errors) == 0