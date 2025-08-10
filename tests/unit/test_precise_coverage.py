"""
精确覆盖率测试
专门针对未覆盖的具体代码行进行测试
"""

import pytest
import asyncio
import sys
import os
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import subprocess
import signal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestDevServerPreciseCoverage:
    """精确测试dev_server.py中未覆盖的代码行"""
    
    def test_hot_reload_handler_init_lines_35_37(self):
        """测试HotReloadEventHandler初始化 (行35-37)"""
        from dev_server import HotReloadEventHandler
        
        mock_dev_server = Mock()
        handler = HotReloadEventHandler(mock_dev_server)
        
        # 精确验证初始化的每个属性 (覆盖行35-37)
        assert handler.dev_server == mock_dev_server  # 行35
        assert handler.last_reload_time == 0  # 行36
        assert handler.reload_cooldown == 1  # 行37
    
    def test_on_modified_detailed_execution_lines_40_60(self):
        """测试on_modified方法的详细执行路径 (行40-60)"""
        from dev_server import HotReloadEventHandler
        
        mock_dev_server = Mock()
        mock_dev_server.restart_backend = AsyncMock()
        mock_dev_server.notify_frontend_reload = AsyncMock()
        
        handler = HotReloadEventHandler(mock_dev_server)
        handler.last_reload_time = 0  # 确保冷却时间已过
        
        # 测试Python文件修改路径 (行40-60)
        mock_event = Mock()
        mock_event.is_directory = False  # 行40-41执行
        mock_event.src_path = "/test/file.py"
        
        with patch('time.time', return_value=10.0), \
             patch('asyncio.create_task') as mock_create_task, \
             patch('dev_server.logger') as mock_logger:
            
            # 执行方法，覆盖行40-60的所有分支
            handler.on_modified(mock_event)
            
            # 验证时间检查和日志记录 (行50-53)
            assert handler.last_reload_time == 10.0
            mock_logger.info.assert_called_with("🔄 文件已修改: /test/file.py")
            
            # 验证Python文件处理 (行55-57)
            mock_create_task.assert_called_once()
            
        # 测试前端文件修改路径 (行58-60)
        mock_event.src_path = "/test/style.css"
        handler.last_reload_time = 0  # 重置时间
        
        with patch('time.time', return_value=20.0), \
             patch('asyncio.create_task') as mock_create_task2:
            
            handler.on_modified(mock_event)
            
            # 验证前端文件处理 (行58-60)
            mock_create_task2.assert_called_once()
    
    def test_dev_server_methods_lines_77_105(self):
        """测试DevServer类的方法执行 (行77-105)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 测试create_app方法的具体执行路径
        with patch('aiohttp.web.Application') as MockApp, \
             patch('pathlib.Path.exists', return_value=True) as mock_exists:
            
            mock_app = Mock()
            mock_router = Mock()
            mock_app.router = mock_router
            mock_app.router.add_get = Mock()
            mock_app.router.add_post = Mock()
            MockApp.return_value = mock_app
            
            # 执行create_app以覆盖行77-105
            result = asyncio.run(server.create_app())
            
            # 验证应用配置
            assert result == mock_app
            mock_router.add_get.assert_called()
            mock_router.add_post.assert_called()
    
    @pytest.mark.asyncio
    async def test_websocket_handler_lines_109_141(self):
        """测试WebSocket处理器的详细执行 (行109-141)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWS:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            mock_ws.close = AsyncMock()
            
            # 模拟消息序列以覆盖所有分支
            messages = [
                # 模拟TEXT消息 (行122-132)
                Mock(type=1, data='{"type": "ping"}'),  # WSMsgType.TEXT
                # 模拟CLOSE消息 (行133-134)
                Mock(type=8)  # WSMsgType.CLOSE
            ]
            
            async def mock_aiter():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = mock_aiter
            MockWS.return_value = mock_ws
            
            mock_request = Mock()
            
            # 执行WebSocket处理器
            result = await server.websocket_handler(mock_request)
            
            # 验证WebSocket设置和消息处理
            assert result == mock_ws
            mock_ws.prepare.assert_called_once_with(mock_request)
            assert mock_ws in server.websocket_clients
    
    @pytest.mark.asyncio
    async def test_notify_frontend_reload_lines_163_181(self):
        """测试前端重载通知的详细执行 (行163-181)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 添加正常和失败的WebSocket客户端
        mock_ws_success = Mock()
        mock_ws_success.send_str = AsyncMock()
        
        mock_ws_fail = Mock()
        mock_ws_fail.send_str = AsyncMock(side_effect=Exception("Connection lost"))
        
        server.websocket_clients.add(mock_ws_success)
        server.websocket_clients.add(mock_ws_fail)
        
        with patch('time.time', return_value=1234567890), \
             patch('json.dumps', return_value='{"test": "data"}') as mock_dumps, \
             patch('dev_server.logger') as mock_logger:
            
            # 执行通知方法
            await server.notify_frontend_reload()
            
            # 验证消息构建 (行164-170)
            mock_dumps.assert_called_once()
            
            # 验证成功客户端收到消息
            mock_ws_success.send_str.assert_called_once()
            
            # 验证失败客户端被移除 (行175-181)
            assert mock_ws_fail not in server.websocket_clients
            assert mock_ws_success in server.websocket_clients
            
            # 验证错误日志
            mock_logger.error.assert_called()
    
    @pytest.mark.asyncio 
    async def test_restart_backend_lines_186_217(self):
        """测试后端重启的详细执行 (行186-217)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 添加WebSocket客户端
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        server.websocket_clients.add(mock_ws)
        
        with patch('time.time', return_value=1234567890), \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep, \
             patch('json.dumps', side_effect=['{"stage": "start"}', '{"stage": "complete"}']):
            
            # 执行重启方法
            await server.restart_backend()
            
            # 验证两阶段通知 (行187-205)
            assert mock_ws.send_str.call_count == 2
            
            # 验证延迟执行 (行206-207)
            mock_sleep.assert_called_once_with(2)
    
    def test_start_file_watcher_lines_223_239(self):
        """测试文件监控启动的详细执行 (行223-239)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch('watchdog.observers.Observer') as MockObserver, \
             patch('dev_server.HotReloadEventHandler') as MockHandler, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_observer = Mock()
            mock_observer.start = Mock()
            mock_observer.schedule = Mock()
            MockObserver.return_value = mock_observer
            
            mock_handler = Mock()
            MockHandler.return_value = mock_handler
            
            # 执行文件监控启动
            server.start_file_watcher()
            
            # 验证Observer配置 (行224-238)
            MockObserver.assert_called_once()
            MockHandler.assert_called_once_with(server)
            mock_observer.schedule.assert_called()
            mock_observer.start.assert_called_once()
            
            # 验证observer保存 (行239)
            assert server.observer == mock_observer
    
    @pytest.mark.asyncio
    async def test_start_method_lines_254_293(self):
        """测试start方法的详细执行 (行254-293)"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch.object(server, 'create_app', new_callable=AsyncMock) as mock_create_app, \
             patch('aiohttp.web.AppRunner') as MockAppRunner, \
             patch('aiohttp.web.TCPSite') as MockTCPSite, \
             patch.object(server, 'start_file_watcher') as mock_start_watcher, \
             patch('webbrowser.open') as mock_browser_open, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # 设置模拟对象
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            mock_runner = Mock()
            mock_runner.setup = AsyncMock()
            MockAppRunner.return_value = mock_runner
            
            mock_site = Mock()
            mock_site.start = AsyncMock()
            MockTCPSite.return_value = mock_site
            
            # 模拟KeyboardInterrupt来终止循环 (行288-290)
            mock_sleep.side_effect = [None, KeyboardInterrupt()]
            
            with patch.object(server, 'cleanup', new_callable=AsyncMock) as mock_cleanup, \
                 patch('dev_server.logger') as mock_logger:
                
                # 执行start方法
                await server.start()
                
                # 验证启动序列 (行255-283)
                mock_create_app.assert_called_once()
                MockAppRunner.assert_called_once_with(mock_app)
                mock_runner.setup.assert_called_once()
                MockTCPSite.assert_called_once()
                mock_site.start.assert_called_once()
                mock_start_watcher.assert_called_once()
                mock_browser_open.assert_called_once()
                
                # 验证清理 (行293)
                mock_cleanup.assert_called_once()
                
                # 验证日志记录 (行291)
                mock_logger.info.assert_called()
    
    def test_check_dependencies_lines_323_326(self):
        """测试依赖检查的具体执行路径 (行323-326)"""
        from dev_server import check_dependencies
        
        # 测试缺失依赖的情况 (行322-326)
        def mock_import_side_effect(name):
            if name == 'aiohttp':
                raise ImportError("No module named 'aiohttp'")
            elif name == 'webbrowser':
                import webbrowser
                return webbrowser
            else:
                return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect), \
             patch('builtins.print') as mock_print:
            
            result = check_dependencies()
            
            # 验证缺失包处理 (行322-326)
            assert result is False
            mock_print.assert_called()
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any('缺少开发环境依赖' in call for call in print_calls)

class TestServerPreciseCoverage:
    """精确测试server.py中未覆盖的代码行"""
    
    def test_real_time_data_manager_init_detailed(self):
        """测试RealTimeDataManager详细初始化"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 验证所有初始化属性
        assert manager.exchanges == {}
        assert manager.websocket_clients == set()
        assert manager.market_data == {}
        assert manager.running is False
    
    @pytest.mark.asyncio
    async def test_initialize_exchanges_detailed_lines_65_86(self):
        """测试交易所初始化的详细路径 (行65-86)"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 测试有凭据但导入失败的情况
        with patch('os.environ.get') as mock_env, \
             patch('ccxt.okx', side_effect=ImportError("ccxt not available")) as mock_okx:
            
            mock_env.side_effect = lambda key, default=None: {
                'OKX_API_KEY': 'test_key',
                'OKX_SECRET': 'test_secret', 
                'OKX_PASSPHRASE': 'test_pass'
            }.get(key, default)
            
            result = await manager.initialize_exchanges()
            
            # 验证导入失败处理
            assert result is False
            assert len(manager.exchanges) == 0
    
    @pytest.mark.asyncio
    async def test_get_market_data_detailed_execution(self):
        """测试市场数据获取的详细执行"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 添加模拟交易所
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 45000.0,
            'bid': 44999.0,
            'ask': 45001.0
        }
        manager.exchanges['okx'] = mock_exchange
        
        # 测试成功获取数据
        result = await manager.get_market_data('BTC/USDT')
        
        assert result is not None
        assert result['symbol'] == 'BTC/USDT'
        assert result['last'] == 45000.0
        mock_exchange.fetch_ticker.assert_called_once_with('BTC/USDT')
    
    @pytest.mark.asyncio
    async def test_broadcast_data_detailed_lines_173_224(self):
        """测试数据广播的详细执行 (行173-224)"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 添加多个客户端，包括会失败的
        mock_ws1 = Mock()
        mock_ws1.send_str = AsyncMock()
        mock_ws2 = Mock() 
        mock_ws2.send_str = AsyncMock(side_effect=ConnectionResetError("Connection lost"))
        mock_ws3 = Mock()
        mock_ws3.send_str = AsyncMock()
        
        manager.websocket_clients.update([mock_ws1, mock_ws2, mock_ws3])
        
        test_data = {'symbol': 'BTC/USDT', 'price': 45000}
        
        with patch('json.dumps', return_value='{"test": "data"}') as mock_dumps:
            await manager.broadcast_data(test_data)
            
            # 验证JSON序列化
            mock_dumps.assert_called_once_with(test_data)
            
            # 验证成功客户端收到消息
            mock_ws1.send_str.assert_called_once()
            mock_ws3.send_str.assert_called_once()
            
            # 验证失败客户端被移除
            assert mock_ws2 not in manager.websocket_clients
            assert len(manager.websocket_clients) == 2
    
    @pytest.mark.asyncio
    async def test_data_collection_loop_detailed(self):
        """测试数据收集循环的详细执行"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        manager.running = True
        
        call_count = 0
        async def mock_get_market_data(symbol):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                manager.running = False
            return {'symbol': symbol, 'price': 45000 + call_count}
        
        with patch.object(manager, 'get_market_data', side_effect=mock_get_market_data) as mock_get, \
             patch.object(manager, 'broadcast_data', new_callable=AsyncMock) as mock_broadcast, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            await manager.data_collection_loop()
            
            # 验证循环执行
            assert call_count >= 3
            assert mock_get.call_count >= 2
            assert mock_broadcast.call_count >= 2
            assert mock_sleep.call_count >= 2

class TestStartDevPreciseCoverage:
    """精确测试start_dev.py中未覆盖的代码行"""
    
    def test_dev_environment_starter_detailed_init(self):
        """测试DevEnvironmentStarter详细初始化"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 验证初始化属性
        assert hasattr(starter, 'project_root')
        assert hasattr(starter, 'python_executable')
        assert starter.project_root.is_dir()
        assert 'python' in starter.python_executable.lower()
    
    def test_check_python_version_detailed_execution(self):
        """测试Python版本检查的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试版本检查逻辑
        with patch('sys.version_info', (3, 7, 0)):  # 版本过低
            result = starter.check_python_version()
            assert result is False
            
        with patch('sys.version_info', (3, 9, 0)):  # 版本满足
            result = starter.check_python_version()
            assert result is True
    
    def test_install_dependencies_detailed_execution(self):
        """测试依赖安装的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试成功安装
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Successfully installed"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['pytest'])
            
            assert result is True
            mock_run.assert_called_once()
            
            # 验证命令构建
            call_args = mock_run.call_args[0][0]
            assert starter.python_executable in call_args
            assert '-m' in call_args
            assert 'pip' in call_args
            assert 'install' in call_args
            assert 'pytest' in call_args
    
    def test_check_port_availability_detailed(self):
        """测试端口可用性检查的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试端口检查逻辑
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket.__enter__ = Mock(return_value=mock_socket)
            mock_socket.__exit__ = Mock(return_value=None)
            mock_socket.connect_ex = Mock(return_value=1)  # 端口空闲
            mock_socket_class.return_value = mock_socket
            
            result = starter.check_port_availability(8000)
            
            assert result is True
            mock_socket.connect_ex.assert_called_once_with(('localhost', 8000))
    
    def test_validate_environment_detailed_execution(self):
        """测试环境验证的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试完整验证流程
        with patch.object(starter, 'check_python_version', return_value=True) as mock_python, \
             patch.object(starter, 'check_project_structure', return_value=True) as mock_structure:
            
            result = starter.validate_environment()
            
            # 验证所有检查都被调用
            mock_python.assert_called_once()
            mock_structure.assert_called_once()
            
            assert isinstance(result, bool)
    
    def test_start_dev_server_detailed_execution(self):
        """测试开发服务器启动的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试服务器启动
        with patch('subprocess.Popen') as mock_popen, \
             patch('webbrowser.open') as mock_browser:
            
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            result = starter.start_dev_server(auto_open_browser=True)
            
            assert result is True
            mock_popen.assert_called_once()
            mock_browser.assert_called_once()
            
            # 验证进程保存
            assert starter.dev_server_process == mock_process

class TestIntegratedPreciseCoverage:
    """集成测试覆盖更多代码路径"""
    
    @pytest.mark.asyncio
    async def test_complete_hot_reload_workflow(self):
        """测试完整的热重载工作流程"""
        from dev_server import DevServer, HotReloadEventHandler
        
        server = DevServer()
        handler = HotReloadEventHandler(server)
        
        # 添加WebSocket客户端
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        server.websocket_clients.add(mock_ws)
        
        # 模拟文件修改事件
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/app.py"
        
        handler.last_reload_time = 0  # 确保冷却时间已过
        
        with patch('time.time', return_value=100.0), \
             patch('asyncio.create_task', new_callable=AsyncMock) as mock_create_task:
            
            # 触发文件修改事件
            handler.on_modified(mock_event)
            
            # 验证任务创建
            mock_create_task.assert_called_once()
            
            # 直接执行重启后端
            await server.restart_backend()
            
            # 验证客户端收到通知
            assert mock_ws.send_str.call_count >= 1
    
    def test_dependency_checking_comprehensive(self):
        """测试全面的依赖检查"""
        from dev_server import check_dependencies as check_dev_deps
        from server import check_dependencies as check_server_deps
        
        # 测试开发服务器依赖检查
        with patch('builtins.print') as mock_print:
            result = check_dev_deps()
            assert isinstance(result, bool)
            
            if not result:
                # 如果依赖缺失，应该有打印输出
                mock_print.assert_called()
        
        # 测试服务器依赖检查
        result = check_server_deps()
        assert isinstance(result, bool)
    
    def test_complete_environment_setup_workflow(self):
        """测试完整环境设置工作流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟完整设置流程
        with patch.object(starter, 'validate_environment', return_value=True) as mock_validate, \
             patch.object(starter, 'install_dependencies', return_value=True) as mock_install, \
             patch.object(starter, 'start_dev_server', return_value=True) as mock_start:
            
            result = starter.setup_development_environment()
            
            # 验证完整流程
            mock_validate.assert_called_once()
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_server_lifecycle_comprehensive(self):
        """测试服务器完整生命周期"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 测试完整生命周期
        assert manager.running is False
        
        # 启动数据收集
        await manager.start_data_collection()
        assert manager.running is True
        
        # 停止数据收集
        await manager.stop_data_collection()
        assert manager.running is False
        
        # 测试客户端管理
        mock_ws = Mock()
        manager.websocket_clients.add(mock_ws)
        assert len(manager.websocket_clients) == 1
        
        manager.websocket_clients.remove(mock_ws)
        assert len(manager.websocket_clients) == 0

class TestErrorPathsPreciseCoverage:
    """测试错误处理路径的精确覆盖"""
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling_detailed(self):
        """测试WebSocket错误处理的详细路径"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 添加会产生各种错误的客户端
        mock_ws1 = Mock()
        mock_ws1.send_str = AsyncMock(side_effect=ConnectionError("Network error"))
        
        mock_ws2 = Mock()
        mock_ws2.send_str = AsyncMock(side_effect=OSError("Socket error"))
        
        mock_ws3 = Mock()
        mock_ws3.send_str = AsyncMock()  # 正常客户端
        
        server.websocket_clients.update([mock_ws1, mock_ws2, mock_ws3])
        
        # 执行通知，测试错误处理
        await server.notify_frontend_reload()
        
        # 验证错误客户端被移除
        assert mock_ws1 not in server.websocket_clients
        assert mock_ws2 not in server.websocket_clients
        assert mock_ws3 in server.websocket_clients
    
    def test_file_system_error_handling(self):
        """测试文件系统错误处理"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试文件不存在的处理
        with patch('pathlib.Path.exists', return_value=False):
            result = starter.check_project_structure()
            # 应该处理文件不存在的情况
            assert isinstance(result, bool)
    
    def test_subprocess_error_handling(self):
        """测试子进程错误处理"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试subprocess调用失败
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'pip')):
            result = starter.install_dependencies(['pytest'])
            assert result is False
        
        # 测试Popen失败
        with patch('subprocess.Popen', side_effect=OSError("Process creation failed")):
            result = starter.start_dev_server()
            assert result is False

class TestConfigurationPreciseCoverage:
    """测试配置相关的精确覆盖"""
    
    def test_environment_variable_handling_detailed(self):
        """测试环境变量处理的详细路径"""
        test_vars = {
            'TEST_VAR1': 'value1',
            'TEST_VAR2': '',
            'TEST_VAR3': '   '
        }
        
        with patch.dict('os.environ', test_vars, clear=True):
            # 测试各种环境变量情况
            assert os.environ.get('TEST_VAR1') == 'value1'
            assert os.environ.get('TEST_VAR2') == ''
            assert os.environ.get('TEST_VAR3') == '   '
            assert os.environ.get('NONEXISTENT_VAR') is None
            assert os.environ.get('NONEXISTENT_VAR', 'default') == 'default'
    
    def test_path_operations_detailed(self):
        """测试路径操作的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试路径解析和验证
        project_root = starter.project_root
        assert isinstance(project_root, Path)
        assert project_root.is_absolute()
        
        # 测试文件检查
        dev_server_file = project_root / 'dev_server.py'
        if dev_server_file.exists():
            assert dev_server_file.is_file()
            assert dev_server_file.suffix == '.py'

class TestPerformancePreciseCoverage:
    """测试性能相关的精确覆盖"""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_detailed(self):
        """测试并发操作的详细执行"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 添加多个客户端
        clients = []
        for i in range(10):
            mock_ws = Mock()
            mock_ws.send_str = AsyncMock()
            clients.append(mock_ws)
            server.websocket_clients.add(mock_ws)
        
        # 并发执行多次通知
        tasks = []
        for i in range(5):
            tasks.append(server.notify_frontend_reload())
        
        await asyncio.gather(*tasks)
        
        # 验证所有客户端都收到了消息
        for client in clients:
            assert client.send_str.call_count >= 1
    
    def test_resource_management_detailed(self):
        """测试资源管理的详细执行"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟资源分配和清理
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        
        starter.dev_server_process = mock_process
        
        # 执行清理
        starter.stop_dev_server()
        
        # 验证资源被正确释放
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert starter.dev_server_process is None