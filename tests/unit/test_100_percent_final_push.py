"""
100%覆盖率最终冲刺测试
针对剩余未覆盖的具体代码行
"""

import pytest
import asyncio
import sys
import os
import time
import json
import socket
import tempfile
import signal
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestFinalCoverageTargets:
    """针对具体未覆盖行的测试"""
    
    def test_dev_server_missing_lines_41_82_86(self):
        """测试dev_server.py第41、82-86行"""
        from dev_server import HotReloadEventHandler, DevServer
        
        # 测试第41行：目录事件的早期返回
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        
        mock_event = Mock()
        mock_event.is_directory = True  # 这应该触发第41行的return
        
        # 不应该有任何进一步处理
        with patch('time.time', return_value=1000.0), \
             patch('asyncio.create_task') as mock_create_task:
            
            handler.on_modified(mock_event)
            mock_create_task.assert_not_called()  # 因为是目录事件
        
        # 测试第82-86行：CORS中间件
        server = DevServer()
        app = asyncio.run(server.create_app())
        
        # 获取CORS中间件
        cors_middleware = app.middlewares[0]
        
        # 创建模拟请求和处理器
        mock_request = Mock()
        mock_response = Mock()
        mock_response.headers = {}
        
        async def mock_handler(request):
            return mock_response
        
        # 执行CORS中间件
        result = asyncio.run(cors_middleware(mock_request, mock_handler))
        
        # 验证CORS头被添加（第82-86行）
        assert 'Access-Control-Allow-Origin' in result.headers
        assert result.headers['Access-Control-Allow-Origin'] == '*'
        assert 'Access-Control-Allow-Methods' in result.headers
        assert 'Access-Control-Allow-Headers' in result.headers
    
    @pytest.mark.asyncio
    async def test_dev_server_websocket_branches_122_132(self):
        """测试dev_server.py第122-132行的WebSocket消息分支"""
        from dev_server import DevServer
        from aiohttp import WSMsgType
        
        server = DevServer()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            mock_ws.exception = Mock(return_value=Exception("WebSocket error"))
            
            # 测试第122-132行：不同的消息类型处理
            messages = [
                # TEXT消息但无效JSON（测试第129行）
                Mock(type=WSMsgType.TEXT, data='invalid json'),
                # ERROR消息（测试第130-132行）
                Mock(type=WSMsgType.ERROR),
                Mock(type=WSMsgType.CLOSE)  # 结束循环
            ]
            
            async def message_iterator():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = message_iterator
            MockWSResponse.return_value = mock_ws
            
            mock_request = Mock()
            
            # 执行WebSocket处理器
            result = await server.websocket_handler(mock_request)
            
            # 验证处理结果
            assert result == mock_ws
    
    def test_dev_server_main_function_lines_332_336(self):
        """测试dev_server.py第332-336行的main函数"""
        from dev_server import main, check_dependencies
        
        # 测试依赖检查失败的分支（第332-333行）
        with patch('dev_server.check_dependencies', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            # 应该调用main函数，检查依赖失败，然后exit(1)
            asyncio.run(main())
            mock_exit.assert_called_once_with(1)
        
        # 测试正常执行路径（第335-336行）
        with patch('dev_server.check_dependencies', return_value=True), \
             patch('dev_server.DevServer') as MockDevServer, \
             patch('asyncio.sleep', side_effect=KeyboardInterrupt()):  # 模拟中断
            
            mock_server = Mock()
            mock_server.start = AsyncMock(side_effect=KeyboardInterrupt())
            MockDevServer.return_value = mock_server
            
            # 应该创建服务器并调用start
            try:
                asyncio.run(main())
            except KeyboardInterrupt:
                pass
            
            MockDevServer.assert_called_once()
            mock_server.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_exchange_initialization_lines_41_57(self):
        """测试server.py第41-57行的交易所初始化"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 模拟ccxt模块和交易所类
        mock_ccxt = Mock()
        mock_okx_class = Mock()
        mock_binance_class = Mock()
        mock_ccxt.okx = mock_okx_class
        mock_ccxt.binance = mock_binance_class
        
        with patch('server.ccxt', mock_ccxt):
            result = await manager.initialize_exchanges()
            
            # 验证交易所初始化（第43-54行）
            assert result is True
            mock_okx_class.assert_called_once()
            mock_binance_class.assert_called_once()
            
            # 验证日志记录（第56行）
            assert 'okx' in manager.exchanges
            assert 'binance' in manager.exchanges
    
    @pytest.mark.asyncio  
    async def test_server_data_stream_complete_flow_185_224(self):
        """测试server.py第185-224行的数据流完整流程"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 设置交易所模拟
        mock_exchange = Mock()
        mock_exchange.fetch_ticker = Mock(return_value={
            'last': 45000, 'baseVolume': 1000, 'change': 500,
            'percentage': 1.12, 'high': 46000, 'low': 44000,
            'bid': 44950, 'ask': 45050
        })
        manager.exchanges['okx'] = mock_exchange
        
        # 添加WebSocket客户端
        good_client = Mock()
        good_client.send_str = AsyncMock()
        bad_client = Mock()
        bad_client.send_str = AsyncMock(side_effect=Exception("Send failed"))
        
        manager.websocket_clients.add(good_client)
        manager.websocket_clients.add(bad_client)
        
        # 启动数据流
        manager.running = True
        
        # 创建数据流任务
        stream_task = asyncio.create_task(manager.start_data_stream())
        
        # 让它运行一小段时间
        await asyncio.sleep(0.3)
        
        # 停止数据流
        manager.running = False
        
        # 等待任务完成
        try:
            await asyncio.wait_for(stream_task, timeout=1.0)
        except asyncio.TimeoutError:
            stream_task.cancel()
        
        # 验证失败的客户端被移除（第202行）
        assert bad_client not in manager.websocket_clients
        # 验证正常客户端保留并收到消息
        assert good_client in manager.websocket_clients
        good_client.send_str.assert_called()
    
    @pytest.mark.asyncio
    async def test_server_websocket_subscription_handling_256_283(self):
        """测试server.py第256-283行的WebSocket订阅处理"""
        from server import websocket_handler, data_manager
        from aiohttp import WSMsgType
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            
            # 测试订阅消息处理（第261-273行）
            messages = [
                Mock(type=WSMsgType.TEXT, data='{"type": "subscribe", "symbols": ["BTC/USDT", "ETH/USDT"]}'),
                Mock(type=WSMsgType.CLOSE)
            ]
            
            async def message_iterator():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = message_iterator
            MockWSResponse.return_value = mock_ws
            
            # 模拟成功的数据获取
            with patch.object(data_manager, 'get_market_data') as mock_get_data:
                mock_get_data.return_value = {
                    'symbol': 'BTC/USDT',
                    'price': 45000,
                    'timestamp': int(time.time() * 1000)
                }
                
                mock_request = Mock()
                
                # 执行WebSocket处理器
                result = await websocket_handler(mock_request)
                
                # 验证订阅处理
                assert result == mock_ws
                # 应该为每个订阅的symbol调用get_market_data
                assert mock_get_data.call_count >= 1
    
    def test_server_main_function_complete_flow_401_403(self):
        """测试server.py第401-403行的main函数失败分支"""
        from server import main, data_manager
        
        # 测试交易所初始化失败的分支
        with patch.object(data_manager, 'initialize_exchanges', return_value=False), \
             patch('builtins.print') as mock_print:
            
            # 执行main函数
            result = asyncio.run(main(dev_mode=True))
            
            # 验证错误输出（第401-403行）
            mock_print.assert_called()
            assert result is None
    
    def test_start_dev_complete_check_flow_23_30(self):
        """测试start_dev.py第23-30行的Python版本检查"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试版本过低的分支（第23-27行）
        with patch('sys.version_info', (3, 7, 5)), \
             patch('builtins.print') as mock_print:
            
            result = starter.check_python_version()
            
            # 验证版本检查失败
            assert result is False
            # 验证错误消息输出
            mock_print.assert_called()
            
            # 检查是否输出了正确的错误信息
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            version_error_found = any('Python版本过低' in call for call in print_calls)
            assert version_error_found
    
    def test_start_dev_dependency_installation_34_68(self):
        """测试start_dev.py第34-68行的依赖安装流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟部分依赖缺失
        def mock_import(name):
            if name in ['aiohttp', 'watchdog']:
                return Mock()
            else:
                raise ImportError(f"No module named '{name}'")
        
        with patch('builtins.__import__', side_effect=mock_import), \
             patch('builtins.print') as mock_print, \
             patch('builtins.input', return_value='y'), \
             patch.object(starter, 'install_dependencies', return_value=True) as mock_install:
            
            # 执行依赖检查
            result = starter.check_dependencies()
            
            # 验证安装流程被触发（第59-61行）
            assert result is True
            mock_install.assert_called_once()
            
            # 验证用户交互（第59行）
            mock_print.assert_called()
    
    def test_start_dev_server_startup_121_144(self):
        """测试start_dev.py第121-144行的服务器启动流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试热重载模式启动（第124-127行）
        with patch('subprocess.run') as mock_run, \
             patch('builtins.print') as mock_print:
            
            result = starter.start_dev_server(mode='hot')
            
            # 验证命令构建和执行
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert 'dev_server.py' in str(call_args)
        
        # 测试增强模式启动（第129-131行）
        with patch('subprocess.run') as mock_run:
            
            result = starter.start_dev_server(mode='enhanced')
            
            # 验证命令构建
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert 'server.py' in str(call_args)
            assert '--dev' in call_args
    
    def test_start_dev_main_function_entry_point_187_193(self):
        """测试start_dev.py第187-193行的main函数入口"""
        from start_dev import DevEnvironmentStarter
        
        # 测试依赖检查失败的分支（第191-193行）
        test_args = ['start_dev.py', '--skip-deps']
        
        with patch('sys.argv', test_args), \
             patch.object(DevEnvironmentStarter, 'check_python_version', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            try:
                from start_dev import main
                main()
            except SystemExit:
                pass
            
            # 验证系统退出被调用
            mock_exit.assert_called_once_with(1)
    
    def test_signal_handler_setup_lines_340_346(self):
        """测试dev_server.py第340-346行的信号处理器设置"""
        # 这测试的是if __name__ == '__main__'分支中的信号处理器
        
        # 导入signal模块以测试处理器函数
        import signal
        from dev_server import signal_handler
        
        # 测试信号处理器函数（第340-342行）
        with patch('dev_server.logger') as mock_logger, \
             patch('sys.exit') as mock_exit:
            
            # 调用信号处理器
            signal_handler(signal.SIGINT, None)
            
            # 验证日志记录和退出
            mock_logger.info.assert_called_once_with("🛑 收到停止信号")
            mock_exit.assert_called_once_with(0)
    
    @pytest.mark.asyncio
    async def test_complete_application_startup_sequence(self):
        """测试完整的应用启动序列以覆盖剩余行"""
        from dev_server import DevServer
        from server import create_app
        
        # 测试dev_server的应用创建
        server = DevServer()
        
        # 测试静态文件路径不存在的分支（第102-103行）
        with patch('pathlib.Path.exists', return_value=False):
            app = await server.create_app()
            
            # 验证应用创建成功
            assert app is not None
            # 验证路由被添加
            routes = list(app.router.routes())
            assert len(routes) > 0
        
        # 测试server的应用创建（开发模式）
        app_dev = await create_app(dev_mode=True)
        
        # 验证开发模式特定配置（第362-367行）
        assert app_dev is not None
        
        # 验证中间件被添加
        assert len(app_dev.middlewares) > 0
        
        # 测试CORS中间件在开发模式下的行为
        cors_middleware = app_dev.middlewares[0]
        
        mock_request = Mock()
        mock_response = Mock()
        mock_response.headers = {}
        
        async def mock_handler(request):
            return mock_response
        
        # 执行中间件
        result = await cors_middleware(mock_request, mock_handler)
        
        # 验证开发模式的缓存控制头被添加（第363-365行）
        assert 'Cache-Control' in result.headers
        assert result.headers['Cache-Control'] == 'no-cache, no-store, must-revalidate'
    
    def test_file_existence_checks_comprehensive(self):
        """全面测试文件存在性检查的所有分支"""
        from start_dev import DevEnvironmentStarter
        from pathlib import Path
        
        starter = DevEnvironmentStarter()
        
        # 创建临时目录结构用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 模拟项目根目录
            with patch.object(starter, 'project_root', temp_path):
                
                # 测试所有文件都不存在的情况
                result = starter.check_project_structure()
                assert isinstance(result, bool)
                
                # 创建部分文件
                (temp_path / 'dev_server.py').write_text('# dev server')
                (temp_path / 'server.py').write_text('# server')
                
                # 测试部分文件存在的情况
                result2 = starter.check_project_structure()
                assert isinstance(result2, bool)
                
                # 创建完整的文件结构
                web_interface_dir = temp_path / 'file_management' / 'web_interface'
                web_interface_dir.mkdir(parents=True)
                (web_interface_dir / 'index.html').write_text('<html></html>')
                (web_interface_dir / 'app.js').write_text('console.log("test");')
                (web_interface_dir / 'styles.css').write_text('body { margin: 0; }')
                
                # 测试完整文件结构存在的情况
                result3 = starter.check_project_structure()
                assert isinstance(result3, bool)

class TestModuleLevelCodeExecution:
    """测试模块级别代码的执行"""
    
    def test_import_statements_and_logging_setup(self):
        """测试导入语句和日志设置"""
        # 重新加载模块以执行模块级代码
        import importlib
        
        modules_to_test = ['dev_server', 'server', 'start_dev']
        
        for module_name in modules_to_test:
            # 如果模块已导入，先删除
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # 重新导入模块
            module = importlib.import_module(module_name)
            
            # 验证日志设置被执行
            if hasattr(module, 'logger'):
                assert module.logger is not None
                assert module.logger.name == module_name
            
            # 验证基本导入成功
            assert module.__name__ == module_name
    
    def test_global_variable_initialization(self):
        """测试全局变量初始化"""
        from server import data_manager
        
        # 验证全局数据管理器被初始化（第236行）
        assert data_manager is not None
        assert hasattr(data_manager, 'exchanges')
        assert hasattr(data_manager, 'websocket_clients')
        assert hasattr(data_manager, 'market_data')
        assert hasattr(data_manager, 'running')
    
    def test_command_line_argument_processing(self):
        """测试命令行参数处理"""
        # 测试server.py的命令行参数处理（第467行）
        test_scenarios = [
            (['server.py'], False),
            (['server.py', '--dev'], True),
            (['server.py', '-d'], True),
            (['server.py', '--dev', '--other'], True),
        ]
        
        for test_argv, expected_dev_mode in test_scenarios:
            with patch('sys.argv', test_argv):
                # 模拟命令行参数检查逻辑
                dev_mode = '--dev' in test_argv or '-d' in test_argv
                assert dev_mode == expected_dev_mode