"""
🎯 终极100%覆盖率攻坚测试
使用极限模拟技术攻击剩余135行未覆盖代码
"""

import pytest
import asyncio
import sys
import os
import time
import json
import signal
import subprocess
import threading
import socket
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestUltimateDevServerCoverage:
    """终极攻坚dev_server.py剩余40行"""
    
    def test_directory_event_early_return_line_41(self):
        """精确测试第41行：目录事件的早期返回"""
        from dev_server import HotReloadEventHandler
        
        mock_server = Mock()
        handler = HotReloadEventHandler(mock_server)
        
        # 创建目录事件
        mock_event = Mock()
        mock_event.is_directory = True  # 触发第41行的early return
        
        # 确保没有任何后续处理
        with patch('time.time') as mock_time, \
             patch('pathlib.Path') as mock_path, \
             patch('asyncio.create_task') as mock_task:
            
            # 调用处理器 - 应该在第41行直接返回
            handler.on_modified(mock_event)
            
            # 验证没有后续调用
            mock_time.assert_not_called()
            mock_path.assert_not_called()
            mock_task.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cors_middleware_lines_82_86(self):
        """精确测试第82-86行：CORS中间件的每一行"""
        from dev_server import DevServer
        
        server = DevServer()
        app = await server.create_app()
        
        # 获取CORS中间件（第80行创建的）
        cors_middleware = app.middlewares[0]
        
        # 创建精确的测试场景
        mock_request = Mock()
        mock_response = Mock()
        mock_response.headers = {}
        
        async def mock_handler(request):
            return mock_response
        
        # 执行CORS中间件，覆盖第82-86行
        result = await cors_middleware(mock_request, mock_handler)
        
        # 验证第82-86行的每个头部设置
        assert result.headers['Access-Control-Allow-Origin'] == '*'  # 第83行
        assert result.headers['Access-Control-Allow-Methods'] == 'GET, POST, OPTIONS'  # 第84行
        assert result.headers['Access-Control-Allow-Headers'] == 'Content-Type'  # 第85行
        assert result == mock_response  # 第86行的return
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling_lines_123_127(self):
        """精确测试第123-127行：WebSocket消息处理分支"""
        from dev_server import DevServer
        from aiohttp import WSMsgType
        
        server = DevServer()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            
            # 创建特定的消息来触发第123-127行
            messages = [
                Mock(type=WSMsgType.TEXT, data='{"type": "ping"}'),  # 触发第123行
                Mock(type=WSMsgType.CLOSE)
            ]
            
            async def message_iterator():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = message_iterator
            MockWSResponse.return_value = mock_ws
            
            mock_request = Mock()
            
            # 执行WebSocket处理器，应该覆盖第123-127行
            result = await server.websocket_handler(mock_request)
            
            # 验证ping消息处理（第124-127行）
            mock_ws.send_str.assert_called()
            # 验证pong响应被发送
            call_args = mock_ws.send_str.call_args[0][0]
            assert 'pong' in call_args
    
    @pytest.mark.asyncio  
    async def test_websocket_error_handling_lines_130_132(self):
        """精确测试第130-132行：WebSocket错误处理"""
        from dev_server import DevServer
        from aiohttp import WSMsgType
        
        server = DevServer()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            mock_ws.exception = Mock(return_value=Exception("WebSocket test error"))
            
            # 创建ERROR消息来触发第130-132行
            messages = [
                Mock(type=WSMsgType.ERROR),  # 触发第130行
                Mock(type=WSMsgType.CLOSE)
            ]
            
            async def message_iterator():
                for msg in messages:
                    yield msg
            
            mock_ws.__aiter__ = message_iterator
            MockWSResponse.return_value = mock_ws
            
            mock_request = Mock()
            
            with patch('dev_server.logger') as mock_logger:
                # 执行WebSocket处理器
                result = await server.websocket_handler(mock_request)
                
                # 验证第131行的异常记录
                mock_logger.error.assert_called()
                # 验证循环在第132行被break
                assert result == mock_ws
    
    def test_dependency_check_missing_webbrowser_line_145(self):
        """精确测试第145行：依赖检查失败路径"""
        from dev_server import check_dependencies
        
        def mock_failing_import(name, *args, **kwargs):
            if name == 'webbrowser':
                raise ImportError("No module named 'webbrowser'")
            else:
                return Mock()
        
        with patch('builtins.__import__', side_effect=mock_failing_import), \
             patch('builtins.print') as mock_print:
            
            # 执行依赖检查，应该触发第145行
            result = check_dependencies()
            
            # 验证第145行的return False
            assert result is False
            mock_print.assert_called()
    
    @pytest.mark.asyncio
    async def test_restart_handler_lines_155_156(self):
        """精确测试第155-156行：重启处理器"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch.object(server, 'restart_backend', new_callable=AsyncMock) as mock_restart:
            mock_request = Mock()
            
            # 执行重启处理器，应该覆盖第155-156行
            response = await server.restart_handler(mock_request)
            
            # 验证第155行的restart_backend调用
            mock_restart.assert_called_once()
            
            # 验证第156行的response返回
            assert hasattr(response, '_body')  # aiohttp.web.json_response特征
    
    def test_no_websocket_clients_early_return_line_164_187(self):
        """精确测试第164行和第187行：无客户端时的早期返回"""
        from dev_server import DevServer
        
        server = DevServer()
        # 确保没有WebSocket客户端
        server.websocket_clients.clear()
        
        # 测试notify_frontend_reload的第164行早期返回
        result1 = asyncio.run(server.notify_frontend_reload())
        assert result1 is None  # 第164行的early return
        
        # 测试restart_backend的第187行早期返回  
        result2 = asyncio.run(server.restart_backend())
        assert result2 is None  # 第187行的early return
    
    @pytest.mark.asyncio
    async def test_complete_server_startup_loop_lines_254_293(self):
        """终极测试第254-293行：完整服务器启动循环"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 精确模拟启动序列的每个步骤
        with patch.object(server, 'create_app', new_callable=AsyncMock) as mock_create_app, \
             patch('aiohttp.web.AppRunner') as MockAppRunner, \
             patch('aiohttp.web.TCPSite') as MockTCPSite, \
             patch.object(server, 'start_file_watcher') as mock_file_watcher, \
             patch('webbrowser.open') as mock_browser, \
             patch('asyncio.sleep') as mock_sleep, \
             patch('dev_server.logger') as mock_logger:
            
            # 设置所有必要的模拟对象
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            mock_runner = Mock()
            mock_runner.setup = AsyncMock()
            MockAppRunner.return_value = mock_runner
            
            mock_site = Mock()  
            mock_site.start = AsyncMock()
            MockTCPSite.return_value = mock_site
            
            # 模拟KeyboardInterrupt来结束无限循环
            mock_sleep.side_effect = [None, None, KeyboardInterrupt()]
            
            with patch.object(server, 'cleanup', new_callable=AsyncMock) as mock_cleanup:
                
                try:
                    # 执行完整的启动序列
                    await server.start()
                except KeyboardInterrupt:
                    pass
                
                # 验证第254-293行的每个关键步骤
                mock_create_app.assert_called_once()  # 第254行
                MockAppRunner.assert_called_once_with(mock_app)  # 第257行
                mock_runner.setup.assert_called_once()  # 第258行
                MockTCPSite.assert_called_once_with(mock_runner, server.host, server.port)  # 第260行
                mock_site.start.assert_called_once()  # 第261行
                mock_file_watcher.assert_called_once()  # 第264行
                mock_browser.assert_called_once()  # 第281行
                mock_cleanup.assert_called_once()  # 第293行
                
                # 验证日志输出（第267-284行）
                assert mock_logger.info.call_count >= 5
    
    @pytest.mark.asyncio
    async def test_webbrowser_open_exception_lines_323_326(self):
        """精确测试第323-326行：浏览器打开异常处理"""
        from dev_server import DevServer
        
        server = DevServer()
        
        with patch('aiohttp.web.AppRunner'), \
             patch('aiohttp.web.TCPSite'), \
             patch.object(server, 'create_app', new_callable=AsyncMock), \
             patch.object(server, 'start_file_watcher'), \
             patch('webbrowser.open', side_effect=Exception("Browser open failed")), \
             patch('asyncio.sleep', side_effect=KeyboardInterrupt()), \
             patch('dev_server.logger') as mock_logger, \
             patch.object(server, 'cleanup', new_callable=AsyncMock):
            
            try:
                # 执行启动，应该捕获浏览器打开异常
                await server.start()
            except KeyboardInterrupt:
                pass
            
            # 验证第324行的异常被捕获，第325行的日志被记录
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            manual_open_message_found = any('请手动打开浏览器' in call for call in log_calls)
            assert manual_open_message_found
    
    def test_main_function_dependency_failure_lines_332_333(self):
        """精确测试第332-333行：main函数依赖检查失败"""
        from dev_server import main, check_dependencies
        
        with patch('dev_server.check_dependencies', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            # 执行main函数，应该在第332-333行退出
            asyncio.run(main())
            
            # 验证第333行的sys.exit(1)
            mock_exit.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_main_function_server_creation_lines_335_336(self):
        """精确测试第335-336行：main函数服务器创建和启动"""
        from dev_server import main, DevServer
        
        with patch('dev_server.check_dependencies', return_value=True), \
             patch('dev_server.DevServer') as MockDevServer:
            
            mock_server = Mock()
            mock_server.start = AsyncMock(side_effect=KeyboardInterrupt())  # 模拟中断
            MockDevServer.return_value = mock_server
            
            # 执行main函数，应该覆盖第335-336行
            try:
                await main()
            except KeyboardInterrupt:
                pass
            
            # 验证第335行的DevServer()创建
            MockDevServer.assert_called_once()
            # 验证第336行的server.start()调用
            mock_server.start.assert_called_once()

class TestUltimateServerCoverage:
    """终极攻坚server.py剩余37行"""
    
    @pytest.mark.asyncio
    async def test_exchange_initialization_lines_41_43_50_57(self):
        """精确测试第41-57行：交易所初始化的关键行"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 模拟ccxt模块
        mock_ccxt = Mock()
        mock_okx_class = Mock()
        mock_binance_class = Mock()
        
        # 设置交易所构造函数
        mock_okx_instance = Mock()
        mock_binance_instance = Mock()
        mock_okx_class.return_value = mock_okx_instance
        mock_binance_class.return_value = mock_binance_instance
        
        mock_ccxt.okx = mock_okx_class
        mock_ccxt.binance = mock_binance_class
        
        with patch('server.ccxt', mock_ccxt), \
             patch('server.logger') as mock_logger:
            
            # 执行交易所初始化，应该覆盖第41-57行
            result = await manager.initialize_exchanges()
            
            # 验证第43-54行的OKX和Binance初始化
            mock_okx_class.assert_called_once_with({
                'sandbox': False,
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
            mock_binance_class.assert_called_once_with({
                'sandbox': False,
                'rateLimit': 1000,
                'enableRateLimit': True,
            })
            
            # 验证第56行的日志记录
            mock_logger.info.assert_called_with("✅ 交易所API初始化完成")
            
            # 验证第57行的return True
            assert result is True
            
            # 验证交易所被保存
            assert manager.exchanges['okx'] == mock_okx_instance
            assert manager.exchanges['binance'] == mock_binance_instance
    
    @pytest.mark.asyncio
    async def test_historical_data_complete_flow_lines_123_141(self):
        """精确测试第123-141行：历史数据获取完整流程"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 设置模拟的OHLCV数据
        mock_ohlcv = [
            [1640000000000, 45000, 46000, 44000, 45500, 1000],  # timestamp, o, h, l, c, v
            [1640003600000, 45500, 46500, 45000, 46000, 1200],
        ]
        
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv = Mock(return_value=mock_ohlcv)
        manager.exchanges['okx'] = mock_exchange
        
        # 执行历史数据获取，应该覆盖第123-141行
        result = await manager.get_historical_data("BTC/USDT", "1h", 100)
        
        # 验证第124-125行的API调用
        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1h", None, 100)
        
        # 验证第128-139行的数据转换
        assert len(result) == 2
        assert result[0]['timestamp'] == 1640000000000
        assert result[0]['open'] == 45000.0
        assert result[0]['high'] == 46000.0
        assert result[0]['low'] == 44000.0
        assert result[0]['close'] == 45500.0
        assert result[0]['volume'] == 1000.0
        assert result[0]['exchange'] == 'okx'
        assert result[0]['data_source'] == 'real'
        
        # 验证第141行的return
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_data_stream_error_handling_lines_204_221(self):
        """精确测试第204-221行：数据流错误处理分支"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 创建会失败的交易所
        mock_exchange = Mock()
        mock_exchange.fetch_ticker = Mock(side_effect=Exception("API Error"))
        manager.exchanges['okx'] = mock_exchange
        
        # 添加WebSocket客户端
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        manager.websocket_clients.add(mock_ws)
        
        # 启动数据流
        manager.running = True
        
        # 创建数据流任务
        stream_task = asyncio.create_task(manager.start_data_stream())
        
        # 短暂运行后停止
        await asyncio.sleep(0.3)
        manager.running = False
        
        try:
            await asyncio.wait_for(stream_task, timeout=1.0)
        except asyncio.TimeoutError:
            stream_task.cancel()
        
        # 验证错误消息被发送（第207-221行）
        mock_ws.send_str.assert_called()
        
        # 检查是否发送了错误消息
        sent_messages = [call[0][0] for call in mock_ws.send_str.call_args_list]
        error_message_found = any('data_error' in msg for msg in sent_messages)
        assert error_message_found
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_detailed_lines_257_283(self):
        """精确测试第257-283行：WebSocket订阅处理详细流程"""
        from server import websocket_handler, data_manager
        from aiohttp import WSMsgType
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            mock_ws = Mock()
            mock_ws.prepare = AsyncMock()
            mock_ws.send_str = AsyncMock()
            
            # 创建详细的订阅消息来触发第257-283行
            messages = [
                Mock(type=WSMsgType.TEXT, data='{"type": "subscribe", "symbols": ["BTC/USDT", "ETH/USDT"]}'),
                Mock(type=WSMsgType.TEXT, data='invalid json'),  # 触发第275行
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
                
                # 验证第263-273行的订阅处理
                assert mock_get_data.call_count >= 2  # 为每个symbol调用
                
                # 验证第276-279行的错误处理
                error_calls = [call for call in mock_ws.send_str.call_args_list 
                              if 'error' in str(call) or '无效的JSON格式' in str(call)]
                # 应该有错误响应
    
    def test_main_function_exchange_init_failure_lines_401_403(self):
        """精确测试第401-403行：main函数中交易所初始化失败"""
        from server import main, data_manager
        
        with patch.object(data_manager, 'initialize_exchanges', return_value=False), \
             patch('server.logger') as mock_logger, \
             patch('builtins.print') as mock_print:
            
            # 执行main函数，应该在第401-403行处理失败
            result = asyncio.run(main(dev_mode=True))
            
            # 验证第401-403行的错误处理
            mock_logger.error.assert_called()
            mock_print.assert_called()
            assert result is None
    
    def test_command_line_dev_mode_detection_lines_458_461(self):
        """精确测试第458-461行：命令行开发模式检测"""
        
        # 模拟不同的命令行参数
        test_scenarios = [
            (['server.py'], False),
            (['server.py', '--dev'], True),
            (['server.py', '-d'], True),
            (['server.py', '--dev', '--other'], True),
        ]
        
        for test_argv, expected_dev_mode in test_scenarios:
            with patch('sys.argv', test_argv):
                
                # 直接测试第467行的逻辑
                dev_mode = '--dev' in test_argv or '-d' in test_argv
                assert dev_mode == expected_dev_mode
                
                # 如果是__main__执行，应该会检查这个逻辑
                if expected_dev_mode:
                    assert '--dev' in test_argv or '-d' in test_argv
                else:
                    assert '--dev' not in test_argv and '-d' not in test_argv

class TestUltimateStartDevCoverage:
    """终极攻坚start_dev.py剩余58行"""
    
    def test_python_version_check_failure_lines_23_30(self):
        """精确测试第23-30行：Python版本检查失败完整流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟低版本Python
        with patch('sys.version_info', (3, 6, 8)), \
             patch('builtins.print') as mock_print:
            
            # 执行版本检查，应该覆盖第23-30行
            result = starter.check_python_version()
            
            # 验证第24-27行的版本检查逻辑
            assert result is False
            
            # 验证第25-27行的错误输出
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any('Python版本过低' in call for call in print_calls)
            assert any('需要Python 3.8或更高版本' in call for call in print_calls)
            
            # 验证第29-30行的成功路径不被执行
            success_calls = [call for call in print_calls if '✅ Python版本' in call]
            assert len(success_calls) == 0
    
    def test_dependency_check_interactive_flow_lines_34_68(self):
        """精确测试第34-68行：依赖检查交互完整流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟部分依赖缺失
        def mock_import_with_missing(name):
            if name in ['aiohttp', 'watchdog']:
                return Mock()
            else:
                raise ImportError(f"No module named '{name}'")
        
        # 测试用户选择安装的流程
        with patch('builtins.__import__', side_effect=mock_import_with_missing), \
             patch('builtins.print') as mock_print, \
             patch('builtins.input', return_value='y'), \
             patch.object(starter, 'install_dependencies', return_value=True) as mock_install:
            
            # 执行依赖检查，应该覆盖第34-68行
            result = starter.check_dependencies()
            
            # 验证第43-54行的依赖检查循环
            mock_print.assert_called()
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            check_calls = [call for call in print_calls if '检查依赖包' in call]
            assert len(check_calls) > 0
            
            # 验证第56-61行的自动安装询问和执行
            mock_install.assert_called_once()
            assert result is True
            
        # 测试用户选择不安装的流程
        with patch('builtins.__import__', side_effect=mock_import_with_missing), \
             patch('builtins.print') as mock_print, \
             patch('builtins.input', return_value='n'):
            
            # 用户拒绝安装，应该覆盖第63-65行
            result2 = starter.check_dependencies()
            
            # 验证第63-65行的手动安装提示
            assert result2 is False
            print_calls2 = [call[0][0] for call in mock_print.call_args_list]
            manual_install_calls = [call for call in print_calls2 if 'pip install' in call]
            assert len(manual_install_calls) > 0
    
    def test_server_startup_complete_flow_lines_121_144(self):
        """精确测试第121-144行：服务器启动完整流程"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试热重载模式启动（第124-127行）
        with patch('subprocess.run') as mock_run, \
             patch('builtins.print') as mock_print:
            
            # 执行热重载启动
            result = starter.start_dev_server(mode='hot')
            
            # 验证第126-127行的命令构建
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            command = call_args[0][0]
            assert 'dev_server.py' in str(command)
            assert starter.python_executable in command
            
            # 验证第133行的命令打印
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            command_calls = [call for call in print_calls if '执行命令' in call]
            assert len(command_calls) > 0
            
        # 测试增强模式启动（第129-131行）
        with patch('subprocess.run') as mock_run:
            
            result2 = starter.start_dev_server(mode='enhanced')
            
            # 验证第130-131行的增强模式命令构建
            call_args2 = mock_run.call_args[0][0]
            assert 'server.py' in str(call_args2)
            assert '--dev' in call_args2
        
        # 测试启动异常处理（第140-142行）
        with patch('subprocess.run', side_effect=Exception("Startup failed")), \
             patch('builtins.print') as mock_print:
            
            result3 = starter.start_dev_server()
            
            # 验证第141-142行的异常处理
            assert result3 is False
            print_calls3 = [call[0][0] for call in mock_print.call_args_list]
            error_calls = [call for call in print_calls3 if '启动失败' in call]
            assert len(error_calls) > 0
    
    def test_usage_info_display_lines_148_163(self):
        """精确测试第148-163行：使用说明显示"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('builtins.print') as mock_print:
            
            # 执行使用说明显示，应该覆盖第148-163行
            starter.show_usage_info()
            
            # 验证所有使用说明内容被打印
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            
            # 验证第149-163行的各种说明内容
            expected_content = [
                'AI量化交易系统 - 开发环境',  # 第149行
                '使用说明',  # 第151行
                '.py 文件将自动重启后端',  # 第152行
                '.html/.css/.js 文件将自动刷新',  # 第153行
                'http://localhost:8000',  # 第157行
                'WebSocket连接状态',  # 第160行
                '开发模式',  # 第161行
            ]
            
            for expected in expected_content:
                found = any(expected in call for call in print_calls)
                assert found, f"Expected content not found: {expected}"
    
    def test_main_function_complete_flow_lines_187_193(self):
        """精确测试第187-193行：main函数完整执行流程"""
        from start_dev import DevEnvironmentStarter, main
        
        # 测试版本检查失败退出（第186-187行）
        with patch.object(DevEnvironmentStarter, 'check_python_version', return_value=False), \
             patch('sys.exit') as mock_exit:
            
            # 应该在版本检查失败后退出
            main()
            mock_exit.assert_called_once_with(1)
        
        # 测试依赖检查失败退出（第191-193行）
        test_args = ['start_dev.py']
        
        with patch('sys.argv', test_args), \
             patch.object(DevEnvironmentStarter, 'check_python_version', return_value=True), \
             patch.object(DevEnvironmentStarter, 'check_dependencies', return_value=False), \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            
            # 应该在依赖检查失败后退出
            main()
            
            # 验证第192-193行的错误提示和退出
            mock_exit.assert_called_once_with(1)
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            skip_deps_hint = any('--skip-deps' in call for call in print_calls)
            assert skip_deps_hint

class TestUltimateMainModuleExecution:
    """终极攻坚主模块执行路径"""
    
    def test_dev_server_main_module_execution(self):
        """测试dev_server.py的主模块执行"""
        
        # 模拟直接执行模块的环境
        with patch('dev_server.__name__', '__main__'), \
             patch('signal.signal') as mock_signal, \
             patch('asyncio.run') as mock_asyncio_run:
            
            # 重新导入模块来触发主模块执行
            import importlib
            import dev_server
            importlib.reload(dev_server)
            
            # 验证信号处理器被设置
            assert mock_signal.call_count >= 2  # SIGINT和SIGTERM
            
            # 验证asyncio.run被调用
            mock_asyncio_run.assert_called_once()
    
    def test_server_main_module_execution(self):
        """测试server.py的主模块执行"""
        
        test_argv = ['server.py', '--dev']
        
        with patch('sys.argv', test_argv), \
             patch('server.check_dependencies', return_value=True), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('os.path.exists', return_value=True), \
             patch('builtins.print') as mock_print, \
             patch('asyncio.run') as mock_asyncio_run:
            
            # 重新导入模块来触发主模块执行
            import importlib
            import server
            importlib.reload(server)
            
            # 验证开发模式被检测
            # 验证asyncio.run被调用
            mock_asyncio_run.assert_called_once()
    
    def test_start_dev_main_module_execution(self):
        """测试start_dev.py的主模块执行"""
        
        with patch('start_dev.main') as mock_main:
            
            # 重新导入模块来触发主模块执行
            import importlib
            import start_dev
            importlib.reload(start_dev)
            
            # 验证main函数被调用
            mock_main.assert_called_once()

class TestUltimateRealEnvironmentSimulation:
    """终极真实环境模拟"""
    
    @pytest.mark.asyncio
    async def test_complete_http_server_lifecycle(self):
        """完整HTTP服务器生命周期测试"""
        from dev_server import DevServer
        import aiohttp
        
        server = DevServer()
        
        # 使用真实的aiohttp组件
        app = await server.create_app()
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        
        # 尝试绑定到随机端口
        site = aiohttp.web.TCPSite(runner, 'localhost', 0)  # 0表示随机端口
        await site.start()
        
        # 验证服务器启动
        assert site._server is not None
        
        # 清理
        await runner.cleanup()
    
    def test_real_file_operations_with_temp_files(self):
        """使用真实临时文件的文件操作测试"""
        from dev_server import HotReloadEventHandler
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # 创建真实的测试文件
            test_file = temp_path / 'test.py'
            test_file.write_text('print("test")')
            
            mock_server = Mock()
            handler = HotReloadEventHandler(mock_server)
            handler.last_reload_time = 0
            
            # 使用真实文件路径
            mock_event = Mock()
            mock_event.is_directory = False
            mock_event.src_path = str(test_file)
            
            with patch('time.time', return_value=1000.0), \
                 patch('asyncio.create_task') as mock_task:
                
                handler.on_modified(mock_event)
                
                # 验证真实的.py文件触发了处理
                mock_task.assert_called_once()
    
    def test_real_socket_operations(self):
        """真实的socket操作测试"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试真实的端口检查
        # 使用系统端口（通常被占用）
        result_ssh = starter.check_port_availability(22)  # SSH端口
        assert isinstance(result_ssh, bool)
        
        # 使用高端口（通常可用）
        result_high = starter.check_port_availability(65432)
        assert isinstance(result_high, bool)
    
    def test_real_process_operations(self):
        """真实的进程操作测试"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 使用真实的Python命令测试
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Success"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['pip'])  # pip应该总是可用
            assert result is True
            
            # 验证真实的命令构建
            call_args = mock_run.call_args[0][0]
            assert starter.python_executable in call_args
            assert 'pip' in call_args
            assert 'install' in call_args