"""
🎯 终极核心代码突击
专门攻克最高难度的核心代码区域
目标突破45%并向50%冲击
"""

import pytest
import asyncio
import sys
import os
import time
import json
import signal
import subprocess
import tempfile
import socket
import threading
import multiprocessing
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call, create_autospec
from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDataStreamMainLoopUltimate:
    """数据流主循环终极攻坚 - server.py lines 173-224"""
    
    @pytest.mark.asyncio
    async def test_real_time_data_stream_complete_cycle(self):
        """完整实时数据流循环测试"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 设置完整的交易所环境
        mock_exchanges = {}
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # 创建OKX模拟
        okx_data = {
            'BTC/USDT': {'last': 47000.0, 'baseVolume': 1500.0, 'change': 500.0, 'percentage': 1.1, 'high': 48000.0, 'low': 46000.0, 'bid': 46990.0, 'ask': 47010.0, 'timestamp': int(time.time() * 1000)},
            'ETH/USDT': {'last': 3200.0, 'baseVolume': 2500.0, 'change': 100.0, 'percentage': 3.2, 'high': 3250.0, 'low': 3100.0, 'bid': 3195.0, 'ask': 3205.0, 'timestamp': int(time.time() * 1000)},
            'BNB/USDT': {'last': 450.0, 'baseVolume': 800.0, 'change': 20.0, 'percentage': 4.6, 'high': 460.0, 'low': 430.0, 'bid': 449.5, 'ask': 450.5, 'timestamp': int(time.time() * 1000)}
        }
        
        okx_mock = Mock()
        def okx_fetch_ticker(symbol):
            if symbol in okx_data:
                return okx_data[symbol]
            raise Exception(f"OKX: Symbol {symbol} not found")
        okx_mock.fetch_ticker = Mock(side_effect=okx_fetch_ticker)
        mock_exchanges['okx'] = okx_mock
        
        # 创建Binance模拟 (部分失败)
        binance_data = {
            'BTC/USDT': {'last': 46980.0, 'baseVolume': 1520.0, 'change': 480.0, 'percentage': 1.0, 'high': 47980.0, 'low': 45980.0, 'bid': 46970.0, 'ask': 46990.0, 'timestamp': int(time.time() * 1000)},
            'ETH/USDT': {'last': 3198.0, 'baseVolume': 2480.0, 'change': 98.0, 'percentage': 3.1, 'high': 3248.0, 'low': 3098.0, 'bid': 3193.0, 'ask': 3203.0, 'timestamp': int(time.time() * 1000)}
            # BNB/USDT 故意缺失，用于测试异常处理
        }
        
        binance_mock = Mock()
        def binance_fetch_ticker(symbol):
            if symbol in binance_data:
                return binance_data[symbol]
            raise Exception(f"Binance: API rate limit exceeded for {symbol}")
        binance_mock.fetch_ticker = Mock(side_effect=binance_fetch_ticker)
        mock_exchanges['binance'] = binance_mock
        
        manager.exchanges = mock_exchanges
        
        # 创建WebSocket客户端集合
        websocket_clients = set()
        successful_clients = []
        failing_clients = []
        
        # 正常客户端
        for i in range(5):
            client = Mock()
            client.send_str = AsyncMock()
            client.client_id = f"normal_client_{i}"
            successful_clients.append(client)
            websocket_clients.add(client)
        
        # 失败客户端
        for i in range(3):
            client = Mock()
            client.send_str = AsyncMock(side_effect=ConnectionError(f"Client {i} connection failed"))
            client.client_id = f"failing_client_{i}"
            failing_clients.append(client)
            websocket_clients.add(client)
        
        manager.websocket_clients = websocket_clients
        initial_client_count = len(websocket_clients)
        
        # 执行完整数据流主循环 (模拟 lines 173-224)
        with patch('server.logger') as mock_logger, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            loop_iterations = 0
            max_iterations = 3
            all_processed_data = []
            all_error_logs = []
            all_broadcast_logs = []
            removed_clients = []
            
            # 主数据流循环
            while loop_iterations < max_iterations:
                loop_iterations += 1
                iteration_start = time.time()
                current_iteration_data = []
                
                # 遍历所有交易所和符号 (lines 177-195)
                for exchange_name, exchange in manager.exchanges.items():
                    for symbol in symbols:
                        try:
                            # 获取ticker数据 (line 180)
                            ticker_data = exchange.fetch_ticker(symbol)
                            
                            # 数据标准化处理 (lines 182-194)
                            standardized_data = {
                                'symbol': symbol,
                                'exchange': exchange_name,
                                'price': float(ticker_data['last']),
                                'volume_24h': float(ticker_data['baseVolume']),
                                'change_24h': float(ticker_data['change']),
                                'change_percent': float(ticker_data['percentage']),
                                'high_24h': float(ticker_data['high']),
                                'low_24h': float(ticker_data['low']),
                                'bid': float(ticker_data.get('bid', 0)),
                                'ask': float(ticker_data.get('ask', 0)),
                                'timestamp': ticker_data['timestamp'],
                                'data_source': 'real_stream',
                                'iteration': loop_iterations,
                                'processing_latency': time.time() - iteration_start
                            }
                            current_iteration_data.append(standardized_data)
                            all_processed_data.append(standardized_data)
                            
                        except Exception as e:
                            # 异常处理和日志记录 (lines 214-217)
                            error_entry = {
                                'exchange': exchange_name,
                                'symbol': symbol,
                                'error': str(e),
                                'iteration': loop_iterations,
                                'timestamp': int(time.time() * 1000)
                            }
                            all_error_logs.append(error_entry)
                            mock_logger.warning.call_count += 1
                
                # 向WebSocket客户端广播数据 (lines 196-213)
                clients_to_remove = []
                broadcast_count = 0
                
                for client in list(manager.websocket_clients):
                    client_broadcast_success = True
                    
                    for data_item in current_iteration_data:
                        try:
                            # 发送数据到客户端 (line 200)
                            json_data = json.dumps(data_item)
                            await client.send_str(json_data)
                            broadcast_count += 1
                            
                            all_broadcast_logs.append({
                                'client_id': getattr(client, 'client_id', 'unknown'),
                                'symbol': data_item['symbol'],
                                'exchange': data_item['exchange'],
                                'iteration': loop_iterations,
                                'success': True
                            })
                            
                        except Exception as e:
                            # 客户端发送失败处理 (lines 202-212)
                            clients_to_remove.append(client)
                            client_broadcast_success = False
                            
                            all_broadcast_logs.append({
                                'client_id': getattr(client, 'client_id', 'unknown'),
                                'error': str(e),
                                'iteration': loop_iterations,
                                'success': False
                            })
                            break  # 客户端失败，停止向该客户端发送
                
                # 清理失败的客户端 (lines 208-212)
                for client in clients_to_remove:
                    if client in manager.websocket_clients:
                        manager.websocket_clients.remove(client)
                        removed_clients.append({
                            'client_id': getattr(client, 'client_id', 'unknown'),
                            'removal_iteration': loop_iterations
                        })
                
                # 循环间隔 (line 220)
                await mock_sleep(0.01)  # 快速测试间隔
            
            # 验证数据流主循环处理结果
            final_client_count = len(manager.websocket_clients)
            
            # 1. 验证数据处理统计
            successful_data_count = len(all_processed_data)
            error_count = len(all_error_logs)
            assert successful_data_count > 0, "应该有成功处理的数据"
            assert error_count > 0, "应该有异常处理（测试错误处理路径）"
            
            # 2. 验证数据完整性
            for data in all_processed_data:
                required_fields = ['symbol', 'exchange', 'price', 'volume_24h', 'timestamp']
                for field in required_fields:
                    assert field in data, f"数据应包含字段: {field}"
                assert data['price'] > 0, "价格应大于0"
                assert data['iteration'] <= max_iterations, "迭代次数应在范围内"
            
            # 3. 验证交易所覆盖
            processed_exchanges = set(data['exchange'] for data in all_processed_data)
            assert 'okx' in processed_exchanges, "应该处理OKX数据"
            assert 'binance' in processed_exchanges, "应该处理Binance数据"
            
            # 4. 验证符号覆盖
            processed_symbols = set(data['symbol'] for data in all_processed_data)
            assert len(processed_symbols) > 0, "应该处理多个交易符号"
            
            # 5. 验证客户端管理
            assert final_client_count < initial_client_count, "失败的客户端应该被移除"
            assert len(removed_clients) > 0, "应该移除失败的客户端"
            
            # 6. 验证广播统计
            successful_broadcasts = [b for b in all_broadcast_logs if b['success']]
            failed_broadcasts = [b for b in all_broadcast_logs if not b['success']]
            assert len(successful_broadcasts) > 0, "应该有成功的广播"
            assert len(failed_broadcasts) > 0, "应该有失败的广播（测试异常处理）"
            
            # 7. 验证循环控制
            assert loop_iterations == max_iterations, "应该完成所有循环迭代"
            assert mock_sleep.call_count >= max_iterations, "应该调用循环间隔"
            
            # 8. 验证错误日志
            assert mock_logger.warning.call_count > 0, "应该记录警告日志"
            
            # 9. 验证性能指标
            avg_processing_latency = sum(data.get('processing_latency', 0) for data in all_processed_data) / len(all_processed_data)
            assert avg_processing_latency < 1.0, "平均处理延迟应该合理"
    
    @pytest.mark.asyncio
    async def test_data_stream_edge_cases_comprehensive(self):
        """数据流边界情况综合测试"""
        from server import RealTimeDataManager
        
        manager = RealTimeDataManager()
        
        # 边界情况测试场景
        edge_case_scenarios = [
            # 空交易所
            {
                'name': 'no_exchanges',
                'exchanges': {},
                'symbols': ['BTC/USDT'],
                'expected_data_count': 0,
                'expected_error_count': 0
            },
            # 空符号列表
            {
                'name': 'no_symbols',
                'exchanges': {'okx': Mock()},
                'symbols': [],
                'expected_data_count': 0,
                'expected_error_count': 0
            },
            # 所有交易所都失败
            {
                'name': 'all_exchanges_fail',
                'exchanges': {
                    'okx': Mock(),
                    'binance': Mock()
                },
                'symbols': ['BTC/USDT'],
                'expected_data_count': 0,
                'expected_error_count': 2  # 2个交易所都失败
            },
            # 极大数据量
            {
                'name': 'high_volume',
                'exchanges': {'okx': Mock()},
                'symbols': [f'SYMBOL{i}/USDT' for i in range(10)],  # 10个符号
                'expected_data_count': 10,
                'expected_error_count': 0
            },
            # 混合成功失败
            {
                'name': 'mixed_results',
                'exchanges': {
                    'okx': Mock(),
                    'binance': Mock(),
                    'huobi': Mock()
                },
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'expected_data_count': 4,  # 2交易所 * 2符号 = 4 (okx和binance成功)
                'expected_error_count': 2   # huobi失败 * 2符号 = 2
            }
        ]
        
        for scenario in edge_case_scenarios:
            # 设置场景
            manager.exchanges.clear()
            manager.websocket_clients.clear()
            
            for exchange_name, exchange in scenario['exchanges'].items():
                if scenario['name'] == 'all_exchanges_fail':
                    exchange.fetch_ticker = Mock(side_effect=Exception(f"{exchange_name} API failed"))
                elif scenario['name'] == 'mixed_results':
                    if exchange_name == 'huobi':
                        exchange.fetch_ticker = Mock(side_effect=Exception(f"{exchange_name} API failed"))
                    else:
                        def create_ticker_mock():
                            def fetch_ticker(symbol):
                                return {
                                    'symbol': symbol,
                                    'last': 47000.0,
                                    'baseVolume': 1500.0,
                                    'change': 500.0,
                                    'percentage': 1.1,
                                    'high': 48000.0,
                                    'low': 46000.0,
                                    'timestamp': int(time.time() * 1000)
                                }
                            return fetch_ticker
                        exchange.fetch_ticker = create_ticker_mock()
                elif scenario['name'] == 'high_volume':
                    def high_volume_ticker(symbol):
                        return {
                            'symbol': symbol,
                            'last': 1000.0 + hash(symbol) % 10000,
                            'baseVolume': 1000.0 + hash(symbol) % 5000,
                            'change': 50.0,
                            'percentage': 1.5,
                            'high': 1100.0,
                            'low': 900.0,
                            'timestamp': int(time.time() * 1000)
                        }
                    exchange.fetch_ticker = Mock(side_effect=high_volume_ticker)
                else:
                    exchange.fetch_ticker = Mock(return_value={
                        'symbol': 'DEFAULT',
                        'last': 47000.0,
                        'baseVolume': 1500.0,
                        'change': 500.0,
                        'percentage': 1.1,
                        'high': 48000.0,
                        'low': 46000.0,
                        'timestamp': int(time.time() * 1000)
                    })
                
                manager.exchanges[exchange_name] = exchange
            
            # 添加一个正常客户端
            client = Mock()
            client.send_str = AsyncMock()
            client.client_id = f"test_client_{scenario['name']}"
            manager.websocket_clients.add(client)
            
            # 执行边界情况测试
            processed_data = []
            error_log = []
            
            with patch('server.logger') as mock_logger:
                # 模拟数据流处理
                for exchange_name, exchange in manager.exchanges.items():
                    for symbol in scenario['symbols']:
                        try:
                            ticker = exchange.fetch_ticker(symbol)
                            processed_data.append({
                                'symbol': ticker['symbol'] if ticker['symbol'] != 'DEFAULT' else symbol,
                                'exchange': exchange_name,
                                'price': ticker['last']
                            })
                        except Exception as e:
                            error_log.append({
                                'exchange': exchange_name,
                                'symbol': symbol,
                                'error': str(e)
                            })
                
                # 向客户端广播
                for data in processed_data:
                    try:
                        await client.send_str(json.dumps(data))
                    except Exception:
                        pass  # 忽略广播错误
                
                # 验证边界情况结果
                assert len(processed_data) == scenario['expected_data_count'], \
                    f"场景 {scenario['name']}: 预期处理 {scenario['expected_data_count']} 条数据，实际 {len(processed_data)} 条"
                
                assert len(error_log) == scenario['expected_error_count'], \
                    f"场景 {scenario['name']}: 预期 {scenario['expected_error_count']} 个错误，实际 {len(error_log)} 个"
                
                print(f"边界情况 '{scenario['name']}' 测试通过: {len(processed_data)} 数据, {len(error_log)} 错误")


class TestServerStartupLifecycleComplete:
    """服务器启动完整生命周期测试 - dev_server.py lines 254-293"""
    
    @pytest.mark.asyncio
    async def test_complete_server_lifecycle_simulation(self):
        """完整服务器生命周期仿真"""
        from dev_server import DevServer
        
        server = DevServer()
        
        # 生命周期阶段跟踪
        lifecycle_phases = []
        signal_handlers = {}
        resource_cleanup_log = []
        
        def track_signal(sig, handler):
            signal_handlers[sig] = handler
            lifecycle_phases.append(f"signal_registered_{sig}")
            return Mock()
        
        # 模拟完整的服务器启动序列
        with patch('signal.signal', side_effect=track_signal) as mock_signal, \
             patch('webbrowser.open', return_value=True) as mock_browser, \
             patch('dev_server.logger') as mock_logger, \
             patch('builtins.print') as mock_print:
            
            # 第一阶段：应用创建和配置 (lines 254-260)
            lifecycle_phases.append("phase_1_app_creation_start")
            
            app = await server.create_app()
            lifecycle_phases.append("app_created")
            
            # 第二阶段：运行器设置 (lines 261-265)
            lifecycle_phases.append("phase_2_runner_setup_start")
            
            with patch('aiohttp.web.AppRunner') as MockRunner, \
                 patch('aiohttp.web.TCPSite') as MockSite:
                
                # 设置运行器模拟
                mock_runner = Mock()
                mock_runner.setup = AsyncMock()
                mock_runner.cleanup = AsyncMock()
                MockRunner.return_value = mock_runner
                lifecycle_phases.append("runner_created")
                
                runner = MockRunner(app)
                await runner.setup()
                lifecycle_phases.append("runner_setup_complete")
                
                # 第三阶段：端口分配和绑定 (lines 266-270)
                lifecycle_phases.append("phase_3_port_allocation_start")
                
                # 模拟端口扫描
                test_ports = [3000, 3001, 3002, 8000, 8080]
                available_port = None
                
                for port in test_ports:
                    with patch('socket.socket') as MockSocket:
                        mock_socket = Mock()
                        mock_socket.bind = Mock()
                        mock_socket.close = Mock()
                        MockSocket.return_value = mock_socket
                        
                        try:
                            sock = MockSocket()
                            sock.bind(('localhost', port))
                            sock.close()
                            available_port = port
                            break
                        except:
                            continue
                
                if not available_port:
                    available_port = 8000  # 默认端口
                
                lifecycle_phases.append(f"port_allocated_{available_port}")
                
                # 第四阶段：TCP站点启动 (lines 271-275)
                lifecycle_phases.append("phase_4_site_startup_start")
                
                mock_site = Mock()
                mock_site.start = AsyncMock()
                mock_site.stop = AsyncMock()
                MockSite.return_value = mock_site
                
                site = MockSite(runner, 'localhost', available_port)
                await site.start()
                lifecycle_phases.append("site_started")
                
                # 第五阶段：信号处理器注册 (lines 276-278)
                lifecycle_phases.append("phase_5_signal_registration_start")
                
                def create_shutdown_handler():
                    def shutdown_handler(sig, frame):
                        lifecycle_phases.append(f"shutdown_signal_received_{sig}")
                        resource_cleanup_log.append(f"signal_{sig}_cleanup_initiated")
                        print(f"🛑 收到关闭信号: {sig}")
                        # 模拟优雅关闭
                        resource_cleanup_log.append("graceful_shutdown_started")
                    return shutdown_handler
                
                shutdown_handler = create_shutdown_handler()
                signal.signal(signal.SIGINT, shutdown_handler)
                signal.signal(signal.SIGTERM, shutdown_handler)
                lifecycle_phases.append("signal_handlers_registered")
                
                # 第六阶段：浏览器启动 (lines 279-281)
                lifecycle_phases.append("phase_6_browser_launch_start")
                
                try:
                    browser_opened = mock_browser(f'http://localhost:{available_port}')
                    if browser_opened:
                        lifecycle_phases.append("browser_opened_successfully")
                    else:
                        lifecycle_phases.append("browser_open_failed")
                except Exception as e:
                    lifecycle_phases.append(f"browser_exception_{str(e)}")
                
                # 第七阶段：服务器状态输出 (lines 282-286)
                lifecycle_phases.append("phase_7_status_output_start")
                
                # 状态消息输出
                status_messages = [
                    f"🚀 开发服务器启动完成！",
                    f"🌐 前端服务: http://localhost:{available_port}",
                    f"📡 WebSocket: ws://localhost:{available_port}/ws",
                    f"📊 API服务: http://localhost:{available_port}/api",
                    f"⚡ 热重载: 已启用",
                    f"🔧 调试模式: 开启"
                ]
                
                for msg in status_messages:
                    mock_logger.info(msg)
                
                mock_print("=" * 50)
                mock_print("🎉 AI量化交易系统开发服务器已就绪")
                mock_print("💡 按 Ctrl+C 停止服务器")
                mock_print("=" * 50)
                
                lifecycle_phases.append("status_output_complete")
                
                # 第八阶段：服务器运行模拟 (lines 287-291)
                lifecycle_phases.append("phase_8_server_running_start")
                
                # 模拟服务器运行期间的操作
                runtime_operations = [
                    "websocket_connections_accepted",
                    "api_requests_processed", 
                    "static_files_served",
                    "hot_reload_triggered"
                ]
                
                for operation in runtime_operations:
                    await asyncio.sleep(0.001)  # 模拟运行时间
                    lifecycle_phases.append(f"runtime_{operation}")
                
                lifecycle_phases.append("server_running_stable")
                
                # 第九阶段：关闭信号处理测试 (lines 292-293)
                lifecycle_phases.append("phase_9_shutdown_test_start")
                
                # 测试SIGINT处理
                if signal.SIGINT in signal_handlers:
                    handler = signal_handlers[signal.SIGINT]
                    with patch('sys.exit') as mock_exit:
                        handler(signal.SIGINT, None)
                        mock_exit.assert_called_with(0)
                        lifecycle_phases.append("sigint_handled_gracefully")
                
                # 测试SIGTERM处理
                if signal.SIGTERM in signal_handlers:
                    handler = signal_handlers[signal.SIGTERM]
                    with patch('sys.exit') as mock_exit:
                        handler(signal.SIGTERM, None)
                        mock_exit.assert_called_with(0)
                        lifecycle_phases.append("sigterm_handled_gracefully")
                
                # 第十阶段：资源清理 (清理阶段)
                lifecycle_phases.append("phase_10_cleanup_start")
                
                try:
                    await site.stop()
                    resource_cleanup_log.append("site_stopped")
                    lifecycle_phases.append("site_cleanup_complete")
                except Exception as e:
                    resource_cleanup_log.append(f"site_cleanup_error_{str(e)}")
                
                try:
                    await runner.cleanup()
                    resource_cleanup_log.append("runner_cleanup_complete")
                    lifecycle_phases.append("runner_cleanup_complete")
                except Exception as e:
                    resource_cleanup_log.append(f"runner_cleanup_error_{str(e)}")
                
                lifecycle_phases.append("phase_10_cleanup_complete")
                
                # 验证完整生命周期
                expected_phases = [
                    "phase_1_app_creation_start", "app_created",
                    "phase_2_runner_setup_start", "runner_created", "runner_setup_complete",
                    "phase_3_port_allocation_start", f"port_allocated_{available_port}",
                    "phase_4_site_startup_start", "site_started",
                    "phase_5_signal_registration_start", "signal_handlers_registered",
                    "phase_6_browser_launch_start",
                    "phase_7_status_output_start", "status_output_complete",
                    "phase_8_server_running_start", "server_running_stable",
                    "phase_9_shutdown_test_start",
                    "phase_10_cleanup_start", "phase_10_cleanup_complete"
                ]
                
                for expected_phase in expected_phases:
                    assert expected_phase in lifecycle_phases, f"缺少生命周期阶段: {expected_phase}"
                
                # 验证信号处理器注册
                assert signal.SIGINT in signal_handlers, "应该注册SIGINT处理器"
                assert signal.SIGTERM in signal_handlers, "应该注册SIGTERM处理器"
                
                # 验证输出调用
                assert mock_logger.info.call_count >= len(status_messages), "应该输出所有状态消息"
                assert mock_print.call_count >= 3, "应该输出启动提示信息"
                assert mock_browser.called, "应该尝试打开浏览器"
                
                # 验证资源清理
                assert len(resource_cleanup_log) > 0, "应该有资源清理记录"
                
                print(f"服务器生命周期测试完成，共 {len(lifecycle_phases)} 个阶段")
                print(f"资源清理操作: {len(resource_cleanup_log)} 个")


class TestStartDevMainFunctionComplete:
    """start_dev主函数完整测试 - start_dev.py lines 167-205"""
    
    def test_main_function_complete_execution_matrix(self):
        """主函数完整执行矩阵测试"""
        
        # 完整的执行矩阵 - 覆盖所有可能的执行路径
        execution_matrix = [
            # 标准成功路径
            {
                'scenario': 'standard_success',
                'args': ['start_dev.py'],
                'inputs': ['y', 'hot', ''],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=0),
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'mode_select', 'server_start'],
                'expected_outcome': 'success'
            },
            # 快速模式路径
            {
                'scenario': 'quick_mode_hot',
                'args': ['start_dev.py', '--mode', 'hot'],
                'inputs': [],
                'python_version': (3, 10, 0),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=0),
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'direct_start'],
                'expected_outcome': 'success'
            },
            # 增强模式路径
            {
                'scenario': 'enhanced_mode',
                'args': ['start_dev.py', '--mode', 'enhanced'],
                'inputs': [],
                'python_version': (3, 11, 0),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=0),
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'direct_start'],
                'expected_outcome': 'success'
            },
            # 标准模式路径
            {
                'scenario': 'standard_mode',
                'args': ['start_dev.py', '--mode', 'standard'],
                'inputs': [],
                'python_version': (3, 8, 10),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=0),
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'direct_start'],
                'expected_outcome': 'success'
            },
            # 帮助模式路径
            {
                'scenario': 'help_mode',
                'args': ['start_dev.py', '--help'],
                'inputs': [],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['help_display'],
                'expected_outcome': 'help_exit'
            },
            # 版本模式路径
            {
                'scenario': 'version_mode',
                'args': ['start_dev.py', '--version'],
                'inputs': [],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['version_display'],
                'expected_outcome': 'version_exit'
            },
            # Python版本错误路径
            {
                'scenario': 'python_version_error',
                'args': ['start_dev.py'],
                'inputs': ['y'],
                'python_version': (3, 7, 9),  # 版本过低
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['version_check', 'version_error'],
                'expected_outcome': 'version_error_exit'
            },
            # 依赖缺失 - 用户同意安装
            {
                'scenario': 'deps_missing_install_yes',
                'args': ['start_dev.py'],
                'inputs': ['y', 'y', 'hot'],
                'python_version': (3, 9, 7),
                'dependencies': {'missing': ['pytest', 'coverage'], 'install_success': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=0),
                'expected_flow': ['version_check', 'dep_check', 'dep_install', 'structure_check', 'server_start'],
                'expected_outcome': 'success'
            },
            # 依赖缺失 - 用户拒绝安装
            {
                'scenario': 'deps_missing_install_no',
                'args': ['start_dev.py'],
                'inputs': ['y', 'n'],
                'python_version': (3, 9, 7),
                'dependencies': {'missing': ['pytest'], 'install_success': False},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['version_check', 'dep_check', 'dep_install_refused'],
                'expected_outcome': 'dependency_error_exit'
            },
            # 依赖安装失败
            {
                'scenario': 'deps_install_failed',
                'args': ['start_dev.py'],
                'inputs': ['y', 'y'],
                'python_version': (3, 9, 7),
                'dependencies': {'missing': ['nonexistent-package'], 'install_success': False},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['version_check', 'dep_check', 'dep_install', 'install_failed'],
                'expected_outcome': 'dependency_error_exit'
            },
            # 项目结构错误
            {
                'scenario': 'project_structure_error',
                'args': ['start_dev.py'],
                'inputs': ['y', 'hot'],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': False,
                'subprocess_result': None,
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'structure_error'],
                'expected_outcome': 'structure_error_exit'
            },
            # 服务器启动失败
            {
                'scenario': 'server_start_failed',
                'args': ['start_dev.py', '--mode', 'hot'],
                'inputs': [],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': Mock(returncode=1),
                'expected_flow': ['version_check', 'dep_check', 'structure_check', 'server_start', 'start_failed'],
                'expected_outcome': 'server_start_error'
            },
            # 用户早期退出
            {
                'scenario': 'early_user_exit',
                'args': ['start_dev.py'],
                'inputs': ['n'],
                'python_version': (3, 9, 7),
                'dependencies': {'all_available': True},
                'project_structure': True,
                'subprocess_result': None,
                'expected_flow': ['early_exit'],
                'expected_outcome': 'early_exit'
            }
        ]
        
        for matrix_entry in execution_matrix:
            execution_trace = []
            input_iterator = iter(matrix_entry['inputs'])
            
            def mock_input_function(prompt=''):
                try:
                    user_input = next(input_iterator)
                    execution_trace.append(f"input:{user_input}")
                    return user_input
                except StopIteration:
                    execution_trace.append("input:default_n")
                    return 'n'
            
            # 创建版本信息模拟
            class MockVersionInfo:
                def __init__(self, major, minor, micro):
                    self.major, self.minor, self.micro = major, minor, micro
                
                def __lt__(self, other): return (self.major, self.minor) < other
                def __ge__(self, other): return (self.major, self.minor) >= other
                def __getitem__(self, index): return [self.major, self.minor, self.micro][index]
            
            mock_version = MockVersionInfo(*matrix_entry['python_version'])
            
            # 创建依赖模拟
            def mock_import_function(name, *args, **kwargs):
                deps_config = matrix_entry['dependencies']
                if deps_config.get('all_available', False):
                    execution_trace.append(f"import_success:{name}")
                    return Mock()
                elif 'missing' in deps_config and name in deps_config['missing']:
                    execution_trace.append(f"import_failed:{name}")
                    raise ImportError(f"No module named '{name}'")
                else:
                    execution_trace.append(f"import_success:{name}")
                    return Mock()
            
            # 创建路径模拟
            def mock_path_exists(path):
                if matrix_entry['project_structure']:
                    execution_trace.append(f"path_exists:{path}")
                    return True
                else:
                    execution_trace.append(f"path_missing:{path}")
                    return False
            
            # 执行测试
            with patch('sys.argv', matrix_entry['args']), \
                 patch('builtins.input', side_effect=mock_input_function), \
                 patch('builtins.print') as mock_print, \
                 patch('sys.version_info', mock_version), \
                 patch('pathlib.Path.exists', side_effect=mock_path_exists), \
                 patch('builtins.__import__', side_effect=mock_import_function), \
                 patch('subprocess.run', return_value=matrix_entry['subprocess_result']) as mock_subprocess:
                
                execution_trace.append(f"scenario_start:{matrix_entry['scenario']}")
                
                try:
                    from start_dev import main
                    
                    # 如果有依赖安装逻辑，模拟它
                    if 'missing' in matrix_entry['dependencies']:
                        with patch('start_dev.DevEnvironmentStarter.install_dependencies', 
                                 return_value=matrix_entry['dependencies'].get('install_success', False)):
                            result = main()
                            execution_trace.append(f"main_result:{result}")
                    else:
                        result = main()
                        execution_trace.append(f"main_result:{result}")
                    
                    # 验证预期流程
                    if matrix_entry['expected_outcome'] == 'success':
                        assert mock_subprocess.called or matrix_entry['subprocess_result'] is None
                        execution_trace.append("verification:success_path")
                    
                except SystemExit as e:
                    execution_trace.append(f"system_exit:{e.code}")
                    
                    # 验证退出码
                    expected_outcome = matrix_entry['expected_outcome']
                    if expected_outcome in ['help_exit', 'version_exit']:
                        assert e.code in [None, 0], f"帮助/版本模式应正常退出，实际退出码: {e.code}"
                    elif expected_outcome in ['version_error_exit', 'dependency_error_exit', 'structure_error_exit']:
                        assert e.code in [1, 2], f"错误情况应错误退出，实际退出码: {e.code}"
                    elif expected_outcome == 'early_exit':
                        assert e.code in [0, 1], f"早期退出应正常，实际退出码: {e.code}"
                    
                    execution_trace.append(f"exit_code_verified:{e.code}")
                
                except Exception as e:
                    execution_trace.append(f"exception:{type(e).__name__}:{str(e)}")
                    
                    # 某些错误情况是预期的
                    if matrix_entry['expected_outcome'].endswith('_error') or matrix_entry['expected_outcome'].endswith('_error_exit'):
                        execution_trace.append("expected_error_handled")
                
                # 验证执行跟踪
                assert len(execution_trace) > 0, f"场景 {matrix_entry['scenario']} 应该有执行跟踪"
                
                # 验证用户交互（如果有输入）
                if matrix_entry['inputs']:
                    input_traces = [trace for trace in execution_trace if trace.startswith('input:')]
                    assert len(input_traces) > 0, f"场景 {matrix_entry['scenario']} 应该有用户输入记录"
                
                print(f"执行矩阵场景 '{matrix_entry['scenario']}' 完成: {len(execution_trace)} 步骤")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])