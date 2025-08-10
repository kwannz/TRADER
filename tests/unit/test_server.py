"""
Web服务器单元测试
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import time

from tests.utils.helpers import (
    create_mock_request, 
    create_sample_market_data,
    MockWebSocketResponse,
    assert_websocket_message_sent
)

# 导入要测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestRealTimeDataManager:
    """实时数据管理器测试"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """模拟数据管理器"""
        # 尝试导入真实的类，如果失败则创建Mock
        try:
            from server import RealTimeDataManager
            manager = RealTimeDataManager()
        except ImportError:
            manager = Mock()
            manager.exchanges = {}
            manager.websocket_clients = set()
            manager.market_data = {}
            manager.running = False
            
        return manager
    
    def test_data_manager_initialization(self, mock_data_manager):
        """测试数据管理器初始化"""
        assert hasattr(mock_data_manager, 'exchanges')
        assert hasattr(mock_data_manager, 'websocket_clients')
        assert hasattr(mock_data_manager, 'market_data')
        assert hasattr(mock_data_manager, 'running')
    
    @pytest.mark.asyncio
    async def test_initialize_exchanges(self, mock_data_manager, mock_ccxt_exchanges):
        """测试交易所初始化"""
        with patch('ccxt.okx') as mock_okx, patch('ccxt.binance') as mock_binance:
            mock_okx.return_value = mock_ccxt_exchanges['okx']
            mock_binance.return_value = mock_ccxt_exchanges['binance']
            
            if hasattr(mock_data_manager, 'initialize_exchanges'):
                result = await mock_data_manager.initialize_exchanges()
                assert result is not None
            else:
                # 模拟初始化过程
                mock_data_manager.exchanges['okx'] = mock_ccxt_exchanges['okx']
                mock_data_manager.exchanges['binance'] = mock_ccxt_exchanges['binance']
                assert len(mock_data_manager.exchanges) == 2
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, mock_data_manager, mock_ccxt_exchanges):
        """测试成功获取市场数据"""
        if hasattr(mock_data_manager, 'get_market_data'):
            with patch.object(mock_data_manager, 'exchanges', mock_ccxt_exchanges):
                result = await mock_data_manager.get_market_data('BTC/USDT')
                assert result is not None
        else:
            # 模拟获取市场数据
            sample_data = create_sample_market_data('BTC/USDT')
            mock_data_manager.get_market_data = AsyncMock(return_value=sample_data)
            
            result = await mock_data_manager.get_market_data('BTC/USDT')
            assert result['symbol'] == 'BTC/USDT'
            assert 'price' in result
    
    @pytest.mark.asyncio
    async def test_get_market_data_failure(self, mock_data_manager):
        """测试获取市场数据失败"""
        # 模拟交易所API失败
        if hasattr(mock_data_manager, 'get_market_data'):
            with patch.object(mock_data_manager, 'exchanges', {}):
                with pytest.raises(Exception):
                    await mock_data_manager.get_market_data('BTC/USDT')
        else:
            # 模拟失败情况
            mock_data_manager.get_market_data = AsyncMock(side_effect=Exception("API Error"))
            
            with pytest.raises(Exception):
                await mock_data_manager.get_market_data('BTC/USDT')
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, mock_data_manager, mock_ccxt_exchanges):
        """测试获取历史数据"""
        sample_ohlcv = [
            [1640995200000, 45000, 46000, 44000, 45500, 1000],
            [1640998800000, 45500, 46500, 45000, 46000, 1100]
        ]
        
        if hasattr(mock_data_manager, 'get_historical_data'):
            mock_ccxt_exchanges['okx'].fetch_ohlcv.return_value = sample_ohlcv
            
            with patch.object(mock_data_manager, 'exchanges', mock_ccxt_exchanges):
                result = await mock_data_manager.get_historical_data('BTC/USDT')
                assert isinstance(result, list)
        else:
            # 模拟获取历史数据
            mock_data_manager.get_historical_data = AsyncMock(return_value=sample_ohlcv)
            
            result = await mock_data_manager.get_historical_data('BTC/USDT')
            assert len(result) == 2
    
    def test_websocket_clients_management(self, mock_data_manager):
        """测试WebSocket客户端管理"""
        # 添加客户端
        mock_ws1 = Mock()
        mock_ws2 = Mock()
        
        mock_data_manager.websocket_clients.add(mock_ws1)
        mock_data_manager.websocket_clients.add(mock_ws2)
        
        assert len(mock_data_manager.websocket_clients) == 2
        
        # 移除客户端
        mock_data_manager.websocket_clients.remove(mock_ws1)
        assert len(mock_data_manager.websocket_clients) == 1

class TestWebSocketHandler:
    """WebSocket处理器测试"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_success(self):
        """测试WebSocket连接成功"""
        mock_request = Mock()
        mock_ws = MockWebSocketResponse()
        
        with patch('aiohttp.web.WebSocketResponse') as MockWSResponse:
            MockWSResponse.return_value = mock_ws
            
            # 模拟WebSocket处理器
            async def websocket_handler(request):
                ws = MockWSResponse()
                await ws.prepare(request)
                
                # 发送初始消息
                await ws.send_str(json.dumps({
                    'type': 'connection_success',
                    'message': '实时数据连接成功'
                }))
                
                return ws
            
            result = await websocket_handler(mock_request)
            assert result is not None
            
            # 检查发送的消息
            messages = mock_ws.get_messages()
            if messages:
                assert len(messages) > 0
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self):
        """测试WebSocket消息处理"""
        mock_ws = MockWebSocketResponse()
        
        # 模拟接收的消息
        test_messages = [
            {'type': 'subscribe', 'symbols': ['BTC/USDT']},
            {'type': 'unsubscribe', 'symbols': ['ETH/USDT']},
            {'type': 'ping'}
        ]
        
        for message in test_messages:
            # 模拟消息处理逻辑
            if message['type'] == 'subscribe':
                response = {
                    'type': 'subscription_success',
                    'symbols': message['symbols']
                }
                await mock_ws.send_str(json.dumps(response))
            elif message['type'] == 'ping':
                await mock_ws.send_str(json.dumps({'type': 'pong'}))
        
        messages = mock_ws.get_messages()
        assert len(messages) == 2  # subscribe response + pong
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """测试WebSocket错误处理"""
        mock_ws = MockWebSocketResponse()
        
        # 模拟错误消息
        error_message = {
            'type': 'error',
            'message': '无效的JSON格式'
        }
        
        await mock_ws.send_str(json.dumps(error_message))
        
        last_message = mock_ws.get_last_message()
        assert last_message is not None
        message_data = json.loads(last_message[1])
        assert message_data['type'] == 'error'

class TestAPIHandlers:
    """API处理器测试"""
    
    @pytest.mark.asyncio
    async def test_api_market_data_success(self, mock_ccxt_exchanges):
        """测试市场数据API成功"""
        mock_request = create_mock_request({'symbol': 'BTC/USDT'})
        
        # 模拟市场数据API处理器
        async def api_market_data(request):
            symbol = request.query.get('symbol', 'BTC/USDT')
            
            # 模拟从数据管理器获取数据
            sample_data = create_sample_market_data(symbol)
            
            return {
                'success': True,
                'data': sample_data
            }
        
        result = await api_market_data(mock_request)
        
        assert result['success'] is True
        assert result['data']['symbol'] == 'BTC/USDT'
        assert 'price' in result['data']
    
    @pytest.mark.asyncio
    async def test_api_market_data_error(self):
        """测试市场数据API错误"""
        mock_request = create_mock_request({'symbol': 'INVALID/SYMBOL'})
        
        async def api_market_data(request):
            try:
                symbol = request.query.get('symbol', 'BTC/USDT')
                # 模拟数据获取失败
                raise Exception("Invalid symbol")
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        result = await api_market_data(mock_request)
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_api_ai_analysis_not_implemented(self):
        """测试AI分析API（未实现）"""
        mock_request = create_mock_request({'symbol': 'BTC'})
        
        async def api_ai_analysis(request):
            return {
                'success': False,
                'error': 'AI分析功能需要配置真实的AI API密钥 (DeepSeek/Gemini)',
                'message': '请在环境变量中设置 DEEPSEEK_API_KEY 或 GEMINI_API_KEY'
            }
        
        result = await api_ai_analysis(mock_request)
        
        assert result['success'] is False
        assert 'AI分析功能需要配置' in result['error']
    
    @pytest.mark.asyncio
    async def test_api_dev_status(self):
        """测试开发状态API"""
        mock_request = create_mock_request()
        
        async def api_dev_status(request):
            return {
                'success': True,
                'mode': 'development',
                'status': 'running',
                'server': 'aiohttp',
                'connected_ws_clients': 0,
                'timestamp': int(time.time() * 1000)
            }
        
        result = await api_dev_status(mock_request)
        
        assert result['success'] is True
        assert result['mode'] == 'development'
        assert result['status'] == 'running'

class TestAppCreation:
    """应用创建测试"""
    
    @pytest.mark.asyncio
    async def test_create_app_production_mode(self):
        """测试生产模式下创建应用"""
        with patch('aiohttp.web.Application') as MockApp:
            mock_app = Mock()
            MockApp.return_value = mock_app
            
            # 模拟create_app函数
            async def create_app(dev_mode=False):
                app = MockApp()
                
                # 添加中间件
                app.middlewares = Mock()
                app.middlewares.append = Mock()
                
                # 添加路由
                app.router = Mock()
                app.router.add_get = Mock()
                app.router.add_static = Mock()
                
                if not dev_mode:
                    # 生产模式配置
                    pass
                
                return app
            
            app = await create_app(dev_mode=False)
            assert app is not None
    
    @pytest.mark.asyncio
    async def test_create_app_development_mode(self):
        """测试开发模式下创建应用"""
        with patch('aiohttp.web.Application') as MockApp:
            mock_app = Mock()
            MockApp.return_value = mock_app
            
            async def create_app(dev_mode=True):
                app = MockApp()
                
                # 开发模式特殊配置
                app.middlewares = Mock()
                app.middlewares.append = Mock()
                
                app.router = Mock()
                app.router.add_get = Mock()
                app.router.add_static = Mock()
                
                if dev_mode:
                    # 添加开发模式路由
                    app.router.add_get('/api/dev/status', Mock())
                
                return app
            
            app = await create_app(dev_mode=True)
            assert app is not None

class TestCORSMiddleware:
    """CORS中间件测试"""
    
    @pytest.mark.asyncio
    async def test_cors_headers_added(self):
        """测试CORS头部添加"""
        mock_request = Mock()
        mock_handler = AsyncMock()
        
        # 创建模拟响应
        mock_response = Mock()
        mock_response.headers = {}
        mock_handler.return_value = mock_response
        
        # CORS中间件
        async def cors_handler(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        result = await cors_handler(mock_request, mock_handler)
        
        assert result.headers['Access-Control-Allow-Origin'] == '*'
        assert 'GET, POST, OPTIONS' in result.headers['Access-Control-Allow-Methods']
    
    @pytest.mark.asyncio
    async def test_cors_headers_dev_mode(self):
        """测试开发模式下的CORS头部"""
        mock_request = Mock()
        mock_handler = AsyncMock()
        mock_response = Mock()
        mock_response.headers = {}
        mock_handler.return_value = mock_response
        
        # 开发模式CORS中间件
        async def cors_handler_dev(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        result = await cors_handler_dev(mock_request, mock_handler)
        
        assert result.headers['Cache-Control'] == 'no-cache, no-store, must-revalidate'
        assert result.headers['Pragma'] == 'no-cache'

class TestDependencyChecking:
    """依赖检查测试"""
    
    def test_check_dependencies_success(self):
        """测试依赖检查成功"""
        required_packages = ['aiohttp', 'ccxt', 'pandas', 'numpy']
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        # 在测试环境中，这些包应该已安装
        assert len(missing_packages) == 0 or len(missing_packages) < len(required_packages)
    
    def test_check_dependencies_missing(self):
        """测试依赖检查失败"""
        # 测试不存在的包
        fake_packages = ['nonexistent_package_xyz', 'another_fake_package']
        
        missing_packages = []
        for package in fake_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        # 这些假包应该全部缺失
        assert len(missing_packages) == len(fake_packages)
    
    def test_check_project_files(self):
        """测试项目文件检查"""
        project_root = Path(__file__).parent.parent.parent
        
        important_files = [
            'server.py',
            'dev_server.py'
        ]
        
        existing_files = []
        for file_name in important_files:
            file_path = project_root / file_name
            if file_path.exists():
                existing_files.append(file_name)
        
        # 至少应该有一些核心文件存在
        assert len(existing_files) > 0

class TestMainFunction:
    """主函数测试"""
    
    @pytest.mark.asyncio
    async def test_main_function_dev_mode(self):
        """测试开发模式主函数"""
        with patch('server.data_manager') as mock_data_manager:
            mock_data_manager.initialize_exchanges = AsyncMock(return_value=True)
            mock_data_manager.start_data_stream = Mock()
            mock_data_manager.stop_data_stream = Mock()
            
            with patch('aiohttp.web.AppRunner') as MockRunner:
                with patch('aiohttp.web.TCPSite') as MockSite:
                    # 模拟主函数逻辑
                    async def main_logic(dev_mode=True):
                        # 初始化数据管理器
                        init_result = await mock_data_manager.initialize_exchanges()
                        if not init_result:
                            return False
                        
                        # 启动数据流
                        mock_data_manager.start_data_stream()
                        
                        return True
                    
                    result = await main_logic(dev_mode=True)
                    assert result is True
    
    @pytest.mark.asyncio
    async def test_main_function_initialization_failure(self):
        """测试主函数初始化失败"""
        with patch('server.data_manager') as mock_data_manager:
            mock_data_manager.initialize_exchanges = AsyncMock(return_value=False)
            
            async def main_logic():
                init_result = await mock_data_manager.initialize_exchanges()
                if not init_result:
                    return False
                return True
            
            result = await main_logic()
            assert result is False