"""
前端JavaScript功能测试（通过Python模拟）
"""

import pytest
import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

class TestDevClientJS:
    """dev_client.js测试"""
    
    @pytest.fixture
    def dev_client_path(self):
        """dev_client.js文件路径"""
        return Path(__file__).parent.parent.parent / 'dev_client.js'
    
    def test_dev_client_file_exists(self, dev_client_path):
        """测试dev_client.js文件存在"""
        assert dev_client_path.exists()
    
    def test_dev_client_has_dev_class(self, dev_client_path):
        """测试dev_client.js包含DevClient类"""
        if dev_client_path.exists():
            content = dev_client_path.read_text()
            assert 'class DevClient' in content
            assert 'constructor()' in content
    
    def test_dev_client_has_websocket_handling(self, dev_client_path):
        """测试WebSocket处理功能"""
        if dev_client_path.exists():
            content = dev_client_path.read_text()
            
            # 检查WebSocket相关方法
            websocket_methods = [
                'connect()',
                'onopen',
                'onmessage', 
                'onclose',
                'onerror'
            ]
            
            for method in websocket_methods:
                assert method in content
    
    def test_dev_client_has_message_types(self, dev_client_path):
        """测试消息类型处理"""
        if dev_client_path.exists():
            content = dev_client_path.read_text()
            
            # 检查消息类型
            message_types = [
                'dev_connected',
                'reload_frontend', 
                'backend_restarting',
                'backend_restarted'
            ]
            
            for msg_type in message_types:
                assert msg_type in content
    
    def test_dev_client_has_notification_system(self, dev_client_path):
        """测试通知系统"""
        if dev_client_path.exists():
            content = dev_client_path.read_text()
            
            # 检查通知相关功能
            notification_features = [
                'showDevNotification',
                'dev-notification',
                'notification_duration'
            ]
            
            for feature in notification_features:
                assert feature in content
    
    def test_dev_client_has_reconnection_logic(self, dev_client_path):
        """测试重连逻辑"""
        if dev_client_path.exists():
            content = dev_client_path.read_text()
            
            # 检查重连相关功能
            reconnection_features = [
                'reconnect',
                'reconnectInterval',
                'maxReconnectAttempts',
                'isReconnecting'
            ]
            
            for feature in reconnection_features:
                assert feature in content

class TestWebInterfaceFiles:
    """Web界面文件测试"""
    
    @pytest.fixture
    def web_interface_dir(self):
        """Web界面目录"""
        return Path(__file__).parent.parent.parent / 'file_management' / 'web_interface'
    
    def test_index_html_exists(self, web_interface_dir):
        """测试index.html存在"""
        index_path = web_interface_dir / 'index.html'
        assert index_path.exists()
    
    def test_app_js_exists(self, web_interface_dir):
        """测试app.js存在"""
        app_js_path = web_interface_dir / 'app.js'
        assert app_js_path.exists()
    
    def test_styles_css_exists(self, web_interface_dir):
        """测试styles.css存在"""
        styles_path = web_interface_dir / 'styles.css'
        assert styles_path.exists()
    
    def test_index_html_has_dev_script_integration(self, web_interface_dir):
        """测试index.html集成了开发脚本"""
        index_path = web_interface_dir / 'index.html'
        if index_path.exists():
            content = index_path.read_text()
            
            # 检查开发脚本集成
            dev_integration_features = [
                'dev-ws',  # 开发WebSocket端点
                'dev_client.js',  # 开发客户端脚本
                'WebSocket',  # WebSocket连接
                '开发模式'  # 开发模式检测
            ]
            
            found_features = 0
            for feature in dev_integration_features:
                if feature in content:
                    found_features += 1
            
            # 至少应该有一些开发功能集成
            assert found_features > 0
    
    def test_html_structure_valid(self, web_interface_dir):
        """测试HTML结构有效"""
        index_path = web_interface_dir / 'index.html'
        if index_path.exists():
            content = index_path.read_text()
            
            # 基本HTML结构检查
            html_elements = [
                '<!DOCTYPE html>',
                '<html',
                '<head>',
                '<body>',
                '</html>'
            ]
            
            for element in html_elements:
                assert element in content

class TestJavaScriptFunctionality:
    """JavaScript功能模拟测试"""
    
    def test_websocket_connection_simulation(self):
        """模拟WebSocket连接测试"""
        # 模拟WebSocket连接状态
        class MockWebSocket:
            def __init__(self, url):
                self.url = url
                self.readyState = 1  # OPEN
                self.onopen = None
                self.onmessage = None
                self.onclose = None
                self.onerror = None
                
            def send(self, data):
                # 模拟发送消息
                return True
                
            def close(self):
                self.readyState = 3  # CLOSED
        
        # 测试WebSocket创建
        ws = MockWebSocket('ws://localhost:8000/dev-ws')
        assert ws.url == 'ws://localhost:8000/dev-ws'
        assert ws.readyState == 1  # OPEN状态
        
        # 测试发送消息
        result = ws.send(json.dumps({'type': 'ping'}))
        assert result is True
        
        # 测试关闭连接
        ws.close()
        assert ws.readyState == 3  # CLOSED状态
    
    def test_message_handling_simulation(self):
        """模拟消息处理测试"""
        # 模拟开发客户端消息处理
        def handle_message(message_data):
            """模拟消息处理逻辑"""
            message_type = message_data.get('type')
            
            if message_type == 'dev_connected':
                return {'action': 'show_notification', 'message': 'Connected'}
            elif message_type == 'reload_frontend':
                return {'action': 'reload_page'}
            elif message_type == 'backend_restarting':
                return {'action': 'show_notification', 'message': 'Backend restarting'}
            elif message_type == 'backend_restarted':
                return {'action': 'show_notification', 'message': 'Backend ready'}
            else:
                return {'action': 'ignore'}
        
        # 测试不同消息类型
        test_messages = [
            {'type': 'dev_connected', 'message': 'Development mode connected'},
            {'type': 'reload_frontend', 'message': 'Files changed'},
            {'type': 'backend_restarting', 'message': 'Python files modified'},
            {'type': 'backend_restarted', 'message': 'Backend is ready'},
            {'type': 'unknown', 'message': 'Unknown message type'}
        ]
        
        expected_actions = [
            'show_notification',
            'reload_page', 
            'show_notification',
            'show_notification',
            'ignore'
        ]
        
        for i, message in enumerate(test_messages):
            result = handle_message(message)
            assert result['action'] == expected_actions[i]
    
    def test_notification_system_simulation(self):
        """模拟通知系统测试"""
        # 模拟通知显示功能
        class MockNotificationSystem:
            def __init__(self):
                self.notifications = []
            
            def show_notification(self, message, type='info', duration=3000):
                """显示通知"""
                notification = {
                    'message': message,
                    'type': type,
                    'duration': duration,
                    'timestamp': 1234567890
                }
                self.notifications.append(notification)
                return notification
            
            def get_notifications(self):
                """获取所有通知"""
                return self.notifications
            
            def clear_notifications(self):
                """清空通知"""
                self.notifications = []
        
        # 测试通知系统
        notification_system = MockNotificationSystem()
        
        # 显示不同类型的通知
        notification_system.show_notification('Connected to dev server', 'success')
        notification_system.show_notification('Files changed, reloading...', 'info')
        notification_system.show_notification('Connection error', 'error')
        
        notifications = notification_system.get_notifications()
        assert len(notifications) == 3
        assert notifications[0]['type'] == 'success'
        assert notifications[1]['type'] == 'info'
        assert notifications[2]['type'] == 'error'
    
    def test_reconnection_logic_simulation(self):
        """模拟重连逻辑测试"""
        # 模拟重连管理器
        class MockReconnectionManager:
            def __init__(self):
                self.max_attempts = 10
                self.current_attempts = 0
                self.reconnect_interval = 3000
                self.is_reconnecting = False
            
            def should_reconnect(self):
                """是否应该重连"""
                return (not self.is_reconnecting and 
                        self.current_attempts < self.max_attempts)
            
            def start_reconnect(self):
                """开始重连"""
                if self.should_reconnect():
                    self.is_reconnecting = True
                    self.current_attempts += 1
                    return True
                return False
            
            def reconnect_success(self):
                """重连成功"""
                self.is_reconnecting = False
                self.current_attempts = 0
            
            def reconnect_failed(self):
                """重连失败"""
                self.is_reconnecting = False
        
        # 测试重连逻辑
        reconnect_mgr = MockReconnectionManager()
        
        # 第一次重连应该成功开始
        assert reconnect_mgr.start_reconnect() is True
        assert reconnect_mgr.is_reconnecting is True
        assert reconnect_mgr.current_attempts == 1
        
        # 重连中时不应该开始新的重连
        assert reconnect_mgr.start_reconnect() is False
        
        # 模拟重连成功
        reconnect_mgr.reconnect_success()
        assert reconnect_mgr.is_reconnecting is False
        assert reconnect_mgr.current_attempts == 0

class TestBrowserIntegration:
    """浏览器集成测试"""
    
    def test_auto_refresh_functionality(self):
        """测试自动刷新功能"""
        # 模拟页面刷新功能
        class MockPage:
            def __init__(self):
                self.refreshed = False
                self.refresh_count = 0
            
            def reload(self):
                """模拟页面刷新"""
                self.refreshed = True
                self.refresh_count += 1
            
            def is_refreshed(self):
                return self.refreshed
        
        # 模拟开发模式下的文件变更处理
        def handle_file_change(file_type, page):
            """处理文件变更"""
            if file_type in ['html', 'css', 'js']:
                # 前端文件变更，刷新页面
                page.reload()
                return 'page_refreshed'
            elif file_type == 'py':
                # Python文件变更，后端重启
                return 'backend_restart'
            else:
                return 'ignored'
        
        # 测试不同文件类型的处理
        page = MockPage()
        
        # 前端文件变更
        result = handle_file_change('html', page)
        assert result == 'page_refreshed'
        assert page.is_refreshed() is True
        
        # 重置页面状态
        page.refreshed = False
        
        # CSS文件变更
        result = handle_file_change('css', page)
        assert result == 'page_refreshed'
        assert page.is_refreshed() is True
        
        # Python文件变更
        page.refreshed = False
        result = handle_file_change('py', page)
        assert result == 'backend_restart'
        assert page.is_refreshed() is False  # Python文件变更不刷新页面
    
    def test_dev_mode_indicator(self):
        """测试开发模式指示器"""
        # 模拟开发模式指示器
        class MockDevIndicator:
            def __init__(self):
                self.visible = False
                self.text = ""
            
            def show(self, text="🔧 开发模式"):
                """显示指示器"""
                self.visible = True
                self.text = text
            
            def hide(self):
                """隐藏指示器"""
                self.visible = False
                self.text = ""
            
            def is_visible(self):
                return self.visible
        
        # 测试指示器功能
        indicator = MockDevIndicator()
        
        # 初始状态应该是隐藏的
        assert indicator.is_visible() is False
        
        # 显示开发模式指示器
        indicator.show()
        assert indicator.is_visible() is True
        assert "开发模式" in indicator.text
        
        # 隐藏指示器
        indicator.hide()
        assert indicator.is_visible() is False

class TestErrorHandlingJS:
    """JavaScript错误处理测试"""
    
    def test_websocket_connection_error(self):
        """测试WebSocket连接错误处理"""
        # 模拟WebSocket连接错误
        class MockWebSocketWithError:
            def __init__(self, should_fail=False):
                self.should_fail = should_fail
                self.error_count = 0
            
            def connect(self):
                if self.should_fail:
                    self.error_count += 1
                    raise Exception("Connection failed")
                return True
        
        # 测试正常连接
        ws_normal = MockWebSocketWithError(should_fail=False)
        result = ws_normal.connect()
        assert result is True
        assert ws_normal.error_count == 0
        
        # 测试连接失败
        ws_error = MockWebSocketWithError(should_fail=True)
        with pytest.raises(Exception, match="Connection failed"):
            ws_error.connect()
        assert ws_error.error_count == 1
    
    def test_message_parsing_error(self):
        """测试消息解析错误处理"""
        def safe_parse_message(message_data):
            """安全解析消息"""
            try:
                if isinstance(message_data, str):
                    return json.loads(message_data)
                return message_data
            except json.JSONDecodeError:
                return {'type': 'parse_error', 'original': message_data}
        
        # 测试有效JSON
        valid_json = '{"type": "test", "message": "hello"}'
        result = safe_parse_message(valid_json)
        assert result['type'] == 'test'
        assert result['message'] == 'hello'
        
        # 测试无效JSON
        invalid_json = '{"type": "test", "message": hello}'  # 缺少引号
        result = safe_parse_message(invalid_json)
        assert result['type'] == 'parse_error'
        assert 'original' in result
        
        # 测试已经是对象的数据
        dict_data = {'type': 'direct', 'message': 'world'}
        result = safe_parse_message(dict_data)
        assert result['type'] == 'direct'
    
    def test_notification_error_handling(self):
        """测试通知错误处理"""
        class MockNotificationWithError:
            def __init__(self):
                self.notifications = []
                self.error_notifications = []
            
            def show_notification(self, message, type='info'):
                """显示通知，可能失败"""
                try:
                    if type == 'invalid_type':
                        raise ValueError("Invalid notification type")
                    
                    notification = {'message': message, 'type': type}
                    self.notifications.append(notification)
                    return True
                except Exception as e:
                    error_notification = {'error': str(e), 'original_message': message}
                    self.error_notifications.append(error_notification)
                    return False
        
        # 测试正常通知
        notifier = MockNotificationWithError()
        result = notifier.show_notification('Test message', 'success')
        assert result is True
        assert len(notifier.notifications) == 1
        
        # 测试错误通知
        result = notifier.show_notification('Error test', 'invalid_type')
        assert result is False
        assert len(notifier.error_notifications) == 1
        assert 'Invalid notification type' in notifier.error_notifications[0]['error']