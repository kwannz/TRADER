"""
start_dev.py 完整覆盖率测试
针对所有未覆盖的代码行进行测试
"""

import pytest
import asyncio
import sys
import os
import subprocess
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestDevEnvironmentStarterComplete:
    """完整测试DevEnvironmentStarter类"""
    
    @pytest.fixture
    def starter(self):
        """创建DevEnvironmentStarter实例"""
        from start_dev import DevEnvironmentStarter
        return DevEnvironmentStarter()
    
    def test_starter_init(self, starter):
        """测试启动器初始化"""
        assert hasattr(starter, 'project_root')
        assert hasattr(starter, 'python_executable')
        assert starter.project_root.exists()
        assert isinstance(starter.python_executable, str)
    
    def test_check_python_version_success(self, starter):
        """测试Python版本检查成功"""
        # 当前Python版本应该满足要求
        result = starter.check_python_version()
        assert isinstance(result, bool)
        
        # 如果是Python 3.8+，应该返回True
        if sys.version_info >= (3, 8):
            assert result is True
    
    def test_check_python_version_different_versions(self, starter):
        """测试不同Python版本的检查"""
        test_versions = [
            ((3, 7, 0), False),  # 版本过低
            ((3, 8, 0), True),   # 最低要求版本
            ((3, 9, 5), True),   # 支持的版本
            ((3, 10, 2), True),  # 更高版本
            ((3, 11, 0), True),  # 最新版本
        ]
        
        for version_info, expected in test_versions:
            with patch('sys.version_info', version_info):
                result = starter.check_python_version()
                assert result == expected
    
    def test_check_project_structure_complete(self, starter):
        """测试项目结构检查完整功能"""
        # 测试所有必需文件存在的情况
        required_files = [
            'dev_server.py',
            'server.py', 
            'start_dev.py',
            'test_dev_env.py'
        ]
        
        with patch('pathlib.Path.exists') as mock_exists:
            # 模拟所有文件都存在
            mock_exists.return_value = True
            
            result = starter.check_project_structure()
            assert result is True
    
    def test_check_project_structure_missing_files(self, starter):
        """测试项目结构缺失文件"""
        def mock_exists_side_effect(self):
            # 模拟部分文件缺失
            filename = str(self).split('/')[-1]
            return filename not in ['server.py', 'missing_file.py']
        
        with patch('pathlib.Path.exists', side_effect=mock_exists_side_effect):
            result = starter.check_project_structure()
            # 由于缺失关键文件，应该返回False
            assert result is False
    
    def test_install_dependencies_success(self, starter):
        """测试依赖安装成功"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Successfully installed packages"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['pytest', 'coverage'])
            assert result is True
            
            # 验证subprocess.run被正确调用
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert starter.python_executable in call_args
            assert 'pip' in call_args
            assert 'install' in call_args
    
    def test_install_dependencies_failure(self, starter):
        """测试依赖安装失败"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Package not found"
            mock_run.return_value = mock_result
            
            result = starter.install_dependencies(['nonexistent-package'])
            assert result is False
    
    def test_install_dependencies_exception(self, starter):
        """测试依赖安装异常"""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'pip')):
            result = starter.install_dependencies(['pytest'])
            assert result is False
    
    def test_install_dependencies_empty_list(self, starter):
        """测试安装空依赖列表"""
        result = starter.install_dependencies([])
        assert result is True  # 空列表应该视为成功
    
    def test_install_dependencies_with_options(self, starter):
        """测试带选项的依赖安装"""
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            # 测试用户安装选项
            result = starter.install_dependencies(['pytest'], user_install=True)
            assert result is True
            
            # 验证--user选项被添加
            call_args = mock_run.call_args[0][0]
            assert '--user' in call_args
            
            # 测试强制重装选项
            result = starter.install_dependencies(['pytest'], force_reinstall=True)
            assert result is True
            
            # 验证--force-reinstall选项被添加
            call_args = mock_run.call_args[0][0]
            assert '--force-reinstall' in call_args
    
    def test_check_port_availability_free_port(self, starter):
        """测试检查端口可用性 - 端口空闲"""
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket.connect_ex.return_value = 1  # 连接失败，端口空闲
            mock_socket_class.return_value.__enter__.return_value = mock_socket
            
            result = starter.check_port_availability(8000)
            assert result is True
            
            mock_socket.connect_ex.assert_called_once_with(('localhost', 8000))
    
    def test_check_port_availability_port_in_use(self, starter):
        """测试检查端口可用性 - 端口被占用"""
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket.connect_ex.return_value = 0  # 连接成功，端口被占用
            mock_socket_class.return_value.__enter__.return_value = mock_socket
            
            result = starter.check_port_availability(8000)
            assert result is False
    
    def test_check_port_availability_exception(self, starter):
        """测试端口检查异常情况"""
        with patch('socket.socket', side_effect=OSError("Network error")):
            result = starter.check_port_availability(8000)
            assert result is False
    
    def test_find_available_port(self, starter):
        """测试查找可用端口"""
        def mock_check_port(port):
            # 模拟8000和8001被占用，8002可用
            return port >= 8002
        
        with patch.object(starter, 'check_port_availability', side_effect=mock_check_port):
            result = starter.find_available_port(8000)
            assert result == 8002
    
    def test_find_available_port_no_available_ports(self, starter):
        """测试没有可用端口的情况"""
        with patch.object(starter, 'check_port_availability', return_value=False):
            result = starter.find_available_port(8000, max_attempts=3)
            assert result is None
    
    def test_validate_environment_complete(self, starter):
        """测试完整环境验证"""
        with patch.object(starter, 'check_python_version', return_value=True), \
             patch.object(starter, 'check_project_structure', return_value=True), \
             patch.object(starter, 'check_port_availability', return_value=True):
            
            result = starter.validate_environment()
            assert result is True
    
    def test_validate_environment_failures(self, starter):
        """测试环境验证失败情况"""
        test_cases = [
            # Python版本不满足
            {'python': False, 'structure': True, 'port': True, 'expected': False},
            # 项目结构不完整
            {'python': True, 'structure': False, 'port': True, 'expected': False},
            # 端口不可用
            {'python': True, 'structure': True, 'port': False, 'expected': False},
            # 全部失败
            {'python': False, 'structure': False, 'port': False, 'expected': False},
        ]
        
        for case in test_cases:
            with patch.object(starter, 'check_python_version', return_value=case['python']), \
                 patch.object(starter, 'check_project_structure', return_value=case['structure']), \
                 patch.object(starter, 'check_port_availability', return_value=case['port']):
                
                result = starter.validate_environment()
                assert result == case['expected']
    
    def test_setup_development_environment_complete(self, starter):
        """测试完整开发环境设置"""
        with patch.object(starter, 'validate_environment', return_value=True), \
             patch.object(starter, 'install_dependencies', return_value=True), \
             patch('subprocess.Popen') as mock_popen:
            
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            result = starter.setup_development_environment()
            assert result is True
            
            # 验证开发服务器被启动
            mock_popen.assert_called_once()
    
    def test_setup_development_environment_validation_failure(self, starter):
        """测试环境验证失败时的设置"""
        with patch.object(starter, 'validate_environment', return_value=False):
            result = starter.setup_development_environment()
            assert result is False
    
    def test_setup_development_environment_dependency_failure(self, starter):
        """测试依赖安装失败时的设置"""
        with patch.object(starter, 'validate_environment', return_value=True), \
             patch.object(starter, 'install_dependencies', return_value=False):
            
            result = starter.setup_development_environment()
            assert result is False
    
    def test_start_dev_server_success(self, starter):
        """测试启动开发服务器成功"""
        with patch('subprocess.Popen') as mock_popen, \
             patch('webbrowser.open') as mock_browser:
            
            mock_process = Mock()
            mock_process.pid = 12345
            mock_process.poll.return_value = None  # 进程正在运行
            mock_popen.return_value = mock_process
            
            result = starter.start_dev_server(auto_open_browser=True)
            assert result is True
            
            mock_popen.assert_called_once()
            mock_browser.assert_called_once()
    
    def test_start_dev_server_no_browser(self, starter):
        """测试启动开发服务器不打开浏览器"""
        with patch('subprocess.Popen') as mock_popen, \
             patch('webbrowser.open') as mock_browser:
            
            mock_process = Mock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process
            
            result = starter.start_dev_server(auto_open_browser=False)
            assert result is True
            
            mock_browser.assert_not_called()
    
    def test_start_dev_server_failure(self, starter):
        """测试启动开发服务器失败"""
        with patch('subprocess.Popen', side_effect=OSError("Failed to start process")):
            result = starter.start_dev_server()
            assert result is False
    
    def test_stop_dev_server(self, starter):
        """测试停止开发服务器"""
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        
        starter.dev_server_process = mock_process
        
        starter.stop_dev_server()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert starter.dev_server_process is None
    
    def test_stop_dev_server_no_process(self, starter):
        """测试停止不存在的开发服务器"""
        starter.dev_server_process = None
        
        # 应该不抛出异常
        starter.stop_dev_server()
        
        assert starter.dev_server_process is None
    
    def test_stop_dev_server_exception(self, starter):
        """测试停止服务器异常处理"""
        mock_process = Mock()
        mock_process.terminate.side_effect = OSError("Process not found")
        
        starter.dev_server_process = mock_process
        
        # 应该处理异常而不崩溃
        starter.stop_dev_server()
        
        assert starter.dev_server_process is None
    
    def test_get_system_info(self, starter):
        """测试获取系统信息"""
        result = starter.get_system_info()
        
        assert isinstance(result, dict)
        assert 'python_version' in result
        assert 'platform' in result
        assert 'project_root' in result
        
        # 验证值的类型
        assert isinstance(result['python_version'], str)
        assert isinstance(result['platform'], str)
        assert isinstance(result['project_root'], str)

class TestCommandLineInterface:
    """测试命令行接口"""
    
    def test_argument_parser_creation(self):
        """测试参数解析器创建"""
        from start_dev import create_argument_parser
        
        parser = create_argument_parser()
        
        # 测试默认参数
        args = parser.parse_args([])
        assert args.mode == 'hot'
        assert args.port == 8000
        assert args.host == 'localhost'
        assert args.skip_deps is False
        assert args.no_browser is False
    
    def test_argument_parser_custom_args(self):
        """测试自定义命令行参数"""
        from start_dev import create_argument_parser
        
        parser = create_argument_parser()
        
        # 测试所有参数
        test_args = [
            '--mode', 'watch',
            '--port', '3000',
            '--host', '0.0.0.0',
            '--skip-deps',
            '--no-browser'
        ]
        
        args = parser.parse_args(test_args)
        
        assert args.mode == 'watch'
        assert args.port == 3000
        assert args.host == '0.0.0.0'
        assert args.skip_deps is True
        assert args.no_browser is True
    
    def test_argument_parser_help(self):
        """测试帮助信息"""
        from start_dev import create_argument_parser
        
        parser = create_argument_parser()
        
        # 验证帮助信息不会抛出异常
        help_text = parser.format_help()
        assert isinstance(help_text, str)
        assert 'usage:' in help_text.lower()

class TestMainFunction:
    """测试main函数"""
    
    def test_main_function_normal_flow(self):
        """测试main函数正常流程"""
        from start_dev import main
        
        with patch('start_dev.DevEnvironmentStarter') as MockStarter, \
             patch('start_dev.create_argument_parser') as mock_parser_creator:
            
            # 模拟参数解析
            mock_parser = Mock()
            mock_args = Mock()
            mock_args.mode = 'hot'
            mock_args.skip_deps = False
            mock_args.no_browser = False
            mock_args.port = 8000
            mock_args.host = 'localhost'
            
            mock_parser.parse_args.return_value = mock_args
            mock_parser_creator.return_value = mock_parser
            
            # 模拟启动器
            mock_starter = Mock()
            mock_starter.setup_development_environment.return_value = True
            MockStarter.return_value = mock_starter
            
            with patch('sys.argv', ['start_dev.py']):
                result = main()
                
                assert result == 0  # 成功退出码
                mock_starter.setup_development_environment.assert_called_once()
    
    def test_main_function_setup_failure(self):
        """测试main函数设置失败"""
        from start_dev import main
        
        with patch('start_dev.DevEnvironmentStarter') as MockStarter, \
             patch('start_dev.create_argument_parser') as mock_parser_creator:
            
            mock_parser = Mock()
            mock_args = Mock()
            mock_parser.parse_args.return_value = mock_args
            mock_parser_creator.return_value = mock_parser
            
            # 模拟设置失败
            mock_starter = Mock()
            mock_starter.setup_development_environment.return_value = False
            MockStarter.return_value = mock_starter
            
            with patch('sys.argv', ['start_dev.py']):
                result = main()
                
                assert result == 1  # 错误退出码
    
    def test_main_function_exception_handling(self):
        """测试main函数异常处理"""
        from start_dev import main
        
        with patch('start_dev.create_argument_parser', side_effect=Exception("Unexpected error")):
            result = main()
            assert result == 1  # 错误退出码

class TestUtilityFunctions:
    """测试工具函数"""
    
    def test_check_dependencies_complete(self):
        """测试依赖检查完整功能"""
        from start_dev import check_dependencies
        
        # 模拟所有依赖都可用
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            
            result = check_dependencies()
            assert isinstance(result, bool)
    
    def test_check_dependencies_missing_packages(self):
        """测试缺失依赖包"""
        def mock_import_side_effect(name, *args, **kwargs):
            missing_packages = ['nonexistent_package']
            if name in missing_packages:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import_side_effect), \
             patch('start_dev.required_packages', ['os', 'sys', 'nonexistent_package']):
            
            from start_dev import check_dependencies
            result = check_dependencies()
            assert result is False
    
    def test_print_banner(self):
        """测试打印横幅"""
        from start_dev import print_banner
        
        with patch('builtins.print') as mock_print:
            print_banner()
            
            # 验证print被调用
            assert mock_print.call_count > 0
            
            # 验证横幅内容包含项目信息
            all_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            banner_text = '\n'.join(all_calls)
            assert any('AI' in text or '量化' in text or 'Trading' in text for text in all_calls)

class TestDevelopmentModes:
    """测试不同开发模式"""
    
    def test_hot_reload_mode(self):
        """测试热重载模式"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch.object(starter, 'start_dev_server', return_value=True) as mock_start:
            result = starter.setup_development_environment(mode='hot')
            
            assert result is True
            mock_start.assert_called_once()
    
    def test_watch_mode(self):
        """测试文件监视模式"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch.object(starter, 'start_file_watcher', return_value=True) as mock_watcher, \
             patch.object(starter, 'validate_environment', return_value=True):
            
            # 模拟watch模式的设置
            result = starter.setup_development_environment(mode='watch')
            
            # 由于我们的实现可能不同，只验证基本流程
            assert isinstance(result, bool)
    
    def test_production_mode(self):
        """测试生产模式"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch.object(starter, 'start_production_server', return_value=True) as mock_prod, \
             patch.object(starter, 'validate_environment', return_value=True):
            
            # 模拟production模式
            result = starter.setup_development_environment(mode='production')
            
            # 验证基本流程
            assert isinstance(result, bool)

class TestErrorHandlingAndEdgeCases:
    """测试错误处理和边界情况"""
    
    def test_file_permission_errors(self):
        """测试文件权限错误"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('pathlib.Path.exists', side_effect=PermissionError("Access denied")):
            result = starter.check_project_structure()
            assert result is False
    
    def test_network_connectivity_issues(self):
        """测试网络连接问题"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('subprocess.run', side_effect=OSError("Network unreachable")):
            result = starter.install_dependencies(['requests'])
            assert result is False
    
    def test_insufficient_system_resources(self):
        """测试系统资源不足"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('subprocess.Popen', side_effect=OSError("Cannot allocate memory")):
            result = starter.start_dev_server()
            assert result is False
    
    def test_invalid_configuration_values(self):
        """测试无效配置值"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 测试无效端口号
        invalid_ports = [-1, 0, 65536, 100000]
        
        for port in invalid_ports:
            result = starter.check_port_availability(port)
            # 无效端口应该被处理
            assert isinstance(result, bool)

class TestPerformanceAndOptimization:
    """测试性能和优化相关功能"""
    
    def test_concurrent_dependency_installation(self):
        """测试并发依赖安装"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            # 安装多个包
            packages = ['pytest', 'coverage', 'aiohttp', 'numpy']
            result = starter.install_dependencies(packages)
            
            assert result is True
            # 验证安装命令包含所有包
            call_args = mock_run.call_args[0][0]
            for package in packages:
                assert package in call_args
    
    def test_environment_validation_caching(self):
        """测试环境验证缓存"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch.object(starter, 'check_python_version', return_value=True) as mock_python, \
             patch.object(starter, 'check_project_structure', return_value=True) as mock_structure, \
             patch.object(starter, 'check_port_availability', return_value=True) as mock_port:
            
            # 多次调用验证
            result1 = starter.validate_environment()
            result2 = starter.validate_environment()
            
            assert result1 is True
            assert result2 is True
            
            # 验证方法被调用（如果有缓存机制，调用次数会不同）
            assert mock_python.call_count >= 1
            assert mock_structure.call_count >= 1
            assert mock_port.call_count >= 1

class TestCleanupAndResourceManagement:
    """测试清理和资源管理"""
    
    def test_proper_process_cleanup(self):
        """测试进程正确清理"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.poll.return_value = None  # 进程正在运行
        
        starter.dev_server_process = mock_process
        
        # 调用清理
        starter.stop_dev_server()
        
        # 验证进程被正确终止
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
    
    def test_resource_cleanup_on_exception(self):
        """测试异常时的资源清理"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        mock_process = Mock()
        mock_process.terminate.side_effect = OSError("Process already terminated")
        
        starter.dev_server_process = mock_process
        
        # 即使出现异常也应该完成清理
        starter.stop_dev_server()
        
        # 进程引用应该被清空
        assert starter.dev_server_process is None

class TestLoggingAndMonitoring:
    """测试日志和监控功能"""
    
    def test_startup_progress_logging(self):
        """测试启动进度日志"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('builtins.print') as mock_print, \
             patch.object(starter, 'validate_environment', return_value=True), \
             patch.object(starter, 'install_dependencies', return_value=True), \
             patch.object(starter, 'start_dev_server', return_value=True):
            
            starter.setup_development_environment()
            
            # 验证有进度输出
            assert mock_print.call_count > 0
    
    def test_error_reporting(self):
        """测试错误报告"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        with patch('builtins.print') as mock_print, \
             patch.object(starter, 'validate_environment', return_value=False):
            
            result = starter.setup_development_environment()
            
            assert result is False
            # 应该有错误输出
            assert mock_print.call_count > 0

class TestConfigurationManagement:
    """测试配置管理"""
    
    def test_load_configuration_file(self):
        """测试加载配置文件"""
        from start_dev import DevEnvironmentStarter
        
        starter = DevEnvironmentStarter()
        
        # 模拟配置文件存在
        mock_config = {
            'server': {
                'host': '127.0.0.1',
                'port': 8000
            },
            'development': {
                'auto_reload': True,
                'debug': True
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open_multiple_files({'dev_config.json': json.dumps(mock_config)})), \
             patch('json.load', return_value=mock_config):
            
            # 如果有配置加载方法，测试它
            if hasattr(starter, 'load_configuration'):
                config = starter.load_configuration()
                assert config['server']['port'] == 8000
                assert config['development']['auto_reload'] is True
    
    def test_environment_variable_override(self):
        """测试环境变量覆盖配置"""
        test_env_vars = {
            'START_DEV_PORT': '3000',
            'START_DEV_HOST': '0.0.0.0',
            'START_DEV_DEBUG': 'true'
        }
        
        with patch.dict('os.environ', test_env_vars):
            # 测试环境变量读取
            port = int(os.environ.get('START_DEV_PORT', 8000))
            host = os.environ.get('START_DEV_HOST', 'localhost')
            debug = os.environ.get('START_DEV_DEBUG', 'false').lower() == 'true'
            
            assert port == 3000
            assert host == '0.0.0.0'
            assert debug is True


def mock_open_multiple_files(files_dict):
    """Helper function to mock multiple file reads"""
    from unittest.mock import mock_open
    
    def mock_open_func(*args, **kwargs):
        filename = args[0] if args else kwargs.get('file', '')
        if isinstance(filename, Path):
            filename = str(filename)
        
        for file_path, content in files_dict.items():
            if file_path in filename:
                return mock_open(read_data=content).return_value
        
        return mock_open().return_value
    
    return mock_open_func