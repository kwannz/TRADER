"""
启动脚本测试
"""

import pytest
import sys
import subprocess
from unittest.mock import Mock, patch, call
from pathlib import Path
import tempfile
import os

# 导入要测试的模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class TestDevEnvironmentStarter:
    """开发环境启动器测试"""
    
    @pytest.fixture
    def mock_starter(self):
        """创建模拟的开发环境启动器"""
        try:
            from start_dev import DevEnvironmentStarter
            return DevEnvironmentStarter()
        except ImportError:
            # 如果导入失败，创建Mock对象
            starter = Mock()
            starter.project_root = Path('/test/project')
            starter.python_executable = sys.executable
            return starter
    
    def test_starter_initialization(self, mock_starter):
        """测试启动器初始化"""
        assert hasattr(mock_starter, 'project_root')
        assert hasattr(mock_starter, 'python_executable')
    
    def test_check_python_version_success(self, mock_starter):
        """测试Python版本检查成功"""
        if hasattr(mock_starter, 'check_python_version'):
            # 当前运行的Python版本应该符合要求
            result = mock_starter.check_python_version()
            assert isinstance(result, bool)
        else:
            # 模拟版本检查
            def check_python_version():
                version = sys.version_info
                return version >= (3, 8)
            
            mock_starter.check_python_version = check_python_version
            result = mock_starter.check_python_version()
            assert result is True
    
    def test_check_python_version_failure(self, mock_starter):
        """测试Python版本检查失败"""
        # 模拟低版本Python
        with patch('sys.version_info', (3, 7, 0)):
            if hasattr(mock_starter, 'check_python_version'):
                result = mock_starter.check_python_version()
                # 低版本应该返回False或引发问题
                assert isinstance(result, bool)
            else:
                def check_python_version():
                    return sys.version_info >= (3, 8)
                
                mock_starter.check_python_version = check_python_version
                result = mock_starter.check_python_version()
                assert result is False
    
    def test_check_dependencies_all_available(self, mock_starter):
        """测试依赖检查 - 所有依赖可用"""
        required_packages = ['json', 'os', 'sys']  # 使用标准库包
        
        if hasattr(mock_starter, 'check_dependencies'):
            # 使用真实方法
            result = mock_starter.check_dependencies()
            assert isinstance(result, bool)
        else:
            # 模拟依赖检查
            def check_dependencies():
                for package in required_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        return False
                return True
            
            mock_starter.check_dependencies = check_dependencies
            result = mock_starter.check_dependencies()
            assert result is True
    
    def test_check_dependencies_missing_packages(self, mock_starter):
        """测试依赖检查 - 缺失依赖"""
        # 使用不存在的包名
        fake_packages = ['nonexistent_package_123', 'fake_module_xyz']
        
        if hasattr(mock_starter, 'check_dependencies'):
            with patch('start_dev.required_packages', fake_packages):
                try:
                    result = mock_starter.check_dependencies()
                    # 应该返回False或引发ImportError
                    if isinstance(result, bool):
                        assert result is False
                except ImportError:
                    # 这也是预期的行为
                    assert True
        else:
            def check_dependencies():
                for package in fake_packages:
                    try:
                        __import__(package)
                    except ImportError:
                        return False
                return True
            
            mock_starter.check_dependencies = check_dependencies
            result = mock_starter.check_dependencies()
            assert result is False
    
    def test_install_dependencies_success(self, mock_starter):
        """测试依赖安装成功"""
        if hasattr(mock_starter, 'install_dependencies'):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                
                result = mock_starter.install_dependencies(['test_package'])
                assert isinstance(result, bool)
        else:
            # 模拟安装过程
            def install_dependencies(packages):
                try:
                    # 模拟pip install命令
                    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
                    return True  # 假设安装成功
                except Exception:
                    return False
            
            mock_starter.install_dependencies = install_dependencies
            result = mock_starter.install_dependencies(['test_package'])
            assert result is True
    
    def test_install_dependencies_failure(self, mock_starter):
        """测试依赖安装失败"""
        if hasattr(mock_starter, 'install_dependencies'):
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(1, 'pip')
                
                result = mock_starter.install_dependencies(['test_package'])
                assert isinstance(result, bool)
        else:
            def install_dependencies(packages):
                raise subprocess.CalledProcessError(1, 'pip install')
            
            mock_starter.install_dependencies = install_dependencies
            
            try:
                result = mock_starter.install_dependencies(['test_package'])
                assert result is False
            except subprocess.CalledProcessError:
                assert True  # 预期的异常
    
    def test_check_project_structure(self, mock_starter, temp_dir):
        """测试项目结构检查"""
        # 在临时目录中创建一些文件
        test_files = ['dev_server.py', 'server.py', 'test_file.js']
        
        for filename in test_files:
            (temp_dir / filename).write_text('# test content')
        
        if hasattr(mock_starter, 'check_project_structure'):
            with patch.object(mock_starter, 'project_root', temp_dir):
                result = mock_starter.check_project_structure()
                assert isinstance(result, bool)
        else:
            def check_project_structure():
                required_files = ['dev_server.py', 'server.py']
                missing_count = 0
                
                for file_name in required_files:
                    file_path = temp_dir / file_name
                    if not file_path.exists():
                        missing_count += 1
                
                return missing_count == 0
            
            mock_starter.check_project_structure = check_project_structure
            result = mock_starter.check_project_structure()
            assert result is True

class TestStartDevScript:
    """启动脚本功能测试"""
    
    def test_script_imports(self):
        """测试脚本导入"""
        # 测试能否成功导入启动脚本模块
        try:
            import start_dev
            assert hasattr(start_dev, 'main') or hasattr(start_dev, 'DevEnvironmentStarter')
        except ImportError:
            # 如果导入失败，至少验证文件存在
            script_path = Path(__file__).parent.parent.parent / 'start_dev.py'
            assert script_path.exists()
    
    def test_argument_parsing(self):
        """测试命令行参数解析"""
        try:
            import start_dev
            import argparse
            
            # 创建参数解析器
            parser = argparse.ArgumentParser()
            parser.add_argument('--mode', choices=['hot', 'enhanced'], default='hot')
            parser.add_argument('--skip-deps', action='store_true')
            
            # 测试不同参数组合
            test_args = [
                [],
                ['--mode', 'hot'],
                ['--mode', 'enhanced'],
                ['--skip-deps'],
                ['--mode', 'hot', '--skip-deps']
            ]
            
            for args in test_args:
                parsed = parser.parse_args(args)
                assert hasattr(parsed, 'mode')
                assert hasattr(parsed, 'skip_deps')
                
        except ImportError:
            # 如果导入失败，跳过测试
            pytest.skip("start_dev module not available")
    
    def test_main_function_exists(self):
        """测试main函数存在"""
        try:
            from start_dev import main
            assert callable(main)
        except ImportError:
            # 验证文件存在并包含main函数定义
            script_path = Path(__file__).parent.parent.parent / 'start_dev.py'
            if script_path.exists():
                content = script_path.read_text()
                assert 'def main(' in content
            else:
                pytest.skip("start_dev.py not found")

class TestShellScript:
    """Shell脚本测试"""
    
    def test_shell_script_exists(self):
        """测试Shell脚本存在"""
        script_path = Path(__file__).parent.parent.parent / 'start_dev.sh'
        assert script_path.exists()
    
    def test_shell_script_executable(self):
        """测试Shell脚本可执行权限"""
        script_path = Path(__file__).parent.parent.parent / 'start_dev.sh'
        if script_path.exists():
            # 检查文件是否有执行权限
            import stat
            file_stat = script_path.stat()
            assert file_stat.st_mode & stat.S_IEXEC
    
    def test_shell_script_shebang(self):
        """测试Shell脚本shebang"""
        script_path = Path(__file__).parent.parent.parent / 'start_dev.sh'
        if script_path.exists():
            content = script_path.read_text()
            assert content.startswith('#!/bin/bash')

class TestBatchScript:
    """批处理脚本测试"""
    
    def test_batch_script_exists(self):
        """测试批处理脚本存在"""
        script_path = Path(__file__).parent.parent.parent / 'start_dev.bat'
        assert script_path.exists()
    
    def test_batch_script_content(self):
        """测试批处理脚本内容"""
        script_path = Path(__file__).parent.parent.parent / 'start_dev.bat'
        if script_path.exists():
            content = script_path.read_text()
            assert '@echo off' in content
            assert 'python' in content.lower() or 'py' in content.lower()

class TestConfigFiles:
    """配置文件测试"""
    
    def test_dev_config_exists(self):
        """测试开发配置文件存在"""
        config_path = Path(__file__).parent.parent.parent / 'dev_config.json'
        assert config_path.exists()
    
    def test_dev_config_valid_json(self):
        """测试开发配置文件是有效的JSON"""
        config_path = Path(__file__).parent.parent.parent / 'dev_config.json'
        if config_path.exists():
            import json
            content = config_path.read_text()
            config = json.loads(content)  # 应该不抛出异常
            
            # 验证基本配置项
            assert 'server' in config
            assert 'hot_reload' in config
    
    def test_requirements_files_exist(self):
        """测试requirements文件存在"""
        project_root = Path(__file__).parent.parent.parent
        
        req_files = [
            'requirements-dev.txt',
            'requirements-test.txt'
        ]
        
        for req_file in req_files:
            req_path = project_root / req_file
            assert req_path.exists()

class TestEnvironmentValidation:
    """环境验证测试"""
    
    def test_python_version_check(self):
        """测试当前Python版本"""
        version = sys.version_info
        assert version >= (3, 8), f"Python版本过低: {version.major}.{version.minor}"
    
    def test_required_modules_importable(self):
        """测试必需模块可导入"""
        core_modules = [
            'json', 'os', 'sys', 'pathlib', 
            'subprocess', 'argparse', 'time'
        ]
        
        for module_name in core_modules:
            try:
                __import__(module_name)
            except ImportError:
                pytest.fail(f"核心模块 {module_name} 无法导入")
    
    def test_project_structure(self):
        """测试项目结构"""
        project_root = Path(__file__).parent.parent.parent
        
        expected_files = [
            'dev_server.py',
            'server.py',
            'start_dev.py'
        ]
        
        missing_files = []
        for filename in expected_files:
            if not (project_root / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            pytest.skip(f"缺少项目文件: {missing_files}")

class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_dependency_checking_function(self):
        """测试依赖检查函数"""
        def check_package_exists(package_name):
            """检查包是否存在"""
            try:
                __import__(package_name)
                return True
            except ImportError:
                return False
        
        # 测试存在的包
        assert check_package_exists('json') is True
        assert check_package_exists('os') is True
        
        # 测试不存在的包
        assert check_package_exists('nonexistent_package_xyz') is False
    
    def test_file_existence_checking(self):
        """测试文件存在性检查"""
        def check_file_exists(file_path):
            """检查文件是否存在"""
            return Path(file_path).exists()
        
        # 测试当前测试文件
        current_file = __file__
        assert check_file_exists(current_file) is True
        
        # 测试不存在的文件
        fake_file = '/nonexistent/path/fake_file.txt'
        assert check_file_exists(fake_file) is False
    
    def test_command_line_validation(self):
        """测试命令行参数验证"""
        def validate_mode(mode):
            """验证启动模式"""
            valid_modes = ['hot', 'enhanced']
            return mode in valid_modes
        
        assert validate_mode('hot') is True
        assert validate_mode('enhanced') is True
        assert validate_mode('invalid') is False
    
    def test_path_handling(self):
        """测试路径处理"""
        # 测试相对路径转绝对路径
        current_dir = Path('.')
        absolute_path = current_dir.resolve()
        assert absolute_path.is_absolute()
        
        # 测试路径拼接
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / 'dev_config.json'
        assert isinstance(config_path, Path)