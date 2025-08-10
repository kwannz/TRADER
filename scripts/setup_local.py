#!/usr/bin/env python3
"""
本地环境安装配置脚本

自动化安装和配置所有依赖组件，包括：
- Python环境和依赖包
- Rust工具链和编译
- MongoDB和Redis数据库
- 系统配置和初始化
"""

import os
import sys
import subprocess
import platform
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class LocalSetup:
    """本地环境安装配置器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.platform = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # 安装状态跟踪
        self.installation_log = []
        self.errors = []
        
    def log_step(self, step: str, status: str = "INFO"):
        """记录安装步骤"""
        message = f"[{status}] {step}"
        print(message)
        self.installation_log.append(message)
        
    def run_command(self, command: List[str], description: str, check: bool = True) -> bool:
        """运行系统命令"""
        try:
            self.log_step(f"执行: {description}")
            print(f"  命令: {' '.join(command)}")
            
            result = subprocess.run(
                command, 
                check=check, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout:
                print(f"  输出: {result.stdout.strip()}")
            
            self.log_step(f"完成: {description}", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"失败: {description} - {e.stderr if e.stderr else str(e)}"
            self.log_step(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False
        except FileNotFoundError:
            error_msg = f"命令未找到: {command[0]} - 请确保已安装相关工具"
            self.log_step(error_msg, "ERROR")
            self.errors.append(error_msg)
            return False

    def check_prerequisites(self) -> bool:
        """检查系统前置条件"""
        self.log_step("检查系统前置条件...")
        
        prerequisites_ok = True
        
        # 检查Python版本
        if sys.version_info < (3, 10):
            self.log_step(f"Python版本过低: {self.python_version}, 需要3.10+", "ERROR")
            prerequisites_ok = False
        else:
            self.log_step(f"Python版本检查通过: {self.python_version}")
        
        # 检查包管理器
        package_managers = {
            'darwin': ['brew', 'Homebrew包管理器'],
            'linux': ['apt', 'APT包管理器'] if shutil.which('apt') else ['yum', 'YUM包管理器'],
            'windows': ['choco', 'Chocolatey包管理器']
        }
        
        if self.platform in package_managers:
            pm_cmd, pm_name = package_managers[self.platform]
            if not shutil.which(pm_cmd):
                self.log_step(f"未找到{pm_name}，请先安装", "WARNING")
        
        return prerequisites_ok

    def install_system_dependencies(self) -> bool:
        """安装系统级依赖"""
        self.log_step("安装系统级依赖...")
        
        if self.platform == 'darwin':  # macOS
            commands = [
                (['brew', '--version'], '检查Homebrew'),
                (['brew', 'update'], '更新Homebrew'),
                (['brew', 'install', 'mongodb-community', 'redis', 'pkg-config'], '安装MongoDB和Redis'),
            ]
        elif self.platform == 'linux':  # Linux
            commands = [
                (['sudo', 'apt', 'update'], '更新包索引'),
                (['sudo', 'apt', 'install', '-y', 'mongodb', 'redis-server', 'pkg-config', 'build-essential'], '安装依赖包'),
            ]
        elif self.platform == 'windows':  # Windows
            commands = [
                (['choco', 'install', 'mongodb', 'redis-64', '-y'], '安装MongoDB和Redis'),
            ]
        else:
            self.log_step(f"不支持的操作系统: {self.platform}", "ERROR")
            return False
        
        success = True
        for command, description in commands:
            if not self.run_command(command, description, check=False):
                success = False
        
        return success

    def setup_rust_environment(self) -> bool:
        """设置Rust开发环境"""
        self.log_step("设置Rust开发环境...")
        
        # 检查是否已安装Rust
        if shutil.which('rustc'):
            self.log_step("Rust已安装，检查版本...")
            self.run_command(['rustc', '--version'], 'Rust版本', check=False)
            self.run_command(['rustup', 'update'], '更新Rust工具链', check=False)
        else:
            self.log_step("安装Rust工具链...")
            # 下载并安装rustup
            if self.platform in ['darwin', 'linux']:
                install_cmd = ['curl', '--proto', '=https', '--tlsv1.2', '-sSf', 
                              'https://sh.rustup.rs', '|', 'sh', '-s', '--', '-y']
                # 由于管道命令的复杂性，直接运行bash
                bash_cmd = 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
                result = os.system(bash_cmd)
                if result != 0:
                    self.log_step("Rust安装失败", "ERROR")
                    return False
            else:  # Windows
                self.log_step("请手动安装Rust: https://rustup.rs/", "WARNING")
                return False
        
        # 添加cargo工具
        self.run_command(['rustup', 'component', 'add', 'clippy', 'rustfmt'], '安装Rust组件', check=False)
        
        # 编译Rust引擎
        rust_engine_path = self.project_root / 'rust_engine'
        if rust_engine_path.exists():
            self.log_step("编译Rust引擎...")
            os.chdir(rust_engine_path)
            success = self.run_command(['cargo', 'build', '--release'], '编译Rust引擎')
            os.chdir(self.project_root)
            return success
        else:
            self.log_step("未找到Rust引擎代码", "WARNING")
            return True

    def setup_python_environment(self) -> bool:
        """设置Python开发环境"""
        self.log_step("设置Python开发环境...")
        
        # 创建虚拟环境
        venv_path = self.project_root / 'venv'
        if not venv_path.exists():
            if not self.run_command([sys.executable, '-m', 'venv', 'venv'], '创建虚拟环境'):
                return False
        
        # 确定虚拟环境中的Python路径
        if self.platform == 'windows':
            venv_python = venv_path / 'Scripts' / 'python.exe'
            venv_pip = venv_path / 'Scripts' / 'pip.exe'
        else:
            venv_python = venv_path / 'bin' / 'python'
            venv_pip = venv_path / 'bin' / 'pip'
        
        # 升级pip
        if not self.run_command([str(venv_pip), 'install', '--upgrade', 'pip'], '升级pip'):
            return False
        
        # 安装Python依赖
        requirements_files = [
            'requirements.txt',
            'python_layer/requirements.txt',
            'fastapi_layer/requirements.txt',
            'cli_interface/requirements.txt'
        ]
        
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                if not self.run_command([str(venv_pip), 'install', '-r', str(req_path)], 
                                      f'安装依赖: {req_file}'):
                    return False
        
        # 安装开发工具
        dev_packages = [
            'pytest', 'pytest-asyncio', 'pytest-cov',
            'black', 'isort', 'flake8', 'mypy',
            'jupyter', 'ipython'
        ]
        
        if not self.run_command([str(venv_pip), 'install'] + dev_packages, '安装开发工具'):
            return False
        
        return True

    def setup_databases(self) -> bool:
        """设置数据库服务"""
        self.log_step("设置数据库服务...")
        
        # 启动MongoDB
        if self.platform == 'darwin':
            mongodb_commands = [
                (['brew', 'services', 'start', 'mongodb/brew/mongodb-community'], '启动MongoDB服务')
            ]
        elif self.platform == 'linux':
            mongodb_commands = [
                (['sudo', 'systemctl', 'start', 'mongod'], '启动MongoDB服务'),
                (['sudo', 'systemctl', 'enable', 'mongod'], '设置MongoDB开机启动')
            ]
        else:
            mongodb_commands = []
        
        # 启动Redis
        if self.platform == 'darwin':
            redis_commands = [
                (['brew', 'services', 'start', 'redis'], '启动Redis服务')
            ]
        elif self.platform == 'linux':
            redis_commands = [
                (['sudo', 'systemctl', 'start', 'redis-server'], '启动Redis服务'),
                (['sudo', 'systemctl', 'enable', 'redis-server'], '设置Redis开机启动')
            ]
        else:
            redis_commands = []
        
        success = True
        for command, description in mongodb_commands + redis_commands:
            if not self.run_command(command, description, check=False):
                success = False
        
        # 等待服务启动
        import time
        time.sleep(3)
        
        # 验证数据库连接
        self.log_step("验证数据库连接...")
        return self.verify_database_connections()

    def verify_database_connections(self) -> bool:
        """验证数据库连接"""
        try:
            # 验证MongoDB
            import pymongo
            mongo_client = pymongo.MongoClient('mongodb://localhost:27017/')
            mongo_client.admin.command('ping')
            self.log_step("MongoDB连接验证成功")
            mongo_client.close()
        except Exception as e:
            self.log_step(f"MongoDB连接失败: {e}", "ERROR")
            return False
        
        try:
            # 验证Redis
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            self.log_step("Redis连接验证成功")
            redis_client.close()
        except Exception as e:
            self.log_step(f"Redis连接失败: {e}", "ERROR")
            return False
        
        return True

    def create_config_files(self) -> bool:
        """创建配置文件"""
        self.log_step("创建配置文件...")
        
        # 创建环境配置文件
        env_config = {
            "database": {
                "mongodb": {
                    "host": "localhost",
                    "port": 27017,
                    "database": "trading_system",
                    "username": "",
                    "password": ""
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "database": 0,
                    "password": ""
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True,
                "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
                "allowed_hosts": ["localhost", "127.0.0.1"]
            },
            "ai": {
                "deepseek": {
                    "api_key": "your_deepseek_api_key_here",
                    "base_url": "https://api.deepseek.com/v1"
                },
                "gemini": {
                    "api_key": "your_gemini_api_key_here"
                }
            },
            "trading": {
                "okx": {
                    "api_key": "",
                    "secret_key": "",
                    "passphrase": "",
                    "sandbox": True
                },
                "binance": {
                    "api_key": "",
                    "secret_key": "",
                    "sandbox": True
                }
            }
        }
        
        config_path = self.project_root / 'config' / 'local.json'
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(env_config, f, indent=2, ensure_ascii=False)
        
        self.log_step(f"配置文件已创建: {config_path}")
        
        # 创建.env文件
        env_content = f"""# 本地开发环境配置
ENVIRONMENT=local
DEBUG=true

# 数据库配置
MONGODB_URL=mongodb://localhost:27017/trading_system
REDIS_URL=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# AI API Keys (请填入你的API密钥)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# 交易所API Keys (请填入你的API密钥)
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
"""
        
        env_path = self.project_root / '.env'
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        self.log_step(f"环境变量文件已创建: {env_path}")
        return True

    def initialize_database(self) -> bool:
        """初始化数据库"""
        self.log_step("初始化数据库...")
        
        try:
            import pymongo
            
            # 连接MongoDB
            client = pymongo.MongoClient('mongodb://localhost:27017/')
            db = client['trading_system']
            
            # 创建集合和索引
            collections_indexes = {
                'strategies': [
                    ('name', pymongo.ASCENDING),
                    ('created_by', pymongo.ASCENDING),
                    ('status', pymongo.ASCENDING),
                    ('created_at', pymongo.DESCENDING)
                ],
                'trades': [
                    ('strategy_id', pymongo.ASCENDING),
                    ('symbol', pymongo.ASCENDING),
                    ('timestamp', pymongo.DESCENDING),
                    ('status', pymongo.ASCENDING)
                ],
                'market_data': [
                    ('symbol', pymongo.ASCENDING),
                    ('timestamp', pymongo.DESCENDING),
                    ('interval', pymongo.ASCENDING)
                ],
                'users': [
                    ('username', pymongo.ASCENDING),
                    ('email', pymongo.ASCENDING)
                ]
            }
            
            for collection_name, indexes in collections_indexes.items():
                collection = db[collection_name]
                for index in indexes:
                    collection.create_index([index])
                    
                self.log_step(f"集合 {collection_name} 初始化完成")
            
            # 创建默认用户
            users_collection = db['users']
            if users_collection.count_documents({}) == 0:
                default_user = {
                    'username': 'admin',
                    'email': 'admin@tradingsystem.ai',
                    'password_hash': 'hashed_password_here',  # 在实际应用中使用proper hash
                    'role': 'admin',
                    'created_at': '2024-01-01T00:00:00Z'
                }
                users_collection.insert_one(default_user)
                self.log_step("默认管理员用户已创建")
            
            client.close()
            self.log_step("数据库初始化完成")
            return True
            
        except Exception as e:
            self.log_step(f"数据库初始化失败: {e}", "ERROR")
            return False

    def run_tests(self) -> bool:
        """运行系统测试"""
        self.log_step("运行系统测试...")
        
        # 获取虚拟环境中的python路径
        venv_path = self.project_root / 'venv'
        if self.platform == 'windows':
            venv_python = venv_path / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_path / 'bin' / 'python'
        
        # 运行CLI验证测试
        cli_tester_path = self.project_root / 'cli_validation' / 'cli_tester.py'
        if cli_tester_path.exists():
            if not self.run_command([str(venv_python), str(cli_tester_path), '--modules', 'python_layer'], 
                                   'CLI模块验证测试', check=False):
                self.log_step("CLI验证测试失败", "WARNING")
        
        return True

    async def run_full_setup(self) -> bool:
        """运行完整安装流程"""
        self.log_step("开始本地环境安装配置...")
        
        steps = [
            ('检查前置条件', self.check_prerequisites),
            ('安装系统依赖', self.install_system_dependencies),
            ('设置Rust环境', self.setup_rust_environment),
            ('设置Python环境', self.setup_python_environment),
            ('设置数据库服务', self.setup_databases),
            ('创建配置文件', self.create_config_files),
            ('初始化数据库', self.initialize_database),
            ('运行系统测试', self.run_tests),
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            self.log_step(f"\n{'='*60}")
            self.log_step(f"步骤: {step_name}")
            self.log_step(f"{'='*60}")
            
            if step_func():
                success_count += 1
                self.log_step(f"✅ {step_name} 完成")
            else:
                self.log_step(f"❌ {step_name} 失败", "ERROR")
        
        # 生成安装报告
        self.generate_setup_report(success_count, len(steps))
        
        return success_count == len(steps)

    def generate_setup_report(self, success_count: int, total_steps: int):
        """生成安装报告"""
        report_path = self.project_root / 'setup_report.txt'
        
        report_content = f"""
# 本地环境安装报告

## 安装概要
- 总步骤数: {total_steps}
- 成功步骤: {success_count}
- 失败步骤: {total_steps - success_count}
- 安装状态: {'✅ 成功' if success_count == total_steps else '❌ 部分失败'}

## 详细日志
"""
        
        for log_entry in self.installation_log:
            report_content += f"{log_entry}\n"
        
        if self.errors:
            report_content += "\n## 错误列表\n"
            for i, error in enumerate(self.errors, 1):
                report_content += f"{i}. {error}\n"
        
        report_content += f"""
## 后续步骤

1. 配置API密钥:
   - 编辑 config/local.json 或 .env 文件
   - 填入DeepSeek和Gemini的API密钥
   - 填入交易所API密钥（如需要）

2. 启动系统:
   ```bash
   # 启动FastAPI服务器
   python scripts/start_server.py
   
   # 或启动CLI界面
   python scripts/start_cli.py
   ```

3. 验证安装:
   ```bash
   # 运行完整验证测试
   python cli_validation/cli_tester.py
   ```

4. 访问文档:
   - API文档: http://localhost:8000/docs
   - 系统状态: http://localhost:8000/health
   - 系统指标: http://localhost:8000/metrics

## 技术支持
如有问题，请查看:
- 项目文档: README.md
- 架构文档: FULLSTACK_ARCHITECTURE.md
- 故障排除: TROUBLESHOOTING.md
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 安装报告已生成: {report_path}")

async def main():
    """主函数"""
    setup = LocalSetup()
    
    print("🚀 AI量化交易系统 - 本地环境安装程序")
    print("="*60)
    
    success = await setup.run_full_setup()
    
    if success:
        print("\n🎉 恭喜！本地环境安装完成！")
        print("\n下一步:")
        print("1. 配置API密钥 (编辑 .env 文件)")
        print("2. 启动系统: python scripts/start_server.py")
        print("3. 访问API文档: http://localhost:8000/docs")
    else:
        print("\n⚠️ 安装过程中出现错误，请查看安装报告")
        print("📄 报告位置: setup_report.txt")

if __name__ == "__main__":
    asyncio.run(main())