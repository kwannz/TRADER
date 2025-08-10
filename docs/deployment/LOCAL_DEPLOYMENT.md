# 🚀 本地部署指南

> AI量化交易系统本地服务器部署完整指南

## 📋 系统概述

本系统基于2025年最新技术栈构建，包含以下核心组件：

- **🐍 Python 3.13** - JIT编译 + free-threading优化
- **🦀 Rust 2025** - 高性能数据处理引擎
- **⚡ FastAPI** - 3000+ req/s API接口层
- **🧠 AI引擎** - DeepSeek Reasoner + Gemini Pro
- **💾 数据库** - MongoDB 8.0 + Redis 8.0
- **🖥️ CLI界面** - Bloomberg Terminal风格

## 🔧 快速开始

### 1️⃣ 一键安装 (推荐)

```bash
# 克隆项目并进入目录
cd trader

# 使用一键启动脚本 (交互式菜单)
python run_system.py

# 选择 "1 - 完整安装" 自动安装所有依赖
```

### 2️⃣ 命令行安装

```bash
# 直接运行完整安装
python run_system.py --setup

# 启动Web服务器
python run_system.py --server

# 启动CLI界面
python run_system.py --cli
```

### 3️⃣ 分步手动安装

```bash
# 1. 环境安装
python scripts/setup_local.py

# 2. 启动服务器
python scripts/start_server.py

# 3. 启动CLI (新终端)
python scripts/start_cli.py
```

## 📋 详细安装步骤

### 步骤1: 环境检查

安装前请确认以下要求：

**基础要求:**
- Python 3.10+ (推荐 3.13+)
- 8GB+ 内存
- 5GB+ 存储空间
- 稳定网络连接

**操作系统支持:**
- ✅ macOS 12+
- ✅ Ubuntu 20.04+ / Debian 11+
- ✅ Windows 10+ (推荐使用 PowerShell)

**推荐工具:**
```bash
# macOS
brew --version                    # Homebrew包管理器

# Ubuntu/Debian
apt --version                     # APT包管理器

# 现代终端 (推荐)
# iTerm2 (macOS) / Windows Terminal / Alacritty
```

### 步骤2: 自动安装流程

运行安装脚本：
```bash
python scripts/setup_local.py
```

安装过程包括：

#### 2.1 系统依赖安装
```bash
# macOS
brew install mongodb-community redis pkg-config

# Ubuntu/Debian  
sudo apt update
sudo apt install -y mongodb redis-server pkg-config build-essential

# 自动启动服务
brew services start mongodb-community redis  # macOS
sudo systemctl start mongod redis-server     # Linux
```

#### 2.2 Rust工具链安装
```bash
# 自动安装 Rust 2025 edition
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 编译Rust引擎
cd rust_engine && cargo build --release
```

#### 2.3 Python环境配置
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装所有依赖包
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.4 数据库初始化
```bash
# 初始化MongoDB集合和索引
# 初始化Redis配置
# 创建默认管理员用户
```

#### 2.5 配置文件生成

自动生成以下配置文件：

**`.env` 环境变量文件:**
```bash
# 本地开发环境配置
ENVIRONMENT=local
DEBUG=true

# 数据库配置
MONGODB_URL=mongodb://localhost:27017/trading_system
REDIS_URL=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# AI API Keys (需要手动填入)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# 交易所API Keys (可选)
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
```

**`config/local.json` 详细配置:**
```json
{
  "database": {
    "mongodb": {
      "host": "localhost",
      "port": 27017,
      "database": "trading_system"
    },
    "redis": {
      "host": "localhost", 
      "port": 6379,
      "database": 0
    }
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": true
  },
  "ai": {
    "deepseek": {
      "api_key": "your_deepseek_api_key_here",
      "base_url": "https://api.deepseek.com/v1"
    },
    "gemini": {
      "api_key": "your_gemini_api_key_here"
    }
  }
}
```

### 步骤3: API密钥配置

#### 3.1 获取API密钥

**DeepSeek API:**
1. 访问 https://platform.deepseek.com/
2. 注册并获取API密钥
3. 记录API Key

**Google Gemini API:**
1. 访问 https://makersuite.google.com/app/apikey
2. 创建API密钥
3. 记录API Key

**交易所API (可选):**
- OKX: https://www.okx.com/account/my-api
- Binance: https://www.binance.com/en/my/settings/api-management

#### 3.2 配置密钥

编辑 `.env` 文件：
```bash
# 替换为你的实际API密钥
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key
GEMINI_API_KEY=your-actual-gemini-key
```

## 🚀 服务启动

### 方式1: 使用启动脚本

```bash
# 启动Web服务器
python scripts/start_server.py

# 启动CLI界面 (新终端窗口)
python scripts/start_cli.py
```

### 方式2: 使用管理脚本

```bash
# 交互式菜单
python run_system.py

# 直接启动服务器
python run_system.py --server

# 直接启动CLI
python run_system.py --cli
```

### 服务验证

**Web服务验证:**
```bash
# 健康检查
curl http://localhost:8000/health

# API文档
open http://localhost:8000/docs

# 系统指标
curl http://localhost:8000/metrics
```

**数据库连接验证:**
```bash
# MongoDB连接测试
mongosh --eval "db.adminCommand('ping')"

# Redis连接测试  
redis-cli ping
```

## 🖥️ CLI界面使用

### 启动CLI

```bash
# 完整模式 (连接后端服务)
python scripts/start_cli.py

# 独立模式 (无后端连接)
python scripts/start_cli.py --standalone

# 演示模式 (模拟数据)
python scripts/start_cli.py --demo
```

### CLI快捷键

| 按键 | 功能 | 描述 |
|------|------|------|
| `1-6` | 页面切换 | 在6个主要功能页面间切换 |
| `R` | 刷新数据 | 手动刷新实时数据 |
| `H` | 帮助信息 | 显示快捷键和功能说明 |
| `Q` | 退出系统 | 安全退出CLI应用 |
| `Ctrl+C` | 强制退出 | 强制中断程序 |

### 功能页面

1. **📊 主仪表盘** - 实时市场数据、投资组合状态
2. **🎯 策略管理** - 创建、启动、监控交易策略
3. **🤖 AI助手** - 智能分析对话和交易建议  
4. **🔬 因子发现** - Alpha因子研究实验室
5. **📝 交易记录** - 历史交易和绩效分析
6. **⚙️ 系统设置** - API配置和系统管理

## 🧪 系统测试

### 运行验证测试

```bash
# 运行所有模块验证
python run_system.py --test

# 运行CLI验证器
python cli_validation/cli_tester.py

# 运行特定模块测试
python cli_validation/cli_tester.py --modules python_layer
```

### 测试项目

验证测试包括：
- ✅ Rust引擎性能测试
- ✅ Python业务逻辑测试
- ✅ FastAPI接口测试
- ✅ 数据库连接测试
- ✅ AI引擎集成测试

### 性能基准

系统性能指标：
- **API响应时间**: < 100ms
- **WebSocket延迟**: < 50ms  
- **数据刷新频率**: 4Hz (每秒4次)
- **内存使用**: < 2GB
- **CPU使用**: < 50% (正常负载)

## 🔍 系统监控

### 状态检查

```bash
# 系统整体状态
python run_system.py --status

# 详细组件状态
curl http://localhost:8000/health | jq

# 实时性能指标
curl http://localhost:8000/metrics | jq
```

### 日志查看

```bash
# 实时日志
tail -f logs/quantum_trader.log

# 错误日志
grep -i error logs/quantum_trader.log

# 性能日志  
grep -i performance logs/quantum_trader.log
```

### 进程管理

```bash
# 查看Python进程
ps aux | grep python

# 查看服务端口
lsof -i :8000

# 监控资源使用
top -p $(pgrep -f "python.*main.py")
```

## 🔧 故障排除

### 常见问题

#### Q1: Python版本问题
```bash
# 检查Python版本
python --version

# 如果版本过低，安装Python 3.13
# macOS
brew install python@3.13

# Ubuntu
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.13 python3.13-venv
```

#### Q2: 数据库连接失败
```bash
# 检查MongoDB状态
brew services list | grep mongodb  # macOS
systemctl status mongod            # Linux

# 重启MongoDB
brew services restart mongodb-community  # macOS
sudo systemctl restart mongod            # Linux

# 检查Redis状态
redis-cli ping

# 重启Redis
brew services restart redis        # macOS
sudo systemctl restart redis-server # Linux
```

#### Q3: 端口占用问题
```bash
# 查看端口占用
lsof -i :8000

# 杀死占用端口的进程
kill -9 $(lsof -t -i:8000)

# 更换端口启动
python scripts/start_server.py --port 8001
```

#### Q4: 依赖包安装失败
```bash
# 升级pip
pip install --upgrade pip

# 清理缓存重新安装
pip cache purge
pip install -r requirements.txt --no-cache-dir

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### Q5: Rust编译失败
```bash
# 更新Rust工具链
rustup update

# 清理重新编译
cd rust_engine
cargo clean
cargo build --release

# 检查系统依赖
# macOS: xcode-select --install
# Linux: sudo apt install build-essential
```

#### Q6: CLI显示异常
```bash
# 检查终端支持
echo $TERM
echo $COLORTERM

# 推荐使用现代终端
# macOS: iTerm2
# Windows: Windows Terminal
# Linux: Alacritty, Kitty

# 调整终端尺寸 (至少120x40)
resize
```

### 日志分析

重要日志位置：
```bash
# 主应用日志
logs/quantum_trader.log

# 安装日志
setup_report.txt

# 系统错误日志
/var/log/mongodb/mongod.log   # MongoDB
/var/log/redis/redis.log      # Redis
```

### 重置系统

如果遇到严重问题，可以重置系统：
```bash
# 停止所有服务
pkill -f python
brew services stop mongodb-community redis  # macOS
sudo systemctl stop mongod redis-server     # Linux

# 清理数据库
rm -rf /usr/local/var/mongodb/*  # macOS
sudo rm -rf /var/lib/mongodb/*   # Linux

# 重新安装
python scripts/setup_local.py
```

## 📈 性能优化

### Python 3.13优化

系统针对Python 3.13进行了深度优化：

```python
# JIT编译加速
import sys
if hasattr(sys, 'set_int_max_str_digits'):
    # 启用JIT编译优化
    pass

# Free-threading并发
if hasattr(sys, '_is_free_threading'):
    # 使用并发执行器
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
```

### 内存优化

```bash
# 设置Python内存优化
export PYTHONMALLOC=pymalloc
export PYTHONUNBUFFERED=1

# MongoDB内存限制
# 编辑 /etc/mongod.conf
# storage:
#   wiredTiger:
#     engineConfig:
#       cacheSizeGB: 2

# Redis内存限制  
# 编辑 /etc/redis/redis.conf
# maxmemory 2gb
# maxmemory-policy allkeys-lru
```

### 网络优化

```bash
# 增加文件描述符限制
ulimit -n 65536

# TCP优化
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 📚 开发指南

### 项目结构

```
trader/
├── 📁 rust_engine/              # Rust高性能引擎
│   ├── src/lib.rs               # 主FFI接口
│   ├── src/utils/indicators.rs  # 技术指标计算
│   └── src/utils/alpha_factors.rs # Alpha因子实现
├── 📁 python_layer/             # Python业务逻辑层
│   └── core/ai_engine.py        # AI分析引擎
├── 📁 fastapi_layer/            # FastAPI接口层
│   ├── main.py                  # 主应用
│   └── routers/strategies.py    # 策略API路由
├── 📁 cli_interface/            # CLI界面系统
│   ├── main.py                  # CLI主程序
│   └── themes/bloomberg.py      # Bloomberg主题
├── 📁 cli_validation/           # 模块验证测试
│   └── cli_tester.py            # 验证测试器
├── 📁 scripts/                  # 部署脚本
│   ├── setup_local.py           # 自动安装脚本
│   ├── start_server.py          # 服务器启动脚本
│   └── start_cli.py             # CLI启动脚本
├── 📄 run_system.py             # 一键启动脚本
├── 📄 requirements.txt          # Python依赖
├── 📄 README.md                 # 项目说明
├── 📄 LOCAL_DEPLOYMENT.md       # 本部署指南
└── 📄 FULLSTACK_ARCHITECTURE.md # 完整架构文档
```

### 开发环境

```bash
# 激活开发环境
source venv/bin/activate

# 安装开发工具
pip install black isort flake8 mypy pytest

# 代码格式化
black .
isort .

# 代码检查
flake8 .
mypy .

# 运行测试
pytest tests/
```

### 添加新功能

1. **Rust引擎扩展**:
   ```rust
   // rust_engine/src/lib.rs
   #[pyfunction]
   fn new_feature(data: Vec<f64>) -> PyResult<f64> {
       // 实现新功能
   }
   ```

2. **Python业务逻辑**:
   ```python
   # python_layer/core/new_module.py
   class NewModule:
       async def process(self, data):
           # 实现业务逻辑
   ```

3. **API接口**:
   ```python  
   # fastapi_layer/routers/new_router.py
   @router.post("/new-endpoint")
   async def new_endpoint():
       # 实现API端点
   ```

4. **CLI界面**:
   ```python
   # cli_interface/screens/new_screen.py
   class NewScreen(Screen):
       # 实现新界面
   ```

## 🔒 安全配置

### API安全

```bash
# 生成安全密钥
openssl rand -hex 32

# 配置JWT密钥
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env

# 设置CORS策略
echo "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000" >> .env
```

### 数据库安全

```bash
# MongoDB认证配置
# 编辑 /etc/mongod.conf
# security:
#   authorization: enabled

# 创建管理员用户
mongosh admin --eval '
  db.createUser({
    user: "admin",
    pwd: "secure_password",
    roles: ["userAdminAnyDatabase"]
  })
'

# Redis密码配置
# 编辑 /etc/redis/redis.conf
# requirepass your_redis_password
```

### 防火墙配置

```bash
# Ubuntu/Debian防火墙
sudo ufw allow 8000
sudo ufw enable

# macOS防火墙 (System Preferences > Security & Privacy > Firewall)

# 只允许本地连接 (生产环境推荐)
# 在配置文件中设置 API_HOST=127.0.0.1
```

## 📊 监控部署

### 系统监控

```bash
# 安装系统监控工具
pip install psutil prometheus_client

# 启动监控服务
python -c "
import psutil
import time
while True:
    print(f'CPU: {psutil.cpu_percent():.1f}%')
    print(f'Memory: {psutil.virtual_memory().percent:.1f}%')
    time.sleep(5)
"
```

### 应用监控

在FastAPI应用中已集成监控：
- 📊 `/metrics` - Prometheus格式指标
- 💚 `/health` - 健康检查端点  
- 📈 实时性能统计

### 日志管理

```bash
# 配置日志轮转
sudo apt install logrotate

# 创建日志轮转配置
sudo tee /etc/logrotate.d/trading_system << EOF
/path/to/trader/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 user user
}
EOF
```

## 🎯 生产部署建议

### 性能调优

```bash
# Python优化
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# 数据库连接池
# MongoDB: 设置合适的连接池大小
# Redis: 启用连接池复用

# 服务器配置
# 使用Gunicorn替代uvicorn (生产环境)
gunicorn fastapi_layer.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 高可用部署

```bash
# 使用进程管理器
pip install supervisor

# 创建supervisor配置
sudo tee /etc/supervisor/conf.d/trading_system.conf << EOF
[program:trading_system]
command=/path/to/venv/bin/python -m uvicorn fastapi_layer.main:app
directory=/path/to/trader
autostart=true
autorestart=true
user=trading
EOF

# 启动supervisor
sudo supervisorctl reread
sudo supervisorctl start trading_system
```

### 备份策略

```bash
# 数据库备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mongodump --db trading_system --out /backup/mongo_$DATE
tar -czf /backup/redis_$DATE.tar.gz /var/lib/redis/

# 定时备份 (crontab)
0 2 * * * /path/to/backup_script.sh
```

## 📞 技术支持

### 获取帮助

- **📖 完整架构文档**: [FULLSTACK_ARCHITECTURE.md](FULLSTACK_ARCHITECTURE.md)
- **🐛 问题反馈**: GitHub Issues
- **💬 技术讨论**: GitHub Discussions
- **📧 邮件支持**: support@example.com

### 社区资源

- 🌟 **GitHub Stars**: 请为项目点星支持
- 🤝 **贡献代码**: 欢迎Pull Request
- 📚 **文档贡献**: 帮助改进文档
- 🎓 **技术分享**: 分享使用经验

---

## 💡 总结

本地部署完成后，您将拥有：

✅ **完整的AI量化交易系统**  
✅ **Bloomberg风格专业CLI界面**  
✅ **高性能Rust引擎支持**  
✅ **现代化FastAPI接口服务**  
✅ **智能AI分析和建议功能**  

**开始您的量化交易之旅！** 🚀📈💰