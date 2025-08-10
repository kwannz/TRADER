# AI量化交易系统 - 开发环境指南

## 🚀 快速开始

### 方式一：一键启动（推荐）

**macOS/Linux:**
```bash
./start_dev.sh
```

**Windows:**
```cmd
start_dev.bat
```

**跨平台Python脚本:**
```bash
python start_dev.py
```

### 方式二：手动启动

1. **安装依赖**
```bash
pip install -r requirements-dev.txt
```

2. **启动热重载服务器**
```bash
python dev_server.py
```

或启动增强版服务器：
```bash
python server.py --dev
```

## 🔥 热重载功能

### 自动重载规则

| 文件类型 | 触发动作 | 响应时间 |
|---------|---------|---------|
| `.py` 文件 | 后端服务器重启 | ~2秒 |
| `.html/.css/.js` 文件 | 浏览器自动刷新 | ~0.5秒 |
| `.json` 配置文件 | 后端重启 | ~2秒 |

### 监控范围

- 📁 **主目录**: `/` (项目根目录)
- 📁 **Web界面**: `/file_management/web_interface/`
- 📁 **核心代码**: `/core/`
- 📁 **源代码**: `/src/`
- 📁 **Python层**: `/python_layer/`

### 忽略文件

自动忽略以下文件和目录：
- `__pycache__/`, `*.pyc`
- `.git/`, `node_modules/`
- `venv/`, `env/`, `trader_venv/`
- `logs/`, `*.log`
- `.DS_Store`

## 🛠️ 开发工具

### 浏览器开发者工具

1. **打开开发者工具** (F12)
2. **查看控制台**: 实时查看热重载日志
3. **网络面板**: 监控WebSocket连接状态
4. **应用面板**: 检查本地存储和缓存

### 开发指示器

- 📍 **左下角**: "🔧 开发模式" 标识
- 📍 **右上角**: 代码更新通知
- 📍 **控制台**: 详细的开发日志

### API端点

| 端点 | 说明 | 示例 |
|------|------|------|
| `/api/dev/status` | 开发状态信息 | 连接数、模式等 |
| `/api/market` | 市场数据API | 实时价格数据 |
| `/dev-ws` | 开发WebSocket | 热重载通知 |
| `/ws` | 应用WebSocket | 业务数据流 |

## 🌐 访问地址

### 主要界面
- **🏠 主界面**: http://localhost:8000
- **📊 仪表盘**: http://localhost:8000 (默认页面)
- **🧠 AI因子**: http://localhost:8000 (切换到因子发现页面)
- **🎯 策略管理**: http://localhost:8000 (切换到策略页面)

### API测试
- **📈 市场数据**: http://localhost:8000/api/market?symbol=BTC/USDT
- **🔧 开发状态**: http://localhost:8000/api/dev/status
- **🔌 WebSocket测试**: ws://localhost:8000/ws

## 🎯 开发工作流

### 1. 前端开发
```bash
# 修改HTML/CSS/JS文件
edit file_management/web_interface/app.js
# 👀 浏览器自动刷新，无需手动操作
```

### 2. 后端开发
```bash
# 修改Python文件
edit core/ai_engine.py
# 👀 服务器自动重启，保持WebSocket连接
```

### 3. 配置修改
```bash
# 修改配置文件
edit config/settings.py
# 👀 服务器自动重启，重新加载配置
```

### 4. 实时调试
1. 在浏览器中打开开发者工具
2. 修改代码观察自动更新
3. 查看控制台日志排查问题
4. 使用Network面板监控API调用

## 📁 项目结构

```
trader/                          # 项目根目录
├── dev_server.py               # 🔥 热重载开发服务器
├── server.py                   # 📡 增强版生产服务器 
├── dev_client.js               # 🔧 前端热重载客户端
├── start_dev.py                # 🚀 Python启动脚本
├── start_dev.sh                # 🚀 Shell启动脚本 (Unix)
├── start_dev.bat               # 🚀 批处理启动脚本 (Windows)
├── requirements-dev.txt        # 📦 开发环境依赖
├── dev_config.json             # ⚙️ 开发环境配置
├── DEV_ENVIRONMENT.md          # 📖 本文档
├── file_management/web_interface/  # 🌐 Web界面文件
│   ├── index.html              # 主页面 (已集成热重载)
│   ├── app.js                  # 前端JavaScript
│   └── styles.css              # 样式文件
├── core/                       # 🧠 核心业务逻辑
├── src/                        # 💼 源代码目录
└── logs/                       # 📝 日志文件
```

## 🔧 高级配置

### 自定义监控目录

编辑 `dev_config.json`:
```json
{
  "hot_reload": {
    "watch_directories": [
      ".", 
      "my_custom_dir",
      "another_dir"
    ]
  }
}
```

### 自定义端口

启动时指定端口:
```bash
# 修改 dev_server.py 中的端口配置
# 或设置环境变量
export DEV_PORT=3000
python dev_server.py
```

### 禁用自动浏览器

编辑 `dev_config.json`:
```json
{
  "browser": {
    "auto_open": false
  }
}
```

## 🐛 故障排除

### 常见问题

**1. WebSocket连接失败**
```bash
# 检查端口是否被占用
netstat -tulpn | grep 8000  # Linux/macOS
netstat -ano | findstr 8000  # Windows

# 尝试更换端口
python dev_server.py --port 3000
```

**2. 文件监控不工作**
```bash
# 检查watchdog是否安装
pip show watchdog

# 手动重启服务器
Ctrl+C 然后重新运行
```

**3. 浏览器不自动刷新**
- 检查浏览器控制台是否有JavaScript错误
- 确认开发WebSocket连接正常
- 尝试手动刷新页面

**4. 依赖包缺失**
```bash
# 重新安装开发依赖
pip install -r requirements-dev.txt

# 检查特定包
python -c "import watchdog; print('OK')"
```

### 调试模式

启用详细日志:
```bash
# 设置环境变量
export DEBUG=1
python dev_server.py
```

或修改 `dev_config.json`:
```json
{
  "logging": {
    "level": "DEBUG",
    "verbose_logging": true
  }
}
```

## 🎨 自定义开发环境

### 添加自定义通知

在 `dev_client.js` 中添加:
```javascript
devClient.showDevNotification('自定义消息', 'info');
```

### 扩展文件监控

在 `dev_server.py` 中添加新的文件类型:
```python
watch_extensions = {'.py', '.html', '.css', '.js', '.json', '.yaml'}
```

### 集成外部工具

可以集成的开发工具:
- **代码格式化**: Black, isort
- **类型检查**: mypy
- **代码质量**: flake8, pylint
- **测试**: pytest
- **文档**: Sphinx

## 💡 最佳实践

### 1. 代码组织
- 保持文件结构清晰
- 使用有意义的文件名
- 定期清理临时文件

### 2. 性能优化
- 避免在监控目录中放置大文件
- 合理设置刷新延迟
- 定期重启开发服务器

### 3. 协作开发
- 使用版本控制忽略开发临时文件
- 共享开发环境配置
- 文档化自定义修改

### 4. 生产部署
- 开发完成后使用生产模式测试
- 移除开发专用代码
- 优化资源加载

## 📚 扩展阅读

- **Watchdog文档**: https://python-watchdog.readthedocs.io/
- **aiohttp文档**: https://docs.aiohttp.org/
- **WebSocket协议**: https://tools.ietf.org/html/rfc6455
- **前端热重载原理**: https://webpack.js.org/concepts/hot-module-replacement/

---

🎉 **祝您开发愉快！** 如果遇到问题，请查看日志文件或提交Issue。