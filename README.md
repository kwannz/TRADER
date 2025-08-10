# AI量化交易系统

一个基于AI的个人量化交易系统，集成了多种技术栈和功能模块。

## 📁 项目结构

```
trader/
├── 📚 docs/                          # 文档目录
│   ├── architecture/                 # 架构文档
│   ├── deployment/                   # 部署文档
│   ├── reports/                      # 系统报告
│   ├── guides/                       # 使用指南
│   └── api/                         # API文档
├── 🔧 system/                        # 系统核心模块
│   ├── monitoring/                   # 监控系统
│   ├── risk_management/              # 风险管理
│   ├── alerts/                       # 告警系统
│   └── data_management/              # 数据管理
├── 💻 file_management/               # 文件管理系统
│   ├── web_interface/                # Web界面
│   ├── cli_interface/                # CLI界面
│   └── api_interface/                # API接口
├── 🛠️ scripts/                       # 脚本工具
│   ├── deployment/                   # 部署脚本
│   ├── testing/                      # 测试脚本
│   └── utilities/                    # 工具脚本
├── ⚙️ config/                        # 配置文件
│   ├── api_keys/                     # API密钥
│   ├── environment/                  # 环境配置
│   └── logging/                      # 日志配置
├── 🧠 core/                          # 核心引擎
├── 🐍 python_layer/                  # Python层
├── 🦀 rust_engine/                   # Rust引擎
├── 📊 src/                           # 源代码
├── 🐼 panda_factor-main/             # PandaFactor集成
├── 🏗️ ctbench/                       # CTBench模块
├── 🔌 services/                      # 服务层
├── 📈 data/                          # 数据目录
├── 📝 logs/                          # 日志目录
└── 🖥️ screens/                       # 界面组件
```

## 🚀 快速开始

### 环境要求
- Python 3.9+
- Rust 1.70+
- Node.js 18+

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd trader
```

2. **安装Python依赖**
```bash
pip install -r config/environment/requirements.txt
```

3. **安装Rust依赖**
```bash
cd rust_engine
cargo build --release
cd ..
```

4. **启动系统**
```bash
python scripts/deployment/start.py
```

## 📖 文档

- **架构文档**: `docs/architecture/`
- **部署指南**: `docs/deployment/`
- **使用指南**: `docs/guides/`
- **API文档**: `docs/api/`

## 🔧 主要功能

### 核心功能
- 🤖 AI驱动的因子发现
- 📊 实时市场监控
- ⚠️ 智能风险预警
- 📈 量化策略回测
- 🔄 自动化交易执行

### 技术栈
- **后端**: Python, Rust, FastAPI
- **前端**: HTML5, CSS3, JavaScript
- **数据库**: MongoDB, Redis
- **AI/ML**: TensorFlow, PyTorch
- **量化**: Pandas, NumPy, CTBench

## 📊 系统模块

### 监控系统 (`system/monitoring/`)
- 实时市场数据监控
- 性能指标跟踪
- 系统状态监控

### 风险管理 (`system/risk_management/`)
- 投资组合风险评估
- 风险指标计算
- 风险预警系统

### 告警系统 (`system/alerts/`)
- 智能告警规则
- 多渠道通知
- 告警历史记录

### 数据管理 (`system/data_management/`)
- 数据存储管理
- 数据质量监控
- 数据备份恢复

## 🛠️ 开发工具

### 脚本工具 (`scripts/`)
- **部署脚本**: 自动化部署流程
- **测试脚本**: 单元测试和集成测试
- **工具脚本**: 开发辅助工具

### 配置文件 (`config/`)
- **环境配置**: 不同环境的配置参数
- **API密钥**: 第三方服务密钥管理
- **日志配置**: 日志级别和输出格式

## 🔒 安全说明

- API密钥存储在 `config/api_keys/` 目录
- 生产环境请使用环境变量管理敏感信息
- 定期更新依赖包以修复安全漏洞

## 📝 更新日志

详细的更新日志请查看 `docs/reports/` 目录下的报告文件。

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 这是一个个人量化交易系统，请根据自身风险承受能力谨慎使用。