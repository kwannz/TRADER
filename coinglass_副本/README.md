# CoinGlass 数据收集系统

一个完整的 CoinGlass API 数据收集、存储和管理系统，使用 JavaScript 和 MongoDB。

## 功能特性

- 🔄 自动收集 CoinGlass 所有可用数据
- 📊 支持合约、现货、期权、ETF 和链上数据
- 🗄️ MongoDB 数据存储和管理
- ⏰ 可配置的定时数据收集
- 📝 完整的日志记录和错误处理
- 🔧 模块化架构，易于扩展

## 数据类型

### 合约数据
- 交易市场数据
- 持仓历史
- 资金费率
- 多空比
- 爆仓数据
- 订单簿深度
- 主动买卖数据

### 现货数据
- 市场数据
- 订单簿深度
- 主动买卖历史

### 期权数据
- 期权最大痛点
- 持仓和成交量历史

### ETF 数据
- 比特币和以太坊 ETF
- 流入流出历史
- 溢价/折扣数据

### 链上数据
- 交易所余额
- 转账数据

### 技术指标
- RSI、基差等合约指标
- 借贷利率、溢价指数
- 宏观情绪指标

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd btcdata
```

### 2. 安装依赖
```bash
npm install
```

### 3. 配置环境变量
```bash
cp env.example .env
# 编辑 .env 文件，设置你的 MongoDB 连接和其他配置
```

### 4. 确保 MongoDB 运行
```bash
# 启动 MongoDB (如果使用本地安装)
mongod

# 或使用 Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 5. 初始化数据库
```bash
npm run init-db
```

### 6. 启动服务
```bash
npm start
```

### 7. 手动收集数据（可选）
```bash
# 收集所有数据
npm run collect

# 收集特定类型数据
npm run collect ContractMarket
npm run collect FundingRate
```

### 8. 开发模式
```bash
npm run dev
```

## 项目结构

```
src/
├── config/          # 配置文件
├── models/          # MongoDB 数据模型
├── services/        # API 服务和业务逻辑
├── collectors/      # 数据收集器
├── utils/           # 工具函数
└── index.js         # 主入口文件

scripts/             # 脚本文件
logs/                # 日志文件
```

## 配置说明

### 环境变量配置
复制 `env.example` 到 `.env` 并修改以下配置：

```bash
# CoinGlass API配置
COINGLASS_API_KEY=51e89d90bf31473384e7e6c61b75afe7
COINGLASS_BASE_URL=https://open-api-v4.coinglass.com

# MongoDB配置
MONGODB_URI=mongodb://localhost:27017/coinglass_data
MONGODB_DB_NAME=coinglass_data

# 数据收集配置
COLLECTION_INTERVAL_MINUTES=5
DATA_RETENTION_DAYS=365
```

### 数据收集频率
可以在 `src/config/index.js` 中调整各类数据的收集频率：

- 合约市场数据：1分钟
- 资金费率：5分钟  
- 持仓数据：5分钟
- 多空比：5分钟
- 爆仓数据：1分钟
- ETF数据：60分钟
- 技术指标：60分钟

## 命令行工具

### 数据库管理
```bash
npm run init-db              # 初始化数据库和索引
```

### 数据收集
```bash
npm run collect              # 手动收集所有数据
npm run collect --help       # 查看收集器帮助
npm run collect ContractMarket # 收集合约市场数据
npm run collect FundingRate    # 收集资金费率数据
npm run collect ETF           # 收集ETF数据
```

### 服务管理
```bash
npm start                    # 启动数据收集服务
npm run dev                  # 开发模式（使用nodemon）
npm test                     # 运行测试
```

## 系统架构

```
btcdata/
├── src/
│   ├── config/              # 配置管理
│   ├── models/              # MongoDB数据模型
│   ├── services/            # API客户端服务
│   ├── collectors/          # 数据收集器
│   ├── utils/               # 工具函数
│   └── index.js             # 主程序入口
├── scripts/                 # 脚本文件
├── logs/                    # 日志文件
├── DATABASE_DESIGN.md       # 数据库设计文档
└── README.md               # 项目说明
```

## 数据模型

系统支持以下类型的数据收集和存储：

### 合约数据
- **ContractMarket**: 合约市场价格和交易量
- **OpenInterest**: 持仓量历史数据
- **FundingRate**: 资金费率数据
- **LongShortRatio**: 多空比统计
- **Liquidation**: 爆仓事件记录
- **ActiveTrade**: 主动买卖统计

### 现货数据
- **SpotMarket**: 现货市场价格数据

### ETF数据
- **ETF**: 比特币和以太坊ETF流入流出数据

### 技术指标
- **Indicator**: RSI、AHR999等技术指标
- **FearGreedIndex**: 恐惧贪婪指数
- **StablecoinMarketCap**: 稳定币市值统计

### 链上数据
- **ExchangeBalance**: 交易所链上余额
- **OnchainTransfer**: ERC20转账记录

详细的数据库设计请参考 [DATABASE_DESIGN.md](./DATABASE_DESIGN.md)

## 监控和日志

### 日志系统
- 日志文件位置: `logs/app.log`
- 错误日志: `logs/error.log`
- 日志级别: debug, info, warn, error

### 系统监控
系统提供实时监控功能：
- 数据库连接状态
- 收集器运行状态
- 错误统计和报告
- 数据收集成功率

### 健康检查
- 自动检测不健康的收集器
- 数据库连接状态监控
- 系统资源使用监控

## 故障排除

### 常见问题

1. **MongoDB连接失败**
   ```bash
   # 检查MongoDB是否运行
   ps aux | grep mongod
   
   # 检查端口是否被占用
   netstat -an | grep 27017
   ```

2. **API请求失败**
   ```bash
   # 检查API密钥是否正确
   # 检查网络连接
   # 查看错误日志
   tail -f logs/error.log
   ```

3. **数据收集器错误**
   ```bash
   # 重置收集器错误状态
   # 查看系统状态
   npm run collect --help
   ```

### 调试模式
设置环境变量启用调试：
```bash
export LOG_LEVEL=debug
npm run dev
```

## 性能优化

### 数据库优化
- 使用复合索引优化查询
- 配置适当的连接池大小
- 定期清理过期数据

### API调用优化
- 实现请求限流和重试机制
- 使用并发控制避免过度请求
- 缓存常用数据减少API调用

### 内存优化
- 使用流式处理大量数据
- 及时释放不需要的对象
- 监控内存使用情况

## 扩展开发

### 添加新的收集器
1. 创建继承自 `BaseCollector` 的新类
2. 实现 `collectData()` 方法
3. 在 `src/collectors/index.js` 中注册
4. 配置收集频率

### 添加新的数据模型
1. 在 `src/models/schemas.js` 中定义模式
2. 在 `src/models/index.js` 中创建模型
3. 配置适当的索引

### API扩展
1. 在 `src/services/coinglassApi.js` 中添加新方法
2. 在 `src/config/index.js` 中配置端点
3. 实现数据转换和验证逻辑

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件 