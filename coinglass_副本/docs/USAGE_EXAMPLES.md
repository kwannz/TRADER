# CoinGlass 数据收集系统使用示例

## 基本使用

### 1. 启动完整数据收集系统

```bash
# 安装依赖
npm install

# 复制配置文件
cp env.example .env

# 初始化数据库
npm run init-db

# 启动系统
npm start
```

### 2. 手动收集特定数据

```bash
# 收集所有数据
npm run collect

# 收集合约市场数据
npm run collect ContractMarket

# 收集资金费率数据
npm run collect FundingRate

# 收集ETF数据
npm run collect ETF
```

## 编程接口使用

### 基本导入和初始化

```javascript
import { CoinGlassDataCollector } from './src/index.js';
import { dbManager } from './src/models/index.js';
import { collectorManager } from './src/collectors/index.js';

// 创建应用实例
const app = new CoinGlassDataCollector();

// 初始化系统
await app.initialize();
```

### 数据收集示例

```javascript
// 手动收集所有数据
const result = await app.collectAllData();
console.log(`收集完成: 成功 ${result.success}, 失败 ${result.failed}`);

// 运行特定收集器
await collectorManager.runCollector('ContractMarket');

// 强制收集所有数据（忽略时间限制）
await collectorManager.forceCollectAll();
```

### 数据库查询示例

```javascript
import { ContractMarket, FundingRate, ETF } from './src/models/index.js';

// 查询最新的BTC合约市场数据
const latestBTCData = await ContractMarket
  .findOne({ symbol: 'BTC' })
  .sort({ timestamp: -1 });

// 查询过去24小时的资金费率
const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
const fundingRates = await FundingRate.find({
  symbol: 'BTC',
  timestamp: { $gte: yesterday }
}).sort({ timestamp: -1 });

// 查询ETF数据统计
const etfStats = await ETF.aggregate([
  { $match: { type: 'bitcoin' } },
  { $group: {
    _id: '$symbol',
    totalNetFlow: { $sum: '$netFlow' },
    avgNetAssets: { $avg: '$netAssets' }
  }}
]);
```

### 系统监控示例

```javascript
// 获取系统状态
const systemStatus = app.getSystemStatus();
console.log('系统状态:', systemStatus);

// 获取收集器状态
const collectorStatus = collectorManager.getCollectorsStatus();
collectorStatus.forEach(collector => {
  console.log(`${collector.name}: ${collector.isHealthy ? '健康' : '异常'}`);
});

// 数据库健康检查
const dbHealth = await dbManager.healthCheck();
console.log('数据库状态:', dbHealth.status);
```

## 数据分析示例

### 1. 市场趋势分析

```javascript
// 分析BTC价格趋势
const btcPrices = await ContractMarket.find({
  symbol: 'BTC',
  timestamp: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
}).sort({ timestamp: 1 });

const priceChange = btcPrices[btcPrices.length - 1].price - btcPrices[0].price;
const percentChange = (priceChange / btcPrices[0].price) * 100;

console.log(`BTC 7天价格变化: ${priceChange.toFixed(2)} USD (${percentChange.toFixed(2)}%)`);
```

### 2. 资金费率分析

```javascript
// 分析平均资金费率
const avgFundingRate = await FundingRate.aggregate([
  { $match: { symbol: 'BTC', fundingTime: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) } }},
  { $group: {
    _id: '$exchange',
    avgRate: { $avg: '$fundingRate' },
    count: { $sum: 1 }
  }},
  { $sort: { avgRate: -1 }}
]);

console.log('24小时平均资金费率:', avgFundingRate);
```

### 3. ETF流向分析

```javascript
// 分析ETF资金流向
const etfFlow = await ETF.aggregate([
  { $match: { 
    type: 'bitcoin',
    timestamp: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
  }},
  { $group: {
    _id: null,
    totalInflow: { $sum: '$inflow' },
    totalOutflow: { $sum: '$outflow' },
    netFlow: { $sum: '$netFlow' }
  }}
]);

console.log('30天ETF资金流向:', etfFlow[0]);
```

### 4. 爆仓数据分析

```javascript
// 分析爆仓趋势
const liquidationTrend = await Liquidation.aggregate([
  { $match: { 
    symbol: 'BTC',
    timestamp: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
  }},
  { $group: {
    _id: { 
      hour: { $hour: '$timestamp' },
      side: '$side'
    },
    totalValue: { $sum: '$value' },
    count: { $sum: 1 }
  }},
  { $sort: { '_id.hour': 1 }}
]);

console.log('24小时爆仓趋势:', liquidationTrend);
```

## 自定义收集器示例

### 创建新的收集器

```javascript
import { BaseCollector } from './src/collectors/BaseCollector.js';
import { coinglassApi } from './src/services/coinglassApi.js';

class CustomDataCollector extends BaseCollector {
  constructor() {
    super('CustomData', YourCustomModel, 60); // 60分钟执行一次
  }

  async collectData() {
    // 实现你的数据收集逻辑
    const data = await coinglassApi.get('/your/custom/endpoint');
    return data;
  }

  transformData(data) {
    // 转换数据格式
    return data.map(item => ({
      // 你的数据映射逻辑
      value: item.value,
      timestamp: new Date(),
      createdAt: new Date(),
      updatedAt: new Date()
    }));
  }

  validateData(data) {
    // 验证数据
    return Array.isArray(data) && data.length > 0;
  }
}

// 使用自定义收集器
const customCollector = new CustomDataCollector();
await customCollector.run();
```

## 定时任务示例

### 设置自定义定时收集

```javascript
import cron from 'node-cron';

// 每小时收集一次特定数据
cron.schedule('0 * * * *', async () => {
  try {
    await collectorManager.runCollector('ETF');
    console.log('ETF数据收集完成');
  } catch (error) {
    console.error('ETF数据收集失败:', error);
  }
});

// 每天凌晨清理旧数据
cron.schedule('0 0 * * *', async () => {
  try {
    const deletedCount = await collectorManager.cleanAllOldData();
    console.log(`清理了 ${deletedCount} 条旧数据`);
  } catch (error) {
    console.error('数据清理失败:', error);
  }
});
```

## 错误处理示例

### 健壮的错误处理

```javascript
async function robustDataCollection() {
  try {
    // 尝试收集数据
    const result = await collectorManager.runAllCollectors();
    
    if (result.failed > 0) {
      console.warn(`部分收集器失败: ${result.failed}/${result.success + result.failed}`);
      
      // 重置错误状态
      collectorManager.resetAllErrors();
      
      // 重试失败的收集器
      setTimeout(async () => {
        await collectorManager.runAllCollectors();
      }, 60000); // 1分钟后重试
    }
    
  } catch (error) {
    console.error('数据收集严重错误:', error);
    
    // 发送告警通知
    // await sendAlert(`数据收集系统异常: ${error.message}`);
    
    // 记录错误日志
    log.error('数据收集系统异常', error);
  }
}
```

## 数据导出示例

### 导出数据到文件

```javascript
import fs from 'fs';

// 导出BTC市场数据到CSV
async function exportBTCData() {
  const data = await ContractMarket.find({
    symbol: 'BTC',
    timestamp: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
  }).sort({ timestamp: 1 });
  
  const csv = [
    'timestamp,price,volume24h,change24hPercent',
    ...data.map(item => 
      `${item.timestamp.toISOString()},${item.price},${item.volume24h},${item.change24hPercent}`
    )
  ].join('\n');
  
  fs.writeFileSync('btc_data.csv', csv);
  console.log('BTC数据已导出到 btc_data.csv');
}

// 导出数据统计报告
async function generateReport() {
  const report = {
    generatedAt: new Date().toISOString(),
    totalRecords: {},
    latestData: {}
  };
  
  // 统计各类数据记录数
  const models = [ContractMarket, FundingRate, ETF, Liquidation];
  
  for (const model of models) {
    const count = await model.countDocuments();
    const latest = await model.findOne().sort({ timestamp: -1 });
    
    report.totalRecords[model.modelName] = count;
    report.latestData[model.modelName] = latest?.timestamp;
  }
  
  fs.writeFileSync('data_report.json', JSON.stringify(report, null, 2));
  console.log('数据报告已生成: data_report.json');
}
```

## API集成示例

### 创建RESTful API服务

```javascript
import express from 'express';
import { ContractMarket, FundingRate } from './src/models/index.js';

const app = express();

// 获取最新市场数据
app.get('/api/market/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const data = await ContractMarket
      .findOne({ symbol: symbol.toUpperCase() })
      .sort({ timestamp: -1 });
    
    if (!data) {
      return res.status(404).json({ error: 'Data not found' });
    }
    
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 获取历史数据
app.get('/api/history/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { days = 7, limit = 100 } = req.query;
    
    const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    
    const data = await ContractMarket.find({
      symbol: symbol.toUpperCase(),
      timestamp: { $gte: startDate }
    })
    .sort({ timestamp: -1 })
    .limit(parseInt(limit));
    
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// 获取系统状态
app.get('/api/status', async (req, res) => {
  try {
    const systemStatus = collectorManager.getSystemStatus();
    const dbHealth = await dbManager.healthCheck();
    
    res.json({
      system: systemStatus,
      database: dbHealth,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('API服务器运行在 http://localhost:3000');
});
```

这些示例展示了如何使用CoinGlass数据收集系统进行各种操作，从基本的数据收集到高级的数据分析和API集成。你可以根据具体需求调整和扩展这些示例。 