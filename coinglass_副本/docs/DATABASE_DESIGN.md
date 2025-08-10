# CoinGlass 数据库设计文档

## 概述

本文档描述了 CoinGlass 数据收集系统的 MongoDB 数据库设计。系统设计用于存储来自 CoinGlass API 的各种加密货币数据。

## 数据库结构

### 数据库名称
- **coinglass_data** - 主数据库

### 集合（Collections）概览

| 集合名称 | 描述 | 主要用途 |
|---------|------|----------|
| contractmarkets | 合约市场数据 | 价格、交易量、涨跌幅 |
| openinterests | 持仓量数据 | 未平仓合约统计 |
| fundingrates | 资金费率 | 合约资金费率历史 |
| longshortraios | 多空比数据 | 账户/持仓多空比例 |
| liquidations | 爆仓数据 | 爆仓事件记录 |
| liquidationheatmaps | 爆仓热力图 | 价格级别爆仓分布 |
| orderbooks | 订单簿数据 | 买卖盘深度信息 |
| largeorders | 大额订单 | 大额买卖订单 |
| activetrades | 主动买卖 | 主动交易统计 |
| spotmarkets | 现货市场 | 现货价格和交易量 |
| options | 期权数据 | 期权相关指标 |
| etfs | ETF数据 | ETF流入流出和净资产 |
| exchangebalances | 交易所余额 | 链上交易所资产 |
| onchaintransfers | 链上转账 | ERC20转账记录 |
| indicators | 技术指标 | RSI、AHR999等指标 |
| feargreedindices | 恐惧贪婪指数 | 市场情绪指标 |
| stablecoinmarketcaps | 稳定币市值 | 稳定币统计 |
| systemstatuses | 系统状态 | 数据收集状态监控 |
| **新增集合** | | |
| aggregated_open_interest | 聚合持仓数据 | 多交易所持仓汇总和分布 |
| liquidation_heatmaps | 爆仓热力图 | 详细价格级别爆仓分析 |
| macro_indicators | 宏观指标 | AHR999、恐惧贪婪等宏观指标 |
| spot_markets | 现货市场扩展 | 现货交易数据扩展 |
| option_aggregates | 期权聚合数据 | 期权市场汇总统计 |

## 详细集合设计

### 1. ContractMarkets (合约市场数据)

```javascript
{
  _id: ObjectId,
  symbol: String,           // 币种符号 (如 "BTC")
  exchange: String,         // 交易所名称
  price: Number,            // 当前价格
  volume24h: Number,        // 24小时交易量
  change24h: Number,        // 24小时涨跌额
  change24hPercent: Number, // 24小时涨跌幅百分比
  high24h: Number,         // 24小时最高价
  low24h: Number,          // 24小时最低价
  timestamp: Date,         // 数据时间戳
  createdAt: Date,         // 创建时间
  updatedAt: Date          // 更新时间
}
```

**索引：**
- `{ symbol: 1, exchange: 1, timestamp: -1 }`
- `{ timestamp: -1 }`

### 2. OpenInterests (持仓数据)

```javascript
{
  _id: ObjectId,
  symbol: String,           // 币种符号
  exchange: String,         // 交易所名称
  timeframe: String,        // 时间框架 (1h, 4h, 1d等)
  openInterest: Number,     // 持仓量
  openInterestValue: Number,// 持仓价值(USD)
  openTime: Date,          // 开盘时间
  closeTime: Date,         // 收盘时间
  timestamp: Date,         // 数据时间戳
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ symbol: 1, exchange: 1, openTime: -1 }`
- `{ timestamp: -1 }`

### 3. FundingRates (资金费率)

```javascript
{
  _id: ObjectId,
  symbol: String,              // 币种符号
  exchange: String,            // 交易所名称
  fundingRate: Number,         // 当前资金费率
  fundingTime: Date,           // 资金费率时间
  nextFundingTime: Date,       // 下次资金费率时间
  predictedFundingRate: Number,// 预测资金费率
  timestamp: Date,             // 数据时间戳
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ symbol: 1, exchange: 1, fundingTime: -1 }`
- `{ timestamp: -1 }`

### 4. LongShortRatios (多空比)

```javascript
{
  _id: ObjectId,
  symbol: String,      // 币种符号
  exchange: String,    // 交易所名称
  longRatio: Number,   // 多头比例
  shortRatio: Number,  // 空头比例
  longAccount: Number, // 多头账户数
  shortAccount: Number,// 空头账户数
  longPosition: Number,// 多头持仓
  shortPosition: Number,// 空头持仓
  timestamp: Date,     // 数据时间戳
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ symbol: 1, exchange: 1, timestamp: -1 }`
- `{ timestamp: -1 }`

### 5. Liquidations (爆仓数据)

```javascript
{
  _id: ObjectId,
  symbol: String,      // 币种符号
  exchange: String,    // 交易所名称
  side: String,        // 爆仓方向 ('long' | 'short')
  size: Number,        // 爆仓数量
  price: Number,       // 爆仓价格
  value: Number,       // 爆仓价值
  timestamp: Date,     // 爆仓时间
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ symbol: 1, exchange: 1, timestamp: -1 }`
- `{ side: 1, timestamp: -1 }`
- `{ timestamp: -1 }`

### 6. ETFs (ETF数据)

```javascript
{
  _id: ObjectId,
  name: String,        // ETF名称
  symbol: String,      // ETF代码
  type: String,        // ETF类型 ('bitcoin' | 'ethereum')
  netAssets: Number,   // 净资产
  inflow: Number,      // 流入金额
  outflow: Number,     // 流出金额
  netFlow: Number,     // 净流入
  premium: Number,     // 溢价
  discount: Number,    // 折价
  price: Number,       // ETF价格
  volume: Number,      // 交易量
  timestamp: Date,     // 数据时间
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ symbol: 1, type: 1, timestamp: -1 }`
- `{ type: 1, timestamp: -1 }`

### 7. Indicators (技术指标)

```javascript
{
  _id: ObjectId,
  name: String,        // 指标名称 (RSI, AHR999等)
  symbol: String,      // 相关币种 (可选)
  value: Number,       // 指标数值
  signal: String,      // 交易信号 ('buy' | 'sell' | 'neutral')
  description: String, // 指标描述
  category: String,    // 指标分类 ('contract' | 'spot' | 'macro')
  timestamp: Date,     // 数据时间
  createdAt: Date,
  updatedAt: Date
}
```

**索引：**
- `{ name: 1, timestamp: -1 }`
- `{ category: 1, timestamp: -1 }`

## 数据存储策略

### 时间序列数据优化
1. **分区策略**: 按月份创建索引，提高查询性能
2. **TTL索引**: 自动清理超过保留期的数据
3. **复合索引**: 优化常用查询路径

### 数据压缩
- 使用 MongoDB 的 WiredTiger 存储引擎压缩
- 数值字段使用适当的数据类型减少存储空间

### 数据保留策略
- 默认保留365天的历史数据
- 可通过配置调整保留期限
- 定时清理任务自动删除过期数据

## 性能优化

### 索引策略
1. **复合索引**: 根据常用查询模式创建
2. **稀疏索引**: 对可选字段使用稀疏索引
3. **部分索引**: 对特定条件数据创建部分索引

### 查询优化
1. **分页查询**: 使用游标分页避免深度分页
2. **聚合管道**: 利用MongoDB聚合框架进行复杂统计
3. **读优化**: 适当使用读首选项配置

### 写入优化
1. **批量写入**: 使用 `insertMany` 进行批量插入
2. **写关注级别**: 根据数据重要性调整写关注级别
3. **连接池**: 优化数据库连接池配置

## 备份和恢复

### 备份策略
1. **定期备份**: 每日自动备份
2. **增量备份**: 使用 oplog 进行增量备份
3. **异地存储**: 备份文件存储到云存储

### 恢复计划
1. **时间点恢复**: 支持任意时间点数据恢复
2. **选择性恢复**: 可恢复特定集合或数据
3. **灾难恢复**: 完整的灾难恢复流程

## 监控和维护

### 性能监控
1. **慢查询监控**: 监控执行时间超过阈值的查询
2. **索引使用率**: 定期检查索引使用情况
3. **存储使用量**: 监控数据库存储空间使用

### 维护任务
1. **索引重建**: 定期重建索引优化性能
2. **数据统计**: 更新集合统计信息
3. **碎片整理**: 定期进行数据库碎片整理

## 扩展性考虑

### 水平扩展
1. **分片策略**: 基于时间或币种进行分片
2. **副本集**: 配置读写分离提高性能
3. **负载均衡**: 使用MongoDB负载均衡

### 垂直扩展
1. **硬件升级**: CPU、内存、存储升级
2. **配置优化**: MongoDB配置参数调优
3. **缓存层**: 添加Redis等缓存层

## 新增集合详细设计

### 8. AggregatedOpenInterest (聚合持仓数据)

```javascript
{
  _id: ObjectId,
  symbol: String,                    // 币种符号
  total_open_interest_usd: Number,   // 总持仓价值(USD)
  total_open_interest_native: Number,// 总持仓数量(原生币)
  exchanges: [{                      // 各交易所分布
    exchange: String,                // 交易所名称
    open_interest_usd: Number,       // 该交易所持仓价值
    open_interest_native: Number,    // 该交易所持仓数量
    percentage: Number               // 占总持仓百分比
  }],
  dominant_exchange: String,         // 主导交易所
  timestamp: Date,                   // 数据时间戳
  created_at: Date                   // 创建时间
}
```

**索引：**
- `{ symbol: 1, timestamp: -1 }`
- `{ timestamp: -1 }`
- `{ dominant_exchange: 1, timestamp: -1 }`

### 9. LiquidationHeatmaps (爆仓热力图)

```javascript
{
  _id: ObjectId,
  symbol: String,                    // 币种符号
  exchange: String,                  // 交易所名称
  price_level: Number,               // 价格级别
  long_liquidation_usd: Number,      // 多头爆仓金额
  short_liquidation_usd: Number,     // 空头爆仓金额
  total_liquidation_usd: Number,     // 总爆仓金额
  cumulative_long: Number,           // 累计多头爆仓
  cumulative_short: Number,          // 累计空头爆仓
  timestamp: Date,                   // 数据时间戳
  created_at: Date                   // 创建时间
}
```

**索引：**
- `{ symbol: 1, exchange: 1, timestamp: -1 }`
- `{ price_level: 1, timestamp: -1 }`
- `{ timestamp: -1 }`

### 10. MacroIndicators (宏观指标)

```javascript
{
  _id: ObjectId,
  name: String,                      // 指标名称
  symbol: String,                    // 相关币种 (可选)
  value: Number,                     // 指标原始值
  normalized_value: Number,          // 标准化值 (0-100)
  signal: String,                    // 交易信号
  signal_strength: Number,           // 信号强度 (0-10)
  description: String,               // 指标描述
  category: String,                  // 指标分类
  timestamp: Date,                   // 数据时间戳
  created_at: Date                   // 创建时间
}
```

**索引：**
- `{ name: 1, timestamp: -1 }`
- `{ category: 1, timestamp: -1 }`
- `{ signal: 1, timestamp: -1 }`

### 11. SpotMarkets (现货市场扩展)

```javascript
{
  _id: ObjectId,
  symbol: String,                    // 币种符号
  exchange: String,                  // 交易所名称
  price: Number,                     // 当前价格
  price_change_24h: Number,          // 24小时价格变化
  price_change_percent_24h: Number,  // 24小时涨跌幅
  volume_24h: Number,                // 24小时交易量
  volume_change_percent_24h: Number, // 交易量变化百分比
  high_24h: Number,                  // 24小时最高价
  low_24h: Number,                   // 24小时最低价
  market_cap: Number,                // 市值
  timestamp: Date,                   // 数据时间戳
  created_at: Date                   // 创建时间
}
```

**索引：**
- `{ symbol: 1, exchange: 1, timestamp: -1 }`
- `{ market_cap: -1, timestamp: -1 }`
- `{ timestamp: -1 }`

### 12. OptionAggregates (期权聚合数据)

```javascript
{
  _id: ObjectId,
  symbol: String,                    // 币种符号
  expiry_date: Date,                 // 到期日期
  max_pain: Number,                  // 最大痛点价格
  put_call_ratio: Number,            // 看跌看涨比
  total_call_oi: Number,             // 总看涨持仓
  total_put_oi: Number,              // 总看跌持仓
  total_call_volume: Number,         // 总看涨交易量
  total_put_volume: Number,          // 总看跌交易量
  implied_volatility: Number,        // 隐含波动率
  delta_neutral_level: Number,       // Delta中性水平
  timestamp: Date,                   // 数据时间戳
  created_at: Date                   // 创建时间
}
```

**索引：**
- `{ symbol: 1, expiry_date: 1, timestamp: -1 }`
- `{ max_pain: 1, timestamp: -1 }`
- `{ timestamp: -1 }`

## 扩展功能特性

### 数据收集优化
1. **智能调度**: 根据API限制自动调整收集频率
2. **优先级管理**: 高价值数据优先收集
3. **错误恢复**: 自动重试和故障恢复机制
4. **数据去重**: 避免重复数据存储

### 数据质量保障
1. **实时验证**: 数据入库前自动验证
2. **异常检测**: 识别和标记异常数据点
3. **完整性检查**: 确保数据链条完整性
4. **一致性维护**: 保持跨集合数据一致性

### 存储优化策略
1. **数据分层**: 热数据和冷数据分离存储
2. **压缩算法**: 使用高效压缩减少存储空间
3. **索引优化**: 智能索引策略提升查询性能
4. **自动清理**: 定期清理过期和无效数据 