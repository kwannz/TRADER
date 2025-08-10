#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXBatchCollector {
  constructor() {
    // 交易对映射 - 处理特殊情况
    this.symbolMapping = {
      'USDT': 'USDT-USDC',  // USDT使用USDT-USDC交易对
      'USDC': 'USDC-USDT',  // USDC使用USDC-USDT交易对
      'USDS': 'USDS-USDT',  // USDS使用USDS-USDT交易对
      'USDE': 'USDE-USDT',  // USDE使用USDE-USDT交易对
      'STETH': 'STETH-ETH', // STETH使用STETH-ETH交易对
      'WBTC': 'WBTC-BTC',   // WBTC使用WBTC-BTC交易对
      // 其他代币使用默认的 {SYMBOL}-USDT
    };
    
    // 时间周期配置
    this.timeframes = {
      '1D': { name: '日线', startYear: 2018, priority: 1 },
      '4H': { name: '4小时', startYear: 2021, priority: 2 }
    };
    
    // 收集配置
    this.config = {
      concurrent: 3,
      requestDelay: 500,
      batchDelay: 2000,
      maxRetries: 3,
      testMode: false,
      symbolLimit: 10  // 先收集前10个
    };
    
    // 统计信息
    this.stats = {
      startTime: Date.now(),
      totalSymbols: 0,
      completedSymbols: 0,
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalDataPoints: 0,
      errors: [],
      skipped: []
    };
    
    // 进度跟踪
    this.progress = {};
    this.symbols = [];
  }

  async loadSymbols() {
    try {
      const data = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      const allSymbols = JSON.parse(data);
      
      // 限制数量（测试模式）
      this.symbols = allSymbols.slice(0, this.config.symbolLimit);
      this.stats.totalSymbols = this.symbols.length;
      
      console.log(`✅ 加载了前 ${this.symbols.length} 个代币进行收集`);
      console.log(`代币列表: ${this.symbols.join(', ')}`);
      
      // 加载进度
      await this.loadProgress();
      
      return true;
    } catch (error) {
      console.error('❌ 加载代币列表失败:', error.message);
      return false;
    }
  }

  async loadProgress() {
    try {
      const progressFile = path.join(OUTPUT_DIR, 'batch-progress.json');
      const data = await fs.readFile(progressFile, 'utf8');
      this.progress = JSON.parse(data);
      console.log('📊 加载了收集进度');
    } catch (error) {
      this.progress = {};
    }
  }

  async saveProgress() {
    try {
      const progressFile = path.join(OUTPUT_DIR, 'batch-progress.json');
      await fs.writeFile(progressFile, JSON.stringify(this.progress, null, 2));
    } catch (error) {
      console.error('⚠️  保存进度失败:', error.message);
    }
  }

  getInstId(symbol) {
    // 检查是否有特殊映射
    if (this.symbolMapping[symbol]) {
      return this.symbolMapping[symbol];
    }
    // 默认使用 {SYMBOL}-USDT
    return `${symbol}-USDT`;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}, retries = 0) {
    try {
      this.stats.totalRequests++;
      
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
      if (response.data && response.data.code === '0') {
        this.stats.successfulRequests++;
        return response.data;
      } else {
        throw new Error(response.data?.msg || 'API返回错误');
      }
    } catch (error) {
      this.stats.failedRequests++;
      
      if (error.response?.status === 429 && retries < this.config.maxRetries) {
        console.log(`⏳ 频率限制，等待后重试 (${retries + 1}/${this.config.maxRetries})...`);
        await this.delay(5000 * (retries + 1));
        return this.makeRequest(endpoint, params, retries + 1);
      }
      
      throw error;
    }
  }

  async saveData(filename, data) {
    try {
      const filePath = path.join(OUTPUT_DIR, filename);
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
      return true;
    } catch (error) {
      console.error(`💾 保存失败: ${filename} - ${error.message}`);
      return false;
    }
  }

  async collectSymbolTimeframe(symbol, timeframe, config) {
    const instId = this.getInstId(symbol);
    const progressKey = `${symbol}_${timeframe}`;
    
    // 检查是否已完成
    if (this.progress[progressKey]?.completed) {
      console.log(`⏭️  ${symbol} ${config.name} 已收集，跳过`);
      return 0;
    }
    
    console.log(`\n📊 收集 ${symbol} ${config.name} 数据 (交易对: ${instId})...`);
    
    const startDate = new Date(`${config.startYear}-01-01`);
    const endDate = new Date();
    
    let allCandles = [];
    let currentEndTime = endDate.getTime();
    const targetStartTime = startDate.getTime();
    let batchNum = 0;
    const maxBatches = 50; // 限制最大批次数，避免无限循环
    
    try {
      while (currentEndTime > targetStartTime && batchNum < maxBatches) {
        batchNum++;
        
        const batchData = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: timeframe,
          limit: '300',
          after: currentEndTime.toString()
        });
        
        if (!batchData || !batchData.data || batchData.data.length === 0) {
          console.log(`   批次 ${batchNum}: 没有更多数据`);
          break;
        }
        
        console.log(`   批次 ${batchNum}: 获取 ${batchData.data.length} 条数据`);
        allCandles = [...allCandles, ...batchData.data];
        
        const oldestTime = parseInt(batchData.data[batchData.data.length - 1][0]);
        currentEndTime = oldestTime;
        
        if (oldestTime <= targetStartTime) {
          break;
        }
        
        await this.delay(this.config.requestDelay);
      }
      
      // 过滤和排序
      const filteredCandles = allCandles
        .filter(candle => parseInt(candle[0]) >= targetStartTime)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
      
      if (filteredCandles.length === 0) {
        console.log(`⚠️  ${symbol} ${config.name} 没有历史数据`);
        this.stats.skipped.push({ symbol, timeframe, reason: '无数据' });
        return 0;
      }
      
      // 转换格式（简化版）
      const formattedCandles = filteredCandles.map(candle => ({
        t: parseInt(candle[0]),  // timestamp
        o: candle[1],            // open
        h: candle[2],            // high
        l: candle[3],            // low
        c: candle[4],            // close
        v: candle[5]             // volume
      }));
      
      // 保存数据
      const metadata = {
        symbol: symbol,
        instId: instId,
        timeframe: timeframe,
        timeframeName: config.name,
        totalCount: formattedCandles.length,
        startTime: new Date(formattedCandles[0].t).toISOString(),
        endTime: new Date(formattedCandles[formattedCandles.length - 1].t).toISOString(),
        collectedAt: new Date().toISOString()
      };
      
      // 保存元数据
      await this.saveData(`${symbol}/${timeframe}/metadata.json`, metadata);
      
      // 保存数据（压缩格式）
      await this.saveData(`${symbol}/${timeframe}/data.json`, {
        ...metadata,
        data: formattedCandles
      });
      
      // 更新进度
      this.progress[progressKey] = {
        completed: true,
        dataPoints: formattedCandles.length,
        timestamp: new Date().toISOString()
      };
      await this.saveProgress();
      
      console.log(`✅ ${symbol} ${config.name} 完成，共 ${formattedCandles.length} 条数据`);
      console.log(`   时间范围: ${metadata.startTime} 至 ${metadata.endTime}`);
      
      this.stats.totalDataPoints += formattedCandles.length;
      return formattedCandles.length;
      
    } catch (error) {
      console.error(`❌ ${symbol} ${config.name} 收集失败:`, error.message);
      this.stats.errors.push({
        symbol,
        timeframe,
        instId,
        error: error.message,
        timestamp: new Date().toISOString()
      });
      
      this.progress[progressKey] = {
        completed: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
      await this.saveProgress();
      
      return 0;
    }
  }

  async collectSymbol(symbol) {
    console.log(`\n${'='.repeat(50)}`);
    console.log(`🪙 开始收集 ${symbol}`);
    console.log(`${'='.repeat(50)}`);
    
    let totalDataPoints = 0;
    
    for (const [timeframe, config] of Object.entries(this.timeframes)) {
      try {
        const dataPoints = await this.collectSymbolTimeframe(symbol, timeframe, config);
        totalDataPoints += dataPoints;
        await this.delay(this.config.batchDelay);
      } catch (error) {
        console.error(`❌ ${symbol} ${timeframe} 失败:`, error.message);
      }
    }
    
    this.stats.completedSymbols++;
    return totalDataPoints;
  }

  async run() {
    console.log('🚀 开始批量收集历史数据\n');
    
    if (!await this.loadSymbols()) {
      return;
    }
    
    console.log('\n📊 收集配置:');
    console.log(`- 时间周期: ${Object.keys(this.timeframes).join(', ')}`);
    console.log(`- 并发数: ${this.config.concurrent}`);
    console.log(`- 请求延迟: ${this.config.requestDelay}ms`);
    
    // 顺序处理每个代币
    for (const symbol of this.symbols) {
      await this.collectSymbol(symbol);
      
      // 显示进度
      const progress = (this.stats.completedSymbols / this.stats.totalSymbols * 100).toFixed(1);
      console.log(`\n📈 总进度: ${progress}% (${this.stats.completedSymbols}/${this.stats.totalSymbols})`);
    }
    
    // 保存最终统计
    await this.saveStats();
  }

  async saveStats() {
    const elapsed = (Date.now() - this.stats.startTime) / 1000;
    
    const summary = {
      ...this.stats,
      endTime: Date.now(),
      elapsedTime: `${(elapsed / 60).toFixed(1)} 分钟`,
      successRate: `${(this.stats.successfulRequests / this.stats.totalRequests * 100).toFixed(1)}%`,
      averageDataPointsPerSymbol: Math.round(this.stats.totalDataPoints / this.stats.completedSymbols),
      collectedSymbols: this.symbols.slice(0, this.stats.completedSymbols)
    };
    
    await this.saveData('batch-collection-summary.json', summary);
    
    console.log('\n\n📊 收集完成统计:');
    console.log(`- 总代币数: ${this.stats.completedSymbols}/${this.stats.totalSymbols}`);
    console.log(`- 总请求数: ${this.stats.totalRequests}`);
    console.log(`- 成功率: ${summary.successRate}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 平均每代币: ${summary.averageDataPointsPerSymbol.toLocaleString()} 条`);
    console.log(`- 总用时: ${summary.elapsedTime}`);
    
    if (this.stats.errors.length > 0) {
      console.log(`\n⚠️  错误数: ${this.stats.errors.length}`);
      this.stats.errors.forEach(err => {
        console.log(`   - ${err.symbol}: ${err.error}`);
      });
    }
    
    if (this.stats.skipped.length > 0) {
      console.log(`\n⏭️  跳过: ${this.stats.skipped.length} 个`);
    }
  }
}

// 主函数
async function main() {
  const collector = new OKXBatchCollector();
  
  try {
    await collector.run();
    console.log('\n✅ 批量收集完成！');
    console.log('📁 数据保存在: historical/storage/raw/okx/{SYMBOL}/');
  } catch (error) {
    console.error('\n💥 收集过程出错:', error);
    await collector.saveStats();
  }
}

// 运行
main();