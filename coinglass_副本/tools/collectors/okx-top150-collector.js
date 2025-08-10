#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXTop150Collector {
  constructor() {
    // 加载前150个代币列表
    this.symbols = [];
    
    // 时间周期配置 - 优先收集日线和4小时数据
    this.timeframes = {
      '1D': { name: '日线', startYear: 2018, priority: 1 },
      '4H': { name: '4小时', startYear: 2020, priority: 2 },
      '1H': { name: '1小时', startYear: 2023, priority: 3 },
      '1W': { name: '周线', startYear: 2018, priority: 4 }
    };
    
    // 收集配置
    this.config = {
      concurrent: 3,           // 同时处理3个代币
      requestDelay: 500,       // 请求间隔(ms)
      batchDelay: 2000,        // 批次间隔(ms)
      maxRetries: 3,           // 最大重试次数
      saveProgress: true,      // 保存进度
      resumeFrom: null         // 从指定代币恢复
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
      errors: []
    };
    
    // 进度跟踪
    this.progress = {};
  }

  async loadSymbols() {
    try {
      const data = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      this.symbols = JSON.parse(data);
      this.stats.totalSymbols = this.symbols.length;
      console.log(`✅ 加载了 ${this.symbols.length} 个代币`);
      
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
      const progressFile = path.join(OUTPUT_DIR, 'collection-progress.json');
      const data = await fs.readFile(progressFile, 'utf8');
      this.progress = JSON.parse(data);
      console.log('📊 加载了收集进度');
    } catch (error) {
      // 进度文件不存在，初始化
      this.progress = {};
    }
  }

  async saveProgress() {
    if (!this.config.saveProgress) return;
    
    try {
      const progressFile = path.join(OUTPUT_DIR, 'collection-progress.json');
      await fs.writeFile(progressFile, JSON.stringify(this.progress, null, 2));
    } catch (error) {
      console.error('⚠️  保存进度失败:', error.message);
    }
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
    const instId = `${symbol}-USDT`;
    const progressKey = `${symbol}_${timeframe}`;
    
    // 检查是否已完成
    if (this.progress[progressKey]?.completed) {
      console.log(`⏭️  ${symbol} ${config.name} 已收集，跳过`);
      return 0;
    }
    
    console.log(`\n📊 收集 ${symbol} ${config.name} 数据...`);
    
    const startDate = new Date(`${config.startYear}-01-01`);
    const endDate = new Date();
    
    let allCandles = [];
    let currentEndTime = endDate.getTime();
    const targetStartTime = startDate.getTime();
    let batchNum = 0;
    
    try {
      while (currentEndTime > targetStartTime) {
        batchNum++;
        
        const batchData = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: timeframe,
          limit: '300',
          after: currentEndTime.toString()
        });
        
        if (!batchData || !batchData.data || batchData.data.length === 0) {
          break;
        }
        
        allCandles = [...allCandles, ...batchData.data];
        
        const oldestTime = parseInt(batchData.data[batchData.data.length - 1][0]);
        currentEndTime = oldestTime;
        
        if (oldestTime <= targetStartTime) {
          break;
        }
        
        // 显示进度
        if (batchNum % 5 === 0) {
          const progress = ((endDate.getTime() - currentEndTime) / (endDate.getTime() - targetStartTime) * 100).toFixed(1);
          console.log(`   批次 ${batchNum} - 进度: ${progress}%`);
        }
        
        await this.delay(this.config.requestDelay);
      }
      
      // 过滤和排序
      const filteredCandles = allCandles
        .filter(candle => parseInt(candle[0]) >= targetStartTime)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
      
      // 转换格式
      const formattedCandles = filteredCandles.map(candle => ({
        timestamp: parseInt(candle[0]),
        time: new Date(parseInt(candle[0])).toISOString(),
        open: candle[1],
        high: candle[2],
        low: candle[3],
        close: candle[4],
        volume: candle[5],
        volCcy: candle[6],
        volCcyQuote: candle[7],
        confirm: candle[8]
      }));
      
      // 保存数据
      const saveData = {
        symbol: symbol,
        instId: instId,
        timeframe: timeframe,
        timeframeName: config.name,
        totalCount: formattedCandles.length,
        startTime: formattedCandles.length > 0 ? formattedCandles[0].time : null,
        endTime: formattedCandles.length > 0 ? formattedCandles[formattedCandles.length - 1].time : null,
        data: formattedCandles
      };
      
      await this.saveData(`${symbol}/${timeframe}/${config.startYear}-present.json`, saveData);
      
      // 更新进度
      this.progress[progressKey] = {
        completed: true,
        dataPoints: formattedCandles.length,
        timestamp: new Date().toISOString()
      };
      await this.saveProgress();
      
      console.log(`✅ ${symbol} ${config.name} 完成，共 ${formattedCandles.length} 条数据`);
      
      this.stats.totalDataPoints += formattedCandles.length;
      return formattedCandles.length;
      
    } catch (error) {
      console.error(`❌ ${symbol} ${config.name} 收集失败:`, error.message);
      this.stats.errors.push({
        symbol,
        timeframe,
        error: error.message,
        timestamp: new Date().toISOString()
      });
      
      // 标记为失败但不是完成
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
    console.log(`\n\n${'='.repeat(50)}`);
    console.log(`🪙 开始收集 ${symbol}`);
    console.log(`${'='.repeat(50)}`);
    
    let totalDataPoints = 0;
    
    // 按优先级收集各时间周期
    const sortedTimeframes = Object.entries(this.timeframes)
      .sort((a, b) => a[1].priority - b[1].priority);
    
    for (const [timeframe, config] of sortedTimeframes) {
      try {
        const dataPoints = await this.collectSymbolTimeframe(symbol, timeframe, config);
        totalDataPoints += dataPoints;
        
        // 批次间延迟
        await this.delay(this.config.batchDelay);
      } catch (error) {
        console.error(`❌ ${symbol} ${timeframe} 失败:`, error.message);
      }
    }
    
    return totalDataPoints;
  }

  async collectBatch(symbols) {
    const promises = symbols.map(symbol => this.collectSymbol(symbol));
    const results = await Promise.all(promises);
    return results.reduce((sum, count) => sum + count, 0);
  }

  async collectAll() {
    console.log('🚀 开始收集市值前150代币历史数据\n');
    console.log('📊 收集配置:');
    console.log(`- 代币数量: ${this.symbols.length}`);
    console.log(`- 时间周期: ${Object.keys(this.timeframes).join(', ')}`);
    console.log(`- 并发数: ${this.config.concurrent}`);
    console.log(`- 预计时间: ${Math.ceil(this.symbols.length / this.config.concurrent * 2)} 分钟\n`);
    
    // 分批处理代币
    for (let i = 0; i < this.symbols.length; i += this.config.concurrent) {
      const batch = this.symbols.slice(i, i + this.config.concurrent);
      const batchNum = Math.floor(i / this.config.concurrent) + 1;
      const totalBatches = Math.ceil(this.symbols.length / this.config.concurrent);
      
      console.log(`\n📦 批次 ${batchNum}/${totalBatches}: ${batch.join(', ')}`);
      
      await this.collectBatch(batch);
      
      this.stats.completedSymbols = i + batch.length;
      
      // 显示进度
      const progress = (this.stats.completedSymbols / this.stats.totalSymbols * 100).toFixed(1);
      const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
      const eta = (elapsed / this.stats.completedSymbols * (this.stats.totalSymbols - this.stats.completedSymbols)).toFixed(1);
      
      console.log(`\n📈 总进度: ${progress}% (${this.stats.completedSymbols}/${this.stats.totalSymbols})`);
      console.log(`⏱️  已用时: ${elapsed.toFixed(1)} 分钟, 预计剩余: ${eta} 分钟`);
      
      // 批次间延迟
      if (i + this.config.concurrent < this.symbols.length) {
        console.log('⏳ 批次间延迟...');
        await this.delay(5000);
      }
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
      averageRequestTime: `${(elapsed / this.stats.totalRequests).toFixed(2)} 秒`,
      successRate: `${(this.stats.successfulRequests / this.stats.totalRequests * 100).toFixed(1)}%`,
      symbols: this.symbols
    };
    
    await this.saveData('top150-collection-summary.json', summary);
    
    console.log('\n\n📊 收集完成统计:');
    console.log(`- 总代币数: ${this.stats.completedSymbols}/${this.stats.totalSymbols}`);
    console.log(`- 总请求数: ${this.stats.totalRequests}`);
    console.log(`- 成功率: ${summary.successRate}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 总用时: ${summary.elapsedTime}`);
    
    if (this.stats.errors.length > 0) {
      console.log(`\n⚠️  错误数: ${this.stats.errors.length}`);
      console.log('详见 top150-collection-summary.json');
    }
  }

  async run() {
    try {
      // 加载代币列表
      if (!await this.loadSymbols()) {
        return;
      }
      
      // 开始收集
      await this.collectAll();
      
      console.log('\n✅ 收集完成！');
      console.log('📁 数据保存在: historical/storage/raw/okx/{SYMBOL}/');
      
    } catch (error) {
      console.error('\n💥 收集过程出错:', error);
      console.error(error.stack);
      
      // 保存错误状态
      await this.saveStats();
    }
  }
}

// 主函数
async function main() {
  const collector = new OKXTop150Collector();
  
  // 可以通过命令行参数配置
  const args = process.argv.slice(2);
  if (args.includes('--fast')) {
    collector.config.concurrent = 5;
    collector.config.requestDelay = 300;
  }
  if (args.includes('--resume')) {
    const index = args.indexOf('--resume');
    if (args[index + 1]) {
      collector.config.resumeFrom = args[index + 1];
    }
  }
  
  await collector.run();
}

// 运行
main();