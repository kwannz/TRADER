#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXFastCollector {
  constructor() {
    // 配置
    this.config = {
      startIndex: 10,      // 从第11个代币开始
      endIndex: 30,        // 先收集到第30个
      concurrent: 5,       // 同时处理5个
      requestDelay: 300,   // 请求间隔
      timeframe: '1D',     // 只收集日线数据
      startYear: 2020      // 从2020年开始
    };
    
    // 特殊交易对映射
    this.specialPairs = {
      'USDT': 'skip',      // 跳过稳定币
      'USDC': 'skip',
      'USDS': 'skip',
      'USDE': 'skip',
      'STETH': 'STETH-ETH',
      'WBTC': 'WBTC-BTC'
    };
    
    this.stats = {
      startTime: Date.now(),
      processed: 0,
      success: 0,
      failed: 0,
      skipped: 0,
      totalDataPoints: 0
    };
    
    this.symbols = [];
  }

  async loadSymbols() {
    try {
      const data = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      const allSymbols = JSON.parse(data);
      this.symbols = allSymbols.slice(this.config.startIndex, this.config.endIndex);
      
      console.log(`✅ 加载代币 ${this.config.startIndex + 1} 到 ${this.config.endIndex}`);
      console.log(`代币列表: ${this.symbols.join(', ')}\n`);
      
      return true;
    } catch (error) {
      console.error('❌ 加载代币列表失败:', error.message);
      return false;
    }
  }

  getInstId(symbol) {
    if (this.specialPairs[symbol] === 'skip') return null;
    if (this.specialPairs[symbol]) return this.specialPairs[symbol];
    return `${symbol}-USDT`;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    try {
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 20000
      });
      
      if (response.data && response.data.code === '0') {
        return response.data;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  async collectSymbol(symbol) {
    const instId = this.getInstId(symbol);
    
    if (!instId) {
      console.log(`⏭️  ${symbol}: 跳过（稳定币）`);
      this.stats.skipped++;
      return 0;
    }
    
    try {
      process.stdout.write(`📊 ${symbol} (${instId})... `);
      
      const startTime = new Date(`${this.config.startYear}-01-01`).getTime();
      const endTime = new Date().getTime();
      
      let allCandles = [];
      let currentEnd = endTime;
      let batches = 0;
      
      // 快速收集 - 最多10批
      while (currentEnd > startTime && batches < 10) {
        const data = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: this.config.timeframe,
          limit: '300',
          after: currentEnd.toString()
        });
        
        if (!data || !data.data || data.data.length === 0) break;
        
        allCandles = [...allCandles, ...data.data];
        currentEnd = parseInt(data.data[data.data.length - 1][0]);
        batches++;
        
        await this.delay(this.config.requestDelay);
      }
      
      if (allCandles.length === 0) {
        console.log(`❌ 无数据`);
        this.stats.failed++;
        return 0;
      }
      
      // 简化数据保存
      const candles = allCandles
        .filter(c => parseInt(c[0]) >= startTime)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .map(c => ({
          t: parseInt(c[0]),
          o: c[1],
          h: c[2],
          l: c[3],
          c: c[4],
          v: c[5]
        }));
      
      // 保存
      const dir = path.join(OUTPUT_DIR, symbol, this.config.timeframe);
      await fs.mkdir(dir, { recursive: true });
      
      await fs.writeFile(
        path.join(dir, 'data.json'),
        JSON.stringify({
          symbol,
          instId,
          count: candles.length,
          start: new Date(candles[0].t).toISOString(),
          end: new Date(candles[candles.length - 1].t).toISOString(),
          data: candles
        }, null, 2)
      );
      
      console.log(`✅ ${candles.length} 条`);
      this.stats.success++;
      this.stats.totalDataPoints += candles.length;
      
      return candles.length;
      
    } catch (error) {
      console.log(`❌ 错误: ${error.message}`);
      this.stats.failed++;
      return 0;
    }
  }

  async collectBatch(batch) {
    const promises = batch.map(symbol => this.collectSymbol(symbol));
    await Promise.all(promises);
  }

  async run() {
    console.log('🚀 快速批量收集器\n');
    console.log(`📊 配置:`);
    console.log(`- 并发数: ${this.config.concurrent}`);
    console.log(`- 时间周期: ${this.config.timeframe}`);
    console.log(`- 起始年份: ${this.config.startYear}\n`);
    
    if (!await this.loadSymbols()) return;
    
    // 分批处理
    for (let i = 0; i < this.symbols.length; i += this.config.concurrent) {
      const batch = this.symbols.slice(i, i + this.config.concurrent);
      const batchNum = Math.floor(i / this.config.concurrent) + 1;
      const totalBatches = Math.ceil(this.symbols.length / this.config.concurrent);
      
      console.log(`\n批次 ${batchNum}/${totalBatches}:`);
      await this.collectBatch(batch);
      
      this.stats.processed = i + batch.length;
      
      // 批次间延迟
      if (i + this.config.concurrent < this.symbols.length) {
        await this.delay(2000);
      }
    }
    
    // 统计
    const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    console.log('\n\n📊 收集完成:');
    console.log(`- 处理: ${this.stats.processed} 个代币`);
    console.log(`- 成功: ${this.stats.success}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 跳过: ${this.stats.skipped}`);
    console.log(`- 数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 用时: ${elapsed.toFixed(1)} 分钟`);
    console.log(`- 速度: ${(this.stats.processed / elapsed).toFixed(1)} 代币/分钟`);
    
    // 保存统计
    await fs.writeFile(
      path.join(OUTPUT_DIR, `fast-collection-${this.config.startIndex}-${this.config.endIndex}.json`),
      JSON.stringify({
        config: this.config,
        stats: this.stats,
        symbols: this.symbols,
        timestamp: new Date().toISOString()
      }, null, 2)
    );
  }
}

// 主函数
async function main() {
  const collector = new OKXFastCollector();
  
  // 支持命令行参数
  const args = process.argv.slice(2);
  if (args.includes('--range')) {
    const rangeIndex = args.indexOf('--range');
    if (args[rangeIndex + 1]) {
      const [start, end] = args[rangeIndex + 1].split('-').map(Number);
      collector.config.startIndex = start;
      collector.config.endIndex = end;
    }
  }
  
  await collector.run();
}

main();