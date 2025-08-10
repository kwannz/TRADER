#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class CollectRemaining {
  constructor() {
    this.allSymbols = [];
    this.collectedSymbols = new Set();
    this.remainingSymbols = [];
    
    // 配置
    this.config = {
      concurrent: 6,       // 提高并发数
      requestDelay: 200,   // 减少延迟
      timeframe: '1D',     
      startYear: 2020,     
      batchSize: 30        // 每批处理30个
    };
    
    this.stats = {
      startTime: Date.now(),
      processed: 0,
      success: 0,
      failed: 0,
      skipped: 0,
      totalDataPoints: 0
    };
  }

  async loadStatus() {
    try {
      // 加载top150列表
      const symbolsData = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      this.allSymbols = JSON.parse(symbolsData);
      
      // 获取已收集的代币
      const dirs = await fs.readdir(OUTPUT_DIR);
      for (const dir of dirs) {
        if (!dir.includes('.json') && !dir.includes('test-')) {
          // 检查是否有数据文件
          try {
            const symbolPath = path.join(OUTPUT_DIR, dir);
            const timeframes = await fs.readdir(symbolPath);
            if (timeframes.some(tf => tf === '1D')) {
              this.collectedSymbols.add(dir);
            }
          } catch (e) {
            // 目录读取失败
          }
        }
      }
      
      // 计算剩余代币
      this.remainingSymbols = this.allSymbols.filter(s => !this.collectedSymbols.has(s));
      
      console.log(`✅ 状态分析:`);
      console.log(`- Top150总数: ${this.allSymbols.length}`);
      console.log(`- 已收集: ${this.collectedSymbols.size}`);
      console.log(`- 剩余: ${this.remainingSymbols.length}`);
      console.log(`- 剩余代币: ${this.remainingSymbols.slice(0, 10).join(', ')}${this.remainingSymbols.length > 10 ? '...' : ''}\n`);
      
      return true;
    } catch (error) {
      console.error('❌ 状态加载失败:', error.message);
      return false;
    }
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
        timeout: 15000
      });
      
      if (response.data && response.data.code === '0') {
        return response.data;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  getInstId(symbol) {
    // 处理特殊情况
    const specialCases = {
      'USDT': 'skip',
      'USDC': 'skip', 
      'USDS': 'skip',
      'USDE': 'skip',
      'FDUSD': 'skip',
      'USDTB': 'skip',
      'PYUSD': 'skip',
      'STETH': 'STETH-ETH',
      'WBTC': 'WBTC-BTC',
      'LSETH': 'LSETH-ETH'
    };
    
    if (specialCases[symbol] === 'skip') return null;
    if (specialCases[symbol]) return specialCases[symbol];
    return `${symbol}-USDT`;
  }

  async collectSymbol(symbol) {
    const instId = this.getInstId(symbol);
    
    if (!instId) {
      console.log(`⏭️  ${symbol}: 跳过（稳定币或特殊代币）`);
      this.stats.skipped++;
      return 0;
    }
    
    try {
      process.stdout.write(`📊 ${symbol.padEnd(8)} `);
      
      const startTime = new Date(`${this.config.startYear}-01-01`).getTime();
      const endTime = new Date().getTime();
      
      let allCandles = [];
      let currentEnd = endTime;
      let batches = 0;
      
      // 快速收集 - 最多12批
      while (currentEnd > startTime && batches < 12) {
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
      
      // 处理数据
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
          timeframe: this.config.timeframe,
          count: candles.length,
          start: new Date(candles[0].t).toISOString(),
          end: new Date(candles[candles.length - 1].t).toISOString(),
          collected: new Date().toISOString(),
          data: candles
        }, null, 2)
      );
      
      console.log(`✅ ${candles.length.toString().padStart(4)} 条`);
      this.stats.success++;
      this.stats.totalDataPoints += candles.length;
      
      return candles.length;
      
    } catch (error) {
      console.log(`❌ 错误`);
      this.stats.failed++;
      return 0;
    }
  }

  async collectBatch(batch) {
    console.log(`\n批次处理: ${batch.join(', ')}`);
    
    // 分组并发处理
    const chunks = [];
    for (let i = 0; i < batch.length; i += this.config.concurrent) {
      chunks.push(batch.slice(i, i + this.config.concurrent));
    }
    
    for (const chunk of chunks) {
      const promises = chunk.map(symbol => this.collectSymbol(symbol));
      await Promise.all(promises);
      
      // 小延迟避免过载
      await this.delay(1000);
    }
  }

  async run() {
    console.log('🚀 收集剩余代币数据\n');
    
    if (!await this.loadStatus()) return;
    
    if (this.remainingSymbols.length === 0) {
      console.log('✅ 所有代币已收集完成！');
      return;
    }
    
    console.log(`📊 开始收集 ${this.remainingSymbols.length} 个剩余代币\n`);
    
    // 分批处理
    for (let i = 0; i < this.remainingSymbols.length; i += this.config.batchSize) {
      const batch = this.remainingSymbols.slice(i, i + this.config.batchSize);
      const batchNum = Math.floor(i / this.config.batchSize) + 1;
      const totalBatches = Math.ceil(this.remainingSymbols.length / this.config.batchSize);
      
      console.log(`\n${'='.repeat(50)}`);
      console.log(`📦 批次 ${batchNum}/${totalBatches} (${batch.length}个代币)`);
      console.log(`${'='.repeat(50)}`);
      
      await this.collectBatch(batch);
      
      this.stats.processed += batch.length;
      
      // 显示进度
      const progress = (this.stats.processed / this.remainingSymbols.length * 100).toFixed(1);
      const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
      
      console.log(`\n📈 批次 ${batchNum} 完成:`);
      console.log(`- 进度: ${progress}% (${this.stats.processed}/${this.remainingSymbols.length})`);
      console.log(`- 成功: ${this.stats.success}, 失败: ${this.stats.failed}, 跳过: ${this.stats.skipped}`);
      console.log(`- 用时: ${elapsed.toFixed(1)} 分钟`);
      
      // 批次间延迟
      if (i + this.config.batchSize < this.remainingSymbols.length) {
        console.log('⏳ 批次间休息3秒...');
        await this.delay(3000);
      }
    }
    
    // 最终统计
    const totalElapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    
    console.log('\n\n🎉 剩余代币收集完成！');
    console.log(`📊 最终统计:`);
    console.log(`- 处理代币: ${this.stats.processed}`);
    console.log(`- 成功: ${this.stats.success}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 跳过: ${this.stats.skipped}`);
    console.log(`- 新增数据: ${this.stats.totalDataPoints.toLocaleString()} 条`);
    console.log(`- 总用时: ${totalElapsed.toFixed(1)} 分钟`);
    console.log(`- 平均速度: ${(this.stats.processed / totalElapsed).toFixed(1)} 代币/分钟`);
    
    // 保存统计
    await fs.writeFile(
      path.join(OUTPUT_DIR, `remaining-collection-${Date.now()}.json`),
      JSON.stringify({
        config: this.config,
        stats: this.stats,
        processed: this.remainingSymbols.slice(0, this.stats.processed),
        timestamp: new Date().toISOString()
      }, null, 2)
    );
    
    console.log('\n💾 统计已保存');
    console.log('📁 运行 node collection-stats.js 查看最新进度');
  }
}

// 运行
async function main() {
  const collector = new CollectRemaining();
  await collector.run();
}

main();