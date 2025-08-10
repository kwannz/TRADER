#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class Complete4HCollection {
  constructor() {
    // 需要补充4小时数据的代币（选择市值较大且有足够数据的）
    this.targetSymbols = [
      // 已有部分4H数据，继续完成
      'BCH', 'LINK', 'LTC', 'DOT', 'ETC', 'FIL', 'NEAR', 'OKB', 'UNI', 'HBAR',
      'LEO', 'INJ', 'STX', 'CRV', 'XLM', 'XTZ', 'THETA', 'MANA', 'IOTA', 'SHIB',
      // 新增重要代币
      'LDO', 'GRT', 'FLOW', 'SAND', 'ICP', 'EGLD', 'AAVE', 'ENS', 'IMX'
    ];
    
    this.stats = {
      processed: 0,
      success: 0,
      failed: 0,
      skipped: 0,
      totalDataPoints: 0,
      startTime: Date.now()
    };
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
        timeout: 30000
      });
      
      if (response.data && response.data.code === '0') {
        return response.data;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  async checkExisting(symbol) {
    try {
      const filePath = path.join(OUTPUT_DIR, symbol, '4H', 'data.json');
      await fs.access(filePath);
      return true;
    } catch (error) {
      return false;
    }
  }

  async collect4HData(symbol) {
    // 检查是否已经有4小时数据
    if (await this.checkExisting(symbol)) {
      console.log(`⏭️  ${symbol.padEnd(8)}: 已有4H数据，跳过`);
      this.stats.skipped++;
      return 0;
    }
    
    const instId = `${symbol}-USDT`;
    
    try {
      process.stdout.write(`📊 ${symbol.padEnd(8)}: `);
      
      const startTime = new Date('2021-01-01').getTime();
      const endTime = new Date().getTime();
      
      let allCandles = [];
      let currentEnd = endTime;
      let batches = 0;
      
      // 收集4小时数据
      while (currentEnd > startTime && batches < 40) {
        const data = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: '4H',
          limit: '300',
          after: currentEnd.toString()
        });
        
        if (!data || !data.data || data.data.length === 0) break;
        
        allCandles = [...allCandles, ...data.data];
        currentEnd = parseInt(data.data[data.data.length - 1][0]);
        batches++;
        
        // 简单进度显示
        process.stdout.write('.');
        await this.delay(400);
      }
      
      if (allCandles.length === 0) {
        console.log(` ❌ 无4H数据`);
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
      const dir = path.join(OUTPUT_DIR, symbol, '4H');
      await fs.mkdir(dir, { recursive: true });
      
      await fs.writeFile(
        path.join(dir, 'data.json'),
        JSON.stringify({
          symbol,
          instId,
          timeframe: '4H',
          count: candles.length,
          start: new Date(candles[0].t).toISOString(),
          end: new Date(candles[candles.length - 1].t).toISOString(),
          collected: new Date().toISOString(),
          data: candles
        }, null, 2)
      );
      
      console.log(` ✅ ${candles.length} 条4H数据`);
      this.stats.success++;
      this.stats.totalDataPoints += candles.length;
      
      return candles.length;
      
    } catch (error) {
      console.log(` ❌ 失败: ${error.message}`);
      this.stats.failed++;
      return 0;
    }
  }

  async run() {
    console.log('🚀 补充主要代币4小时数据\n');
    console.log(`目标代币 (${this.targetSymbols.length}个):`);
    console.log(this.targetSymbols.join(', '));
    console.log();
    
    for (const symbol of this.targetSymbols) {
      this.stats.processed++;
      await this.collect4HData(symbol);
      
      // 进度显示
      const progress = (this.stats.processed / this.targetSymbols.length * 100).toFixed(1);
      console.log(`   进度: ${progress}% (${this.stats.processed}/${this.targetSymbols.length})`);
      
      await this.delay(2000);
    }
    
    // 最终统计
    const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    
    console.log('\n📊 4小时数据补充完成:');
    console.log(`- 处理: ${this.stats.processed} 个代币`);
    console.log(`- 成功: ${this.stats.success}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 跳过: ${this.stats.skipped}`);
    console.log(`- 新增数据: ${this.stats.totalDataPoints.toLocaleString()} 条`);
    console.log(`- 用时: ${elapsed.toFixed(1)} 分钟`);
    
    // 保存统计
    await fs.writeFile(
      path.join(OUTPUT_DIR, `4h-completion-${Date.now()}.json`),
      JSON.stringify({
        stats: this.stats,
        symbols: this.targetSymbols,
        timestamp: new Date().toISOString()
      }, null, 2)
    );
    
    console.log('\n💾 统计已保存');
    console.log('📁 运行 node collection-stats.js 查看更新后的状态');
  }
}

// 运行
async function main() {
  const collector = new Complete4HCollection();
  await collector.run();
}

main();