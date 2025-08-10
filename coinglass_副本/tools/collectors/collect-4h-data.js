#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class Collect4HData {
  constructor() {
    // 选择前20个主要代币收集4小时数据
    this.targetSymbols = [
      'ADA', 'ALGO', 'APT', 'ARB', 'ATOM', 'AVAX', 'BCH', 'LINK',
      'LTC', 'DOT', 'ETC', 'FIL', 'NEAR', 'OKB', 'UNI', 'HBAR',
      'LEO', 'INJ', 'STX', 'CRV'
    ];
    
    this.stats = {
      processed: 0,
      success: 0,
      failed: 0,
      totalDataPoints: 0
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

  async collect4HData(symbol) {
    const instId = `${symbol}-USDT`;
    
    try {
      console.log(`📊 收集 ${symbol} 4小时数据...`);
      
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
        
        process.stdout.write(`\r   批次 ${batches}: ${allCandles.length} 条`);
        await this.delay(500);
      }
      
      console.log(); // 换行
      
      if (allCandles.length === 0) {
        console.log(`❌ ${symbol}: 无4小时数据`);
        this.stats.failed++;
        return;
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
          data: candles
        }, null, 2)
      );
      
      console.log(`✅ ${symbol}: ${candles.length} 条4小时数据`);
      this.stats.success++;
      this.stats.totalDataPoints += candles.length;
      
    } catch (error) {
      console.error(`❌ ${symbol}: 失败 - ${error.message}`);
      this.stats.failed++;
    }
  }

  async run() {
    console.log('🚀 收集主要代币4小时数据\n');
    console.log(`目标代币: ${this.targetSymbols.join(', ')}\n`);
    
    for (const symbol of this.targetSymbols) {
      this.stats.processed++;
      await this.collect4HData(symbol);
      await this.delay(2000);
    }
    
    console.log('\n📊 4小时数据收集完成:');
    console.log(`- 处理: ${this.stats.processed} 个代币`);
    console.log(`- 成功: ${this.stats.success}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 新增数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
  }
}

// 运行
async function main() {
  const collector = new Collect4HData();
  await collector.run();
}

main();