#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

// 测试收集器 - 只收集前5个代币的日线数据
class OKXTestCollector {
  constructor() {
    // 测试前5个代币
    this.testSymbols = ['BTC', 'ETH', 'USDT', 'XRP', 'BNB'];
    
    this.stats = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0
    };
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    try {
      this.stats.totalRequests++;
      console.log(`📡 请求: ${params.instId} ${params.bar}`);
      
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
        this.stats.failedRequests++;
        console.log(`⚠️  API错误: ${response.data?.msg}`);
        return null;
      }
    } catch (error) {
      this.stats.failedRequests++;
      console.error(`❌ 请求失败: ${error.message}`);
      return null;
    }
  }

  async collectSymbol(symbol) {
    console.log(`\n🪙 测试收集 ${symbol} 日线数据...`);
    
    const instId = `${symbol}-USDT`;
    let allCandles = [];
    
    // 只获取最近300条数据作为测试
    const data = await this.makeRequest('/api/v5/market/history-candles', {
      instId: instId,
      bar: '1D',
      limit: '300'
    });
    
    if (data && data.data) {
      allCandles = data.data;
      
      // 转换格式
      const formattedCandles = allCandles.map(candle => ({
        timestamp: parseInt(candle[0]),
        time: new Date(parseInt(candle[0])).toISOString(),
        open: candle[1],
        high: candle[2],
        low: candle[3],
        close: candle[4],
        volume: candle[5],
        volCcy: candle[6]
      }));
      
      // 保存测试数据
      const saveData = {
        symbol: symbol,
        instId: instId,
        timeframe: '1D',
        dataCount: formattedCandles.length,
        latestDate: formattedCandles[0]?.time,
        oldestDate: formattedCandles[formattedCandles.length - 1]?.time,
        sampleData: formattedCandles.slice(0, 5)
      };
      
      const filePath = path.join(OUTPUT_DIR, `test-data/${symbol}-1D-test.json`);
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, JSON.stringify(saveData, null, 2));
      
      console.log(`✅ ${symbol}: 获取 ${formattedCandles.length} 条数据`);
      console.log(`   时间范围: ${saveData.oldestDate} 至 ${saveData.latestDate}`);
    }
    
    await this.delay(1000);
  }

  async run() {
    console.log('🧪 OKX Top150 收集器测试\n');
    console.log(`将测试以下代币: ${this.testSymbols.join(', ')}\n`);
    
    for (const symbol of this.testSymbols) {
      await this.collectSymbol(symbol);
    }
    
    console.log('\n📊 测试统计:');
    console.log(`- 总请求: ${this.stats.totalRequests}`);
    console.log(`- 成功: ${this.stats.successfulRequests}`);
    console.log(`- 失败: ${this.stats.failedRequests}`);
    console.log(`- 成功率: ${(this.stats.successfulRequests / this.stats.totalRequests * 100).toFixed(1)}%`);
    
    console.log('\n✅ 测试完成！');
    console.log('📁 测试数据保存在: historical/storage/raw/okx/test-data/');
  }
}

// 运行测试
async function main() {
  const tester = new OKXTestCollector();
  await tester.run();
}

main();