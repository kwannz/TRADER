#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';
const OUTPUT_DIR = './historical/storage/raw';

class PreciseBTCETHCollector {
  constructor() {
    this.symbols = ['BTC', 'ETH'];
    this.requestCount = 0;
    this.successCount = 0;
    this.errorCount = 0;
    
    // 确认可用的端点
    this.endpoints = {
      // 期货数据
      'futures-supported-coins': '/api/futures/supported-coins',
      'futures-ohlc': '/api/futures/ohlc-history',
      'futures-open-interest': '/api/futures/open-interest-ohlc-history',
      'futures-funding-rate': '/api/futures/funding-rate-history',
      'futures-liquidation': '/api/futures/liquidation-history',
      
      // 现货数据
      'spot-supported-coins': '/api/spot/supported-coins',
      'spot-ohlc': '/api/spot/ohlc-history',
      
      // ETF数据
      'etf-flows': '/api/etf/flows-history',
      
      // 指标数据
      'fear-greed': '/api/index/fear-greed-history'
    };
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    try {
      this.requestCount++;
      console.log(`📡 [${this.requestCount}] 请求：${endpoint}`);
      
      const response = await axios.get(`${API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'CG-API-KEY': API_KEY,
          'accept': 'application/json'
        },
        timeout: 30000
      });
      
      this.successCount++;
      console.log(`✅ [${this.requestCount}] 成功`);
      return response.data;
    } catch (error) {
      this.errorCount++;
      console.log(`❌ [${this.requestCount}] 失败：${error.message} (${error.response?.status || 'N/A'})`);
      
      if (error.response?.status === 429) {
        console.log('⏳ API限制，等待60秒...');
        await this.delay(60000);
      }
      return null;
    }
  }

  async saveData(filename, data) {
    try {
      const filePath = path.join(OUTPUT_DIR, filename);
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
      console.log(`💾 已保存：${filename}`);
    } catch (error) {
      console.log(`💾 保存失败：${filename} - ${error.message}`);
    }
  }

  async collectHistoricalData() {
    console.log('🚀 开始收集BTC和ETH 2018-2025年历史数据...\n');

    // 时间参数（Unix时间戳）
    const startYear = 2018;
    const currentYear = new Date().getFullYear();

    for (const symbol of this.symbols) {
      console.log(`\n📊 收集 ${symbol} 历史数据...`);
      
      for (let year = startYear; year <= currentYear; year++) {
        console.log(`\n📅 ${symbol} ${year}年数据...`);
        
        const startTime = Math.floor(new Date(`${year}-01-01`).getTime() / 1000);
        const endTime = Math.floor(new Date(`${year}-12-31`).getTime() / 1000);
        
        // 1. 期货OHLC历史数据
        const futuresOHLC = await this.makeRequest('/api/futures/ohlc-history', {
          symbol: symbol,
          interval: '1d',
          startTime: startTime,
          endTime: endTime
        });
        if (futuresOHLC) {
          await this.saveData(`${symbol}/futures/ohlc-${year}.json`, futuresOHLC);
        }
        await this.delay(4000);
        
        // 2. 持仓量OHLC历史数据
        const openInterestOHLC = await this.makeRequest('/api/futures/open-interest-ohlc-history', {
          symbol: symbol,
          interval: '1d',
          startTime: startTime,
          endTime: endTime
        });
        if (openInterestOHLC) {
          await this.saveData(`${symbol}/futures/open-interest-ohlc-${year}.json`, openInterestOHLC);
        }
        await this.delay(4000);
        
        // 3. 资金费率历史数据
        const fundingRate = await this.makeRequest('/api/futures/funding-rate-history', {
          symbol: symbol,
          startTime: startTime,
          endTime: endTime
        });
        if (fundingRate) {
          await this.saveData(`${symbol}/futures/funding-rate-${year}.json`, fundingRate);
        }
        await this.delay(4000);
        
        // 4. 爆仓历史数据
        const liquidation = await this.makeRequest('/api/futures/liquidation-history', {
          symbol: symbol,
          interval: '1d',
          startTime: startTime,
          endTime: endTime
        });
        if (liquidation) {
          await this.saveData(`${symbol}/futures/liquidation-${year}.json`, liquidation);
        }
        await this.delay(4000);
        
        // 5. 现货OHLC历史数据
        const spotOHLC = await this.makeRequest('/api/spot/ohlc-history', {
          symbol: symbol,
          interval: '1d',
          startTime: startTime,
          endTime: endTime
        });
        if (spotOHLC) {
          await this.saveData(`${symbol}/spot/ohlc-${year}.json`, spotOHLC);
        }
        await this.delay(4000);
        
        console.log(`✅ ${symbol} ${year}年数据收集完成`);
      }
    }

    // 收集额外的市场指标
    console.log('\n📈 收集市场指标历史数据...');
    
    // 恐惧贪婪指数
    const fearGreed = await this.makeRequest('/api/index/fear-greed-history', {
      startTime: Math.floor(new Date('2018-01-01').getTime() / 1000),
      endTime: Math.floor(new Date().getTime() / 1000)
    });
    if (fearGreed) {
      await this.saveData('indicators/fear-greed-index.json', fearGreed);
    }

    console.log('\n📈 统计信息：');
    console.log(`总请求数：${this.requestCount}`);
    console.log(`成功：${this.successCount}`);
    console.log(`失败：${this.errorCount}`);
    console.log(`成功率：${((this.successCount / this.requestCount) * 100).toFixed(1)}%`);
  }

  async testEndpoints() {
    console.log('🔍 测试所有端点...\n');
    
    for (const [name, endpoint] of Object.entries(this.endpoints)) {
      console.log(`测试：${name}`);
      const result = await this.makeRequest(endpoint);
      if (result) {
        console.log(`✅ ${name} - 可用`);
      } else {
        console.log(`❌ ${name} - 不可用`);
      }
      await this.delay(2000);
    }
  }
}

// 主函数
async function main() {
  const collector = new PreciseBTCETHCollector();
  
  try {
    // 先测试端点
    console.log('=== 端点测试 ===');
    await collector.testEndpoints();
    
    console.log('\n=== 开始历史数据收集 ===');
    await collector.collectHistoricalData();
    
    console.log('\n🎉 2018-2025年BTC和ETH完整历史数据收集完成！');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
  }
}

// 运行
main();