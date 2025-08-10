#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';
const OUTPUT_DIR = './historical/storage/raw';

class WorkingBTCETHCollector {
  constructor() {
    this.symbols = ['BTC', 'ETH'];
    this.requestCount = 0;
    this.successCount = 0;
    this.errorCount = 0;
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
      console.log(`❌ [${this.requestCount}] 失败：${error.response?.status || 'N/A'} - ${error.message}`);
      
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

  async collectAvailableData() {
    console.log('🚀 收集所有可用的BTC和ETH数据...\n');

    // 1. 获取基础信息
    console.log('📋 收集基础信息...');
    
    const futuresCoins = await this.makeRequest('/api/futures/supported-coins');
    if (futuresCoins) {
      await this.saveData('futures-supported-coins.json', futuresCoins);
    }
    await this.delay(3000);

    const spotCoins = await this.makeRequest('/api/spot/supported-coins');
    if (spotCoins) {
      await this.saveData('spot-supported-coins.json', spotCoins);
    }
    await this.delay(3000);

    // 2. 尝试所有可能的实时数据端点
    const endpoints = [
      // 期货市场数据
      { path: '/api/futures/coins-markets', name: 'futures-markets' },
      { path: '/api/futures/liquidation/history', name: 'liquidation-history' },
      { path: '/api/futures/open-interest/history', name: 'open-interest-history' },
      { path: '/api/futures/funding-rate/history', name: 'funding-rate-history' },
      { path: '/api/futures/long-short-ratio/history', name: 'long-short-ratio-history' },
      
      // 现货市场数据  
      { path: '/api/spot/coins-markets', name: 'spot-markets' },
      { path: '/api/spot/orderbook/history', name: 'spot-orderbook-history' },
      { path: '/api/spot/taker-buysell-ratio/history', name: 'spot-taker-history' },
      
      // ETF数据
      { path: '/api/etf/bitcoin/list', name: 'bitcoin-etf-list' },
      { path: '/api/etf/bitcoin/netassets-history', name: 'bitcoin-etf-netassets' },
      
      // 指标数据
      { path: '/api/index/fear-greed-history', name: 'fear-greed-index' },
      { path: '/api/index/crypto-fear-greed-history', name: 'crypto-fear-greed' },
      
      // 链上数据
      { path: '/api/onchain/exchange-transfer', name: 'exchange-transfers' }
    ];

    for (const endpoint of endpoints) {
      console.log(`\n🔍 测试端点：${endpoint.name}`);
      const data = await this.makeRequest(endpoint.path);
      if (data) {
        await this.saveData(`${endpoint.name}.json`, data);
      }
      await this.delay(3000);
    }

    // 3. 尝试获取具体币种的数据
    for (const symbol of this.symbols) {
      console.log(`\n📊 尝试收集 ${symbol} 具体数据...`);
      
      // 期货数据（带币种参数）
      const futuresEndpoints = [
        { path: '/api/futures/coins-markets', name: `${symbol}-futures-markets`, params: { symbol } },
        { path: '/api/futures/liquidation/history', name: `${symbol}-liquidation`, params: { symbol } },
        { path: '/api/futures/open-interest/history', name: `${symbol}-open-interest`, params: { symbol } },
        { path: '/api/futures/funding-rate/history', name: `${symbol}-funding-rate`, params: { symbol } }
      ];

      for (const endpoint of futuresEndpoints) {
        const data = await this.makeRequest(endpoint.path, endpoint.params);
        if (data) {
          await this.saveData(`${symbol}/${endpoint.name}.json`, data);
        }
        await this.delay(3000);
      }

      // 现货数据（带币种参数）
      const spotEndpoints = [
        { path: '/api/spot/coins-markets', name: `${symbol}-spot-markets`, params: { symbol } },
        { path: '/api/spot/taker-buysell-ratio/history', name: `${symbol}-spot-taker`, params: { symbol } }
      ];

      for (const endpoint of spotEndpoints) {
        const data = await this.makeRequest(endpoint.path, endpoint.params);
        if (data) {
          await this.saveData(`${symbol}/${endpoint.name}.json`, data);
        }
        await this.delay(3000);
      }
    }

    console.log('\n📈 统计信息：');
    console.log(`总请求数：${this.requestCount}`);
    console.log(`成功：${this.successCount}`);
    console.log(`失败：${this.errorCount}`);
    console.log(`成功率：${((this.successCount / this.requestCount) * 100).toFixed(1)}%`);
  }
}

// 主函数
async function main() {
  const collector = new WorkingBTCETHCollector();
  
  try {
    await collector.collectAvailableData();
    console.log('\n🎉 所有可用的BTC和ETH数据收集完成！');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
  }
}

// 运行
main();