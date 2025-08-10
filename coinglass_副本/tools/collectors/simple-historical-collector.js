#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';
const OUTPUT_DIR = './historical/storage/raw';

class SimpleHistoricalCollector {
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
      
      // 检查响应是否成功
      if (response.data && response.data.code === '0') {
        this.successCount++;
        console.log(`✅ [${this.requestCount}] 成功`);
        return response.data;
      } else {
        this.errorCount++;
        console.log(`⚠️  [${this.requestCount}] API返回错误：${response.data?.msg || '未知错误'}`);
        return null;
      }
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

  async collectHistoricalData() {
    console.log('🚀 开始收集可用的历史数据...\n');

    // 1. 收集恐惧贪婪指数历史
    console.log('📊 收集恐惧贪婪指数历史数据...');
    const fearGreedData = await this.makeRequest('/api/index/fear-greed-history');
    if (fearGreedData) {
      await this.saveData('indicators/fear-greed-history-complete.json', fearGreedData);
    }
    await this.delay(3000);

    // 2. 尝试获取支持的交易所列表
    console.log('\n📊 获取支持的交易所列表...');
    const exchangesData = await this.makeRequest('/api/futures/supported-exchanges');
    if (exchangesData) {
      await this.saveData('supported-exchanges.json', exchangesData);
      console.log('支持的交易所：', exchangesData.data);
    }
    await this.delay(3000);

    // 3. 测试可用的历史数据端点
    const testEndpoints = [
      // 期货历史端点
      { path: '/api/futures/liquidation/history', name: 'liquidation-history' },
      { path: '/api/futures/open-interest/history', name: 'open-interest-history' },
      { path: '/api/futures/funding-rate/history', name: 'funding-rate-history' },
      { path: '/api/futures/long-short-ratio/history', name: 'long-short-ratio-history' },
      
      // 现货历史端点
      { path: '/api/spot/orderbook/history', name: 'spot-orderbook-history' },
      { path: '/api/spot/taker-buysell-ratio/history', name: 'spot-taker-history' },
      
      // ETF历史端点
      { path: '/api/etf/bitcoin/netassets-history', name: 'bitcoin-etf-netassets' },
      { path: '/api/etf/flows-history', name: 'etf-flows-history' }
    ];

    console.log('\n📊 测试历史数据端点...');
    for (const endpoint of testEndpoints) {
      console.log(`\n测试：${endpoint.name}`);
      
      // 先测试不带参数
      let data = await this.makeRequest(endpoint.path);
      if (data) {
        await this.saveData(`test/${endpoint.name}-noparams.json`, data);
      }
      await this.delay(3000);
      
      // 测试带币种参数
      for (const symbol of this.symbols) {
        data = await this.makeRequest(endpoint.path, { symbol });
        if (data) {
          await this.saveData(`test/${symbol}/${endpoint.name}.json`, data);
        }
        await this.delay(3000);
      }
    }

    // 4. 尝试获取最近的数据（不指定时间范围）
    console.log('\n📊 收集最近的市场数据...');
    for (const symbol of this.symbols) {
      console.log(`\n收集 ${symbol} 最近数据...`);
      
      // 期货数据
      const futuresEndpoints = [
        { path: '/api/futures/liquidation/info', name: 'liquidation-recent' },
        { path: '/api/futures/open-interest/ohlc-statistics', name: 'open-interest-stats' },
        { path: '/api/futures/funding-rate/ohlc-statistics', name: 'funding-rate-stats' }
      ];
      
      for (const endpoint of futuresEndpoints) {
        const data = await this.makeRequest(endpoint.path, { symbol });
        if (data) {
          await this.saveData(`${symbol}/recent/${endpoint.name}.json`, data);
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
  const collector = new SimpleHistoricalCollector();
  
  try {
    await collector.collectHistoricalData();
    console.log('\n🎉 可用历史数据收集完成！');
    console.log('\n💡 提示：');
    console.log('- 大部分历史数据端点需要升级API计划');
    console.log('- 当前计划可以获取实时数据和部分历史指标');
    console.log('- 已收集的数据保存在 historical/storage/raw 目录');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
  }
}

// 运行
main();