#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';

class CoinGlassAPI {
  constructor() {
    this.apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    this.baseURL = 'https://open-api-v4.coinglass.com';
    this.requestDelay = 2000; // 2秒延迟避免限流
    this.maxRetries = 3;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        console.log(`📡 请求: ${endpoint} (尝试 ${attempt}/${this.maxRetries})`);
        
        const response = await axios.get(url, {
          params: params,
          headers: {
            'Accept': 'application/json',
            'User-Agent': 'CoinGlass-Data-Collector/1.0',
            'CG-API-KEY': this.apiKey
          },
          timeout: 30000
        });

        if (response.data && response.data.success) {
          console.log(`✅ 成功获取数据: ${endpoint}`);
          return response.data;
        } else {
          console.log(`⚠️  API返回错误: ${response.data?.msg || 'Unknown error'}`);
          return null;
        }

      } catch (error) {
        console.log(`❌ 请求失败 (${attempt}/${this.maxRetries}): ${error.message}`);
        
        if (attempt < this.maxRetries) {
          const delayTime = this.requestDelay * attempt;
          console.log(`⏳ 等待 ${delayTime/1000}s 后重试...`);
          await this.delay(delayTime);
        }
      }
    }
    
    return null;
  }

  // 获取支持的币种列表
  async getSupportedCoins() {
    return await this.makeRequest('/futures/supported-coins');
  }

  // 获取期货持仓量OHLC历史数据
  async getFuturesOpenInterestOHLC(symbol = 'BTC', interval = '1h') {
    return await this.makeRequest('/api/futures/openInterest/ohlc-history', {
      symbol,
      interval
    });
  }

  // 获取资金费率OHLC历史数据
  async getFundingRateOHLC(symbol = 'BTC', interval = '8h') {
    return await this.makeRequest('/api/futures/fundingRate/ohlc-history', {
      symbol,
      interval
    });
  }

  // 获取期货市场数据
  async getFuturesMarkets() {
    return await this.makeRequest('/api/futures/coins-markets');
  }

  // 获取期货价格OHLC历史
  async getPriceOHLC(symbol = 'BTC', interval = '1h') {
    return await this.makeRequest('/api/price/ohlc-history', {
      symbol,
      interval
    });
  }

  // 获取比特币ETF列表
  async getBitcoinETFList() {
    return await this.makeRequest('/api/etf/bitcoin/list');
  }

  // 获取比特币ETF净资产历史
  async getBitcoinETFNetAssets() {
    return await this.makeRequest('/api/etf/bitcoin/net-assets/history');
  }

  // 获取比特币ETF流入流出历史
  async getBitcoinETFFlow() {
    return await this.makeRequest('/api/etf/bitcoin/flow-history');
  }

  // 获取资金费率交易所列表
  async getFundingRateExchanges(symbol = 'BTC') {
    return await this.makeRequest('/api/futures/fundingRate/exchange-list', {
      symbol
    });
  }

  // 测试API连通性
  async testConnection() {
    console.log('🔄 测试CoinGlass API连接...\n');
    
    // 尝试一些基础端点，看看哪些是免费的
    const testEndpoints = [
      '/futures/supported-coins',
      '/api/futures/pairs-markets', 
      '/futures/supported-exchange-pairs',
      '/futures/price-change-list',
      '/api/etf/bitcoin/list'
    ];

    const tests = testEndpoints.map(endpoint => ({
      name: endpoint.replace('/api/', '').replace('/', ' '),
      method: () => this.makeRequest(endpoint)
    }));

    const results = {};
    
    for (const test of tests) {
      try {
        const result = await test.method();
        results[test.name] = {
          success: !!result,
          dataCount: result?.data?.length || 0,
          error: result ? null : 'No data returned'
        };
        
        await this.delay(this.requestDelay);
      } catch (error) {
        results[test.name] = {
          success: false,
          dataCount: 0,
          error: error.message
        };
      }
    }

    // 输出测试结果
    console.log('📊 API连接测试结果:\n');
    Object.entries(results).forEach(([name, result]) => {
      const status = result.success ? '✅' : '❌';
      const info = result.success ? `${result.dataCount} 条数据` : result.error;
      console.log(`${status} ${name}: ${info}`);
    });

    const successCount = Object.values(results).filter(r => r.success).length;
    console.log(`\n📈 成功率: ${successCount}/${tests.length} (${(successCount/tests.length*100).toFixed(1)}%)`);
    
    return results;
  }
}

// 如果直接运行此文件，执行API测试
if (import.meta.url === `file://${process.argv[1]}`) {
  const api = new CoinGlassAPI();
  await api.testConnection();
}

export default CoinGlassAPI;