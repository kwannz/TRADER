#!/usr/bin/env node

import axios from 'axios';

class CoinGlassAuthTester {
  constructor() {
    this.apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    this.baseURL = 'https://open-api-v4.coinglass.com';
  }

  async testAuthMethod(method, endpoint, params = {}) {
    console.log(`🔍 测试认证方法: ${method}`);
    console.log(`📡 端点: ${endpoint}`);
    
    try {
      let config = {
        timeout: 30000
      };

      switch (method) {
        case 'query-param':
          config.params = { ...params, apikey: this.apiKey };
          break;
        case 'header-cg-api-key':
          config.headers = { 'CG-API-KEY': this.apiKey };
          config.params = params;
          break;
        case 'header-authorization':
          config.headers = { 'Authorization': `Bearer ${this.apiKey}` };
          config.params = params;
          break;
        case 'header-x-api-key':
          config.headers = { 'X-API-KEY': this.apiKey };
          config.params = params;
          break;
      }

      const response = await axios.get(`${this.baseURL}${endpoint}`, config);
      
      console.log(`✅ 成功! 状态码: ${response.status}`);
      console.log(`📊 响应数据:`, JSON.stringify(response.data, null, 2).substring(0, 500));
      return { success: true, data: response.data };
      
    } catch (error) {
      console.log(`❌ 失败: ${error.response?.status} - ${error.response?.data?.msg || error.message}`);
      return { success: false, error: error.message };
    }
  }

  async runAuthTests() {
    console.log('🚀 开始CoinGlass API认证测试\n');

    const authMethods = [
      'query-param',
      'header-cg-api-key', 
      'header-authorization',
      'header-x-api-key'
    ];

    // 测试端点列表（从简单到复杂）
    const testEndpoints = [
      { path: '/api/futures/coins-markets', params: {} },
      { path: '/api/futures/pairs-markets', params: { symbol: 'BTC' } },
      { path: '/api/etf/bitcoin/list', params: {} },
      { path: '/api/futures/openInterest/ohlc-history', params: { symbol: 'BTC', interval: '1h' } }
    ];

    const results = {};

    for (const endpoint of testEndpoints) {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`🎯 测试端点: ${endpoint.path}`);
      console.log(`${'='.repeat(60)}\n`);
      
      results[endpoint.path] = {};
      
      for (const method of authMethods) {
        const result = await this.testAuthMethod(method, endpoint.path, endpoint.params);
        results[endpoint.path][method] = result;
        
        console.log(''); // 空行分隔
        
        // 如果成功，跳过其他认证方法测试这个端点
        if (result.success) {
          console.log(`🎉 找到有效认证方法: ${method} for ${endpoint.path}`);
          break;
        }
        
        // 延迟避免限流
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // 汇总结果
    console.log(`\n${'='.repeat(60)}`);
    console.log('📊 测试结果汇总');
    console.log(`${'='.repeat(60)}\n`);

    Object.entries(results).forEach(([endpoint, methods]) => {
      const successMethod = Object.entries(methods).find(([_, result]) => result.success);
      if (successMethod) {
        console.log(`✅ ${endpoint}: ${successMethod[0]} 认证成功`);
      } else {
        console.log(`❌ ${endpoint}: 所有认证方法失败`);
      }
    });

    return results;
  }
}

// 运行测试
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new CoinGlassAuthTester();
  await tester.runAuthTests();
}

export default CoinGlassAuthTester;