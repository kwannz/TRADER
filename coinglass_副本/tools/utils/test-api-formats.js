#!/usr/bin/env node

import axios from 'axios';

class APIFormatTester {
  constructor() {
    this.apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    this.baseURL = 'https://open-api-v4.coinglass.com';
  }

  async testFormat(description, config) {
    console.log(`🔍 测试: ${description}`);
    
    try {
      const response = await axios.get(`${this.baseURL}/api/futures/coins-markets`, {
        ...config,
        timeout: 15000
      });
      
      const data = response.data;
      if (data.code === "0" || data.success) {
        console.log(`✅ 成功! 返回数据:`);
        console.log(JSON.stringify(data, null, 2).substring(0, 300));
        return true;
      } else {
        console.log(`⚠️  API响应: ${data.msg || data.message || '未知错误'}`);
        return false;
      }
      
    } catch (error) {
      console.log(`❌ 错误: ${error.response?.data?.msg || error.message}`);
      return false;
    }
  }

  async runTests() {
    console.log('🚀 测试CoinGlass API密钥格式\n');

    const tests = [
      // Query parameter variations
      {
        description: 'apikey参数',
        config: { params: { apikey: this.apiKey } }
      },
      {
        description: 'api_key参数', 
        config: { params: { api_key: this.apiKey } }
      },
      {
        description: 'key参数',
        config: { params: { key: this.apiKey } }
      },
      {
        description: 'token参数',
        config: { params: { token: this.apiKey } }
      },
      
      // Header variations
      {
        description: 'CG-API-KEY头',
        config: { headers: { 'CG-API-KEY': this.apiKey } }
      },
      {
        description: 'X-API-KEY头',
        config: { headers: { 'X-API-KEY': this.apiKey } }
      },
      {
        description: 'Authorization Bearer头',
        config: { headers: { 'Authorization': `Bearer ${this.apiKey}` } }
      },
      {
        description: 'Authorization Token头',
        config: { headers: { 'Authorization': `Token ${this.apiKey}` } }
      },
      
      // Mixed approaches
      {
        description: '头和参数组合',
        config: { 
          headers: { 'CG-API-KEY': this.apiKey },
          params: { apikey: this.apiKey }
        }
      }
    ];

    let successCount = 0;
    
    for (const test of tests) {
      const success = await this.testFormat(test.description, test.config);
      if (success) {
        successCount++;
        console.log(`🎉 找到有效格式: ${test.description}\n`);
        break; // 找到有效格式就停止
      }
      console.log('');
      
      // 延迟避免限流
      await new Promise(resolve => setTimeout(resolve, 1500));
    }

    if (successCount === 0) {
      console.log('❌ 所有格式测试失败。可能的原因：');
      console.log('1. API密钥已过期或无效');
      console.log('2. 需要不同的认证方式');
      console.log('3. 需要账户升级或验证');
      console.log('4. API使用了不同的认证机制');
      
      // 最后尝试一个简单的无认证请求，看看错误信息
      console.log('\n🔍 测试无认证请求以获取错误信息:');
      try {
        const response = await axios.get(`${this.baseURL}/api/futures/coins-markets`, {
          timeout: 10000
        });
        console.log('无认证响应:', JSON.stringify(response.data, null, 2));
      } catch (error) {
        console.log('无认证错误:', error.response?.data || error.message);
      }
    }
  }
}

// 运行测试
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new APIFormatTester();
  await tester.runTests();
}

export default APIFormatTester;