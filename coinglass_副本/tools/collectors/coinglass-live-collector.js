#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

class CoinGlassLiveCollector {
  constructor() {
    this.apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    this.baseURL = 'https://open-api-v4.coinglass.com';
    this.outputDir = './data/coinglass';
    this.requestDelay = 3000;
    this.stats = {
      startTime: Date.now(),
      success: 0,
      failed: 0,
      totalDataPoints: 0
    };
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      console.log(`📡 请求: ${endpoint}`);
      
      const response = await axios.get(url, {
        headers: {
          'Accept': 'application/json',
          'CG-API-KEY': this.apiKey
        },
        params: params,
        timeout: 30000
      });

      const data = response.data;
      
      if (data.code === "0" && data.data) {
        console.log(`✅ 成功获取数据`);
        return data;
      } else {
        console.log(`⚠️  API响应: ${data.msg || '未知错误'}`);
        return { error: data.msg, endpoint };
      }

    } catch (error) {
      console.log(`❌ 请求失败: ${error.response?.data?.msg || error.message}`);
      return { error: error.message, endpoint };
    }
  }

  async saveData(category, filename, data, metadata = {}) {
    const dir = path.join(this.outputDir, category);
    await fs.mkdir(dir, { recursive: true });
    
    const fileData = {
      ...metadata,
      collectedAt: new Date().toISOString(),
      endpoint: metadata.endpoint || 'unknown',
      dataCount: Array.isArray(data) ? data.length : (data.data ? (Array.isArray(data.data) ? data.data.length : 1) : 1),
      data: data
    };

    const filePath = path.join(dir, filename);
    await fs.writeFile(filePath, JSON.stringify(fileData, null, 2));
    
    console.log(`💾 数据已保存: ${filePath}`);
    return fileData.dataCount;
  }

  async tryFreeEndpoints() {
    console.log('🔍 尝试免费的API端点...\n');

    // 尝试一些可能免费的基础端点
    const freeEndpoints = [
      // 基础信息端点
      { 
        endpoint: '/futures/supported-coins', 
        category: 'futures', 
        filename: 'supported-coins-live.json',
        description: '支持的币种列表'
      },
      { 
        endpoint: '/futures/supported-exchange-pairs', 
        category: 'futures', 
        filename: 'exchange-pairs-live.json',
        description: '支持的交易所对'
      },
      
      // ETF相关（通常是公开数据）
      { 
        endpoint: '/api/etf/bitcoin/list', 
        category: 'etf', 
        filename: 'bitcoin-list-live.json',
        description: '比特币ETF列表'
      },
      
      // 尝试一些市场数据
      { 
        endpoint: '/api/futures/coins-markets', 
        category: 'futures', 
        filename: 'coins-markets-live.json',
        description: '币种市场数据'
      },
      
      // 特定币种的基础数据
      { 
        endpoint: '/api/futures/pairs-markets', 
        params: { symbol: 'BTC' },
        category: 'futures', 
        filename: 'btc-pairs-live.json',
        description: 'BTC交易对市场'
      },
      { 
        endpoint: '/api/futures/pairs-markets', 
        params: { symbol: 'ETH' },
        category: 'futures', 
        filename: 'eth-pairs-live.json',
        description: 'ETH交易对市场'
      }
    ];

    const results = [];
    
    for (const test of freeEndpoints) {
      console.log(`📊 尝试获取: ${test.description}`);
      
      const data = await this.makeRequest(test.endpoint, test.params || {});
      
      if (data && !data.error && data.data) {
        // 成功获取数据
        const count = await this.saveData(test.category, test.filename, data, {
          endpoint: test.endpoint,
          description: test.description,
          params: test.params
        });
        
        results.push({
          endpoint: test.endpoint,
          success: true,
          dataCount: count,
          description: test.description
        });
        
        this.stats.success++;
        this.stats.totalDataPoints += count;
        
        console.log(`✅ ${test.description}: ${count} 条数据\n`);
        
      } else {
        // 失败
        results.push({
          endpoint: test.endpoint,
          success: false,
          error: data?.error || 'Unknown error',
          description: test.description
        });
        
        this.stats.failed++;
        console.log(`❌ ${test.description}: ${data?.error || 'Failed'}\n`);
      }
      
      await this.delay(this.requestDelay);
    }
    
    return results;
  }

  async generateReport(results) {
    const successfulEndpoints = results.filter(r => r.success);
    const failedEndpoints = results.filter(r => !r.success);
    
    console.log('📊 收集结果汇总:');
    console.log(`- 成功端点: ${successfulEndpoints.length}`);
    console.log(`- 失败端点: ${failedEndpoints.length}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints}`);
    
    if (successfulEndpoints.length > 0) {
      console.log('\n✅ 成功的端点:');
      successfulEndpoints.forEach(ep => {
        console.log(`   ${ep.description}: ${ep.dataCount} 条数据`);
      });
    }
    
    if (failedEndpoints.length > 0) {
      console.log('\n❌ 失败的端点:');
      failedEndpoints.forEach(ep => {
        console.log(`   ${ep.description}: ${ep.error}`);
      });
    }

    // 保存收集报告
    const report = {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      results: results,
      summary: {
        successRate: results.length > 0 ? (successfulEndpoints.length / results.length * 100).toFixed(1) + '%' : '0%',
        availableEndpoints: successfulEndpoints.map(ep => ep.endpoint),
        restrictedEndpoints: failedEndpoints.map(ep => ({ endpoint: ep.endpoint, error: ep.error }))
      }
    };

    const reportPath = path.join(this.outputDir, 'live-collection-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 详细报告已保存: ${reportPath}`);
  }

  async run() {
    console.log('🚀 开始CoinGlass实时数据收集\n');
    console.log(`API密钥: ${this.apiKey.substring(0, 8)}...`);
    console.log(`基础URL: ${this.baseURL}\n`);
    
    const results = await this.tryFreeEndpoints();
    await this.generateReport(results);
    
    const elapsed = (Date.now() - this.stats.startTime) / 1000;
    console.log(`\n⏱️  总用时: ${elapsed.toFixed(1)} 秒`);
  }
}

// 运行收集器
if (import.meta.url === `file://${process.argv[1]}`) {
  const collector = new CoinGlassLiveCollector();
  await collector.run();
}

export default CoinGlassLiveCollector;