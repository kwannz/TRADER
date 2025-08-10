#!/usr/bin/env node

import CoinGlassAPI from '../utils/coinglass-api.js';
import fs from 'fs/promises';
import path from 'path';

class CoinGlassCollector {
  constructor() {
    this.api = new CoinGlassAPI();
    this.outputDir = './data/coinglass';
    this.symbols = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOGE']; // 主要代币
    this.stats = {
      startTime: Date.now(),
      processed: 0,
      success: 0,
      failed: 0,
      totalDataPoints: 0
    };
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async ensureDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      // Directory already exists
    }
  }

  async saveData(filePath, data, metadata = {}) {
    await this.ensureDirectory(path.dirname(filePath));
    
    const fileData = {
      ...metadata,
      collectedAt: new Date().toISOString(),
      dataCount: Array.isArray(data) ? data.length : 1,
      data: data
    };

    await fs.writeFile(filePath, JSON.stringify(fileData, null, 2));
    return fileData.dataCount;
  }

  // 收集期货市场对数据  
  async collectFuturesPairsMarkets(symbol) {
    try {
      console.log(`📊 收集 ${symbol} 期货市场数据...`);
      
      const data = await this.api.makeRequest('/api/futures/pairs-markets', { symbol });
      
      if (data && data.data) {
        const filePath = path.join(this.outputDir, 'futures', `${symbol}-pairs-markets.json`);
        const count = await this.saveData(filePath, data.data, {
          symbol,
          type: 'futures-pairs-markets',
          endpoint: '/api/futures/pairs-markets'
        });
        
        console.log(`✅ ${symbol} 期货市场: ${count} 条数据`);
        this.stats.success++;
        this.stats.totalDataPoints += count;
        return count;
      } else {
        console.log(`❌ ${symbol} 期货市场: 无数据`);
        this.stats.failed++;
        return 0;
      }
    } catch (error) {
      console.log(`❌ ${symbol} 期货市场: ${error.message}`);
      this.stats.failed++;
      return 0;
    }
  }

  // 尝试收集其他可用数据
  async collectAvailableData() {
    console.log('🔍 探索可用的CoinGlass API端点...\n');

    // 测试各种端点组合
    const testEndpoints = [
      {
        path: '/api/futures/pairs-markets',
        params: { symbol: 'BTC' },
        output: 'futures/btc-pairs-markets.json'
      },
      {
        path: '/api/futures/pairs-markets', 
        params: { symbol: 'ETH' },
        output: 'futures/eth-pairs-markets.json'
      },
      // 可以添加更多端点测试
    ];

    for (const test of testEndpoints) {
      try {
        console.log(`📡 测试: ${test.path} ${JSON.stringify(test.params)}`);
        
        const data = await this.api.makeRequest(test.path, test.params);
        
        if (data && (data.data || data.success)) {
          const filePath = path.join(this.outputDir, test.output);
          const count = await this.saveData(filePath, data.data || data, {
            endpoint: test.path,
            params: test.params,
            type: 'exploratory-data'
          });
          
          console.log(`✅ 成功: ${count} 条数据保存到 ${test.output}`);
          this.stats.success++;
          this.stats.totalDataPoints += count;
        } else {
          console.log(`⚠️  无数据: ${test.path}`);
          this.stats.failed++;
        }
        
        await this.delay(2000);
        
      } catch (error) {
        console.log(`❌ 错误: ${test.path} - ${error.message}`);
        this.stats.failed++;
      }
    }
  }

  // 主收集流程
  async run() {
    console.log('🚀 开始CoinGlass数据收集\n');
    
    // 首先探索可用端点
    await this.collectAvailableData();
    
    console.log('\n📊 开始主要代币数据收集...\n');
    
    // 收集主要代币的期货市场数据
    for (const symbol of this.symbols) {
      this.stats.processed++;
      await this.collectFuturesPairsMarkets(symbol);
      await this.delay(2000); // API限流保护
    }

    // 最终统计
    const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    
    console.log('\n🎉 CoinGlass数据收集完成!');
    console.log(`📊 收集统计:`);
    console.log(`- 处理端点: ${this.stats.processed}`);
    console.log(`- 成功: ${this.stats.success}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 用时: ${elapsed.toFixed(1)} 分钟`);
    
    // 保存收集报告
    const reportPath = path.join(this.outputDir, 'collection-report.json');
    await this.saveData(reportPath, {
      stats: this.stats,
      symbols: this.symbols,
      timestamp: new Date().toISOString()
    }, {
      type: 'collection-report'
    });
    
    console.log(`📄 收集报告已保存: ${reportPath}`);
  }
}

// 运行收集器
if (import.meta.url === `file://${process.argv[1]}`) {
  const collector = new CoinGlassCollector();
  await collector.run();
}

export default CoinGlassCollector;