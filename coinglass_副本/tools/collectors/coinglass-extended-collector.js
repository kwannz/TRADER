#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

class CoinGlassExtendedCollector {
  constructor() {
    this.apiKey = '51e89d90bf31473384e7e6c61b75afe7';
    this.baseURL = 'https://open-api-v4.coinglass.com';
    this.outputDir = './data/coinglass';
    this.requestDelay = 3000;
    this.symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP'];
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
        return { success: true, data: data };
      } else {
        return { success: false, error: data.msg, data: data };
      }

    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.msg || error.message,
        statusCode: error.response?.status
      };
    }
  }

  async saveData(category, filename, data, metadata = {}) {
    const dir = path.join(this.outputDir, category);
    await fs.mkdir(dir, { recursive: true });
    
    const fileData = {
      ...metadata,
      collectedAt: new Date().toISOString(),
      dataCount: Array.isArray(data.data) ? data.data.length : 1,
      data: data
    };

    const filePath = path.join(dir, filename);
    await fs.writeFile(filePath, JSON.stringify(fileData, null, 2));
    
    return { filePath, dataCount: fileData.dataCount };
  }

  async collectMarketData() {
    console.log('📊 收集市场数据...\n');
    
    const results = [];
    
    // 收集各个主要币种的市场数据
    for (const symbol of this.symbols) {
      console.log(`📈 收集 ${symbol} 交易对数据...`);
      
      const result = await this.makeRequest('/api/futures/pairs-markets', { symbol });
      
      if (result.success) {
        const saved = await this.saveData(
          'futures', 
          `${symbol.toLowerCase()}-pairs-live.json`,
          result.data,
          {
            endpoint: '/api/futures/pairs-markets',
            symbol: symbol,
            description: `${symbol}交易对市场数据`
          }
        );
        
        console.log(`✅ ${symbol}: ${saved.dataCount} 条数据`);
        results.push({ symbol, success: true, dataCount: saved.dataCount });
        this.stats.success++;
        this.stats.totalDataPoints += saved.dataCount;
        
      } else {
        console.log(`❌ ${symbol}: ${result.error}`);
        results.push({ symbol, success: false, error: result.error });
        this.stats.failed++;
      }
      
      await this.delay(this.requestDelay);
    }
    
    return results;
  }

  async collectETFData() {
    console.log('\n💼 收集ETF数据...\n');
    
    const etfEndpoints = [
      {
        path: '/api/etf/bitcoin/list',
        filename: 'bitcoin-list-live.json',
        description: '比特币ETF列表'
      },
      {
        path: '/api/etf/bitcoin/net-assets/history',
        filename: 'bitcoin-net-assets-history.json',
        description: '比特币ETF净资产历史'
      },
      {
        path: '/api/etf/bitcoin/flow-history',
        filename: 'bitcoin-flow-history.json', 
        description: '比特币ETF流入流出历史'
      }
    ];

    const results = [];
    
    for (const endpoint of etfEndpoints) {
      console.log(`📊 收集: ${endpoint.description}`);
      
      const result = await this.makeRequest(endpoint.path);
      
      if (result.success) {
        const saved = await this.saveData(
          'etf',
          endpoint.filename,
          result.data,
          {
            endpoint: endpoint.path,
            description: endpoint.description
          }
        );
        
        console.log(`✅ ${endpoint.description}: ${saved.dataCount} 条数据`);
        results.push({ 
          endpoint: endpoint.path, 
          success: true, 
          dataCount: saved.dataCount 
        });
        this.stats.success++;
        this.stats.totalDataPoints += saved.dataCount;
        
      } else {
        console.log(`❌ ${endpoint.description}: ${result.error}`);
        results.push({ 
          endpoint: endpoint.path, 
          success: false, 
          error: result.error 
        });
        this.stats.failed++;
      }
      
      await this.delay(this.requestDelay);
    }
    
    return results;
  }

  async tryHistoricalData() {
    console.log('\n📚 尝试历史数据端点...\n');
    
    // 尝试一些历史数据端点
    const historyEndpoints = [
      {
        path: '/api/price/ohlc-history',
        params: { symbol: 'BTC', interval: '1d' },
        filename: 'btc-price-daily-history.json',
        description: 'BTC日线价格历史'
      },
      {
        path: '/api/futures/openInterest/ohlc-history',
        params: { symbol: 'BTC', interval: '1d' },
        filename: 'btc-oi-daily-history.json',
        description: 'BTC持仓量日线历史'
      },
      {
        path: '/api/futures/fundingRate/ohlc-history',
        params: { symbol: 'BTC', interval: '8h' },
        filename: 'btc-funding-rate-history.json',
        description: 'BTC资金费率历史'
      }
    ];

    const results = [];
    
    for (const endpoint of historyEndpoints) {
      console.log(`📊 尝试: ${endpoint.description}`);
      
      const result = await this.makeRequest(endpoint.path, endpoint.params);
      
      if (result.success) {
        const saved = await this.saveData(
          'futures',
          endpoint.filename,
          result.data,
          {
            endpoint: endpoint.path,
            params: endpoint.params,
            description: endpoint.description
          }
        );
        
        console.log(`✅ ${endpoint.description}: ${saved.dataCount} 条数据`);
        results.push({ 
          endpoint: endpoint.path, 
          success: true, 
          dataCount: saved.dataCount 
        });
        this.stats.success++;
        this.stats.totalDataPoints += saved.dataCount;
        
      } else {
        console.log(`❌ ${endpoint.description}: ${result.error}`);
        results.push({ 
          endpoint: endpoint.path, 
          success: false, 
          error: result.error 
        });
        this.stats.failed++;
      }
      
      await this.delay(this.requestDelay);
    }
    
    return results;
  }

  async generateComprehensiveReport(marketResults, etfResults, historyResults) {
    const allResults = [...marketResults, ...etfResults, ...historyResults];
    const successfulResults = allResults.filter(r => r.success);
    
    console.log('\n📋 综合收集报告:');
    console.log(`- 总请求数: ${allResults.length}`);
    console.log(`- 成功: ${successfulResults.length}`);
    console.log(`- 失败: ${allResults.length - successfulResults.length}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 成功率: ${(successfulResults.length / allResults.length * 100).toFixed(1)}%`);

    // 详细报告
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalRequests: allResults.length,
        successfulRequests: successfulResults.length,
        failedRequests: allResults.length - successfulResults.length,
        totalDataPoints: this.stats.totalDataPoints,
        successRate: (successfulResults.length / allResults.length * 100).toFixed(1) + '%'
      },
      categories: {
        market: {
          attempted: marketResults.length,
          successful: marketResults.filter(r => r.success).length,
          results: marketResults
        },
        etf: {
          attempted: etfResults.length,
          successful: etfResults.filter(r => r.success).length,
          results: etfResults
        },
        history: {
          attempted: historyResults.length,
          successful: historyResults.filter(r => r.success).length,
          results: historyResults
        }
      },
      availableEndpoints: successfulResults.map(r => ({
        endpoint: r.endpoint || 'market-data',
        dataCount: r.dataCount
      })),
      stats: this.stats
    };

    const reportPath = path.join(this.outputDir, 'comprehensive-collection-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 详细报告已保存: ${reportPath}`);
  }

  async run() {
    console.log('🚀 开始CoinGlass扩展数据收集\n');
    console.log(`🔑 API密钥: ${this.apiKey.substring(0, 8)}...`);
    console.log(`🌐 基础URL: ${this.baseURL}`);
    console.log(`💰 目标币种: ${this.symbols.join(', ')}\n`);
    
    // 执行各种数据收集
    const marketResults = await this.collectMarketData();
    const etfResults = await this.collectETFData();
    const historyResults = await this.tryHistoricalData();
    
    // 生成综合报告
    await this.generateComprehensiveReport(marketResults, etfResults, historyResults);
    
    const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    console.log(`\n⏱️  总用时: ${elapsed.toFixed(1)} 分钟`);
    console.log('🎉 扩展收集完成!');
  }
}

// 运行收集器
if (import.meta.url === `file://${process.argv[1]}`) {
  const collector = new CoinGlassExtendedCollector();
  await collector.run();
}

export default CoinGlassExtendedCollector;