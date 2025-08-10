#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';
const OUTPUT_DIR = './historical/storage/raw';

class SimpleBTCETHCollector {
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
      console.log(`❌ [${this.requestCount}] 失败：${error.message}`);
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

  async collectBasicData() {
    console.log('🚀 开始收集BTC和ETH基础数据...\n');

    // 1. 获取支持的币种
    console.log('📋 获取支持的币种...');
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

    // 2. 获取每个币种的市场数据
    for (const symbol of this.symbols) {
      console.log(`\n📊 收集 ${symbol} 数据...`);
      
      // 合约市场数据
      const contractMarkets = await this.makeRequest('/api/futures/coin-markets', { 
        coin: symbol.toLowerCase() 
      });
      if (contractMarkets) {
        await this.saveData(`${symbol}/contract-markets.json`, contractMarkets);
      }
      await this.delay(3000);

      // 现货市场数据
      const spotMarkets = await this.makeRequest('/api/spot/coin-markets', { 
        coin: symbol.toLowerCase() 
      });
      if (spotMarkets) {
        await this.saveData(`${symbol}/spot-markets.json`, spotMarkets);
      }
      await this.delay(3000);

      // 持仓量数据
      const openInterest = await this.makeRequest('/api/futures/coin-open-interest', { 
        coin: symbol.toLowerCase() 
      });
      if (openInterest) {
        await this.saveData(`${symbol}/open-interest.json`, openInterest);
      }
      await this.delay(3000);

      // 资金费率
      const fundingRates = await this.makeRequest('/api/futures/coin-funding-rates', { 
        coin: symbol.toLowerCase() 
      });
      if (fundingRates) {
        await this.saveData(`${symbol}/funding-rates.json`, fundingRates);
      }
      await this.delay(3000);
    }

    console.log('\n📈 统计信息：');
    console.log(`总请求数：${this.requestCount}`);
    console.log(`成功：${this.successCount}`);
    console.log(`失败：${this.errorCount}`);
    console.log(`成功率：${((this.successCount / this.requestCount) * 100).toFixed(1)}%`);
  }

  async collectHistoricalData() {
    console.log('\n🕒 开始收集历史数据...');
    
    // 时间范围：2018年至今，按年分段
    const startYear = 2018;
    const currentYear = new Date().getFullYear();
    
    for (const symbol of this.symbols) {
      console.log(`\n📅 收集 ${symbol} 历史数据...`);
      
      for (let year = startYear; year <= currentYear; year++) {
        const startTime = Math.floor(new Date(`${year}-01-01`).getTime() / 1000);
        const endTime = Math.floor(new Date(`${year}-12-31`).getTime() / 1000);
        
        console.log(`\n📆 ${symbol} ${year}年数据...`);
        
        // 持仓量历史
        const oiHistory = await this.makeRequest('/api/futures/open-interest-history', {
          coin: symbol.toLowerCase(),
          start_time: startTime,
          end_time: endTime
        });
        if (oiHistory) {
          await this.saveData(`${symbol}/history/open-interest-${year}.json`, oiHistory);
        }
        await this.delay(4000);
        
        // 资金费率历史
        const fundingHistory = await this.makeRequest('/api/futures/funding-rate-history', {
          coin: symbol.toLowerCase(),
          start_time: startTime,
          end_time: endTime
        });
        if (fundingHistory) {
          await this.saveData(`${symbol}/history/funding-rate-${year}.json`, fundingHistory);
        }
        await this.delay(4000);
      }
    }
  }
}

// 主函数
async function main() {
  const collector = new SimpleBTCETHCollector();
  
  try {
    // 先收集基础数据
    await collector.collectBasicData();
    
    // 然后收集历史数据
    await collector.collectHistoricalData();
    
    console.log('\n🎉 数据收集完成！');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
  }
}

// 运行
main();