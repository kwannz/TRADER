#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const API_BASE = 'https://open-api-v4.coinglass.com';
const API_KEY = '51e89d90bf31473384e7e6c61b75afe7';
const OUTPUT_DIR = './historical/storage/raw';

class HistoricalPriceCollector {
  constructor() {
    this.symbols = ['BTC', 'ETH'];
    this.requestCount = 0;
    this.successCount = 0;
    this.errorCount = 0;
    this.startYear = 2018;
    this.currentYear = new Date().getFullYear();
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    try {
      this.requestCount++;
      console.log(`📡 [${this.requestCount}] 请求：${endpoint}`);
      console.log(`   参数：`, params);
      
      const response = await axios.get(`${API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'CG-API-KEY': API_KEY,
          'accept': 'application/json'
        },
        timeout: 30000
      });
      
      this.successCount++;
      console.log(`✅ [${this.requestCount}] 成功 - 数据点数：${response.data?.data?.length || 0}`);
      return response.data;
    } catch (error) {
      this.errorCount++;
      console.log(`❌ [${this.requestCount}] 失败：${error.response?.status || 'N/A'} - ${error.message}`);
      
      if (error.response?.data) {
        console.log(`   错误详情：`, error.response.data);
      }
      
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

  async collectPriceHistory() {
    console.log('🚀 开始收集BTC和ETH 2018年至今的历史价格数据...\n');

    for (const symbol of this.symbols) {
      console.log(`\n📊 收集 ${symbol} 历史价格数据...`);
      
      // 首先测试端点是否可用
      console.log(`\n🔍 测试 ${symbol} 价格历史端点...`);
      const testStartTime = Math.floor(new Date('2024-01-01').getTime() / 1000);
      const testEndTime = Math.floor(new Date('2024-01-31').getTime() / 1000);
      
      const testData = await this.makeRequest('/api/futures/price/history', {
        symbol: symbol,
        interval: '1d',
        startTime: testStartTime,
        endTime: testEndTime
      });
      
      if (!testData) {
        console.log(`⚠️  ${symbol} 价格历史端点不可用，尝试其他参数...`);
        // 尝试不同的参数组合
        const alternativeTest = await this.makeRequest('/api/futures/price/history', {
          symbol: symbol,
          interval: '1d'
        });
        
        if (!alternativeTest) {
          console.log(`❌ ${symbol} 价格历史端点完全不可用，跳过此币种`);
          continue;
        }
      }
      
      await this.delay(3000);
      
      // 按年份收集数据
      for (let year = this.startYear; year <= this.currentYear; year++) {
        console.log(`\n📅 收集 ${symbol} ${year}年价格数据...`);
        
        // 按月份收集，避免单次请求数据量过大
        for (let month = 1; month <= 12; month++) {
          // 如果是当前年份且月份超过当前月份，停止
          if (year === this.currentYear && month > new Date().getMonth() + 1) {
            break;
          }
          
          const monthStr = month.toString().padStart(2, '0');
          const lastDay = new Date(year, month, 0).getDate();
          
          const startTime = Math.floor(new Date(`${year}-${monthStr}-01`).getTime() / 1000);
          const endTime = Math.floor(new Date(`${year}-${monthStr}-${lastDay}`).getTime() / 1000);
          
          console.log(`📆 ${year}年${month}月数据...`);
          
          // 尝试获取价格历史数据
          const priceData = await this.makeRequest('/api/futures/price/history', {
            symbol: symbol,
            interval: '1d',
            startTime: startTime,
            endTime: endTime
          });
          
          if (priceData && priceData.data && priceData.data.length > 0) {
            // 按月保存数据
            await this.saveData(
              `${symbol}/prices/${year}/${monthStr}.json`, 
              priceData
            );
          }
          
          // 延迟以避免速率限制
          await this.delay(4000);
        }
        
        console.log(`✅ ${symbol} ${year}年价格数据收集完成`);
      }
    }

    // 尝试收集更细粒度的数据（最近30天的小时线）
    console.log('\n📊 尝试收集最近30天的小时线数据...');
    
    const thirtyDaysAgo = Math.floor((Date.now() - 30 * 24 * 60 * 60 * 1000) / 1000);
    const now = Math.floor(Date.now() / 1000);
    
    for (const symbol of this.symbols) {
      const hourlyData = await this.makeRequest('/api/futures/price/history', {
        symbol: symbol,
        interval: '1h',
        startTime: thirtyDaysAgo,
        endTime: now
      });
      
      if (hourlyData && hourlyData.data && hourlyData.data.length > 0) {
        await this.saveData(`${symbol}/prices/recent-hourly.json`, hourlyData);
      }
      
      await this.delay(3000);
    }

    console.log('\n📈 统计信息：');
    console.log(`总请求数：${this.requestCount}`);
    console.log(`成功：${this.successCount}`);
    console.log(`失败：${this.errorCount}`);
    console.log(`成功率：${((this.successCount / this.requestCount) * 100).toFixed(1)}%`);
  }

  async testAlternativeEndpoints() {
    console.log('\n🔍 测试其他可能的价格端点...\n');
    
    const alternativeEndpoints = [
      '/api/futures/price/history',
      '/api/spot/price/history',
      '/api/futures/ohlc-history',
      '/api/spot/ohlc-history',
      '/api/futures/candle-history',
      '/api/spot/candle-history'
    ];
    
    for (const endpoint of alternativeEndpoints) {
      console.log(`测试端点：${endpoint}`);
      const result = await this.makeRequest(endpoint, {
        symbol: 'BTC',
        interval: '1d',
        limit: 10
      });
      
      if (result) {
        console.log(`✅ ${endpoint} - 可用`);
        if (result.data) {
          console.log(`   数据结构：`, Object.keys(result));
          if (Array.isArray(result.data) && result.data.length > 0) {
            console.log(`   示例数据：`, result.data[0]);
          }
        }
      } else {
        console.log(`❌ ${endpoint} - 不可用`);
      }
      
      await this.delay(2000);
    }
  }
}

// 主函数
async function main() {
  const collector = new HistoricalPriceCollector();
  
  try {
    // 先测试端点
    console.log('=== 测试价格历史端点 ===');
    await collector.testAlternativeEndpoints();
    
    console.log('\n=== 开始历史价格数据收集 ===');
    await collector.collectPriceHistory();
    
    console.log('\n🎉 历史价格数据收集完成！');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
    console.error(error.stack);
  }
}

// 运行
main();