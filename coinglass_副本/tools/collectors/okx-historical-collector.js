#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXHistoricalCollector {
  constructor() {
    // OKX交易对格式
    this.symbols = {
      'BTC': 'BTC-USDT',
      'ETH': 'ETH-USDT'
    };
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
      console.log(`📡 [${this.requestCount}] 请求OKX：${endpoint}`);
      console.log(`   参数：`, params);
      
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
      // OKX API返回格式检查
      if (response.data && response.data.code === '0') {
        this.successCount++;
        console.log(`✅ [${this.requestCount}] 成功 - 数据条数：${response.data.data?.length || 0}`);
        return response.data;
      } else {
        this.errorCount++;
        console.log(`⚠️  [${this.requestCount}] API返回错误：${response.data?.msg || '未知错误'}`);
        return null;
      }
    } catch (error) {
      this.errorCount++;
      console.log(`❌ [${this.requestCount}] 失败：${error.response?.status || 'N/A'} - ${error.message}`);
      
      if (error.response?.data) {
        console.log(`   错误详情：`, error.response.data);
      }
      
      // OKX限制：20次/2s
      if (error.response?.status === 429) {
        console.log('⏳ 触发频率限制，等待3秒...');
        await this.delay(3000);
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

  // 将时间戳转换为OKX需要的毫秒格式
  toOKXTimestamp(date) {
    return new Date(date).getTime();
  }

  async collectHistoricalCandles() {
    console.log('🚀 开始从OKX收集BTC和ETH历史K线数据...\n');
    console.log('📌 说明：OKX历史K线数据最多返回1440条记录\n');

    for (const [symbol, instId] of Object.entries(this.symbols)) {
      console.log(`\n📊 收集 ${symbol} (${instId}) 历史数据...`);
      
      // OKX历史K线数据是从最新往历史查询
      // 我们需要分段获取数据
      
      // 先测试最新数据
      console.log(`\n🔍 测试获取 ${symbol} 最新K线数据...`);
      const testData = await this.makeRequest('/api/v5/market/history-candles', {
        instId: instId,
        bar: '1D',  // 日线
        limit: '100'
      });
      
      if (!testData || !testData.data || testData.data.length === 0) {
        console.log(`❌ 无法获取 ${symbol} 数据，跳过`);
        continue;
      }
      
      console.log(`✅ 成功获取最新数据，第一条时间：${new Date(parseInt(testData.data[testData.data.length - 1][0])).toISOString()}`);
      console.log(`   最后一条时间：${new Date(parseInt(testData.data[0][0])).toISOString()}`);
      
      await this.saveData(`${symbol}/candles/latest.json`, testData);
      await this.delay(500); // OKX限制20次/2s，保守一点
      
      // 按月份往前获取历史数据
      let currentEndTime = new Date().getTime();
      const targetStartTime = new Date('2018-01-01').getTime();
      let allCandles = [];
      let monthCount = 0;
      
      console.log(`\n📅 开始按时间段收集 ${symbol} 历史数据...`);
      
      while (currentEndTime > targetStartTime && monthCount < 100) { // 限制最多100个月，避免无限循环
        monthCount++;
        
        // 使用after参数获取更早的数据
        const batchData = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: '1D',
          limit: '300',  // 每次获取300条（约10个月）
          after: currentEndTime.toString()
        });
        
        if (!batchData || !batchData.data || batchData.data.length === 0) {
          console.log(`⚠️  没有更多历史数据了`);
          break;
        }
        
        // OKX返回的数据是倒序的（最新的在前）
        const oldestTime = parseInt(batchData.data[batchData.data.length - 1][0]);
        const newestTime = parseInt(batchData.data[0][0]);
        
        console.log(`📊 获取了 ${batchData.data.length} 条数据`);
        console.log(`   时间范围：${new Date(oldestTime).toISOString().split('T')[0]} 到 ${new Date(newestTime).toISOString().split('T')[0]}`);
        
        // 将数据添加到总数组（注意去重）
        allCandles = [...allCandles, ...batchData.data];
        
        // 更新结束时间为本批次最早的时间
        currentEndTime = oldestTime;
        
        // 如果已经到达目标时间，停止
        if (oldestTime <= targetStartTime) {
          console.log(`✅ 已到达目标时间 2018-01-01`);
          break;
        }
        
        await this.delay(500);
      }
      
      // 按年份整理和保存数据
      console.log(`\n📁 整理 ${symbol} 数据并按年份保存...`);
      const candlesByYear = {};
      
      for (const candle of allCandles) {
        const timestamp = parseInt(candle[0]);
        const year = new Date(timestamp).getFullYear();
        
        if (!candlesByYear[year]) {
          candlesByYear[year] = [];
        }
        
        // 转换为更易读的格式
        candlesByYear[year].push({
          timestamp: timestamp,
          time: new Date(timestamp).toISOString(),
          open: candle[1],
          high: candle[2],
          low: candle[3],
          close: candle[4],
          volume: candle[5],
          volCcy: candle[6],  // 成交额
          volCcyQuote: candle[7], // 成交额（计价货币）
          confirm: candle[8]  // 是否完结
        });
      }
      
      // 保存各年份数据
      for (const [year, data] of Object.entries(candlesByYear)) {
        // 按时间正序排序
        data.sort((a, b) => a.timestamp - b.timestamp);
        
        await this.saveData(`${symbol}/candles/${year}.json`, {
          symbol: symbol,
          instId: instId,
          year: year,
          count: data.length,
          data: data
        });
        
        console.log(`💾 已保存 ${year} 年数据：${data.length} 条`);
      }
      
      // 保存完整数据
      allCandles.sort((a, b) => parseInt(a[0]) - parseInt(b[0])); // 按时间正序
      await this.saveData(`${symbol}/candles/all-historical.json`, {
        symbol: symbol,
        instId: instId,
        totalCount: allCandles.length,
        startTime: allCandles.length > 0 ? new Date(parseInt(allCandles[0][0])).toISOString() : null,
        endTime: allCandles.length > 0 ? new Date(parseInt(allCandles[allCandles.length - 1][0])).toISOString() : null,
        data: allCandles
      });
      
      console.log(`\n✅ ${symbol} 历史数据收集完成，共 ${allCandles.length} 条`);
    }

    console.log('\n📈 统计信息：');
    console.log(`总请求数：${this.requestCount}`);
    console.log(`成功：${this.successCount}`);
    console.log(`失败：${this.errorCount}`);
    console.log(`成功率：${((this.successCount / this.requestCount) * 100).toFixed(1)}%`);
  }

  async testOKXAPI() {
    console.log('🔍 测试OKX API连接...\n');
    
    // 测试获取交易产品信息
    const instrumentsData = await this.makeRequest('/api/v5/public/instruments', {
      instType: 'SPOT'
    });
    
    if (instrumentsData && instrumentsData.data) {
      console.log(`✅ API连接成功，获取到 ${instrumentsData.data.length} 个现货交易对`);
      
      // 查找BTC和ETH交易对
      const btcPairs = instrumentsData.data.filter(inst => inst.baseCcy === 'BTC');
      const ethPairs = instrumentsData.data.filter(inst => inst.baseCcy === 'ETH');
      
      console.log(`\nBTC交易对：${btcPairs.length} 个`);
      console.log(`ETH交易对：${ethPairs.length} 个`);
      
      // 显示USDT交易对
      const btcUsdt = btcPairs.find(inst => inst.instId === 'BTC-USDT');
      const ethUsdt = ethPairs.find(inst => inst.instId === 'ETH-USDT');
      
      if (btcUsdt) console.log(`\nBTC-USDT 信息：`, btcUsdt);
      if (ethUsdt) console.log(`\nETH-USDT 信息：`, ethUsdt);
    }
  }
}

// 主函数
async function main() {
  const collector = new OKXHistoricalCollector();
  
  try {
    // 先测试API
    console.log('=== 测试OKX API ===');
    await collector.testOKXAPI();
    
    console.log('\n=== 开始收集历史K线数据 ===');
    await collector.collectHistoricalCandles();
    
    console.log('\n🎉 OKX历史数据收集完成！');
    console.log('📁 数据保存在：historical/storage/raw/okx/');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
    console.error(error.stack);
  }
}

// 运行
main();