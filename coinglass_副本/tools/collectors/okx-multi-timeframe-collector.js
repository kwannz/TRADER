#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXMultiTimeframeCollector {
  constructor() {
    // OKX交易对格式
    this.symbols = {
      'BTC': 'BTC-USDT',
      'ETH': 'ETH-USDT'
    };
    
    // OKX支持的时间周期
    // [1m/3m/5m/15m/30m/1H/2H/4H/6H/8H/12H/1D/3D/1W/1M/3M]
    this.timeframes = {
      '1m': { name: '1分钟', maxDays: 7 },       // 1分钟线最多查询7天
      '5m': { name: '5分钟', maxDays: 30 },      // 5分钟线最多查询30天
      '15m': { name: '15分钟', maxDays: 60 },    // 15分钟线最多查询60天
      '30m': { name: '30分钟', maxDays: 90 },    // 30分钟线最多查询90天
      '1H': { name: '1小时', maxDays: 180 },     // 1小时线最多查询180天
      '4H': { name: '4小时', maxDays: 730 },     // 4小时线最多查询2年
      '1D': { name: '日线', maxDays: 3650 },     // 日线最多查询10年
      '1W': { name: '周线', maxDays: 3650 },     // 周线
      '1M': { name: '月线', maxDays: 3650 }      // 月线
    };
    
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
      
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
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

  // 计算每个时间周期需要的请求次数
  calculateBatchCount(timeframe, startDate, endDate) {
    const msPerDay = 24 * 60 * 60 * 1000;
    const totalDays = Math.ceil((endDate - startDate) / msPerDay);
    const maxDaysPerRequest = this.timeframes[timeframe].maxDays;
    
    // OKX每次最多返回300条数据
    let candlesPerDay;
    switch(timeframe) {
      case '1m': candlesPerDay = 1440; break;  // 24*60
      case '5m': candlesPerDay = 288; break;   // 24*60/5
      case '15m': candlesPerDay = 96; break;  // 24*60/15
      case '30m': candlesPerDay = 48; break;  // 24*60/30
      case '1H': candlesPerDay = 24; break;   // 24
      case '4H': candlesPerDay = 6; break;    // 24/4
      case '1D': candlesPerDay = 1; break;
      case '1W': candlesPerDay = 1/7; break;
      case '1M': candlesPerDay = 1/30; break;
      default: candlesPerDay = 1;
    }
    
    const daysPerBatch = Math.min(Math.floor(300 / candlesPerDay), maxDaysPerRequest);
    const batchCount = Math.ceil(totalDays / daysPerBatch);
    
    return { batchCount, daysPerBatch, candlesPerDay };
  }

  async collectTimeframeData(symbol, instId, timeframe, startYear = 2018) {
    console.log(`\n📊 收集 ${symbol} ${this.timeframes[timeframe].name} 数据...`);
    
    const startDate = new Date(`${startYear}-01-01`);
    const endDate = new Date();
    const { batchCount, daysPerBatch } = this.calculateBatchCount(timeframe, startDate, endDate);
    
    console.log(`📅 时间范围：${startDate.toISOString().split('T')[0]} 至 ${endDate.toISOString().split('T')[0]}`);
    console.log(`📦 预计需要 ${batchCount} 批次请求，每批 ${daysPerBatch} 天`);
    
    let allCandles = [];
    let currentEndTime = endDate.getTime();
    const targetStartTime = startDate.getTime();
    let batchNum = 0;
    
    while (currentEndTime > targetStartTime && batchNum < batchCount + 10) { // 额外10次防止计算误差
      batchNum++;
      console.log(`\n批次 ${batchNum}/${batchCount}...`);
      
      const batchData = await this.makeRequest('/api/v5/market/history-candles', {
        instId: instId,
        bar: timeframe,
        limit: '300',
        after: currentEndTime.toString()
      });
      
      if (!batchData || !batchData.data || batchData.data.length === 0) {
        console.log(`⚠️  没有更多历史数据了`);
        break;
      }
      
      const oldestTime = parseInt(batchData.data[batchData.data.length - 1][0]);
      const newestTime = parseInt(batchData.data[0][0]);
      
      console.log(`   获取 ${batchData.data.length} 条数据`);
      console.log(`   时间：${new Date(oldestTime).toISOString()} 至 ${new Date(newestTime).toISOString()}`);
      
      // 合并数据
      allCandles = [...allCandles, ...batchData.data];
      
      // 更新时间
      currentEndTime = oldestTime;
      
      if (oldestTime <= targetStartTime) {
        console.log(`✅ 已到达目标时间`);
        break;
      }
      
      await this.delay(500);
    }
    
    // 过滤掉目标时间之前的数据
    const filteredCandles = allCandles.filter(candle => {
      return parseInt(candle[0]) >= targetStartTime;
    });
    
    // 按时间正序排序
    filteredCandles.sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    
    // 转换为易读格式
    const formattedCandles = filteredCandles.map(candle => ({
      timestamp: parseInt(candle[0]),
      time: new Date(parseInt(candle[0])).toISOString(),
      open: candle[1],
      high: candle[2],
      low: candle[3],
      close: candle[4],
      volume: candle[5],
      volCcy: candle[6],
      volCcyQuote: candle[7],
      confirm: candle[8]
    }));
    
    // 保存数据
    const saveData = {
      symbol: symbol,
      instId: instId,
      timeframe: timeframe,
      timeframeName: this.timeframes[timeframe].name,
      totalCount: formattedCandles.length,
      startTime: formattedCandles.length > 0 ? formattedCandles[0].time : null,
      endTime: formattedCandles.length > 0 ? formattedCandles[formattedCandles.length - 1].time : null,
      data: formattedCandles
    };
    
    await this.saveData(`${symbol}/${timeframe}/${startYear}-present.json`, saveData);
    
    // 如果数据量大，也按年份保存
    if (formattedCandles.length > 1000) {
      const candlesByYear = {};
      formattedCandles.forEach(candle => {
        const year = new Date(candle.timestamp).getFullYear();
        if (!candlesByYear[year]) {
          candlesByYear[year] = [];
        }
        candlesByYear[year].push(candle);
      });
      
      for (const [year, yearData] of Object.entries(candlesByYear)) {
        await this.saveData(`${symbol}/${timeframe}/${year}.json`, {
          symbol: symbol,
          instId: instId,
          timeframe: timeframe,
          year: year,
          count: yearData.length,
          data: yearData
        });
      }
    }
    
    console.log(`\n✅ ${symbol} ${this.timeframes[timeframe].name} 数据收集完成，共 ${formattedCandles.length} 条`);
    
    return formattedCandles.length;
  }

  async collectAllTimeframes(selectedTimeframes = null) {
    console.log('🚀 开始收集多时间周期历史数据...\n');
    
    const timeframesToCollect = selectedTimeframes || Object.keys(this.timeframes);
    const summary = {};
    
    for (const [symbol, instId] of Object.entries(this.symbols)) {
      summary[symbol] = {};
      
      for (const timeframe of timeframesToCollect) {
        if (!this.timeframes[timeframe]) {
          console.log(`⚠️  不支持的时间周期：${timeframe}`);
          continue;
        }
        
        // 根据时间周期决定起始年份
        let startYear;
        switch(timeframe) {
          case '1m':
          case '5m':
            startYear = 2024; // 分钟级数据只收集近期
            break;
          case '15m':
          case '30m':
          case '1H':
            startYear = 2023; // 小时级数据收集近2年
            break;
          default:
            startYear = 2018; // 其他周期收集更长历史
        }
        
        const count = await this.collectTimeframeData(symbol, instId, timeframe, startYear);
        summary[symbol][timeframe] = count;
        
        // 延迟避免频率限制
        await this.delay(1000);
      }
    }
    
    // 保存汇总信息
    await this.saveData('collection-summary.json', {
      collectionTime: new Date().toISOString(),
      totalRequests: this.requestCount,
      successfulRequests: this.successCount,
      failedRequests: this.errorCount,
      summary: summary
    });
    
    return summary;
  }
}

// 主函数
async function main() {
  const collector = new OKXMultiTimeframeCollector();
  
  try {
    console.log('=== OKX多时间周期数据收集器 ===\n');
    console.log('可用时间周期：');
    Object.entries(collector.timeframes).forEach(([key, value]) => {
      console.log(`  ${key}: ${value.name}`);
    });
    
    // 收集指定的时间周期
    // 您可以修改这个数组来选择需要的时间周期
    const selectedTimeframes = ['4H', '1D', '1W'];  // 4小时、日线、周线
    
    console.log(`\n将收集以下时间周期：${selectedTimeframes.join(', ')}\n`);
    
    const summary = await collector.collectAllTimeframes(selectedTimeframes);
    
    console.log('\n📈 收集完成统计：');
    console.log(`总请求数：${collector.requestCount}`);
    console.log(`成功：${collector.successCount}`);
    console.log(`失败：${collector.errorCount}`);
    console.log(`成功率：${((collector.successCount / collector.requestCount) * 100).toFixed(1)}%`);
    
    console.log('\n📊 数据汇总：');
    console.log(JSON.stringify(summary, null, 2));
    
    console.log('\n🎉 多时间周期历史数据收集完成！');
    console.log('📁 数据保存在：historical/storage/raw/okx/');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
    console.error(error.stack);
  }
}

// 运行
main();