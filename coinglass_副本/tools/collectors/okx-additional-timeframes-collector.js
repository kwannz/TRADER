#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class OKXAdditionalTimeframesCollector {
  constructor() {
    // OKX交易对格式
    this.symbols = {
      'BTC': 'BTC-USDT',
      'ETH': 'ETH-USDT'
    };
    
    // 额外的时间周期配置
    this.timeframes = {
      '1H': { name: '1小时', startYear: 2022, daysPerBatch: 180 },   // 近3年
      '30m': { name: '30分钟', startYear: 2023, daysPerBatch: 90 },  // 近2年
      '15m': { name: '15分钟', startYear: 2024, daysPerBatch: 60 },  // 近1年
      '1M': { name: '月线', startYear: 2018, daysPerBatch: 3650 }    // 全部历史
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
        console.log('⏳ 触发频率限制，等待5秒...');
        await this.delay(5000);
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

  async collectTimeframeData(symbol, instId, timeframe, config) {
    console.log(`\n📊 收集 ${symbol} ${config.name} 数据...`);
    
    const startDate = new Date(`${config.startYear}-01-01`);
    const endDate = new Date();
    
    console.log(`📅 时间范围：${startDate.toISOString().split('T')[0]} 至 ${endDate.toISOString().split('T')[0]}`);
    
    let allCandles = [];
    let currentEndTime = endDate.getTime();
    const targetStartTime = startDate.getTime();
    let batchNum = 0;
    const maxBatches = 200; // 防止无限循环
    
    while (currentEndTime > targetStartTime && batchNum < maxBatches) {
      batchNum++;
      
      // 显示进度
      const progress = ((endDate.getTime() - currentEndTime) / (endDate.getTime() - targetStartTime) * 100).toFixed(1);
      console.log(`\n批次 ${batchNum} - 进度: ${progress}%`);
      
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
      
      // 根据时间周期调整延迟
      const delayMs = timeframe === '15m' ? 1000 : 500;
      await this.delay(delayMs);
    }
    
    // 过滤并排序
    const filteredCandles = allCandles
      .filter(candle => parseInt(candle[0]) >= targetStartTime)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    
    // 转换格式
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
    
    // 保存完整数据
    await this.saveData(`${symbol}/${timeframe}/${config.startYear}-present.json`, {
      symbol: symbol,
      instId: instId,
      timeframe: timeframe,
      timeframeName: config.name,
      totalCount: formattedCandles.length,
      startTime: formattedCandles.length > 0 ? formattedCandles[0].time : null,
      endTime: formattedCandles.length > 0 ? formattedCandles[formattedCandles.length - 1].time : null,
      data: formattedCandles
    });
    
    // 按年份保存（如果数据量大）
    if (formattedCandles.length > 1000) {
      console.log('\n📁 按年份保存数据...');
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
          timeframeName: config.name,
          year: year,
          count: yearData.length,
          data: yearData
        });
        console.log(`💾 保存 ${year} 年数据：${yearData.length} 条`);
      }
    }
    
    console.log(`\n✅ ${symbol} ${config.name} 数据收集完成，共 ${formattedCandles.length} 条`);
    
    return formattedCandles.length;
  }

  async collectAll() {
    console.log('🚀 开始收集额外时间周期的历史数据...\n');
    
    const summary = {
      collectionTime: new Date().toISOString(),
      symbols: {}
    };
    
    for (const [symbol, instId] of Object.entries(this.symbols)) {
      summary.symbols[symbol] = {};
      
      for (const [timeframe, config] of Object.entries(this.timeframes)) {
        try {
          const count = await this.collectTimeframeData(symbol, instId, timeframe, config);
          summary.symbols[symbol][timeframe] = {
            count: count,
            timeframeName: config.name,
            startYear: config.startYear
          };
          
          // 延迟避免频率限制
          await this.delay(2000);
        } catch (error) {
          console.error(`❌ 收集 ${symbol} ${config.name} 数据时出错：`, error.message);
          summary.symbols[symbol][timeframe] = {
            error: error.message,
            timeframeName: config.name
          };
        }
      }
    }
    
    // 保存汇总信息
    await this.saveData('additional-timeframes-summary.json', {
      ...summary,
      totalRequests: this.requestCount,
      successfulRequests: this.successCount,
      failedRequests: this.errorCount,
      successRate: ((this.successCount / this.requestCount) * 100).toFixed(1) + '%'
    });
    
    return summary;
  }
}

// 主函数
async function main() {
  const collector = new OKXAdditionalTimeframesCollector();
  
  try {
    console.log('=== OKX额外时间周期数据收集器 ===\n');
    console.log('将收集以下时间周期：');
    Object.entries(collector.timeframes).forEach(([key, value]) => {
      console.log(`  ${key}: ${value.name} (从${value.startYear}年开始)`);
    });
    
    const summary = await collector.collectAll();
    
    console.log('\n📈 收集完成统计：');
    console.log(`总请求数：${collector.requestCount}`);
    console.log(`成功：${collector.successCount}`);
    console.log(`失败：${collector.errorCount}`);
    console.log(`成功率：${((collector.successCount / collector.requestCount) * 100).toFixed(1)}%`);
    
    console.log('\n📊 数据汇总：');
    for (const [symbol, data] of Object.entries(summary.symbols)) {
      console.log(`\n${symbol}:`);
      for (const [timeframe, info] of Object.entries(data)) {
        if (info.count) {
          console.log(`  ${info.timeframeName}: ${info.count} 条数据`);
        } else if (info.error) {
          console.log(`  ${info.timeframeName}: 失败 - ${info.error}`);
        }
      }
    }
    
    console.log('\n🎉 额外时间周期历史数据收集完成！');
    console.log('📁 数据保存在：historical/storage/raw/okx/');
  } catch (error) {
    console.error('💥 收集过程出错：', error.message);
    console.error(error.stack);
  }
}

// 运行
main();