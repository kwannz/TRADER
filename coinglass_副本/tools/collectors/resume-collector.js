#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = './historical/storage/raw/okx';

class ResumeCollector {
  constructor() {
    // 时间周期配置
    this.timeframes = {
      '1D': { name: '日线', startYear: 2018, priority: 1 },
      '4H': { name: '4小时', startYear: 2021, priority: 2 }
    };
    
    // 统计
    this.stats = {
      startTime: Date.now(),
      resumed: 0,
      completed: 0,
      failed: 0,
      totalDataPoints: 0
    };
    
    this.progress = {};
    this.symbols = [];
  }

  async loadProgress() {
    try {
      // 加载进度
      const progressData = await fs.readFile(path.join(OUTPUT_DIR, 'batch-progress.json'), 'utf8');
      this.progress = JSON.parse(progressData);
      
      // 加载代币列表
      const symbolsData = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      const allSymbols = JSON.parse(symbolsData);
      this.symbols = allSymbols.slice(0, 10); // 继续前10个
      
      console.log('✅ 加载进度成功');
      return true;
    } catch (error) {
      console.error('❌ 加载进度失败:', error.message);
      return false;
    }
  }

  async checkIncomplete() {
    const incomplete = [];
    
    for (const symbol of this.symbols) {
      for (const timeframe of Object.keys(this.timeframes)) {
        const key = `${symbol}_${timeframe}`;
        if (!this.progress[key] || !this.progress[key].completed) {
          incomplete.push({ symbol, timeframe, status: this.progress[key] });
        }
      }
    }
    
    return incomplete;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params = {}) {
    try {
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params: params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
      if (response.data && response.data.code === '0') {
        return response.data;
      } else {
        throw new Error(response.data?.msg || 'API错误');
      }
    } catch (error) {
      throw error;
    }
  }

  async collectSymbolTimeframe(symbol, timeframe) {
    const config = this.timeframes[timeframe];
    const instId = symbol === 'USDT' ? 'USDT-USDC' : 
                   symbol === 'USDC' ? 'USDC-USDT' : 
                   `${symbol}-USDT`;
    
    console.log(`\n📊 收集 ${symbol} ${config.name} (${instId})...`);
    
    const startDate = new Date(`${config.startYear}-01-01`);
    const endDate = new Date();
    
    let allCandles = [];
    let currentEndTime = endDate.getTime();
    const targetStartTime = startDate.getTime();
    let batchNum = 0;
    
    try {
      while (currentEndTime > targetStartTime && batchNum < 50) {
        batchNum++;
        
        const batchData = await this.makeRequest('/api/v5/market/history-candles', {
          instId: instId,
          bar: timeframe,
          limit: '300',
          after: currentEndTime.toString()
        });
        
        if (!batchData || !batchData.data || batchData.data.length === 0) {
          break;
        }
        
        process.stdout.write(`\r   批次 ${batchNum}: ${batchData.data.length} 条`);
        allCandles = [...allCandles, ...batchData.data];
        
        const oldestTime = parseInt(batchData.data[batchData.data.length - 1][0]);
        currentEndTime = oldestTime;
        
        if (oldestTime <= targetStartTime) break;
        
        await this.delay(500);
      }
      
      console.log(); // 换行
      
      // 处理数据
      const filteredCandles = allCandles
        .filter(candle => parseInt(candle[0]) >= targetStartTime)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
      
      if (filteredCandles.length > 0) {
        // 保存数据
        const formattedCandles = filteredCandles.map(candle => ({
          t: parseInt(candle[0]),
          o: candle[1],
          h: candle[2],
          l: candle[3],
          c: candle[4],
          v: candle[5]
        }));
        
        const metadata = {
          symbol,
          instId,
          timeframe,
          timeframeName: config.name,
          totalCount: formattedCandles.length,
          startTime: new Date(formattedCandles[0].t).toISOString(),
          endTime: new Date(formattedCandles[formattedCandles.length - 1].t).toISOString()
        };
        
        await fs.mkdir(path.join(OUTPUT_DIR, symbol, timeframe), { recursive: true });
        await fs.writeFile(
          path.join(OUTPUT_DIR, symbol, timeframe, 'data.json'),
          JSON.stringify({ ...metadata, data: formattedCandles }, null, 2)
        );
        
        // 更新进度
        this.progress[`${symbol}_${timeframe}`] = {
          completed: true,
          dataPoints: formattedCandles.length,
          timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(
          path.join(OUTPUT_DIR, 'batch-progress.json'),
          JSON.stringify(this.progress, null, 2)
        );
        
        console.log(`✅ 完成: ${formattedCandles.length} 条数据`);
        this.stats.completed++;
        this.stats.totalDataPoints += formattedCandles.length;
        
        return true;
      }
    } catch (error) {
      console.error(`❌ 失败: ${error.message}`);
      this.stats.failed++;
      
      // 更新失败状态
      this.progress[`${symbol}_${timeframe}`] = {
        completed: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
      
      await fs.writeFile(
        path.join(OUTPUT_DIR, 'batch-progress.json'),
        JSON.stringify(this.progress, null, 2)
      );
      
      return false;
    }
  }

  async run() {
    console.log('🔄 恢复收集任务\n');
    
    if (!await this.loadProgress()) {
      return;
    }
    
    // 检查未完成的任务
    const incomplete = await this.checkIncomplete();
    console.log(`📊 发现 ${incomplete.length} 个未完成任务\n`);
    
    if (incomplete.length === 0) {
      console.log('✅ 所有任务已完成！');
      return;
    }
    
    // 显示待处理任务
    console.log('待处理任务:');
    incomplete.forEach((task, i) => {
      console.log(`${i + 1}. ${task.symbol} - ${task.timeframe}`);
    });
    console.log();
    
    // 开始处理
    for (const task of incomplete) {
      this.stats.resumed++;
      await this.collectSymbolTimeframe(task.symbol, task.timeframe);
      await this.delay(2000);
    }
    
    // 最终统计
    const elapsed = (Date.now() - this.stats.startTime) / 1000 / 60;
    console.log('\n📊 恢复收集完成:');
    console.log(`- 恢复任务: ${this.stats.resumed}`);
    console.log(`- 成功完成: ${this.stats.completed}`);
    console.log(`- 失败: ${this.stats.failed}`);
    console.log(`- 新增数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`- 用时: ${elapsed.toFixed(1)} 分钟`);
  }
}

// 运行
async function main() {
  const collector = new ResumeCollector();
  await collector.run();
}

main();