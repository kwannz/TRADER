#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OKX_API_BASE = 'https://www.okx.com';
const DATA_DIR = path.join(__dirname, '../../data/okx');

class MultiTimeframeCollector {
  constructor() {
    // Special trading pair mappings
    this.specialPairs = {
      'STETH': 'STETH-ETH',
      'WBTC': 'WBTC-BTC',
      'PYUSD': 'PYUSD-USDT',
      'DAI': 'DAI-USDT',
      'JITOSOL': 'JITOSOL-SOL',
      'USDC': 'USDC-USDT',
      'USDS': 'USDS-USDT',
      'USDE': 'USDE-USDT'
    };
    
    // Timeframe configurations
    this.timeframeConfigs = {
      '1D': {
        name: '日线',
        bar: '1D',
        maxDays: 3650,    // 10 years
        candlesPerDay: 1
      },
      '4H': {
        name: '4小时',
        bar: '4H',
        maxDays: 730,     // 2 years
        candlesPerDay: 6
      },
      '1H': {
        name: '1小时',
        bar: '1H',
        maxDays: 180,     // 6 months
        candlesPerDay: 24
      },
      '30m': {
        name: '30分钟',
        bar: '30m',
        maxDays: 90,      // 3 months
        candlesPerDay: 48
      }
    };
    
    // Collection configuration
    this.config = {
      concurrent: 3,       // Parallel requests per symbol
      requestDelay: 500,   // Delay between requests
      batchSize: 300,      // OKX limit
      retryLimit: 3,
      retryDelay: 2000
    };
    
    // Progress tracking
    this.stats = {
      startTime: Date.now(),
      totalCollected: 0,
      failedCollections: [],
      totalDataPoints: 0
    };
  }

  getInstId(symbol) {
    return this.specialPairs[symbol] || `${symbol}-USDT`;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async makeRequest(endpoint, params, retries = 0) {
    try {
      const response = await axios.get(`${OKX_API_BASE}${endpoint}`, {
        params,
        headers: {
          'accept': 'application/json',
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });
      
      if (response.data && response.data.code === '0') {
        return response.data.data;
      } else {
        throw new Error(response.data?.msg || 'API error');
      }
    } catch (error) {
      if (error.response?.status === 429 && retries < this.config.retryLimit) {
        await this.delay(this.config.retryDelay * (retries + 1));
        return this.makeRequest(endpoint, params, retries + 1);
      }
      throw error;
    }
  }

  async getTokenStartDate(symbol, timeframe) {
    // Calculate start date based on timeframe limits and token listing
    const config = this.timeframeConfigs[timeframe];
    const maxStartDate = new Date();
    maxStartDate.setDate(maxStartDate.getDate() - config.maxDays);
    
    // Try to get actual listing date from 1D data
    try {
      const dailyDataPath = path.join(DATA_DIR, symbol, '1D', 'data.json');
      const dailyData = JSON.parse(await fs.readFile(dailyDataPath, 'utf8'));
      if (dailyData.start || dailyData.startTime) {
        const listingDate = new Date(dailyData.start || dailyData.startTime);
        // Return the later of the two dates (more recent)
        const selectedDate = listingDate > maxStartDate ? listingDate : maxStartDate;
        return selectedDate;
      }
    } catch (error) {
      // No daily data found
    }
    
    return maxStartDate;
  }

  async checkExistingData(symbol, timeframe) {
    try {
      const dataPath = path.join(DATA_DIR, symbol, timeframe, 'data.json');
      await fs.access(dataPath);
      const stats = await fs.stat(dataPath);
      
      if (stats.size > 1000) {
        const data = JSON.parse(await fs.readFile(dataPath, 'utf8'));
        if (data.totalCount > 100) {
          return { exists: true, count: data.totalCount };
        }
      }
    } catch (error) {
      // File doesn't exist or is invalid
    }
    return { exists: false, count: 0 };
  }

  async collectTimeframeData(symbol, timeframe) {
    const instId = this.getInstId(symbol);
    const config = this.timeframeConfigs[timeframe];
    const startDate = await this.getTokenStartDate(symbol, timeframe);
    const endTime = Date.now();
    
    // Ensure we don't try to fetch future data
    if (startDate.getTime() > endTime) {
      console.log(`     ⚠️  Start date ${startDate.toISOString()} is in the future`);
      return { success: false, error: 'Start date is in the future' };
    }
    
    console.log(`  📊 Collecting ${timeframe} data (${config.name})`);
    
    const allCandles = [];
    let currentEnd = endTime;
    let batchCount = 0;
    const maxBatches = Math.ceil(config.maxDays * config.candlesPerDay / this.config.batchSize);
    
    try {
      while (batchCount < maxBatches && currentEnd > startDate.getTime()) {
        await this.delay(this.config.requestDelay);
        
        const params = {
          instId,
          bar: config.bar,
          before: currentEnd - 1,
          limit: this.config.batchSize
        };
        
        // For first request, remove the before parameter to get most recent data
        if (batchCount === 0 && currentEnd === endTime) {
          delete params.before;
        }
        
        const candles = await this.makeRequest('/api/v5/market/history-candles', params);
        if (!candles || candles.length === 0) break;
        
        // Format candles
        const formattedCandles = candles.map(candle => ({
          t: parseInt(candle[0]),
          o: candle[1],
          h: candle[2],
          l: candle[3],
          c: candle[4],
          v: candle[5]
        }));
        
        // Filter out candles before start date
        const validCandles = formattedCandles.filter(c => c.t >= startDate.getTime());
        allCandles.push(...validCandles);
        
        // Update currentEnd to the oldest timestamp in this batch
        const oldestTimestamp = Math.min(...candles.map(c => parseInt(c[0])));
        currentEnd = oldestTimestamp;
        batchCount++;
        
        if (batchCount % 5 === 0) {
          console.log(`     Progress: ${allCandles.length} candles collected`);
        }
        
        // Stop if we've reached the start date
        if (currentEnd <= startDate.getTime() || validCandles.length < formattedCandles.length) {
          break;
        }
      }
      
      // Sort by timestamp
      allCandles.sort((a, b) => a.t - b.t);
      
      if (allCandles.length === 0) {
        console.log(`     ⚠️  No data available for ${symbol}/${timeframe}`);
        return { success: false, error: 'No data available' };
      }
      
      // Prepare output data
      const outputData = {
        symbol,
        instId,
        timeframe,
        timeframeName: config.name,
        totalCount: allCandles.length,
        startTime: new Date(allCandles[0].t).toISOString(),
        endTime: new Date(allCandles[allCandles.length - 1].t).toISOString(),
        collectedAt: new Date().toISOString(),
        data: allCandles
      };
      
      // Save data
      const outputPath = path.join(DATA_DIR, symbol, timeframe);
      await fs.mkdir(outputPath, { recursive: true });
      
      await fs.writeFile(
        path.join(outputPath, 'data.json'),
        JSON.stringify(outputData, null, 2)
      );
      
      // Save metadata
      const metadata = {
        symbol: outputData.symbol,
        instId: outputData.instId,
        timeframe: outputData.timeframe,
        timeframeName: outputData.timeframeName,
        totalCount: outputData.totalCount,
        startTime: outputData.startTime,
        endTime: outputData.endTime,
        collectedAt: outputData.collectedAt
      };
      
      await fs.writeFile(
        path.join(outputPath, 'metadata.json'),
        JSON.stringify(metadata, null, 2)
      );
      
      console.log(`     ✅ Saved ${allCandles.length} candles`);
      this.stats.totalDataPoints += allCandles.length;
      
      return { success: true, count: allCandles.length };
      
    } catch (error) {
      console.error(`     ❌ Failed: ${error.message}`);
      return { success: false, error: error.message };
    }
  }

  async collectSymbol(symbol, timeframes) {
    console.log(`\n🪙 Collecting ${symbol}...`);
    
    const results = {};
    
    for (const timeframe of timeframes) {
      // Check if already exists
      const existing = await this.checkExistingData(symbol, timeframe);
      if (existing.exists) {
        console.log(`  ⏭️  Skipping ${timeframe} - already has ${existing.count} candles`);
        results[timeframe] = { skipped: true, count: existing.count };
        continue;
      }
      
      // Collect data
      const result = await this.collectTimeframeData(symbol, timeframe);
      results[timeframe] = result;
      
      if (result.success) {
        this.stats.totalCollected++;
      } else {
        this.stats.failedCollections.push({ symbol, timeframe, error: result.error });
      }
    }
    
    return results;
  }

  async collectMultiple(symbols, timeframes) {
    console.log('🚀 Multi-Timeframe Collection');
    console.log('='.repeat(60));
    console.log(`📊 Symbols: ${symbols.join(', ')}`);
    console.log(`⏱️  Timeframes: ${timeframes.join(', ')}`);
    console.log(`⚙️  Config:`, this.config);
    console.log('='.repeat(60));
    
    const allResults = {};
    
    for (const symbol of symbols) {
      const results = await this.collectSymbol(symbol, timeframes);
      allResults[symbol] = results;
    }
    
    // Final report
    const duration = (Date.now() - this.stats.startTime) / 1000;
    
    console.log('\n' + '='.repeat(60));
    console.log('📊 Collection Complete!');
    console.log('='.repeat(60));
    console.log(`✅ Successful collections: ${this.stats.totalCollected}`);
    console.log(`📈 Total data points: ${this.stats.totalDataPoints.toLocaleString()}`);
    console.log(`❌ Failed collections: ${this.stats.failedCollections.length}`);
    console.log(`⏱️  Duration: ${duration.toFixed(1)}s`);
    
    if (this.stats.failedCollections.length > 0) {
      console.log('\n❌ Failed collections:');
      this.stats.failedCollections.forEach(f => {
        console.log(`  - ${f.symbol}/${f.timeframe}: ${f.error}`);
      });
    }
    
    // Save collection report
    const reportPath = path.join(DATA_DIR, 'metadata', `multi-timeframe-report-${Date.now()}.json`);
    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.writeFile(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      duration: duration + 's',
      symbols,
      timeframes,
      stats: this.stats,
      results: allResults
    }, null, 2));
    
    console.log(`\n📄 Report saved to: ${reportPath}`);
  }
}

// CLI usage
const collector = new MultiTimeframeCollector();

// Parse command line arguments
const args = process.argv.slice(2);
let symbols = [];
let timeframes = ['1H', '30m']; // Default to new timeframes

// Parse arguments
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--symbols' || args[i] === '-s') {
    symbols = args[++i].split(',');
  } else if (args[i] === '--timeframes' || args[i] === '-t') {
    timeframes = args[++i].split(',');
  } else if (!args[i].startsWith('-')) {
    symbols.push(args[i]);
  }
}

// Show usage if no symbols provided
if (symbols.length === 0) {
  console.log('Usage: node multi-timeframe-collector.js [symbols...] [options]');
  console.log('');
  console.log('Options:');
  console.log('  --symbols, -s     Comma-separated list of symbols');
  console.log('  --timeframes, -t  Comma-separated list of timeframes (default: 1H,30m)');
  console.log('');
  console.log('Examples:');
  console.log('  node multi-timeframe-collector.js BTC ETH');
  console.log('  node multi-timeframe-collector.js --symbols BTC,ETH,SOL --timeframes 1H,30m');
  console.log('  node multi-timeframe-collector.js BTC --timeframes 1D,4H,1H,30m');
  process.exit(0);
}

// Run collection
collector.collectMultiple(symbols, timeframes).catch(console.error);