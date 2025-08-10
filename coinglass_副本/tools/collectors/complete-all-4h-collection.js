#!/usr/bin/env node

import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OKX_API_BASE = 'https://www.okx.com';
const OUTPUT_DIR = path.join(__dirname, '../../data/okx');

class Complete4HCollector {
  constructor() {
    // All tokens that need 4H data
    this.tokensNeed4H = [
      'A', 'AAVE', 'APE', 'BONK', 'BTT', 'CFX', 'COMP', 'CORE', 'CRO', 
      'DAI', 'DYDX', 'EGLD', 'EIGEN', 'ENS', 'ETHFI', 'FET', 'FLOKI', 
      'FLOW', 'FLR', 'GALA', 'GRT', 'ICP', 'IMX', 'IOTA', 'IP', 'JITOSOL', 
      'JTO', 'JUP', 'KAIA', 'LDO', 'MANA', 'MORPHO', 'MOVE', 'NEO', 'NFT', 
      'ONDO', 'OP', 'PENDLE', 'PENGU', 'PEPE', 'POL', 'PYTH', 'PYUSD', 
      'RAY', 'RENDER', 'RSR', 'S', 'SAND', 'SHIB', 'STETH', 'STRK', 'SUI', 
      'TIA', 'TON', 'TRUMP', 'WBTC', 'WIF', 'WLD', 'XAUT'
    ];
    
    // Special trading pair mappings
    this.specialPairs = {
      'STETH': 'STETH-ETH',
      'WBTC': 'WBTC-BTC',
      'PYUSD': 'PYUSD-USDT',
      'DAI': 'DAI-USDT',
      'JITOSOL': 'JITOSOL-SOL'
    };
    
    // Configuration
    this.config = {
      timeframe: '4H',
      startDate: '2021-01-01',
      batchSize: 300,      // OKX limit per request
      maxBatches: 40,      // ~12,000 candles total
      concurrent: 5,       // Parallel requests
      requestDelay: 500,   // Delay between requests
      retryLimit: 3,
      retryDelay: 2000
    };
    
    // Progress tracking
    this.progress = {
      total: this.tokensNeed4H.length,
      completed: 0,
      failed: [],
      skipped: [],
      startTime: Date.now()
    };
    
    // Rate limiting
    this.requestQueue = [];
    this.activeRequests = 0;
  }

  getInstId(symbol) {
    return this.specialPairs[symbol] || `${symbol}-USDT`;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async checkExisting4HData(symbol) {
    try {
      const dataPath = path.join(OUTPUT_DIR, symbol, '4H', 'data.json');
      await fs.access(dataPath);
      const stats = await fs.stat(dataPath);
      
      // Check if file has substantial data (not just empty or error)
      if (stats.size > 1000) {
        const data = JSON.parse(await fs.readFile(dataPath, 'utf8'));
        if (data.totalCount > 100) {
          console.log(`✅ ${symbol} already has 4H data (${data.totalCount} candles)`);
          return true;
        }
      }
    } catch (error) {
      // File doesn't exist or is invalid
    }
    return false;
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
        console.log(`⏳ Rate limit hit, waiting ${this.config.retryDelay}ms...`);
        await this.delay(this.config.retryDelay * (retries + 1));
        return this.makeRequest(endpoint, params, retries + 1);
      }
      throw error;
    }
  }

  async getTokenStartDate(symbol) {
    // Try to get the start date from existing 1D data
    try {
      const dailyDataPath = path.join(OUTPUT_DIR, symbol, '1D', 'data.json');
      const dailyData = JSON.parse(await fs.readFile(dailyDataPath, 'utf8'));
      if (dailyData.start || dailyData.startTime) {
        return new Date(dailyData.start || dailyData.startTime).getTime();
      }
    } catch (error) {
      // Fallback to 2021 if no daily data found
    }
    return new Date(this.config.startDate).getTime();
  }

  async collect4HData(symbol) {
    const instId = this.getInstId(symbol);
    const startTime = await this.getTokenStartDate(symbol);
    const endTime = Date.now();
    
    console.log(`\n📊 Collecting 4H data for ${symbol} (${instId})`);
    
    const allCandles = [];
    let currentEnd = endTime;
    let batchCount = 0;
    
    try {
      while (batchCount < this.config.maxBatches && currentEnd > startTime) {
        await this.delay(this.config.requestDelay);
        
        const candles = await this.makeRequest('/api/v5/market/history-candles', {
          instId,
          bar: this.config.timeframe,
          before: currentEnd - 1,
          limit: this.config.batchSize
        });
        
        if (!candles || candles.length === 0) break;
        
        // OKX returns: [timestamp, open, high, low, close, volume, volCcy]
        const formattedCandles = candles.map(candle => ({
          t: parseInt(candle[0]),
          o: candle[1],
          h: candle[2],
          l: candle[3],
          c: candle[4],
          v: candle[5]
        }));
        
        allCandles.push(...formattedCandles);
        currentEnd = parseInt(candles[candles.length - 1][0]);
        batchCount++;
        
        console.log(`  📈 Batch ${batchCount}: ${candles.length} candles (total: ${allCandles.length})`);
        
        // Check if we've reached the start date
        if (currentEnd <= startTime) break;
      }
      
      // Sort by timestamp (oldest first)
      allCandles.sort((a, b) => a.t - b.t);
      
      // Save data
      const outputData = {
        symbol,
        instId,
        timeframe: this.config.timeframe,
        timeframeName: '4小时',
        totalCount: allCandles.length,
        startTime: new Date(allCandles[0]?.t || startTime).toISOString(),
        endTime: new Date(allCandles[allCandles.length - 1]?.t || endTime).toISOString(),
        collectedAt: new Date().toISOString(),
        data: allCandles
      };
      
      const outputPath = path.join(OUTPUT_DIR, symbol, '4H');
      await fs.mkdir(outputPath, { recursive: true });
      await fs.writeFile(
        path.join(outputPath, 'data.json'),
        JSON.stringify(outputData, null, 2)
      );
      
      console.log(`✅ ${symbol}: Saved ${allCandles.length} candles`);
      return { success: true, count: allCandles.length };
      
    } catch (error) {
      console.error(`❌ ${symbol}: ${error.message}`);
      return { success: false, error: error.message };
    }
  }

  async processQueue() {
    while (this.requestQueue.length > 0 && this.activeRequests < this.config.concurrent) {
      const task = this.requestQueue.shift();
      this.activeRequests++;
      
      task().finally(() => {
        this.activeRequests--;
        this.processQueue();
      });
    }
  }

  async collectAll() {
    console.log('🚀 Starting 4H data collection for missing tokens');
    console.log(`📊 Total tokens to process: ${this.tokensNeed4H.length}`);
    console.log(`⚙️  Configuration:`, this.config);
    
    for (const symbol of this.tokensNeed4H) {
      const task = async () => {
        // Check if already has data
        if (await this.checkExisting4HData(symbol)) {
          this.progress.skipped.push(symbol);
          this.progress.completed++;
          return;
        }
        
        // Collect data
        const result = await this.collect4HData(symbol);
        
        if (result.success) {
          this.progress.completed++;
        } else {
          this.progress.failed.push({ symbol, error: result.error });
        }
        
        // Progress report
        const elapsed = (Date.now() - this.progress.startTime) / 1000 / 60;
        const rate = this.progress.completed / elapsed;
        const remaining = (this.progress.total - this.progress.completed) / rate;
        
        console.log(`\n📊 Progress: ${this.progress.completed}/${this.progress.total} (${Math.round(this.progress.completed / this.progress.total * 100)}%)`);
        console.log(`⏱️  Elapsed: ${elapsed.toFixed(1)}min | Rate: ${rate.toFixed(1)}/min | ETA: ${remaining.toFixed(1)}min`);
      };
      
      this.requestQueue.push(task);
    }
    
    // Start processing queue
    await this.processQueue();
    
    // Wait for all tasks to complete
    while (this.activeRequests > 0 || this.requestQueue.length > 0) {
      await this.delay(1000);
    }
    
    // Final report
    console.log('\n' + '='.repeat(60));
    console.log('📊 Collection Complete!');
    console.log('='.repeat(60));
    console.log(`✅ Successfully collected: ${this.progress.completed - this.progress.skipped.length}`);
    console.log(`⏭️  Skipped (already exists): ${this.progress.skipped.length}`);
    console.log(`❌ Failed: ${this.progress.failed.length}`);
    console.log(`⏱️  Total time: ${((Date.now() - this.progress.startTime) / 1000 / 60).toFixed(1)} minutes`);
    
    if (this.progress.failed.length > 0) {
      console.log('\n❌ Failed tokens:');
      this.progress.failed.forEach(f => console.log(`  - ${f.symbol}: ${f.error}`));
    }
    
    // Save collection report
    const reportPath = path.join(OUTPUT_DIR, 'metadata', '4h-collection-report.json');
    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.writeFile(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      config: this.config,
      results: {
        total: this.progress.total,
        completed: this.progress.completed,
        newlyCollected: this.progress.completed - this.progress.skipped.length,
        skipped: this.progress.skipped,
        failed: this.progress.failed
      },
      duration: ((Date.now() - this.progress.startTime) / 1000).toFixed(2) + 's'
    }, null, 2));
    
    console.log(`\n📄 Report saved to: ${reportPath}`);
  }
}

// Run the collector
const collector = new Complete4HCollector();
collector.collectAll().catch(console.error);