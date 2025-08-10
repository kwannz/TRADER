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

// Test with tokens that have longer history
const TEST_TOKENS = ['AAVE', 'COMP', 'GRT'];

class Test4HCollector {
  constructor() {
    this.specialPairs = {
      'STETH': 'STETH-ETH',
      'WBTC': 'WBTC-BTC',
      'PYUSD': 'PYUSD-USDT',
      'DAI': 'DAI-USDT',
      'JITOSOL': 'JITOSOL-SOL'
    };
    
    this.config = {
      timeframe: '4H',
      startDate: '2021-01-01',
      batchSize: 300,
      maxBatches: 3,  // Just 3 batches for testing
      requestDelay: 500
    };
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
      return stats.size > 1000;
    } catch {
      return false;
    }
  }

  async makeRequest(endpoint, params) {
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
      console.error(`Request failed: ${error.message}`);
      throw error;
    }
  }

  async collect4HData(symbol) {
    const instId = this.getInstId(symbol);
    console.log(`\n📊 Testing 4H collection for ${symbol} (${instId})`);
    
    if (await this.checkExisting4HData(symbol)) {
      console.log(`✅ ${symbol} already has 4H data, skipping...`);
      return { success: true, skipped: true };
    }
    
    try {
      const allCandles = [];
      let currentEnd = Date.now();
      
      for (let batch = 0; batch < this.config.maxBatches; batch++) {
        await this.delay(this.config.requestDelay);
        
        console.log(`  📡 Fetching batch ${batch + 1}/${this.config.maxBatches}...`);
        console.log(`     Request params: instId=${instId}, bar=${this.config.timeframe}, before=${currentEnd - 1}`);
        
        const candles = await this.makeRequest('/api/v5/market/history-candles', {
          instId,
          bar: this.config.timeframe,
          before: currentEnd - 1,
          limit: this.config.batchSize
        });
        
        console.log(`     Response: ${candles ? candles.length : 0} candles`);
        if (!candles || candles.length === 0) {
          console.log(`     ⚠️  No data returned for ${symbol}`);
          break;
        }
        
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
        
        console.log(`  ✅ Got ${candles.length} candles (total: ${allCandles.length})`);
      }
      
      allCandles.sort((a, b) => a.t - b.t);
      
      // Handle empty candles array
      if (allCandles.length === 0) {
        console.log(`  ⚠️  No data found for ${symbol}`);
        return { success: false, error: 'No data available' };
      }
      
      const outputData = {
        symbol,
        instId,
        timeframe: this.config.timeframe,
        timeframeName: '4小时',
        totalCount: allCandles.length,
        startTime: new Date(allCandles[0].t).toISOString(),
        endTime: new Date(allCandles[allCandles.length - 1].t).toISOString(),
        collectedAt: new Date().toISOString(),
        data: allCandles
      };
      
      const outputPath = path.join(OUTPUT_DIR, symbol, '4H');
      await fs.mkdir(outputPath, { recursive: true });
      await fs.writeFile(
        path.join(outputPath, 'data.json'),
        JSON.stringify(outputData, null, 2)
      );
      
      console.log(`✅ ${symbol}: Saved ${allCandles.length} candles to ${outputPath}/data.json`);
      return { success: true, count: allCandles.length };
      
    } catch (error) {
      console.error(`❌ ${symbol}: ${error.message}`);
      return { success: false, error: error.message };
    }
  }

  async runTest() {
    console.log('🧪 Testing 4H data collection');
    console.log(`📊 Test tokens: ${TEST_TOKENS.join(', ')}`);
    console.log(`⚙️  Config:`, this.config);
    
    const results = [];
    
    for (const symbol of TEST_TOKENS) {
      const result = await this.collect4HData(symbol);
      results.push({ symbol, ...result });
    }
    
    console.log('\n' + '='.repeat(40));
    console.log('📊 Test Results:');
    console.log('='.repeat(40));
    
    results.forEach(r => {
      if (r.success) {
        if (r.skipped) {
          console.log(`⏭️  ${r.symbol}: Skipped (already exists)`);
        } else {
          console.log(`✅ ${r.symbol}: Success (${r.count} candles)`);
        }
      } else {
        console.log(`❌ ${r.symbol}: Failed - ${r.error}`);
      }
    });
    
    console.log('\n✅ Test complete!');
  }
}

const tester = new Test4HCollector();
tester.runTest().catch(console.error);