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

class IncrementalUpdater {
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
    
    // Configuration
    this.config = {
      requestDelay: 500,      // Delay between requests
      batchSize: 300,         // OKX limit
      maxNewCandles: 5000,    // Safety limit for new data
      retryLimit: 3,
      retryDelay: 2000
    };
    
    // Supported timeframes
    this.timeframes = ['1D', '4H', '1H', '30m'];
    
    // Statistics
    this.stats = {
      totalUpdated: 0,
      totalNewCandles: 0,
      failedUpdates: [],
      skippedTokens: [],
      startTime: Date.now()
    };
  }

  getInstId(symbol) {
    return this.specialPairs[symbol] || `${symbol}-USDT`;
  }

  async delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async loadExistingData(symbol, timeframe) {
    try {
      const dataPath = path.join(DATA_DIR, symbol, timeframe, 'data.json');
      const content = await fs.readFile(dataPath, 'utf8');
      return JSON.parse(content);
    } catch (error) {
      return null;
    }
  }

  async saveData(symbol, timeframe, data) {
    try {
      const outputPath = path.join(DATA_DIR, symbol, timeframe);
      await fs.mkdir(outputPath, { recursive: true });
      
      // Save main data file
      await fs.writeFile(
        path.join(outputPath, 'data.json'),
        JSON.stringify(data, null, 2)
      );
      
      // Update metadata
      const metadata = {
        symbol: data.symbol,
        instId: data.instId,
        timeframe: data.timeframe,
        timeframeName: data.timeframeName,
        totalCount: data.totalCount || data.data.length,
        startTime: data.startTime || data.start,
        endTime: data.endTime || data.end,
        lastUpdated: new Date().toISOString()
      };
      
      await fs.writeFile(
        path.join(outputPath, 'metadata.json'),
        JSON.stringify(metadata, null, 2)
      );
      
      return true;
    } catch (error) {
      console.error(`Failed to save ${symbol}/${timeframe}: ${error.message}`);
      return false;
    }
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

  getTimeframeBar(timeframe) {
    // Map our timeframe names to OKX bar values
    const mapping = {
      '1D': '1D',
      '4H': '4H',
      '1H': '1H',
      '30m': '30m'
    };
    return mapping[timeframe] || timeframe;
  }

  getTimeframeName(timeframe) {
    const names = {
      '1D': '日线',
      '4H': '4小时',
      '1H': '1小时',
      '30m': '30分钟'
    };
    return names[timeframe] || timeframe;
  }

  async updateTimeframe(symbol, timeframe, existingData) {
    const instId = this.getInstId(symbol);
    const bar = this.getTimeframeBar(timeframe);
    
    // Get the latest timestamp from existing data
    const existingCandles = existingData.data || [];
    if (existingCandles.length === 0) {
      console.log(`  ⚠️  No existing data for ${symbol}/${timeframe}`);
      return { updated: false, newCandles: 0 };
    }
    
    // Find the latest timestamp
    const latestTimestamp = Math.max(...existingCandles.map(c => c.t));
    const latestDate = new Date(latestTimestamp);
    
    console.log(`  📅 Last candle: ${latestDate.toISOString()}`);
    
    // Fetch new candles since the last timestamp
    const newCandles = [];
    let hasMore = true;
    let after = latestTimestamp;
    
    while (hasMore && newCandles.length < this.config.maxNewCandles) {
      await this.delay(this.config.requestDelay);
      
      const candles = await this.makeRequest('/api/v5/market/history-candles', {
        instId,
        bar,
        after,
        limit: this.config.batchSize
      });
      
      if (!candles || candles.length === 0) {
        hasMore = false;
        break;
      }
      
      // Convert and filter candles
      const formattedCandles = candles
        .map(candle => ({
          t: parseInt(candle[0]),
          o: candle[1],
          h: candle[2],
          l: candle[3],
          c: candle[4],
          v: candle[5]
        }))
        .filter(c => c.t > latestTimestamp); // Only keep new candles
      
      if (formattedCandles.length === 0) {
        hasMore = false;
        break;
      }
      
      newCandles.push(...formattedCandles);
      after = Math.max(...formattedCandles.map(c => c.t));
    }
    
    if (newCandles.length === 0) {
      console.log(`  ✅ Already up to date`);
      return { updated: false, newCandles: 0 };
    }
    
    // Merge new candles with existing data
    const allCandles = [...existingCandles, ...newCandles];
    allCandles.sort((a, b) => a.t - b.t);
    
    // Remove any duplicates (just in case)
    const uniqueCandles = [];
    const seen = new Set();
    for (const candle of allCandles) {
      if (!seen.has(candle.t)) {
        seen.add(candle.t);
        uniqueCandles.push(candle);
      }
    }
    
    // Update the data structure
    const updatedData = {
      ...existingData,
      totalCount: uniqueCandles.length,
      count: uniqueCandles.length,
      endTime: new Date(Math.max(...uniqueCandles.map(c => c.t))).toISOString(),
      end: new Date(Math.max(...uniqueCandles.map(c => c.t))).toISOString(),
      lastUpdated: new Date().toISOString(),
      data: uniqueCandles
    };
    
    // Save updated data
    await this.saveData(symbol, timeframe, updatedData);
    
    console.log(`  ✅ Added ${newCandles.length} new candles (total: ${uniqueCandles.length})`);
    return { updated: true, newCandles: newCandles.length };
  }

  async updateSymbol(symbol) {
    console.log(`\n🔄 Updating ${symbol}...`);
    
    let symbolUpdated = false;
    let totalNewCandles = 0;
    
    for (const timeframe of this.timeframes) {
      // Check if this timeframe exists
      const existingData = await this.loadExistingData(symbol, timeframe);
      if (!existingData) {
        console.log(`  ⏭️  Skipping ${timeframe} - no existing data`);
        continue;
      }
      
      console.log(`  📊 Updating ${timeframe}...`);
      
      try {
        const result = await this.updateTimeframe(symbol, timeframe, existingData);
        if (result.updated) {
          symbolUpdated = true;
          totalNewCandles += result.newCandles;
          this.stats.totalNewCandles += result.newCandles;
        }
      } catch (error) {
        console.error(`  ❌ Failed to update ${timeframe}: ${error.message}`);
        this.stats.failedUpdates.push({ symbol, timeframe, error: error.message });
      }
    }
    
    if (symbolUpdated) {
      this.stats.totalUpdated++;
      console.log(`✅ ${symbol} updated with ${totalNewCandles} new candles`);
    } else {
      console.log(`✅ ${symbol} already up to date`);
    }
    
    return symbolUpdated;
  }

  async getAllSymbols() {
    try {
      const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
      const symbols = entries
        .filter(entry => entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'metadata')
        .map(entry => entry.name)
        .sort();
      
      return symbols;
    } catch (error) {
      console.error('Failed to read symbols:', error.message);
      return [];
    }
  }

  async updateAll(specificSymbols = null) {
    console.log('🚀 Starting Incremental Data Update');
    console.log('='.repeat(60));
    
    // Get symbols to update
    const allSymbols = await this.getAllSymbols();
    const symbolsToUpdate = specificSymbols || allSymbols;
    
    console.log(`📊 Found ${allSymbols.length} total symbols`);
    console.log(`🔄 Updating ${symbolsToUpdate.length} symbols`);
    console.log(`📅 Current time: ${new Date().toISOString()}`);
    console.log('='.repeat(60));
    
    // Update each symbol
    for (const symbol of symbolsToUpdate) {
      try {
        await this.updateSymbol(symbol);
      } catch (error) {
        console.error(`❌ Failed to update ${symbol}: ${error.message}`);
        this.stats.failedUpdates.push({ symbol, error: error.message });
      }
    }
    
    // Final report
    const duration = (Date.now() - this.stats.startTime) / 1000;
    
    console.log('\n' + '='.repeat(60));
    console.log('📊 Update Complete!');
    console.log('='.repeat(60));
    console.log(`✅ Symbols updated: ${this.stats.totalUpdated}`);
    console.log(`📈 New candles added: ${this.stats.totalNewCandles}`);
    console.log(`❌ Failed updates: ${this.stats.failedUpdates.length}`);
    console.log(`⏱️  Duration: ${duration.toFixed(1)}s`);
    
    if (this.stats.failedUpdates.length > 0) {
      console.log('\n❌ Failed updates:');
      this.stats.failedUpdates.forEach(f => {
        console.log(`  - ${f.symbol}${f.timeframe ? '/' + f.timeframe : ''}: ${f.error}`);
      });
    }
    
    // Save update report
    const reportPath = path.join(DATA_DIR, 'metadata', 'last-update-report.json');
    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.writeFile(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      duration: duration + 's',
      stats: this.stats,
      symbolsUpdated: symbolsToUpdate
    }, null, 2));
    
    console.log(`\n📄 Report saved to: ${reportPath}`);
  }
}

// CLI usage
const updater = new IncrementalUpdater();

// Check if specific symbols were provided
const args = process.argv.slice(2);
const specificSymbols = args.length > 0 ? args : null;

if (specificSymbols) {
  console.log(`Updating specific symbols: ${specificSymbols.join(', ')}`);
}

// Run the update
updater.updateAll(specificSymbols).catch(console.error);