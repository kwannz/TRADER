#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';
import zlib from 'zlib';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const gunzip = promisify(zlib.gunzip);

class DataReader {
  constructor() {
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  // Detect data format
  detectFormat(data) {
    if (typeof data !== 'object') return 'unknown';
    
    // Check for optimized format
    if (data.meta && data.data && data.data.t && Array.isArray(data.data.t)) {
      return 'optimized';
    }
    
    // Check for minified format
    if (data.s && data.i && data.d && Array.isArray(data.d)) {
      return 'minified';
    }
    
    // Standard format
    if (data.symbol && data.data && Array.isArray(data.data)) {
      return 'standard';
    }
    
    return 'unknown';
  }

  // Convert any format to standard
  normalize(data) {
    const format = this.detectFormat(data);
    
    switch (format) {
      case 'standard':
        return data;
        
      case 'minified':
        return this.unminifyData(data);
        
      case 'optimized':
        return this.convertFromOptimized(data);
        
      default:
        throw new Error(`Unknown data format: ${format}`);
    }
  }

  // Convert minified back to standard
  unminifyData(minified) {
    return {
      symbol: minified.s,
      instId: minified.i,
      timeframe: minified.tf,
      timeframeName: this.getTimeframeName(minified.tf),
      totalCount: minified.n,
      startTime: minified.st,
      endTime: minified.et,
      collectedAt: minified.ca,
      data: minified.d.map(candle => ({
        t: candle[0],
        o: candle[1].toString(),
        h: candle[2].toString(),
        l: candle[3].toString(),
        c: candle[4].toString(),
        v: candle[5].toString()
      }))
    };
  }

  // Convert optimized format to standard
  convertFromOptimized(optimized) {
    const { meta, data } = optimized;
    const candles = [];
    
    for (let i = 0; i < data.t.length; i++) {
      candles.push({
        t: data.t[i],
        o: data.o[i].toString(),
        h: data.h[i].toString(),
        l: data.l[i].toString(),
        c: data.c[i].toString(),
        v: data.v[i].toString()
      });
    }
    
    return {
      symbol: meta.symbol,
      instId: meta.instId,
      timeframe: meta.timeframe,
      timeframeName: this.getTimeframeName(meta.timeframe),
      totalCount: meta.count,
      startTime: meta.start,
      endTime: meta.end,
      collectedAt: meta.collected,
      data: candles
    };
  }

  getTimeframeName(timeframe) {
    const names = {
      '1D': '日线',
      '4H': '4小时',
      '1H': '1小时',
      '30m': '30分钟',
      '15m': '15分钟',
      '5m': '5分钟'
    };
    return names[timeframe] || timeframe;
  }

  // Read data file with automatic format detection
  async readData(filePath, options = {}) {
    const { useCache = true, normalize = true } = options;
    
    // Check cache
    if (useCache && this.cache.has(filePath)) {
      const cached = this.cache.get(filePath);
      if (Date.now() - cached.timestamp < this.cacheTimeout) {
        return cached.data;
      }
    }
    
    try {
      let data;
      
      // Check if it's a gzip file
      if (filePath.endsWith('.gz')) {
        const compressed = await fs.readFile(filePath);
        const decompressed = await gunzip(compressed);
        data = JSON.parse(decompressed.toString());
      } else {
        const content = await fs.readFile(filePath, 'utf8');
        data = JSON.parse(content);
      }
      
      // Normalize if requested
      if (normalize) {
        data = this.normalize(data);
      }
      
      // Cache the result
      if (useCache) {
        this.cache.set(filePath, {
          data,
          timestamp: Date.now()
        });
      }
      
      return data;
      
    } catch (error) {
      throw new Error(`Failed to read ${filePath}: ${error.message}`);
    }
  }

  // Get specific candles by time range
  async getCandlesByTimeRange(filePath, startTime, endTime) {
    const data = await this.readData(filePath);
    
    const start = new Date(startTime).getTime();
    const end = new Date(endTime).getTime();
    
    return data.data.filter(candle => 
      candle.t >= start && candle.t <= end
    );
  }

  // Get latest N candles
  async getLatestCandles(filePath, count = 100) {
    const data = await this.readData(filePath);
    return data.data.slice(-count);
  }

  // Get data statistics
  async getDataStats(filePath) {
    const data = await this.readData(filePath);
    const candles = data.data;
    
    if (candles.length === 0) {
      return null;
    }
    
    // Calculate price statistics
    const prices = candles.map(c => parseFloat(c.c));
    const volumes = candles.map(c => parseFloat(c.v));
    
    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);
    const priceAvg = prices.reduce((a, b) => a + b, 0) / prices.length;
    
    const volumeMin = Math.min(...volumes);
    const volumeMax = Math.max(...volumes);
    const volumeAvg = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    const volumeTotal = volumes.reduce((a, b) => a + b, 0);
    
    return {
      symbol: data.symbol,
      timeframe: data.timeframe,
      candleCount: data.totalCount,
      dateRange: {
        start: data.startTime,
        end: data.endTime
      },
      priceStats: {
        min: priceMin.toFixed(2),
        max: priceMax.toFixed(2),
        avg: priceAvg.toFixed(2),
        current: prices[prices.length - 1].toFixed(2)
      },
      volumeStats: {
        min: volumeMin.toFixed(2),
        max: volumeMax.toFixed(2),
        avg: volumeAvg.toFixed(2),
        total: volumeTotal.toFixed(2)
      }
    };
  }

  // Clear cache
  clearCache() {
    this.cache.clear();
  }
}

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const reader = new DataReader();
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node data-reader.js <command> <file> [options]');
    console.log('');
    console.log('Commands:');
    console.log('  read <file>        Read and display data info');
    console.log('  stats <file>       Show data statistics');
    console.log('  latest <file> [n]  Show latest n candles (default: 10)');
    console.log('');
    console.log('Examples:');
    console.log('  node data-reader.js read data/okx/BTC/1D/data.json');
    console.log('  node data-reader.js stats data/okx/BTC/1D/data.json');
    console.log('  node data-reader.js latest data/okx/BTC/1D/data.json 5');
    process.exit(0);
  }
  
  const command = args[0];
  const filePath = args[1];
  
  if (!filePath) {
    console.error('Error: File path required');
    process.exit(1);
  }
  
  (async () => {
    try {
      if (command === 'read') {
        const data = await reader.readData(filePath);
        console.log('📊 Data Info:');
        console.log(`Symbol: ${data.symbol}`);
        console.log(`Timeframe: ${data.timeframe} (${data.timeframeName})`);
        console.log(`Candles: ${data.totalCount}`);
        console.log(`Period: ${data.startTime} to ${data.endTime}`);
        console.log(`Format: ${reader.detectFormat(data)}`);
        
      } else if (command === 'stats') {
        const stats = await reader.getDataStats(filePath);
        console.log('📊 Data Statistics:');
        console.log(JSON.stringify(stats, null, 2));
        
      } else if (command === 'latest') {
        const count = parseInt(args[2]) || 10;
        const candles = await reader.getLatestCandles(filePath, count);
        
        console.log(`📊 Latest ${count} candles:`);
        candles.forEach(candle => {
          const date = new Date(candle.t).toISOString();
          console.log(`${date}: O=${candle.o} H=${candle.h} L=${candle.l} C=${candle.c} V=${candle.v}`);
        });
      }
    } catch (error) {
      console.error(`Error: ${error.message}`);
      process.exit(1);
    }
  })();
}

export default DataReader;