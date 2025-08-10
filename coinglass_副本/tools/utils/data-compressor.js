#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';
import zlib from 'zlib';
import { promisify } from 'util';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

class DataCompressor {
  constructor() {
    this.compressionStrategies = {
      'compact': this.compactJSON.bind(this),
      'minified': this.minifiedJSON.bind(this),
      'gzip': this.gzipCompress.bind(this),
      'optimized': this.optimizedFormat.bind(this)
    };
    
    this.stats = {
      originalSize: 0,
      compressedSize: 0,
      filesProcessed: 0,
      compressionRatio: 0
    };
  }

  // Strategy 1: Compact JSON - Remove whitespace
  async compactJSON(data) {
    const compacted = JSON.stringify(data);
    return {
      format: 'compact',
      data: compacted,
      size: Buffer.byteLength(compacted)
    };
  }

  // Strategy 2: Minified JSON - Shorten keys
  async minifiedJSON(data) {
    const minified = this.minifyData(data);
    const compressed = JSON.stringify(minified);
    return {
      format: 'minified',
      data: compressed,
      size: Buffer.byteLength(compressed)
    };
  }

  // Strategy 3: GZIP compression
  async gzipCompress(data) {
    const jsonStr = JSON.stringify(data);
    const compressed = await gzip(jsonStr);
    return {
      format: 'gzip',
      data: compressed,
      size: compressed.length
    };
  }

  // Strategy 4: Optimized format - Array-based with metadata
  async optimizedFormat(data) {
    const optimized = this.convertToOptimized(data);
    const compressed = JSON.stringify(optimized);
    return {
      format: 'optimized',
      data: compressed,
      size: Buffer.byteLength(compressed)
    };
  }

  // Convert full format to minified
  minifyData(data) {
    return {
      s: data.symbol,
      i: data.instId,
      tf: data.timeframe,
      n: data.totalCount || data.count,
      st: data.startTime || data.start,
      et: data.endTime || data.end,
      ca: data.collectedAt || data.lastUpdated,
      d: data.data.map(candle => [
        candle.t,
        parseFloat(candle.o),
        parseFloat(candle.h),
        parseFloat(candle.l),
        parseFloat(candle.c),
        parseFloat(candle.v)
      ])
    };
  }

  // Convert to optimized array format
  convertToOptimized(data) {
    const candles = data.data;
    
    // Extract arrays for each field
    const timestamps = [];
    const opens = [];
    const highs = [];
    const lows = [];
    const closes = [];
    const volumes = [];
    
    candles.forEach(candle => {
      timestamps.push(candle.t);
      opens.push(parseFloat(candle.o));
      highs.push(parseFloat(candle.h));
      lows.push(parseFloat(candle.l));
      closes.push(parseFloat(candle.c));
      volumes.push(parseFloat(candle.v));
    });
    
    return {
      meta: {
        symbol: data.symbol,
        instId: data.instId,
        timeframe: data.timeframe,
        count: data.totalCount || data.count,
        start: data.startTime || data.start,
        end: data.endTime || data.end,
        collected: data.collectedAt || data.lastUpdated
      },
      data: {
        t: timestamps,
        o: opens,
        h: highs,
        l: lows,
        c: closes,
        v: volumes
      }
    };
  }

  // Decompress data based on format
  async decompress(compressedData, format) {
    switch (format) {
      case 'compact':
      case 'minified':
      case 'optimized':
        return JSON.parse(compressedData);
      
      case 'gzip':
        const decompressed = await gunzip(compressedData);
        return JSON.parse(decompressed.toString());
      
      default:
        throw new Error(`Unknown format: ${format}`);
    }
  }

  // Convert minified back to full format
  unminifyData(minified) {
    return {
      symbol: minified.s,
      instId: minified.i,
      timeframe: minified.tf,
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

  // Convert optimized format back to standard
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
      totalCount: meta.count,
      startTime: meta.start,
      endTime: meta.end,
      collectedAt: meta.collected,
      data: candles
    };
  }

  // Analyze compression effectiveness
  async analyzeCompression(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const data = JSON.parse(content);
      const originalSize = Buffer.byteLength(content);
      
      console.log(`\nAnalyzing: ${path.basename(filePath)}`);
      console.log(`Original size: ${this.formatSize(originalSize)}`);
      console.log('-'.repeat(50));
      
      const results = {};
      
      for (const [strategy, compressor] of Object.entries(this.compressionStrategies)) {
        const compressed = await compressor(data);
        const ratio = ((originalSize - compressed.size) / originalSize * 100).toFixed(1);
        
        results[strategy] = {
          size: compressed.size,
          ratio: ratio,
          format: compressed.format
        };
        
        console.log(`${strategy.padEnd(12)}: ${this.formatSize(compressed.size).padEnd(10)} (${ratio}% reduction)`);
      }
      
      return results;
    } catch (error) {
      console.error(`Error analyzing ${filePath}: ${error.message}`);
      return null;
    }
  }

  // Compress a single file
  async compressFile(filePath, strategy = 'optimized', outputPath = null) {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const data = JSON.parse(content);
      const originalSize = Buffer.byteLength(content);
      
      const compressor = this.compressionStrategies[strategy];
      if (!compressor) {
        throw new Error(`Unknown strategy: ${strategy}`);
      }
      
      const compressed = await compressor(data);
      
      // Determine output path
      if (!outputPath) {
        const dir = path.dirname(filePath);
        const base = path.basename(filePath, '.json');
        outputPath = path.join(dir, `${base}.${strategy}.json`);
        
        if (strategy === 'gzip') {
          outputPath += '.gz';
        }
      }
      
      // Save compressed data
      if (strategy === 'gzip') {
        await fs.writeFile(outputPath, compressed.data);
      } else {
        await fs.writeFile(outputPath, compressed.data);
      }
      
      // Update stats
      this.stats.originalSize += originalSize;
      this.stats.compressedSize += compressed.size;
      this.stats.filesProcessed++;
      
      const ratio = ((originalSize - compressed.size) / originalSize * 100).toFixed(1);
      console.log(`✅ Compressed ${path.basename(filePath)}: ${this.formatSize(originalSize)} → ${this.formatSize(compressed.size)} (${ratio}% reduction)`);
      
      return {
        original: originalSize,
        compressed: compressed.size,
        ratio: ratio,
        outputPath
      };
      
    } catch (error) {
      console.error(`Error compressing ${filePath}: ${error.message}`);
      return null;
    }
  }

  // Format bytes to human readable
  formatSize(bytes) {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  }

  // Generate compression report
  generateReport() {
    const totalRatio = ((this.stats.originalSize - this.stats.compressedSize) / this.stats.originalSize * 100).toFixed(1);
    
    return {
      summary: {
        filesProcessed: this.stats.filesProcessed,
        originalSize: this.formatSize(this.stats.originalSize),
        compressedSize: this.formatSize(this.stats.compressedSize),
        savedSpace: this.formatSize(this.stats.originalSize - this.stats.compressedSize),
        compressionRatio: `${totalRatio}%`
      },
      timestamp: new Date().toISOString()
    };
  }
}

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const compressor = new DataCompressor();
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node data-compressor.js <command> [options]');
    console.log('');
    console.log('Commands:');
    console.log('  analyze <file>     Analyze compression strategies for a file');
    console.log('  compress <file>    Compress a file (default: optimized)');
    console.log('  --strategy <name>  Compression strategy: compact, minified, gzip, optimized');
    console.log('');
    console.log('Examples:');
    console.log('  node data-compressor.js analyze data/okx/BTC/1D/data.json');
    console.log('  node data-compressor.js compress data/okx/BTC/1D/data.json --strategy optimized');
    process.exit(0);
  }
  
  const command = args[0];
  const filePath = args[1];
  
  if (command === 'analyze' && filePath) {
    compressor.analyzeCompression(filePath);
  } else if (command === 'compress' && filePath) {
    const strategyIndex = args.indexOf('--strategy');
    const strategy = strategyIndex > -1 ? args[strategyIndex + 1] : 'optimized';
    
    compressor.compressFile(filePath, strategy).then(result => {
      if (result) {
        console.log(`\nOutput saved to: ${result.outputPath}`);
      }
    });
  }
}

export default DataCompressor;