#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import DataCompressor from './data-compressor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATA_DIR = path.join(__dirname, '../../data/okx');

class BatchCompressor {
  constructor() {
    this.compressor = new DataCompressor();
    this.stats = {
      totalFiles: 0,
      processedFiles: 0,
      failedFiles: 0,
      originalSize: 0,
      compressedSize: 0,
      startTime: Date.now()
    };
  }

  async findDataFiles(dir) {
    const files = [];
    
    async function walk(currentDir) {
      const entries = await fs.readdir(currentDir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(currentDir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          await walk(fullPath);
        } else if (entry.isFile() && entry.name === 'data.json') {
          files.push(fullPath);
        }
      }
    }
    
    await walk(dir);
    return files;
  }

  async compressDirectory(dir, strategy = 'optimized', createBackup = true) {
    console.log('🚀 Batch Compression Tool');
    console.log('='.repeat(60));
    console.log(`📁 Directory: ${dir}`);
    console.log(`📊 Strategy: ${strategy}`);
    console.log(`💾 Backup: ${createBackup ? 'Yes' : 'No'}`);
    console.log('='.repeat(60));
    
    // Find all data.json files
    const files = await this.findDataFiles(dir);
    this.stats.totalFiles = files.length;
    
    console.log(`\n📊 Found ${files.length} data files to compress\n`);
    
    // Process each file
    for (const file of files) {
      try {
        // Get relative path for display
        const relativePath = path.relative(DATA_DIR, file);
        
        // Read original file
        const originalContent = await fs.readFile(file, 'utf8');
        const originalSize = Buffer.byteLength(originalContent);
        this.stats.originalSize += originalSize;
        
        // Create backup if requested
        if (createBackup) {
          const backupPath = file.replace('.json', '.backup.json');
          await fs.writeFile(backupPath, originalContent);
        }
        
        // Compress file
        const result = await this.compressor.compressFile(file, strategy, file);
        
        if (result) {
          this.stats.processedFiles++;
          this.stats.compressedSize += result.compressed;
          console.log(`✅ ${relativePath}: ${this.formatSize(originalSize)} → ${this.formatSize(result.compressed)} (${result.ratio}% reduction)`);
        } else {
          this.stats.failedFiles++;
          console.log(`❌ ${relativePath}: Compression failed`);
        }
        
      } catch (error) {
        this.stats.failedFiles++;
        console.error(`❌ ${path.relative(DATA_DIR, file)}: ${error.message}`);
      }
    }
    
    // Generate final report
    this.generateReport();
  }

  async restoreFromBackup(dir) {
    console.log('🔄 Restoring from Backup');
    console.log('='.repeat(60));
    
    const backupFiles = [];
    
    async function findBackups(currentDir) {
      const entries = await fs.readdir(currentDir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(currentDir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          await findBackups(fullPath);
        } else if (entry.isFile() && entry.name === 'data.backup.json') {
          backupFiles.push(fullPath);
        }
      }
    }
    
    await findBackups(dir);
    
    console.log(`Found ${backupFiles.length} backup files\n`);
    
    let restored = 0;
    for (const backupFile of backupFiles) {
      try {
        const originalPath = backupFile.replace('.backup.json', '.json');
        const content = await fs.readFile(backupFile, 'utf8');
        await fs.writeFile(originalPath, content);
        await fs.unlink(backupFile); // Remove backup after restore
        restored++;
        console.log(`✅ Restored: ${path.relative(DATA_DIR, originalPath)}`);
      } catch (error) {
        console.error(`❌ Failed to restore ${backupFile}: ${error.message}`);
      }
    }
    
    console.log(`\n✅ Restored ${restored} files`);
  }

  formatSize(bytes) {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
  }

  generateReport() {
    const duration = (Date.now() - this.stats.startTime) / 1000;
    const savedSpace = this.stats.originalSize - this.stats.compressedSize;
    const compressionRatio = ((savedSpace / this.stats.originalSize) * 100).toFixed(1);
    
    console.log('\n' + '='.repeat(60));
    console.log('📊 Compression Complete!');
    console.log('='.repeat(60));
    console.log(`✅ Files processed: ${this.stats.processedFiles}/${this.stats.totalFiles}`);
    console.log(`❌ Failed: ${this.stats.failedFiles}`);
    console.log(`📦 Original size: ${this.formatSize(this.stats.originalSize)}`);
    console.log(`📦 Compressed size: ${this.formatSize(this.stats.compressedSize)}`);
    console.log(`💾 Space saved: ${this.formatSize(savedSpace)} (${compressionRatio}%)`);
    console.log(`⏱️  Duration: ${duration.toFixed(1)}s`);
    
    // Save report
    const report = {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      duration: duration + 's',
      compressionRatio: compressionRatio + '%',
      savedSpace: this.formatSize(savedSpace)
    };
    
    const reportPath = path.join(DATA_DIR, 'metadata', 'compression-report.json');
    fs.writeFile(reportPath, JSON.stringify(report, null, 2))
      .then(() => console.log(`\n📄 Report saved to: ${reportPath}`))
      .catch(console.error);
  }

  async analyzeDirectory(dir) {
    console.log('📊 Compression Analysis');
    console.log('='.repeat(60));
    
    const files = await this.findDataFiles(dir);
    console.log(`Found ${files.length} data files\n`);
    
    const strategies = ['compact', 'minified', 'gzip', 'optimized'];
    const totals = {};
    strategies.forEach(s => totals[s] = { size: 0, count: 0 });
    
    let totalOriginal = 0;
    
    for (const file of files.slice(0, 10)) { // Sample first 10 files
      const content = await fs.readFile(file, 'utf8');
      const data = JSON.parse(content);
      const originalSize = Buffer.byteLength(content);
      totalOriginal += originalSize;
      
      console.log(`\nAnalyzing: ${path.relative(DATA_DIR, file)}`);
      console.log(`Original: ${this.formatSize(originalSize)}`);
      
      for (const strategy of strategies) {
        const compressed = await this.compressor.compressionStrategies[strategy](data);
        totals[strategy].size += compressed.size;
        totals[strategy].count++;
        
        const ratio = ((originalSize - compressed.size) / originalSize * 100).toFixed(1);
        console.log(`  ${strategy}: ${this.formatSize(compressed.size)} (${ratio}% reduction)`);
      }
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 Average Compression Ratios (based on sample):');
    console.log('='.repeat(60));
    
    for (const [strategy, data] of Object.entries(totals)) {
      const avgRatio = ((totalOriginal - data.size) / totalOriginal * 100).toFixed(1);
      console.log(`${strategy.padEnd(12)}: ${avgRatio}% reduction`);
    }
    
    // Projection
    const totalFiles = files.length;
    const avgFileSize = totalOriginal / 10;
    const projectedTotal = avgFileSize * totalFiles;
    
    console.log('\n📈 Storage Projections:');
    console.log(`Current (estimated): ${this.formatSize(projectedTotal)}`);
    
    for (const [strategy, data] of Object.entries(totals)) {
      const avgRatio = (totalOriginal - data.size) / totalOriginal;
      const projected = projectedTotal * (1 - avgRatio);
      console.log(`With ${strategy}: ${this.formatSize(projected)}`);
    }
  }
}

// CLI usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const batch = new BatchCompressor();
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node batch-compress.js <command> [options]');
    console.log('');
    console.log('Commands:');
    console.log('  compress [dir]     Compress all data files in directory');
    console.log('  analyze [dir]      Analyze compression potential');
    console.log('  restore [dir]      Restore from backup files');
    console.log('');
    console.log('Options:');
    console.log('  --strategy <name>  Compression strategy: compact, minified, gzip, optimized (default)');
    console.log('  --no-backup        Skip creating backup files');
    console.log('');
    console.log('Examples:');
    console.log('  node batch-compress.js compress');
    console.log('  node batch-compress.js compress --strategy gzip');
    console.log('  node batch-compress.js analyze');
    console.log('  node batch-compress.js restore');
    process.exit(0);
  }
  
  const command = args[0];
  const dir = args[1] || DATA_DIR;
  
  if (command === 'compress') {
    const strategyIndex = args.indexOf('--strategy');
    const strategy = strategyIndex > -1 ? args[strategyIndex + 1] : 'optimized';
    const createBackup = !args.includes('--no-backup');
    
    batch.compressDirectory(dir, strategy, createBackup).catch(console.error);
    
  } else if (command === 'analyze') {
    batch.analyzeDirectory(dir).catch(console.error);
    
  } else if (command === 'restore') {
    batch.restoreFromBackup(dir).catch(console.error);
  }
}

export default BatchCompressor;