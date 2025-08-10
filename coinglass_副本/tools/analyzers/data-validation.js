#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';

const OUTPUT_DIR = './data/okx';

class DataValidator {
  constructor() {
    this.issues = [];
    this.stats = {
      totalFiles: 0,
      validFiles: 0,
      invalidFiles: 0,
      totalDataPoints: 0,
      duplicates: 0,
      missingData: 0
    };
  }

  async validateFile(filePath) {
    try {
      const data = await fs.readFile(filePath, 'utf8');
      const json = JSON.parse(data);
      
      this.stats.totalFiles++;
      
      // 检查基本结构
      if (!json.data || !Array.isArray(json.data)) {
        this.issues.push({
          file: filePath,
          type: 'structure',
          message: '缺少data数组'
        });
        this.stats.invalidFiles++;
        return false;
      }
      
      // 检查数据点
      const dataPoints = json.data.length;
      this.stats.totalDataPoints += dataPoints;
      
      if (dataPoints === 0) {
        this.issues.push({
          file: filePath,
          type: 'empty',
          message: '数据为空'
        });
        this.stats.missingData++;
        return false;
      }
      
      // 检查时间戳连续性
      const timestamps = json.data.map(d => d.t || d.timestamp).filter(t => t);
      
      if (timestamps.length !== dataPoints) {
        this.issues.push({
          file: filePath,
          type: 'timestamp',
          message: '时间戳缺失'
        });
      }
      
      // 检查重复
      const uniqueTimestamps = new Set(timestamps);
      if (uniqueTimestamps.size !== timestamps.length) {
        const duplicateCount = timestamps.length - uniqueTimestamps.size;
        this.issues.push({
          file: filePath,
          type: 'duplicate',
          message: `发现 ${duplicateCount} 个重复时间戳`
        });
        this.stats.duplicates += duplicateCount;
      }
      
      // 检查价格数据
      for (let i = 0; i < Math.min(5, dataPoints); i++) {
        const point = json.data[i];
        const prices = [point.o, point.h, point.l, point.c].map(Number);
        
        if (prices.some(p => isNaN(p) || p <= 0)) {
          this.issues.push({
            file: filePath,
            type: 'price',
            message: '价格数据异常'
          });
          break;
        }
        
        // 检查OHLC逻辑
        const [open, high, low, close] = prices;
        if (high < Math.max(open, close) || low > Math.min(open, close)) {
          this.issues.push({
            file: filePath,
            type: 'ohlc',
            message: 'OHLC数据逻辑错误'
          });
          break;
        }
      }
      
      this.stats.validFiles++;
      return true;
      
    } catch (error) {
      this.issues.push({
        file: filePath,
        type: 'parse',
        message: `解析错误: ${error.message}`
      });
      this.stats.invalidFiles++;
      return false;
    }
  }

  async findDataFiles() {
    const files = [];
    
    try {
      const dirs = await fs.readdir(OUTPUT_DIR);
      
      for (const dir of dirs) {
        if (dir.includes('.json') || dir.includes('test-')) continue;
        
        try {
          const symbolPath = path.join(OUTPUT_DIR, dir);
          const timeframes = await fs.readdir(symbolPath);
          
          for (const tf of timeframes) {
            const dataFile = path.join(symbolPath, tf, 'data.json');
            
            try {
              await fs.access(dataFile);
              files.push(dataFile);
            } catch (e) {
              // 文件不存在
            }
          }
        } catch (e) {
          // 目录读取失败
        }
      }
    } catch (error) {
      console.error('读取目录失败:', error.message);
    }
    
    return files;
  }

  async run() {
    console.log('🔍 开始数据质量验证\n');
    
    const files = await this.findDataFiles();
    console.log(`找到 ${files.length} 个数据文件\n`);
    
    let processed = 0;
    for (const file of files) {
      processed++;
      process.stdout.write(`\r验证进度: ${processed}/${files.length} (${(processed/files.length*100).toFixed(1)}%)`);
      
      await this.validateFile(file);
    }
    
    console.log('\n\n📊 验证结果:');
    console.log(`- 总文件数: ${this.stats.totalFiles}`);
    console.log(`- 有效文件: ${this.stats.validFiles}`);
    console.log(`- 无效文件: ${this.stats.invalidFiles}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}`);
    
    if (this.stats.duplicates > 0) {
      console.log(`- 重复数据: ${this.stats.duplicates}`);
    }
    
    if (this.stats.missingData > 0) {
      console.log(`- 空数据文件: ${this.stats.missingData}`);
    }
    
    // 显示问题汇总
    if (this.issues.length > 0) {
      console.log('\n⚠️  发现的问题:');
      
      const issueTypes = {};
      this.issues.forEach(issue => {
        issueTypes[issue.type] = (issueTypes[issue.type] || 0) + 1;
      });
      
      Object.entries(issueTypes).forEach(([type, count]) => {
        console.log(`- ${type}: ${count} 个文件`);
      });
      
      // 显示前5个具体问题
      if (this.issues.length <= 10) {
        console.log('\n具体问题:');
        this.issues.forEach((issue, i) => {
          console.log(`${i + 1}. ${path.basename(issue.file)}: ${issue.message}`);
        });
      }
    } else {
      console.log('\n✅ 所有数据文件验证通过！');
    }
    
    // 计算质量评分
    const qualityScore = this.stats.totalFiles > 0 ? 
      (this.stats.validFiles / this.stats.totalFiles * 100).toFixed(1) : 0;
    
    console.log(`\n📈 数据质量评分: ${qualityScore}%`);
    
    // 保存验证报告
    const report = {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      qualityScore: parseFloat(qualityScore),
      issues: this.issues,
      summary: {
        totalSymbols: new Set(files.map(f => path.basename(path.dirname(path.dirname(f))))).size,
        avgDataPointsPerFile: Math.round(this.stats.totalDataPoints / this.stats.totalFiles),
        issueRate: this.stats.totalFiles > 0 ? 
          (this.issues.length / this.stats.totalFiles * 100).toFixed(2) + '%' : '0%'
      }
    };
    
    await fs.writeFile(
      path.join(OUTPUT_DIR, 'data-validation-report.json'),
      JSON.stringify(report, null, 2)
    );
    
    console.log('\n📄 验证报告已保存到: data-validation-report.json');
  }
}

// 运行
async function main() {
  const validator = new DataValidator();
  await validator.run();
}

main();