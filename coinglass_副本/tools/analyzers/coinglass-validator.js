#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';

class CoinGlassValidator {
  constructor() {
    this.dataDir = './data/coinglass';
    this.stats = {
      totalFiles: 0,
      validFiles: 0,
      invalidFiles: 0,
      totalDataPoints: 0,
      categories: {}
    };
    this.issues = [];
  }

  async validateFile(filePath) {
    try {
      const data = await fs.readFile(filePath, 'utf8');
      const json = JSON.parse(data);
      
      this.stats.totalFiles++;
      
      // 基本结构检查
      if (!json.data) {
        this.issues.push({
          file: filePath,
          type: 'structure',
          message: '缺少data字段'
        });
        this.stats.invalidFiles++;
        return false;
      }

      // 统计数据点
      let dataPoints = 0;
      if (Array.isArray(json.data)) {
        dataPoints = json.data.length;
      } else if (json.data.data_list && Array.isArray(json.data.data_list)) {
        dataPoints = json.data.data_list.length;
      } else if (typeof json.data === 'object') {
        dataPoints = 1;
      }

      this.stats.totalDataPoints += dataPoints;

      // 检查元数据
      if (!json.collectedAt) {
        this.issues.push({
          file: filePath,
          type: 'metadata',
          message: '缺少collectedAt时间戳'
        });
      }

      // 检查数据质量
      if (dataPoints === 0) {
        this.issues.push({
          file: filePath,
          type: 'empty',
          message: '数据为空'
        });
      }

      this.stats.validFiles++;
      return true;
      
    } catch (error) {
      this.issues.push({
        file: filePath,
        type: 'parse',
        message: `JSON解析错误: ${error.message}`
      });
      this.stats.invalidFiles++;
      return false;
    }
  }

  async scanDirectory() {
    const categories = ['futures', 'spot', 'etf', 'indicators', 'options', 'onchain'];
    
    for (const category of categories) {
      const categoryPath = path.join(this.dataDir, category);
      
      try {
        const files = await fs.readdir(categoryPath);
        const jsonFiles = files.filter(f => f.endsWith('.json'));
        
        this.stats.categories[category] = {
          totalFiles: jsonFiles.length,
          validFiles: 0,
          dataPoints: 0
        };

        for (const file of jsonFiles) {
          const filePath = path.join(categoryPath, file);
          const isValid = await this.validateFile(filePath);
          
          if (isValid) {
            this.stats.categories[category].validFiles++;
          }
        }
        
      } catch (error) {
        if (error.code !== 'ENOENT') {
          console.log(`⚠️  无法读取目录 ${category}: ${error.message}`);
        }
      }
    }
  }

  async generateSummary() {
    console.log('📊 CoinGlass数据验证结果:\n');
    
    // 总体统计
    console.log('🔍 总体统计:');
    console.log(`- 总文件数: ${this.stats.totalFiles}`);
    console.log(`- 有效文件: ${this.stats.validFiles}`);
    console.log(`- 无效文件: ${this.stats.invalidFiles}`);
    console.log(`- 总数据点: ${this.stats.totalDataPoints.toLocaleString()}\n`);

    // 分类统计
    console.log('📁 分类统计:');
    Object.entries(this.stats.categories).forEach(([category, stats]) => {
      if (stats.totalFiles > 0) {
        console.log(`- ${category}: ${stats.validFiles}/${stats.totalFiles} 文件有效`);
      }
    });

    // 质量评分
    const qualityScore = this.stats.totalFiles > 0 ? 
      (this.stats.validFiles / this.stats.totalFiles * 100).toFixed(1) : 0;
    console.log(`\n📈 数据质量评分: ${qualityScore}%`);

    // 显示问题
    if (this.issues.length > 0) {
      console.log('\n⚠️  发现的问题:');
      const issueTypes = {};
      this.issues.forEach(issue => {
        issueTypes[issue.type] = (issueTypes[issue.type] || 0) + 1;
      });
      
      Object.entries(issueTypes).forEach(([type, count]) => {
        console.log(`- ${type}: ${count} 个文件`);
      });
    } else {
      console.log('\n✅ 所有数据文件验证通过！');
    }
  }

  async generateDataPreview() {
    console.log('\n📋 数据预览:\n');
    
    const previewFiles = [
      'indicators/fear-greed-index.json',
      'futures/btc-funding-rate.json', 
      'etf/bitcoin-etf-list.json',
      'spot/markets.json'
    ];

    for (const file of previewFiles) {
      try {
        const filePath = path.join(this.dataDir, file);
        const data = await fs.readFile(filePath, 'utf8');
        const json = JSON.parse(data);
        
        console.log(`📄 ${file}:`);
        console.log(`   类型: ${json.type || 'unknown'}`);
        console.log(`   数据点: ${json.dataCount || 'unknown'}`);
        console.log(`   收集时间: ${json.collectedAt || 'unknown'}`);
        console.log('');
        
      } catch (error) {
        console.log(`📄 ${file}: 无法读取`);
      }
    }
  }

  async run() {
    console.log('🔍 开始CoinGlass数据验证\n');
    
    await this.scanDirectory();
    await this.generateSummary();
    await this.generateDataPreview();
    
    // 保存验证报告
    const report = {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      qualityScore: this.stats.totalFiles > 0 ? 
        parseFloat((this.stats.validFiles / this.stats.totalFiles * 100).toFixed(1)) : 0,
      issues: this.issues
    };
    
    const reportPath = path.join(this.dataDir, 'validation-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📄 验证报告已保存: ${reportPath}`);
  }
}

// 运行验证工具
if (import.meta.url === `file://${process.argv[1]}`) {
  const validator = new CoinGlassValidator();
  await validator.run();
}

export default CoinGlassValidator;