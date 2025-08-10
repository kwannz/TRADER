#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';

class CoinGlassDataOrganizer {
  constructor() {
    this.archiveDir = './data/archive';
    this.coinglassDir = './data/coinglass';
    this.stats = {
      processed: 0,
      moved: 0,
      skipped: 0
    };
  }

  async ensureDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      // Directory already exists
    }
  }

  async organizeDataFile(filePath, category, newName) {
    try {
      const data = await fs.readFile(filePath, 'utf8');
      const jsonData = JSON.parse(data);
      
      const targetDir = path.join(this.coinglassDir, category);
      await this.ensureDirectory(targetDir);
      
      const targetPath = path.join(targetDir, newName);
      
      // 添加元数据
      const organizedData = {
        originalFile: path.basename(filePath),
        category,
        type: category.replace(/s$/, ''), // 去掉复数形式
        collectedAt: new Date().toISOString(),
        dataCount: this.getDataCount(jsonData),
        data: jsonData
      };
      
      await fs.writeFile(targetPath, JSON.stringify(organizedData, null, 2));
      console.log(`✅ 移动: ${filePath} → ${targetPath}`);
      
      this.stats.moved++;
      return true;
    } catch (error) {
      console.log(`❌ 处理失败: ${filePath} - ${error.message}`);
      this.stats.skipped++;
      return false;
    }
  }

  getDataCount(data) {
    if (Array.isArray(data)) return data.length;
    if (data.data) {
      if (Array.isArray(data.data)) return data.data.length;
      if (data.data.data_list && Array.isArray(data.data.data_list)) {
        return data.data.data_list.length;
      }
    }
    return 1;
  }

  async run() {
    console.log('🗂️  开始整理CoinGlass数据...\n');

    // 定义文件映射规则
    const fileMap = [
      // 指标数据
      {
        source: 'fear-greed-index.json',
        category: 'indicators',
        newName: 'fear-greed-index.json'
      },
      {
        source: 'indicators/fear-greed-history-complete.json',
        category: 'indicators', 
        newName: 'fear-greed-history.json'
      },
      {
        source: 'indicators/fear-greed-index.json',
        category: 'indicators',
        newName: 'fear-greed-current.json'
      },
      
      // 期货数据
      {
        source: 'funding-rate-history.json',
        category: 'futures',
        newName: 'funding-rate-history.json'
      },
      {
        source: 'futures-markets.json',
        category: 'futures',
        newName: 'markets.json'
      },
      {
        source: 'futures-supported-coins.json',
        category: 'futures', 
        newName: 'supported-coins.json'
      },
      {
        source: 'liquidation-history.json',
        category: 'futures',
        newName: 'liquidation-history.json'
      },
      {
        source: 'open-interest-history.json',
        category: 'futures',
        newName: 'open-interest-history.json'
      },
      
      // 现货数据
      {
        source: 'spot-markets.json',
        category: 'spot',
        newName: 'markets.json'
      },
      {
        source: 'spot-orderbook-history.json',
        category: 'spot',
        newName: 'orderbook-history.json'
      },
      {
        source: 'spot-supported-coins.json',
        category: 'spot',
        newName: 'supported-coins.json'
      },
      
      // ETF数据
      {
        source: 'bitcoin-etf-list.json',
        category: 'etf',
        newName: 'bitcoin-etf-list.json'
      }
    ];

    // 处理BTC和ETH特定数据
    const symbolData = ['BTC', 'ETH'];
    for (const symbol of symbolData) {
      const symbolFiles = [
        { suffix: 'funding-rate.json', category: 'futures' },
        { suffix: 'futures-markets.json', category: 'futures' },
        { suffix: 'liquidation.json', category: 'futures' },
        { suffix: 'open-interest.json', category: 'futures' },
        { suffix: 'spot-markets.json', category: 'spot' }
      ];
      
      for (const file of symbolFiles) {
        fileMap.push({
          source: `${symbol}/${symbol}-${file.suffix}`,
          category: file.category,
          newName: `${symbol.toLowerCase()}-${file.suffix}`
        });
      }
    }

    // 处理所有文件
    for (const mapping of fileMap) {
      this.stats.processed++;
      const sourcePath = path.join(this.archiveDir, mapping.source);
      
      try {
        await fs.access(sourcePath);
        await this.organizeDataFile(sourcePath, mapping.category, mapping.newName);
      } catch (error) {
        console.log(`⏭️  跳过: ${sourcePath} (文件不存在)`);
        this.stats.skipped++;
      }
    }

    // 生成整理报告
    await this.generateReport();
    
    console.log('\n📊 整理完成统计:');
    console.log(`- 处理文件: ${this.stats.processed}`);
    console.log(`- 成功移动: ${this.stats.moved}`);
    console.log(`- 跳过: ${this.stats.skipped}`);
  }

  async generateReport() {
    const report = {
      organizedAt: new Date().toISOString(),
      stats: this.stats,
      structure: {}
    };

    // 扫描整理后的目录结构
    try {
      const categories = await fs.readdir(this.coinglassDir);
      
      for (const category of categories) {
        const categoryPath = path.join(this.coinglassDir, category);
        const stat = await fs.stat(categoryPath);
        
        if (stat.isDirectory()) {
          const files = await fs.readdir(categoryPath);
          report.structure[category] = files.length;
        }
      }
    } catch (error) {
      console.log(`警告: 无法生成完整报告 - ${error.message}`);
    }

    const reportPath = path.join(this.coinglassDir, 'organization-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📄 整理报告已保存: ${reportPath}`);
  }
}

// 运行整理工具
if (import.meta.url === `file://${process.argv[1]}`) {
  const organizer = new CoinGlassDataOrganizer();
  await organizer.run();
}

export default CoinGlassDataOrganizer;