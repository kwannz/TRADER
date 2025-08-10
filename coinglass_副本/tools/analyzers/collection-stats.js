#!/usr/bin/env node

import fs from 'fs/promises';
import path from 'path';

const OUTPUT_DIR = './data/okx';

async function getStats() {
  console.log('📊 OKX数据收集统计\n');
  
  try {
    // 获取所有目录
    const dirs = await fs.readdir(OUTPUT_DIR);
    const symbolDirs = dirs.filter(d => !d.includes('.json') && !d.includes('test-data'));
    
    console.log(`已收集代币: ${symbolDirs.length} 个\n`);
    
    let totalDataPoints = 0;
    let totalSize = 0;
    const symbolStats = [];
    
    // 统计每个代币
    for (const symbol of symbolDirs.sort()) {
      let symbolData = { symbol, timeframes: {}, totalPoints: 0 };
      
      try {
        const timeframeDirs = await fs.readdir(path.join(OUTPUT_DIR, symbol));
        
        for (const tf of timeframeDirs) {
          const dataPath = path.join(OUTPUT_DIR, symbol, tf, 'data.json');
          
          try {
            const stat = await fs.stat(dataPath);
            const data = await fs.readFile(dataPath, 'utf8');
            const json = JSON.parse(data);
            
            const count = json.count || json.totalCount || (json.data ? json.data.length : 0);
            symbolData.timeframes[tf] = count;
            symbolData.totalPoints += count;
            totalDataPoints += count;
            totalSize += stat.size;
          } catch (e) {
            // 文件不存在或读取失败
          }
        }
        
        if (symbolData.totalPoints > 0) {
          symbolStats.push(symbolData);
        }
      } catch (e) {
        // 目录读取失败
      }
    }
    
    // 显示统计
    console.log('代币数据详情:');
    console.log('─'.repeat(60));
    console.log('代币\t日线\t4小时\t1小时\t30分钟\t15分钟\t总计');
    console.log('─'.repeat(60));
    
    symbolStats.forEach(s => {
      const line = [
        s.symbol.padEnd(8),
        (s.timeframes['1D'] || '-').toString().padEnd(8),
        (s.timeframes['4H'] || '-').toString().padEnd(8),
        (s.timeframes['1H'] || '-').toString().padEnd(8),
        (s.timeframes['30m'] || '-').toString().padEnd(8),
        (s.timeframes['15m'] || '-').toString().padEnd(8),
        s.totalPoints.toLocaleString()
      ];
      console.log(line.join('\t'));
    });
    
    console.log('─'.repeat(60));
    
    // 汇总统计
    console.log('\n📈 汇总统计:');
    console.log(`- 总代币数: ${symbolStats.length}`);
    console.log(`- 总数据点: ${totalDataPoints.toLocaleString()}`);
    console.log(`- 总文件大小: ${(totalSize / 1024 / 1024).toFixed(1)} MB`);
    console.log(`- 平均每代币: ${Math.round(totalDataPoints / symbolStats.length).toLocaleString()} 条数据`);
    
    // 加载top150列表对比
    try {
      const top150Data = await fs.readFile('./historical/storage/raw/okx/top150-symbols.json', 'utf8');
      const top150 = JSON.parse(top150Data);
      const collected = new Set(symbolStats.map(s => s.symbol));
      const missing = top150.filter(s => !collected.has(s));
      
      console.log(`\n📋 收集进度: ${collected.size}/${top150.length} (${(collected.size/top150.length*100).toFixed(1)}%)`);
      
      if (missing.length > 0 && missing.length <= 20) {
        console.log(`\n未收集代币: ${missing.join(', ')}`);
      } else if (missing.length > 20) {
        console.log(`\n未收集代币: ${missing.slice(0, 20).join(', ')}... 等${missing.length}个`);
      }
    } catch (e) {
      // 无法加载top150列表
    }
    
  } catch (error) {
    console.error('错误:', error.message);
  }
}

// 运行
getStats();