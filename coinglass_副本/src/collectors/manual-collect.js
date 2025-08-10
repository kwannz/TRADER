#!/usr/bin/env node

require('dotenv').config();
const collectorManager = require('./collector-manager');
const JsonStorageManager = require('../database/json-storage');
const { logger } = require('../utils/logger');

class ManualCollector {
  constructor() {
    this.isInitialized = false;
    this.jsonStorage = new JsonStorageManager();
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      console.log('🔄 Initializing CoinGlass Data Collector...');
      
      // 初始化JSON存储
      await this.jsonStorage.connect();
      console.log('✅ JSON storage initialized successfully');
      
      this.isInitialized = true;
      console.log('✅ Initialization completed\n');
    } catch (error) {
      console.error('❌ Initialization failed:', error.message);
      process.exit(1);
    }
  }

  async runCollection(collectorName = null) {
    await this.initialize();

    try {
      console.log('🚀 Starting data collection...\n');
      
      let result;
      if (collectorName) {
        console.log(`📊 Running collector: ${collectorName}`);
        result = await collectorManager.runCollector(collectorName);
        this.displaySingleResult(collectorName, result);
      } else {
        console.log('📊 Running all collectors...');
        result = await collectorManager.runAllCollectors();
        this.displayAllResults(result);
      }
      
      console.log('\n🎉 Collection completed successfully!');
      return result;
      
    } catch (error) {
      console.error('\n❌ Collection failed:', error.message);
      logger.error('Manual collection failed', error);
      throw error;
    }
  }

  displaySingleResult(collectorName, result) {
    console.log(`\n📈 Results for ${collectorName}:`);
    console.log(`   • Documents collected: ${result.count || 0}`);
    console.log(`   • Documents saved: ${result.saved || 0}`);
    console.log(`   • Duplicates removed: ${result.duplicatesRemoved || 0}`);
    console.log(`   • Duration: ${result.duration}ms`);
  }

  displayAllResults(result) {
    console.log('\n📈 Collection Summary:');
    console.log(`   • Total collectors: ${result.totalCollectors}`);
    console.log(`   • Successful: ${result.successfulCollectors}`);
    console.log(`   • Failed: ${result.failedCollectors}`);
    console.log(`   • Total duration: ${result.duration}ms`);
    
    console.log('\n📊 Individual Results:');
    for (const [name, collectorResult] of Object.entries(result.results)) {
      const status = collectorResult.success ? '✅' : '❌';
      console.log(`   ${status} ${name}:`);
      
      if (collectorResult.success) {
        console.log(`      - Collected: ${collectorResult.count || 0} documents`);
        console.log(`      - Saved: ${collectorResult.saved || 0} documents`);
        console.log(`      - Duration: ${collectorResult.duration || 0}ms`);
      } else {
        console.log(`      - Error: ${collectorResult.error}`);
      }
    }
  }

  async showStatus() {
    await this.initialize();

    try {
      console.log('📊 System Status:\n');
      
      // 数据库状态
      const dbHealth = await this.jsonStorage.healthCheck();
      console.log('🗄️  Database Status:');
      console.log(`   • Status: ${dbHealth.status}`);
      console.log(`   • Collections: ${dbHealth.collections || 0}`);
      console.log(`   • Data size: ${dbHealth.dataSize || 'N/A'}`);
      
      // 收集器状态
      const collectorStatus = collectorManager.getStatus();
      console.log('\n🔧 Collector Manager:');
      console.log(`   • Running: ${collectorStatus.isRunning ? 'Yes' : 'No'}`);
      console.log(`   • Total collectors: ${collectorStatus.totalCollectors}`);
      console.log(`   • Last run: ${collectorStatus.stats.lastRunTime || 'Never'}`);
      
      // 各收集器健康状态
      const healthStatus = await collectorManager.getAllHealthStatus();
      console.log('\n🩺 Collector Health:');
      for (const [name, health] of Object.entries(healthStatus)) {
        const statusIcon = health.status === 'healthy' ? '✅' : 
                          health.status === 'warning' ? '⚠️' : '❌';
        console.log(`   ${statusIcon} ${name}: ${health.reason}`);
      }
      
    } catch (error) {
      console.error('❌ Failed to get status:', error.message);
      throw error;
    }
  }

  async showStats() {
    await this.initialize();

    try {
      console.log('📈 Collection Statistics:\n');
      
      const allStats = collectorManager.getAllStats();
      
      // 管理器统计
      console.log('🔧 Manager Stats:');
      console.log(`   • Total runs: ${allStats.manager.totalRuns}`);
      console.log(`   • Successful runs: ${allStats.manager.successfulRuns}`);
      console.log(`   • Failed runs: ${allStats.manager.failedRuns}`);
      console.log(`   • Success rate: ${((allStats.manager.successfulRuns / Math.max(allStats.manager.totalRuns, 1)) * 100).toFixed(1)}%`);
      
      // 各收集器统计
      console.log('\n📊 Collector Stats:');
      for (const [name, stats] of Object.entries(allStats.collectors)) {
        console.log(`   🔹 ${name}:`);
        console.log(`      - Success rate: ${stats.successRate}%`);
        console.log(`      - Total documents: ${stats.totalDocuments}`);
        console.log(`      - Last run: ${stats.lastRunTime || 'Never'}`);
        console.log(`      - Errors: ${stats.errorCount}`);
      }
      
    } catch (error) {
      console.error('❌ Failed to get statistics:', error.message);
      throw error;
    }
  }

  async listCollectors() {
    await this.initialize();
    
    console.log('📋 Available Collectors:\n');
    const collectorNames = collectorManager.getCollectorNames();
    
    collectorNames.forEach((name, index) => {
      console.log(`   ${index + 1}. ${name}`);
    });
    
    console.log(`\nTotal: ${collectorNames.length} collectors available`);
  }

  async cleanup(days = 365) {
    await this.initialize();

    try {
      console.log(`🧹 Cleaning up data older than ${days} days...`);
      
      const result = await collectorManager.cleanupAllData(days);
      
      console.log('\n📈 Cleanup Results:');
      console.log(`   • Total deleted: ${result.totalDeleted} documents`);
      
      for (const [name, cleanupResult] of Object.entries(result.results)) {
        if (cleanupResult.success && cleanupResult.deleted > 0) {
          console.log(`   • ${name}: ${cleanupResult.deleted} documents deleted`);
        }
      }
      
      console.log('\n✅ Cleanup completed!');
      
    } catch (error) {
      console.error('❌ Cleanup failed:', error.message);
      throw error;
    }
  }

  displayHelp() {
    console.log(`
🔧 CoinGlass Data Collector - Manual Collection Tool

Usage: node src/collectors/manual-collect.js [command] [options]

Commands:
  collect [collector]     Run data collection (all collectors or specific one)
  status                  Show system status
  stats                   Show collection statistics  
  list                    List all available collectors
  cleanup [days]          Clean up old data (default: 365 days)
  help                    Show this help message

Examples:
  node src/collectors/manual-collect.js collect
  node src/collectors/manual-collect.js collect ContractMarketCollector
  node src/collectors/manual-collect.js status
  node src/collectors/manual-collect.js cleanup 30

Available Collectors:
  • ContractMarketCollector   - Contract market data
  • OpenInterestCollector     - Open interest data
  • FundingRateCollector      - Funding rate data
  • ETFCollector              - ETF flow data
  • IndicatorCollector        - Technical indicators
`);
  }

  async cleanup_resources() {
    try {
      await this.jsonStorage.disconnect();
      console.log('✅ Resources cleaned up');
    } catch (error) {
      console.error('⚠️  Warning: Failed to cleanup resources:', error.message);
    }
  }
}

// 主函数
async function main() {
  const collector = new ManualCollector();
  const args = process.argv.slice(2);
  const command = args[0] || 'help';
  
  try {
    switch (command.toLowerCase()) {
      case 'collect':
        const collectorName = args[1];
        await collector.runCollection(collectorName);
        break;
        
      case 'status':
        await collector.showStatus();
        break;
        
      case 'stats':
        await collector.showStats();
        break;
        
      case 'list':
        await collector.listCollectors();
        break;
        
      case 'cleanup':
        const days = parseInt(args[1]) || 365;
        await collector.cleanup(days);
        break;
        
      case 'help':
      default:
        collector.displayHelp();
        break;
    }
  } catch (error) {
    console.error('\n💥 Operation failed:', error.message);
    process.exit(1);
  } finally {
    await collector.cleanup_resources();
  }
}

// 处理程序退出
process.on('SIGINT', async () => {
  console.log('\n⏹️  Received interrupt signal, shutting down...');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n⏹️  Received termination signal, shutting down...');
  process.exit(0);
});

// 运行主函数
if (require.main === module) {
  main().catch(error => {
    console.error('💥 Unexpected error:', error);
    process.exit(1);
  });
}

module.exports = ManualCollector; 