#!/usr/bin/env node

require('dotenv').config();

const JsonStorageManager = require('./database/json-storage');
const collectorManager = require('./collectors/collector-manager');
const { logger, logSystemHealth } = require('./utils/logger');
const CoinGlassClient = require('./api/coinglass-client');

class CoinGlassCollectorApp {
  constructor() {
    this.isRunning = false;
    this.startTime = null;
    this.coinGlassClient = new CoinGlassClient(process.env.COINGLASS_API_KEY);
    this.jsonStorage = new JsonStorageManager();
  }

  /**
   * 初始化应用程序
   */
  async initialize() {
    try {
      logger.info('🚀 Initializing CoinGlass Data Collector...');
      
      // 验证环境变量
      this.validateEnvironment();
      
      // 初始化JSON存储
      logger.info('📡 Initializing JSON storage...');
      await this.jsonStorage.connect();
      
      // 测试API连接
      logger.info('🔌 Testing API connection...');
      const apiTest = await this.coinGlassClient.testConnection();
      if (!apiTest.success) {
        throw new Error(`API connection failed: ${apiTest.message}`);
      }
      logger.info(`✅ API connection successful (${apiTest.supportedCoins} coins supported)`);
      
      // 显示系统信息
      this.displaySystemInfo();
      
      logger.info('✅ Initialization completed successfully');
      
    } catch (error) {
      logger.error('❌ Initialization failed', error);
      throw error;
    }
  }

  /**
   * 验证必需的环境变量
   */
  validateEnvironment() {
    const required = ['COINGLASS_API_KEY', 'MONGODB_URI'];
    const missing = required.filter(key => !process.env[key]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
    
    logger.info('✅ Environment variables validated');
  }

  /**
   * 显示系统信息
   */
  displaySystemInfo() {
    const apiStatus = this.coinGlassClient.getStatus();
    
    logger.info('📊 System Information', {
      node_version: process.version,
      platform: process.platform,
      api_base_url: apiStatus.baseURL,
      has_api_key: apiStatus.hasApiKey,
      rate_limit: apiStatus.rateLimit,
      collectors: collectorManager.getCollectorNames().length
    });
  }

  /**
   * 启动应用程序
   */
  async start() {
    try {
      await this.initialize();
      
      this.isRunning = true;
      this.startTime = new Date();
      
      logger.info('🎯 Starting CoinGlass Data Collector Service');
      
      // 显示可用的收集器
      const collectorNames = collectorManager.getCollectorNames();
      logger.info(`📋 Available collectors: ${collectorNames.join(', ')}`);
      
      // 执行一次数据收集
      logger.info('📊 Running initial data collection...');
      const result = await collectorManager.runAllCollectors();
      
      logger.info('🎉 Initial collection completed', {
        successful_collectors: result.successfulCollectors,
        failed_collectors: result.failedCollectors,
        total_duration: result.duration
      });
      
      // 显示结果摘要
      this.displayCollectionSummary(result);
      
      logger.info('✅ CoinGlass Data Collector is now running');
      
      return result;
      
    } catch (error) {
      logger.error('❌ Failed to start application', error);
      throw error;
    }
  }

  /**
   * 显示收集结果摘要
   */
  displayCollectionSummary(result) {
    console.log('\n📈 Collection Summary:');
    console.log(`   • Total collectors: ${result.totalCollectors}`);
    console.log(`   • Successful: ${result.successfulCollectors}`);
    console.log(`   • Failed: ${result.failedCollectors}`);
    console.log(`   • Duration: ${(result.duration / 1000).toFixed(2)}s`);
    
    if (result.failedCollectors > 0) {
      console.log('\n❌ Failed collectors:');
      for (const [name, collectorResult] of Object.entries(result.results)) {
        if (!collectorResult.success) {
          console.log(`   • ${name}: ${collectorResult.error}`);
        }
      }
    }
    
    console.log('\n✅ Data collection completed successfully!\n');
  }

  /**
   * 运行健康检查
   */
  async healthCheck() {
    try {
      const health = {
        timestamp: new Date().toISOString(),
        uptime: this.isRunning ? Date.now() - this.startTime.getTime() : 0,
        status: 'healthy'
      };
      
      // 检查数据库
      const dbHealth = await this.jsonStorage.healthCheck();
      health.database = dbHealth;
      
      // 检查收集器
      const collectorHealth = await collectorManager.getAllHealthStatus();
      health.collectors = collectorHealth;
      
      // 检查API
      const apiStatus = this.coinGlassClient.getStatus();
      health.api = {
        status: apiStatus.hasApiKey ? 'configured' : 'missing_key',
        request_count: apiStatus.requestCount,
        rate_limit: apiStatus.rateLimit
      };
      
      // 确定整体状态
      if (dbHealth.status !== 'healthy') {
        health.status = 'unhealthy';
      } else if (Object.values(collectorHealth).some(c => c.status === 'unhealthy')) {
        health.status = 'degraded';
      }
      
      logSystemHealth('CoinGlassCollectorApp', health.status, health);
      
      return health;
      
    } catch (error) {
      const health = {
        timestamp: new Date().toISOString(),
        status: 'unhealthy',
        error: error.message
      };
      
      logSystemHealth('CoinGlassCollectorApp', 'unhealthy', health);
      return health;
    }
  }

  /**
   * 获取统计信息
   */
  getStats() {
    const collectorStats = collectorManager.getAllStats();
    
    return {
      app: {
        name: 'CoinGlass Data Collector',
        version: '1.0.0',
        uptime: this.isRunning ? Date.now() - this.startTime.getTime() : 0,
        start_time: this.startTime,
        is_running: this.isRunning
      },
      ...collectorStats
    };
  }

  /**
   * 停止应用程序
   */
  async stop() {
    try {
      logger.info('🛑 Stopping CoinGlass Data Collector...');
      
      // 停止收集器
      await collectorManager.stopAll();
      
      // 断开数据库连接
      await this.jsonStorage.disconnect();
      
      this.isRunning = false;
      
      logger.info('✅ CoinGlass Data Collector stopped successfully');
      
    } catch (error) {
      logger.error('❌ Error during shutdown', error);
      throw error;
    }
  }
}

// 创建应用实例
const app = new CoinGlassCollectorApp();

// 命令行界面
async function cli() {
  const args = process.argv.slice(2);
  const command = args[0] || 'start';
  
  switch (command.toLowerCase()) {
    case 'start':
      try {
        await app.start();
        
        // 在开发模式下，启动后保持运行
        if (process.env.NODE_ENV === 'development') {
          console.log('📱 Development mode: Press Ctrl+C to stop\n');
          
          // 设置定期健康检查
          setInterval(async () => {
            try {
              await app.healthCheck();
            } catch (error) {
              logger.error('Health check failed', error);
            }
          }, 5 * 60 * 1000); // 每5分钟检查一次
          
          // 保持进程运行
          process.stdin.resume();
        }
        
      } catch (error) {
        console.error('❌ Failed to start:', error.message);
        process.exit(1);
      }
      break;
      
    case 'health':
      try {
        await app.initialize();
        const health = await app.healthCheck();
        console.log(JSON.stringify(health, null, 2));
      } catch (error) {
        console.error('❌ Health check failed:', error.message);
        process.exit(1);
      } finally {
        await app.stop();
      }
      break;
      
    case 'stats':
      try {
        await app.initialize();
        const stats = app.getStats();
        console.log(JSON.stringify(stats, null, 2));
      } catch (error) {
        console.error('❌ Failed to get stats:', error.message);
        process.exit(1);
      } finally {
        await app.stop();
      }
      break;
      
    case 'test':
      try {
        await app.initialize();
        const testResult = await app.coinGlassClient.testConnection();
        console.log('🧪 API Test Result:', testResult);
      } catch (error) {
        console.error('❌ Test failed:', error.message);
        process.exit(1);
      } finally {
        await app.stop();
      }
      break;
      
    case 'help':
    default:
      console.log(`
🔧 CoinGlass Data Collector

Usage: node src/index.js [command]

Commands:
  start     Start the data collector (default)
  health    Show health status
  stats     Show statistics
  test      Test API connection
  help      Show this help

Examples:
  node src/index.js start
  node src/index.js health
  node src/index.js test
`);
      break;
  }
}

// 优雅退出处理
function setupGracefulShutdown() {
  const shutdown = async (signal) => {
    logger.info(`Received ${signal}, shutting down gracefully...`);
    
    try {
      await app.stop();
      process.exit(0);
    } catch (error) {
      logger.error('Error during shutdown', error);
      process.exit(1);
    }
  };
  
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  
  // 处理未捕获的异常
  process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception', error);
    process.exit(1);
  });
  
  process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Promise Rejection', { reason, promise });
    process.exit(1);
  });
}

// 设置优雅退出
setupGracefulShutdown();

// 运行CLI
if (require.main === module) {
  cli().catch(error => {
    logger.error('CLI error', error);
    process.exit(1);
  });
}

module.exports = {
  CoinGlassCollectorApp,
  app
}; 