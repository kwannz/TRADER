const { logDataCollection, logError, schedulerLogger } = require('../utils/logger');
const CoinGlassClient = require('../api/coinglass-client');

// 导入所有收集器
const ContractMarketCollector = require('./contract-market-collector');
const OpenInterestCollector = require('./open-interest-collector');
const FundingRateCollector = require('./funding-rate-collector');
const ETFCollector = require('./etf-collector');
const IndicatorCollector = require('./indicator-collector');

class CollectorManager {
  constructor() {
    this.collectors = new Map();
    this.isRunning = false;
    this.stats = {
      totalRuns: 0,
      successfulRuns: 0,
      failedRuns: 0,
      lastRunTime: null,
      startTime: new Date()
    };
    
    // 首先创建CoinGlass客户端
    this.coinGlassClient = new CoinGlassClient(process.env.COINGLASS_API_KEY);
    
    // 然后初始化收集器
    this.initializeCollectors();
  }

  /**
   * 初始化所有收集器
   */
  initializeCollectors() {
    try {
      // 注册所有收集器，传递CoinGlass客户端实例
      this.registerCollector(new ContractMarketCollector(this.coinGlassClient));
      this.registerCollector(new OpenInterestCollector(this.coinGlassClient));
      this.registerCollector(new FundingRateCollector(this.coinGlassClient));
      this.registerCollector(new ETFCollector(this.coinGlassClient));
      this.registerCollector(new IndicatorCollector(this.coinGlassClient));
      
      schedulerLogger.info(`Initialized ${this.collectors.size} collectors`, {
        collectors: Array.from(this.collectors.keys())
      });
    } catch (error) {
      logError('Failed to initialize collectors', error);
      throw error;
    }
  }

  /**
   * 注册收集器
   */
  registerCollector(collector) {
    if (!collector || !collector.name) {
      throw new Error('Invalid collector provided');
    }
    
    this.collectors.set(collector.name, collector);
    schedulerLogger.debug(`Registered collector: ${collector.name}`);
  }

  /**
   * 获取收集器
   */
  getCollector(name) {
    return this.collectors.get(name);
  }

  /**
   * 获取所有收集器名称
   */
  getCollectorNames() {
    return Array.from(this.collectors.keys());
  }

  /**
   * 运行所有收集器
   */
  async runAllCollectors() {
    if (this.isRunning) {
      throw new Error('Collection is already running');
    }

    this.isRunning = true;
    const startTime = Date.now();
    const results = {};
    let successCount = 0;
    let failureCount = 0;

    try {
      schedulerLogger.info('Starting data collection for all collectors');

      for (const [name, collector] of this.collectors) {
        try {
          schedulerLogger.info(`Starting collection for: ${name}`);
          const result = await collector.collectWithRetry();
          
          results[name] = {
            success: true,
            ...result
          };
          
          successCount++;
          schedulerLogger.info(`Completed collection for: ${name}`, {
            count: result.count,
            duration: result.duration
          });
          
        } catch (error) {
          results[name] = {
            success: false,
            error: error.message
          };
          
          failureCount++;
          logError(`Collection failed for: ${name}`, error);
        }
        
        // 在收集器之间添加延迟，避免API限制
        await this.sleep(1000);
      }

      const duration = Date.now() - startTime;
      this.stats.totalRuns++;
      this.stats.lastRunTime = new Date();
      
      if (failureCount === 0) {
        this.stats.successfulRuns++;
      } else {
        this.stats.failedRuns++;
      }

      const summary = {
        success: failureCount === 0,
        duration,
        totalCollectors: this.collectors.size,
        successfulCollectors: successCount,
        failedCollectors: failureCount,
        results
      };

      logDataCollection('CollectorManager', 'runAll', summary);
      
      return summary;

    } catch (error) {
      this.stats.totalRuns++;
      this.stats.failedRuns++;
      logError('Failed to run all collectors', error);
      throw error;
    } finally {
      this.isRunning = false;
    }
  }

  /**
   * 运行指定的收集器
   */
  async runCollector(collectorName) {
    const collector = this.collectors.get(collectorName);
    if (!collector) {
      throw new Error(`Collector ${collectorName} not found`);
    }

    try {
      schedulerLogger.info(`Running single collector: ${collectorName}`);
      const result = await collector.collectWithRetry();
      
      logDataCollection('CollectorManager', 'runSingle', {
        success: true,
        collector: collectorName,
        ...result
      });
      
      return result;
    } catch (error) {
      logError(`Failed to run collector ${collectorName}`, error);
      throw error;
    }
  }

  /**
   * 获取所有收集器的统计信息
   */
  getAllStats() {
    const collectorStats = {};
    
    for (const [name, collector] of this.collectors) {
      collectorStats[name] = collector.getStats();
    }

    return {
      manager: this.stats,
      collectors: collectorStats
    };
  }

  /**
   * 获取所有收集器的健康状态
   */
  async getAllHealthStatus() {
    const healthStatus = {};
    
    for (const [name, collector] of this.collectors) {
      try {
        healthStatus[name] = await collector.healthCheck();
      } catch (error) {
        healthStatus[name] = {
          status: 'unhealthy',
          reason: 'Health check failed',
          error: error.message
        };
      }
    }

    return healthStatus;
  }

  /**
   * 重置所有收集器的统计信息
   */
  resetAllStats() {
    for (const [name, collector] of this.collectors) {
      collector.resetStats();
    }
    
    this.stats = {
      totalRuns: 0,
      successfulRuns: 0,
      failedRuns: 0,
      lastRunTime: null,
      startTime: new Date()
    };
    
    schedulerLogger.info('Reset all collector statistics');
  }

  /**
   * 清理所有收集器的过期数据
   */
  async cleanupAllData(retentionDays = 365) {
    const results = {};
    let totalDeleted = 0;

    try {
      schedulerLogger.info(`Starting data cleanup with ${retentionDays} days retention`);

      for (const [name, collector] of this.collectors) {
        try {
          const deleted = await collector.cleanupOldData(retentionDays);
          results[name] = { success: true, deleted };
          totalDeleted += deleted;
          
          if (deleted > 0) {
            schedulerLogger.info(`Cleaned up ${deleted} documents from ${name}`);
          }
        } catch (error) {
          results[name] = { success: false, error: error.message };
          logError(`Failed to cleanup data for ${name}`, error);
        }
      }

      schedulerLogger.info(`Data cleanup completed. Total deleted: ${totalDeleted} documents`);
      
      return {
        success: true,
        totalDeleted,
        results
      };

    } catch (error) {
      logError('Failed to cleanup data', error);
      throw error;
    }
  }

  /**
   * 获取运行状态
   */
  getStatus() {
    return {
      isRunning: this.isRunning,
      totalCollectors: this.collectors.size,
      collectorNames: this.getCollectorNames(),
      stats: this.stats
    };
  }

  /**
   * 停止所有正在运行的收集器
   */
  async stopAll() {
    try {
      // 这里可以实现停止逻辑，如果收集器支持中断的话
      this.isRunning = false;
      schedulerLogger.info('Stopped all collectors');
    } catch (error) {
      logError('Failed to stop collectors', error);
      throw error;
    }
  }

  /**
   * 延迟函数
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// 创建单例实例
const collectorManager = new CollectorManager();

module.exports = collectorManager; 